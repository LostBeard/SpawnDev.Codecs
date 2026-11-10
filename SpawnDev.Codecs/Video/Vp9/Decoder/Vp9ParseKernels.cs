// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// VP9 tile parse on the accelerator: partitions, mode info, motion vectors and coefficient tokens of every block,
// one thread per tile column (its tile rows in order: the above contexts carry over between tile rows). A port of
// libvpx v1.17.0 (BSD-3-Clause) row-MT parse path:
//   vp9/decoder/vp9_decodeframe.c  decode_tiles / process_partition (PARSE) / parse_block / read_partition /
//                                  dec_partition_plane_context / dec_update_partition_context / set_offsets /
//                                  parse_intra_block_row_mt / parse_inter_block_row_mt
//   vp9/decoder/vp9_decodemv.c     vp9_read_mode_info (all of it) incl. dec_find_mv_refs, append_sub8x8_mvs_for_idx
//   vp9/decoder/vp9_detokenize.c   decode_coefs / vp9_decode_block_tokens
//   vp9/common/vp9_entropymv.c     vp9_inc_mv (counts)
// The recursion of process_partition becomes an explicit stack: only PARTITION_SPLIT above 8x8 recurses and it has
// no work after its children, so pre-order stack frames give libvpx's order exactly.
// Outputs follow libvpx's row-MT buffers: per superblock the partitions in visit order and, per plane, the dequantized
// coefficients of each coded transform block in parse order (16 << (2 * tx_size) slots each) with its eob.

using ILGPU;
using SpawnDev.Codecs.Video.Vp8;

namespace SpawnDev.Codecs.Video.Vp9.Decoder;

/// <summary>Per-frame int arguments of the parse kernel.</summary>
internal static class Vp9PArg
{
    public const int MiCols = 0, MiRows = 1, Log2TileCols = 2, Log2TileRows = 3, IntraOnly = 4, CountsOn = 5, Interp = 6,
        AllowHp = 7, UsePrevMvs = 8, Lossless = 9, SegEnabled = 10, SegUpdateMap = 11, SegTemporal = 12,
        SegTree = 13, SegPred = 20, SegMask = 23, SegData = 31, SignBias = 63, Dequant = 67, SbCols = 99,
        GridOffset = 101, LastSegOffset = 102, CurSegOffset = 103, CurMvOffset = 104, PrevMvOffset = 105,
        DqStride = 109, DqU = 110, DqV = 111, StatusOffset = 112, SsX = 113, SsY = 114,
        TileBuf = 128;
    public const int Count = TileBuf + 2 * 4 * 64;
}

/// <summary>Buffer layouts of the parse kernel.</summary>
internal static class Vp9PLayout
{
    /// <summary>
    /// Per superblock (sb buffer): 85 partitions, 256 eobs per plane, then the block table: a count and up to 64 entries
    /// (one per decoded block, parse order) of <see cref="BlkStride"/> ints: mode info record, bwl, bhl, and the
    /// coefficient (absolute) and eob (relative) positions where the block's data starts per plane.
    /// </summary>
    public const int SbPartitions = 0, SbEobY = 85, SbEobU = 85 + 256, SbEobV = 85 + 512, SbBlockCount = 85 + 768, SbBlocks = SbBlockCount + 1;
    public const int BlkRec = 0, BlkBwl = 1, BlkBhl = 2, BlkDqY = 3, BlkDqU = 4, BlkDqV = 5, BlkEobY = 6, BlkEobU = 7, BlkEobV = 8, BlkStride = 9;
    public const int SbStride = SbBlocks + 64 * BlkStride;
    /// <summary>Partition stack entries (row, col, bsize, n4 log2) of the per thread local stack.</summary>
    public const int StackEntries = 16;
    /// <summary>Per mi position (maps buffer, MV regions): ref_frame[0], ref_frame[1], mv[0], mv[1].</summary>
    public const int MvStride = 4;
}

/// <summary>
/// The parse thread's local memory (one int array per tile column thread), so the hot loop's read-after-write state
/// does not go through device memory: the above partition contexts (mi columns relative to the tile start; a tile column
/// is at most 64 superblocks = 512 mi wide), the left entropy and partition contexts of the superblock row, and the
/// token cache (libvpx token_cache, by coefficient position). 1592 ints + the 64-int partition stack = 6.5 KB, inside
/// WGSL's 8192-byte limit on a function's variables (the above entropy contexts, 3 x 1024 more, stay in the work buffer).
/// </summary>
internal static class Vp9PLocal
{
    public const int AbovePart = 0, LeftCtx = 512, LeftPart = LeftCtx + 48, TokenCache = LeftPart + 8, Ints = TokenCache + 1024;
}

/// <summary>
/// Coefficient scan tables packed for the token loop: per (tx_size, tx_type) and scan index c in [0, 16 << 2 tx], one int
/// <c>scan[c] | neighbors[2c] &lt;&lt; 10 | neighbors[2c + 1] &lt;&lt; 20</c> (libvpx vp9_scan_orders; the entry at the end
/// carries the neighbours' padding), so one load per coefficient gives its position and the next context's neighbours.
/// Appended to the parse kernel's table buffer after <see cref="Vp9T.Packed"/>.
/// </summary>
internal static class Vp9PScan
{
    /// <summary>Offset of the packed table of (<paramref name="tx"/>, <paramref name="type"/>) in <see cref="Tables"/>.</summary>
    public static int Offset(int tx, int type) =>
        Vp9T.Count + (tx == 0 ? 0 : tx == 1 ? 4 * 17 : tx == 2 ? 4 * (17 + 65) : 4 * (17 + 65 + 257)) + type * ((16 << (2 * tx)) + 1);

    public const int Ints = 4 * (17 + 65 + 257 + 1025);

    // After the scans, the parse kernel's own small trees (vpx_tree_index form: a leaf is -value), so a single bool with a
    // partition result and the coefficient token tree go through the kernel's one tree reader:
    /// <summary>read_partition with only the columns inside the frame: bit ? PARTITION_SPLIT : PARTITION_HORZ.</summary>
    public const int PartHorzTree = Vp9T.Count + Ints;
    /// <summary>read_partition with only the rows inside the frame: bit ? PARTITION_SPLIT : PARTITION_VERT.</summary>
    public const int PartVertTree = PartHorzTree + 2;
    /// <summary>libvpx vp9_coef_con_tree (TWO_TOKEN .. CATEGORY6_TOKEN under the pivot node, probabilities vp9_pareto8_full).</summary>
    public const int CoefConTree = PartVertTree + 2;
    /// <summary>
    /// Per token (index = token, TWO_TOKEN 2 .. CATEGORY6_TOKEN 10): its token cache value (vp9_pt_energy_class, bits 0-2),
    /// its smallest value (bits 3-9), its number of extra bits (bits 10-13) and the extra bits' probabilities (offset in
    /// <see cref="Tables"/>, bits 14 and up).
    /// </summary>
    public const int TokenInfo = CoefConTree + 16;
    /// <summary>All of <see cref="Tables"/>.</summary>
    public const int TotalInts = TokenInfo + 11;

    /// <summary><see cref="Vp9T.Packed"/> followed by the packed scans and the small trees (built once from libvpx's constant tables).</summary>
    public static readonly int[] Tables = Build();

    private static int[] Build()
    {
        var t = Vp9T.Packed;
        var all = new int[TotalInts];
        Array.Copy(t, all, Vp9T.Count);
        // PARTITION_HORZ 1, PARTITION_VERT 2, PARTITION_SPLIT 3
        all[PartHorzTree] = -1; all[PartHorzTree + 1] = -3;
        all[PartVertTree] = -2; all[PartVertTree + 1] = -3;
        // vp9_coef_con_tree: TWO 2, THREE 3, FOUR 4, CATEGORY1..6 5..10
        int[] con = { 2, 6, -2, 4, -3, -4, 8, 10, -5, -6, 12, 14, -7, -8, -9, -10 };
        Array.Copy(con, 0, all, CoefConTree, con.Length);
        // dct_value_tokens: smallest value and extra bits per token (CAT1_MIN_VAL 5 .. CAT6_MIN_VAL 67; 8-bit cat6 = 14 bits)
        int[] minVal = { 0, 0, 2, 3, 4, 5, 7, 11, 19, 35, 67 };
        int[] bits = { 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 14 };
        int[] probs = { 0, 0, 0, 0, 0, Vp9T.Cat1Prob, Vp9T.Cat2Prob, Vp9T.Cat3Prob, Vp9T.Cat4Prob, Vp9T.Cat5Prob, Vp9T.Cat6Prob };
        for (int tok = 2; tok <= 10; tok++)
            all[TokenInfo + tok] = t[Vp9T.PtEnergyClass + tok] | (minVal[tok] << 3) | (bits[tok] << 10) | (probs[tok] << 14);
        for (int tx = 0; tx < 4; tx++)
            for (int type = 0; type < 4; type++)
            {
                int n = 16 << (2 * tx), scan = Vp9ParseKernels.ScanOffset(tx, type), nb = Vp9ParseKernels.NeighborsOffset(tx, type);
                for (int c = 0; c <= n; c++)
                    all[Offset(tx, type) + c] = (c < n ? t[scan + c] : 0) | (t[nb + 2 * c] << 10) | (t[nb + 2 * c + 1] << 20);
                // the token loop computes the band (Vp9ParseKernels.CoefBand) instead of loading it
                int bands = tx == 0 ? Vp9T.CoefbandTrans4x4 : Vp9T.CoefbandTrans8x8plus;
                for (int c = 0; c < n; c++)
                    if (Vp9ParseKernels.CoefBand(tx, c) != t[bands + c]) throw new InvalidOperationException($"VP9 coefficient band {tx}/{c}");
            }
        return all;
    }
}

/// <summary>libvpx FRAME_COUNTS layout (unsigned int arrays in struct order), one block per tile column.</summary>
internal static class Vp9Cnt
{
    public const int YMode = 0, UvMode = 40, Partition = 140, Coef = 204, EobBranch = 2508, SwitchableInterp = 3084,
        InterMode = 3096, IntraInter = 3124, CompInter = 3132, SingleRef = 3142, CompRef = 3162, Tx32 = 3172, Tx16 = 3180,
        Tx8 = 3186, TxTotals = 3190, Skip = 3194, MvJoints = 3200, MvComp = 3204, MvCompStride = 51;
    public const int MvSign = 0, MvClasses = 2, MvClass0 = 13, MvBits = 15, MvClass0Fp = 35, MvFp = 43, MvClass0Hp = 47, MvHp = 49;
    public const int Size = 3306;
}

internal static partial class Vp9ParseKernels
{
    // libvpx enums
    private const int Block8x8 = 3, Block64x64 = 12;
    private const int PartitionNone = 0, PartitionHorz = 1, PartitionVert = 2, PartitionSplit = 3;
    private const int DcPred = 0, NearestMv = 10, NearMv = 11, ZeroMv = 12, NewMv = 13;
    private const int SegLvlAltQ = 0, SegLvlRefFrame = 2, SegLvlSkip = 3;
    private const int TxModeSelect = 4;
    private const int MvUpp = (1 << 14) - 1, MvLow = -(1 << 14);
    private const int MvBorder = 16 << 3;

    // ---------------------------------------------------------------- bool reading

    /// <summary>
    /// vpx_read (libvpx vpx_dsp/bitreader.h), the same arithmetic as <see cref="Vp8BoolDecoderGpu.DecodeBool"/> with the
    /// renormalising shift (vpx_norm[range]) from a leading zero count: range is in [1, 255] here, so the shift is
    /// lzcnt32(range) - 24, one instruction instead of a chain of compares on the parse thread's every bool.
    /// </summary>
    private static int Read(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, int prob)
    {
        uint split = 1u + (((r.Range - 1u) * (uint)prob) >> 8);
        if (r.Count < 0) Vp8BoolDecoderGpu.Fill(ref r, d);
        ulong bigsplit = (ulong)split << (Vp8BoolDecoderGpu.BdValueSize - 8);
        uint range;
        int bit;
        if (r.Value >= bigsplit)
        {
            range = r.Range - split;
            r.Value -= bigsplit;
            bit = 1;
        }
        else
        {
            range = split;
            bit = 0;
        }
        int shift = System.Numerics.BitOperations.LeadingZeroCount(range) - 24;
        r.Range = range << shift;
        r.Value <<= shift;
        r.Count -= shift;
        return bit;
    }

    /// <summary>Where <see cref="ReadSym"/> finds the probabilities: the frame context, the constant tables, the frame arguments.</summary>
    private const int SrcCtx = 0, SrcTab = 1, SrcArg = 2;

    /// <summary>A single bool as a tree: libvpx vp9_mv_class0_tree is {-0, -1}.</summary>
    private const int BoolTree = Vp9T.MvClass0Tree;

    /// <summary>
    /// vpx_read_tree, the kernel's one tree reader (trees in the tables, probabilities from <paramref name="src"/>).
    /// Chrome on Windows compiles WGSL through D3D's compiler, which inlines every function, so the code it compiles is
    /// call sites x body: the parse kernel keeps every bool read (Read and this) to as few call sites as possible and
    /// walks syntax elements in loops instead of repeating reads per element. Each buffer is indexed in its own branch
    /// (WGSL cannot choose at run time between views of different buffers); a constant <paramref name="src"/> folds the
    /// choice away at its call site.
    /// </summary>
    private static int ReadSym(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> t, ArrayView<int> ctx, ArrayView<int> a,
        int treeAt, int src, int probAt)
    {
        int i = 0;
        do
        {
            int at = probAt + (i >> 1);
            int prob = src == SrcCtx ? ctx[at] : src == SrcTab ? t[at] : a[at];
            i = t[treeAt + i + Read(ref r, d, prob)];
        } while (i > 0);
        return -i;
    }

    // ---------------------------------------------------------------- MVs (int_mv.as_int: row low 16 bits, col high)

    private static int MvRow(int mv) => (short)(mv & 0xFFFF);
    private static int MvCol(int mv) => mv >> 16;
    private static int PackMv(int row, int col) => (row & 0xFFFF) | (col << 16);

    private static bool SegFeatureActive(ArrayView<int> a, int segId, int feature) =>
        a[Vp9PArg.SegEnabled] != 0 & (a[Vp9PArg.SegMask + segId] & (1 << feature)) != 0;   // & not &&: see ReadSegmentId

    private static int SegData(ArrayView<int> a, int segId, int feature) => a[Vp9PArg.SegData + segId * 4 + feature];

    // ---------------------------------------------------------------- mode info: segment id, skip, tx size

    private static void SetSegmentId(ArrayView<int> maps, ArrayView<int> a, int miOffset, int xMis, int yMis, int segId)
    {
        int miCols = a[Vp9PArg.MiCols], cur = a[Vp9PArg.CurSegOffset];
        for (int y = 0; y < yMis; y++)
            for (int x = 0; x < xMis; x++) maps[cur + miOffset + y * miCols + x] = segId;
    }

    private static void CopySegmentId(ArrayView<int> maps, ArrayView<int> a, int miOffset, int xMis, int yMis)
    {
        int miCols = a[Vp9PArg.MiCols], cur = a[Vp9PArg.CurSegOffset], last = a[Vp9PArg.LastSegOffset];
        for (int y = 0; y < yMis; y++)
            for (int x = 0; x < xMis; x++) maps[cur + miOffset + y * miCols + x] = maps[last + miOffset + y * miCols + x];
    }

    /// <summary>
    /// read_intra_segment_id (<paramref name="intraOnly"/>) / read_inter_segment_id, one function: the temporal prediction
    /// flag (inter frames with temporal update) and the segment tree are steps of one loop over one tree read.
    /// </summary>
    private static int ReadSegmentId(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> t, ArrayView<int> ctx, ArrayView<int> a,
        ArrayView<int> maps, ArrayView<int> mi, int rec, int above, int left, int miOffset, int xMis, int yMis, bool intraOnly)
    {
        // One exit per helper (here and in every inlined helper of the kernel): inlined, an early return makes WGSL's
        // structured control flow repeat the code after it, and short-circuit conditions (&&, ||) in front of a body
        // do the same to the body, so conditions without side effects use & and |.
        int segId = 0;
        if (a[Vp9PArg.SegEnabled] != 0)
        {
            if (!intraOnly)
            {
                int miCols = a[Vp9PArg.MiCols], last = a[Vp9PArg.LastSegOffset];
                segId = 8;   // dec_get_segment_id: the smallest id over the block's positions in the last map
                for (int y = 0; y < yMis; y++)
                    for (int x = 0; x < xMis; x++)
                    {
                        int s = maps[last + miOffset + y * miCols + x];
                        if (s < segId) segId = s;
                    }
            }
            if (a[Vp9PArg.SegUpdateMap] == 0) CopySegmentId(maps, a, miOffset, xMis, yMis);   // the predicted id (0 intra)
            else
            {
                // step 0: seg_id_predicted (prob seg.pred_probs[ctx]); step 1: the segment tree (seg.tree_probs)
                for (int step = !intraOnly & a[Vp9PArg.SegTemporal] != 0 ? 0 : 1; step < 2; step++)
                {
                    int v = ReadSym(ref r, d, t, ctx, a, step == 0 ? BoolTree : Vp9T.SegmentTree, SrcArg,
                        step == 0 ? Vp9PArg.SegPred + Vp9ParseContexts.SegIdPredContext(mi, above, left) : Vp9PArg.SegTree);
                    if (step == 1) segId = v;
                    else
                    {
                        mi[rec * Vp9Mi.Stride + Vp9Mi.SegIdPredicted] = v;
                        if (v != 0) break;   // predicted
                    }
                }
                SetSegmentId(maps, a, miOffset, xMis, yMis, segId);
            }
        }
        return segId;
    }

    private static int ReadSkip(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> ctx, int fc, ArrayView<int> a,
        ArrayView<int> mi, int above, int left, int segId, ArrayView<int> counts, int cnt)
    {
        int skip = 1;
        if (!SegFeatureActive(a, segId, SegLvlSkip))
        {
            int c = Vp9ParseContexts.SkipContext(mi, above, left);
            skip = Read(ref r, d, ctx[fc + Vp9Ctx.Skip + c]);
            if (cnt >= 0) counts[cnt + Vp9Cnt.Skip + c * 2 + skip]++;
        }
        return skip;
    }

    private static int ReadTxSize(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> ctx, int fc, ArrayView<int> t,
        ArrayView<int> mi, int above, int left, int bsize, int txMode, bool allowSelect, ArrayView<int> counts, int cnt)
    {
        int maxTx = t[Vp9T.MaxTxsizeLookup + bsize];
        int biggest = t[Vp9T.TxModeToBiggestTxSize + txMode];
        int txSize = maxTx < biggest ? maxTx : biggest;
        if (allowSelect & txMode == TxModeSelect & bsize >= Block8x8)
        {
            int c = Vp9ParseContexts.TxSizeContext(mi, above, left, maxTx);
            // get_tx_probs: p8x8[ctx][1], p16x16[ctx][2], p32x32[ctx][3]
            int probs = maxTx == 1 ? fc + Vp9Ctx.Tx8 + c : maxTx == 2 ? fc + Vp9Ctx.Tx16 + c * 2 : fc + Vp9Ctx.Tx32 + c * 3;
            // read_selected_tx_size: unary, tx_probs[k] for the bit after k ones, at most maxTx bits
            txSize = 0;
            while (txSize < maxTx && Read(ref r, d, ctx[probs + txSize]) != 0) txSize++;
            if (cnt >= 0)
            {
                int at = maxTx == 1 ? cnt + Vp9Cnt.Tx8 + c * 2 : maxTx == 2 ? cnt + Vp9Cnt.Tx16 + c * 3 : cnt + Vp9Cnt.Tx32 + c * 4;
                counts[at + txSize]++;
            }
        }
        return txSize;
    }

    // ---------------------------------------------------------------- intra mode info

    /// <summary>get_y_mode of a neighbour: its bmi mode below 8x8, else its mode.</summary>
    private static int YMode(ArrayView<int> mi, int rec, int block) =>
        mi[rec * Vp9Mi.Stride + Vp9Mi.SbType] < Block8x8 ? mi[rec * Vp9Mi.Stride + Vp9Mi.BmiMode + block] : mi[rec * Vp9Mi.Stride + Vp9Mi.Mode];

    /// <summary>Key frame y mode probabilities: vp9_kf_y_mode_prob[above][left] (vp9_above_block_mode / vp9_left_block_mode).</summary>
    private static int KfYModeProbs(ArrayView<int> mi, int rec, int above, int left, int block)
    {
        int aboveMode, leftMode;
        if (block == 0 || block == 1)
            aboveMode = above < 0 || Vp9Mi.IsInter(mi, above) ? DcPred : YMode(mi, above, block + 2);
        else aboveMode = mi[rec * Vp9Mi.Stride + Vp9Mi.BmiMode + block - 2];
        if (block == 0 || block == 2)
            leftMode = left < 0 || Vp9Mi.IsInter(mi, left) ? DcPred : YMode(mi, left, block + 1);
        else leftMode = mi[rec * Vp9Mi.Stride + Vp9Mi.BmiMode + block - 1];
        return Vp9T.KfYModeProb + (aboveMode * 10 + leftMode) * 9;
    }

    /// <summary>
    /// The intra modes of a block, read_intra_frame_mode_info's (<paramref name="kf"/>: key frame / intra-only
    /// probabilities from the tables, not counted) or read_intra_block_mode_info's (frame context, counted): one y mode
    /// per 4x4 sub-block below 8x8 (the 4x8 / 8x4 halves copied), else one, then the uv mode, all steps of one loop over
    /// one tree read.
    /// </summary>
    private static void ReadIntraModes(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> ctx, int fc, ArrayView<int> t,
        ArrayView<int> a, ArrayView<int> mi, int rec, int above, int left, int bsize, int bmodeWl, int bmodeHl, bool kf,
        ArrayView<int> counts, int cnt)
    {
        int b = rec * Vp9Mi.Stride;
        bool sub8 = bsize < Block8x8;
        int n4w = sub8 ? 1 << bmodeWl : 2, n4h = sub8 ? 1 << bmodeHl : 2;
        int sizeGroup = sub8 ? 0 : t[Vp9T.SizeGroupLookup + bsize];
        int j = 0, mode = 0;
        while (true)
        {
            bool uv = j >= 4;   // the y modes are sub-blocks 0..3, then the uv mode
            int probAt = uv ? (kf ? Vp9T.KfUvModeProb : fc + Vp9Ctx.UvMode) + mode * 9
                : kf ? KfYModeProbs(mi, rec, above, left, j) : fc + Vp9Ctx.YMode + sizeGroup * 9;
            int m = ReadSym(ref r, d, t, ctx, a, Vp9T.IntraModeTree, kf ? SrcTab : SrcCtx, probAt);
            if (!kf & cnt >= 0) counts[cnt + (uv ? Vp9Cnt.UvMode + mode * 10 : Vp9Cnt.YMode + sizeGroup * 10) + m]++;
            if (uv)
            {
                mi[b + Vp9Mi.UvMode] = m;
                break;
            }
            if (sub8)
            {
                mi[b + Vp9Mi.BmiMode + j] = m;
                if (n4h == 2) mi[b + Vp9Mi.BmiMode + j + 2] = m;
                if (n4w == 2) mi[b + Vp9Mi.BmiMode + j + 1] = m;
            }
            mode = m;   // the block's mode is its last sub-block's
            mi[b + Vp9Mi.Mode] = m;
            int idx = (j & 1) + n4w, idy = j >> 1;
            if (idx >= 2)
            {
                idx = 0;
                idy += n4h;
            }
            j = idy >= 2 ? 4 : idy * 2 + idx;
        }
    }
}
