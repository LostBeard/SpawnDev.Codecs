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

    /// <summary><see cref="Vp9T.Packed"/> followed by the packed scans (built once from libvpx's constant tables).</summary>
    public static readonly int[] Tables = Build();

    private static int[] Build()
    {
        var t = Vp9T.Packed;
        var all = new int[Vp9T.Count + Ints];
        Array.Copy(t, all, Vp9T.Count);
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
    private const int Block4x4 = 0, Block4x8 = 1, Block8x4 = 2, Block8x8 = 3, Block64x64 = 12;
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

    /// <summary>vpx_read_tree: tree and probabilities are int views (packed tables / context).</summary>
    private static int ReadTree(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> tree, int treeAt, ArrayView<int> probs, int probAt)
    {
        int i = 0;
        while ((i = tree[treeAt + i + Read(ref r, d, probs[probAt + (i >> 1)])]) > 0) { }
        return -i;
    }

    // ---------------------------------------------------------------- MVs (int_mv.as_int: row low 16 bits, col high)

    private static int MvRow(int mv) => (short)(mv & 0xFFFF);
    private static int MvCol(int mv) => mv >> 16;
    private static int PackMv(int row, int col) => (row & 0xFFFF) | (col << 16);

    private static bool SegFeatureActive(ArrayView<int> a, int segId, int feature) =>
        a[Vp9PArg.SegEnabled] != 0 && (a[Vp9PArg.SegMask + segId] & (1 << feature)) != 0;

    private static int SegData(ArrayView<int> a, int segId, int feature) => a[Vp9PArg.SegData + segId * 4 + feature];

    // ---------------------------------------------------------------- mode info: segment id, skip, tx size

    private static int ReadSegmentId(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> t, ArrayView<int> a) =>
        ReadTree(ref r, d, t, Vp9T.SegmentTree, a, Vp9PArg.SegTree);

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

    private static int ReadIntraSegmentId(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> t, ArrayView<int> a,
        ArrayView<int> maps, int miOffset, int xMis, int yMis)
    {
        if (a[Vp9PArg.SegEnabled] == 0) return 0;
        if (a[Vp9PArg.SegUpdateMap] == 0)
        {
            CopySegmentId(maps, a, miOffset, xMis, yMis);
            return 0;
        }
        int segId = ReadSegmentId(ref r, d, t, a);
        SetSegmentId(maps, a, miOffset, xMis, yMis, segId);
        return segId;
    }

    private static int ReadInterSegmentId(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> t, ArrayView<int> a,
        ArrayView<int> maps, ArrayView<int> mi, int rec, int above, int left, int miOffset, int xMis, int yMis)
    {
        if (a[Vp9PArg.SegEnabled] == 0) return 0;
        int miCols = a[Vp9PArg.MiCols], last = a[Vp9PArg.LastSegOffset];
        int predicted = 8;   // dec_get_segment_id: the smallest id over the block's positions in the last map
        for (int y = 0; y < yMis; y++)
            for (int x = 0; x < xMis; x++)
            {
                int s = maps[last + miOffset + y * miCols + x];
                if (s < predicted) predicted = s;
            }
        if (a[Vp9PArg.SegUpdateMap] == 0)
        {
            CopySegmentId(maps, a, miOffset, xMis, yMis);
            return predicted;
        }
        int segId;
        if (a[Vp9PArg.SegTemporal] != 0)
        {
            int prob = a[Vp9PArg.SegPred + Vp9ParseContexts.SegIdPredContext(mi, above, left)];
            int pred = Read(ref r, d, prob);
            mi[rec * Vp9Mi.Stride + Vp9Mi.SegIdPredicted] = pred;
            segId = pred != 0 ? predicted : ReadSegmentId(ref r, d, t, a);
        }
        else segId = ReadSegmentId(ref r, d, t, a);
        SetSegmentId(maps, a, miOffset, xMis, yMis, segId);
        return segId;
    }

    private static int ReadSkip(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> ctx, int fc, ArrayView<int> a,
        ArrayView<int> mi, int above, int left, int segId, ArrayView<int> counts, int cnt)
    {
        if (SegFeatureActive(a, segId, SegLvlSkip)) return 1;
        int c = Vp9ParseContexts.SkipContext(mi, above, left);
        int skip = Read(ref r, d, ctx[fc + Vp9Ctx.Skip + c]);
        if (cnt >= 0) counts[cnt + Vp9Cnt.Skip + c * 2 + skip]++;
        return skip;
    }

    private static int ReadTxSize(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> ctx, int fc, ArrayView<int> t,
        ArrayView<int> mi, int above, int left, int bsize, int txMode, bool allowSelect, ArrayView<int> counts, int cnt)
    {
        int maxTx = t[Vp9T.MaxTxsizeLookup + bsize];
        if (allowSelect && txMode == TxModeSelect && bsize >= Block8x8)
        {
            int c = Vp9ParseContexts.TxSizeContext(mi, above, left, maxTx);
            // get_tx_probs: p8x8[ctx][1], p16x16[ctx][2], p32x32[ctx][3]
            int probs = maxTx == 1 ? fc + Vp9Ctx.Tx8 + c : maxTx == 2 ? fc + Vp9Ctx.Tx16 + c * 2 : fc + Vp9Ctx.Tx32 + c * 3;
            int tx = Read(ref r, d, ctx[probs]);
            if (tx != 0 && maxTx >= 2)
            {
                tx += Read(ref r, d, ctx[probs + 1]);
                if (tx != 1 && maxTx >= 3) tx += Read(ref r, d, ctx[probs + 2]);
            }
            if (cnt >= 0)
            {
                int at = maxTx == 1 ? cnt + Vp9Cnt.Tx8 + c * 2 : maxTx == 2 ? cnt + Vp9Cnt.Tx16 + c * 3 : cnt + Vp9Cnt.Tx32 + c * 4;
                counts[at + tx]++;
            }
            return tx;
        }
        int biggest = t[Vp9T.TxModeToBiggestTxSize + txMode];
        return maxTx < biggest ? maxTx : biggest;
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

    private static void ReadIntraFrameModeInfo(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> ctx, int fc,
        ArrayView<int> t, ArrayView<int> a, ArrayView<int> mi, ArrayView<int> maps, int rec, int above, int left, int bsize,
        int txMode, int miOffset, int xMis, int yMis, ArrayView<int> counts, int cnt)
    {
        int b = rec * Vp9Mi.Stride;
        int segId = ReadIntraSegmentId(ref r, d, t, a, maps, miOffset, xMis, yMis);
        mi[b + Vp9Mi.SegmentId] = segId;
        mi[b + Vp9Mi.Skip] = ReadSkip(ref r, d, ctx, fc, a, mi, above, left, segId, counts, cnt);
        mi[b + Vp9Mi.TxSize] = ReadTxSize(ref r, d, ctx, fc, t, mi, above, left, bsize, txMode, true, counts, cnt);
        mi[b + Vp9Mi.Ref0] = Vp9Mi.IntraFrame;
        mi[b + Vp9Mi.Ref1] = Vp9Mi.NoRefFrame;
        int mode;
        if (bsize == Block4x4)
        {
            for (int i = 0; i < 4; i++)
                mi[b + Vp9Mi.BmiMode + i] = ReadTree(ref r, d, t, Vp9T.IntraModeTree, t, KfYModeProbs(mi, rec, above, left, i));
            mode = mi[b + Vp9Mi.BmiMode + 3];
        }
        else if (bsize == Block4x8)
        {
            int m0 = ReadTree(ref r, d, t, Vp9T.IntraModeTree, t, KfYModeProbs(mi, rec, above, left, 0));
            mi[b + Vp9Mi.BmiMode] = m0;
            mi[b + Vp9Mi.BmiMode + 2] = m0;
            mode = ReadTree(ref r, d, t, Vp9T.IntraModeTree, t, KfYModeProbs(mi, rec, above, left, 1));
            mi[b + Vp9Mi.BmiMode + 1] = mode;
            mi[b + Vp9Mi.BmiMode + 3] = mode;
        }
        else if (bsize == Block8x4)
        {
            int m0 = ReadTree(ref r, d, t, Vp9T.IntraModeTree, t, KfYModeProbs(mi, rec, above, left, 0));
            mi[b + Vp9Mi.BmiMode] = m0;
            mi[b + Vp9Mi.BmiMode + 1] = m0;
            mode = ReadTree(ref r, d, t, Vp9T.IntraModeTree, t, KfYModeProbs(mi, rec, above, left, 2));
            mi[b + Vp9Mi.BmiMode + 2] = mode;
            mi[b + Vp9Mi.BmiMode + 3] = mode;
        }
        else mode = ReadTree(ref r, d, t, Vp9T.IntraModeTree, t, KfYModeProbs(mi, rec, above, left, 0));
        mi[b + Vp9Mi.Mode] = mode;
        mi[b + Vp9Mi.UvMode] = ReadTree(ref r, d, t, Vp9T.IntraModeTree, t, Vp9T.KfUvModeProb + mode * 9);
    }

    private static int ReadIntraModeY(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> ctx, int fc, ArrayView<int> t,
        int sizeGroup, ArrayView<int> counts, int cnt)
    {
        int m = ReadTree(ref r, d, t, Vp9T.IntraModeTree, ctx, fc + Vp9Ctx.YMode + sizeGroup * 9);
        if (cnt >= 0) counts[cnt + Vp9Cnt.YMode + sizeGroup * 10 + m]++;
        return m;
    }

    private static void ReadIntraBlockModeInfo(ref Vp8BoolDecoderGpuState r, ArrayView<byte> d, ArrayView<int> ctx, int fc,
        ArrayView<int> t, ArrayView<int> mi, int rec, int bsize, ArrayView<int> counts, int cnt)
    {
        int b = rec * Vp9Mi.Stride, mode;
        if (bsize == Block4x4)
        {
            for (int i = 0; i < 4; i++) mi[b + Vp9Mi.BmiMode + i] = ReadIntraModeY(ref r, d, ctx, fc, t, 0, counts, cnt);
            mode = mi[b + Vp9Mi.BmiMode + 3];
        }
        else if (bsize == Block4x8)
        {
            int m0 = ReadIntraModeY(ref r, d, ctx, fc, t, 0, counts, cnt);
            mi[b + Vp9Mi.BmiMode] = m0;
            mi[b + Vp9Mi.BmiMode + 2] = m0;
            mode = ReadIntraModeY(ref r, d, ctx, fc, t, 0, counts, cnt);
            mi[b + Vp9Mi.BmiMode + 1] = mode;
            mi[b + Vp9Mi.BmiMode + 3] = mode;
        }
        else if (bsize == Block8x4)
        {
            int m0 = ReadIntraModeY(ref r, d, ctx, fc, t, 0, counts, cnt);
            mi[b + Vp9Mi.BmiMode] = m0;
            mi[b + Vp9Mi.BmiMode + 1] = m0;
            mode = ReadIntraModeY(ref r, d, ctx, fc, t, 0, counts, cnt);
            mi[b + Vp9Mi.BmiMode + 2] = mode;
            mi[b + Vp9Mi.BmiMode + 3] = mode;
        }
        else mode = ReadIntraModeY(ref r, d, ctx, fc, t, t[Vp9T.SizeGroupLookup + bsize], counts, cnt);
        mi[b + Vp9Mi.Mode] = mode;
        int uv = ReadTree(ref r, d, t, Vp9T.IntraModeTree, ctx, fc + Vp9Ctx.UvMode + mode * 9);
        if (cnt >= 0) counts[cnt + Vp9Cnt.UvMode + mode * 10 + uv]++;
        mi[b + Vp9Mi.UvMode] = uv;
        mi[b + Vp9Mi.Interp] = Vp9Mi.SwitchableFilters;
        mi[b + Vp9Mi.Ref0] = Vp9Mi.IntraFrame;
        mi[b + Vp9Mi.Ref1] = Vp9Mi.NoRefFrame;
    }
}
