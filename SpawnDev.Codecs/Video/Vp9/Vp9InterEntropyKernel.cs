// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// VP9 inter frame encoder, the entropy pass: the tile's bool-coded bitstream from the frame's mode info and levels
// (Plans/PLAN-VP9-Encoder-Inter-Frames.md). One thread walks the superblocks in bounded, resumable launches, like the
// keyframe pass (Vp9FrameEntropyKernel), and writes what the decoder's parse reads, in its order: the write-side
// mirror of libvpx v1.17.0 (BSD-3-Clause) vp9/decoder/vp9_decodeframe.c decode_partition / parse_block and
// vp9/decoder/vp9_decodemv.c read_inter_frame_mode_info / read_inter_block_mode_info / read_mv, as ported in
// Decoder/Vp9ParseKernels*.cs. It reads the same mode info array the decoder's kernels use, so the contexts
// (Vp9ParseContexts) and the MV candidates (Vp9ParseKernels.FindMvRefs) are the decoder's own code.
//
// Per leaf: the partition symbols that precede it, skip, is_inter (1), the reference (LAST), the inter mode, the MV
// difference of a NEWMV, then the coefficient tokens (none when skipped). The inter mode is chosen here from the MV
// the leaf already has: NEARESTMV / NEARMV / ZEROMV when the MV equals that candidate (the most probable one of those),
// else NEWMV. The pass writes the mode it coded into the leaf's record: the following leaves' mode contexts read it.
//
// Every mode info bool goes into a small queue that ONE loop writes (one bool encoder call site: Chrome's D3D
// compiler inlines every call).

using ILGPU;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.Codecs.Video.Vp9.Decoder;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>The inter entropy pass's frame arguments: the parse arguments its decoder helpers read, then its own.</summary>
internal static class Vp9EArg
{
    /// <summary>
    /// vp9_mv_class_tree tokens, per class (11): the path's bits (first decision in the highest bit) and its length,
    /// built on the host from the decoder's tree (<see cref="BuildClassTokens"/>).
    /// </summary>
    public const int ClassTok = Vp9PArg.Count;
    /// <summary>Log2 of the frame's tile columns (Vp9TileGrid): a leaf's left neighbour and MV candidates stay in its tile.</summary>
    public const int TileLog2Cols = ClassTok + 22;
    public const int Count = TileLog2Cols + 1;

    /// <summary>(bits, length) of every leaf of the tree at <paramref name="treeAt"/> in <see cref="Vp9T.Packed"/>.</summary>
    public static void BuildClassTokens(int[] a)
    {
        var t = Vp9T.Packed;
        void Walk(int node, int bits, int len)
        {
            for (int bit = 0; bit < 2; bit++)
            {
                int next = t[Vp9T.MvClassTree + node + bit];
                int b = (bits << 1) | bit;
                if (next <= 0)
                {
                    a[ClassTok + -next * 2] = b;
                    a[ClassTok + -next * 2 + 1] = len + 1;
                }
                else Walk(next, b, len + 1);
            }
        }
        Walk(0, 0, 0);
    }
}

internal static class Vp9InterEntropy
{
    /// <summary>
    /// The mode-info columns [start, end) of the tile column holding <paramref name="miCol"/> (libvpx get_tile_offset:
    /// superblock (c * sbCols) >> log2, the decoder's formula).
    /// </summary>
    public static void TileColumns(ArrayView<int> a, int miCol, out int start, out int end)
    {
        int log2 = a[Vp9EArg.TileLog2Cols], sbCols = a[Vp9PArg.SbCols], miCols = a[Vp9PArg.MiCols];
        int sb = miCol >> 3, n = 1 << log2;
        int s = 0, e = sbCols;
        for (int c = 0; c < n; c++)
        {
            int cs = (c * sbCols) >> log2, ce = ((c + 1) * sbCols) >> log2;
            bool here = sb >= cs & sb < ce;
            if (here) { s = cs; e = ce; }
        }
        start = s * 8;
        end = e * 8 < miCols ? e * 8 : miCols;
    }

    /// <summary>The widest picture, in 8-pixel columns (the keyframe pass's limit).</summary>
    public const int MaxMiCols = Vp9FrameEntropyKernel.MaxMiColsAligned;
    /// <summary>Leaf slots (16 per superblock, z-order) one launch codes.</summary>
    public const int LeavesPerLaunch = Vp9FrameEntropyKernel.LeavesPerLaunch;

    // The pass's state (ints, device memory, carried between launches): the walk, the bool encoder, then the contexts:
    // above entropy contexts (3 planes x 2 per mi column), above partition contexts, left entropy contexts (3 planes x
    // 16), left partition contexts.
    public const int StStarted = 0, StDone = 1, StNextSlot = 2, StEnc = 3;
    public const int Above = 8, AbovePart = Above + 6 * MaxMiCols, Left = AbovePart + MaxMiCols, LeftPart = Left + 48;
    /// <summary>The frame's symbol counts (the decoder's FRAME_COUNTS layout, <see cref="Vp9Cnt"/>, one tile column):
    /// kept in the state buffer, so the kernel binds no more buffers than before.</summary>
    public const int Counts = LeftPart + 8;
    public const int StateInts = Counts + Vp9Cnt.Size;

    private const int QueueLength = 80;

    /// <summary>One bool for the queue: the bit and its probability.</summary>
    private static void Push(ArrayView<int> queue, ref int n, int bit, int prob)
    {
        queue[n] = bit | (prob << 1);
        n++;
    }

    /// <summary>The partition symbol's context (libvpx partition_plane_context): its three probabilities are at
    /// <c>Vp9Ctx.Partition + context * 3</c>, its four counts at <c>Vp9Cnt.Partition + context * 4</c>.</summary>
    internal static int PartitionCtx(ArrayView<int> st, int sizeIdx, int miRow, int miCol)
    {
        int leftBit = (st[LeftPart + (miRow & 7)] >> sizeIdx) & 1;
        int aboveBit = (st[AbovePart + miCol] >> sizeIdx) & 1;
        return sizeIdx * 4 + leftBit * 2 + aboveBit;
    }

    /// <summary>
    /// The partition symbol of a block that is split further (64x64 and 32x32 here): SPLIT (bits 1, 1, 1) inside the
    /// picture; at the bottom edge one bool at probs[1] (SPLIT, not HORZ), at the right edge one at probs[2] (SPLIT, not
    /// VERT); at the corner SPLIT is implied.
    /// </summary>
    private static void PushSplit(ArrayView<int> queue, ref int n, ArrayView<int> st, ArrayView<int> ctx, int sizeIdx, int miRow, int miCol,
        int miRows, int miCols, ArrayView<int> counts, int cnt)
    {
        int half = 1 << sizeIdx >> 1;   // in mi: 4 for 64x64, 2 for 32x32
        bool hasRows = miRow + half < miRows, hasCols = miCol + half < miCols;
        int pctx = PartitionCtx(st, sizeIdx, miRow, miCol);
        int probs = Vp9Ctx.Partition + pctx * 3;
        if (cnt >= 0) counts[cnt + Vp9Cnt.Partition + pctx * 4 + 3]++;   // PARTITION_SPLIT: counted when implied too (read_partition)
        if (hasRows & hasCols)
        {
            for (int k = 0; k < 3; k++) Push(queue, ref n, 1, ctx[probs + k]);
        }
        else if (hasCols) Push(queue, ref n, 1, ctx[probs + 1]);
        else if (hasRows) Push(queue, ref n, 1, ctx[probs + 2]);
    }

    /// <summary>read_mv's component, written: sign, class, the integer bits, the fraction (high precision is off).</summary>
    private static void PushMvComponent(ArrayView<int> queue, ref int n, ArrayView<int> ctx, ArrayView<int> t, ArrayView<int> a, int comp, int v)
    {
        Push(queue, ref n, v < 0 ? 1 : 0, ctx[comp + Vp9Ctx.MvSign]);
        int z = (v < 0 ? -v : v) - 1;
        // vp9_get_mv_class: class 10 from 8192, else floor(log2(z >> 3)) (0 below 2)
        int c = 0, x = z >> 3;
        for (int i = 0; i < 10; i++)
            if (x > 1)
            {
                x >>= 1;
                c++;
            }
        if (z >= 8192) c = 10;
        int o = z - (c != 0 ? 2 << (c + 2) : 0);   // the offset in the class (mv_class_base)
        int d = o >> 3, fr = (o >> 1) & 3;        // integer part, quarter-pel fraction; o & 1 = 1 (hp implied)
        // the class: a walk of vp9_mv_class_tree along the class's token
        int bits = a[Vp9EArg.ClassTok + c * 2], len = a[Vp9EArg.ClassTok + c * 2 + 1];
        int node = 0;
        for (int i = len - 1; i >= 0; i--)
        {
            int bit = (bits >> i) & 1;
            Push(queue, ref n, bit, ctx[comp + Vp9Ctx.MvClasses + (node >> 1)]);
            node = t[Vp9T.MvClassTree + node + bit];
        }
        // the integer part: class 0 one bool, class c its c bits, lowest first
        if (c == 0) Push(queue, ref n, d, ctx[comp + Vp9Ctx.MvClass0]);
        for (int i = 0; i < c; i++) Push(queue, ref n, (d >> i) & 1, ctx[comp + Vp9Ctx.MvBits + i]);
        // the fraction: vp9_mv_fp_tree (0: 0; 1: 1 0; 2: 1 1 0; 3: 1 1 1)
        int fp = comp + (c == 0 ? Vp9Ctx.MvClass0Fp + d * 3 : Vp9Ctx.MvFp);
        Push(queue, ref n, fr == 0 ? 0 : 1, ctx[fp]);
        if (fr != 0)
        {
            Push(queue, ref n, fr == 1 ? 0 : 1, ctx[fp + 1]);
            if (fr != 1) Push(queue, ref n, fr == 2 ? 0 : 1, ctx[fp + 2]);
        }
    }

    /// <summary>
    /// One 16x16 position of the superblock z-order walk (slot = sb * 16 + q32 * 4 + q16): the partition symbols that
    /// precede it, then the leaf. Positions outside the picture code nothing.
    /// </summary>
    private static void EncodeLeafSlot(ref Vp8BoolEncoderGpuState state, ArrayView<byte> outBuf, ArrayView<short> levels, ArrayView<int> mi,
        ArrayView<int> a, ArrayView<int> ctx, ArrayView<int> t, ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts, ArrayView<int> st,
        ArrayView<int> maps, ArrayView<int> counts, ArrayView<int> queue, ArrayView<byte> tokenCache, int sbRow, int sbCol, int q,
        int tileMiCol, int tileMiColEnd)
    {
        int miCols = a[Vp9PArg.MiCols], miRows = a[Vp9PArg.MiRows];
        // the frame's symbol counts when the frame adapts (counts = the state buffer, at Counts)
        int cnt = a[Vp9PArg.CountsOn] != 0 ? Counts : -1;
        int sb = sbRow * a[Vp9PArg.SbCols] + sbCol;
        int miRow64 = sbRow * 8, miCol64 = sbCol * 8;
        int q32 = q >> 2, q16 = q & 3;
        int miRow32 = miRow64 + (q32 >> 1) * 4, miCol32 = miCol64 + (q32 & 1) * 4;
        int miRow = miRow32 + (q16 >> 1) * 2, miCol = miCol32 + (q16 & 1) * 2;
        // (the caller skips positions outside the picture: slot 0 of a superblock and the first slot of a 32x32 block
        // are inside whenever anything under them is, so the symbols queued below are always followed by a leaf)

        int n = 0;
        if (q == 0)
        {
            if (miCol64 == tileMiCol)
                for (int k = 0; k < 56; k++) st[Left + k] = 0;   // left entropy + partition contexts: each superblock row of the tile
            PushSplit(queue, ref n, st, ctx, 3, miRow64, miCol64, miRows, miCols, counts, cnt);
        }
        if (q16 == 0) PushSplit(queue, ref n, st, ctx, 2, miRow32, miCol32, miRows, miCols, counts, cnt);

        // the leaf and its partition symbol at the 16x16 block: NONE inside the picture; at the bottom edge one bool at
        // probs[1] (HORZ, the top 16X8), at the right edge one at probs[2] (VERT, the left 8X16); at the corner SPLIT is
        // implied and the top-left 8x8 codes its own partition, NONE
        int shape = Vp9InterEncodeKernels.LeafShape(miRow, miCol, miRows, miCols);
        if (shape == 3 & cnt >= 0) counts[cnt + Vp9Cnt.Partition + PartitionCtx(st, 1, miRow, miCol) * 4 + 3]++;   // the implied SPLIT of the 16x16
        int pctx16 = PartitionCtx(st, shape == 3 ? 0 : 1, miRow, miCol);
        int probs16 = Vp9Ctx.Partition + pctx16 * 3;
        int partition = 0;   // PARTITION_NONE; HORZ 1 at the bottom edge, VERT 2 at the right edge
        if (shape == 1) partition = 1;
        if (shape == 2) partition = 2;
        Push(queue, ref n, 0, ctx[probs16 + partition]);
        if (cnt >= 0) counts[cnt + Vp9Cnt.Partition + pctx16 * 4 + partition]++;
        // dec_update_partition_context: partition_context_lookup[subsize] {above, left} over the coded block's width
        int pa = shape == 2 | shape == 3 ? 14 : 12, pl = shape == 1 | shape == 3 ? 14 : 12;
        int span = shape == 3 ? 1 : 2;
        for (int i = 0; i < span; i++)
        {
            st[AbovePart + miCol + i] = pa;
            st[LeftPart + ((miRow + i) & 7)] = pl;
        }

        // ---- mode info (read_inter_frame_mode_info) ----
        int wide = shape == 0 | shape == 1 ? 2 : 1, high = shape == 0 | shape == 2 ? 2 : 1;   // in mi
        int grid = a[Vp9PArg.GridOffset];
        int rec = miRow * miCols + miCol, b = rec * Vp9Mi.Stride;
        int above = -1, left = -1;
        if (miRow != 0) above = mi[grid + (miRow - 1) * miCols + miCol];
        if (miCol > tileMiCol) left = mi[grid + miRow * miCols + miCol - 1];   // none at the tile's left edge (libvpx set_mi_row_col)
        int skip = mi[b + Vp9Mi.Skip];
        int skipCtx = Vp9ParseContexts.SkipContext(mi, above, left);
        int interCtx = Vp9ParseContexts.IntraInterContext(mi, above, left);
        int refCtx = Vp9ParseContexts.SingleRefP1Context(mi, above, left);
        Push(queue, ref n, skip, ctx[Vp9Ctx.Skip + skipCtx]);
        Push(queue, ref n, 1, ctx[Vp9Ctx.IntraInter + interCtx]);   // is_inter
        // tx_size: not coded (tx_mode ALLOW_32X32: the largest the block allows)
        // read_ref_frames, single reference: one bool, 0 = LAST
        Push(queue, ref n, 0, ctx[Vp9Ctx.SingleRef + refCtx * 2]);
        if (cnt >= 0)
        {
            counts[cnt + Vp9Cnt.Skip + skipCtx * 2 + skip]++;
            counts[cnt + Vp9Cnt.IntraInter + interCtx * 2 + 1]++;
            counts[cnt + Vp9Cnt.SingleRef + refCtx * 4]++;   // single_ref p1 = 0
        }

        // read_inter_block_mode_info: the mode context from the first two candidates, then the candidates
        int bsize = mi[b + Vp9Mi.SbType];
        var p = new Vp9ParseKernels.BlockPos
        {
            MiRow = miRow, MiCol = miCol, MiColStart = tileMiCol, MiColEnd = tileMiColEnd,
            ToTop = -(miRow * 8 * 8), ToBottom = (miRows - high - miRow) * 8 * 8,
            ToLeft = -(miCol * 8 * 8), ToRight = (miCols - wide - miCol) * 8 * 8,
        };
        int search = Vp9T.MvRefBlocks + bsize * 16, counter = 0;
        for (int i = 0; i < 2; i++)
        {
            int cand = Vp9ParseKernels.Candidate(mi, a, p, t[search + i * 2], t[search + i * 2 + 1]);
            if (cand >= 0) counter += t[Vp9T.Mode2Counter + mi[cand * Vp9Mi.Stride + Vp9Mi.Mode]];
        }
        int modeCtx = t[Vp9T.CounterToContext + counter];
        int modeProbs = Vp9Ctx.InterMode + modeCtx * 3;
        // NEARMV's search returns both candidates: list0 is what NEARESTMV's and NEWMV's (early break) return
        Vp9ParseKernels.FindMvRefs(mi, maps, a, t, p, bsize, Vp9ParseKernels.NearMv, Vp9Mi.LastFrame, -1, out int l0, out int l1);
        int nearest = Vp9ParseKernels.LowerMvPrecision(l0, false), near = Vp9ParseKernels.LowerMvPrecision(l1, false);
        int mv = mi[b + Vp9Mi.Mv0];
        // vp9_inter_mode_tree: ZEROMV 0; NEARESTMV 1 0; NEARMV 1 1 0; NEWMV 1 1 1. Of the modes that give this MV
        // without coding it, the most probable (weights = probability x 2^24).
        int p0 = ctx[modeProbs], p1 = ctx[modeProbs + 1], p2 = ctx[modeProbs + 2];
        int mode = Vp9ParseKernels.NewMv, weight = -1;
        if (mv == 0)
        {
            mode = Vp9ParseKernels.ZeroMv;
            weight = p0 * 65536;
        }
        int wNearest = (256 - p0) * p1 * 256, wNear = (256 - p0) * (256 - p1) * p2;
        if (mv == nearest & wNearest > weight)
        {
            mode = Vp9ParseKernels.NearestMv;
            weight = wNearest;
        }
        if (mv == near & wNear > weight) mode = Vp9ParseKernels.NearMv;
        mi[b + Vp9Mi.Mode] = mode;
        if (cnt >= 0) counts[cnt + Vp9Cnt.InterMode + modeCtx * 4 + (mode - Vp9ParseKernels.NearestMv)]++;
        // the MV map of the frame, as the decoder writes it (vp9_read_mode_info): what the candidate search of the
        // next frame reads as the previous frame's MVs
        int curMv = a[Vp9PArg.CurMvOffset];
        for (int y = 0; y < high; y++)
            for (int x = 0; x < wide; x++)
            {
                int at = curMv + (rec + y * miCols + x) * Vp9PLayout.MvStride;
                maps[at] = mi[b + Vp9Mi.Ref0];
                maps[at + 1] = mi[b + Vp9Mi.Ref1];
                maps[at + 2] = mi[b + Vp9Mi.Mv0];
                maps[at + 3] = mi[b + Vp9Mi.Mv1];
            }
        Push(queue, ref n, mode == Vp9ParseKernels.ZeroMv ? 0 : 1, p0);
        if (mode != Vp9ParseKernels.ZeroMv)
        {
            Push(queue, ref n, mode == Vp9ParseKernels.NearestMv ? 0 : 1, p1);
            if (mode != Vp9ParseKernels.NearestMv) Push(queue, ref n, mode == Vp9ParseKernels.NearMv ? 0 : 1, p2);
        }
        // interp_filter: not coded (the frame's EIGHTTAP)
        if (mode == Vp9ParseKernels.NewMv)
        {
            // read_mv: the difference to the best reference MV (NEARESTMV's candidate), row then col
            int dRow = Vp9ParseKernels.MvRow(mv) - Vp9ParseKernels.MvRow(nearest), dCol = Vp9ParseKernels.MvCol(mv) - Vp9ParseKernels.MvCol(nearest);
            int joint = (dRow != 0 ? 2 : 0) | (dCol != 0 ? 1 : 0);
            // vp9_mv_joint_tree: ZERO 0; HNZVZ (col) 1 0; HZVNZ (row) 1 1 0; HNZVNZ 1 1 1
            Push(queue, ref n, joint == 0 ? 0 : 1, ctx[Vp9Ctx.MvJoints]);
            if (joint != 0)
            {
                Push(queue, ref n, joint == 1 ? 0 : 1, ctx[Vp9Ctx.MvJoints + 1]);
                if (joint != 1) Push(queue, ref n, joint == 2 ? 0 : 1, ctx[Vp9Ctx.MvJoints + 2]);
            }
            for (int k = 0; k < 2; k++)
                if ((joint & (2 >> k)) != 0)
                    PushMvComponent(queue, ref n, ctx, t, a, Vp9Ctx.MvComp + k * Vp9Ctx.MvCompStride, k == 0 ? dRow : dCol);
            if (cnt >= 0)
            {
                // vp9_inc_mv on the coded difference (the decoder's own helper)
                counts[cnt + Vp9Cnt.MvJoints + joint]++;
                for (int k = 0; k < 2; k++)
                    if ((joint & (2 >> k)) != 0)
                        Vp9ParseKernels.IncMvComponent(k == 0 ? dRow : dCol, counts, cnt + Vp9Cnt.MvComp + k * Vp9Cnt.MvCompStride);
            }
        }

        // every queued bool, in order
        for (int i = 0; i < n; i++) Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, queue[i] & 1, queue[i] >> 1);

        // ---- the coefficient tokens, or the skipped block's contexts (dec_reset_skip_context) ----
        int dqSb = sb * a[Vp9PArg.DqStride];
        bool two = shape == 1 | shape == 2;
        int jobs = two ? 6 : 3;
        for (int j = 0; j < jobs; j++)
        {
            int plane = two ? j >> 1 : j;
            int half = two ? j & 1 : 0;
            int across = shape == 1 ? half : 0, down = shape == 2 ? half : 0;
            int s = plane == 0 ? 0 : 1;
            int tx = (shape == 0 ? 2 : 1) - s;
            int cells = 1 << tx, count = 16 << (tx << 1);
            int aboveAt = Above + plane * 2 * MaxMiCols + ((miCol * 2) >> s) + across * cells;
            int leftAt = Left + plane * 16 + (((miRow * 2) & 15) >> s) + down * cells;
            int nz = 0;
            if (skip == 0)
            {
                int ac = 0, lc = 0;
                for (int k = 0; k < cells; k++)
                {
                    ac |= st[aboveAt + k];
                    lc |= st[leftAt + k];
                }
                int scanAt, neighborsAt;
                if (tx == 2)
                {
                    scanAt = Vp9KeyframeConstantsGpu.Scan16x16Offset;
                    neighborsAt = Vp9KeyframeConstantsGpu.Neighbors16x16Offset;
                }
                else if (tx == 1)
                {
                    scanAt = Vp9KeyframeConstantsGpu.Scan8x8Offset;
                    neighborsAt = Vp9KeyframeConstantsGpu.Neighbors8x8Offset;
                }
                else
                {
                    scanAt = Vp9KeyframeConstantsGpu.Scan4x4Offset;
                    neighborsAt = Vp9KeyframeConstantsGpu.Neighbors4x4Offset;
                }
                // the frame context's probabilities and the frame's counts of [tx][plane type][inter]
                int typeRef = (tx * 2 + (plane == 0 ? 0 : 1)) * 2 + 1;
                int at = dqSb + q * (plane == 0 ? 256 : 64) + half * count;
                if (plane == 1) at += a[Vp9PArg.DqU];
                if (plane == 2) at += a[Vp9PArg.DqV];
                int eob = Vp9BlockCoefEncoderGpu.EncodeBlockAdaptive(ref state, outBuf, levels.SubView(at, count),
                    ushortConsts.SubView(scanAt, count), ushortConsts.SubView(neighborsAt, count * 2),
                    ctx, Vp9Ctx.Coef + typeRef * 108,
                    counts, cnt >= 0 ? cnt + Vp9Cnt.Coef + typeRef * 144 : -1, cnt + Vp9Cnt.EobBranch + typeRef * 36,
                    byteConsts.SubView(Vp9KeyframeConstantsGpu.CoefConstsOffset, Vp9KeyframeConstantsGpu.CoefConstsLength),
                    tokenCache, count,
                    initialCtx: (ac != 0 ? 1 : 0) + (lc != 0 ? 1 : 0),
                    isTx4x4: tx == 0 ? 1 : 0);
                nz = eob > 0 ? 1 : 0;
            }
            for (int k = 0; k < cells; k++)
            {
                st[aboveAt + k] = nz;
                st[leftAt + k] = nz;
            }
        }
    }

    /// <summary>
    /// One bounded step of the walk of one tile: the next <paramref name="leaves"/> leaf slots. One thread.
    /// <paramref name="tileOrigin"/> = first superblock column | first superblock row &lt;&lt; 16, <paramref name="tileSize"/>
    /// = superblock columns | rows &lt;&lt; 16; the slots are the tile's superblocks in raster order, 16 each.
    /// </summary>
    public static void StepKernel(Index1D index, ArrayView<byte> outBuf, ArrayView<long> outLen, ArrayView<short> levels, ArrayView<int> mi,
        ArrayView<int> a, ArrayView<int> ctx, ArrayView<int> t, ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts, ArrayView<int> st,
        ArrayView<int> maps, int leaves, int tileOrigin, int tileSize)
    {
        if (st[StDone] != 0) return;
        var queue = LocalMemory.Allocate<int>(QueueLength);
        var tokenCache = LocalMemory.Allocate<byte>(256);
        Vp8BoolEncoderGpuState state;
        if (st[StStarted] == 0)
        {
            state = Vp8BoolEncoderGpu.Init();
            Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, 128);   // the marker bit
            st[StStarted] = 1;
        }
        else state = Vp8BoolEncoderGpu.Load(st, StEnc);

        int tileSbCols = tileSize & 0xFFFF;
        int slotCount = (tileSize >> 16) * tileSbCols * 16;
        int slot = st[StNextSlot];
        int slotEnd = slot + leaves;
        if (slotEnd > slotCount) slotEnd = slotCount;
        int miRows = a[Vp9PArg.MiRows], miCols = a[Vp9PArg.MiCols];
        int tileMiCol = (tileOrigin & 0xFFFF) * 8;
        int tileMiColEnd = ((tileOrigin & 0xFFFF) + tileSbCols) * 8;
        if (tileMiColEnd > miCols) tileMiColEnd = miCols;
        for (; slot < slotEnd; slot++)
        {
            // the slot's 16x16 position; outside the picture nothing is coded
            int sbLocal = slot >> 4, q = slot & 15;
            int sbRowLocal = sbLocal / tileSbCols;
            int sbRow = (tileOrigin >> 16) + sbRowLocal, sbCol = (tileOrigin & 0xFFFF) + sbLocal - sbRowLocal * tileSbCols;
            int row = sbRow * 8 + ((q >> 3) & 1) * 4 + ((q >> 1) & 1) * 2, col = sbCol * 8 + ((q >> 2) & 1) * 4 + (q & 1) * 2;
            if (row < miRows & col < miCols)
                EncodeLeafSlot(ref state, outBuf, levels, mi, a, ctx, t, byteConsts, ushortConsts, st, maps, st, queue, tokenCache,
                    sbRow, sbCol, q, tileMiCol, tileMiColEnd);
        }
        if (slot >= slotCount)
        {
            Vp8BoolEncoderGpu.StopVp9(ref state, outBuf);
            outLen[0] = state.OutLen;
            st[StDone] = 1;
        }
        st[StNextSlot] = slot;
        Vp8BoolEncoderGpu.Save(st, StEnc, ref state);
    }
}
