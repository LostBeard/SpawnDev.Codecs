// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// VP9 inter frame tile symbols in parallel, for Vp8BoolEncoderParallelGpu: the symbols the serial pass
// (Vp9InterEntropy.StepKernel / EncodeLeafSlot) codes, in its order, so the coded tile is byte-identical, and the same
// symbol counts for backward adaptation.
//
// The serial pass carries above / left context arrays and decides each leaf's inter mode as it goes. Two passes replace it:
//   1. ModeKernel, one launch per anti-diagonal of 16x16 leaves (row + col = d): the leaf's inter mode. The choice reads
//      the mode context, i.e. the modes of the first two MV reference candidates (for these block sizes the leaf above
//      and the leaf left: one diagonal earlier), and the candidate MVs (FindMvRefs: neighbours' references and MVs and
//      the previous frame's map, all fixed before entropy coding). It writes the mode into the mode info, the frame's
//      MV map (what the next frame reads as the previous frame's MVs) and the leaf's NEARESTMV for the MV difference.
//   2. SymbolsKernel, one thread per 16x16 leaf slot (count mode, then write mode per coder window): every context from
//      the neighbours. Partition contexts as in Vp9KeyframeSymbolsKernel (12 / 14 by the neighbour leaf's shape; the left
//      array restarts at every superblock row). Skip / is-inter / reference contexts: the decoder's own helpers on the
//      mode info. Coefficient contexts: a transform's cells hold (eob > 0) of the last transform that wrote them, 0 for a
//      skipped leaf: from the leaf above (its only transform, or its lower one when stacked), the leaf left (its only
//      one, or its right one when side by side), or this leaf's first transform of the plane.
// The symbol counts (count mode only) are added with atomics: sums, so the order does not matter.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.Codecs.Video.Vp9.Decoder;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>Per-launch arguments of <see cref="Vp9InterSymbols.SymbolsKernel"/>.</summary>
public struct Vp9InterSymbolParams
{
    /// <summary>0 = count each slot's symbols (and add the frame's symbol counts), 1 = write the window's symbols.</summary>
    public int Write;
    /// <summary>Stream index of the window's first symbol and the window's size.</summary>
    public int WindowBase, WindowSymbols;
    /// <summary>The tile (superblocks): first column and row, width. The thread index is the slot inside the tile.</summary>
    public int SbColStart, SbRowStart, TileSbCols;
}

internal static class Vp9InterSymbols
{
    /// <summary>
    /// Most symbols one slot can code: partitions 3 + 3 + 1, skip, is-inter, reference, 3 mode bools, an MV difference
    /// (3 joint bools + per component sign, 10 class bools, 10 integer bits, 3 fraction bools), and the coefficients of
    /// at most 6 transforms (an end-of-block symbol each, at most 22 per coefficient over 384 coefficients); plus the
    /// stream's marker bit.
    /// </summary>
    public const int MaxSymbolsPerSlot = 7 + 3 + 3 + 3 + 2 * 24 + 6 + 22 * 384 + 1;

    /// <summary>(mi row, mi col) of slot <paramref name="q"/> of superblock (sbRow, sbCol).</summary>
    private static void SlotPosition(int sbRow, int sbCol, int q, out int miRow, out int miCol)
    {
        miRow = sbRow * 8 + ((q >> 3) & 1) * 4 + ((q >> 1) & 1) * 2;
        miCol = sbCol * 8 + ((q >> 2) & 1) * 4 + (q & 1) * 2;
    }

    /// <summary>The slot (sb * 16 + q) of the 16x16 leaf whose top-left mode info is (miRow, miCol), both even.</summary>
    private static int SlotOf(int miRow, int miCol, int sbCols)
    {
        int sb = (miRow >> 3) * sbCols + (miCol >> 3);
        int q = ((miRow >> 2) & 1) * 8 + ((miCol >> 2) & 1) * 4 + ((miRow >> 1) & 1) * 2 + ((miCol >> 1) & 1);
        return sb * 16 + q;
    }

    // ---- pass 1: the inter modes, one anti-diagonal of 16x16 leaves per launch ----

    /// <summary>
    /// Thread i: the i-th leaf of anti-diagonal <paramref name="diag"/> (leaf row + leaf col = diag, rows from the top).
    /// The mode choice of the serial pass (read_inter_block_mode_info, written): of the modes that give the leaf's MV
    /// without coding it, the most probable; else NEWMV.
    /// </summary>
    public static void ModeKernel(Index1D i, ArrayView<int> mi, ArrayView<int> a, ArrayView<int> ctx, ArrayView<int> t,
        ArrayView<int> maps, ArrayView<int> nearestBuf, int diag)
    {
        int miRows = a[Vp9PArg.MiRows], miCols = a[Vp9PArg.MiCols];
        int leafCols = (miCols + 1) >> 1, leafRows = (miRows + 1) >> 1;
        int rMin = diag - (leafCols - 1);
        if (rMin < 0) rMin = 0;
        int lr = rMin + i, lc = diag - lr;
        bool inside = lr < leafRows & lc >= 0;
        if (!inside) return;
        int miRow = lr * 2, miCol = lc * 2;
        int shape = Vp9InterEncodeKernels.LeafShape(miRow, miCol, miRows, miCols);
        int wide = shape == 0 | shape == 1 ? 2 : 1, high = shape == 0 | shape == 2 ? 2 : 1;
        int rec = miRow * miCols + miCol, b = rec * Vp9Mi.Stride;
        int bsize = mi[b + Vp9Mi.SbType];
        Vp9InterEntropy.TileColumns(a, miCol, out int tileMiCol, out int tileMiColEnd);
        var p = new Vp9ParseKernels.BlockPos
        {
            MiRow = miRow, MiCol = miCol, MiColStart = tileMiCol, MiColEnd = tileMiColEnd,
            ToTop = -(miRow * 8 * 8), ToBottom = (miRows - high - miRow) * 8 * 8,
            ToLeft = -(miCol * 8 * 8), ToRight = (miCols - wide - miCol) * 8 * 8,
        };
        int modeCtx = ModeContext(mi, a, t, p, bsize);
        int modeProbs = Vp9Ctx.InterMode + modeCtx * 3;
        Vp9ParseKernels.FindMvRefs(mi, maps, a, t, p, bsize, Vp9ParseKernels.NearMv, Vp9Mi.LastFrame, -1, out int l0, out int l1);
        int nearest = Vp9ParseKernels.LowerMvPrecision(l0, false), near = Vp9ParseKernels.LowerMvPrecision(l1, false);
        int mv = mi[b + Vp9Mi.Mv0];
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
        nearestBuf[SlotOf(miRow, miCol, a[Vp9PArg.SbCols])] = nearest;
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
    }

    /// <summary>The inter mode context: counter_to_context of the first two candidates' modes.</summary>
    private static int ModeContext(ArrayView<int> mi, ArrayView<int> a, ArrayView<int> t, in Vp9ParseKernels.BlockPos p, int bsize)
    {
        int search = Vp9T.MvRefBlocks + bsize * 16, counter = 0;
        for (int i = 0; i < 2; i++)
        {
            int cand = Vp9ParseKernels.Candidate(mi, a, p, t[search + i * 2], t[search + i * 2 + 1]);
            if (cand >= 0) counter += t[Vp9T.Mode2Counter + mi[cand * Vp9Mi.Stride + Vp9Mi.Mode]];
        }
        return t[Vp9T.CounterToContext + counter];
    }

    // ---- pass 2: the symbols ----

    private static void Count(ArrayView<int> counts, int cnt, int at)
    {
        if (cnt >= 0) Atomic.Add(ref counts[cnt + at], 1);
    }

    /// <summary>Partition context state at (miRow, miCol) for level <paramref name="sizeIdx"/>: sizeIdx * 4 + left * 2 + above.</summary>
    private static int PartitionCtx(int sizeIdx, int miRow, int miCol, int miRows, int miCols, int tileMiCol)
    {
        int above = 0, left = 0;
        if (miRow > 0)
        {
            int s = Vp9InterEncodeKernels.LeafShape((miRow - 1) & ~1, miCol & ~1, miRows, miCols);
            above = s == 2 | s == 3 ? 14 : 12;
        }
        if (miCol > tileMiCol)
        {
            int s = Vp9InterEncodeKernels.LeafShape(miRow & ~1, (miCol - 1) & ~1, miRows, miCols);
            left = s == 1 | s == 3 ? 14 : 12;
        }
        return sizeIdx * 4 + ((left >> sizeIdx) & 1) * 2 + ((above >> sizeIdx) & 1);
    }

    private static void PutSplit(ref Vp8BoolSymbolSink sink, ArrayView<ushort> window, ArrayView<int> ctx, int sizeIdx, int miRow, int miCol,
        int miRows, int miCols, ArrayView<int> counts, int cnt, int tileMiCol)
    {
        int half = 1 << sizeIdx >> 1;
        bool hasRows = miRow + half < miRows, hasCols = miCol + half < miCols;
        int pctx = PartitionCtx(sizeIdx, miRow, miCol, miRows, miCols, tileMiCol);
        int probs = Vp9Ctx.Partition + pctx * 3;
        Count(counts, cnt, Vp9Cnt.Partition + pctx * 4 + 3);   // PARTITION_SPLIT: counted when implied too (read_partition)
        if (hasRows & hasCols)
        {
            for (int k = 0; k < 3; k++) Vp8BoolSymbolSink.Put(ref sink, window, 1, ctx[probs + k]);
        }
        else if (hasCols) Vp8BoolSymbolSink.Put(ref sink, window, 1, ctx[probs + 1]);
        else if (hasRows) Vp8BoolSymbolSink.Put(ref sink, window, 1, ctx[probs + 2]);
    }

    /// <summary>1 when the leaf at (miRow, miCol) coded a nonzero coefficient in transform <paramref name="half"/> of <paramref name="plane"/>.</summary>
    private static int LeafNonzero(ArrayView<short> levels, ArrayView<int> mi, ArrayView<int> a, int miRow, int miCol, int plane, int shape, int half)
    {
        int miCols = a[Vp9PArg.MiCols];
        int nz = 0;
        if (mi[(miRow * miCols + miCol) * Vp9Mi.Stride + Vp9Mi.Skip] == 0)
        {
            int slot = SlotOf(miRow, miCol, a[Vp9PArg.SbCols]);
            int s = plane == 0 ? 0 : 1;
            int tx = (shape == 0 ? 2 : 1) - s;
            int count = 16 << (tx << 1);
            int at = (slot >> 4) * a[Vp9PArg.DqStride] + (slot & 15) * (plane == 0 ? 256 : 64) + half * count;
            if (plane == 1) at += a[Vp9PArg.DqU];
            if (plane == 2) at += a[Vp9PArg.DqV];
            for (int k = 0; k < count; k++) nz |= levels[at + k];
        }
        return nz != 0 ? 1 : 0;
    }

    /// <summary>
    /// Thread = 16x16 leaf slot (sb * 16 + q). Count mode: the slot's symbol count into slotCounts, and the frame's
    /// symbol counts (when the frame adapts). Write mode: the slot's symbols that fall in the coder's window.
    /// </summary>
    public static void SymbolsKernel(Index1D slot, ArrayView<short> levels, ArrayView<int> mi, ArrayView<int> a, ArrayView<int> ctx,
        ArrayView<int> t, ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts, ArrayView<int> counts, ArrayView<int> nearestBuf,
        ArrayView<int> slotCounts, ArrayView<int> offsets, ArrayView<ushort> window, Vp9InterSymbolParams sp)
    {
        int miRows = a[Vp9PArg.MiRows], miCols = a[Vp9PArg.MiCols];
        // slot = the superblock in the tile (raster inside the tile) * 16 + its z-order position
        int sbLocal = slot >> 4, q = slot & 15;
        int sbRowLocal = sbLocal / sp.TileSbCols;
        int sbRow = sp.SbRowStart + sbRowLocal, sbCol = sp.SbColStart + sbLocal - sbRowLocal * sp.TileSbCols;
        SlotPosition(sbRow, sbCol, q, out int miRow, out int miCol);
        int tileMiCol = sp.SbColStart * 8;
        int tileMiColEnd = (sp.SbColStart + sp.TileSbCols) * 8;
        if (tileMiColEnd > miCols) tileMiColEnd = miCols;
        bool inside = miRow < miRows & miCol < miCols;
        var tokenCache = LocalMemory.Allocate<byte>(256);
        var sink = new Vp8BoolSymbolSink();
        int cnt = -1;
        if (sp.Write != 0)
        {
            int start = offsets[slot];
            int end = start + slotCounts[slot];
            int hi = sp.WindowBase + sp.WindowSymbols;
            bool outside = end <= sp.WindowBase | start >= hi;
            if (outside) return;
            sink.Next = start;
            sink.Lo = sp.WindowBase;
            sink.Hi = hi;
        }
        else if (a[Vp9PArg.CountsOn] != 0) cnt = Vp9InterEntropy.Counts;
        if (slot == 0) Vp8BoolSymbolSink.Put(ref sink, window, 0, 128);   // the marker bit (each tile has a bool coder of its own)
        if (inside)
        {
            EmitLeafSlot(ref sink, window, levels, mi, a, ctx, t, byteConsts, ushortConsts, counts, cnt, nearestBuf, tokenCache,
                q, miRow, miCol, miRows, miCols, tileMiCol, tileMiColEnd);
        }
        if (sp.Write == 0) slotCounts[slot] = sink.Next;
    }

    private static void EmitLeafSlot(ref Vp8BoolSymbolSink sink, ArrayView<ushort> window, ArrayView<short> levels, ArrayView<int> mi,
        ArrayView<int> a, ArrayView<int> ctx, ArrayView<int> t, ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        ArrayView<int> counts, int cnt, ArrayView<int> nearestBuf, ArrayView<byte> tokenCache, int q, int miRow, int miCol, int miRows, int miCols,
        int tileMiCol, int tileMiColEnd)
    {
        int q32 = q >> 2, q16 = q & 3;
        int miRow64 = miRow & ~7, miCol64 = miCol & ~7;
        int miRow32 = miRow & ~3, miCol32 = miCol & ~3;
        if (q == 0) PutSplit(ref sink, window, ctx, 3, miRow64, miCol64, miRows, miCols, counts, cnt, tileMiCol);
        if (q16 == 0) PutSplit(ref sink, window, ctx, 2, miRow32, miCol32, miRows, miCols, counts, cnt, tileMiCol);

        int shape = Vp9InterEncodeKernels.LeafShape(miRow, miCol, miRows, miCols);
        if (shape == 3) Count(counts, cnt, Vp9Cnt.Partition + PartitionCtx(1, miRow, miCol, miRows, miCols, tileMiCol) * 4 + 3);   // the implied SPLIT of the 16x16
        int pctx16 = PartitionCtx(shape == 3 ? 0 : 1, miRow, miCol, miRows, miCols, tileMiCol);
        int partition = shape == 1 ? 1 : shape == 2 ? 2 : 0;
        Vp8BoolSymbolSink.Put(ref sink, window, 0, ctx[Vp9Ctx.Partition + pctx16 * 3 + partition]);
        Count(counts, cnt, Vp9Cnt.Partition + pctx16 * 4 + partition);

        // mode info
        int wide = shape == 0 | shape == 1 ? 2 : 1, high = shape == 0 | shape == 2 ? 2 : 1;
        int grid = a[Vp9PArg.GridOffset];
        int rec = miRow * miCols + miCol, b = rec * Vp9Mi.Stride;
        int above = -1, left = -1;
        if (miRow != 0) above = mi[grid + (miRow - 1) * miCols + miCol];
        if (miCol > tileMiCol) left = mi[grid + miRow * miCols + miCol - 1];   // none at the tile's left edge
        int skip = mi[b + Vp9Mi.Skip];
        int skipCtx = Vp9ParseContexts.SkipContext(mi, above, left);
        int interCtx = Vp9ParseContexts.IntraInterContext(mi, above, left);
        int refCtx = Vp9ParseContexts.SingleRefP1Context(mi, above, left);
        Vp8BoolSymbolSink.Put(ref sink, window, skip, ctx[Vp9Ctx.Skip + skipCtx]);
        Vp8BoolSymbolSink.Put(ref sink, window, 1, ctx[Vp9Ctx.IntraInter + interCtx]);
        Vp8BoolSymbolSink.Put(ref sink, window, 0, ctx[Vp9Ctx.SingleRef + refCtx * 2]);
        Count(counts, cnt, Vp9Cnt.Skip + skipCtx * 2 + skip);
        Count(counts, cnt, Vp9Cnt.IntraInter + interCtx * 2 + 1);
        Count(counts, cnt, Vp9Cnt.SingleRef + refCtx * 4);

        int bsize = mi[b + Vp9Mi.SbType];
        var p = new Vp9ParseKernels.BlockPos
        {
            MiRow = miRow, MiCol = miCol, MiColStart = tileMiCol, MiColEnd = tileMiColEnd,
            ToTop = -(miRow * 8 * 8), ToBottom = (miRows - high - miRow) * 8 * 8,
            ToLeft = -(miCol * 8 * 8), ToRight = (miCols - wide - miCol) * 8 * 8,
        };
        int modeCtx = ModeContext(mi, a, t, p, bsize);
        int modeProbs = Vp9Ctx.InterMode + modeCtx * 3;
        int mode = mi[b + Vp9Mi.Mode];
        Count(counts, cnt, Vp9Cnt.InterMode + modeCtx * 4 + (mode - Vp9ParseKernels.NearestMv));
        Vp8BoolSymbolSink.Put(ref sink, window, mode == Vp9ParseKernels.ZeroMv ? 0 : 1, ctx[modeProbs]);
        if (mode != Vp9ParseKernels.ZeroMv)
        {
            Vp8BoolSymbolSink.Put(ref sink, window, mode == Vp9ParseKernels.NearestMv ? 0 : 1, ctx[modeProbs + 1]);
            if (mode != Vp9ParseKernels.NearestMv) Vp8BoolSymbolSink.Put(ref sink, window, mode == Vp9ParseKernels.NearMv ? 0 : 1, ctx[modeProbs + 2]);
        }
        if (mode == Vp9ParseKernels.NewMv)
        {
            int mv = mi[b + Vp9Mi.Mv0];
            int nearest = nearestBuf[SlotOf(miRow, miCol, a[Vp9PArg.SbCols])];
            int dRow = Vp9ParseKernels.MvRow(mv) - Vp9ParseKernels.MvRow(nearest), dCol = Vp9ParseKernels.MvCol(mv) - Vp9ParseKernels.MvCol(nearest);
            int joint = (dRow != 0 ? 2 : 0) | (dCol != 0 ? 1 : 0);
            Vp8BoolSymbolSink.Put(ref sink, window, joint == 0 ? 0 : 1, ctx[Vp9Ctx.MvJoints]);
            if (joint != 0)
            {
                Vp8BoolSymbolSink.Put(ref sink, window, joint == 1 ? 0 : 1, ctx[Vp9Ctx.MvJoints + 1]);
                if (joint != 1) Vp8BoolSymbolSink.Put(ref sink, window, joint == 2 ? 0 : 1, ctx[Vp9Ctx.MvJoints + 2]);
            }
            for (int k = 0; k < 2; k++)
                if ((joint & (2 >> k)) != 0)
                    PutMvComponent(ref sink, window, ctx, t, a, Vp9Ctx.MvComp + k * Vp9Ctx.MvCompStride, k == 0 ? dRow : dCol);
            if (cnt >= 0)
            {
                Count(counts, cnt, Vp9Cnt.MvJoints + joint);
                for (int k = 0; k < 2; k++)
                    if ((joint & (2 >> k)) != 0)
                        CountMvComponent(k == 0 ? dRow : dCol, counts, cnt + Vp9Cnt.MvComp + k * Vp9Cnt.MvCompStride);
            }
        }

        // the coefficient tokens (none when skipped); contexts from the neighbours
        if (skip != 0) return;
        int shapeAbove = miRow > 0 ? Vp9InterEncodeKernels.LeafShape(miRow - 2, miCol, miRows, miCols) : 0;
        int shapeLeft = miCol > tileMiCol ? Vp9InterEncodeKernels.LeafShape(miRow, miCol - 2, miRows, miCols) : 0;
        int slot = SlotOf(miRow, miCol, a[Vp9PArg.SbCols]);
        int dqSb = (slot >> 4) * a[Vp9PArg.DqStride];
        bool two = shape == 1 | shape == 2;
        int jobs = two ? 6 : 3;
        int firstNz = 0;
        for (int j = 0; j < jobs; j++)
        {
            int plane = two ? j >> 1 : j;
            int half = two ? j & 1 : 0;
            int s = plane == 0 ? 0 : 1;
            int tx = (shape == 0 ? 2 : 1) - s;
            int count = 16 << (tx << 1);
            int aboveNz, leftNz;
            if (shape == 2 & half == 1) aboveNz = firstNz;
            else if (miRow == 0) aboveNz = 0;
            else aboveNz = LeafNonzero(levels, mi, a, miRow - 2, miCol, plane, shapeAbove, shapeAbove == 2 ? 1 : 0);
            if (shape == 1 & half == 1) leftNz = firstNz;
            else if (miCol <= tileMiCol) leftNz = 0;
            else leftNz = LeafNonzero(levels, mi, a, miRow, miCol - 2, plane, shapeLeft, shapeLeft == 1 ? 1 : 0);
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
            int typeRef = (tx * 2 + (plane == 0 ? 0 : 1)) * 2 + 1;
            int at = dqSb + (slot & 15) * (plane == 0 ? 256 : 64) + half * count;
            if (plane == 1) at += a[Vp9PArg.DqU];
            if (plane == 2) at += a[Vp9PArg.DqV];
            int eob = PutBlockAdaptive(ref sink, window, levels.SubView(at, count),
                ushortConsts.SubView(scanAt, count), ushortConsts.SubView(neighborsAt, count * 2),
                ctx, Vp9Ctx.Coef + typeRef * 108,
                counts, cnt >= 0 ? cnt + Vp9Cnt.Coef + typeRef * 144 : -1, cnt + Vp9Cnt.EobBranch + typeRef * 36,
                byteConsts.SubView(Vp9KeyframeConstantsGpu.CoefConstsOffset, Vp9KeyframeConstantsGpu.CoefConstsLength),
                tokenCache, count, aboveNz + leftNz, tx == 0 ? 1 : 0);
            if (half == 0) firstNz = eob > 0 ? 1 : 0;
        }
    }

    /// <summary>read_mv's component, written (Vp9InterEntropy.PushMvComponent).</summary>
    private static void PutMvComponent(ref Vp8BoolSymbolSink sink, ArrayView<ushort> window, ArrayView<int> ctx, ArrayView<int> t, ArrayView<int> a, int comp, int v)
    {
        Vp8BoolSymbolSink.Put(ref sink, window, v < 0 ? 1 : 0, ctx[comp + Vp9Ctx.MvSign]);
        int z = (v < 0 ? -v : v) - 1;
        int c = 0, x = z >> 3;
        for (int i = 0; i < 10; i++)
            if (x > 1)
            {
                x >>= 1;
                c++;
            }
        if (z >= 8192) c = 10;
        int o = z - (c != 0 ? 2 << (c + 2) : 0);
        int d = o >> 3, fr = (o >> 1) & 3;
        int bits = a[Vp9EArg.ClassTok + c * 2], len = a[Vp9EArg.ClassTok + c * 2 + 1];
        int node = 0;
        for (int i = len - 1; i >= 0; i--)
        {
            int bit = (bits >> i) & 1;
            Vp8BoolSymbolSink.Put(ref sink, window, bit, ctx[comp + Vp9Ctx.MvClasses + (node >> 1)]);
            node = t[Vp9T.MvClassTree + node + bit];
        }
        if (c == 0) Vp8BoolSymbolSink.Put(ref sink, window, d, ctx[comp + Vp9Ctx.MvClass0]);
        for (int i = 0; i < c; i++) Vp8BoolSymbolSink.Put(ref sink, window, (d >> i) & 1, ctx[comp + Vp9Ctx.MvBits + i]);
        int fp = comp + (c == 0 ? Vp9Ctx.MvClass0Fp + d * 3 : Vp9Ctx.MvFp);
        Vp8BoolSymbolSink.Put(ref sink, window, fr == 0 ? 0 : 1, ctx[fp]);
        if (fr != 0)
        {
            Vp8BoolSymbolSink.Put(ref sink, window, fr == 1 ? 0 : 1, ctx[fp + 1]);
            if (fr != 1) Vp8BoolSymbolSink.Put(ref sink, window, fr == 2 ? 0 : 1, ctx[fp + 2]);
        }
    }

    /// <summary>vp9_inc_mv's component (Vp9ParseKernels.IncMvComponent) with atomic adds.</summary>
    private static void CountMvComponent(int v, ArrayView<int> counts, int comp)
    {
        int s = v < 0 ? 1 : 0;
        Atomic.Add(ref counts[comp + Vp9Cnt.MvSign + s], 1);
        int z = (s != 0 ? -v : v) - 1;
        int c;
        if (z >= 2 * 4096) c = 10;
        else
        {
            int x = z >> 3;
            c = x == 0 ? 0 : 31 - System.Numerics.BitOperations.LeadingZeroCount((uint)x);
        }
        int o = z - (c != 0 ? 2 << (c + 2) : 0);
        Atomic.Add(ref counts[comp + Vp9Cnt.MvClasses + c], 1);
        int dd = o >> 3, f = (o >> 1) & 3, e = o & 1;
        if (c == 0)
        {
            Atomic.Add(ref counts[comp + Vp9Cnt.MvClass0 + dd], 1);
            Atomic.Add(ref counts[comp + Vp9Cnt.MvClass0Fp + dd * 4 + f], 1);
            Atomic.Add(ref counts[comp + Vp9Cnt.MvClass0Hp + e], 1);
        }
        else
        {
            for (int i = 0; i < c; i++) Atomic.Add(ref counts[comp + Vp9Cnt.MvBits + i * 2 + ((dd >> i) & 1)], 1);
            Atomic.Add(ref counts[comp + Vp9Cnt.MvFp + f], 1);
            Atomic.Add(ref counts[comp + Vp9Cnt.MvHp + e], 1);
        }
    }

    /// <summary>Vp9BlockCoefEncoderGpu.EncodeBlockAdaptive, written to the sink, its counts added with atomics.</summary>
    private static int PutBlockAdaptive(ref Vp8BoolSymbolSink sink, ArrayView<ushort> window, ArrayView<short> coefs,
        ArrayView<ushort> scan, ArrayView<ushort> neighbors, ArrayView<int> probs, int probsBase,
        ArrayView<int> counts, int cntBase, int eobBase, ArrayView<byte> consts, ArrayView<byte> tokenCache,
        int maxCoefs, int initialCtx, int isTx4x4)
    {
        for (int i = 0; i < maxCoefs; i++) tokenCache[i] = 0;
        int eob = 0;
        for (int i = maxCoefs - 1; i >= 0; i--)
        {
            int raster = scan[i];
            if (coefs[raster] != 0) { eob = i + 1; break; }
        }
        bool counting = cntBase >= 0;
        int c = 0;
        int ctx = initialCtx;
        bool more = true;
        while (more)
        {
            int band = isTx4x4 != 0 ? consts[Vp9BlockCoefEncoderGpu.Band4x4Offset + c] : consts[Vp9BlockCoefEncoderGpu.Band8x8PlusOffset + c];
            int cell = band * 6 + ctx;
            int m0 = probs[probsBase + cell * 3];
            int m1 = probs[probsBase + cell * 3 + 1];
            int m2 = probs[probsBase + cell * 3 + 2];
            if (counting) Atomic.Add(ref counts[eobBase + cell], 1);
            if (c == eob)
            {
                Vp8BoolSymbolSink.Put(ref sink, window, 0, m0);                               // EOB
                if (counting) Atomic.Add(ref counts[cntBase + cell * 4 + 3], 1);
                return eob;
            }
            Vp8BoolSymbolSink.Put(ref sink, window, 1, m0);
            bool zero = coefs[(int)scan[c]] == 0;
            while (zero)
            {
                Vp8BoolSymbolSink.Put(ref sink, window, 0, m1);                               // ZERO
                if (counting) Atomic.Add(ref counts[cntBase + cell * 4], 1);
                c++;
                // c < eob <= maxCoefs here: a zero before the last nonzero coefficient
                band = isTx4x4 != 0 ? consts[Vp9BlockCoefEncoderGpu.Band4x4Offset + c] : consts[Vp9BlockCoefEncoderGpu.Band8x8PlusOffset + c];
                cell = band * 6 + Vp9BlockCoefEncoderGpu.GetCoefContext(neighbors, tokenCache, c);
                m1 = probs[probsBase + cell * 3 + 1];
                m2 = probs[probsBase + cell * 3 + 2];
                zero = coefs[(int)scan[c]] == 0;
            }
            Vp8BoolSymbolSink.Put(ref sink, window, 1, m1);
            int value = coefs[(int)scan[c]];
            int magnitude = value < 0 ? -value : value;
            byte token;
            if (magnitude == 1)
            {
                Vp8BoolSymbolSink.Put(ref sink, window, 0, m2);
                if (counting) Atomic.Add(ref counts[cntBase + cell * 4 + 1], 1);
                token = (byte)Vp9CoefToken.One;
            }
            else
            {
                Vp8BoolSymbolSink.Put(ref sink, window, 1, m2);
                if (counting) Atomic.Add(ref counts[cntBase + cell * 4 + 2], 1);
                token = Vp9BlockCoefEncoderGpu.MagnitudeToToken(magnitude, 0);
                Vp9KeyframeSymbolsKernel.PutTreeWalk(ref sink, window, consts, Vp9BlockCoefEncoderGpu.Pareto8FullOffset + (m2 - 1) * 8, token);
                if (token >= (byte)Vp9CoefToken.Category1) Vp9KeyframeSymbolsKernel.PutCategory(ref sink, window, consts, token, magnitude);
            }
            Vp8BoolSymbolSink.Put(ref sink, window, value < 0 ? 1 : 0, 128);
            tokenCache[(int)scan[c]] = consts[Vp9BlockCoefEncoderGpu.PtEnergyClassOffset + token];
            c++;
            more = c < maxCoefs;
            if (more) ctx = Vp9BlockCoefEncoderGpu.GetCoefContext(neighbors, tokenCache, c);
        }
        return eob;
    }
}
