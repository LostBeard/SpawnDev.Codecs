// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 key frame tile symbols, one thread per 16x16 leaf slot, for Vp8BoolEncoderParallelGpu. Emits the symbols the
// serial walker (Vp9FrameEntropyKernel: EncodeLeafSlot / EncodeBlock16x16 / EncodeLeaf / EncodePlaneCoefs, with
// Vp9BlockCoefEncoderGpu.EncodeBlock) codes for the same slot, in the same order, so the coded tile is byte-identical.
//
// The serial walker carries its contexts in above / left arrays it updates leaf by leaf. Every one of them is a function
// of the neighbouring leaves, all known before entropy coding starts, so each slot derives its own:
//   - skip context: every leaf codes skip = 0 and the arrays start at 0: always 0;
//   - Y mode context: every leaf is DC_PRED (0) and the arrays start at 0: always row 0;
//   - partition context: written only by the leaves (libvpx dec_update_partition_context), 12 / 14 by the leaf's shape.
//     The value read at (miRow, miCol) is the one of the leaf directly above (the most recent writer of that column in
//     z-order / raster order) or directly left (the left array restarts at every superblock row, i.e. at miCol 0); 0 at
//     the frame's top / left edge. Shapes come from the frame's display size (Vp9FrameSequentialEncodeKernel.BoundaryShape):
//     a leaf above another is never on the bottom row (shape 0 or 2, both write both columns), a leaf left of another is
//     never on the last column (shape 0 or 1, both write both rows);
//   - coefficient contexts: a transform's cells hold (eob > 0) of the last transform that wrote them, i.e. "has a
//     nonzero coefficient". From the leaf above: its only transform (shape 0) or its lower one (shape 2, stacked); from
//     the leaf left: its only transform (shape 0) or its right one (shape 1, side by side); inside the leaf, the first
//     transform of the plane for the second one. The cells a shape-2 leaf above leaves unwritten are only read by
//     leaves on the last column, which read none of them (same for shape 1 on the left).
//
// Two modes: count (each slot's symbol count) and write (the symbols whose stream index falls in the coder's window).
// The stream's first symbol is the VP9 marker bit (bool 0 at 128) the serial walker codes before the first slot.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>Per-frame arguments of <see cref="Vp9KeyframeSymbolsKernel"/>.</summary>
public struct Vp9KeySymbolParams
{
    /// <summary>Macroblock columns / rows of the working (64-aligned) frame.</summary>
    public int MbCols, MbRows;
    /// <summary>Display size in 8x8 mode-info units.</summary>
    public int FrameMiCols, FrameMiRows;
    /// <summary>0 = count each slot's symbols, 1 = write the symbols in [WindowBase, WindowBase + WindowSymbols).</summary>
    public int Write;
    /// <summary>Stream index of the window's first symbol and the window's size.</summary>
    public int WindowBase, WindowSymbols;
}

/// <summary>VP9 key frame tile symbols in parallel (one thread per 16x16 leaf slot) for <see cref="Vp8BoolEncoderParallelGpu"/>.</summary>
public sealed class Vp9KeyframeSymbolsKernel
{
    /// <summary>
    /// Most symbols one slot can code: partitions 3 + 3 + 1, skip, Y and UV mode, and per transform an end-of-block
    /// symbol plus at most 22 per coefficient (more-coefficients, zero / non-zero, one, 4 tree bits, 14 Cat6 bits,
    /// sign) over 256 + 64 + 64 coefficients and up to 6 transforms; plus the stream's marker bit.
    /// </summary>
    public const int MaxSymbolsPerSlot = 7 + 3 + 6 + 22 * 384 + 1;

    private readonly Action<Index1D, ArrayView<short>, ArrayView<short>, ArrayView<short>, ArrayView<byte>, ArrayView<ushort>,
        ArrayView<int>, ArrayView<int>, ArrayView<ushort>, Vp9KeySymbolParams> _kernel;

    /// <summary>Compile.</summary>
    public Vp9KeyframeSymbolsKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<short>, ArrayView<short>, ArrayView<short>,
            ArrayView<byte>, ArrayView<ushort>, ArrayView<int>, ArrayView<int>, ArrayView<ushort>, Vp9KeySymbolParams>(SymbolsKernel);
    }

    /// <summary>Slots of a frame: 16 per superblock.</summary>
    public static int SlotCount(int mbCols, int mbRows) => (mbCols >> 2) * (mbRows >> 2) * 16;

    /// <summary>Each slot's symbol count into <paramref name="counts"/>. Queued.</summary>
    public void Count(ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts, ArrayView<int> counts, ArrayView<ushort> window,
        int mbCols, int mbRows, int frameMiCols, int frameMiRows)
    {
        var p = new Vp9KeySymbolParams { MbCols = mbCols, MbRows = mbRows, FrameMiCols = frameMiCols, FrameMiRows = frameMiRows };
        _kernel(SlotCount(mbCols, mbRows), yCoefs, uCoefs, vCoefs, byteConsts, ushortConsts, counts, counts, window, p);
    }

    /// <summary>The symbols with a stream index in the coder's window of wave <paramref name="wave"/>. Queued.</summary>
    public void Write(ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts, ArrayView<int> counts, ArrayView<int> offsets,
        ArrayView<ushort> window, int windowSymbols, int wave,
        int mbCols, int mbRows, int frameMiCols, int frameMiRows)
    {
        var p = new Vp9KeySymbolParams
        {
            MbCols = mbCols, MbRows = mbRows, FrameMiCols = frameMiCols, FrameMiRows = frameMiRows,
            Write = 1, WindowBase = checked(wave * windowSymbols), WindowSymbols = windowSymbols,
        };
        _kernel(SlotCount(mbCols, mbRows), yCoefs, uCoefs, vCoefs, byteConsts, ushortConsts, counts, offsets, window, p);
    }

    private static void SymbolsKernel(Index1D slot,
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        ArrayView<int> counts, ArrayView<int> offsets, ArrayView<ushort> window, Vp9KeySymbolParams p)
    {
        var sink = new Vp8BoolSymbolSink();
        if (p.Write != 0)
        {
            int start = offsets[slot];
            int end = start + counts[slot];
            int hi = p.WindowBase + p.WindowSymbols;
            bool outside = end <= p.WindowBase | start >= hi;
            if (outside) return;
            sink.Next = start;
            sink.Lo = p.WindowBase;
            sink.Hi = hi;
        }
        var tokenCache = LocalMemory.Allocate<byte>(256);
        EmitSlot(ref sink, window, slot, yCoefs, uCoefs, vCoefs, byteConsts, ushortConsts, tokenCache, p);
        if (p.Write == 0) counts[slot] = sink.Next;
    }

    // ---- neighbours ----

    /// <summary>The leaf shape at macroblock (mbRow, mbCol): Vp9FrameSequentialEncodeKernel.BoundaryShape.</summary>
    private static int Shape(int mbRow, int mbCol, int frameMiRows, int frameMiCols)
    {
        int bottom = frameMiRows < mbRow * 2 + 2 ? 1 : 0;
        int right = frameMiCols < mbCol * 2 + 2 ? 2 : 0;
        return bottom | right;
    }

    /// <summary>Partition context state (left bit * 2 + above bit) at mode-info (miRow, miCol), level <paramref name="bsl"/>.</summary>
    private static int PartitionState(int miRow, int miCol, int bsl, int frameMiRows, int frameMiCols)
    {
        int above = 0, left = 0;
        if (miRow > 0)
        {
            int s = Shape((miRow - 1) >> 1, miCol >> 1, frameMiRows, frameMiCols);
            above = s == 2 | s == 3 ? 14 : 12;
        }
        if (miCol > 0)
        {
            int s = Shape(miRow >> 1, (miCol - 1) >> 1, frameMiRows, frameMiCols);
            left = s == 1 | s == 3 ? 14 : 12;
        }
        return ((left >> bsl) & 1) * 2 + ((above >> bsl) & 1);
    }

    /// <summary>1 when transform <paramref name="half"/> of <paramref name="plane"/> in leaf (mbRow, mbCol) of shape <paramref name="shape"/> has a nonzero coefficient.</summary>
    private static int Nonzero(ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        int plane, int mbRow, int mbCol, int mbCols, int shape, int half)
    {
        long mbIdx = (long)mbRow * mbCols + mbCol;
        int nz = 0;
        if (plane == 0)
        {
            long b = mbIdx * 256 + half * 64;
            int n = shape != 0 ? 64 : 256;
            for (int i = 0; i < n; i++) nz |= yCoefs[b + i];
        }
        else if (plane == 1)
        {
            long b = mbIdx * 64 + half * 16;
            int n = shape != 0 ? 16 : 64;
            for (int i = 0; i < n; i++) nz |= uCoefs[b + i];
        }
        else
        {
            long b = mbIdx * 64 + half * 16;
            int n = shape != 0 ? 16 : 64;
            for (int i = 0; i < n; i++) nz |= vCoefs[b + i];
        }
        return nz != 0 ? 1 : 0;
    }

    // ---- the slot (EncodeLeafSlot) ----

    private static void EmitSlot(ref Vp8BoolSymbolSink sink, ArrayView<ushort> window, int slot,
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts, ArrayView<byte> tokenCache, Vp9KeySymbolParams p)
    {
        int frameMiCols = p.FrameMiCols;
        int frameMiRows = p.FrameMiRows;
        int sbCols = p.MbCols >> 2;
        int sb = slot >> 4;
        int q = slot & 15;
        int sbRow = sb / sbCols;
        int sbCol = sb - sbRow * sbCols;
        int miRow64 = sbRow * 8;
        int miCol64 = sbCol * 8;

        // the VP9 marker bit, coded before the first slot
        if (slot == 0) Vp8BoolSymbolSink.Put(ref sink, window, 0, 128);

        if (q == 0)
        {
            bool hasRows64 = (miRow64 + 4) < frameMiRows;
            bool hasCols64 = (miCol64 + 4) < frameMiCols;
            EmitPartition(ref sink, window, byteConsts, 3, 3, miRow64, miCol64, hasRows64, hasCols64, frameMiRows, frameMiCols, 1);
        }

        int q32 = q >> 2;
        int q16 = q & 3;
        int miRow32 = miRow64 + ((q32 & 2) >> 1) * 4;
        int miCol32 = miCol64 + (q32 & 1) * 4;
        bool in32 = (miRow32 < frameMiRows) & (miCol32 < frameMiCols);
        if (!in32) return;

        if (q16 == 0)
        {
            bool hasRows32 = (miRow32 + 2) < frameMiRows;
            bool hasCols32 = (miCol32 + 2) < frameMiCols;
            EmitPartition(ref sink, window, byteConsts, 2, 2, miRow32, miCol32, hasRows32, hasCols32, frameMiRows, frameMiCols, 1);
        }

        int miRow16 = miRow32 + ((q16 & 2) >> 1) * 2;
        int miCol16 = miCol32 + (q16 & 1) * 2;
        bool in16 = (miRow16 < frameMiRows) & (miCol16 < frameMiCols);
        if (!in16) return;

        // EncodeBlock16x16: the 16x16's partition (NONE inside, HORZ / VERT at the bottom / right edge, the corner's
        // implied SPLIT and then its 8x8's NONE)
        bool hasRows = (miRow16 + 1) < frameMiRows;
        bool hasCols = (miCol16 + 1) < frameMiCols;
        int shape = hasRows & hasCols ? 0 : hasCols ? 1 : hasRows ? 2 : 3;
        int sizeIdx = shape == 3 ? 0 : 1;
        int bsl = shape == 3 ? 0 : 1;
        int st = PartitionState(miRow16, miCol16, bsl, frameMiRows, frameMiCols);
        long probsBase = Vp9KeyframeConstantsGpu.KfPartitionProbsOffset + ((long)sizeIdx * 4 + st) * 3;
        int probIdx = shape == 1 ? 1 : shape == 2 ? 2 : 0;
        Vp8BoolSymbolSink.Put(ref sink, window, 0, byteConsts[probsBase + probIdx]);

        // EncodeLeaf: skip 0 (context 0), Y mode DC_PRED (context row 0), UV mode DC_PRED
        Vp8BoolSymbolSink.Put(ref sink, window, 0, byteConsts[Vp9KeyframeConstantsGpu.SkipProbsOffset]);
        Vp8BoolSymbolSink.Put(ref sink, window, 0, byteConsts[Vp9KeyframeConstantsGpu.KfYModeProbsOffset]);
        Vp8BoolSymbolSink.Put(ref sink, window, 0, byteConsts[Vp9KeyframeConstantsGpu.KfUvModeProbsOffset]);

        int mbR = miRow16 >> 1, mbC = miCol16 >> 1, mbCols = p.MbCols;
        long mbIdx = (long)mbR * mbCols + mbC;
        int shapeAbove = mbR > 0 ? Shape(mbR - 1, mbC, frameMiRows, frameMiCols) : 0;
        int shapeLeft = mbC > 0 ? Shape(mbR, mbC - 1, frameMiRows, frameMiCols) : 0;
        int two = shape == 1 | shape == 2 ? 1 : 0;
        int jobs = two != 0 ? 6 : 3;
        int firstNz = 0;   // the plane's first transform's (eob > 0), for its second
        for (int j = 0; j < jobs; j++)
        {
            int plane = two != 0 ? (j >> 1) : j;
            int half = two != 0 ? (j & 1) : 0;
            // above: this leaf's first transform for the lower of two stacked ones, else the leaf above (its lower
            // transform when it has two stacked); left: this leaf's first transform for the right of two side by side,
            // else the leaf left (its right transform when it has two side by side)
            int aboveNz;
            if (shape == 2 & half == 1) aboveNz = firstNz;
            else if (mbR == 0) aboveNz = 0;
            else aboveNz = Nonzero(yCoefs, uCoefs, vCoefs, plane, mbR - 1, mbC, mbCols, shapeAbove, shapeAbove == 2 ? 1 : 0);
            int leftNz;
            if (shape == 1 & half == 1) leftNz = firstNz;
            else if (mbC == 0) leftNz = 0;
            else leftNz = Nonzero(yCoefs, uCoefs, vCoefs, plane, mbR, mbC - 1, mbCols, shapeLeft, shapeLeft == 1 ? 1 : 0);
            int initialCtx = aboveNz + leftNz;

            long coefBase;
            int coefCount, tx, planeType, isTx4x4;
            if (plane == 0)
            {
                coefBase = mbIdx * 256 + half * 64;
                coefCount = shape != 0 ? 64 : 256;
                isTx4x4 = 0;
                tx = shape != 0 ? (int)Vp9TxSize.Tx8x8 : (int)Vp9TxSize.Tx16x16;
                planeType = (int)Vp9BlockCoefEnums.PlaneType.Y;
            }
            else
            {
                coefBase = mbIdx * 64 + half * 16;
                coefCount = shape != 0 ? 16 : 64;
                isTx4x4 = shape != 0 ? 1 : 0;
                tx = shape != 0 ? (int)Vp9TxSize.Tx4x4 : (int)Vp9TxSize.Tx8x8;
                planeType = (int)Vp9BlockCoefEnums.PlaneType.Uv;
            }
            long scanBase, neighborsBase, coefProbsBase;
            if (tx == (int)Vp9TxSize.Tx4x4)
            {
                scanBase = Vp9KeyframeConstantsGpu.Scan4x4Offset;
                neighborsBase = Vp9KeyframeConstantsGpu.Neighbors4x4Offset;
                coefProbsBase = Vp9KeyframeConstantsGpu.CoefProbs4x4Offset;
            }
            else if (tx == (int)Vp9TxSize.Tx8x8)
            {
                scanBase = Vp9KeyframeConstantsGpu.Scan8x8Offset;
                neighborsBase = Vp9KeyframeConstantsGpu.Neighbors8x8Offset;
                coefProbsBase = Vp9KeyframeConstantsGpu.CoefProbs8x8Offset;
            }
            else
            {
                scanBase = Vp9KeyframeConstantsGpu.Scan16x16Offset;
                neighborsBase = Vp9KeyframeConstantsGpu.Neighbors16x16Offset;
                coefProbsBase = Vp9KeyframeConstantsGpu.CoefProbs16x16Offset;
            }
            var coefs = plane == 0 ? yCoefs.SubView(coefBase, coefCount) : plane == 1 ? uCoefs.SubView(coefBase, coefCount) : vCoefs.SubView(coefBase, coefCount);
            int eob = EmitBlock(ref sink, window, coefs,
                ushortConsts.SubView(scanBase, coefCount), ushortConsts.SubView(neighborsBase, (long)coefCount * 2),
                byteConsts.SubView(coefProbsBase, 432),
                byteConsts.SubView(Vp9KeyframeConstantsGpu.CoefConstsOffset, Vp9KeyframeConstantsGpu.CoefConstsLength),
                tokenCache, coefCount, planeType, initialCtx, isTx4x4);
            if (half == 0) firstNz = eob > 0 ? 1 : 0;
        }
    }

    /// <summary>
    /// A partition symbol at a 64x64 / 32x32 level (<paramref name="split"/> = 1: SPLIT inside the frame = three 1 bits;
    /// at the bottom edge one 1 bit at probs[1], at the right edge one at probs[2], none at the corner).
    /// </summary>
    private static void EmitPartition(ref Vp8BoolSymbolSink sink, ArrayView<ushort> window, ArrayView<byte> byteConsts,
        int sizeIdx, int bsl, int miRow, int miCol, bool hasRows, bool hasCols, int frameMiRows, int frameMiCols, int split)
    {
        int st = PartitionState(miRow, miCol, bsl, frameMiRows, frameMiCols);
        long probsBase = Vp9KeyframeConstantsGpu.KfPartitionProbsOffset + ((long)sizeIdx * 4 + st) * 3;
        if (hasRows & hasCols)
        {
            Vp8BoolSymbolSink.Put(ref sink, window, 1, byteConsts[probsBase + 0]);
            Vp8BoolSymbolSink.Put(ref sink, window, 1, byteConsts[probsBase + 1]);
            Vp8BoolSymbolSink.Put(ref sink, window, 1, byteConsts[probsBase + 2]);
        }
        else if (hasCols) Vp8BoolSymbolSink.Put(ref sink, window, 1, byteConsts[probsBase + 1]);
        else if (hasRows) Vp8BoolSymbolSink.Put(ref sink, window, 1, byteConsts[probsBase + 2]);
    }

    // ---- one transform's tokens (Vp9BlockCoefEncoderGpu.EncodeBlock, intra, 8 bits) ----

    private static int EmitBlock(ref Vp8BoolSymbolSink sink, ArrayView<ushort> window, ArrayView<short> coefs,
        ArrayView<ushort> scan, ArrayView<ushort> neighbors, ArrayView<byte> coefProbs, ArrayView<byte> consts,
        ArrayView<byte> tokenCache, int maxCoefs, int planeType, int initialCtx, int isTx4x4)
    {
        for (int i = 0; i < maxCoefs; i++) tokenCache[i] = 0;
        int eob = 0;
        for (int i = maxCoefs - 1; i >= 0; i--)
        {
            int raster = scan[i];
            if (coefs[raster] != 0) { eob = i + 1; break; }
        }

        int c = 0;
        int ctx = initialCtx;
        bool more = true;
        while (more)
        {
            int band = isTx4x4 != 0 ? consts[Vp9BlockCoefEncoderGpu.Band4x4Offset + c] : consts[Vp9BlockCoefEncoderGpu.Band8x8PlusOffset + c];
            int modelBase = Vp9CoefProbs.Index4x4(planeType, (int)Vp9BlockCoefEnums.RefType.Intra, band, ctx, 0);
            if (c == eob)
            {
                Vp8BoolSymbolSink.Put(ref sink, window, 0, coefProbs[modelBase]);   // EOB
                return eob;
            }
            Vp8BoolSymbolSink.Put(ref sink, window, 1, coefProbs[modelBase]);       // more coefficients
            int m1 = coefProbs[modelBase + 1];
            int m2 = coefProbs[modelBase + 2];
            bool zero = coefs[(int)scan[c]] == 0;
            while (zero)
            {
                Vp8BoolSymbolSink.Put(ref sink, window, 0, m1);                     // ZERO
                c++;
                // c < eob <= maxCoefs here: a zero before the last nonzero coefficient
                band = isTx4x4 != 0 ? consts[Vp9BlockCoefEncoderGpu.Band4x4Offset + c] : consts[Vp9BlockCoefEncoderGpu.Band8x8PlusOffset + c];
                ctx = Vp9BlockCoefEncoderGpu.GetCoefContext(neighbors, tokenCache, c);
                modelBase = Vp9CoefProbs.Index4x4(planeType, (int)Vp9BlockCoefEnums.RefType.Intra, band, ctx, 0);
                m1 = coefProbs[modelBase + 1];
                m2 = coefProbs[modelBase + 2];
                zero = coefs[(int)scan[c]] == 0;
            }
            Vp8BoolSymbolSink.Put(ref sink, window, 1, m1);                         // not ZERO
            int value = coefs[(int)scan[c]];
            int magnitude = value < 0 ? -value : value;
            byte token;
            if (magnitude == 1)
            {
                Vp8BoolSymbolSink.Put(ref sink, window, 0, m2);                     // ONE
                token = (byte)Vp9CoefToken.One;
            }
            else
            {
                Vp8BoolSymbolSink.Put(ref sink, window, 1, m2);
                token = Vp9BlockCoefEncoderGpu.MagnitudeToToken(magnitude, 0);
                PutTreeWalk(ref sink, window, consts, Vp9BlockCoefEncoderGpu.Pareto8FullOffset + (m2 - 1) * 8, token);
                if (token >= (byte)Vp9CoefToken.Category1) PutCategory(ref sink, window, consts, token, magnitude);
            }
            Vp8BoolSymbolSink.Put(ref sink, window, value < 0 ? 1 : 0, 128);        // sign
            tokenCache[(int)scan[c]] = consts[Vp9BlockCoefEncoderGpu.PtEnergyClassOffset + token];
            c++;
            more = c < maxCoefs;
            if (more) ctx = Vp9BlockCoefEncoderGpu.GetCoefContext(neighbors, tokenCache, c);
        }
        return eob;
    }

    // The constrained tree (Vp9BlockCoefEncoderGpu.EmitTreeWalk): the path to each token as (bit, pareto8 node) pairs.
    internal static void PutTreeWalk(ref Vp8BoolSymbolSink sink, ArrayView<ushort> window, ArrayView<byte> consts, int row, byte token)
    {
        int t = token;
        if (t <= (int)Vp9CoefToken.Four)
        {
            Vp8BoolSymbolSink.Put(ref sink, window, 0, consts[row + 0]);
            Vp8BoolSymbolSink.Put(ref sink, window, t == (int)Vp9CoefToken.Two ? 0 : 1, consts[row + 1]);
            if (t != (int)Vp9CoefToken.Two) Vp8BoolSymbolSink.Put(ref sink, window, t == (int)Vp9CoefToken.Three ? 0 : 1, consts[row + 2]);
        }
        else
        {
            Vp8BoolSymbolSink.Put(ref sink, window, 1, consts[row + 0]);
            if (t <= (int)Vp9CoefToken.Category2)
            {
                Vp8BoolSymbolSink.Put(ref sink, window, 0, consts[row + 3]);
                Vp8BoolSymbolSink.Put(ref sink, window, t == (int)Vp9CoefToken.Category1 ? 0 : 1, consts[row + 4]);
            }
            else
            {
                Vp8BoolSymbolSink.Put(ref sink, window, 1, consts[row + 3]);
                if (t <= (int)Vp9CoefToken.Category4)
                {
                    Vp8BoolSymbolSink.Put(ref sink, window, 0, consts[row + 5]);
                    Vp8BoolSymbolSink.Put(ref sink, window, t == (int)Vp9CoefToken.Category3 ? 0 : 1, consts[row + 6]);
                }
                else
                {
                    Vp8BoolSymbolSink.Put(ref sink, window, 1, consts[row + 5]);
                    Vp8BoolSymbolSink.Put(ref sink, window, t == (int)Vp9CoefToken.Category5 ? 0 : 1, consts[row + 7]);
                }
            }
        }
    }

    // The extra magnitude bits of a category token, most significant first (Vp9BlockCoefEncoderGpu.EmitCategoryMagnitude, 8 bits).
    internal static void PutCategory(ref Vp8BoolSymbolSink sink, ArrayView<ushort> window, ArrayView<byte> consts, byte token, int magnitude)
    {
        int probOffset, probLen, minVal;
        if (token == (byte)Vp9CoefToken.Category1) { probOffset = Vp9BlockCoefEncoderGpu.Cat1ProbOffset; probLen = 1; minVal = Vp9CoefProbs.CatMinVal.Cat1; }
        else if (token == (byte)Vp9CoefToken.Category2) { probOffset = Vp9BlockCoefEncoderGpu.Cat2ProbOffset; probLen = 2; minVal = Vp9CoefProbs.CatMinVal.Cat2; }
        else if (token == (byte)Vp9CoefToken.Category3) { probOffset = Vp9BlockCoefEncoderGpu.Cat3ProbOffset; probLen = 3; minVal = Vp9CoefProbs.CatMinVal.Cat3; }
        else if (token == (byte)Vp9CoefToken.Category4) { probOffset = Vp9BlockCoefEncoderGpu.Cat4ProbOffset; probLen = 4; minVal = Vp9CoefProbs.CatMinVal.Cat4; }
        else if (token == (byte)Vp9CoefToken.Category5) { probOffset = Vp9BlockCoefEncoderGpu.Cat5ProbOffset; probLen = 5; minVal = Vp9CoefProbs.CatMinVal.Cat5; }
        else { probOffset = Vp9BlockCoefEncoderGpu.Cat6ProbOffset; probLen = 14; minVal = Vp9CoefProbs.CatMinVal.Cat6; }
        int extra = magnitude - minVal;
        for (int i = 0; i < probLen; i++)
            Vp8BoolSymbolSink.Put(ref sink, window, (extra >> (probLen - 1 - i)) & 1, consts[probOffset + i]);
    }
}
