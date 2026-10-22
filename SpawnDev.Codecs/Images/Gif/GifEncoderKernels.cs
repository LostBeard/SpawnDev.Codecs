// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The GIF encode kernels. Every stage that touches pixels or the bitstream runs on the accelerator; the host
// allocates, dispatches in order, and reads back one length (and, for Floyd-Steinberg / the exact-or-quantize
// decision, a few ints). Nothing here depends on thread scheduling: every result is deterministic.
//
//   Palette
//     1. ExactPaletteBuilder (shared with BMP)   hash set of the opaque colours of ALL frames: when there are at most
//                                                maxColors (minus one reserved transparent index) the palette is
//                                                EXACT (sorted ascending by 0xRRGGBB) and shared by every frame.
//     2. else, per frame: exact again, or QUANTIZED:
//        HistKernel (1 thread per pixel)         32768-bin histogram of the 5-5-5 bit colour with the 3 low bits of
//                                                every channel summed per bin (so the palette keeps full precision).
//        MedianCutKernel (1 thread)              median cut over the occupied bins: repeatedly split the box with the
//                                                highest (population >> s) * longest extent along its longest axis
//                                                (ties: R, then G, then B) at the population median; the palette entry
//                                                of a box is its pixel-weighted mean colour; entries sorted ascending.
//   Pixels
//     3. MapKernel (1 thread per pixel)          nearest palette entry by squared RGB distance (lowest index wins a
//                                                tie), binary search for exact palettes, optional ordered dither
//                                                (8x8 Bayer, +-16 levels). A pixel with alpha < threshold gets the
//                                                transparent index.
//        FsWaveKernel (1 thread per row)         Floyd-Steinberg, run in anti-diagonal waves (t = x + 2y) so that the
//                                                result equals the sequential algorithm: the error of a pixel is
//                                                pulled from its left, upper-left, upper and upper-right neighbours
//                                                with weights 7/3/5/1 (sum + 8) >> 4.
//   Bitstream
//     4. LzwSegmentKernel (1 thread per segment) LZW over SegmentPixels pixels: a clear code ends every segment (the
//                                                last one ends with the end code), so segments are independent and the
//                                                bit streams just concatenate. The dictionary is an open-addressing hash
//                                                table in per-thread scratch.
//     5. SegOffsetsKernel / FrameOffsetsKernel   prefix sums of the segment bit lengths and of the frame lengths.
//     6. GatherKernel                            OR every segment's bits into its frame payload at its bit offset.
//     7. WritePrefixKernel / CopyFramedKernel    graphic control extension, image descriptor, local table, minimum code
//                                                size, then the payload cut into 255 byte sub-blocks, straight into the
//                                                caller's output view. HeaderKernel / TrailerKernel write the rest.
//
// Loop conditions and branch conditions with several terms are bool locals built with non-short-circuit & / |
// (a compound `while (a && b)` hung the WebGPU device).

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Images.Gif;

/// <summary>Slots of the encoder's global state block.</summary>
internal static class GifEncGlobal
{
    public const int Size = 16;
    /// <summary>1 when every opaque colour of the whole animation fits the palette (one exact palette for all frames).</summary>
    public const int Exact = 0;
    /// <summary>Entries of the global exact palette.</summary>
    public const int Count = 1;
    /// <summary>1 when any pixel of any frame is transparent: then every frame carries the same transparent index.</summary>
    public const int AnimHasTrans = 2;
    /// <summary>1 when all frames use the palette of frame 0, written once as the global colour table.</summary>
    public const int SinglePalette = 3;
    /// <summary>Bytes of the file header (screen descriptor, global table, loop extension).</summary>
    public const int HeaderLen = 4;
    /// <summary>Output offset where the next frame goes.</summary>
    public const int Running = 5;
    /// <summary>Total file length once the trailer is written.</summary>
    public const int TotalLen = 6;
    /// <summary>1 when frames get a graphic control extension.</summary>
    public const int GceMode = 7;
    /// <summary>1 when the NETSCAPE loop extension is written.</summary>
    public const int LoopExt = 8;
}

/// <summary>Slots of one frame's entry in the frame info block.</summary>
internal static class GifEncFrame
{
    public const int Stride = 12;
    public const int PalCount = 0;
    public const int Trans = 1;
    public const int Exact = 2;
    public const int TableSize = 3;
    public const int Mcs = 4;
    public const int PayloadBytes = 5;
    public const int OutOffset = 6;
    public const int PrefixLen = 7;
    public const int FramedLen = 8;
    public const int HasGce = 9;
    public const int Local = 10;
    public const int Disposal = 11;
}

/// <summary>Slots of the encoder's parameter block (uploaded once per call).</summary>
internal static class GifEncParam
{
    public const int Size = 16;
    public const int Frames = 0;
    public const int Width = 1;
    public const int Height = 2;
    public const int MaxColors = 3;
    public const int AlphaThreshold = 4;
    public const int Dither = 5;
    public const int LoopCount = 6;
    public const int SegPixels = 7;
    public const int SegCount = 8;
}

/// <summary>Loads and launches the GIF encode kernels.</summary>
internal sealed class GifEncoderKernels : IDisposable
{
    /// <summary>Pixels per LZW segment (independent bit streams).</summary>
    public const int SegmentPixels = 1 << 15;
    /// <summary>Slots of the per-thread LZW hash table (the dictionary holds at most 4096 entries: load factor 0.5).</summary>
    public const int HashSlots = 8192;

    // quantizer scratch layout (ints)
    public const int HistOff = 0;
    public const int HistInts = 32768 * 4;
    public const int BinsOff = HistOff + HistInts;
    public const int TmpOff = BinsOff + 32768;
    public const int BoxOff = TmpOff + 32768;
    public const int BoxStride = 10;
    public const int H1Off = BoxOff + 256 * BoxStride;
    public const int QuantScratchInts = H1Off + 32 + 64;

    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _prepGlobal;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _framePalette;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _hist;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _medianCut;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _layout;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _map;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int> _fsWave;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int, int> _lzw;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _segOffsets;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int, int> _frameOffsets;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int> _gather;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _writePrefix;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int, int, int> _copyFramed;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _header;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>> _trailer;

    public GifEncoderKernels(Accelerator a)
    {
        ArgumentNullException.ThrowIfNull(a);
        _prepGlobal = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(PrepGlobalKernel);
        _framePalette = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(FramePaletteKernel);
        _hist = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(HistKernel);
        _medianCut = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(MedianCutKernel);
        _layout = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(LayoutKernel);
        _map = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(MapKernel);
        _fsWave = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(FsWaveKernel);
        _lzw = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int, int>(LzwSegmentKernel);
        _segOffsets = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(SegOffsetsKernel);
        _frameOffsets = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, int, int>(FrameOffsetsKernel);
        _gather = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(GatherKernel);
        _writePrefix = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(WritePrefixKernel);
        _copyFramed = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int, int, int>(CopyFramedKernel);
        _header = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(HeaderKernel);
        _trailer = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>>(TrailerKernel);
    }

    // ------------------------------------------------------------------
    // launchers
    // ------------------------------------------------------------------

    /// <summary>Publish the global exact-palette verdict and, when exact, give every frame the shared palette.</summary>
    public void PrepGlobal(ArrayView<int> builderState, ArrayView<int> builderPalette, ArrayView<int> gstate,
        ArrayView<int> fpal, ArrayView<int> finfo, ArrayView<int> prm, int frames)
        => _prepGlobal(frames, builderState, builderPalette, gstate, fpal, finfo, prm);

    /// <summary>Copy frame <paramref name="frame"/>'s exact palette (if it is exact and the animation has no shared palette) into its slot.</summary>
    public void FramePalette(ArrayView<int> builderState, ArrayView<int> builderPalette, ArrayView<int> gstate,
        ArrayView<int> fpal, ArrayView<int> finfo, int frame)
        => _framePalette(1, builderState, builderPalette, gstate, fpal, finfo, frame);

    public void Hist(ArrayView<byte> rgbaFrame, ArrayView<int> scratch, ArrayView<int> finfo, ArrayView<int> gstate,
        ArrayView<int> prm, int frame, int pixels)
        => _hist(pixels, rgbaFrame, scratch, finfo, gstate, prm, frame);

    public void MedianCut(ArrayView<int> scratch, ArrayView<int> finfo, ArrayView<int> gstate, ArrayView<int> fpal,
        ArrayView<int> prm, int frame)
        => _medianCut(1, scratch, finfo, gstate, fpal, prm, frame);

    /// <summary>Per-frame table size, minimum code size, prefix length, header length, running offset.</summary>
    public void Layout(ArrayView<int> gstate, ArrayView<int> finfo, ArrayView<int> prm, ArrayView<int> disposals)
        => _layout(1, gstate, finfo, prm, disposals);

    public void Map(ArrayView<byte> rgbaBatch, ArrayView<byte> idxBatch, ArrayView<int> fpal, ArrayView<int> finfo,
        ArrayView<int> prm, int batchFirst, int threads)
        => _map(threads, rgbaBatch, idxBatch, fpal, finfo, prm, batchFirst);

    public void FsWave(ArrayView<byte> rgbaFrame, ArrayView<byte> idxFrame, ArrayView<int> err, ArrayView<int> fpal,
        ArrayView<int> finfo, ArrayView<int> prm, int wave, int frame, int rows)
        => _fsWave(rows, rgbaFrame, idxFrame, err, fpal, finfo, prm, wave, frame);

    public void LzwSegments(ArrayView<byte> idxBatch, ArrayView<byte> seg, ArrayView<int> segInfo, ArrayView<int> hash,
        ArrayView<int> finfo, ArrayView<int> prm, int batchFirst, int segStart, int segBound, int threads)
        => _lzw(threads, idxBatch, seg, segInfo, hash, finfo, prm, batchFirst, segStart, segBound);

    public void SegOffsets(ArrayView<int> segInfo, ArrayView<int> segOff, ArrayView<int> finfo, ArrayView<int> prm,
        int batchFirst, int framesInBatch)
        => _segOffsets(framesInBatch, segInfo, segOff, finfo, prm, batchFirst);

    public void FrameOffsets(ArrayView<int> gstate, ArrayView<int> finfo, int batchFirst, int framesInBatch)
        => _frameOffsets(1, gstate, finfo, batchFirst, framesInBatch);

    public void Gather(ArrayView<byte> seg, ArrayView<int> segInfo, ArrayView<int> segOff, ArrayView<int> payload,
        ArrayView<int> prm, int segBound, int payloadWords, int threads)
        => _gather(threads, seg, segInfo, segOff, payload, prm, segBound, payloadWords);

    public void WritePrefix(ArrayView<byte> output, ArrayView<int> gstate, ArrayView<int> finfo, ArrayView<int> fpal,
        ArrayView<int> prm, ArrayView<int> delays, int batchFirst, int framesInBatch)
        => _writePrefix(framesInBatch, output, gstate, finfo, fpal, prm, delays, batchFirst);

    public void CopyFramed(ArrayView<byte> output, ArrayView<int> finfo, ArrayView<int> payload, int batchFirst,
        int payloadWords, int framedBound, int threads)
        => _copyFramed(threads, output, finfo, payload, batchFirst, payloadWords, framedBound);

    public void Header(ArrayView<byte> output, ArrayView<int> gstate, ArrayView<int> finfo, ArrayView<int> fpal,
        ArrayView<int> prm)
        => _header(1, output, gstate, finfo, fpal, prm);

    public void Trailer(ArrayView<byte> output, ArrayView<int> gstate)
        => _trailer(1, output, gstate);

    // ------------------------------------------------------------------
    // small helpers
    // ------------------------------------------------------------------

    private static int Clamp255(int v)
    {
        int r = v;
        if (r < 0) r = 0;
        if (r > 255) r = 255;
        return r;
    }

    /// <summary>Value (0..63) of the 8x8 Bayer matrix at (x, y).</summary>
    private static int Bayer8(int x, int y)
    {
        int xv = x & 7;
        int yv = y & 7;
        int xy = xv ^ yv;
        return ((xy & 1) << 5) | ((yv & 1) << 4) | ((xy & 2) << 2) | ((yv & 2) << 1) | ((xy & 4) >> 1) | ((yv & 4) >> 2);
    }

    private static int NearestIndex(ArrayView<int> fpal, int baseOff, int count, int r, int g, int b)
    {
        int best = 0;
        int bestD = int.MaxValue;
        for (int i = 0; i < count; i++)
        {
            int c = fpal[baseOff + i];
            int dr = r - ((c >> 16) & 255);
            int dg = g - ((c >> 8) & 255);
            int db = b - (c & 255);
            int d = dr * dr + dg * dg + db * db;
            if (d < bestD)
            {
                bestD = d;
                best = i;
            }
        }
        return best;
    }

    /// <summary>Index of <paramref name="key"/> in the ascending palette (0 when absent: callers only ask for colours they inserted).</summary>
    private static int ExactIndex(ArrayView<int> fpal, int baseOff, int count, int key)
    {
        int lo = 0;
        int hi = count - 1;
        int found = 0;
        bool go = lo <= hi;
        while (go)
        {
            int mid = (lo + hi) >> 1;
            int v = fpal[baseOff + mid];
            if (v == key)
            {
                found = mid;
                go = false;
            }
            else
            {
                if (v < key) lo = mid + 1;
                else hi = mid - 1;
                go = lo <= hi;
            }
        }
        return found;
    }

    // ------------------------------------------------------------------
    // palette
    // ------------------------------------------------------------------

    private static void PrepGlobalKernel(Index1D index, ArrayView<int> bstate, ArrayView<int> bpal, ArrayView<int> gstate,
        ArrayView<int> fpal, ArrayView<int> finfo, ArrayView<int> prm)
    {
        int f = index.X;
        int frames = prm[GifEncParam.Frames];
        int exact = bstate[ExactPaletteState.Exact];
        int count = bstate[ExactPaletteState.PaletteCount];
        int anim = bstate[ExactPaletteState.HasTransparent] != 0 ? 1 : 0;
        if (f == 0)
        {
            gstate[GifEncGlobal.Exact] = exact;
            gstate[GifEncGlobal.Count] = count;
            gstate[GifEncGlobal.AnimHasTrans] = anim;
            gstate[GifEncGlobal.SinglePalette] = (exact != 0) | (frames == 1) ? 1 : 0;
        }
        int fi = f * GifEncFrame.Stride;
        for (int k = 0; k < GifEncFrame.Stride; k++) finfo[fi + k] = 0;
        finfo[fi + GifEncFrame.Trans] = -1;
        if (exact != 0)
        {
            for (int i = 0; i < count; i++) fpal[f * 256 + i] = bpal[i];
            finfo[fi + GifEncFrame.PalCount] = count;
            finfo[fi + GifEncFrame.Exact] = 1;
            if (anim != 0) finfo[fi + GifEncFrame.Trans] = count;
        }
    }

    private static void FramePaletteKernel(Index1D _, ArrayView<int> bstate, ArrayView<int> bpal, ArrayView<int> gstate,
        ArrayView<int> fpal, ArrayView<int> finfo, int frame)
    {
        int fi = frame * GifEncFrame.Stride;
        bool take = (gstate[GifEncGlobal.Exact] == 0) & (bstate[ExactPaletteState.Exact] != 0);
        if (take)
        {
            int count = bstate[ExactPaletteState.PaletteCount];
            for (int i = 0; i < count; i++) fpal[frame * 256 + i] = bpal[i];
            finfo[fi + GifEncFrame.PalCount] = count;
            finfo[fi + GifEncFrame.Exact] = 1;
            finfo[fi + GifEncFrame.Trans] = gstate[GifEncGlobal.AnimHasTrans] != 0 ? count : -1;
        }
    }

    private static void HistKernel(Index1D index, ArrayView<byte> rgbaFrame, ArrayView<int> sc, ArrayView<int> finfo,
        ArrayView<int> gstate, ArrayView<int> prm, int frame)
    {
        bool need = (gstate[GifEncGlobal.Exact] == 0) & (finfo[frame * GifEncFrame.Stride + GifEncFrame.Exact] == 0);
        int o = index.X * 4;
        bool opaque = rgbaFrame[o + 3] >= prm[GifEncParam.AlphaThreshold];
        if (need & opaque)
        {
            int r = rgbaFrame[o];
            int g = rgbaFrame[o + 1];
            int b = rgbaFrame[o + 2];
            int bin = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3);
            int h = HistOff + bin * 4;
            Atomic.Add(ref sc[h], 1);
            Atomic.Add(ref sc[h + 1], r & 7);
            Atomic.Add(ref sc[h + 2], g & 7);
            Atomic.Add(ref sc[h + 3], b & 7);
        }
    }

    /// <summary>Population, min/max of R, G, B (5-bit values) of the bins [lo, hi) into box slot <paramref name="box"/>.</summary>
    private static void BoxStats(ArrayView<int> sc, int box, int lo, int hi)
    {
        int bo = BoxOff + box * BoxStride;
        int c = 0;
        int minR = 99, maxR = -1, minG = 99, maxG = -1, minB = 99, maxB = -1;
        for (int i = lo; i < hi; i++)
        {
            int bin = sc[BinsOff + i];
            c += sc[HistOff + bin * 4];
            int r = bin >> 10;
            int g = (bin >> 5) & 31;
            int b = bin & 31;
            if (r < minR) minR = r;
            if (r > maxR) maxR = r;
            if (g < minG) minG = g;
            if (g > maxG) maxG = g;
            if (b < minB) minB = b;
            if (b > maxB) maxB = b;
        }
        sc[bo] = lo;
        sc[bo + 1] = hi;
        sc[bo + 2] = c;
        sc[bo + 3] = minR;
        sc[bo + 4] = maxR;
        sc[bo + 5] = minG;
        sc[bo + 6] = maxG;
        sc[bo + 7] = minB;
        sc[bo + 8] = maxB;
    }

    private static int AxisValue(int bin, int axis)
    {
        int v = bin & 31;
        if (axis == 0) v = bin >> 10;
        else if (axis == 1) v = (bin >> 5) & 31;
        return v;
    }

    private static void MedianCutKernel(Index1D _, ArrayView<int> sc, ArrayView<int> finfo, ArrayView<int> gstate,
        ArrayView<int> fpal, ArrayView<int> prm, int frame)
    {
        int fi = frame * GifEncFrame.Stride;
        bool need = (gstate[GifEncGlobal.Exact] == 0) & (finfo[fi + GifEncFrame.Exact] == 0);
        if (need)
        {
            int anim = gstate[GifEncGlobal.AnimHasTrans];
            int target = prm[GifEncParam.MaxColors] - anim;

            // occupied bins, ascending
            int n = 0;
            int total = 0;
            for (int b = 0; b < 32768; b++)
            {
                int c = sc[HistOff + b * 4];
                if (c != 0)
                {
                    sc[BinsOff + n] = b;
                    n++;
                    total += c;
                }
            }
            int sh = 0;
            bool shifting = (total >> sh) >= (1 << 26);
            while (shifting)
            {
                sh++;
                shifting = (total >> sh) >= (1 << 26);
            }

            int boxes = 0;
            if (n > 0)
            {
                BoxStats(sc, 0, 0, n);
                boxes = 1;
            }
            bool go = (boxes > 0) & (boxes < target);
            while (go)
            {
                int best = -1;
                int bestScore = -1;
                for (int i = 0; i < boxes; i++)
                {
                    int bo = BoxOff + i * BoxStride;
                    if (sc[bo + 1] - sc[bo] >= 2)
                    {
                        int er = sc[bo + 4] - sc[bo + 3];
                        int eg = sc[bo + 6] - sc[bo + 5];
                        int eb = sc[bo + 8] - sc[bo + 7];
                        int ext = er;
                        if (eg > ext) ext = eg;
                        if (eb > ext) ext = eb;
                        int score = (sc[bo + 2] >> sh) * ext;
                        if (score > bestScore)
                        {
                            bestScore = score;
                            best = i;
                        }
                    }
                }
                if (best < 0)
                {
                    go = false;
                }
                else
                {
                    int bo = BoxOff + best * BoxStride;
                    int lo = sc[bo];
                    int hi = sc[bo + 1];
                    int cnt = sc[bo + 2];
                    int er = sc[bo + 4] - sc[bo + 3];
                    int eg = sc[bo + 6] - sc[bo + 5];
                    int eb = sc[bo + 8] - sc[bo + 7];
                    int axis = 0;
                    int axisExt = er;
                    if (eg > axisExt)
                    {
                        axis = 1;
                        axisExt = eg;
                    }
                    if (eb > axisExt)
                    {
                        axis = 2;
                        axisExt = eb;
                    }
                    int axisMin = sc[bo + 3 + axis * 2];
                    int axisMax = sc[bo + 4 + axis * 2];

                    // population per axis value
                    for (int v = 0; v < 32; v++) sc[H1Off + v] = 0;
                    for (int i = lo; i < hi; i++)
                    {
                        int bin = sc[BinsOff + i];
                        sc[H1Off + AxisValue(bin, axis)] += sc[HistOff + bin * 4];
                    }
                    int cum = 0;
                    int split = axisMin;
                    bool seeking = true;
                    for (int v = axisMin; v <= axisMax; v++)
                    {
                        if (seeking)
                        {
                            cum += sc[H1Off + v];
                            if (cum * 2 >= cnt)
                            {
                                split = v;
                                seeking = false;
                            }
                        }
                    }
                    if (split >= axisMax) split = axisMax - 1;

                    // stable partition of the bins by axis value <= split
                    int w = lo;
                    for (int i = lo; i < hi; i++)
                    {
                        int bin = sc[BinsOff + i];
                        if (AxisValue(bin, axis) <= split)
                        {
                            sc[TmpOff + w] = bin;
                            w++;
                        }
                    }
                    int mid = w;
                    for (int i = lo; i < hi; i++)
                    {
                        int bin = sc[BinsOff + i];
                        if (AxisValue(bin, axis) > split)
                        {
                            sc[TmpOff + w] = bin;
                            w++;
                        }
                    }
                    for (int i = lo; i < hi; i++) sc[BinsOff + i] = sc[TmpOff + i];
                    BoxStats(sc, best, lo, mid);
                    BoxStats(sc, boxes, mid, hi);
                    boxes++;
                    go = boxes < target;
                }
            }

            // palette = pixel-weighted mean colour of every box, then sorted ascending
            int pb = frame * 256;
            for (int i = 0; i < boxes; i++)
            {
                int bo = BoxOff + i * BoxStride;
                int lo = sc[bo];
                int hi = sc[bo + 1];
                int cnt = sc[bo + 2];
                int s = 0;
                bool scaling = (cnt >> s) >= (1 << 23);
                while (scaling)
                {
                    s++;
                    scaling = (cnt >> s) >= (1 << 23);
                }
                int wsum = 0;
                int accR = 0, accG = 0, accB = 0;
                for (int k = lo; k < hi; k++)
                {
                    int bin = sc[BinsOff + k];
                    int h = HistOff + bin * 4;
                    int bc = sc[h];
                    int wt = bc >> s;
                    int half = bc / 2;
                    int cr = ((bin >> 10) << 3) + (sc[h + 1] + half) / bc;
                    int cg = (((bin >> 5) & 31) << 3) + (sc[h + 2] + half) / bc;
                    int cb = ((bin & 31) << 3) + (sc[h + 3] + half) / bc;
                    accR += wt * cr;
                    accG += wt * cg;
                    accB += wt * cb;
                    wsum += wt;
                }
                int mr = (accR + wsum / 2) / wsum;
                int mg = (accG + wsum / 2) / wsum;
                int mb = (accB + wsum / 2) / wsum;
                fpal[pb + i] = (mr << 16) | (mg << 8) | mb;
            }
            for (int i = 1; i < boxes; i++)
            {
                int v = fpal[pb + i];
                int j = i - 1;
                bool moving = true;
                while (moving)
                {
                    if (j >= 0)
                    {
                        if (fpal[pb + j] > v)
                        {
                            fpal[pb + j + 1] = fpal[pb + j];
                            j--;
                        }
                        else
                        {
                            moving = false;
                        }
                    }
                    else
                    {
                        moving = false;
                    }
                }
                fpal[pb + j + 1] = v;
            }
            finfo[fi + GifEncFrame.PalCount] = boxes;
            finfo[fi + GifEncFrame.Exact] = 0;
            finfo[fi + GifEncFrame.Trans] = anim != 0 ? boxes : -1;
        }
    }

    // ------------------------------------------------------------------
    // layout
    // ------------------------------------------------------------------

    private static int TableSizeFor(int entries)
    {
        int size = 2;
        while (size < entries) size <<= 1;
        return size;
    }

    private static int Log2(int powerOfTwo)
    {
        int n = 0;
        int v = powerOfTwo;
        while (v > 1)
        {
            v >>= 1;
            n++;
        }
        return n;
    }

    private static void LayoutKernel(Index1D _, ArrayView<int> gstate, ArrayView<int> finfo, ArrayView<int> prm, ArrayView<int> disposals)
    {
        int frames = prm[GifEncParam.Frames];
        int anim = gstate[GifEncGlobal.AnimHasTrans];
        int single = gstate[GifEncGlobal.SinglePalette];
        int gce = (frames > 1) | (anim != 0) ? 1 : 0;
        int loopExt = (frames > 1) & (prm[GifEncParam.LoopCount] >= 0) ? 1 : 0;
        gstate[GifEncGlobal.GceMode] = gce;
        gstate[GifEncGlobal.LoopExt] = loopExt;
        int header = 13 + (loopExt != 0 ? 19 : 0);
        for (int f = 0; f < frames; f++)
        {
            int fi = f * GifEncFrame.Stride;
            int entries = finfo[fi + GifEncFrame.PalCount] + (finfo[fi + GifEncFrame.Trans] >= 0 ? 1 : 0);
            int size = TableSizeFor(entries);
            finfo[fi + GifEncFrame.TableSize] = size;
            finfo[fi + GifEncFrame.Mcs] = Log2(size) < 2 ? 2 : Log2(size);
            int local = single != 0 ? 0 : 1;
            finfo[fi + GifEncFrame.Local] = local;
            finfo[fi + GifEncFrame.HasGce] = gce;
            // -1 = automatic: an animation with transparency clears each frame before the next (restore to
            // background) so that every frame shows exactly its own pixels; without transparency frames just stay.
            int disposal = disposals[f];
            if (disposal < 0)
            {
                disposal = 0;
                if (frames > 1) disposal = anim != 0 ? 2 : 1;
            }
            finfo[fi + GifEncFrame.Disposal] = disposal;
            finfo[fi + GifEncFrame.PrefixLen] = (gce != 0 ? 8 : 0) + 10 + (local != 0 ? 3 * size : 0) + 1;
            if ((f == 0) & (single != 0)) header += 3 * size;
        }
        gstate[GifEncGlobal.HeaderLen] = header;
        gstate[GifEncGlobal.Running] = header;
    }

    // ------------------------------------------------------------------
    // pixels
    // ------------------------------------------------------------------

    private static void MapKernel(Index1D index, ArrayView<byte> rgbaBatch, ArrayView<byte> idxBatch, ArrayView<int> fpal,
        ArrayView<int> finfo, ArrayView<int> prm, int batchFirst)
    {
        int width = prm[GifEncParam.Width];
        int pixels = width * prm[GifEncParam.Height];
        int i = index.X;
        int fb = i / pixels;
        int p = i - fb * pixels;
        int f = batchFirst + fb;
        int fi = f * GifEncFrame.Stride;
        int exact = finfo[fi + GifEncFrame.Exact];
        int dither = prm[GifEncParam.Dither];
        bool fsFrame = (dither == 2) & (exact == 0);
        if (!fsFrame)
        {
            int o = i * 4;
            int result = 0;
            if (rgbaBatch[o + 3] < prm[GifEncParam.AlphaThreshold])
            {
                result = finfo[fi + GifEncFrame.Trans];
            }
            else
            {
                int r = rgbaBatch[o];
                int g = rgbaBatch[o + 1];
                int b = rgbaBatch[o + 2];
                int count = finfo[fi + GifEncFrame.PalCount];
                if (exact != 0)
                {
                    result = ExactIndex(fpal, f * 256, count, (r << 16) | (g << 8) | b);
                }
                else
                {
                    if (dither == 1)
                    {
                        int x = p % width;
                        int y = p / width;
                        int t = ((2 * Bayer8(x, y) - 63) * 16) >> 6;
                        r = Clamp255(r + t);
                        g = Clamp255(g + t);
                        b = Clamp255(b + t);
                    }
                    result = NearestIndex(fpal, f * 256, count, r, g, b);
                }
            }
            idxBatch[i] = (byte)result;
        }
    }

    private static void FsWaveKernel(Index1D index, ArrayView<byte> rgbaFrame, ArrayView<byte> idxFrame, ArrayView<int> err,
        ArrayView<int> fpal, ArrayView<int> finfo, ArrayView<int> prm, int wave, int frame)
    {
        int width = prm[GifEncParam.Width];
        int y = index.X;
        int x = wave - 2 * y;
        bool inside = (x >= 0) & (x < width);
        if (inside)
        {
            int p = y * width + x;
            int o = p * 4;
            int fi = frame * GifEncFrame.Stride;
            int e0 = 0, e1 = 0, e2 = 0;
            int result = 0;
            if (rgbaFrame[o + 3] < prm[GifEncParam.AlphaThreshold])
            {
                result = finfo[fi + GifEncFrame.Trans];
            }
            else
            {
                int accR = 0, accG = 0, accB = 0;
                if (x > 0)
                {
                    int q = (p - 1) * 3;
                    accR += 7 * err[q];
                    accG += 7 * err[q + 1];
                    accB += 7 * err[q + 2];
                }
                if (y > 0)
                {
                    if (x + 1 < width)
                    {
                        int q = (p - width + 1) * 3;
                        accR += 3 * err[q];
                        accG += 3 * err[q + 1];
                        accB += 3 * err[q + 2];
                    }
                    int q2 = (p - width) * 3;
                    accR += 5 * err[q2];
                    accG += 5 * err[q2 + 1];
                    accB += 5 * err[q2 + 2];
                    if (x > 0)
                    {
                        int q3 = (p - width - 1) * 3;
                        accR += err[q3];
                        accG += err[q3 + 1];
                        accB += err[q3 + 2];
                    }
                }
                int r = Clamp255(rgbaFrame[o] + ((accR + 8) >> 4));
                int g = Clamp255(rgbaFrame[o + 1] + ((accG + 8) >> 4));
                int b = Clamp255(rgbaFrame[o + 2] + ((accB + 8) >> 4));
                int count = finfo[fi + GifEncFrame.PalCount];
                result = NearestIndex(fpal, frame * 256, count, r, g, b);
                int c = fpal[frame * 256 + result];
                e0 = r - ((c >> 16) & 255);
                e1 = g - ((c >> 8) & 255);
                e2 = b - (c & 255);
            }
            idxFrame[p] = (byte)result;
            err[p * 3] = e0;
            err[p * 3 + 1] = e1;
            err[p * 3 + 2] = e2;
        }
    }

    // ------------------------------------------------------------------
    // LZW
    // ------------------------------------------------------------------

    private struct BitWriter
    {
        public int Pos;
        public int Acc;
        public int Bits;
    }

    private static void Emit(ref BitWriter w, ArrayView<byte> seg, int ob, int code, int width)
    {
        w.Acc |= code << w.Bits;
        w.Bits += width;
        bool flush = w.Bits >= 8;
        while (flush)
        {
            seg[ob + w.Pos] = (byte)(w.Acc & 255);
            w.Pos++;
            w.Acc >>= 8;
            w.Bits -= 8;
            flush = w.Bits >= 8;
        }
    }

    private static void LzwSegmentKernel(Index1D index, ArrayView<byte> idxBatch, ArrayView<byte> seg, ArrayView<int> segInfo,
        ArrayView<int> hash, ArrayView<int> finfo, ArrayView<int> prm, int batchFirst, int segStart, int segBound)
    {
        int segCount = prm[GifEncParam.SegCount];
        int segPixels = prm[GifEncParam.SegPixels];
        int pixels = prm[GifEncParam.Width] * prm[GifEncParam.Height];
        int g = segStart + index.X;
        int fb = g / segCount;
        int s = g - fb * segCount;
        int f = batchFirst + fb;
        int pixBase = fb * pixels + s * segPixels;
        int cnt = pixels - s * segPixels;
        if (cnt > segPixels) cnt = segPixels;
        int mcs = finfo[f * GifEncFrame.Stride + GifEncFrame.Mcs];
        int clear = 1 << mcs;
        int eoi = clear + 1;
        int hb = index.X * HashSlots;
        int ob = g * segBound;

        for (int i = 0; i < HashSlots; i++) hash[hb + i] = 0;
        int next = eoi + 1;
        int width = mcs + 1;
        var w = new BitWriter { Pos = 0, Acc = 0, Bits = 0 };
        if (s == 0) Emit(ref w, seg, ob, clear, width);

        int prefix = idxBatch[pixBase];
        for (int k = 1; k < cnt; k++)
        {
            int c = idxBatch[pixBase + k];
            uint key = (uint)((prefix << 8) | c);
            int slot = (int)((key * 2654435761u) >> 19);
            int found = -1;
            bool probing = true;
            while (probing)
            {
                uint e = (uint)hash[hb + slot];
                if (e == 0)
                {
                    probing = false;
                }
                else if ((e >> 12) == key)
                {
                    found = (int)(e & 4095u);
                    probing = false;
                }
                else
                {
                    slot = (slot + 1) & (HashSlots - 1);
                }
            }
            if (found >= 0)
            {
                prefix = found;
            }
            else
            {
                Emit(ref w, seg, ob, prefix, width);
                hash[hb + slot] = (int)((key << 12) | (uint)next);
                next++;
                if ((next > (1 << width)) & (width < 12)) width++;
                if (next == 4096)
                {
                    Emit(ref w, seg, ob, clear, width);
                    for (int i = 0; i < HashSlots; i++) hash[hb + i] = 0;
                    next = eoi + 1;
                    width = mcs + 1;
                }
                prefix = c;
            }
        }
        Emit(ref w, seg, ob, prefix, width);
        // The decoder adds an entry after the last data code too, and may widen before it reads the next code.
        if (next < 4096)
        {
            next++;
            if ((next > (1 << width)) & (width < 12)) width++;
        }
        Emit(ref w, seg, ob, s == segCount - 1 ? eoi : clear, width);
        int bitLen = w.Pos * 8 + w.Bits;
        if (w.Bits > 0) seg[ob + w.Pos] = (byte)(w.Acc & 255);
        segInfo[g] = bitLen;
    }

    private static void SegOffsetsKernel(Index1D index, ArrayView<int> segInfo, ArrayView<int> segOff, ArrayView<int> finfo,
        ArrayView<int> prm, int batchFirst)
    {
        int segCount = prm[GifEncParam.SegCount];
        int fb = index.X;
        int running = 0;
        for (int s = 0; s < segCount; s++)
        {
            segOff[fb * segCount + s] = running;
            running += segInfo[fb * segCount + s];
        }
        finfo[(batchFirst + fb) * GifEncFrame.Stride + GifEncFrame.PayloadBytes] = (running + 7) >> 3;
    }

    private static void FrameOffsetsKernel(Index1D _, ArrayView<int> gstate, ArrayView<int> finfo, int batchFirst, int frames)
    {
        int running = gstate[GifEncGlobal.Running];
        for (int j = 0; j < frames; j++)
        {
            int fi = (batchFirst + j) * GifEncFrame.Stride;
            int payload = finfo[fi + GifEncFrame.PayloadBytes];
            int framed = payload + (payload + 254) / 255 + 1;
            finfo[fi + GifEncFrame.FramedLen] = framed;
            finfo[fi + GifEncFrame.OutOffset] = running;
            running += finfo[fi + GifEncFrame.PrefixLen] + framed;
        }
        gstate[GifEncGlobal.Running] = running;
    }

    private static void GatherKernel(Index1D index, ArrayView<byte> seg, ArrayView<int> segInfo, ArrayView<int> segOff,
        ArrayView<int> payload, ArrayView<int> prm, int segBound, int payloadWords)
    {
        int segCount = prm[GifEncParam.SegCount];
        int g = index.X;
        int fb = g / segCount;
        int bits = segInfo[g];
        int off = segOff[g];
        int nbytes = (bits + 7) >> 3;
        int pw = fb * payloadWords;
        for (int i = 0; i < nbytes; i++)
        {
            uint b = seg[g * segBound + i];
            int o = off + i * 8;
            int wi = pw + (o >> 5);
            int sh = o & 31;
            Atomic.Or(ref payload[wi], (int)(b << sh));
            if (sh > 24) Atomic.Or(ref payload[wi + 1], (int)(b >> (32 - sh)));
        }
    }

    // ------------------------------------------------------------------
    // file assembly
    // ------------------------------------------------------------------

    private static void Put16(ArrayView<byte> o, int p, int v)
    {
        o[p] = (byte)(v & 255);
        o[p + 1] = (byte)((v >> 8) & 255);
    }

    /// <summary>Writes the table of frame <paramref name="f"/>: entries, the transparent entry (black), zero padding up to the table size.</summary>
    private static void PutTable(ArrayView<byte> o, int p, ArrayView<int> fpal, ArrayView<int> finfo, int f)
    {
        int fi = f * GifEncFrame.Stride;
        int count = finfo[fi + GifEncFrame.PalCount];
        int size = finfo[fi + GifEncFrame.TableSize];
        for (int i = 0; i < size; i++)
        {
            int c = 0;
            if (i < count) c = fpal[f * 256 + i];
            o[p + i * 3] = (byte)((c >> 16) & 255);
            o[p + i * 3 + 1] = (byte)((c >> 8) & 255);
            o[p + i * 3 + 2] = (byte)(c & 255);
        }
    }

    private static void HeaderKernel(Index1D _, ArrayView<byte> output, ArrayView<int> gstate, ArrayView<int> finfo,
        ArrayView<int> fpal, ArrayView<int> prm)
    {
        output[0] = 0x47; output[1] = 0x49; output[2] = 0x46; output[3] = 0x38; output[4] = 0x39; output[5] = 0x61;
        Put16(output, 6, prm[GifEncParam.Width]);
        Put16(output, 8, prm[GifEncParam.Height]);
        int p = 13;
        int packed = 0;
        if (gstate[GifEncGlobal.SinglePalette] != 0)
        {
            int size = finfo[GifEncFrame.TableSize];
            packed = 0x80 | (7 << 4) | (Log2(size) - 1);
            PutTable(output, 13, fpal, finfo, 0);
            p += 3 * size;
        }
        output[10] = (byte)packed;
        output[11] = 0;
        output[12] = 0;
        if (gstate[GifEncGlobal.LoopExt] != 0)
        {
            output[p] = 0x21; output[p + 1] = 0xFF; output[p + 2] = 0x0B;
            output[p + 3] = 0x4E; output[p + 4] = 0x45; output[p + 5] = 0x54; output[p + 6] = 0x53;
            output[p + 7] = 0x43; output[p + 8] = 0x41; output[p + 9] = 0x50; output[p + 10] = 0x45;
            output[p + 11] = 0x32; output[p + 12] = 0x2E; output[p + 13] = 0x30;
            output[p + 14] = 3; output[p + 15] = 1;
            Put16(output, p + 16, prm[GifEncParam.LoopCount]);
            output[p + 18] = 0;
        }
    }

    private static void WritePrefixKernel(Index1D index, ArrayView<byte> output, ArrayView<int> gstate, ArrayView<int> finfo,
        ArrayView<int> fpal, ArrayView<int> prm, ArrayView<int> delays, int batchFirst)
    {
        int f = batchFirst + index.X;
        int fi = f * GifEncFrame.Stride;
        int p = finfo[fi + GifEncFrame.OutOffset];
        int trans = finfo[fi + GifEncFrame.Trans];
        if (finfo[fi + GifEncFrame.HasGce] != 0)
        {
            output[p] = 0x21; output[p + 1] = 0xF9; output[p + 2] = 4;
            output[p + 3] = (byte)((finfo[fi + GifEncFrame.Disposal] << 2) | (trans >= 0 ? 1 : 0));
            Put16(output, p + 4, delays[f]);
            output[p + 6] = (byte)(trans >= 0 ? trans : 0);
            output[p + 7] = 0;
            p += 8;
        }
        output[p] = 0x2C;
        Put16(output, p + 1, 0);
        Put16(output, p + 3, 0);
        Put16(output, p + 5, prm[GifEncParam.Width]);
        Put16(output, p + 7, prm[GifEncParam.Height]);
        int size = finfo[fi + GifEncFrame.TableSize];
        bool local = finfo[fi + GifEncFrame.Local] != 0;
        output[p + 9] = (byte)(local ? (0x80 | (Log2(size) - 1)) : 0);
        p += 10;
        if (local)
        {
            PutTable(output, p, fpal, finfo, f);
            p += 3 * size;
        }
        output[p] = (byte)finfo[fi + GifEncFrame.Mcs];
    }

    private static void CopyFramedKernel(Index1D index, ArrayView<byte> output, ArrayView<int> finfo, ArrayView<int> payload,
        int batchFirst, int payloadWords, int framedBound)
    {
        int i = index.X;
        int fb = i / framedBound;
        int j = i - fb * framedBound;
        int fi = (batchFirst + fb) * GifEncFrame.Stride;
        int framed = finfo[fi + GifEncFrame.FramedLen];
        if (j < framed)
        {
            int total = finfo[fi + GifEncFrame.PayloadBytes];
            int v = 0;
            if (j < framed - 1)
            {
                int block = j >> 8;
                int o = j & 255;
                if (o == 0)
                {
                    int left = total - block * 255;
                    v = left > 255 ? 255 : left;
                }
                else
                {
                    int k = block * 255 + o - 1;
                    int word = payload[fb * payloadWords + (k >> 2)];
                    v = (word >> ((k & 3) * 8)) & 255;
                }
            }
            output[finfo[fi + GifEncFrame.OutOffset] + finfo[fi + GifEncFrame.PrefixLen] + j] = (byte)v;
        }
    }

    private static void TrailerKernel(Index1D _, ArrayView<byte> output, ArrayView<int> gstate)
    {
        int p = gstate[GifEncGlobal.Running];
        output[p] = 0x3B;
        gstate[GifEncGlobal.TotalLen] = p + 1;
    }

    /// <summary>No-op (kernels are owned by the accelerator).</summary>
    public void Dispose() { }
}
