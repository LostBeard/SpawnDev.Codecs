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
//        MedianCut (a chain of bounded launches) median cut over the occupied bins: repeatedly split the box with the
//                                                highest (population >> s) * longest extent along its longest axis
//                                                (ties: R, then G, then B) at the population median; the palette entry
//                                                of a box is its pixel-weighted mean colour; entries sorted ascending.
//                                                Bin work runs over MedLanes lanes (compaction, axis histogram, stable
//                                                partition, extents, colour sums with integer atomics); per split only
//                                                the box choice, a 32-value median search and a prefix over the lanes
//                                                of the box are one thread. MaxColors - 1 split rounds (5 launches each)
//                                                are launched, a round after the last split does nothing.
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
//     4. LzwInitKernel / LzwStepKernel           LZW over SegmentPixels pixels: a clear code ends every segment (the
//        (1 thread per segment, resumable)       last one ends with the end code), so segments are independent and the
//                                                bit streams just concatenate. The dictionary is an open-addressing hash
//                                                table in per-thread scratch (cleared by the host before the first step);
//                                                one launch encodes LzwChunk pixels of every segment and keeps the
//                                                dictionary / bit writer state in device memory for the next one.
//     5. SegOffsetsKernel / FrameOffsetsKernel   prefix sums of the segment bit lengths (at most 4096 segments per
//                                                frame) and of the frame lengths (at most 64 frames per batch).
//     6. GatherKernel (1 thread per 64 bytes)    OR every segment's bits into its frame payload at its bit offset.
//
// No launch does serial work that grows with the input (a long launch stalls the desktop compositor and trips the OS
// GPU watchdog): every one-thread loop above is bounded by a constant, the serial stages are resumable chains of
// launches whose count the host knows in advance.
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
    /// <summary>
    /// Pixels of every segment one LZW launch encodes: a pixel is a hash probe chain of dependent loads (about 0.7 us on
    /// one desktop GPU thread), plus at most two dictionary resets of HashSlots stores per chunk: 4096 was a 17.8 ms launch
    /// on CUDA (Nsight), 1024 is ~4 ms.
    /// </summary>
    public const int LzwChunk = 1024;
    /// <summary>Ints of resumable state per LZW thread.</summary>
    public const int LzwStateInts = 8;
    /// <summary>Segment bytes one gather thread ORs into the payload.</summary>
    public const int GatherBytes = 64;
    /// <summary>Lanes of the median cut's bin kernels (MedLanes * MedLaneBins = the 32768 bins).</summary>
    public const int MedLanes = 1024;
    /// <summary>Consecutive bin positions per lane.</summary>
    public const int MedLaneBins = 32;

    // LZW state fields (relative to thread * LzwStateInts)
    private const int LeK = 0, LePrefix = 1, LeNext = 2, LeWidth = 3, LePos = 4, LeAcc = 5, LeBits = 6, LeDone = 7;

    // quantizer scratch layout (ints)
    public const int HistOff = 0;
    public const int HistInts = 32768 * 4;
    public const int BinsOff = HistOff + HistInts;
    public const int TmpOff = BinsOff + 32768;
    public const int BoxOff = TmpOff + 32768;
    public const int BoxStride = 10;
    public const int H1Off = BoxOff + 256 * BoxStride;
    /// <summary>Box that owns each position of Bins (kept up to date as boxes split).</summary>
    public const int OwnerOff = H1Off + 32 + 64;
    /// <summary>Per-lane counts / offsets.</summary>
    public const int LaneOff = OwnerOff + 32768;
    /// <summary>Per-lane population sums.</summary>
    public const int LaneSumOff = LaneOff + MedLanes;
    /// <summary>Per-box weighted R, G, B sums and weight sum.</summary>
    public const int AccOff = LaneSumOff + MedLanes;
    /// <summary>Per-box mean colour before sorting.</summary>
    public const int MeanOff = AccOff + 256 * 4;
    /// <summary>The split loop's control block (C* slots).</summary>
    public const int CtrlOff = MeanOff + 256;
    public const int QuantScratchInts = CtrlOff + 32;

    // control block slots
    private const int CNeed = 0, CTarget = 1, CSh = 2, CN = 3, CBoxes = 4, CActive = 5, CGo = 6, CBest = 7, CLo = 8,
        CHi = 9, CCnt = 10, CAxis = 11, CAxisMin = 12, CAxisMax = 13, CSplit = 14, CMid = 15, CPrevLo = 16, CPrevHi = 17,
        CPrevMid = 18, CPrevBest = 19, CPrevNew = 20;

    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _prepGlobal;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _framePalette;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _hist;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _medLaneCount;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _medSetup;
    private readonly Action<Index1D, ArrayView<int>> _medCompact;
    private readonly Action<Index1D, ArrayView<int>> _medSelect;
    private readonly Action<Index1D, ArrayView<int>> _medAxisHist;
    private readonly Action<Index1D, ArrayView<int>> _medPartCount;
    private readonly Action<Index1D, ArrayView<int>> _medPrefix;
    private readonly Action<Index1D, ArrayView<int>> _medScatter;
    private readonly Action<Index1D, ArrayView<int>> _medFinalCopy;
    private readonly Action<Index1D, ArrayView<int>> _medAccumulate;
    private readonly Action<Index1D, ArrayView<int>> _medMeans;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _medSort;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _layout;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int> _quantFlags;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _map;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int> _fsWave;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int, int> _lzwInit;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int, int> _lzwStep;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _segOffsets;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int, int> _frameOffsets;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int, int> _gather;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> _writePrefix;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int, int, int> _copyFramed;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>> _header;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>> _trailer;

    private readonly LaunchPacer _pacer;

    public GifEncoderKernels(Accelerator a)
    {
        ArgumentNullException.ThrowIfNull(a);
        _pacer = new LaunchPacer(a);
        _prepGlobal = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(PrepGlobalKernel);
        _framePalette = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(FramePaletteKernel);
        _hist = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(HistKernel);
        _medLaneCount = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(MedLaneCountKernel);
        _medSetup = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(MedSetupKernel);
        _medCompact = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(MedCompactKernel);
        _medSelect = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(MedSelectKernel);
        _medAxisHist = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(MedAxisHistKernel);
        _medPartCount = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(MedPartCountKernel);
        _medPrefix = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(MedPrefixKernel);
        _medScatter = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(MedScatterKernel);
        _medFinalCopy = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(MedFinalCopyKernel);
        _medAccumulate = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(MedAccumulateKernel);
        _medMeans = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(MedMeansKernel);
        _medSort = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(MedSortKernel);
        _layout = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(LayoutKernel);
        _quantFlags = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, int>(QuantFlagsKernel);
        _map = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(MapKernel);
        _fsWave = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(FsWaveKernel);
        _lzwInit = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int, int>(LzwInitKernel);
        _lzwStep = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int, int>(LzwStepKernel);
        _segOffsets = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(SegOffsetsKernel);
        _frameOffsets = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, int, int>(FrameOffsetsKernel);
        _gather = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int, int>(GatherKernel);
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

    /// <summary>
    /// Median cut palette of frame <paramref name="frame"/> from the histogram in <paramref name="scratch"/>
    /// (<see cref="Hist"/> first): a chain of bounded launches, <paramref name="maxColors"/> - 1 split rounds (an upper
    /// bound of the splits; a round after the last split does nothing). Nothing happens for a frame with an exact palette.
    /// </summary>
    public async ValueTask MedianCutAsync(ArrayView<int> scratch, ArrayView<int> finfo, ArrayView<int> gstate, ArrayView<int> fpal,
        ArrayView<int> prm, int frame, int maxColors)
    {
        _medLaneCount(MedLanes, scratch, finfo, gstate, frame);
        _medSetup(1, scratch, finfo, gstate, prm, frame);
        _medCompact(MedLanes, scratch);
        for (int round = 1; round < maxColors; round++)
        {
            _medSelect(1, scratch);
            _medAxisHist(MedLanes, scratch);
            _medPartCount(MedLanes, scratch);
            _medPrefix(1, scratch);
            _medScatter(MedLanes, scratch);
            await _pacer.LaunchedAsync();
        }
        _medFinalCopy(MedLanes, scratch);
        _medAccumulate(MedLanes, scratch);
        _medMeans(256, scratch);
        _medSort(256, scratch, finfo, gstate, fpal, frame);
    }

    /// <summary>Per-frame table size, minimum code size, prefix length, header length, running offset.</summary>
    public void Layout(ArrayView<int> gstate, ArrayView<int> finfo, ArrayView<int> prm, ArrayView<int> disposals, int frames)
        => _layout(frames, gstate, finfo, prm, disposals);

    /// <summary>Bit f of <paramref name="flags"/> (32 per int) = frame f has no exact palette ((frames + 31) / 32 ints).</summary>
    public void QuantFlags(ArrayView<int> finfo, ArrayView<int> flags, int frames)
        => _quantFlags((frames + 31) / 32, finfo, flags, frames);

    public void Map(ArrayView<byte> rgbaBatch, ArrayView<byte> idxBatch, ArrayView<int> fpal, ArrayView<int> finfo,
        ArrayView<int> prm, int batchFirst, int threads)
        => _map(threads, rgbaBatch, idxBatch, fpal, finfo, prm, batchFirst);

    public void FsWave(ArrayView<byte> rgbaFrame, ArrayView<byte> idxFrame, ArrayView<int> err, ArrayView<int> fpal,
        ArrayView<int> finfo, ArrayView<int> prm, int wave, int frame, int rows)
        => _fsWave(rows, rgbaFrame, idxFrame, err, fpal, finfo, prm, wave, frame);

    /// <summary>Launches of <see cref="LzwStep"/> that finish every segment of a <paramref name="pixels"/> pixel frame.</summary>
    public static int LzwSteps(int pixels)
    {
        int cnt = Math.Min(pixels, SegmentPixels);
        return Math.Max(1, (cnt - 1 + LzwChunk - 1) / LzwChunk);
    }

    /// <summary>
    /// Start <paramref name="threads"/> segments (from <paramref name="segStart"/>) of the batch: their hash tables
    /// must be zero. Then <see cref="LzwStep"/> <see cref="LzwSteps"/> times.
    /// </summary>
    public void LzwInit(ArrayView<byte> idxBatch, ArrayView<byte> seg, ArrayView<int> state, ArrayView<int> finfo,
        ArrayView<int> prm, int batchFirst, int segStart, int segBound, int threads)
        => _lzwInit(threads, idxBatch, seg, state, finfo, prm, batchFirst, segStart, segBound);

    /// <summary>The next <see cref="LzwChunk"/> pixels of each of <paramref name="threads"/> segments.</summary>
    public void LzwStep(ArrayView<byte> idxBatch, ArrayView<byte> seg, ArrayView<int> segInfo, ArrayView<int> hash,
        ArrayView<int> finfo, ArrayView<int> prm, ArrayView<int> state, int batchFirst, int segStart, int segBound, int threads)
        => _lzwStep(threads, idxBatch, seg, segInfo, hash, finfo, prm, state, batchFirst, segStart, segBound);

    public void SegOffsets(ArrayView<int> segInfo, ArrayView<int> segOff, ArrayView<int> finfo, ArrayView<int> prm,
        int batchFirst, int framesInBatch)
        => _segOffsets(framesInBatch, segInfo, segOff, finfo, prm, batchFirst);

    public void FrameOffsets(ArrayView<int> gstate, ArrayView<int> finfo, int batchFirst, int framesInBatch)
        => _frameOffsets(1, gstate, finfo, batchFirst, framesInBatch);

    /// <summary>OR the bits of <paramref name="segments"/> segments into their frame payloads (one thread per <see cref="GatherBytes"/> bytes).</summary>
    public void Gather(ArrayView<byte> seg, ArrayView<int> segInfo, ArrayView<int> segOff, ArrayView<int> payload,
        ArrayView<int> prm, int segBound, int payloadWords, int segments)
    {
        int groups = (segBound + GatherBytes - 1) / GatherBytes;
        _gather(checked(segments * groups), seg, segInfo, segOff, payload, prm, segBound, payloadWords, groups);
    }

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

    private static int AxisValue(int bin, int axis)
    {
        int v = bin & 31;
        if (axis == 0) v = bin >> 10;
        else if (axis == 1) v = (bin >> 5) & 31;
        return v;
    }

    // Median cut, split into bounded launches. The occupied bins (ascending) live in Bins[0, n); a box is a range
    // [lo, hi) of Bins plus its population and 5-bit R/G/B extents. Work over bins is spread over MedLanes lanes of
    // MedLaneBins consecutive bin positions each; only the box choice, the median search over 32 values and a prefix
    // over the lanes a box covers are one thread. Sums, minima and maxima are combined with integer atomics, so the
    // result does not depend on the thread schedule and equals the sequential algorithm exactly.

    private static bool MedNeed(ArrayView<int> finfo, ArrayView<int> gstate, int frame)
        => (gstate[GifEncGlobal.Exact] == 0) & (finfo[frame * GifEncFrame.Stride + GifEncFrame.Exact] == 0);

    private static void InitBox(ArrayView<int> sc, int box, int lo, int hi, int count)
    {
        int bo = BoxOff + box * BoxStride;
        sc[bo] = lo;
        sc[bo + 1] = hi;
        sc[bo + 2] = count;
        sc[bo + 3] = 99;
        sc[bo + 4] = -1;
        sc[bo + 5] = 99;
        sc[bo + 6] = -1;
        sc[bo + 7] = 99;
        sc[bo + 8] = -1;
    }

    /// <summary>Folds lane-local population and R/G/B extents into box <paramref name="box"/>.</summary>
    private static void MergeBox(ArrayView<int> sc, int box, int count, int minR, int maxR, int minG, int maxG, int minB, int maxB)
    {
        int bo = BoxOff + box * BoxStride;
        Atomic.Add(ref sc[bo + 2], count);
        Atomic.Min(ref sc[bo + 3], minR);
        Atomic.Max(ref sc[bo + 4], maxR);
        Atomic.Min(ref sc[bo + 5], minG);
        Atomic.Max(ref sc[bo + 6], maxG);
        Atomic.Min(ref sc[bo + 7], minB);
        Atomic.Max(ref sc[bo + 8], maxB);
    }

    /// <summary>Lane <c>index</c>: occupied bins and their population among its MedLaneBins bins.</summary>
    private static void MedLaneCountKernel(Index1D index, ArrayView<int> sc, ArrayView<int> finfo, ArrayView<int> gstate, int frame)
    {
        int lane = index.X;
        int occupied = 0;
        int sum = 0;
        if (MedNeed(finfo, gstate, frame))
        {
            int b0 = lane * MedLaneBins;
            for (int j = 0; j < MedLaneBins; j++)
            {
                int c = sc[HistOff + (b0 + j) * 4];
                if (c != 0)
                {
                    occupied++;
                    sum += c;
                }
            }
        }
        sc[LaneOff + lane] = occupied;
        sc[LaneSumOff + lane] = sum;
    }

    /// <summary>One thread: lane offsets of the compaction, n, the score shift, box 0, the split loop state.</summary>
    private static void MedSetupKernel(Index1D _, ArrayView<int> sc, ArrayView<int> finfo, ArrayView<int> gstate,
        ArrayView<int> prm, int frame)
    {
        int need = MedNeed(finfo, gstate, frame) ? 1 : 0;
        int n = 0;
        int total = 0;
        for (int lane = 0; lane < MedLanes; lane++)
        {
            int c = sc[LaneOff + lane];
            sc[LaneOff + lane] = n;
            n += c;
            total += sc[LaneSumOff + lane];
        }
        int sh = 0;
        bool shifting = (total >> sh) >= (1 << 26);
        while (shifting)
        {
            sh++;
            shifting = (total >> sh) >= (1 << 26);
        }
        int target = prm[GifEncParam.MaxColors] - gstate[GifEncGlobal.AnimHasTrans];
        int boxes = 0;
        if ((need != 0) & (n > 0))
        {
            InitBox(sc, 0, 0, n, total);
            boxes = 1;
        }
        int ct = CtrlOff;
        sc[ct + CNeed] = need;
        sc[ct + CTarget] = target;
        sc[ct + CSh] = sh;
        sc[ct + CN] = n;
        sc[ct + CBoxes] = boxes;
        sc[ct + CActive] = (boxes > 0) & (boxes < target) ? 1 : 0;
        sc[ct + CGo] = 0;
        sc[ct + CPrevLo] = 0;
        sc[ct + CPrevHi] = 0;
        sc[ct + CPrevMid] = 0;
        sc[ct + CPrevBest] = 0;
        sc[ct + CPrevNew] = 0;
    }

    /// <summary>Lane <c>index</c>: writes its occupied bins (ascending) to Bins, all in box 0, and folds them into box 0's extents.</summary>
    private static void MedCompactKernel(Index1D index, ArrayView<int> sc)
    {
        int lane = index.X;
        if (sc[CtrlOff + CNeed] != 0)
        {
            int w = sc[LaneOff + lane];
            int b0 = lane * MedLaneBins;
            int found = 0;
            int minR = 99, maxR = -1, minG = 99, maxG = -1, minB = 99, maxB = -1;
            for (int j = 0; j < MedLaneBins; j++)
            {
                int bin = b0 + j;
                if (sc[HistOff + bin * 4] != 0)
                {
                    sc[BinsOff + w] = bin;
                    sc[OwnerOff + w] = 0;
                    w++;
                    found++;
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
            }
            // box 0's population is already the total: only the extents are folded in
            if (found > 0) MergeBox(sc, 0, 0, minR, maxR, minG, maxG, minB, maxB);
        }
    }

    /// <summary>
    /// One thread, once per split: the box with the highest (population >> shift) * longest extent (first wins a tie)
    /// among the boxes of at least 2 bins, and its longest axis (ties: R, then G, then B). Clears the axis histogram.
    /// </summary>
    private static void MedSelectKernel(Index1D _, ArrayView<int> sc)
    {
        int ct = CtrlOff;
        int go = 0;
        bool active = (sc[ct + CNeed] != 0) & (sc[ct + CActive] != 0);
        if (active)
        {
            int boxes = sc[ct + CBoxes];
            int sh = sc[ct + CSh];
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
                sc[ct + CActive] = 0;
            }
            else
            {
                int bo = BoxOff + best * BoxStride;
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
                sc[ct + CBest] = best;
                sc[ct + CLo] = sc[bo];
                sc[ct + CHi] = sc[bo + 1];
                sc[ct + CCnt] = sc[bo + 2];
                sc[ct + CAxis] = axis;
                sc[ct + CAxisMin] = sc[bo + 3 + axis * 2];
                sc[ct + CAxisMax] = sc[bo + 4 + axis * 2];
                for (int v = 0; v < 32; v++) sc[H1Off + v] = 0;
                go = 1;
            }
        }
        sc[ct + CGo] = go;
    }

    /// <summary>
    /// Lane <c>index</c>: first finishes the previous split (copies its partitioned range back from Tmp to Bins and
    /// records the owner box of every position), then adds the population of the chosen box's bins to the axis histogram.
    /// </summary>
    private static void MedAxisHistKernel(Index1D index, ArrayView<int> sc)
    {
        int ct = CtrlOff;
        int k0 = index.X * MedLaneBins;
        MedCopyBack(sc, k0);
        if (sc[ct + CGo] != 0)
        {
            int lo = sc[ct + CLo];
            int hi = sc[ct + CHi];
            int axis = sc[ct + CAxis];
            for (int j = 0; j < MedLaneBins; j++)
            {
                int k = k0 + j;
                if ((k >= lo) & (k < hi))
                {
                    int bin = sc[BinsOff + k];
                    Atomic.Add(ref sc[H1Off + AxisValue(bin, axis)], sc[HistOff + bin * 4]);
                }
            }
        }
    }

    /// <summary>Positions [k0, k0 + MedLaneBins) of the previous split's range: Tmp back to Bins, owner = the box the position now belongs to.</summary>
    private static void MedCopyBack(ArrayView<int> sc, int k0)
    {
        int ct = CtrlOff;
        int plo = sc[ct + CPrevLo];
        int phi = sc[ct + CPrevHi];
        int pmid = sc[ct + CPrevMid];
        int pbest = sc[ct + CPrevBest];
        int pnew = sc[ct + CPrevNew];
        for (int j = 0; j < MedLaneBins; j++)
        {
            int k = k0 + j;
            if ((k >= plo) & (k < phi))
            {
                sc[BinsOff + k] = sc[TmpOff + k];
                sc[OwnerOff + k] = k < pmid ? pbest : pnew;
            }
        }
    }

    /// <summary>The axis value at which the chosen box is split: the population median, at most axisMax - 1.</summary>
    private static int MedSplitValue(ArrayView<int> sc)
    {
        int ct = CtrlOff;
        int axisMin = sc[ct + CAxisMin];
        int axisMax = sc[ct + CAxisMax];
        int cnt = sc[ct + CCnt];
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
        return split;
    }

    /// <summary>Lane <c>index</c>: how many of its positions inside the chosen box go to the low side (axis value &lt;= split).</summary>
    private static void MedPartCountKernel(Index1D index, ArrayView<int> sc)
    {
        int ct = CtrlOff;
        if (sc[ct + CGo] != 0)
        {
            int lane = index.X;
            int split = MedSplitValue(sc);
            if (lane == 0) sc[ct + CSplit] = split;
            int lo = sc[ct + CLo];
            int hi = sc[ct + CHi];
            int axis = sc[ct + CAxis];
            int k0 = lane * MedLaneBins;
            bool overlaps = (k0 + MedLaneBins > lo) & (k0 < hi);
            if (overlaps)
            {
                int low = 0;
                for (int j = 0; j < MedLaneBins; j++)
                {
                    int k = k0 + j;
                    if ((k >= lo) & (k < hi))
                    {
                        if (AxisValue(sc[BinsOff + k], axis) <= split) low++;
                    }
                }
                sc[LaneOff + lane] = low;
            }
        }
    }

    /// <summary>
    /// One thread: prefix of the low counts over the lanes the chosen box covers (at most MedLanes), the split point,
    /// the two new boxes (the low half keeps the box's slot, the high half is appended).
    /// </summary>
    private static void MedPrefixKernel(Index1D _, ArrayView<int> sc)
    {
        int ct = CtrlOff;
        if (sc[ct + CGo] != 0)
        {
            int lo = sc[ct + CLo];
            int hi = sc[ct + CHi];
            int l0 = lo / MedLaneBins;
            int l1 = (hi - 1) / MedLaneBins;
            int running = 0;
            for (int lane = l0; lane <= l1; lane++)
            {
                int c = sc[LaneOff + lane];
                sc[LaneOff + lane] = running;
                running += c;
            }
            int mid = lo + running;
            int best = sc[ct + CBest];
            int boxes = sc[ct + CBoxes];
            InitBox(sc, best, lo, mid, 0);
            InitBox(sc, boxes, mid, hi, 0);
            sc[ct + CMid] = mid;
            sc[ct + CPrevLo] = lo;
            sc[ct + CPrevHi] = hi;
            sc[ct + CPrevMid] = mid;
            sc[ct + CPrevBest] = best;
            sc[ct + CPrevNew] = boxes;
            boxes++;
            sc[ct + CBoxes] = boxes;
            sc[ct + CActive] = boxes < sc[ct + CTarget] ? 1 : 0;
        }
    }

    /// <summary>
    /// Lane <c>index</c>: stable partition of its positions of the chosen box into Tmp (low side from lo, high side from
    /// mid, both in their original order), and their population and extents folded into the two new boxes.
    /// </summary>
    private static void MedScatterKernel(Index1D index, ArrayView<int> sc)
    {
        int ct = CtrlOff;
        if (sc[ct + CGo] != 0)
        {
            int lane = index.X;
            int lo = sc[ct + CLo];
            int hi = sc[ct + CHi];
            int k0 = lane * MedLaneBins;
            bool overlaps = (k0 + MedLaneBins > lo) & (k0 < hi);
            if (overlaps)
            {
                int axis = sc[ct + CAxis];
                int split = sc[ct + CSplit];
                int mid = sc[ct + CMid];
                int lows = sc[LaneOff + lane];      // low positions of the box before this lane
                int cL = 0, minRL = 99, maxRL = -1, minGL = 99, maxGL = -1, minBL = 99, maxBL = -1;
                int cH = 0, minRH = 99, maxRH = -1, minGH = 99, maxGH = -1, minBH = 99, maxBH = -1;
                int nL = 0, nH = 0;
                for (int j = 0; j < MedLaneBins; j++)
                {
                    int k = k0 + j;
                    if ((k >= lo) & (k < hi))
                    {
                        int bin = sc[BinsOff + k];
                        int c = sc[HistOff + bin * 4];
                        int r = bin >> 10;
                        int g = (bin >> 5) & 31;
                        int b = bin & 31;
                        if (AxisValue(bin, axis) <= split)
                        {
                            sc[TmpOff + lo + lows] = bin;
                            lows++;
                            nL++;
                            cL += c;
                            if (r < minRL) minRL = r;
                            if (r > maxRL) maxRL = r;
                            if (g < minGL) minGL = g;
                            if (g > maxGL) maxGL = g;
                            if (b < minBL) minBL = b;
                            if (b > maxBL) maxBL = b;
                        }
                        else
                        {
                            sc[TmpOff + mid + (k - lo - lows)] = bin;
                            nH++;
                            cH += c;
                            if (r < minRH) minRH = r;
                            if (r > maxRH) maxRH = r;
                            if (g < minGH) minGH = g;
                            if (g > maxGH) maxGH = g;
                            if (b < minBH) minBH = b;
                            if (b > maxBH) maxBH = b;
                        }
                    }
                }
                if (nL > 0) MergeBox(sc, sc[ct + CPrevBest], cL, minRL, maxRL, minGL, maxGL, minBL, maxBL);
                if (nH > 0) MergeBox(sc, sc[ct + CPrevNew], cH, minRH, maxRH, minGH, maxGH, minBH, maxBH);
            }
        }
    }

    /// <summary>Lane <c>index</c>: finishes the last split (copy back, owners) and clears the colour accumulators.</summary>
    private static void MedFinalCopyKernel(Index1D index, ArrayView<int> sc)
    {
        if (sc[CtrlOff + CNeed] != 0)
        {
            MedCopyBack(sc, index.X * MedLaneBins);
            if (index.X < 256)
            {
                int a = AccOff + index.X * 4;
                sc[a] = 0;
                sc[a + 1] = 0;
                sc[a + 2] = 0;
                sc[a + 3] = 0;
            }
        }
    }

    /// <summary>Weight shift of a box of <paramref name="cnt"/> pixels (keeps the weighted sums inside an int).</summary>
    private static int MeanShift(int cnt)
    {
        int s = 0;
        bool scaling = (cnt >> s) >= (1 << 23);
        while (scaling)
        {
            s++;
            scaling = (cnt >> s) >= (1 << 23);
        }
        return s;
    }

    /// <summary>
    /// Lane <c>index</c>: adds the weighted colour of each of its bins to its box's accumulators (positions of one box
    /// are contiguous, so a lane flushes once per box it touches).
    /// </summary>
    private static void MedAccumulateKernel(Index1D index, ArrayView<int> sc)
    {
        int ct = CtrlOff;
        if (sc[ct + CNeed] != 0)
        {
            int n = sc[ct + CN];
            int k0 = index.X * MedLaneBins;
            int cur = -1;
            int s = 0;
            int accR = 0, accG = 0, accB = 0, wsum = 0;
            for (int j = 0; j < MedLaneBins; j++)
            {
                int k = k0 + j;
                if (k < n)
                {
                    int box = sc[OwnerOff + k];
                    if (box != cur)
                    {
                        if (cur >= 0)
                        {
                            int a = AccOff + cur * 4;
                            Atomic.Add(ref sc[a], accR);
                            Atomic.Add(ref sc[a + 1], accG);
                            Atomic.Add(ref sc[a + 2], accB);
                            Atomic.Add(ref sc[a + 3], wsum);
                        }
                        cur = box;
                        s = MeanShift(sc[BoxOff + box * BoxStride + 2]);
                        accR = 0;
                        accG = 0;
                        accB = 0;
                        wsum = 0;
                    }
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
            }
            if (cur >= 0)
            {
                int a = AccOff + cur * 4;
                Atomic.Add(ref sc[a], accR);
                Atomic.Add(ref sc[a + 1], accG);
                Atomic.Add(ref sc[a + 2], accB);
                Atomic.Add(ref sc[a + 3], wsum);
            }
        }
    }

    /// <summary>Box <c>index</c>: its palette entry, the pixel-weighted mean colour.</summary>
    private static void MedMeansKernel(Index1D index, ArrayView<int> sc)
    {
        int ct = CtrlOff;
        int i = index.X;
        bool live = (sc[ct + CNeed] != 0) & (i < sc[ct + CBoxes]);
        if (live)
        {
            int a = AccOff + i * 4;
            int wsum = sc[a + 3];
            int mr = (sc[a] + wsum / 2) / wsum;
            int mg = (sc[a + 1] + wsum / 2) / wsum;
            int mb = (sc[a + 2] + wsum / 2) / wsum;
            sc[MeanOff + i] = (mr << 16) | (mg << 8) | mb;
        }
    }

    /// <summary>
    /// Box <c>index</c>: its entry goes to its rank in ascending order (equal entries keep their box order, like a
    /// stable sort); thread 0 records the palette size and the transparent index.
    /// </summary>
    private static void MedSortKernel(Index1D index, ArrayView<int> sc, ArrayView<int> finfo, ArrayView<int> gstate,
        ArrayView<int> fpal, int frame)
    {
        int ct = CtrlOff;
        int i = index.X;
        if (sc[ct + CNeed] != 0)
        {
            int boxes = sc[ct + CBoxes];
            if (i < boxes)
            {
                int v = sc[MeanOff + i];
                int rank = 0;
                for (int j = 0; j < boxes; j++)
                {
                    int u = sc[MeanOff + j];
                    bool before = (u < v) | ((u == v) & (j < i));
                    if (before) rank++;
                }
                fpal[frame * 256 + rank] = v;
            }
            if (i == 0)
            {
                int fi = frame * GifEncFrame.Stride;
                finfo[fi + GifEncFrame.PalCount] = boxes;
                finfo[fi + GifEncFrame.Exact] = 0;
                finfo[fi + GifEncFrame.Trans] = gstate[GifEncGlobal.AnimHasTrans] != 0 ? boxes : -1;
            }
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

    /// <summary>Frame <c>index</c>: table size, minimum code size, local table, disposal, prefix length; frame 0 also the header length.</summary>
    private static void LayoutKernel(Index1D index, ArrayView<int> gstate, ArrayView<int> finfo, ArrayView<int> prm, ArrayView<int> disposals)
    {
        int f = index.X;
        int frames = prm[GifEncParam.Frames];
        int anim = gstate[GifEncGlobal.AnimHasTrans];
        int single = gstate[GifEncGlobal.SinglePalette];
        int gce = (frames > 1) | (anim != 0) ? 1 : 0;
        int loopExt = (frames > 1) & (prm[GifEncParam.LoopCount] >= 0) ? 1 : 0;
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
        if (f == 0)
        {
            int header = 13 + (loopExt != 0 ? 19 : 0);
            if (single != 0) header += 3 * size;
            gstate[GifEncGlobal.GceMode] = gce;
            gstate[GifEncGlobal.LoopExt] = loopExt;
            gstate[GifEncGlobal.HeaderLen] = header;
            gstate[GifEncGlobal.Running] = header;
        }
    }

    /// <summary>Word <c>index</c> of the quantize flags: bit j = frame index * 32 + j has no exact palette (it gets a median cut palette).</summary>
    private static void QuantFlagsKernel(Index1D index, ArrayView<int> finfo, ArrayView<int> flags, int frames)
    {
        int bits = 0;
        for (int j = 0; j < 32; j++)
        {
            int f = index.X * 32 + j;
            if (f < frames)
            {
                if (finfo[f * GifEncFrame.Stride + GifEncFrame.Exact] == 0) bits |= 1 << j;
            }
        }
        flags[index.X] = bits;
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

    /// <summary>Which segment an LZW thread encodes.</summary>
    private struct LzwSegment
    {
        /// <summary>Segment index in the batch, frame in the batch, segment in the frame, first pixel, pixel count.</summary>
        public int G, Fb, S, PixBase, Cnt;
    }

    private static LzwSegment SegmentOf(ArrayView<int> prm, int segStart, int t)
    {
        int segCount = prm[GifEncParam.SegCount];
        int segPixels = prm[GifEncParam.SegPixels];
        int pixels = prm[GifEncParam.Width] * prm[GifEncParam.Height];
        var r = new LzwSegment();
        r.G = segStart + t;
        r.Fb = r.G / segCount;
        r.S = r.G - r.Fb * segCount;
        r.PixBase = r.Fb * pixels + r.S * segPixels;
        int cnt = pixels - r.S * segPixels;
        if (cnt > segPixels) cnt = segPixels;
        r.Cnt = cnt;
        return r;
    }

    /// <summary>
    /// Segment <c>segStart + index</c>: initial dictionary state (the hash table was cleared by the host), the leading
    /// clear code of a frame's first segment, the first pixel as the current prefix.
    /// </summary>
    private static void LzwInitKernel(Index1D index, ArrayView<byte> idxBatch, ArrayView<byte> seg, ArrayView<int> state,
        ArrayView<int> finfo, ArrayView<int> prm, int batchFirst, int segStart, int segBound)
    {
        int t = index.X;
        var sg = SegmentOf(prm, segStart, t);
        int mcs = finfo[(batchFirst + sg.Fb) * GifEncFrame.Stride + GifEncFrame.Mcs];
        int clear = 1 << mcs;
        var w = new BitWriter { Pos = 0, Acc = 0, Bits = 0 };
        if (sg.S == 0) Emit(ref w, seg, sg.G * segBound, clear, mcs + 1);
        int st = t * LzwStateInts;
        state[st + LeK] = 1;
        state[st + LePrefix] = idxBatch[sg.PixBase];
        state[st + LeNext] = clear + 2;
        state[st + LeWidth] = mcs + 1;
        state[st + LePos] = w.Pos;
        state[st + LeAcc] = w.Acc;
        state[st + LeBits] = w.Bits;
        state[st + LeDone] = 0;
    }

    /// <summary>
    /// Segment <c>segStart + index</c>: LZW over the next <see cref="LzwChunk"/> pixels from the state the previous
    /// launch left; the launch that reaches the end of the segment writes the last code, the trailing clear / end code
    /// and the segment's bit length. A launch after that does nothing.
    /// </summary>
    private static void LzwStepKernel(Index1D index, ArrayView<byte> idxBatch, ArrayView<byte> seg, ArrayView<int> segInfo,
        ArrayView<int> hash, ArrayView<int> finfo, ArrayView<int> prm, ArrayView<int> state, int batchFirst, int segStart, int segBound)
    {
        int t = index.X;
        int st = t * LzwStateInts;
        if (state[st + LeDone] == 0)
        {
            var sg = SegmentOf(prm, segStart, t);
            int g = sg.G;
            int s = sg.S;
            int pixBase = sg.PixBase;
            int cnt = sg.Cnt;
            int segCount = prm[GifEncParam.SegCount];
            int mcs = finfo[(batchFirst + sg.Fb) * GifEncFrame.Stride + GifEncFrame.Mcs];
            int clear = 1 << mcs;
            int eoi = clear + 1;
            int hb = t * HashSlots;
            int ob = g * segBound;

            int k = state[st + LeK];
            int prefix = state[st + LePrefix];
            int next = state[st + LeNext];
            int width = state[st + LeWidth];
            var w = new BitWriter { Pos = state[st + LePos], Acc = state[st + LeAcc], Bits = state[st + LeBits] };
            int limit = k + LzwChunk;
            if (limit > cnt) limit = cnt;
            bool going = k < limit;
            while (going)
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
                        // At most one dictionary reset per 4093 codes: a chunk of LzwChunk pixels sees at most two.
                        Emit(ref w, seg, ob, clear, width);
                        for (int i = 0; i < HashSlots; i++) hash[hb + i] = 0;
                        next = eoi + 1;
                        width = mcs + 1;
                    }
                    prefix = c;
                }
                k++;
                going = k < limit;
            }
            int done = 0;
            if (k >= cnt)
            {
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
                done = 1;
            }
            state[st + LeK] = k;
            state[st + LePrefix] = prefix;
            state[st + LeNext] = next;
            state[st + LeWidth] = width;
            state[st + LePos] = w.Pos;
            state[st + LeAcc] = w.Acc;
            state[st + LeBits] = w.Bits;
            state[st + LeDone] = done;
        }
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

    /// <summary>Bytes [part * GatherBytes, +GatherBytes) of segment g (thread = g * groups + part): OR them into the frame payload at the segment's bit offset.</summary>
    private static void GatherKernel(Index1D index, ArrayView<byte> seg, ArrayView<int> segInfo, ArrayView<int> segOff,
        ArrayView<int> payload, ArrayView<int> prm, int segBound, int payloadWords, int groups)
    {
        int segCount = prm[GifEncParam.SegCount];
        int g = index.X / groups;
        int part = index.X - g * groups;
        int fb = g / segCount;
        int bits = segInfo[g];
        int off = segOff[g];
        int nbytes = (bits + 7) >> 3;
        int pw = fb * payloadWords;
        int i0 = part * GatherBytes;
        int i1 = i0 + GatherBytes;
        if (i1 > nbytes) i1 = nbytes;
        for (int i = i0; i < i1; i++)
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
