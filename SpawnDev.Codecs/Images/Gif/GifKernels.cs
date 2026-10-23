// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The GIF decode kernels. Every stage that touches the file bytes or the pixels runs on the accelerator; the
// host only dispatches and reads back the metadata block, the frame table and per-frame status ints.
//
//   1. ScanInitKernel  (1 thread)            header, logical screen, global colour table; resets the walk state.
//      ScanStepKernel  (1 thread, resumable) the block walk: Graphic Control Extension, Netscape loop extension,
//                                            image descriptors (frame table); other extensions are skipped safely.
//                                            One launch does at most ScanChunk steps (a step = one block
//                                            introducer or one data sub-block hop) and continues from the walk
//                                            state kept in the meta header (ScanPhase..ScanDone).
//      IdentityKernel  (1 thread per frame)  whether the frame's colour table is the identity gray ramp.
//   2. LzwInitKernel   (1 thread per frame)  resets the frame's decoder state.
//      LzwStepKernel   (1 thread per frame,  variable-width LZW (2..12 bits, clear / end codes, deferred clear,
//                       resumable)           full 4096-entry table) reading the data sub-blocks in place into a
//                                            palette-index buffer. LZW is a sequential dependency chain, so a
//                                            frame is one thread; frames of a batch decode in parallel. One launch
//                                            decodes codes until LzwBudget work units (1 per code + 1 per pixel
//                                            written) are spent and stops at a code boundary; the bit reader and
//                                            dictionary state stay in device memory (the code table in the scratch,
//                                            the rest in the state block) for the next launch.
//      No launch does serial work that grows with the input: a long launch stalls the desktop compositor and trips
//      the OS GPU watchdog. The host knows an upper bound of the launches each stage needs (see the launchers).
//   3. CompositeKernel (1 thread per pixel)  one STEP of the compositing plan the host built from the frame table:
//                                            fill a rectangle (or the whole canvas) with a colour, save / restore a
//                                            frame rectangle (disposal), or draw a frame (de-interlace, palette
//                                            lookup, transparent index). The step carries every rule that differs
//                                            between the two compositing modes, so the kernel has no mode switch.
//
// Compositing modes (GifCompositing; the host plans the steps, see GifDecoderGpu):
//   Pillow (default): bit-exact with Pillow 13 (GifImagePlugin, default LOADING_STRATEGY, every frame read with
//     Image.seek + convert("RGBA")). Pillow's rules, all reproduced:
//       * the first frame is an indexed canvas: filled with its transparent index (else index 0), so the part a
//         smaller first frame does not cover is palette colour 0 (opaque) or transparent; an indexed pixel equal to
//         the first frame's transparent index has alpha 0 but KEEPS its palette colour (rgb is not zeroed);
//       * from the second frame on the canvas is RGBA (if the first frame had a transparent index) or RGB (opaque)
//         and a pixel equal to the current frame's transparent index leaves the canvas unchanged;
//       * the disposal method is STICKY: a frame whose graphic control extension says 0, or that has none, keeps
//         the last non-zero method seen; 4..7 act like 3;
//       * disposal 2 fills the frame rectangle with the frame's transparent index colour (alpha 0) if it has one,
//         else with the logical screen background colour (opaque; index 0 when there is no global table);
//         disposal 3 restores the rectangle, except on the FIRST frame, which is left in place unless it has a
//         transparent index (then the rectangle becomes transparent);
//       * a colour table that is the identity grayscale ramp is dropped by Pillow (mode "L"): an index is then its
//         own gray level, even beyond the table; a palette index beyond a real table is black.
//     Not reproduced (Pillow grows the canvas, ours is fixed): frames that extend beyond the logical screen are
//     clipped. A first frame with an identity palette followed by a frame with a real palette is erratic in Pillow
//     (it errors or returns black); here every frame uses its own palette.
//   Web: what browsers show. The canvas starts transparent (0, 0, 0, 0); disposal 2 clears the rectangle to
//     transparent (the background colour index is ignored); a pixel equal to the transparent index leaves the
//     canvas unchanged, every other pixel is opaque; disposal 3 restores the rectangle (also on the first frame,
//     where the canvas was transparent); 4..7 behave like 0; an index beyond the table is opaque black.
// Common to both: frame pixels outside the logical screen are clipped.
//
// Loop conditions are bool locals built with non-short-circuit & / | (a compound `while (a && b)` hung the
// WebGPU device).

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Images.Gif;

/// <summary>Running state of one LZW bit reader over the data sub-blocks.</summary>
internal struct LzwBits
{
    /// <summary>Next file byte to read (a sub-block length byte when <see cref="BlockLeft"/> is 0).</summary>
    public int Pos;
    /// <summary>Payload bytes left in the current sub-block.</summary>
    public int BlockLeft;
    /// <summary>Bit buffer (low bits are the next bits of the stream).</summary>
    public uint Buf;
    /// <summary>Valid bits in <see cref="Buf"/>.</summary>
    public int Cnt;
    /// <summary>1 once the block terminator was reached with no more data.</summary>
    public int Err;
}

/// <summary>Loads and launches the GIF decode kernels.</summary>
public sealed class GifKernels : IDisposable
{
    /// <summary>Ints of scratch one LZW thread needs (the 4096-entry code table).</summary>
    public const int LzwScratchInts = 4096;
    /// <summary>Ints per LZW plan entry: frame, index offset, status, pixels produced.</summary>
    public const int PlanStride = 4;
    /// <summary>Ints of resumable decoder state one LZW thread keeps between launches (see <c>Ls*</c>).</summary>
    public const int LzwStateInts = 12;

    /// <summary>
    /// Work units one LZW launch may spend per frame: 1 per code plus 1 per pixel written. A launch stops at the first
    /// code boundary past the budget, so it does at most LzwBudget + 4096 units (a code stands for at most 4096 pixels).
    /// One unit is a dependent load and a byte store on one GPU thread: 12k units are a few ms on a desktop GPU.
    /// </summary>
    public const int LzwBudget = 8192;

    /// <summary>
    /// Steps one scan launch may walk (a step reads one block introducer or one data sub-block length byte and
    /// consumes at least one file byte, so the walk of a file takes at most fileLength + 1 steps).
    /// </summary>
    public const int ScanChunk = 4096;

    // ---- LZW state fields (relative to thread * LzwStateInts) ----
    private const int LsWidth = 0, LsNext = 1, LsPrev = 2, LsFirstChar = 3, LsProduced = 4, LsStatus = 5,
        LsPos = 6, LsBlockLeft = 7, LsBuf = 8, LsCnt = 9, LsErr = 10;

    // ---- scan walk phases ----
    private const int PhaseBlock = 0, PhaseExtension = 1, PhaseImage = 2;

    /// <summary>LZW status: decoded all pixels.</summary>
    public const int LzwOk = 0;
    /// <summary>LZW status: a code was not in the table.</summary>
    public const int LzwBadCode = 1;
    /// <summary>LZW status: the end code arrived before all pixels were produced.</summary>
    public const int LzwEarlyEnd = 2;
    /// <summary>LZW status: the data sub-blocks ended before all pixels were produced.</summary>
    public const int LzwDataEnded = 3;

    /// <summary>Ints per compositing step.</summary>
    public const int StepStride = 10;

    // ---- step fields (relative to step * StepStride) ----
    /// <summary>Frame whose rectangle, palette and interlacing the step uses.</summary>
    public const int StepFrame = 0;
    /// <summary>One of the <c>Op*</c> constants.</summary>
    public const int StepOp = 1;
    /// <summary>OpDraw: where the frame's palette indices start in the index buffer.</summary>
    public const int StepIndexOffset = 2;
    /// <summary>OpDraw: an index equal to this leaves the canvas pixel alone (-1 = none).</summary>
    public const int StepSkip = 3;
    /// <summary>OpDraw: an index equal to this is written with alpha 0 (-1 = none).</summary>
    public const int StepAlphaZero = 4;
    /// <summary>OpFill*: the palette index whose colour fills the area.</summary>
    public const int StepFillIdx = 5;
    /// <summary>OpFill*: one of the <c>Fill*</c> rules.</summary>
    public const int StepFillRule = 6;
    /// <summary>OpFill*: 1 = the fill has alpha 0 (its colour is kept), else alpha 255.</summary>
    public const int StepFillAlphaZero = 7;
    /// <summary>1 = the canvas is opaque RGB: every written pixel gets alpha 255.</summary>
    public const int StepOpaque = 8;
    /// <summary>1 = a frame whose colour table is the identity ramp maps an index to its own gray level.</summary>
    public const int StepGrayRule = 9;

    /// <summary>Op: fill the frame rectangle (clipped to the canvas).</summary>
    public const int OpFillRect = 0;
    /// <summary>Op: copy the frame rectangle of the canvas to the saved copy.</summary>
    public const int OpSave = 1;
    /// <summary>Op: copy the frame rectangle of the saved copy back to the canvas.</summary>
    public const int OpRestore = 2;
    /// <summary>Op: draw the frame.</summary>
    public const int OpDraw = 3;
    /// <summary>Op: fill the whole canvas.</summary>
    public const int OpFillCanvas = 4;

    /// <summary>Fill rule: the index is looked up in the frame's table; beyond the table is black.</summary>
    public const int FillThroughPalette = 0;
    /// <summary>Fill rule: like <see cref="FillThroughPalette"/> but an index beyond the table means entry 0 (Pillow's _rgb).</summary>
    public const int FillEntryZeroBeyond = 1;
    /// <summary>Fill rule: transparent black (0, 0, 0, 0).</summary>
    public const int FillTransparent = 2;

    private readonly Action<Index1D, ArrayView<byte>, int, ArrayView<int>> _scanInit;
    private readonly Action<Index1D, ArrayView<byte>, int, ArrayView<int>> _scanStep;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>> _identity;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>> _lzwInit;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<byte>, ArrayView<int>, ArrayView<int>> _lzwStep;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int> _composite;

    /// <summary>Compile the kernels.</summary>
    public GifKernels(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _scanInit = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, int, ArrayView<int>>(ScanInitKernel);
        _scanStep = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, int, ArrayView<int>>(ScanStepKernel);
        _identity = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>>(IdentityKernel);
        _lzwInit = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(LzwInitKernel);
        _lzwStep = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<byte>, ArrayView<int>, ArrayView<int>>(LzwStepKernel);
        _composite = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, int>(CompositeKernel);
    }

    /// <summary>Launches of <see cref="ScanStep"/> that always finish the walk of a <paramref name="fileLength"/> byte file.</summary>
    public static long ScanStepBound(int fileLength) => ((long)fileLength + 1 + ScanChunk - 1) / ScanChunk;

    /// <summary>
    /// Start the block-structure scan: header, logical screen, global colour table, and the walk state in the meta
    /// header. Then <see cref="ScanStep"/> until <c>meta[GifMeta.ScanDone]</c> is 1 (at most <see cref="ScanStepBound"/>
    /// launches), then <see cref="Identity"/> over the frames found.
    /// </summary>
    public void ScanInit(ArrayView<byte> file, int fileLength, ArrayView<int> meta)
    {
        _scanInit(1, file, fileLength, meta);
    }

    /// <summary>Walk at most <see cref="ScanChunk"/> more steps of the block structure (nothing once the walk is done).</summary>
    public void ScanStep(ArrayView<byte> file, int fileLength, ArrayView<int> meta)
    {
        _scanStep(1, file, fileLength, meta);
    }

    /// <summary>Fill <see cref="GifMeta.FramePaletteIdentity"/> of the first <paramref name="frameCount"/> frames.</summary>
    public void Identity(ArrayView<byte> file, ArrayView<int> meta, int frameCount)
    {
        if (frameCount > 0) _identity(frameCount, file, meta);
    }

    /// <summary>
    /// Launches of <see cref="LzwStep"/> that always finish a frame of <paramref name="pixels"/> pixels whose LZW data
    /// has <paramref name="dataBytes"/> payload bytes and minimum code size <paramref name="minCodeSize"/>: every code
    /// is at least minCodeSize + 1 bits, every launch but the last spends the whole <see cref="LzwBudget"/>.
    /// </summary>
    public static long LzwStepBound(long pixels, long dataBytes, int minCodeSize)
    {
        long codes = dataBytes * 8 / (minCodeSize + 1) + 1;
        return (codes + pixels + 4096) / LzwBudget + 1;
    }

    /// <summary>
    /// Start decoding <paramref name="frameCount"/> frames' LZW data in parallel. <paramref name="plan"/> holds
    /// <see cref="PlanStride"/> ints per frame: frame index, offset of the frame's first index byte in the index
    /// buffer; <paramref name="state"/> holds <see cref="LzwStateInts"/> ints per frame. Then <see cref="LzwStep"/>
    /// the number of times <see cref="LzwStepBound"/> gives for the largest frame; the steps write the status and the
    /// pixels produced into the plan.
    /// </summary>
    public void LzwInit(ArrayView<int> meta, ArrayView<int> plan, ArrayView<int> state, int frameCount)
    {
        _lzwInit(frameCount, meta, plan, state);
    }

    /// <summary>Decode the next codes (about <see cref="LzwBudget"/> work units) of each of <paramref name="frameCount"/> frames.</summary>
    public void LzwStep(ArrayView<byte> file, ArrayView<int> meta, ArrayView<int> plan, ArrayView<byte> indices,
        ArrayView<int> scratch, ArrayView<int> state, int frameCount)
    {
        _lzwStep(frameCount, file, meta, plan, indices, scratch, state);
    }

    /// <summary>
    /// Run compositing step <paramref name="step"/> of <paramref name="steps"/> (<see cref="StepStride"/> ints each).
    /// <paramref name="threads"/> = the step's rectangle clipped to the canvas (width x height), or the whole canvas
    /// for <see cref="OpFillCanvas"/>.
    /// </summary>
    public void Composite(ArrayView<byte> file, ArrayView<int> meta, ArrayView<int> steps, ArrayView<byte> indices,
        ArrayView<byte> canvas, ArrayView<byte> saved, int step, int threads)
    {
        _composite(threads, file, meta, steps, indices, canvas, saved, step);
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private static int Le16(ArrayView<byte> f, int p) => f[p] | (f[p + 1] << 8);

    /// <summary>1 when entry i of the <paramref name="count"/>-entry table at <paramref name="off"/> is (i, i, i) for every i.</summary>
    private static int IsIdentityPalette(ArrayView<byte> file, int off, int count)
    {
        int bad = 0;
        for (int i = 0; i < count; i++)
        {
            int e = off + i * 3;
            bad |= file[e] ^ i;
            bad |= file[e + 1] ^ i;
            bad |= file[e + 2] ^ i;
        }
        return bad == 0 ? 1 : 0;
    }

    private static void SetFrameFlag(ArrayView<int> meta, int fe, int bit)
    {
        meta[fe + GifMeta.FrameFlags] = meta[fe + GifMeta.FrameFlags] | bit;
    }

    // ------------------------------------------------------------------
    // 1. Scan
    // ------------------------------------------------------------------

    /// <summary>The block walk state carried between scan launches (kept in the meta header).</summary>
    private struct ScanState
    {
        public int Phase, Pos, ChainPos, Payload, HasGce, GceDisposal, GceTransFlag, GceTransIndex, GceDelay, Frames, Err, Done;
    }

    private static ScanState LoadScan(ArrayView<int> meta)
    {
        return new ScanState
        {
            Phase = meta[GifMeta.ScanPhase],
            Pos = meta[GifMeta.ScanPos],
            ChainPos = meta[GifMeta.ScanChainPos],
            Payload = meta[GifMeta.ScanPayload],
            HasGce = meta[GifMeta.ScanHasGce],
            GceDisposal = meta[GifMeta.ScanGceDisposal],
            GceTransFlag = meta[GifMeta.ScanGceTransFlag],
            GceTransIndex = meta[GifMeta.ScanGceTransIndex],
            GceDelay = meta[GifMeta.ScanGceDelay],
            Frames = meta[GifMeta.ScanFrames],
            Err = meta[GifMeta.ScanErr],
            Done = meta[GifMeta.ScanDone],
        };
    }

    private static void StoreScan(ArrayView<int> meta, ref ScanState s)
    {
        meta[GifMeta.ScanPhase] = s.Phase;
        meta[GifMeta.ScanPos] = s.Pos;
        meta[GifMeta.ScanChainPos] = s.ChainPos;
        meta[GifMeta.ScanPayload] = s.Payload;
        meta[GifMeta.ScanHasGce] = s.HasGce;
        meta[GifMeta.ScanGceDisposal] = s.GceDisposal;
        meta[GifMeta.ScanGceTransFlag] = s.GceTransFlag;
        meta[GifMeta.ScanGceTransIndex] = s.GceTransIndex;
        meta[GifMeta.ScanGceDelay] = s.GceDelay;
        meta[GifMeta.ScanFrames] = s.Frames;
        meta[GifMeta.ScanErr] = s.Err;
        meta[GifMeta.ScanDone] = s.Done;
        // The published results follow the walk (final once Done is 1).
        meta[GifMeta.FrameCount] = s.Frames;
        meta[GifMeta.Error] = s.Err;
    }

    private static void ScanFail(ArrayView<int> meta, ref ScanState s, GifError error)
    {
        s.Err = (int)error;
        meta[GifMeta.ErrorOffset] = s.Pos;
        s.Done = 1;
    }

    private static void ScanInitKernel(Index1D _, ArrayView<byte> file, int fileLen, ArrayView<int> meta)
    {
        for (int i = 0; i < GifMeta.Size; i++) meta[i] = 0;
        meta[GifMeta.GlobalPaletteOffset] = -1;
        meta[GifMeta.LoopCount] = -1;
        meta[GifMeta.FileLength] = fileLen;

        int err = 0;
        if (fileLen < 6)
        {
            err = (int)GifError.BadSignature;
        }
        else
        {
            int bad = file[0] ^ 0x47;
            bad |= file[1] ^ 0x49;
            bad |= file[2] ^ 0x46;
            bad |= file[3] ^ 0x38;
            bad |= file[5] ^ 0x61;
            bool v = (file[4] == 0x37) | (file[4] == 0x39);
            if ((bad != 0) | !v) err = (int)GifError.BadSignature;
        }
        if ((err == 0) & (fileLen < 13)) err = (int)GifError.Truncated;

        int pos = 13;
        if (err == 0)
        {
            meta[GifMeta.Width] = Le16(file, 6);
            meta[GifMeta.Height] = Le16(file, 8);
            int packed = file[10];
            meta[GifMeta.BackgroundIndex] = file[11];
            if ((packed & 0x80) != 0)
            {
                int count = 2 << (packed & 7);
                if (pos + 3 * count > fileLen)
                {
                    err = (int)GifError.Truncated;
                    meta[GifMeta.ErrorOffset] = pos;
                }
                else
                {
                    meta[GifMeta.GlobalPaletteOffset] = pos;
                    meta[GifMeta.GlobalPaletteCount] = count;
                    pos += 3 * count;
                }
            }
        }
        var s = new ScanState { Phase = PhaseBlock, Pos = pos, Err = err, Done = err != 0 ? 1 : 0 };
        StoreScan(meta, ref s);
    }

    /// <summary>The extension at s.Pos is complete (its sub-blocks are inside the file): apply what it says.</summary>
    private static void FinishExtension(ArrayView<byte> file, int fileLen, ArrayView<int> meta, ref ScanState s)
    {
        int label = file[s.Pos + 1];
        int first = s.Pos + 2;     // first sub-block length byte
        int len0 = file[first];
        if (label == 0xF9)
        {
            // GCE: packed, delay (LE16), transparent index. Two GCEs before one image (never written by an encoder)
            // merge the way Pillow merges them: a later "unspecified" disposal never overrides a specified one, and a
            // later GCE cannot clear the transparent index an earlier one set.
            if (len0 >= 4)
            {
                int gp = file[first + 1];
                s.HasGce = 1;
                int dbits = (gp >> 2) & 7;
                if (dbits != 0) s.GceDisposal = dbits;
                if ((gp & 1) != 0)
                {
                    s.GceTransFlag = 1;
                    s.GceTransIndex = file[first + 4];
                }
                s.GceDelay = Le16(file, first + 2);
            }
        }
        else if (label == 0xFF)
        {
            // Application extension "NETSCAPE2.0" (or "ANIMEXTS1.0") before the first image carries the loop count
            // in the next sub-block: [3, 1, count LE16].
            if ((len0 >= 11) & (s.Frames == 0))
            {
                bool netscape = (file[first + 1] == 0x4E) & (file[first + 2] == 0x45) & (file[first + 3] == 0x54)
                              & (file[first + 4] == 0x53) & (file[first + 5] == 0x43) & (file[first + 6] == 0x41)
                              & (file[first + 7] == 0x50) & (file[first + 8] == 0x45) & (file[first + 9] == 0x32)
                              & (file[first + 10] == 0x2E) & (file[first + 11] == 0x30);
                bool animexts = (file[first + 1] == 0x41) & (file[first + 2] == 0x4E) & (file[first + 3] == 0x49)
                              & (file[first + 4] == 0x4D) & (file[first + 5] == 0x45) & (file[first + 6] == 0x58)
                              & (file[first + 7] == 0x54) & (file[first + 8] == 0x53) & (file[first + 9] == 0x31)
                              & (file[first + 10] == 0x2E) & (file[first + 11] == 0x30);
                int second = first + 1 + len0;
                bool room = second + 4 <= fileLen;
                if ((netscape | animexts) & room)
                {
                    bool isLoop = (file[second] >= 3) & (file[second + 1] == 1);
                    if (isLoop) meta[GifMeta.LoopCount] = Le16(file, second + 2);
                }
            }
        }
        s.Pos = s.ChainPos;
        s.Phase = PhaseBlock;
    }

    /// <summary>The image at s.Pos is complete (its data sub-blocks are inside the file): record it in the frame table.</summary>
    private static void FinishImage(ArrayView<byte> file, ArrayView<int> meta, ref ScanState s)
    {
        if (s.Frames >= GifMeta.MaxFrames)
        {
            ScanFail(meta, ref s, GifError.TooManyFrames);
        }
        else
        {
            int pos = s.Pos;
            int fpacked = file[pos + 9];
            int p = pos + 10;
            int palOff = meta[GifMeta.GlobalPaletteOffset];
            int palCount = meta[GifMeta.GlobalPaletteCount];
            if ((fpacked & 0x80) != 0)
            {
                palOff = p;
                palCount = 2 << (fpacked & 7);
                p += 3 * palCount;
            }
            int fe = GifMeta.FrameTable + s.Frames * GifMeta.FrameStride;
            meta[fe + GifMeta.FrameX] = Le16(file, pos + 1);
            meta[fe + GifMeta.FrameY] = Le16(file, pos + 3);
            meta[fe + GifMeta.FrameW] = Le16(file, pos + 5);
            meta[fe + GifMeta.FrameH] = Le16(file, pos + 7);
            meta[fe + GifMeta.FrameFlags] = 0;
            if ((fpacked & 0x40) != 0) SetFrameFlag(meta, fe, 1);
            if ((s.HasGce != 0) & (s.GceTransFlag != 0)) SetFrameFlag(meta, fe, 2);
            meta[fe + GifMeta.FrameTransparentIndex] = s.GceTransIndex;
            meta[fe + GifMeta.FrameDelayCs] = s.GceDelay;
            meta[fe + GifMeta.FrameDisposal] = s.GceDisposal;
            meta[fe + GifMeta.FramePaletteOffset] = palOff;
            meta[fe + GifMeta.FramePaletteCount] = palCount;
            meta[fe + GifMeta.FrameMinCodeSize] = file[p];
            meta[fe + GifMeta.FrameDataOffset] = p + 1;
            meta[fe + GifMeta.FramePaletteIdentity] = 0;     // IdentityKernel fills it once the walk is done
            meta[fe + GifMeta.FrameDataBytes] = s.Payload;
            s.Frames++;
            s.HasGce = 0;
            s.GceDisposal = 0;
            s.GceTransFlag = 0;
            s.GceTransIndex = 0;
            s.GceDelay = 0;
            s.Pos = s.ChainPos;
            s.Phase = PhaseBlock;
        }
    }

    /// <summary>
    /// At most <see cref="ScanChunk"/> steps of the block walk. A step is either a block introducer (the trailer, an
    /// extension or an image descriptor + its colour table) or one data sub-block hop of the block being walked; a
    /// block's effect is applied when its terminating sub-block is reached, so a truncated chain records nothing.
    /// </summary>
    private static void ScanStepKernel(Index1D _, ArrayView<byte> file, int fileLen, ArrayView<int> meta)
    {
        var s = LoadScan(meta);
        int steps = 0;
        bool go = s.Done == 0;
        while (go)
        {
            if (s.Phase == PhaseBlock)
            {
                if (s.Pos >= fileLen)
                {
                    // A missing trailer is tolerated once at least one frame is complete.
                    if (s.Frames == 0) ScanFail(meta, ref s, GifError.Truncated);
                    s.Done = 1;
                }
                else
                {
                    int b = file[s.Pos];
                    if (b == 0x3B)
                    {
                        s.Done = 1;
                    }
                    else if (b == 0x21)
                    {
                        if (s.Pos + 2 > fileLen)
                        {
                            ScanFail(meta, ref s, GifError.Truncated);
                        }
                        else
                        {
                            s.ChainPos = s.Pos + 2;
                            s.Phase = PhaseExtension;
                        }
                    }
                    else if (b == 0x2C)
                    {
                        if (s.Pos + 10 > fileLen)
                        {
                            ScanFail(meta, ref s, GifError.Truncated);
                        }
                        else
                        {
                            int fpacked = file[s.Pos + 9];
                            int p = s.Pos + 10;
                            bool ok = true;
                            if ((fpacked & 0x80) != 0)
                            {
                                int count = 2 << (fpacked & 7);
                                if (p + 3 * count > fileLen) ok = false;
                                else p += 3 * count;
                            }
                            // LZW minimum code size byte, then the data sub-blocks.
                            if (ok & (p < fileLen))
                            {
                                s.ChainPos = p + 1;
                                s.Payload = 0;
                                s.Phase = PhaseImage;
                            }
                            else
                            {
                                ScanFail(meta, ref s, GifError.Truncated);
                            }
                        }
                    }
                    else
                    {
                        ScanFail(meta, ref s, GifError.BadBlock);
                    }
                }
            }
            else
            {
                // One sub-block of the extension / image data at s.Pos.
                if (s.ChainPos >= fileLen)
                {
                    ScanFail(meta, ref s, GifError.Truncated);
                }
                else
                {
                    int len = file[s.ChainPos];
                    s.ChainPos += 1 + len;
                    if (s.ChainPos > fileLen)
                    {
                        ScanFail(meta, ref s, GifError.Truncated);
                    }
                    else if (len != 0)
                    {
                        s.Payload += len;
                    }
                    else if (s.Phase == PhaseExtension)
                    {
                        FinishExtension(file, fileLen, meta, ref s);
                    }
                    else
                    {
                        FinishImage(file, meta, ref s);
                    }
                }
            }
            steps++;
            go = (s.Done == 0) & (steps < ScanChunk);
        }
        StoreScan(meta, ref s);
    }

    /// <summary>Frame <c>index</c>: is its colour table the identity gray ramp (at most 256 entries).</summary>
    private static void IdentityKernel(Index1D index, ArrayView<byte> file, ArrayView<int> meta)
    {
        int fe = GifMeta.FrameTable + index.X * GifMeta.FrameStride;
        int palOff = meta[fe + GifMeta.FramePaletteOffset];
        int identity = 0;
        if (palOff >= 0) identity = IsIdentityPalette(file, palOff, meta[fe + GifMeta.FramePaletteCount]);
        meta[fe + GifMeta.FramePaletteIdentity] = identity;
    }

    // ------------------------------------------------------------------
    // 2. LZW
    // ------------------------------------------------------------------

    /// <summary>Next <paramref name="width"/>-bit code (LSB first), or -1 when the sub-blocks ran out.</summary>
    private static int NextCode(ref LzwBits st, ArrayView<byte> file, int width)
    {
        bool need = (st.Cnt < width) & (st.Err == 0);
        while (need)
        {
            if (st.BlockLeft == 0)
            {
                int len = file[st.Pos];
                st.Pos++;
                st.BlockLeft = len;
                if (len == 0) st.Err = 1;
            }
            if (st.Err == 0)
            {
                st.Buf |= (uint)file[st.Pos] << st.Cnt;
                st.Pos++;
                st.BlockLeft--;
                st.Cnt += 8;
            }
            need = (st.Cnt < width) & (st.Err == 0);
        }
        int code = -1;
        if (st.Err == 0)
        {
            code = (int)(st.Buf & ((1u << width) - 1u));
            st.Buf >>= width;
            st.Cnt -= width;
        }
        return code;
    }

    private static void LzwInitKernel(Index1D index, ArrayView<int> meta, ArrayView<int> plan, ArrayView<int> state)
    {
        int t = index.X;
        int frame = plan[t * PlanStride];
        int fe = GifMeta.FrameTable + frame * GifMeta.FrameStride;
        int minCode = meta[fe + GifMeta.FrameMinCodeSize];
        int clear = 1 << minCode;
        int sb = t * LzwStateInts;
        state[sb + LsWidth] = minCode + 1;
        state[sb + LsNext] = clear + 2;
        state[sb + LsPrev] = -1;
        state[sb + LsFirstChar] = 0;
        state[sb + LsProduced] = 0;
        state[sb + LsStatus] = LzwOk;
        state[sb + LsPos] = meta[fe + GifMeta.FrameDataOffset];
        state[sb + LsBlockLeft] = 0;
        state[sb + LsBuf] = 0;
        state[sb + LsCnt] = 0;
        state[sb + LsErr] = 0;
        plan[t * PlanStride + 2] = LzwOk;
        plan[t * PlanStride + 3] = 0;
    }

    /// <summary>
    /// The next codes of frame <c>plan[index]</c>: decodes until <see cref="LzwBudget"/> work units are spent (or the
    /// frame is complete / failed), then stores the state for the next launch. A launch after the end does nothing.
    /// </summary>
    private static void LzwStepKernel(
        Index1D index, ArrayView<byte> file, ArrayView<int> meta, ArrayView<int> plan,
        ArrayView<byte> indices, ArrayView<int> scratch, ArrayView<int> state)
    {
        int t = index.X;
        int frame = plan[t * PlanStride];
        int outBase = plan[t * PlanStride + 1];
        int fe = GifMeta.FrameTable + frame * GifMeta.FrameStride;
        int total = meta[fe + GifMeta.FrameW] * meta[fe + GifMeta.FrameH];
        int minCode = meta[fe + GifMeta.FrameMinCodeSize];
        int sb = t * LzwScratchInts;
        int ss = t * LzwStateInts;

        int clear = 1 << minCode;
        int eoi = clear + 1;
        int width = state[ss + LsWidth];
        int next = state[ss + LsNext];
        int prev = state[ss + LsPrev];
        int firstChar = state[ss + LsFirstChar];
        int produced = state[ss + LsProduced];
        int status = state[ss + LsStatus];
        var st = new LzwBits
        {
            Pos = state[ss + LsPos],
            BlockLeft = state[ss + LsBlockLeft],
            Buf = (uint)state[ss + LsBuf],
            Cnt = state[ss + LsCnt],
            Err = state[ss + LsErr],
        };

        int units = 0;
        bool go = (status == LzwOk) & (produced < total);
        while (go)
        {
            units++;
            int code = NextCode(ref st, file, width);
            if (code < 0)
            {
                status = LzwDataEnded;
            }
            else if (code == clear)
            {
                width = minCode + 1;
                next = clear + 2;
                prev = -1;
            }
            else if (code == eoi)
            {
                status = LzwEarlyEnd;
            }
            else
            {
                bool bad = false;
                if (prev < 0)
                {
                    if (code >= clear) bad = true;
                }
                else if (code > next)
                {
                    bad = true;
                }
                if (bad)
                {
                    status = LzwBadCode;
                }
                else if (prev < 0)
                {
                    // First code after a clear: a literal.
                    indices[outBase + produced] = (byte)code;
                    produced++;
                    units++;
                    firstChar = code;
                    prev = code;
                }
                else
                {
                    int prevLen = 1;
                    if (prev >= clear) prevLen = (int)((uint)scratch[sb + prev] >> 20);
                    // code == next (KwKwK): the new entry is prev + the first character of prev.
                    if (code == next)
                        scratch[sb + next] = prev | (firstChar << 12) | ((prevLen + 1) << 20);

                    // Write string(code), last character first, walking the prefix chain.
                    int len = 1;
                    if (code >= clear) len = (int)((uint)scratch[sb + code] >> 20);
                    units += len;
                    int c = code;
                    int k = len - 1;
                    int ch = 0;
                    bool walk = true;
                    while (walk)
                    {
                        if (c < clear)
                        {
                            ch = c;
                        }
                        else
                        {
                            int e = scratch[sb + c];
                            ch = (e >> 12) & 0xFF;
                            c = e & 0xFFF;
                        }
                        int dst = produced + k;
                        if (dst < total) indices[outBase + dst] = (byte)ch;
                        k--;
                        walk = k >= 0;
                    }
                    int room = total - produced;
                    produced += len < room ? len : room;

                    if (next < 4096)
                    {
                        if (code != next)
                            scratch[sb + next] = prev | (ch << 12) | ((prevLen + 1) << 20);
                        next++;
                        if ((next == (1 << width)) & (width < 12)) width++;
                    }
                    firstChar = ch;
                    prev = code;
                }
            }
            go = (status == LzwOk) & (produced < total) & (units < LzwBudget);
        }
        state[ss + LsWidth] = width;
        state[ss + LsNext] = next;
        state[ss + LsPrev] = prev;
        state[ss + LsFirstChar] = firstChar;
        state[ss + LsProduced] = produced;
        state[ss + LsStatus] = status;
        state[ss + LsPos] = st.Pos;
        state[ss + LsBlockLeft] = st.BlockLeft;
        state[ss + LsBuf] = (int)st.Buf;
        state[ss + LsCnt] = st.Cnt;
        state[ss + LsErr] = st.Err;
        plan[t * PlanStride + 2] = status;
        plan[t * PlanStride + 3] = produced;
    }

    // ------------------------------------------------------------------
    // 3. Compositing
    // ------------------------------------------------------------------

    /// <summary>Source row (in the stored, interlaced order) of output row <paramref name="y"/> of an <paramref name="h"/> row frame.</summary>
    private static int InterlacedSourceRow(int y, int h)
    {
        int n1 = (h + 7) >> 3;
        int n2 = 0;
        if (h > 4) n2 = (h - 4 + 7) >> 3;
        int n3 = 0;
        if (h > 2) n3 = (h - 2 + 3) >> 2;
        int m = y & 7;
        int r;
        if (m == 0) r = y >> 3;
        else if (m == 4) r = n1 + ((y - 4) >> 3);
        else if ((y & 3) == 2) r = n1 + n2 + ((y - 2) >> 2);
        else r = n1 + n2 + n3 + ((y - 1) >> 1);
        return r;
    }

    /// <summary>
    /// Colour (0xRRGGBB) of palette index <paramref name="idx"/> in the table of frame entry <paramref name="fe"/>.
    /// With <paramref name="gray"/> set, an identity-ramp table maps an index to its own gray level (even beyond the
    /// table); otherwise an index beyond the table is black, or entry 0 when <paramref name="beyondIsEntryZero"/>.
    /// </summary>
    private static int PaletteColor(ArrayView<byte> file, ArrayView<int> meta, int fe, int idx, int gray, bool beyondIsEntryZero)
    {
        int count = meta[fe + GifMeta.FramePaletteCount];
        int color = 0;
        bool identity = (gray != 0) & (meta[fe + GifMeta.FramePaletteIdentity] != 0);
        if (identity)
        {
            color = idx * 0x010101;
        }
        else
        {
            int e = idx;
            if ((idx >= count) & beyondIsEntryZero) e = 0;
            if (e < count)
            {
                int o = meta[fe + GifMeta.FramePaletteOffset] + e * 3;
                color = (file[o] << 16) | (file[o + 1] << 8) | file[o + 2];
            }
        }
        return color;
    }

    private static void CompositeKernel(
        Index1D index, ArrayView<byte> file, ArrayView<int> meta, ArrayView<int> steps, ArrayView<byte> indices,
        ArrayView<byte> canvas, ArrayView<byte> saved, int step)
    {
        int sb = step * StepStride;
        int op = steps[sb + StepOp];
        int fe = GifMeta.FrameTable + steps[sb + StepFrame] * GifMeta.FrameStride;
        int fw = meta[fe + GifMeta.FrameW];
        int fh = meta[fe + GifMeta.FrameH];
        int cw = meta[GifMeta.Width];

        int i = index.X;
        int fy = 0;
        int fx = 0;
        int p = i * 4;
        if (op != OpFillCanvas)
        {
            // The dispatch covers the frame rectangle clipped to the canvas.
            int x0 = meta[fe + GifMeta.FrameX];
            int y0 = meta[fe + GifMeta.FrameY];
            int clipW = fw;
            if (x0 + clipW > cw) clipW = cw - x0;
            fy = i / clipW;
            fx = i - fy * clipW;
            p = ((y0 + fy) * cw + x0 + fx) * 4;
        }

        if (op == OpSave)
        {
            saved[p] = canvas[p]; saved[p + 1] = canvas[p + 1]; saved[p + 2] = canvas[p + 2]; saved[p + 3] = canvas[p + 3];
        }
        else if (op == OpRestore)
        {
            canvas[p] = saved[p]; canvas[p + 1] = saved[p + 1]; canvas[p + 2] = saved[p + 2]; canvas[p + 3] = saved[p + 3];
        }
        else if (op == OpDraw)
        {
            int row = fy;
            if ((meta[fe + GifMeta.FrameFlags] & 1) != 0) row = InterlacedSourceRow(fy, fh);
            int idx = indices[steps[sb + StepIndexOffset] + row * fw + fx];
            if (idx != steps[sb + StepSkip])
            {
                int color = PaletteColor(file, meta, fe, idx, steps[sb + StepGrayRule], false);
                int a = 255;
                if ((idx == steps[sb + StepAlphaZero]) & (steps[sb + StepOpaque] == 0)) a = 0;
                canvas[p] = (byte)(color >> 16);
                canvas[p + 1] = (byte)(color >> 8);
                canvas[p + 2] = (byte)color;
                canvas[p + 3] = (byte)a;
            }
        }
        else
        {
            // OpFillRect / OpFillCanvas
            int rule = steps[sb + StepFillRule];
            int color = 0;
            int a = 0;
            if (rule != FillTransparent)
            {
                color = PaletteColor(file, meta, fe, steps[sb + StepFillIdx], steps[sb + StepGrayRule], rule == FillEntryZeroBeyond);
                a = 255;
                if ((steps[sb + StepFillAlphaZero] != 0) & (steps[sb + StepOpaque] == 0)) a = 0;
            }
            canvas[p] = (byte)(color >> 16);
            canvas[p + 1] = (byte)(color >> 8);
            canvas[p + 2] = (byte)color;
            canvas[p + 3] = (byte)a;
        }
    }

    /// <summary>No-op (kernels are owned by the accelerator).</summary>
    public void Dispose() { }
}
