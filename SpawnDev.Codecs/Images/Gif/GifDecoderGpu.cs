// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GIF decoder (87a / 89a, animated), 100% accelerator-resident. The host is a pure coordinator: it uploads the
// file (necessary I/O), dispatches the GifKernels chain, and reads back the block-structure scan (header ints and
// 16 ints per frame) plus one status int per frame. The LZW data, the palette indices and the RGBA canvas never
// touch the host unless the caller asks for managed output.
//
// API shape (same as the PNG decoder, plus the animation API of Plans/PLAN-Image-Codecs.md): three overloads per
// entry point, all funnelling into the ArrayView1D one.
//   - ArrayView1D : file already on the accelerator, RGBA pixels written into a caller-owned view.
//   - TypedArray  : browser. JS bytes in via CopyFromJS, JS Uint8Array pixels out; nothing enters the .NET heap.
//   - ReadOnlyMemory<byte> : managed. Desktop and small inputs; pixels come back as byte[].
//
// Animation: ReadFramesAsync lists the frames as stored (rectangle, DurationMs = delay x 10, Dispose, Blend = Over
// for a frame with a transparent colour index, else Source). DecodeFrameAsync(k) returns the FULL canvas (logical
// screen size) as it looks while frame k is displayed; DecodeAsync is DecodeFrameAsync(0). Frames 0..k are
// LZW-decoded (in parallel, in batches that bound memory) and composited on the accelerator; random access
// therefore costs O(k) per call. ReadLoopCountAsync returns the NETSCAPE loop count.
//
// No GPU launch grows with the input: the block scan and the LZW decode are chains of bounded, resumable launches
// (GifKernels.ScanChunk steps / GifKernels.LzwBudget work units each); the host launches an upper bound of steps
// computed from the file length / the frame table (the scan checks a done flag every ScanBatch launches).
//
// Compositing: GifDecoderGpu.Compositing selects the rules (default GifCompositing.Pillow = bit-exact with
// Pillow 13, GifCompositing.Web = what browsers show). The host turns the frame table into a short list of
// compositing STEPS (fill, save, restore, draw: a few ints each) and dispatches them; the rules themselves are in
// GifKernels' header comment. Planning is arithmetic on the frame table (a handful of ints per frame), never on
// pixels.
//
// Errors (ImageDecodeException, specific reason): bad signature, truncation anywhere in the block structure
// (a missing trailer after a complete frame is tolerated), unknown block introducer, zero-size screen or frame,
// frame without a colour table, invalid LZW minimum code size, a frame larger than its LZW data can describe, and
// per frame: invalid LZW code, LZW data that ends early (end code or data exhausted). Structural errors surface
// from the scan, so they are detected for the whole file; LZW errors surface when the frame is decoded
// (DecodeFrameAsync(k) checks frames 0..k).

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Images.Gif;

/// <summary>How <see cref="GifDecoderGpu"/> combines frames on the canvas.</summary>
public enum GifCompositing
{
    /// <summary>
    /// Bit-exact with Pillow 13 (<c>Image.seek(k)</c> then <c>convert("RGBA")</c>, default loading strategy):
    /// sticky disposal method, disposal 2 fills with the transparent colour or the screen background colour,
    /// a pixel with the first frame's transparent index keeps its palette colour under alpha 0, an identity-ramp
    /// colour table is read as gray levels. See the header of <c>GifKernels.cs</c> for every rule.
    /// </summary>
    Pillow = 0,
    /// <summary>
    /// What browsers show: transparent initial canvas, disposal 2 clears to transparent, a pixel with the
    /// transparent index leaves the canvas alone, disposal 4..7 leaves the frame in place, undefined colours are black.
    /// </summary>
    Web = 1,
}

/// <summary>GIF decoder running on an ILGPU accelerator.</summary>
/// <remarks>Not thread-safe: one in-flight call per instance (the metadata block and staging buffers are shared).</remarks>
public sealed class GifDecoderGpu : IDisposable
{
    private const int MaxBatchFrames = 64;
    private const long IndexBudgetBytes = 32L << 20;
    /// <summary>Steps one frame can add: dispose of the previous frame, save, draw (+ the initial canvas fill for frame 0).</summary>
    private const int MaxStepsPerBatch = MaxBatchFrames * 3 + 2;
    /// <summary>Scan launches between two checks of the walk's done flag (a small file needs one batch: no extra readback).</summary>
    private const int ScanBatch = 16;

    private readonly Accelerator _accelerator;
    private readonly GifKernels _kernels;
    private readonly ImagePixelKernels _pixel;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _meta;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _plan;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _steps;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _lzwScratch;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _lzwState;
    private readonly StageBuffer<byte> _stageIn;
    private readonly StageBuffer<byte> _stageOut;
    private readonly StageBuffer<byte> _indices;
    private readonly StageBuffer<byte> _canvas;
    private readonly StageBuffer<byte> _saved;

    /// <summary>
    /// The compositing rules for frames after the first and for the canvas around a smaller frame.
    /// Default <see cref="GifCompositing.Pillow"/>.
    /// </summary>
    public GifCompositing Compositing { get; set; } = GifCompositing.Pillow;

    /// <summary>Compile the kernels.</summary>
    public GifDecoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernels = new GifKernels(accelerator);
        _pixel = new ImagePixelKernels(accelerator);
        _meta = accelerator.Allocate1D<int>(GifMeta.BufferInts);
        _plan = accelerator.Allocate1D<int>(MaxBatchFrames * GifKernels.PlanStride);
        _steps = accelerator.Allocate1D<int>(MaxStepsPerBatch * GifKernels.StepStride);
        _lzwScratch = accelerator.Allocate1D<int>(MaxBatchFrames * GifKernels.LzwScratchInts);
        _lzwState = accelerator.Allocate1D<int>(MaxBatchFrames * GifKernels.LzwStateInts);
        _stageIn = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
        _indices = new StageBuffer<byte>(accelerator);
        _canvas = new StageBuffer<byte>(accelerator);
        _saved = new StageBuffer<byte>(accelerator);
    }

    // ------------------------------------------------------------------
    // Layout derived from the scan (pure metadata arithmetic, 64-bit)
    // ------------------------------------------------------------------

    private sealed class Layout
    {
        public int Width, Height, FrameCount, LoopCount;
        /// <summary>Logical screen background colour index when a global colour table exists, else 0 (Pillow's rule).</summary>
        public int BackgroundOrZero;
        /// <summary>Raw frame table: <see cref="GifMeta.FrameStride"/> ints per frame.</summary>
        public int[] Table = System.Array.Empty<int>();
        public ImageFrameInfo[] Frames = System.Array.Empty<ImageFrameInfo>();
        public bool HasTransparentIndex, IsInterlaced;

        public int Field(int frame, int field) => Table[frame * GifMeta.FrameStride + field];
        public bool HasTrans(int frame) => (Field(frame, GifMeta.FrameFlags) & 2) != 0;
    }

    private ImageInfo ToInfo(Layout l)
    {
        bool alpha = l.HasTransparentIndex;
        if (Compositing == GifCompositing.Web)
        {
            // Web: a "restore to background" frame leaves transparent pixels, and so does a first frame that does not cover the screen.
            for (int f = 0; f < l.FrameCount; f++)
                if (l.Frames[f].Dispose == FrameDispose.Background) alpha = true;
            var first = l.Frames[0];
            if (first.X != 0 || first.Y != 0 || first.Width < l.Width || first.Height < l.Height) alpha = true;
        }
        return new ImageInfo(ImageFormat.Gif, l.Width, l.Height, alpha, 8, l.FrameCount, 1, l.IsInterlaced);
    }

    private static string DescribeScanError(int[] head) => (GifError)head[GifMeta.Error] switch
    {
        GifError.BadSignature => "not a GIF file (bad signature)",
        GifError.Truncated => $"file is truncated (ends inside a block that starts at offset {head[GifMeta.ErrorOffset]})",
        GifError.BadBlock => $"invalid block introducer at offset {head[GifMeta.ErrorOffset]}",
        GifError.TooManyFrames => $"more than {GifMeta.MaxFrames} frames",
        var e => e.ToString(),
    };

    private async Task<Layout> ScanAsync(ArrayView1D<byte, Stride1D.Dense> file, int fileLength)
    {
        if (fileLength <= 0 || fileLength > file.Length) throw new ArgumentOutOfRangeException(nameof(fileLength));
        // The walk is a chain of bounded launches (ScanStepBound always finishes it); the done flag is checked
        // after every ScanBatch launches, so a file that needs fewer costs no readback beyond the header's.
        _kernels.ScanInit(file, fileLength, _meta.View);
        long bound = GifKernels.ScanStepBound(fileLength);
        long launched = 0;
        int[] head;
        while (true)
        {
            long batch = Math.Min(ScanBatch, bound - launched);
            for (long i = 0; i < batch; i++) _kernels.ScanStep(file, fileLength, _meta.View);
            launched += batch;
            await _accelerator.SynchronizeAsync();
            head = await _meta.View.SubView(0, GifMeta.Size).CopyToHostAsync();
            if (head[GifMeta.ScanDone] != 0) break;
            if (launched >= bound)
                throw new InvalidOperationException($"GIF block scan did not finish within {bound} launches (decoder bug).");
        }
        if ((GifError)head[GifMeta.Error] != GifError.None)
            throw new ImageDecodeException(ImageFormat.Gif, DescribeScanError(head));

        int count = head[GifMeta.FrameCount];
        var l = new Layout
        {
            Width = head[GifMeta.Width],
            Height = head[GifMeta.Height],
            FrameCount = count,
            LoopCount = head[GifMeta.LoopCount],
            BackgroundOrZero = head[GifMeta.GlobalPaletteOffset] >= 0 ? head[GifMeta.BackgroundIndex] : 0,
        };
        if (l.Width <= 0 || l.Height <= 0)
            throw new ImageDecodeException(ImageFormat.Gif, $"logical screen has zero size ({l.Width}x{l.Height})");
        if ((long)l.Width * l.Height * 4 > int.MaxValue)
            throw new ImageDecodeException(ImageFormat.Gif, $"logical screen {l.Width}x{l.Height} is too large");
        if (count == 0)
            throw new ImageDecodeException(ImageFormat.Gif, "no image data (the file has no frames)");

        _kernels.Identity(file, _meta.View, count);
        await _accelerator.SynchronizeAsync();
        l.Table = await _meta.View.SubView(GifMeta.FrameTable, (long)count * GifMeta.FrameStride).CopyToHostAsync();
        l.Frames = new ImageFrameInfo[count];
        for (int f = 0; f < count; f++)
        {
            int w = l.Field(f, GifMeta.FrameW), h = l.Field(f, GifMeta.FrameH);
            if (w <= 0 || h <= 0)
                throw new ImageDecodeException(ImageFormat.Gif, $"frame {f} has zero size ({w}x{h})");
            if ((long)w * h > int.MaxValue)
                throw new ImageDecodeException(ImageFormat.Gif, $"frame {f} ({w}x{h}) is too large");
            int minCode = l.Field(f, GifMeta.FrameMinCodeSize);
            if (minCode < 2 || minCode > 8)
                throw new ImageDecodeException(ImageFormat.Gif, $"frame {f} has an invalid LZW minimum code size {minCode}");
            if (l.Field(f, GifMeta.FramePaletteOffset) < 0)
                throw new ImageDecodeException(ImageFormat.Gif, $"frame {f} has no colour table (neither local nor global)");
            // A code is at least minCode + 1 bits and stands for at most 4095 pixels: a header that claims more
            // pixels than the data can produce is a lie, and must not size a multi-gigabyte index buffer.
            long maxPixels = (long)l.Field(f, GifMeta.FrameDataBytes) * 8 / (minCode + 1) * 4095;
            if ((long)w * h > maxPixels)
                throw new ImageDecodeException(ImageFormat.Gif,
                    $"frame {f} is {w}x{h} pixels but its {l.Field(f, GifMeta.FrameDataBytes)} bytes of LZW data can describe at most {maxPixels}");

            int flags = l.Field(f, GifMeta.FrameFlags);
            int disposal = l.Field(f, GifMeta.FrameDisposal);
            var dispose = disposal == 2 ? FrameDispose.Background : disposal == 3 ? FrameDispose.Previous : FrameDispose.None;
            var blend = (flags & 2) != 0 ? FrameBlend.Over : FrameBlend.Source;
            l.Frames[f] = new ImageFrameInfo(f, l.Field(f, GifMeta.FrameX), l.Field(f, GifMeta.FrameY), w, h,
                l.Field(f, GifMeta.FrameDelayCs) * 10, dispose, blend);
            if ((flags & 1) != 0) l.IsInterlaced = true;
            if ((flags & 2) != 0) l.HasTransparentIndex = true;
        }
        return l;
    }

    // ------------------------------------------------------------------
    // Compositing plan
    // ------------------------------------------------------------------

    /// <summary>
    /// Turns the frame table into compositing steps, one frame at a time (state carries from frame to frame and
    /// across LZW batches). Pure arithmetic on a few ints per frame.
    /// </summary>
    private sealed class Planner
    {
        // Pillow canvas stages: the first frame lives in an indexed canvas ("P", or "L" for an identity ramp);
        // later frames are composited on an RGBA (first frame had a transparent index) or RGB canvas.
        private const int StageP = 0, StageL = 1, StageRgb = 2, StageRgba = 3;
        private const int PendNone = 0, PendFill = 1, PendRestore = 2;

        private readonly Layout _l;
        private readonly bool _pillow;
        private readonly int _canvasThreads;
        private int _sticky;
        private int _stage;
        private int _infoTrans = -1;
        private int _pend;
        private int _pendIdx, _pendRule, _pendAlphaZero, _pendOpaque;

        public readonly List<int> Steps = new();
        public readonly List<int> Threads = new();

        public Planner(Layout l, bool pillow)
        {
            _l = l;
            _pillow = pillow;
            _canvasThreads = l.Width * l.Height;
        }

        public int StepCount => Threads.Count;

        private void Add(int frame, int op, int indexOffset, int skip, int alphaZero, int fillIdx, int fillRule,
            int fillAlphaZero, int opaque, int gray, int threads)
        {
            if (threads <= 0) return;
            Steps.Add(frame); Steps.Add(op); Steps.Add(indexOffset); Steps.Add(skip); Steps.Add(alphaZero);
            Steps.Add(fillIdx); Steps.Add(fillRule); Steps.Add(fillAlphaZero); Steps.Add(opaque); Steps.Add(gray);
            Threads.Add(threads);
        }

        private int Clipped(int frame) => ClippedThreads(_l.Frames[frame], _l.Width, _l.Height);

        private void EmitPending(int prevFrame)
        {
            if (_pend == PendFill)
                Add(prevFrame, GifKernels.OpFillRect, 0, -1, -1, _pendIdx, _pendRule, _pendAlphaZero, _pendOpaque, _pillow ? 1 : 0, Clipped(prevFrame));
            else if (_pend == PendRestore)
                Add(prevFrame, GifKernels.OpRestore, 0, -1, -1, 0, 0, 0, 0, 0, Clipped(prevFrame));
            _pend = PendNone;
        }

        public void PlanFrame(int f, int indexOffset)
        {
            if (_pillow) PlanPillow(f, indexOffset);
            else PlanWeb(f, indexOffset);
        }

        private void PlanWeb(int f, int indexOffset)
        {
            if (f == 0)
                Add(0, GifKernels.OpFillCanvas, 0, -1, -1, 0, GifKernels.FillTransparent, 0, 0, 0, _canvasThreads);
            else
                EmitPending(f - 1);
            int disposal = _l.Field(f, GifMeta.FrameDisposal);
            int trans = _l.HasTrans(f) ? _l.Field(f, GifMeta.FrameTransparentIndex) : -1;
            int threads = Clipped(f);
            _pend = PendNone;
            if (disposal == 3)
            {
                Add(f, GifKernels.OpSave, 0, -1, -1, 0, 0, 0, 0, 0, threads);
                _pend = PendRestore;
            }
            else if (disposal == 2)
            {
                _pend = PendFill; _pendIdx = 0; _pendRule = GifKernels.FillTransparent; _pendAlphaZero = 0; _pendOpaque = 0;
            }
            Add(f, GifKernels.OpDraw, indexOffset, trans, -1, 0, 0, 0, 0, 0, threads);
        }

        private void PlanPillow(int f, int indexOffset)
        {
            int trans = _l.HasTrans(f) ? _l.Field(f, GifMeta.FrameTransparentIndex) : -1;
            int threads = Clipped(f);

            // Pillow _seek(f): first the previous frame's disposal is applied to the canvas ...
            if (f > 0) EmitPending(f - 1);

            // ... then the canvas stage is set up / converted ...
            if (f == 0)
            {
                _stage = _l.Field(0, GifMeta.FramePaletteIdentity) != 0 ? StageL : StageP;
                _infoTrans = trans;
                // Image.core.fill(mode, size, transparent index or 0): every pixel is that index; it has alpha 0 when it IS the transparent index.
                Add(0, GifKernels.OpFillCanvas, 0, -1, -1, trans >= 0 ? trans : 0, GifKernels.FillThroughPalette,
                    trans >= 0 ? 1 : 0, 0, 1, _canvasThreads);
            }
            else if (_stage == StageP)
            {
                // P -> RGBA when the first frame had a transparent index, else P -> RGB. (The RGBA canvas already holds the converted pixels.)
                _stage = _infoTrans >= 0 ? StageRgba : StageRgb;
                _infoTrans = -1;
            }

            // ... then this frame's own disposal is prepared (the method is sticky: 0 never overrides).
            int disposal = _l.Field(f, GifMeta.FrameDisposal);
            if (disposal != 0) _sticky = disposal;
            _pend = PendNone;
            bool opaque = _stage == StageRgb;
            if (_sticky == 2)
            {
                if (_stage == StageP || _stage == StageL)
                {
                    int color = _infoTrans >= 0 ? _infoTrans : trans >= 0 ? trans : _l.BackgroundOrZero;
                    _pend = PendFill; _pendIdx = color; _pendRule = GifKernels.FillThroughPalette;
                    _pendAlphaZero = (_infoTrans >= 0 && color == _infoTrans) ? 1 : 0; _pendOpaque = 0;
                }
                else
                {
                    int color = trans >= 0 ? trans : _l.BackgroundOrZero;
                    _pend = PendFill; _pendIdx = color; _pendRule = GifKernels.FillEntryZeroBeyond;
                    _pendAlphaZero = trans >= 0 ? 1 : 0; _pendOpaque = opaque ? 1 : 0;
                }
            }
            else if (_sticky >= 3)
            {
                if (f > 0)
                {
                    Add(f, GifKernels.OpSave, 0, -1, -1, 0, 0, 0, 0, 0, threads);
                    _pend = PendRestore;
                }
                else if (trans >= 0)
                {
                    // Frame 0 has no canvas to copy yet: Pillow fills the rectangle with the transparent index instead.
                    _pend = PendFill; _pendIdx = trans; _pendRule = GifKernels.FillThroughPalette; _pendAlphaZero = 1; _pendOpaque = 0;
                }
            }

            // ... and finally the frame is drawn.
            int skip = f == 0 ? -1 : trans;
            int alphaZero = f == 0 ? trans : (_stage == StageL ? _infoTrans : -1);
            Add(f, GifKernels.OpDraw, indexOffset, skip, alphaZero, 0, 0, 0, opaque ? 1 : 0, 1, threads);
        }
    }

    // ------------------------------------------------------------------
    // Pipeline
    // ------------------------------------------------------------------

    /// <summary>
    /// Read the facts about a GIF that is on the accelerator (block scan + a metadata-sized readback; no pixels
    /// are decoded). <see cref="ImageInfo.Width"/> / <see cref="ImageInfo.Height"/> are the logical screen.
    /// </summary>
    public async Task<ImageInfo> ReadInfoAsync(ArrayView1D<byte, Stride1D.Dense> file, int fileLength)
    {
        return ToInfo(await ScanAsync(file, fileLength));
    }

    /// <summary>
    /// List the frames of a GIF on the accelerator, as stored: rectangle, duration (delay x 10 ms; browsers raise
    /// very short delays, this reports the stored value), disposal (the method in the frame's own graphic control
    /// extension; <see cref="GifCompositing.Pillow"/> additionally carries the last non-zero method forward) and
    /// blend. Metadata only.
    /// </summary>
    public async Task<ImageFrameInfo[]> ReadFramesAsync(ArrayView1D<byte, Stride1D.Dense> file, int fileLength)
    {
        return (await ScanAsync(file, fileLength)).Frames;
    }

    /// <summary>
    /// The NETSCAPE loop count of a GIF on the accelerator: 0 = repeat forever, n = the value stored in the file,
    /// -1 = the file has no loop extension before its first image (it plays once).
    /// </summary>
    public async Task<int> ReadLoopCountAsync(ArrayView1D<byte, Stride1D.Dense> file, int fileLength)
    {
        return (await ScanAsync(file, fileLength)).LoopCount;
    }

    /// <summary>
    /// Decode the first frame (composited on the logical screen canvas) of a GIF on the accelerator into a
    /// caller-owned pixel view.
    /// </summary>
    public Task<ImageInfo> DecodeAsync(
        ArrayView1D<byte, Stride1D.Dense> file, int fileLength,
        ArrayView1D<byte, Stride1D.Dense> output, PixelFormat format = PixelFormat.Rgba8)
        => DecodeFrameAsync(file, fileLength, 0, output, format);

    /// <summary>
    /// Decode frame <paramref name="frameIndex"/>: the full canvas as it looks while that frame is displayed.
    /// Frames 0..frameIndex are decoded and composited on the accelerator; only metadata scalars cross to the host.
    /// </summary>
    /// <param name="file">The GIF file bytes (may be longer than <paramref name="fileLength"/>).</param>
    /// <param name="fileLength">Number of valid bytes in <paramref name="file"/>.</param>
    /// <param name="frameIndex">0-based frame index.</param>
    /// <param name="output">Receives the canvas; at least <see cref="ImageInfo.GetPixelBufferSize"/> bytes.</param>
    /// <param name="format">Pixel layout (Rgba16 widens each 8-bit sample v to v * 257).</param>
    /// <returns>The facts about the image (canvas size, frame count).</returns>
    /// <exception cref="ImageDecodeException">The file is corrupt, truncated or uses an unsupported feature.</exception>
    public async Task<ImageInfo> DecodeFrameAsync(
        ArrayView1D<byte, Stride1D.Dense> file, int fileLength, int frameIndex,
        ArrayView1D<byte, Stride1D.Dense> output, PixelFormat format = PixelFormat.Rgba8)
    {
        var layout = await ScanAsync(file, fileLength);
        var info = ToInfo(layout);
        CheckFrameIndex(layout, frameIndex);
        long outBytes = info.GetPixelBufferSize(format);
        if (output.Length < outBytes)
            throw new ArgumentException($"output must hold {outBytes} bytes for {info.Width}x{info.Height} {format}.", nameof(output));
        await RunPipelineAsync(file, layout, frameIndex, output, format);
        return info;
    }

    private static void CheckFrameIndex(Layout layout, int frameIndex)
    {
        if (frameIndex < 0 || frameIndex >= layout.FrameCount)
            throw new ArgumentOutOfRangeException(nameof(frameIndex), $"frame {frameIndex} of {layout.FrameCount}");
    }

    private async Task<ImageInfo> DecodeToStageAsync(int fileLength, int frameIndex, PixelFormat format)
    {
        var layout = await ScanAsync(_stageIn.View, fileLength);
        CheckFrameIndex(layout, frameIndex);
        var info = ToInfo(layout);
        _stageOut.Ensure(info.GetPixelBufferSize(format));
        await RunPipelineAsync(_stageIn.View, layout, frameIndex, _stageOut.View, format);
        return info;
    }

    private async Task RunPipelineAsync(
        ArrayView1D<byte, Stride1D.Dense> file, Layout layout, int target,
        ArrayView1D<byte, Stride1D.Dense> output, PixelFormat format)
    {
        int cw = layout.Width, ch = layout.Height;
        int pixels = cw * ch;
        var canvas = format == PixelFormat.Rgba8 ? output : _canvas.Ensure((long)pixels * 4);

        // Both modes may restore a rectangle (disposal 3 and, in Pillow mode, 4..7).
        bool anySave = false;
        for (int f = 0; f <= target; f++)
            if (layout.Field(f, GifMeta.FrameDisposal) >= 3) anySave = true;
        var saved = anySave ? _saved.Ensure((long)pixels * 4) : canvas;

        // Plan the batches first so the index buffer is sized once, before any dispatch reads it.
        var batches = new List<(int First, int Count, long Bytes)>();
        long maxBytes = 1;
        for (int f = 0; f <= target;)
        {
            int first = f;
            long bytes = 0;
            int count = 0;
            while (f <= target && count < MaxBatchFrames)
            {
                long fb = (long)layout.Frames[f].Width * layout.Frames[f].Height;
                if (count > 0 && bytes + fb > IndexBudgetBytes) break;
                bytes += fb;
                count++;
                f++;
            }
            batches.Add((first, count, bytes));
            if (bytes > maxBytes) maxBytes = bytes;
        }
        if (maxBytes > int.MaxValue) throw new ImageDecodeException(ImageFormat.Gif, "a batch of frames is too large to decode");
        var indices = _indices.Ensure(maxBytes);

        var planner = new Planner(layout, Compositing == GifCompositing.Pillow);
        foreach (var (first, count, _) in batches)
        {
            var plan = new int[count * GifKernels.PlanStride];
            long offset = 0;
            long lzwSteps = 1;
            for (int j = 0; j < count; j++)
            {
                int frame = first + j;
                long px = (long)layout.Frames[frame].Width * layout.Frames[frame].Height;
                plan[j * GifKernels.PlanStride] = frame;
                plan[j * GifKernels.PlanStride + 1] = (int)offset;
                offset += px;
                lzwSteps = Math.Max(lzwSteps, GifKernels.LzwStepBound(px, layout.Field(frame, GifMeta.FrameDataBytes), layout.Field(frame, GifMeta.FrameMinCodeSize)));
            }
            _plan.View.SubView(0, plan.Length).CopyFromCPU(ref plan[0], plan.Length);
            // LZW is resumable: bounded launches continue from the state the previous one left (the bound always finishes every frame).
            _kernels.LzwInit(_meta.View, _plan.View, _lzwState.View, count);
            var pacer = new LaunchPacer(_accelerator);
            for (long s = 0; s < lzwSteps; s++)
            {
                _kernels.LzwStep(file, _meta.View, _plan.View, indices, _lzwScratch.View, _lzwState.View, count);
                await pacer.LaunchedAsync();
            }
            await _accelerator.SynchronizeAsync();

            var status = await _plan.View.SubView(0, plan.Length).CopyToHostAsync();
            for (int j = 0; j < count; j++)
            {
                int frame = first + j;
                int s = status[j * GifKernels.PlanStride + 2];
                int produced = status[j * GifKernels.PlanStride + 3];
                long expected = (long)layout.Frames[frame].Width * layout.Frames[frame].Height;
                if (s == GifKernels.LzwBadCode)
                    throw new ImageDecodeException(ImageFormat.Gif, $"frame {frame}: invalid LZW code (not in the code table) after {produced} of {expected} pixels");
                if (s == GifKernels.LzwEarlyEnd)
                    throw new ImageDecodeException(ImageFormat.Gif, $"frame {frame}: LZW end code after {produced} of {expected} pixels");
                if (s == GifKernels.LzwDataEnded)
                    throw new ImageDecodeException(ImageFormat.Gif, $"frame {frame}: LZW data ends after {produced} of {expected} pixels");
            }

            // Plan and run this batch's compositing steps in frame order.
            planner.Steps.Clear();
            planner.Threads.Clear();
            long frameOffset = 0;
            for (int j = 0; j < count; j++)
            {
                planner.PlanFrame(first + j, (int)frameOffset);
                frameOffset += (long)layout.Frames[first + j].Width * layout.Frames[first + j].Height;
            }
            var stepInts = planner.Steps.ToArray();
            if (stepInts.Length > 0)
            {
                _steps.View.SubView(0, stepInts.Length).CopyFromCPU(ref stepInts[0], stepInts.Length);
                var compositePacer = new LaunchPacer(_accelerator);
                for (int s = 0; s < planner.StepCount; s++)
                {
                    _kernels.Composite(file, _meta.View, _steps.View, indices, canvas, saved, s, planner.Threads[s]);
                    await compositePacer.LaunchedAsync();
                }
            }
            // The next batch overwrites the steps and the index buffer: let these kernels finish first.
            await _accelerator.SynchronizeAsync();
        }

        if (format == PixelFormat.Rgba16) _pixel.Widen(canvas, output, pixels * 4);
        await _accelerator.SynchronizeAsync();
    }

    /// <summary>Threads covering the frame rectangle clipped to the canvas (0 when it lies outside).</summary>
    private static int ClippedThreads(ImageFrameInfo f, int canvasW, int canvasH)
    {
        long w = Math.Min((long)f.Width, (long)canvasW - f.X);
        long h = Math.Min((long)f.Height, (long)canvasH - f.Y);
        return w <= 0 || h <= 0 ? 0 : (int)(w * h);
    }

    // ------------------------------------------------------------------
    // Managed overloads
    // ------------------------------------------------------------------

    /// <summary>Read the facts about a managed GIF file (no pixels are decoded).</summary>
    public async Task<ImageInfo> ReadInfoAsync(ReadOnlyMemory<byte> file)
    {
        StageManaged(file);
        return await ReadInfoAsync(_stageIn.View, file.Length);
    }

    /// <summary>List the frames of a managed GIF file.</summary>
    public async Task<ImageFrameInfo[]> ReadFramesAsync(ReadOnlyMemory<byte> file)
    {
        StageManaged(file);
        return await ReadFramesAsync(_stageIn.View, file.Length);
    }

    /// <summary>The NETSCAPE loop count of a managed GIF file (0 = forever, -1 = no loop extension).</summary>
    public async Task<int> ReadLoopCountAsync(ReadOnlyMemory<byte> file)
    {
        StageManaged(file);
        return await ReadLoopCountAsync(_stageIn.View, file.Length);
    }

    /// <summary>
    /// Decode the first frame of a managed GIF file. Returns the pixels as a managed array (a terminal CPU
    /// sink; in a browser prefer the <see cref="TypedArray"/> overload).
    /// </summary>
    public Task<DecodedImage> DecodeAsync(ReadOnlyMemory<byte> file, PixelFormat format = PixelFormat.Rgba8)
        => DecodeFrameAsync(file, 0, format);

    /// <summary>Decode frame <paramref name="frameIndex"/> of a managed GIF file (the full canvas as displayed).</summary>
    public async Task<DecodedImage> DecodeFrameAsync(ReadOnlyMemory<byte> file, int frameIndex, PixelFormat format = PixelFormat.Rgba8)
    {
        StageManaged(file);
        var info = await DecodeToStageAsync(file.Length, frameIndex, format);
        var pixels = await DeviceTransfer.DownloadAsync(_stageOut.View, info.GetPixelBufferSize(format));
        return new DecodedImage(info, pixels, format);
    }

    /// <summary>Not async: it holds a span.</summary>
    private void StageManaged(ReadOnlyMemory<byte> file)
    {
        if (file.IsEmpty) throw new ArgumentException("File is empty.", nameof(file));
        _stageIn.Ensure(file.Length);
        DeviceTransfer.Upload(_stageIn.View, file.Span);
    }

    // ------------------------------------------------------------------
    // TypedArray overloads (browser: bulk bytes stay in JS)
    // ------------------------------------------------------------------

    /// <summary>Read the facts about a GIF held in a JS typed array (browser accelerators only).</summary>
    [SupportedOSPlatform("browser")]
    public async Task<ImageInfo> ReadInfoAsync(TypedArray file)
    {
        int length = StageTyped(file);
        return await ReadInfoAsync(_stageIn.View, length);
    }

    /// <summary>List the frames of a GIF held in a JS typed array (browser accelerators only).</summary>
    [SupportedOSPlatform("browser")]
    public async Task<ImageFrameInfo[]> ReadFramesAsync(TypedArray file)
    {
        int length = StageTyped(file);
        return await ReadFramesAsync(_stageIn.View, length);
    }

    /// <summary>The NETSCAPE loop count of a GIF held in a JS typed array (browser accelerators only).</summary>
    [SupportedOSPlatform("browser")]
    public async Task<int> ReadLoopCountAsync(TypedArray file)
    {
        int length = StageTyped(file);
        return await ReadLoopCountAsync(_stageIn.View, length);
    }

    /// <summary>
    /// Decode the first frame of a GIF held in a JS typed array (browser accelerators only). The bytes go to
    /// the GPU with <c>CopyFromJS</c> and the pixels come back as a JS <see cref="Uint8Array"/>; none of it
    /// enters the .NET heap. The caller disposes the result.
    /// </summary>
    [SupportedOSPlatform("browser")]
    public Task<DecodedImageJS> DecodeAsync(TypedArray file, PixelFormat format = PixelFormat.Rgba8)
        => DecodeFrameAsync(file, 0, format);

    /// <summary>Decode frame <paramref name="frameIndex"/> of a GIF held in a JS typed array (browser accelerators only).</summary>
    [SupportedOSPlatform("browser")]
    public async Task<DecodedImageJS> DecodeFrameAsync(TypedArray file, int frameIndex, PixelFormat format = PixelFormat.Rgba8)
    {
        int length = StageTyped(file);
        var info = await DecodeToStageAsync(length, frameIndex, format);
        var pixels = await DeviceTransfer.DownloadUint8ArrayAsync(_stageOut.View, info.GetPixelBufferSize(format));
        return new DecodedImageJS(info, pixels, format);
    }

    [SupportedOSPlatform("browser")]
    private int StageTyped(TypedArray file)
    {
        ArgumentNullException.ThrowIfNull(file);
        long length = file.ByteLength;
        _stageIn.Ensure(length);
        DeviceTransfer.Upload(_stageIn.View, file);
        return checked((int)length);
    }

    /// <summary>Release kernel resources and buffers.</summary>
    public void Dispose()
    {
        _kernels.Dispose();
        _pixel.Dispose();
        _meta.Dispose();
        _plan.Dispose();
        _steps.Dispose();
        _lzwScratch.Dispose();
        _lzwState.Dispose();
        _stageIn.Dispose();
        _stageOut.Dispose();
        _indices.Dispose();
        _canvas.Dispose();
        _saved.Dispose();
    }
}
