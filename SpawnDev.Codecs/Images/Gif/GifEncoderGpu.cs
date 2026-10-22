// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GIF encoder (89a, still and animated), 100% accelerator-resident: palette discovery, quantization, dithering,
// LZW, sub-blocking and file assembly are kernels (see GifEncoderKernels for the kernel chain). The host is a pure
// coordinator: it uploads the pixels (necessary I/O), uploads a few parameter ints once, enqueues the kernels in
// order, and reads back the file length (plus, for Floyd-Steinberg, one small block to learn which frames need
// error diffusion).
//
// What it writes
//   * Every frame covers the whole canvas (no delta rectangles). A pixel with alpha < AlphaThreshold is transparent.
//   * One exact palette (sorted ascending by 0xRRGGBB) for the whole animation when its opaque colours fit
//     MaxColors (minus one index reserved for transparency): lossless. Otherwise every frame is exact if it can be,
//     else QUANTIZED on the accelerator (median cut, see GifEncoderKernels) with its own local colour table
//     (a single still image gets a global table). Dithering applies to quantized frames only.
//   * When any pixel of the animation is transparent, EVERY frame gets the same graphic-control transparent index
//     (the entry after the colours) and, by default, disposal 2 (restore to background), so each decoder - Pillow
//     and browsers alike - shows exactly the pixels of the frame. Without transparency the frames use disposal 1.
//   * Animations (2+ frames) get a NETSCAPE2.0 loop extension (LoopCount, -1 = none) and per-frame delays.
//   * The LZW stream is cut into independent segments (SegmentPixels = 32768 pixels each, a clear code between
//     them) so a single large frame is encoded by many threads; the segments' bits are concatenated on the
//     accelerator. Images of up to 32768 pixels are one segment, i.e. byte-identical to a plain sequential encoder.
//
// API shape (Plans/PLAN-Image-Codecs.md): three overloads per entry point, all funnelling into the ArrayView1D one.
// Not thread-safe: one in-flight call per instance.

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Images.Gif;

/// <summary>Dithering applied while mapping pixels to a QUANTIZED palette (an exact palette is lossless and never dithered).</summary>
public enum GifDither
{
    /// <summary>Nearest palette colour.</summary>
    None = 0,
    /// <summary>8x8 Bayer ordered dither (+-16 levels added to every channel before the nearest-colour search).</summary>
    Ordered = 1,
    /// <summary>Floyd-Steinberg error diffusion (weights 7/3/5/1; run in anti-diagonal waves, identical to the sequential algorithm).</summary>
    FloydSteinberg = 2,
}

/// <summary>Options for <see cref="GifEncoderGpu"/>.</summary>
public sealed record GifEncodeOptions
{
    /// <summary>Largest palette, transparent index included (2..256). Default 256.</summary>
    public int MaxColors { get; init; } = 256;
    /// <summary>Dithering for quantized frames. Default <see cref="GifDither.None"/>.</summary>
    public GifDither Dither { get; init; } = GifDither.None;
    /// <summary>Pixels with alpha below this are transparent (0 = none, 256 = all). Default 128.</summary>
    public int AlphaThreshold { get; init; } = 128;
    /// <summary>Animation repeat count written to the NETSCAPE extension: 0 = forever, n = n, -1 = no extension. Default 0.</summary>
    public int LoopCount { get; init; } = 0;
    /// <summary>Frame delay in milliseconds when <see cref="FrameDelaysMs"/> is not given (stored in 10 ms units, rounded). Default 100.</summary>
    public int FrameDelayMs { get; init; } = 100;
    /// <summary>Per-frame delays in milliseconds (one per frame), or null for <see cref="FrameDelayMs"/>.</summary>
    public IReadOnlyList<int>? FrameDelaysMs { get; init; }
    /// <summary>
    /// Per-frame disposal (one per frame), or null for automatic: Background for every frame when the animation has
    /// transparent pixels (so each frame shows only its own pixels), None otherwise.
    /// </summary>
    public IReadOnlyList<FrameDispose>? FrameDisposals { get; init; }
}

/// <summary>GIF encoder running on an ILGPU accelerator.</summary>
/// <remarks>Not thread-safe: one in-flight call per instance.</remarks>
public sealed class GifEncoderGpu : IDisposable
{
    /// <summary>Largest picture (pixels per frame) the encoder accepts.</summary>
    public const int MaxPixels = 1 << 27;
    private const int PixelsPerBatch = 1 << 22;
    private const int MaxFramesPerBatch = 64;
    private const int MaxLzwThreads = 1024;

    private readonly Accelerator _accelerator;
    private readonly GifEncoderKernels _kernels;
    private readonly ExactPaletteBuilder _globalBuilder;
    private readonly ExactPaletteBuilder _frameBuilder;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _prm;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _gstate;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _quant;
    private readonly StageBuffer<int> _fpal;
    private readonly StageBuffer<int> _finfo;
    private readonly StageBuffer<int> _delays;
    private readonly StageBuffer<int> _disposals;
    private readonly StageBuffer<byte> _idx;
    private readonly StageBuffer<byte> _seg;
    private readonly StageBuffer<int> _segInfo;
    private readonly StageBuffer<int> _segOff;
    private readonly StageBuffer<int> _hash;
    private readonly StageBuffer<int> _payload;
    private readonly StageBuffer<int> _err;
    private readonly StageBuffer<byte> _stageIn;
    private readonly StageBuffer<byte> _stageOut;

    /// <summary>Compile the kernels.</summary>
    public GifEncoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernels = new GifEncoderKernels(accelerator);
        _globalBuilder = new ExactPaletteBuilder(accelerator);
        _frameBuilder = new ExactPaletteBuilder(accelerator);
        _prm = accelerator.Allocate1D<int>(GifEncParam.Size);
        _gstate = accelerator.Allocate1D<int>(GifEncGlobal.Size);
        _quant = accelerator.Allocate1D<int>(GifEncoderKernels.QuantScratchInts);
        _fpal = new StageBuffer<int>(accelerator);
        _finfo = new StageBuffer<int>(accelerator);
        _delays = new StageBuffer<int>(accelerator);
        _disposals = new StageBuffer<int>(accelerator);
        _idx = new StageBuffer<byte>(accelerator);
        _seg = new StageBuffer<byte>(accelerator);
        _segInfo = new StageBuffer<int>(accelerator);
        _segOff = new StageBuffer<int>(accelerator);
        _hash = new StageBuffer<int>(accelerator);
        _payload = new StageBuffer<int>(accelerator);
        _err = new StageBuffer<int>(accelerator);
        _stageIn = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
    }

    // ------------------------------------------------------------------
    // sizes
    // ------------------------------------------------------------------

    private sealed record Geometry(int Pixels, int Segments, int SegBound, int PayloadBound, int PayloadWords, int FramedBound);

    private static Geometry GeometryFor(int width, int height)
    {
        if (width <= 0 || width > 65535) throw new ArgumentOutOfRangeException(nameof(width), "GIF dimensions are 1..65535.");
        if (height <= 0 || height > 65535) throw new ArgumentOutOfRangeException(nameof(height), "GIF dimensions are 1..65535.");
        long px = (long)width * height;
        if (px > MaxPixels) throw new ArgumentOutOfRangeException(nameof(width), $"{width}x{height} exceeds the encoder's {MaxPixels} pixel limit.");
        int pixels = (int)px;
        int segPx = GifEncoderKernels.SegmentPixels;
        int segments = (pixels + segPx - 1) / segPx;
        // A code is at most 12 bits; a segment emits at most one code per pixel plus a clear per 4093 codes plus its own
        // framing codes (leading clear, trailing clear / end).
        long segBound = ((long)(segPx + segPx / 4093 + 3) * 12 + 7) / 8 + 8;
        long payloadBound = (((long)pixels + pixels / 4093 + 3L * segments) * 12 + 7) / 8 + 8;
        long framedBound = payloadBound + (payloadBound + 254) / 255 + 1;
        return new Geometry(pixels, segments, (int)segBound, (int)payloadBound, (int)(payloadBound / 4 + 2), (int)framedBound);
    }

    /// <summary>Upper bound on the encoded size of one picture.</summary>
    public static int GetMaxEncodedSize(int width, int height, GifEncodeOptions? options = null)
        => GetMaxEncodedSize(width, height, 1, options);

    /// <summary>Upper bound on the encoded size of an animation of <paramref name="frameCount"/> full-canvas frames.</summary>
    public static int GetMaxEncodedSize(int width, int height, int frameCount, GifEncodeOptions? options = null)
    {
        if (frameCount <= 0 || frameCount > GifMeta.MaxFrames)
            throw new ArgumentOutOfRangeException(nameof(frameCount), $"1..{GifMeta.MaxFrames} frames.");
        var g = GeometryFor(width, height);
        // header + global table + loop extension, per frame (GCE 8 + descriptor 10 + local table 768 + min code size 1 + framed data), trailer
        long total = 13 + 768 + 19 + (long)frameCount * (8 + 10 + 768 + 1 + g.FramedBound) + 1;
        if (total > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(frameCount), "The encoded animation could exceed 2 GB.");
        return (int)total;
    }

    // ------------------------------------------------------------------
    // encode
    // ------------------------------------------------------------------

    /// <summary>
    /// Encode one RGBA8 picture that is already on the accelerator into a caller-owned output view. Only the file
    /// length (and a few metadata ints) cross to the host.
    /// </summary>
    /// <param name="rgba">Tightly packed RGBA8, rows top to bottom (<c>width * height * 4</c> bytes).</param>
    /// <param name="width">Picture width (1..65535).</param>
    /// <param name="height">Picture height (1..65535).</param>
    /// <param name="output">Receives the file; at least <see cref="GetMaxEncodedSize(int, int, GifEncodeOptions?)"/> bytes.</param>
    /// <param name="options">Palette size, dithering, alpha threshold; defaults when null.</param>
    /// <returns>The file length in bytes.</returns>
    public Task<int> EncodeAsync(
        ArrayView1D<byte, Stride1D.Dense> rgba, int width, int height,
        ArrayView1D<byte, Stride1D.Dense> output, GifEncodeOptions? options = null)
        => EncodeAnimationAsync(rgba, width, height, 1, output, options);

    /// <summary>
    /// Encode <paramref name="frameCount"/> RGBA8 frames of the same size (stored back to back in
    /// <paramref name="rgbaFrames"/>) as an animated GIF into a caller-owned output view.
    /// </summary>
    /// <returns>The file length in bytes.</returns>
    public async Task<int> EncodeAnimationAsync(
        ArrayView1D<byte, Stride1D.Dense> rgbaFrames, int width, int height, int frameCount,
        ArrayView1D<byte, Stride1D.Dense> output, GifEncodeOptions? options = null)
    {
        options ??= new GifEncodeOptions();
        int max = GetMaxEncodedSize(width, height, frameCount, options);
        var geo = GeometryFor(width, height);
        Validate(options, frameCount);
        long needed = (long)geo.Pixels * 4 * frameCount;
        if (rgbaFrames.Length < needed)
            throw new ArgumentException($"rgba must hold {needed} bytes for {frameCount} frame(s) of {width}x{height}.", nameof(rgbaFrames));
        if (output.Length < max)
            throw new ArgumentException($"output must hold {max} bytes (GetMaxEncodedSize).", nameof(output));
        return await EncodeCoreAsync(rgbaFrames, width, height, frameCount, geo, output, options);
    }

    private static void Validate(GifEncodeOptions o, int frames)
    {
        if (o.MaxColors < 2 || o.MaxColors > 256) throw new ArgumentOutOfRangeException(nameof(o), "MaxColors must be 2..256.");
        if (o.AlphaThreshold < 0 || o.AlphaThreshold > 256) throw new ArgumentOutOfRangeException(nameof(o), "AlphaThreshold must be 0..256.");
        if (!Enum.IsDefined(o.Dither)) throw new ArgumentOutOfRangeException(nameof(o), "Unknown dither mode.");
        if (o.LoopCount < -1 || o.LoopCount > 65535) throw new ArgumentOutOfRangeException(nameof(o), "LoopCount must be -1..65535.");
        if (o.FrameDelayMs < 0 || o.FrameDelayMs > 655350) throw new ArgumentOutOfRangeException(nameof(o), "FrameDelayMs must be 0..655350.");
        if (o.FrameDelaysMs != null)
        {
            if (o.FrameDelaysMs.Count != frames) throw new ArgumentException($"FrameDelaysMs has {o.FrameDelaysMs.Count} entries for {frames} frame(s).", nameof(o));
            foreach (var d in o.FrameDelaysMs)
                if (d < 0 || d > 655350) throw new ArgumentOutOfRangeException(nameof(o), "Frame delays must be 0..655350 ms.");
        }
        if (o.FrameDisposals != null)
        {
            if (o.FrameDisposals.Count != frames) throw new ArgumentException($"FrameDisposals has {o.FrameDisposals.Count} entries for {frames} frame(s).", nameof(o));
            foreach (var d in o.FrameDisposals)
                if (!Enum.IsDefined(d)) throw new ArgumentOutOfRangeException(nameof(o), "Unknown frame disposal.");
        }
    }

    private async Task<int> EncodeCoreAsync(
        ArrayView1D<byte, Stride1D.Dense> rgba, int width, int height, int frames, Geometry geo,
        ArrayView1D<byte, Stride1D.Dense> output, GifEncodeOptions o)
    {
        int pixels = geo.Pixels;
        long frameBytes = (long)pixels * 4;

        // ---- parameters, uploaded once (never rewritten while kernels are in flight) ----
        var prm = new int[GifEncParam.Size];
        prm[GifEncParam.Frames] = frames;
        prm[GifEncParam.Width] = width;
        prm[GifEncParam.Height] = height;
        prm[GifEncParam.MaxColors] = o.MaxColors;
        prm[GifEncParam.AlphaThreshold] = o.AlphaThreshold;
        prm[GifEncParam.Dither] = (int)o.Dither;
        prm[GifEncParam.LoopCount] = o.LoopCount;
        prm[GifEncParam.SegPixels] = GifEncoderKernels.SegmentPixels;
        prm[GifEncParam.SegCount] = geo.Segments;
        _prm.View.CopyFromCPU(ref prm[0], prm.Length);

        var delays = new int[frames];
        var disposals = new int[frames];
        for (int f = 0; f < frames; f++)
        {
            int ms = o.FrameDelaysMs != null ? o.FrameDelaysMs[f] : o.FrameDelayMs;
            delays[f] = Math.Min(65535, (ms + 5) / 10);
            disposals[f] = o.FrameDisposals == null ? -1 : o.FrameDisposals[f] switch
            {
                FrameDispose.Background => 2,
                FrameDispose.Previous => 3,
                _ => 1,
            };
        }
        var delayView = _delays.Ensure(frames);
        var dispView = _disposals.Ensure(frames);
        delayView.SubView(0, frames).CopyFromCPU(ref delays[0], frames);
        dispView.SubView(0, frames).CopyFromCPU(ref disposals[0], frames);

        var fpal = _fpal.Ensure((long)frames * 256);
        var finfo = _finfo.Ensure((long)frames * GifEncFrame.Stride);
        _gstate.View.MemSetToZero();

        // ---- palette: one exact palette for the whole animation if it fits ----
        var gb = _globalBuilder;
        gb.Reset();
        for (int f = 0; f < frames; f++) gb.ScanAlpha(rgba.SubView(f * frameBytes, frameBytes), pixels, o.AlphaThreshold);
        gb.SetLimit(o.MaxColors);
        for (int f = 0; f < frames; f++) gb.Insert(rgba.SubView(f * frameBytes, frameBytes), pixels, o.AlphaThreshold);
        gb.Finish();
        _kernels.PrepGlobal(gb.State, gb.Palette, _gstate.View, fpal, finfo, _prm.View, frames);
        await _accelerator.SynchronizeAsync();
        var g0 = await _gstate.View.SubView(0, GifEncGlobal.Size).CopyToHostAsync();
        bool globalExact = g0[GifEncGlobal.Exact] != 0;

        // ---- not exact: every frame gets its own exact-or-quantized palette ----
        if (!globalExact)
        {
            var fb = _frameBuilder;
            for (int f = 0; f < frames; f++)
            {
                var frameView = rgba.SubView(f * frameBytes, frameBytes);
                fb.Reset();
                fb.SetLimitFrom(_gstate.View, GifEncGlobal.AnimHasTrans, o.MaxColors);
                fb.Insert(frameView, pixels, o.AlphaThreshold);
                fb.Finish();
                _kernels.FramePalette(fb.State, fb.Palette, _gstate.View, fpal, finfo, f);
                _quant.View.SubView(GifEncoderKernels.HistOff, GifEncoderKernels.HistInts).MemSetToZero();
                _kernels.Hist(frameView, _quant.View, finfo, _gstate.View, _prm.View, f, pixels);
                _kernels.MedianCut(_quant.View, finfo, _gstate.View, fpal, _prm.View, f);
            }
        }
        _kernels.Layout(_gstate.View, finfo, _prm.View, dispView);
        _kernels.Header(output, _gstate.View, finfo, fpal, _prm.View);

        // Floyd-Steinberg only touches frames whose palette is quantized: learn which ones (a few ints).
        bool[]? fsFrames = null;
        if (o.Dither == GifDither.FloydSteinberg && !globalExact)
        {
            await _accelerator.SynchronizeAsync();
            var info = await finfo.SubView(0, (long)frames * GifEncFrame.Stride).CopyToHostAsync();
            fsFrames = new bool[frames];
            for (int f = 0; f < frames; f++) fsFrames[f] = info[f * GifEncFrame.Stride + GifEncFrame.Exact] == 0;
        }

        // ---- frames, in batches that bound the scratch memory ----
        int perBatch = Math.Max(1, Math.Min(MaxFramesPerBatch, PixelsPerBatch / pixels));
        perBatch = Math.Min(perBatch, frames);
        int segs = geo.Segments;
        var idx = _idx.Ensure((long)perBatch * pixels);
        var seg = _seg.Ensure((long)perBatch * segs * geo.SegBound);
        var segInfo = _segInfo.Ensure((long)perBatch * segs);
        var segOff = _segOff.Ensure((long)perBatch * segs);
        int chunk = Math.Min(MaxLzwThreads, perBatch * segs);
        var hash = _hash.Ensure((long)chunk * GifEncoderKernels.HashSlots);
        var payload = _payload.Ensure((long)perBatch * geo.PayloadWords);
        ArrayView1D<int, Stride1D.Dense> err = default;
        if (fsFrames != null) err = _err.Ensure((long)pixels * 3);

        for (int f0 = 0; f0 < frames; f0 += perBatch)
        {
            int n = Math.Min(perBatch, frames - f0);
            var rgbaBatch = rgba.SubView(f0 * frameBytes, n * frameBytes);
            _kernels.Map(rgbaBatch, idx, fpal, finfo, _prm.View, f0, n * pixels);
            if (fsFrames != null)
            {
                for (int j = 0; j < n; j++)
                {
                    if (!fsFrames[f0 + j]) continue;
                    var rgbaFrame = rgba.SubView((f0 + j) * frameBytes, frameBytes);
                    var idxFrame = idx.SubView((long)j * pixels, pixels);
                    for (int wave = 0; wave < width + 2 * (height - 1); wave++)
                        _kernels.FsWave(rgbaFrame, idxFrame, err, fpal, finfo, _prm.View, wave, f0 + j, height);
                }
            }
            payload.SubView(0, (long)n * geo.PayloadWords).MemSetToZero();
            int totalSegs = n * segs;
            for (int s = 0; s < totalSegs; s += chunk)
                _kernels.LzwSegments(idx, seg, segInfo, hash, finfo, _prm.View, f0, s, geo.SegBound, Math.Min(chunk, totalSegs - s));
            _kernels.SegOffsets(segInfo, segOff, finfo, _prm.View, f0, n);
            _kernels.FrameOffsets(_gstate.View, finfo, f0, n);
            _kernels.Gather(seg, segInfo, segOff, payload, _prm.View, geo.SegBound, geo.PayloadWords, totalSegs);
            _kernels.WritePrefix(output, _gstate.View, finfo, fpal, _prm.View, delayView, f0, n);
            _kernels.CopyFramed(output, finfo, payload, f0, geo.PayloadWords, geo.FramedBound, checked(n * geo.FramedBound));
            // The next batch reuses the scratch buffers.
            await _accelerator.SynchronizeAsync();
        }

        _kernels.Trailer(output, _gstate.View);
        await _accelerator.SynchronizeAsync();
        var g1 = await _gstate.View.SubView(0, GifEncGlobal.Size).CopyToHostAsync();
        return g1[GifEncGlobal.TotalLen];
    }

    // ------------------------------------------------------------------
    // Managed overloads
    // ------------------------------------------------------------------

    /// <summary>
    /// Encode one managed RGBA8 picture. Returns the file as a managed array (a terminal CPU sink; in a browser
    /// prefer the <see cref="TypedArray"/> overload).
    /// </summary>
    public Task<byte[]> EncodeAsync(ReadOnlyMemory<byte> rgba, int width, int height, GifEncodeOptions? options = null)
        => EncodeAnimationAsync(rgba, width, height, 1, options);

    /// <summary>Encode managed RGBA8 frames (back to back) as an animated GIF. Returns the file as a managed array.</summary>
    public async Task<byte[]> EncodeAnimationAsync(ReadOnlyMemory<byte> rgbaFrames, int width, int height, int frameCount, GifEncodeOptions? options = null)
    {
        int max = GetMaxEncodedSize(width, height, frameCount, options);
        long needed = (long)width * height * 4 * frameCount;
        if (rgbaFrames.Length < needed)
            throw new ArgumentException($"rgba must hold {needed} bytes for {frameCount} frame(s) of {width}x{height}.", nameof(rgbaFrames));
        _stageIn.Ensure(needed);
        DeviceTransfer.Upload(_stageIn.View, rgbaFrames.Span.Slice(0, (int)needed));
        _stageOut.Ensure(max);
        int length = await EncodeAnimationAsync(_stageIn.View, width, height, frameCount, _stageOut.View, options);
        return await DeviceTransfer.DownloadAsync(_stageOut.View, length);
    }

    // ------------------------------------------------------------------
    // TypedArray overloads (browser: bulk bytes stay in JS)
    // ------------------------------------------------------------------

    /// <summary>
    /// Encode one RGBA8 picture held in a JS typed array (browser accelerators only). The pixels go to the GPU with
    /// <c>CopyFromJS</c> and the file comes back as a JS <see cref="Uint8Array"/>; none of it enters the .NET heap.
    /// The caller disposes the result.
    /// </summary>
    [SupportedOSPlatform("browser")]
    public Task<Uint8Array> EncodeAsync(TypedArray rgba, int width, int height, GifEncodeOptions? options = null)
        => EncodeAnimationAsync(rgba, width, height, 1, options);

    /// <summary>Encode RGBA8 frames (back to back) held in a JS typed array as an animated GIF (browser accelerators only).</summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> EncodeAnimationAsync(TypedArray rgbaFrames, int width, int height, int frameCount, GifEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(rgbaFrames);
        int max = GetMaxEncodedSize(width, height, frameCount, options);
        long needed = (long)width * height * 4 * frameCount;
        long length = rgbaFrames.ByteLength;
        if (length < needed)
            throw new ArgumentException($"rgba must hold {needed} bytes for {frameCount} frame(s) of {width}x{height}.", nameof(rgbaFrames));
        _stageIn.Ensure(length);
        DeviceTransfer.Upload(_stageIn.View, rgbaFrames);
        _stageOut.Ensure(max);
        int encoded = await EncodeAnimationAsync(_stageIn.View, width, height, frameCount, _stageOut.View, options);
        return await DeviceTransfer.DownloadUint8ArrayAsync(_stageOut.View, encoded);
    }

    /// <summary>Release kernel resources and buffers.</summary>
    public void Dispose()
    {
        _kernels.Dispose();
        _globalBuilder.Dispose();
        _frameBuilder.Dispose();
        _prm.Dispose();
        _gstate.Dispose();
        _quant.Dispose();
        _fpal.Dispose();
        _finfo.Dispose();
        _delays.Dispose();
        _disposals.Dispose();
        _idx.Dispose();
        _seg.Dispose();
        _segInfo.Dispose();
        _segOff.Dispose();
        _hash.Dispose();
        _payload.Dispose();
        _err.Dispose();
        _stageIn.Dispose();
        _stageOut.Dispose();
    }
}
