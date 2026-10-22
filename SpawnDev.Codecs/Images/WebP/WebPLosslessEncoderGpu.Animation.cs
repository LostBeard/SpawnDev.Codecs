// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Animated lossless WebP encoder. Every frame is a lossless VP8L image encoded by the same accelerator pipeline as a still
// image (palette / predictor / cross colour / LZ77 / colour cache / meta prefix codes, strategy trials), written straight
// into its ANMF chunk of the output; the container headers (RIFF, VP8X, ICCP, ANIM, ANMF, the VP8L chunk headers, EXIF,
// XMP) are written by kernels. The host only sequences the frames and reads back a few scalars per frame (the stream
// length, the alpha fact).
//
// Frames are what the caller gives: a rectangle on the canvas (even offsets: WebP stores them halved), its RGBA pixels, a
// duration, a blend method (Source = replace, Over = alpha-blend onto the canvas) and a disposal (None or Background).
// FrameDispose.Previous does not exist in WebP and is rejected. The encoder does not diff frames or pick sub-rectangles:
// that is a choice about the CONTENT the caller makes (and a candidate for a later option).
//
// Three overloads per entry point, funnelling into the ArrayView1D one, like every entry point of this class.

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Images.WebP;

public sealed partial class WebPLosslessEncoderGpu
{
    private const int MaxCanvasDimension = 1 << 24;

    private static void ValidateAnimation(IReadOnlyList<ImageFrameInfo> frames, int canvasWidth, int canvasHeight, WebPAnimationEncodeOptions? options)
    {
        ArgumentNullException.ThrowIfNull(frames);
        if (frames.Count == 0) throw new ArgumentException("An animation needs at least one frame.", nameof(frames));
        if (canvasWidth < 1 || canvasWidth > MaxCanvasDimension) throw new ArgumentOutOfRangeException(nameof(canvasWidth), $"Canvas width must be 1..{MaxCanvasDimension}.");
        if (canvasHeight < 1 || canvasHeight > MaxCanvasDimension) throw new ArgumentOutOfRangeException(nameof(canvasHeight), $"Canvas height must be 1..{MaxCanvasDimension}.");
        if ((long)canvasWidth * canvasHeight >= (1L << 32)) throw new ArgumentOutOfRangeException(nameof(canvasWidth), "Canvas has too many pixels (width x height must be below 2^32).");
        if (options != null)
        {
            if (options.Quality < 0 || options.Quality > 100) throw new ArgumentOutOfRangeException(nameof(options), "Quality must be 0..100.");
            if (options.LoopCount < 0 || options.LoopCount > 65535) throw new ArgumentOutOfRangeException(nameof(options), "LoopCount must be 0..65535.");
        }
        for (int k = 0; k < frames.Count; k++)
        {
            var f = frames[k] ?? throw new ArgumentException($"Frame {k} is null.", nameof(frames));
            string at = $"frame {k}";
            ValidateSize(f.Width, f.Height);
            if (f.X < 0 || f.Y < 0 || (f.X & 1) != 0 || (f.Y & 1) != 0)
                throw new ArgumentException($"{at}: offsets must be even and not negative (WebP stores them halved); got ({f.X}, {f.Y}).", nameof(frames));
            if ((long)f.X + f.Width > canvasWidth || (long)f.Y + f.Height > canvasHeight)
                throw new ArgumentException($"{at}: the {f.Width}x{f.Height} rectangle at ({f.X}, {f.Y}) does not fit the {canvasWidth}x{canvasHeight} canvas.", nameof(frames));
            if (f.DurationMs < 0 || f.DurationMs >= (1 << 24))
                throw new ArgumentException($"{at}: duration must be 0..{(1 << 24) - 1} ms.", nameof(frames));
            if (f.Dispose != FrameDispose.None && f.Dispose != FrameDispose.Background)
                throw new ArgumentException($"{at}: WebP supports only FrameDispose.None and FrameDispose.Background.", nameof(frames));
            if (f.Blend != FrameBlend.Source && f.Blend != FrameBlend.Over)
                throw new ArgumentException($"{at}: unknown blend method.", nameof(frames));
        }
    }

    /// <summary>
    /// Upper bound on the size of the animated file, for sizing the output buffer of
    /// <see cref="EncodeAnimationAsync(IReadOnlyList{ArrayView1D{byte, Stride1D.Dense}}, IReadOnlyList{ImageFrameInfo}, int, int, ArrayView1D{byte, Stride1D.Dense}, WebPAnimationEncodeOptions?)"/>.
    /// </summary>
    public static int GetMaxAnimationEncodedSize(IReadOnlyList<ImageFrameInfo> frames, WebPAnimationEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(frames);
        long size = 12 + 18 + 14 + 8 * 4;
        if (options?.IccProfile is { } icc) size += icc.Length + 1;
        if (ExifPayload(options?.Exif) is { } exif) size += exif.Length + 1;
        if (options?.Xmp is { } xmp) size += xmp.Length + 1;
        foreach (var f in frames)
        {
            ValidateSize(f.Width, f.Height);
            size += 8 + 16 + 8 + 5L * f.Width * f.Height + 64 * 1024;
        }
        if (size > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(frames), "The animation is too large to encode.");
        return (int)size;
    }

    /// <summary>
    /// Encode an animation whose frames' RGBA8 pixels are already on the accelerator into a caller-owned view. Only a few
    /// scalars per frame cross to the host.
    /// </summary>
    /// <param name="frameRgba">Per frame, tightly packed RGBA8 pixels (straight alpha, rows top to bottom) of exactly the frame's rectangle.</param>
    /// <param name="frames">Per frame, its rectangle on the canvas, duration, disposal and blend (the <c>Index</c> is ignored: list order is playback order).</param>
    /// <param name="canvasWidth">Canvas width (1..2^24).</param>
    /// <param name="canvasHeight">Canvas height (1..2^24).</param>
    /// <param name="output">Receives the WebP file; at least <see cref="GetMaxAnimationEncodedSize"/> bytes.</param>
    /// <param name="options">Encoder options (null = defaults): quality and metadata apply to the whole file, loop count and background colour to the animation.</param>
    /// <returns>The length of the file in <paramref name="output"/>.</returns>
    public async Task<int> EncodeAnimationAsync(
        IReadOnlyList<ArrayView1D<byte, Stride1D.Dense>> frameRgba, IReadOnlyList<ImageFrameInfo> frames,
        int canvasWidth, int canvasHeight, ArrayView1D<byte, Stride1D.Dense> output, WebPAnimationEncodeOptions? options = null)
    {
        options ??= new WebPAnimationEncodeOptions();
        ValidateAnimation(frames, canvasWidth, canvasHeight, options);
        ArgumentNullException.ThrowIfNull(frameRgba);
        if (frameRgba.Count != frames.Count)
            throw new ArgumentException($"{frameRgba.Count} pixel buffers for {frames.Count} frames.", nameof(frameRgba));
        for (int k = 0; k < frames.Count; k++)
        {
            long need = (long)frames[k].Width * frames[k].Height * 4;
            if (frameRgba[k].Length < need)
                throw new ArgumentException($"frame {k}: rgba must hold {need} bytes for {frames[k].Width}x{frames[k].Height}.", nameof(frameRgba));
        }
        int maxSize = GetMaxAnimationEncodedSize(frames, options);
        if (output.Length < maxSize)
            throw new ArgumentException($"output must hold {maxSize} bytes (GetMaxAnimationEncodedSize).", nameof(output));

        byte[]? icc = options.IccProfile is { Length: > 0 } ? options.IccProfile : null;
        byte[]? exif = ExifPayload(options.Exif);
        byte[]? xmp = options.Xmp is { Length: > 0 } ? options.Xmp : null;
        var blobs = UploadBlobs(icc, exif, xmp);
        var meta = _meta.View;
        int exact = options.Exact ? 1 : 0;

        var hostMeta = new int[WebPEncMeta.Size];
        hostMeta[WebPEncMeta.OutputCapacity] = (int)Math.Min(output.Length, int.MaxValue);
        hostMeta[WebPEncMeta.ExtendedFile] = 1;
        hostMeta[WebPEncMeta.IccLength] = icc?.Length ?? 0;
        hostMeta[WebPEncMeta.ExifLength] = exif?.Length ?? 0;
        hostMeta[WebPEncMeta.XmpLength] = xmp?.Length ?? 0;

        // [RIFF 12][VP8X 18][ICCP][ANIM 14] then the ANMF chunks
        int cursor = 12 + 18 + 14;
        if (icc != null) cursor += 8 + icc.Length + (icc.Length & 1);
        bool anyAlpha = false;
        for (int k = 0; k < frames.Count; k++)
        {
            var f = frames[k];
            var pixels = frameRgba[k];
            int n = f.Width * f.Height;
            hostMeta[WebPEncMeta.PayloadStart] = cursor + 8 + 16 + 8; // ANMF header, ANMF parameters, VP8L chunk header
            hostMeta[WebPEncMeta.FrameX] = f.X;
            hostMeta[WebPEncMeta.FrameY] = f.Y;
            hostMeta[WebPEncMeta.FrameDuration] = f.DurationMs;
            hostMeta[WebPEncMeta.FrameFlags] = (f.Dispose == FrameDispose.Background ? 1 : 0) | (f.Blend == FrameBlend.Source ? 2 : 0);
            await EncodeStreamAsync(
                new StreamJob(f.Width, f.Height, options.Quality, a => _kernels.ToArgb(pixels, a, n, exact), Plane: false),
                hostMeta, output);
            _kernels.FinalizeFrame(output, meta);
            await _accelerator.SynchronizeAsync();
            var m = await meta.CopyToHostAsync();
            if (m[WebPEncMeta.Error] != 0 || m[WebPEncMeta.Total] > output.Length)
                throw new InvalidOperationException(
                    $"WebP encode overflowed the output buffer at frame {k} (needed {m[WebPEncMeta.Total]} bytes, GetMaxAnimationEncodedSize said {maxSize}).");
            anyAlpha |= m[WebPEncMeta.HasAlpha] == 1;
            cursor = m[WebPEncMeta.Total];
        }

        hostMeta[WebPEncMeta.CanvasWidth] = canvasWidth;
        hostMeta[WebPEncMeta.CanvasHeight] = canvasHeight;
        hostMeta[WebPEncMeta.LoopCount] = options.LoopCount;
        hostMeta[WebPEncMeta.Background] = unchecked((int)options.BackgroundColor);
        hostMeta[WebPEncMeta.AnimEnd] = cursor;
        hostMeta[WebPEncMeta.AnimAlpha] = anyAlpha ? 1 : 0;
        meta.SubView(WebPEncMeta.HostStart, WebPEncMeta.Size - WebPEncMeta.HostStart)
            .CopyFromCPU(ref hostMeta[WebPEncMeta.HostStart], WebPEncMeta.Size - WebPEncMeta.HostStart);
        _kernels.FinalizeAnimation(output, meta);
        CopyBlobs(blobs, output, icc, exif, xmp);
        await _accelerator.SynchronizeAsync();

        var result = await meta.CopyToHostAsync();
        if (result[WebPEncMeta.Total] > output.Length)
            throw new InvalidOperationException(
                $"WebP encode overflowed the output buffer (needed {result[WebPEncMeta.Total]} bytes, GetMaxAnimationEncodedSize said {maxSize}).");
        return result[WebPEncMeta.Total];
    }

    /// <summary>
    /// Encode an animation of managed RGBA8 frames. Returns the file as a managed array (a terminal CPU sink; in a browser
    /// prefer the <see cref="TypedArray"/> overload).
    /// </summary>
    public async Task<byte[]> EncodeAnimationAsync(
        IReadOnlyList<ReadOnlyMemory<byte>> frameRgba, IReadOnlyList<ImageFrameInfo> frames,
        int canvasWidth, int canvasHeight, WebPAnimationEncodeOptions? options = null)
    {
        ValidateAnimation(frames, canvasWidth, canvasHeight, options);
        ArgumentNullException.ThrowIfNull(frameRgba);
        if (frameRgba.Count != frames.Count)
            throw new ArgumentException($"{frameRgba.Count} pixel buffers for {frames.Count} frames.", nameof(frameRgba));
        long total = 0;
        for (int k = 0; k < frames.Count; k++)
        {
            long need = (long)frames[k].Width * frames[k].Height * 4;
            if (frameRgba[k].Length < need)
                throw new ArgumentException($"frame {k}: rgba must hold {need} bytes for {frames[k].Width}x{frames[k].Height}.", nameof(frameRgba));
            total += need;
        }
        _stageIn.Ensure(total);
        var views = new ArrayView1D<byte, Stride1D.Dense>[frames.Count];
        long at = 0;
        for (int k = 0; k < frames.Count; k++)
        {
            long need = (long)frames[k].Width * frames[k].Height * 4;
            views[k] = _stageIn.View.SubView(at, need);
            DeviceTransfer.Upload(views[k], frameRgba[k].Span.Slice(0, (int)need));
            at += need;
        }
        _stageOut.Ensure(GetMaxAnimationEncodedSize(frames, options));
        int length = await EncodeAnimationAsync(views, frames, canvasWidth, canvasHeight, _stageOut.View, options);
        return await DeviceTransfer.DownloadAsync(_stageOut.View, length);
    }

    /// <summary>
    /// Encode an animation whose frames' RGBA8 pixels are held in JS typed arrays (browser accelerators only). The bytes go to
    /// the GPU with <c>CopyFromJS</c> and the file comes back as a JS <see cref="Uint8Array"/>; none of it enters the .NET heap.
    /// The caller disposes the result.
    /// </summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> EncodeAnimationAsync(
        IReadOnlyList<TypedArray> frameRgba, IReadOnlyList<ImageFrameInfo> frames,
        int canvasWidth, int canvasHeight, WebPAnimationEncodeOptions? options = null)
    {
        ValidateAnimation(frames, canvasWidth, canvasHeight, options);
        ArgumentNullException.ThrowIfNull(frameRgba);
        if (frameRgba.Count != frames.Count)
            throw new ArgumentException($"{frameRgba.Count} pixel buffers for {frames.Count} frames.", nameof(frameRgba));
        long total = 0;
        for (int k = 0; k < frames.Count; k++)
        {
            long need = (long)frames[k].Width * frames[k].Height * 4;
            if (frameRgba[k].ByteLength < need)
                throw new ArgumentException($"frame {k}: rgba must hold {need} bytes for {frames[k].Width}x{frames[k].Height}.", nameof(frameRgba));
            total += need;
        }
        _stageIn.Ensure(total);
        var views = new ArrayView1D<byte, Stride1D.Dense>[frames.Count];
        long at = 0;
        for (int k = 0; k < frames.Count; k++)
        {
            long need = (long)frames[k].Width * frames[k].Height * 4;
            views[k] = _stageIn.View.SubView(at, need);
            DeviceTransfer.Upload(views[k], frameRgba[k]);
            at += need;
        }
        _stageOut.Ensure(GetMaxAnimationEncodedSize(frames, options));
        int length = await EncodeAnimationAsync(views, frames, canvasWidth, canvasHeight, _stageOut.View, options);
        return await DeviceTransfer.DownloadUint8ArrayAsync(_stageOut.View, length);
    }
}
