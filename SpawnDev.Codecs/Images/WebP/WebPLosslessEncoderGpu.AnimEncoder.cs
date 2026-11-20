// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Animated lossless WebP the way libwebp's WebPAnimEncoder writes it (WebPAnimEncoder.cs): full-canvas frames in,
// change rectangles, disposal, blending with IncreaseTransparency, dropped repeats and key frames out - what Pillow's
// `save(save_all=True, lossless=True, quality, method, exif)` does (ComfyUI's Save Animated WEBP default). Every frame
// is this encoder's lossless still, so the pixels are exact; the VP8L bytes are our coder's, not libwebp's, so the file
// is not byte-identical with Pillow's and libwebp's size-based key-frame choice can differ.
//
// The caller-rectangle EncodeAnimationAsync overloads (WebPLosslessEncoderGpu.Animation.cs) are unchanged: those write
// exactly the frames given.

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Images.WebP;

public sealed partial class WebPLosslessEncoderGpu
{
    private WebPAnimEncoder? _animEncoder;

    private WebPAnimEncoder AnimEncoder => _animEncoder ??= new WebPAnimEncoder(_accelerator);

    /// <summary>A frame of the animation: this encoder's lossless still file (RIFF + VP8L).</summary>
    private AnimFrameEncoder LosslessFrameEncoder(WebPEncodeOptions o) => async (rgba, width, height, blend, file) =>
    {
        var view = await file.EnsureDrainedAsync(GetMaxEncodedSize(width, height, o));
        var (length, hasAlpha) = await EncodeCoreAsync(rgba, width, height, view, o);
        return new AnimEncoded(length, hasAlpha, 12);
    };

    private static void ValidateAnimEncode(int frameCount, int width, int height, IReadOnlyList<int> durationsMs, WebPEncodeOptions o)
    {
        WebPAnimEncoder.ValidateDurations(frameCount, durationsMs);
        ValidateSize(width, height);
        if (o.Quality < 0 || o.Quality > 100) throw new ArgumentOutOfRangeException(nameof(o), "Quality must be 0..100.");
        if (o.IccProfile != null || o.Exif != null || o.Xmp != null)
            throw new ArgumentException("An animation takes its ICC / EXIF / XMP from WebPAnimEncoderOptions, not the per-frame options.");
    }

    /// <summary>
    /// Encode an animation whose frames (full-canvas RGBA8, straight alpha) are on the accelerator, as libwebp's
    /// WebPAnimEncoder does for Pillow's animated lossless save; for Pillow's behaviour pass
    /// <c>Exact = false</c> (libwebp's default: RGB under fully transparent pixels may be zeroed). The file goes to
    /// <paramref name="output"/>; returns its length. Throws <see cref="ArgumentException"/> naming the needed size when
    /// <paramref name="output"/> is too small.
    /// </summary>
    public async Task<int> EncodeAnimationAsync(
        IReadOnlyList<ArrayView1D<byte, Stride1D.Dense>> frames, int width, int height, IReadOnlyList<int> durationsMs,
        ArrayView1D<byte, Stride1D.Dense> output, WebPEncodeOptions? options = null, WebPAnimEncoderOptions? animation = null)
    {
        ArgumentNullException.ThrowIfNull(frames);
        var o = options ?? new WebPEncodeOptions();
        ValidateAnimEncode(frames.Count, width, height, durationsMs, o);
        long need = (long)width * height * 4;
        for (int k = 0; k < frames.Count; k++)
            if (frames[k].Length < need) throw new ArgumentException($"frame {k}: must hold {need} bytes.", nameof(frames));
        var anim = AnimEncoder;
        var sink = WebPAnimEncoder.Sink.Fixed(anim, output);
        int length = await anim.EncodeAsync(frames.Count, k => new ValueTask<ArrayView1D<byte, Stride1D.Dense>>(frames[k]),
            width, height, durationsMs, true, o.Quality, LosslessFrameEncoder(o), animation ?? new WebPAnimEncoderOptions(), sink);
        if (sink.Overflowed) throw new ArgumentException($"output must hold {length} bytes.", nameof(output));
        return length;
    }

    /// <summary>
    /// Upper bound on the size of a WebPAnimEncoder animation of <paramref name="frameCount"/> frames on a
    /// <paramref name="width"/> x <paramref name="height"/> canvas (every frame full size, the metadata of
    /// <paramref name="animation"/>), for sizing the output of
    /// <see cref="EncodeAnimationAsync(IReadOnlyList{ArrayView1D{byte, Stride1D.Dense}}, int, int, IReadOnlyList{int}, ArrayView1D{byte, Stride1D.Dense}, WebPEncodeOptions?, WebPAnimEncoderOptions?)"/>.
    /// A guarantee from the encoder's limits, far above real files: to get the file at its exact size instead, use
    /// <see cref="EncodeAnimationToDeviceAsync"/> or, in a browser, <see cref="EncodeAnimationToUint8ArrayAsync"/>.
    /// </summary>
    public static long GetMaxAnimationEncodedSize(int frameCount, int width, int height, WebPEncodeOptions? options = null,
        WebPAnimEncoderOptions? animation = null)
    {
        if (frameCount < 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
        // a frame is a still file's chunks inside an ANMF chunk (header 8 + 16); a still file at most GetMaxEncodedSize
        long frame = 8 + 16 + GetMaxEncodedSize(width, height) + 1;
        long size = 12 + 18 + 14 + frameCount * frame;
        if (animation?.IccProfile is { } icc) size += 8 + icc.Length + 1;
        if (animation?.Exif is { } exif) size += 8 + exif.Length + 1;
        if (animation?.Xmp is { } xmp) size += 8 + xmp.Length + 1;
        return size;
    }

    /// <summary>
    /// Encode an animation whose frames are on the accelerator (as the ArrayView1D overload) and hand the file back where
    /// the encoder built it: a view of exactly the file's length into a buffer this encoder owns, valid until its next
    /// encode or its disposal. No worst-case output buffer: the file's own size is all that is held.
    /// </summary>
    public async Task<ArrayView1D<byte, Stride1D.Dense>> EncodeAnimationToDeviceAsync(
        IReadOnlyList<ArrayView1D<byte, Stride1D.Dense>> frames, int width, int height, IReadOnlyList<int> durationsMs,
        WebPEncodeOptions? options = null, WebPAnimEncoderOptions? animation = null)
    {
        ArgumentNullException.ThrowIfNull(frames);
        var o = options ?? new WebPEncodeOptions();
        ValidateAnimEncode(frames.Count, width, height, durationsMs, o);
        long need = (long)width * height * 4;
        for (int k = 0; k < frames.Count; k++)
            if (frames[k].Length < need) throw new ArgumentException($"frame {k}: must hold {need} bytes.", nameof(frames));
        var anim = AnimEncoder;
        int length = await anim.EncodeAsync(frames.Count, k => new ValueTask<ArrayView1D<byte, Stride1D.Dense>>(frames[k]),
            width, height, durationsMs, true, o.Quality, LosslessFrameEncoder(o), animation ?? new WebPAnimEncoderOptions(), WebPAnimEncoder.Sink.Growing(anim));
        return anim.Output.SubView(0, length);
    }

    /// <summary>
    /// Encode an animation whose frames are on the accelerator; the file comes back as a JS Uint8Array of exactly its
    /// length (browser accelerators; the bytes never enter the .NET heap). The caller disposes the result.
    /// </summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> EncodeAnimationToUint8ArrayAsync(
        IReadOnlyList<ArrayView1D<byte, Stride1D.Dense>> frames, int width, int height, IReadOnlyList<int> durationsMs,
        WebPEncodeOptions? options = null, WebPAnimEncoderOptions? animation = null)
    {
        var file = await EncodeAnimationToDeviceAsync(frames, width, height, durationsMs, options, animation);
        return await DeviceTransfer.DownloadUint8ArrayAsync(file, file.Length);
    }

    /// <summary>
    /// Encode an animation of managed RGBA8 frames (full canvas each) like libwebp's WebPAnimEncoder (see the ArrayView1D
    /// overload). Returns the file as a managed array (a terminal CPU sink). Frames are uploaded one at a time.
    /// </summary>
    public async Task<byte[]> EncodeAnimationAsync(
        IReadOnlyList<ReadOnlyMemory<byte>> frames, int width, int height, IReadOnlyList<int> durationsMs,
        WebPEncodeOptions? options = null, WebPAnimEncoderOptions? animation = null)
    {
        ArgumentNullException.ThrowIfNull(frames);
        var o = options ?? new WebPEncodeOptions();
        ValidateAnimEncode(frames.Count, width, height, durationsMs, o);
        int need = width * height * 4;
        for (int k = 0; k < frames.Count; k++)
            if (frames[k].Length < need) throw new ArgumentException($"frame {k}: must hold {need} bytes.", nameof(frames));
        var anim = AnimEncoder;
        var input = await anim.FrameIn.EnsureDrainedAsync(need);
        int length = await anim.EncodeAsync(frames.Count, async k =>
        {
            await _accelerator.SynchronizeAsync();   // the previous frame's kernels are done with the input buffer
            DeviceTransfer.Upload(input.SubView(0, need), frames[k].Span.Slice(0, need));
            return input;
        }, width, height, durationsMs, true, o.Quality, LosslessFrameEncoder(o), animation ?? new WebPAnimEncoderOptions(), WebPAnimEncoder.Sink.Growing(anim));
        return await DeviceTransfer.DownloadAsync(anim.Output, length);
    }

    /// <summary>Encode an animation of RGBA8 frames held in JS typed arrays (browser accelerators); the file comes back as a JS Uint8Array.</summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> EncodeAnimationAsync(
        IReadOnlyList<TypedArray> frames, int width, int height, IReadOnlyList<int> durationsMs,
        WebPEncodeOptions? options = null, WebPAnimEncoderOptions? animation = null)
    {
        ArgumentNullException.ThrowIfNull(frames);
        var o = options ?? new WebPEncodeOptions();
        ValidateAnimEncode(frames.Count, width, height, durationsMs, o);
        int need = width * height * 4;
        for (int k = 0; k < frames.Count; k++)
            if (frames[k] == null || frames[k].ByteLength < need) throw new ArgumentException($"frame {k}: must hold {need} bytes.", nameof(frames));
        var anim = AnimEncoder;
        var input = await anim.FrameIn.EnsureDrainedAsync(need);
        int length = await anim.EncodeAsync(frames.Count, async k =>
        {
            await _accelerator.SynchronizeAsync();
            DeviceTransfer.Upload(input.SubView(0, need), frames[k]);
            return input;
        }, width, height, durationsMs, true, o.Quality, LosslessFrameEncoder(o), animation ?? new WebPAnimEncoderOptions(), WebPAnimEncoder.Sink.Growing(anim));
        return await DeviceTransfer.DownloadUint8ArrayAsync(anim.Output, length);
    }
}
