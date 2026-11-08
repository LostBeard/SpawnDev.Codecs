// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Animated lossy WebP: libwebp's WebPAnimEncoder (WebPAnimEncoder.cs) with this encoder's still pictures as frames, as
// Pillow writes `save(save_all=True, lossless=False, quality, method, exif)` (ComfyUI's Save Animated WEBP with
// lossless off). Every frame's VP8 chunk is byte-identical with libwebp's WebPEncode; a frame with transparent pixels
// (flattened blocks of a blended frame, or a transparent source) carries an ALPH chunk coded by our VP8L coder, which
// decodes to the same alpha plane but can have another size, and libwebp picks key frames by size.

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Images.WebP.Lossy;

public sealed partial class WebPLossyEncoderGpu
{
    private WebPAnimEncoder? _animEncoder;

    private WebPAnimEncoder Anim => _animEncoder ??= new WebPAnimEncoder(_accelerator);

    /// <summary>A frame of the animation: this encoder's still file (loop filter off when blended, as libwebp).</summary>
    private AnimFrameEncoder FrameEncoder(WebPLossyEncodeOptions o) => async (rgba, width, height, blend, file) =>
    {
        var config = blend ? o with { FilterStrength = 0 } : o;
        var (simpleLength, alphaLength) = await EncodeToScratchAsync(rgba, width, height, config);
        int length = FileLength(simpleLength, alphaLength);
        var view = await file.EnsureDrainedAsync(length);
        Assemble(view, length, alphaLength);
        return new AnimEncoded(length, alphaLength >= 0, alphaLength >= 0 ? 30 : 12);
    };

    private static void ValidateAnimation(int frameCount, int width, int height, IReadOnlyList<int> durationsMs, WebPLossyEncodeOptions o)
    {
        WebPAnimEncoder.ValidateDurations(frameCount, durationsMs);
        if (o.IccProfile != null || o.Exif != null || o.Xmp != null)
            throw new ArgumentException("An animation takes its ICC / EXIF / XMP from WebPAnimEncoderOptions, not the per-frame options.");
        Validate(width, height, o);
    }

    /// <summary>
    /// Encode an animation whose frames (full-canvas RGBA8, straight alpha) are on the accelerator, as libwebp's
    /// WebPAnimEncoder does for Pillow's animated lossy save. The file goes to <paramref name="output"/>; returns its length.
    /// Throws <see cref="ArgumentException"/> naming the needed size when <paramref name="output"/> is too small.
    /// </summary>
    /// <param name="frames">Per frame, <paramref name="width"/> x <paramref name="height"/> x 4 bytes of RGBA8 (rows top to bottom).</param>
    /// <param name="width">Canvas width (1..16383).</param>
    /// <param name="height">Canvas height (1..16383).</param>
    /// <param name="durationsMs">Per frame, how long it shows (0..2^24 - 1 ms).</param>
    /// <param name="output">Receives the WebP file.</param>
    /// <param name="options">Lossy settings of every frame (quality, method; Pillow: quality 80, method as given).</param>
    /// <param name="animation">Animation settings and metadata chunks (null = Pillow's defaults, no metadata).</param>
    public async Task<int> EncodeAnimationAsync(
        IReadOnlyList<ArrayView1D<byte, Stride1D.Dense>> frames, int width, int height, IReadOnlyList<int> durationsMs,
        ArrayView1D<byte, Stride1D.Dense> output, WebPLossyEncodeOptions? options = null, WebPAnimEncoderOptions? animation = null)
    {
        ArgumentNullException.ThrowIfNull(frames);
        var o = options ?? new WebPLossyEncodeOptions();
        ValidateAnimation(frames.Count, width, height, durationsMs, o);
        long need = (long)width * height * 4;
        for (int k = 0; k < frames.Count; k++)
            if (frames[k].Length < need) throw new ArgumentException($"frame {k}: must hold {need} bytes.", nameof(frames));
        var sink = WebPAnimEncoder.Sink.Fixed(Anim, output);
        int length = await Anim.EncodeAsync(frames.Count, k => new ValueTask<ArrayView1D<byte, Stride1D.Dense>>(frames[k]),
            width, height, durationsMs, false, o.Quality, FrameEncoder(o), animation ?? new WebPAnimEncoderOptions(), sink);
        if (sink.Overflowed) throw new ArgumentException($"output must hold {length} bytes.", nameof(output));
        return length;
    }

    /// <summary>
    /// Encode an animation of managed RGBA8 frames (full canvas each). Returns the file as a managed array (a terminal CPU
    /// sink; in a browser prefer the <see cref="TypedArray"/> overload). Frames are uploaded one at a time.
    /// </summary>
    public async Task<byte[]> EncodeAnimationAsync(
        IReadOnlyList<ReadOnlyMemory<byte>> frames, int width, int height, IReadOnlyList<int> durationsMs,
        WebPLossyEncodeOptions? options = null, WebPAnimEncoderOptions? animation = null)
    {
        ArgumentNullException.ThrowIfNull(frames);
        var o = options ?? new WebPLossyEncodeOptions();
        ValidateAnimation(frames.Count, width, height, durationsMs, o);
        int need = width * height * 4;
        for (int k = 0; k < frames.Count; k++)
            if (frames[k].Length < need) throw new ArgumentException($"frame {k}: must hold {need} bytes.", nameof(frames));
        var anim = Anim;
        var input = await anim.FrameIn.EnsureDrainedAsync(need);
        int length = await anim.EncodeAsync(frames.Count, async k =>
        {
            await _accelerator.SynchronizeAsync();   // the previous frame's kernels are done with the input buffer
            DeviceTransfer.Upload(input.SubView(0, need), frames[k].Span.Slice(0, need));
            return input;
        }, width, height, durationsMs, false, o.Quality, FrameEncoder(o), animation ?? new WebPAnimEncoderOptions(), WebPAnimEncoder.Sink.Growing(anim));
        return await DeviceTransfer.DownloadAsync(anim.Output, length);
    }

    /// <summary>Encode an animation of RGBA8 frames held in JS typed arrays (browser accelerators); the file comes back as a JS Uint8Array.</summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> EncodeAnimationAsync(
        IReadOnlyList<TypedArray> frames, int width, int height, IReadOnlyList<int> durationsMs,
        WebPLossyEncodeOptions? options = null, WebPAnimEncoderOptions? animation = null)
    {
        ArgumentNullException.ThrowIfNull(frames);
        var o = options ?? new WebPLossyEncodeOptions();
        ValidateAnimation(frames.Count, width, height, durationsMs, o);
        int need = width * height * 4;
        for (int k = 0; k < frames.Count; k++)
            if (frames[k] == null || frames[k].ByteLength < need) throw new ArgumentException($"frame {k}: must hold {need} bytes.", nameof(frames));
        var anim = Anim;
        var input = await anim.FrameIn.EnsureDrainedAsync(need);
        int length = await anim.EncodeAsync(frames.Count, async k =>
        {
            await _accelerator.SynchronizeAsync();
            DeviceTransfer.Upload(input.SubView(0, need), frames[k]);
            return input;
        }, width, height, durationsMs, false, o.Quality, FrameEncoder(o), animation ?? new WebPAnimEncoderOptions(), WebPAnimEncoder.Sink.Growing(anim));
        return await DeviceTransfer.DownloadUint8ArrayAsync(anim.Output, length);
    }
}
