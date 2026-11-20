// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Upper bounds on the lossy encoder's files, for sizing caller-owned output buffers, and the animation entry points that
// hand back the file at its exact length instead (no worst-case buffer: SpawnGen's 121-frame 1280x704 animations, 2026-11-19).
//
// The bound of one frame, from what the encoder can write:
//   VP8 chunk  = 10 (frame header) + partition 0 (its capacity: 4096 + 64 per macroblock, the encoder fails beyond it)
//              + the token partition: every token codes one bit at a probability of at least 1/256, so at most a byte;
//                tokens per macroblock at most 25 blocks x (1 + 16 coefficients x 19), from VP8RecordCoeffTokens
//                (WebPLossyEncodeRowKernel.RecordCoeffTokens: 5 tree decisions, 2 category bits, 11 extra bits of
//                category 6, the sign and the end-of-block bit) + 1024 bytes of coder slack;
//   ALPH chunk = 1 + width x height (WebPLosslessEncoderGpu.GetMaxAlphaEncodedSize: a longer stream is stored raw).
// Real files are a small fraction of this (the bound is a guarantee, not an estimate).

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Images.WebP.Lossy;

public sealed partial class WebPLossyEncoderGpu
{
    /// <summary>Tokens one macroblock can record at most (25 blocks x (1 + 16 x 19); see the file header).</summary>
    private const long MaxTokensPerMacroblock = 25L * (1 + 16 * 19);

    // the VP8 chunk's payload at most, and the ALPH chunk's payload at most
    private static long MaxVp8Payload(int width, int height)
    {
        long mbs = (long)((width + 15) >> 4) * ((height + 15) >> 4);
        return 10 + (4096 + mbs * 64) + mbs * MaxTokensPerMacroblock + 1024;
    }

    private static long Chunk(long payload) => 8 + payload + (payload & 1);

    private static long MetadataBound(byte[]? icc, byte[]? exif, byte[]? xmp) =>
        (icc != null ? Chunk(icc.Length) : 0) + (exif != null ? Chunk(exif.Length) : 0) + (xmp != null ? Chunk(xmp.Length) : 0);

    /// <summary>
    /// Upper bound on the size of a still lossy file of <paramref name="width"/> x <paramref name="height"/> (with an ALPH
    /// chunk and the metadata of <paramref name="options"/>), for sizing the output of
    /// <see cref="EncodeAsync(ArrayView1D{byte, Stride1D.Dense}, int, int, ArrayView1D{byte, Stride1D.Dense}, WebPLossyEncodeOptions?)"/>.
    /// A guarantee from the encoder's limits, far above real files.
    /// </summary>
    public static long GetMaxEncodedSize(int width, int height, WebPLossyEncodeOptions? options = null)
    {
        if (width <= 0 || width > 16383) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0 || height > 16383) throw new ArgumentOutOfRangeException(nameof(height));
        return 12 + Chunk(10) + Chunk(1 + (long)width * height) + Chunk(MaxVp8Payload(width, height))
            + MetadataBound(options?.IccProfile, options?.Exif, options?.Xmp);
    }

    /// <summary>
    /// Upper bound on the size of a lossy animation of <paramref name="frameCount"/> frames on a
    /// <paramref name="width"/> x <paramref name="height"/> canvas (every frame full size with an ALPH chunk, the metadata
    /// of <paramref name="animation"/>), for sizing the output of
    /// <see cref="EncodeAnimationAsync(IReadOnlyList{ArrayView1D{byte, Stride1D.Dense}}, int, int, IReadOnlyList{int}, ArrayView1D{byte, Stride1D.Dense}, WebPLossyEncodeOptions?, WebPAnimEncoderOptions?)"/>.
    /// A guarantee, far above real files: to get the file at its exact size instead, use
    /// <see cref="EncodeAnimationToDeviceAsync"/> or, in a browser, <see cref="EncodeAnimationToUint8ArrayAsync"/>.
    /// </summary>
    public static long GetMaxAnimationEncodedSize(int frameCount, int width, int height, WebPLossyEncodeOptions? options = null,
        WebPAnimEncoderOptions? animation = null)
    {
        if (frameCount < 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
        if (width <= 0 || width > 16383) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0 || height > 16383) throw new ArgumentOutOfRangeException(nameof(height));
        long frame = Chunk(16 + Chunk(1 + (long)width * height) + Chunk(MaxVp8Payload(width, height)));   // ANMF
        return 12 + Chunk(10) + Chunk(6) + frameCount * frame
            + MetadataBound(animation?.IccProfile, animation?.Exif, animation?.Xmp);
    }

    /// <summary>
    /// Encode an animation whose frames are on the accelerator (as the ArrayView1D overload) and hand the file back where
    /// the encoder built it: a view of exactly the file's length into a buffer this encoder owns, valid until its next
    /// encode or its disposal. No worst-case output buffer: the file's own size is all that is held.
    /// </summary>
    public async Task<ArrayView1D<byte, Stride1D.Dense>> EncodeAnimationToDeviceAsync(
        IReadOnlyList<ArrayView1D<byte, Stride1D.Dense>> frames, int width, int height, IReadOnlyList<int> durationsMs,
        WebPLossyEncodeOptions? options = null, WebPAnimEncoderOptions? animation = null)
    {
        ArgumentNullException.ThrowIfNull(frames);
        var o = options ?? new WebPLossyEncodeOptions();
        ValidateAnimation(frames.Count, width, height, durationsMs, o);
        long need = (long)width * height * 4;
        for (int k = 0; k < frames.Count; k++)
            if (frames[k].Length < need) throw new ArgumentException($"frame {k}: must hold {need} bytes.", nameof(frames));
        var anim = Anim;
        int length = await anim.EncodeAsync(frames.Count, k => new ValueTask<ArrayView1D<byte, Stride1D.Dense>>(frames[k]),
            width, height, durationsMs, false, o.Quality, FrameEncoder(o), animation ?? new WebPAnimEncoderOptions(), WebPAnimEncoder.Sink.Growing(anim));
        return anim.Output.SubView(0, length);
    }

    /// <summary>
    /// Encode an animation whose frames are on the accelerator; the file comes back as a JS Uint8Array of exactly its
    /// length (browser accelerators; the bytes never enter the .NET heap). The caller disposes the result.
    /// </summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> EncodeAnimationToUint8ArrayAsync(
        IReadOnlyList<ArrayView1D<byte, Stride1D.Dense>> frames, int width, int height, IReadOnlyList<int> durationsMs,
        WebPLossyEncodeOptions? options = null, WebPAnimEncoderOptions? animation = null)
    {
        var file = await EncodeAnimationToDeviceAsync(frames, width, height, durationsMs, options, animation);
        return await DeviceTransfer.DownloadUint8ArrayAsync(file, file.Length);
    }
}
