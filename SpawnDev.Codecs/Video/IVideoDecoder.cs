// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

namespace SpawnDev.Codecs.Video;

/// <summary>
/// Decodes a compressed video frame (or super-frame packet) into a planar YUV
/// image. Stateful - holds codec state, reference frames, and probability
/// tables across packets. Create one instance per stream.
/// </summary>
public interface IVideoDecoder : IAsyncDisposable
{
    /// <summary>Which codec this decoder implements.</summary>
    VideoCodec Codec { get; }

    /// <summary>Output frame width in pixels (reported after the first keyframe).</summary>
    int Width { get; }

    /// <summary>Output frame height in pixels.</summary>
    int Height { get; }

    /// <summary>
    /// Decode one compressed packet. The decoder may emit zero, one, or
    /// several frames per packet (VP9 super-frames, AV1 temporal unit with
    /// multiple OBUs, etc.). Returns the number of decoded frames.
    /// </summary>
    ValueTask<int> DecodeFrameAsync(
        ReadOnlyMemory<byte> compressedPacket,
        IVideoFrameSink frameSink,
        CancellationToken ct = default);
}

/// <summary>
/// Callback invoked once per decoded frame. Implementations copy Y/U/V planes
/// out of the decoder's reference pool into whatever target they need
/// (texture upload, WebGPU canvas, file write, etc.).
/// </summary>
public interface IVideoFrameSink
{
    /// <summary>Called once per decoded frame, in display order.</summary>
    /// <param name="yPlane">Luma plane, row-major, length = width * height.</param>
    /// <param name="yStride">Luma row stride in bytes.</param>
    /// <param name="uPlane">Chroma-U plane; for 4:2:0 this is half-width and half-height.</param>
    /// <param name="uStride">Chroma-U row stride.</param>
    /// <param name="vPlane">Chroma-V plane.</param>
    /// <param name="vStride">Chroma-V row stride.</param>
    /// <param name="pts">Presentation timestamp in codec-defined units (packet-supplied).</param>
    ValueTask OnFrameAsync(
        ReadOnlyMemory<byte> yPlane, int yStride,
        ReadOnlyMemory<byte> uPlane, int uStride,
        ReadOnlyMemory<byte> vPlane, int vStride,
        long pts);
}
