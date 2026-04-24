// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 decoder scaffold. Full implementation is scoped for Phase 1b and will
// port libvpx's vp8/decoder/ with ILGPU-accelerated intra prediction,
// inverse transforms, and loop filter.

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// VP8 decoder (RFC 6386). Scaffolding only at this point; all frame decode
/// currently throws <see cref="NotImplementedException"/> with a descriptive
/// message. Construction is intentionally cheap so callers can route by
/// codec enum without catching.
/// </summary>
public sealed class Vp8Decoder : IVideoDecoder
{
    /// <inheritdoc/>
    public VideoCodec Codec => VideoCodec.Vp8;

    /// <inheritdoc/>
    public int Width { get; private set; }

    /// <inheritdoc/>
    public int Height { get; private set; }

    /// <inheritdoc/>
    public ValueTask<int> DecodeFrameAsync(ReadOnlyMemory<byte> compressedPacket, IVideoFrameSink frameSink, CancellationToken ct = default)
    {
        throw new NotImplementedException(
            "VP8 decode is not yet implemented. Phase 1b target: port libvpx vp8/decoder/ " +
            "with ILGPU-accelerated intra prediction, inverse DCT, and loop filter. " +
            "The rest of SpawnDev.Codecs (audio codecs + containers) is usable today.");
    }

    /// <inheritdoc/>
    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}
