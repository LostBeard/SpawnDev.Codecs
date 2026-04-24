// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 decoder scaffold. Full implementation is scoped for Phase 1c and will
// port libvpx vp9/decoder/ with ILGPU-accelerated MC, inverse transforms,
// and loop filter across all 6 backends.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 decoder. Scaffold only; all frame decode currently throws
/// <see cref="NotImplementedException"/>.
/// </summary>
public sealed class Vp9Decoder : IVideoDecoder
{
    /// <inheritdoc/>
    public VideoCodec Codec => VideoCodec.Vp9;

    /// <inheritdoc/>
    public int Width { get; private set; }

    /// <inheritdoc/>
    public int Height { get; private set; }

    /// <inheritdoc/>
    public ValueTask<int> DecodeFrameAsync(ReadOnlyMemory<byte> compressedPacket, IVideoFrameSink frameSink, CancellationToken ct = default)
    {
        throw new NotImplementedException(
            "VP9 decode is not yet implemented. Phase 1c target: port libvpx vp9/decoder/ " +
            "with ILGPU-accelerated motion compensation, inverse ADST/DCT, and loop filter. " +
            "VP9 super-frames will be unpacked at this layer before per-subframe decode.");
    }

    /// <inheritdoc/>
    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}
