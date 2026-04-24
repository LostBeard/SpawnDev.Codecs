// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 decoder scaffold. Full implementation will take multiple phases and
// will follow dav1d's structure (OBU-driven), with ILGPU-accelerated
// inter/intra prediction, inverse transforms, CDEF, loop restoration, and
// film-grain synthesis across all 6 backends. The unique value here is a
// pure-.NET patent-clean AV1 decoder that runs in Blazor WASM.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// AV1 decoder. Scaffold only; full implementation spans multiple phases
/// (entropy decode, inter/intra prediction, IT, CDEF, loop restoration,
/// film-grain synthesis). All frame decode currently throws
/// <see cref="NotImplementedException"/>.
/// </summary>
public sealed class Av1Decoder : IVideoDecoder
{
    /// <inheritdoc/>
    public VideoCodec Codec => VideoCodec.Av1;

    /// <inheritdoc/>
    public int Width { get; private set; }

    /// <inheritdoc/>
    public int Height { get; private set; }

    /// <inheritdoc/>
    public ValueTask<int> DecodeFrameAsync(ReadOnlyMemory<byte> compressedPacket, IVideoFrameSink frameSink, CancellationToken ct = default)
    {
        throw new NotImplementedException(
            "AV1 decode is not yet implemented. Scoped across multiple phases: entropy " +
            "decode (OBU-driven per dav1d), inter/intra prediction, inverse transforms, " +
            "CDEF, loop restoration, film-grain synthesis - all ILGPU-accelerated across " +
            "the 6 backends. Pure-.NET AV1 decoder in Blazor WASM is the unique value.");
    }

    /// <inheritdoc/>
    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}
