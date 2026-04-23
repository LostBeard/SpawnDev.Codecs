// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

namespace SpawnDev.Codecs.Audio;

/// <summary>
/// Decodes a compressed audio packet to PCM samples. Stateful (holds codec state across
/// packets), not thread-safe - create one instance per stream.
/// </summary>
public interface IAudioDecoder : IAsyncDisposable
{
    /// <summary>Which codec this decoder implements.</summary>
    AudioCodec Codec { get; }

    /// <summary>Output sample rate in Hz. Opus supports 8000, 12000, 16000, 24000, or 48000.</summary>
    int SampleRateHz { get; }

    /// <summary>Output channel count. 1 = mono, 2 = stereo.</summary>
    int ChannelCount { get; }

    /// <summary>
    /// Decodes one compressed packet into PCM samples (float, range -1.0 to +1.0).
    /// Caller sizes <paramref name="pcmOutput"/> to hold the maximum possible samples for the
    /// packet's codec at the configured sample rate (for Opus at 48 kHz, 5760 samples * channels
    /// covers the 120 ms maximum packet duration).
    /// </summary>
    /// <returns>Number of decoded sample frames per channel (not interleaved count, not bytes).</returns>
    ValueTask<int> DecodePacketAsync(
        ReadOnlyMemory<byte> compressedPacket,
        Memory<float> pcmOutput,
        CancellationToken ct = default);

    /// <summary>
    /// Convenience overload for 16-bit signed PCM (WebRTC RTP native, capture device native).
    /// Equivalent to the float path with an internal narrow pass.
    /// </summary>
    /// <returns>Number of decoded sample frames per channel.</returns>
    ValueTask<int> DecodePacketAsync(
        ReadOnlyMemory<byte> compressedPacket,
        Memory<short> pcmOutput,
        CancellationToken ct = default);
}
