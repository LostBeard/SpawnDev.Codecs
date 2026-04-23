// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

namespace SpawnDev.Codecs.Audio.Opus;

/// <summary>
/// Configuration for an Opus decoder. Frame size is NOT configured here - Opus decoders
/// accept any packet containing any mix of the RFC 6716 frame durations (2.5 / 5 / 10 / 20
/// / 40 / 60 ms), with the actual duration discovered from each packet's TOC byte.
/// </summary>
public sealed record OpusDecoderConfig
{
    /// <summary>Output sample rate in Hz. Must be one of 8000, 12000, 16000, 24000, or 48000.</summary>
    public required int SampleRateHz { get; init; }

    /// <summary>Output channel count. 1 = mono, 2 = stereo.</summary>
    public required int ChannelCount { get; init; }

    /// <summary>
    /// Validates this config. Throws <see cref="ArgumentOutOfRangeException"/> if any value
    /// is outside the RFC 6716 allowed set. Called by factories before constructing a decoder.
    /// </summary>
    public void Validate()
    {
        if (SampleRateHz is not (8000 or 12000 or 16000 or 24000 or 48000))
        {
            throw new ArgumentOutOfRangeException(
                nameof(SampleRateHz),
                SampleRateHz,
                "Opus sample rate must be 8000, 12000, 16000, 24000, or 48000 Hz.");
        }
        if (ChannelCount is not (1 or 2))
        {
            throw new ArgumentOutOfRangeException(
                nameof(ChannelCount),
                ChannelCount,
                "Opus channel count must be 1 (mono) or 2 (stereo).");
        }
    }
}
