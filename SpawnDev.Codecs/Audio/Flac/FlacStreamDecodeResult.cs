// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// Whole-stream FLAC decode result: parsed STREAMINFO plus interleaved PCM
/// samples concatenated across every frame in the stream.
/// </summary>
public sealed record FlacStreamDecodeResult
{
    /// <summary>STREAMINFO metadata block parsed from the stream prelude.</summary>
    public required FlacStreamInfo StreamInfo { get; init; }

    /// <summary>
    /// Fully-decoded PCM samples interleaved across channels:
    /// <c>[ch0[0], ch1[0], ch0[1], ch1[1], ...]</c>. Length equals
    /// <see cref="TotalSamplesPerChannel"/> × <see cref="FlacStreamInfo.Channels"/>.
    /// </summary>
    public required int[] InterleavedSamples { get; init; }

    /// <summary>Total decoded samples per channel summed across all frames.</summary>
    public int TotalSamplesPerChannel { get; init; }
}
