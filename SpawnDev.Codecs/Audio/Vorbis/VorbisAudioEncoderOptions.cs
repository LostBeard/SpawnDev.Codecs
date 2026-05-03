// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// Configuration record for Vorbis encoders. Consumed by both the GPU
/// encoder (<c>VorbisAudioEncoderGpu</c>, main library) and the CPU
/// reference encoder (<c>VorbisAudioEncoder</c>, in
/// <c>SpawnDev.Codecs.References</c>).
/// </summary>
public sealed record VorbisAudioEncoderOptions
{
    /// <summary>Audio sample rate in Hz.</summary>
    public int SampleRateHz { get; init; } = 44100;

    /// <summary>Channel count. Currently only 1 (mono) is supported.</summary>
    public int Channels { get; init; } = 1;

    /// <summary>
    /// Block size in samples (power of 2 in [64, 8192]). Same value used for
    /// both blocksize_0 and blocksize_1; the encoder emits short-block-only
    /// audio packets with no transition windows. Default 1024 matches what
    /// libvorbis uses by default for the short block size.
    /// </summary>
    public int BlockSize { get; init; } = 1024;
}
