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

    /// <summary>
    /// Verify the STREAMINFO MD5 signature against the decoded samples. Returns
    /// <c>true</c> if the signatures match (integrity preserved) or the stored
    /// signature is all-zero (the encoder chose not to compute it). Returns
    /// <c>false</c> when the stored signature is non-zero AND does not match.
    /// </summary>
    public bool VerifyMd5()
    {
        bool allZero = true;
        for (int i = 0; i < 16; i++)
        {
            if (StreamInfo.Md5Signature[i] != 0) { allZero = false; break; }
        }
        if (allZero) return true; // not computed
        byte[] recomputed = FlacMd5.Compute(InterleavedSamples, StreamInfo.BitsPerSample);
        for (int i = 0; i < 16; i++)
        {
            if (recomputed[i] != StreamInfo.Md5Signature[i]) return false;
        }
        return true;
    }
}
