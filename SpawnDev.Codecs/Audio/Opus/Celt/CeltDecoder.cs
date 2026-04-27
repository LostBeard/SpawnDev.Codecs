// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
// See NOTICE.md for upstream attributions.

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>
/// CELT decoder. Not yet implemented - Phase 1b target with ILGPU-accelerated
/// IMDCT kernels. This class exists so callers can construct a CELT decoder
/// and get a clear <see cref="NotImplementedException"/> when they attempt to
/// decode a CELT-mode Opus frame, rather than a cryptic crash deeper in the
/// stack.
/// </summary>
public sealed class CeltDecoder
{
    private readonly CeltMode _mode;

    /// <summary>Create a CELT decoder for the given mode.</summary>
    public CeltDecoder(CeltMode mode)
    {
        _mode = mode ?? throw new ArgumentNullException(nameof(mode));
    }

    /// <summary>Audio sample rate in Hz (48000 for all Opus CELT modes).</summary>
    public int SampleRateHz => _mode.SampleRateHz;

    /// <summary>Frame size in samples at <see cref="SampleRateHz"/>.</summary>
    public int FrameSize => _mode.FrameSize;

    /// <summary>CELT band count used for this decoder.</summary>
    public int EndBand => _mode.EndBand;

    /// <summary>
    /// Decode a CELT-mode Opus frame into PCM. Not yet implemented. Throws
    /// <see cref="NotImplementedException"/> with a descriptive message.
    ///
    /// Implementation scope (per RFC 6716 sec 4.3 / libopus celt/celt_decoder.c):
    ///   1. Silence flag + post-filter parameters (range-coded)
    ///   2. Coarse band energy (Q8 prediction-coded)
    ///   3. Tf change flags + spread decision
    ///   4. Anti-collapse seed
    ///   5. Bit allocation derivation
    ///   6. Fine band energy
    ///   7. PVQ shape decode per band (codebook search inverse)
    ///   8. Stereo de-coupling (when applicable)
    ///   9. Inverse pitch prefilter / postfilter
    ///   10. IMDCT per short-block (2.5 ms or 5 ms units)
    ///   11. Window overlap-add into the synthesis buffer
    ///   12. Deemphasis filter
    ///
    /// This is on the order of 2000 lines of carefully-ported code from
    /// libopus and not feasible as a single drop-in. SILK-mode Opus packets
    /// ARE supported via <see cref="OpusDecoder"/> for now.
    /// </summary>
    public int DecodeFrame(ReadOnlySpan<byte> payload, Span<float> pcmOut, int channels)
    {
        _ = payload;
        _ = pcmOut;
        _ = channels;
        throw new NotImplementedException(
            $"CELT decode (Mode={_mode.FrameSize}-sample frame, EndBand={_mode.EndBand}) is not yet implemented. " +
            "Requires the full RFC 6716 sec 4.3 pipeline: coarse/fine energy, PVQ shape, " +
            "stereo de-coupling, pitch postfilter, IMDCT, and deemphasis. " +
            "SILK-mode Opus packets ARE supported via OpusDecoder.");
    }
}
