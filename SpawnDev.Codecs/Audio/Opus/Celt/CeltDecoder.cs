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
    /// <see cref="NotImplementedException"/> with a descriptive message. Will land
    /// in a future slice as a multi-step port: range-coded bit allocation decode +
    /// PVQ decode + IMDCT (ILGPU-accelerated) + post-filter + deemphasis.
    /// </summary>
    public int DecodeFrame(ReadOnlySpan<byte> payload, Span<float> pcmOut, int channels)
    {
        _ = payload;
        _ = pcmOut;
        _ = channels;
        throw new NotImplementedException(
            $"CELT decode (Mode={_mode.FrameSize}-sample frame, EndBand={_mode.EndBand}) is not yet implemented. " +
            "Phase 1b target: port libopus celt/celt_decoder.c with ILGPU-accelerated IMDCT. " +
            "SILK-mode Opus packets ARE supported via OpusDecoder.");
    }
}
