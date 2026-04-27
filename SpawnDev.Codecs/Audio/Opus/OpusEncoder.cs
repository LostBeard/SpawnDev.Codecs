// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Top-level Opus encoder. Routes per-frame encode requests to the SILK or
// CELT subsystem based on the configured mode. Mirrors the structure of
// OpusDecoder so consumers can encode/decode through symmetric APIs.
//
// State (Phase 1a): the SILK and CELT encoder paths are not yet implemented;
// each throws NotImplementedException with a descriptive message. Packet
// framing (TOC byte, frame count, padding) is wired so once a per-frame
// encoder lands the top-level structure can be reused.

namespace SpawnDev.Codecs.Audio.Opus;

/// <summary>
/// Configuration for an <see cref="OpusEncoder"/>. Mirrors the shape of
/// <see cref="OpusDecoderConfig"/>.
/// </summary>
public sealed record OpusEncoderConfig
{
    /// <summary>Sample rate in Hz. Must be 8000, 12000, 16000, 24000, or 48000.</summary>
    public required int SampleRateHz { get; init; }

    /// <summary>Channel count (1 for mono, 2 for stereo).</summary>
    public required int ChannelCount { get; init; }

    /// <summary>
    /// Application hint - influences automatic mode + bitrate decisions in a
    /// real encoder. Currently informational.
    /// </summary>
    public OpusEncoderApplication Application { get; init; } = OpusEncoderApplication.Audio;

    /// <summary>Target bitrate in bits per second. 0 means "auto".</summary>
    public int BitrateBitsPerSecond { get; init; } = 0;

    /// <summary>Validate the configuration; throws on invalid values.</summary>
    public void Validate()
    {
        if (SampleRateHz is not (8000 or 12000 or 16000 or 24000 or 48000))
            throw new ArgumentException(
                $"OpusEncoderConfig.SampleRateHz must be 8000/12000/16000/24000/48000, got {SampleRateHz}.");
        if (ChannelCount is not (1 or 2))
            throw new ArgumentException(
                $"OpusEncoderConfig.ChannelCount must be 1 or 2, got {ChannelCount}.");
        if (BitrateBitsPerSecond < 0)
            throw new ArgumentException(
                $"OpusEncoderConfig.BitrateBitsPerSecond must be >= 0, got {BitrateBitsPerSecond}.");
    }
}

/// <summary>
/// Opus application hint (matches libopus <c>OPUS_APPLICATION_*</c>). Distinct
/// name from Concentus's <c>OpusApplication</c> so test projects that reference
/// both have no ambiguity.
/// </summary>
public enum OpusEncoderApplication
{
    /// <summary>VoIP - prioritises speech intelligibility.</summary>
    Voip = 2048,
    /// <summary>General audio - balances speech and music.</summary>
    Audio = 2049,
    /// <summary>Restricted low-delay - removes algorithmic delay introducing modes.</summary>
    RestrictedLowDelay = 2051,
}

/// <summary>
/// Top-level Opus encoder. Phase 1a state: structure and packet framing are
/// in place; the per-frame SILK and CELT encode paths are stubs. Subsequent
/// slices will wire up Concentus-derived SILK encode and a fresh CELT encode
/// that mirrors libopus.
/// </summary>
public sealed class OpusEncoder
{
    private readonly OpusEncoderConfig _config;
    private bool _disposed;

    /// <summary>Construct an encoder with the given configuration.</summary>
    public OpusEncoder(OpusEncoderConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        config.Validate();
        _config = config;
    }

    /// <summary>The codec this encoder produces.</summary>
    public AudioCodec Codec => AudioCodec.Opus;

    /// <summary>Sample rate in Hz.</summary>
    public int SampleRateHz => _config.SampleRateHz;

    /// <summary>Channel count.</summary>
    public int ChannelCount => _config.ChannelCount;

    /// <summary>
    /// Encode one PCM frame into a complete Opus packet. The frame size in
    /// samples per channel must be one of the Opus-legal values
    /// (2.5/5/10/20/40/60 ms at the configured sample rate).
    /// Currently throws <see cref="NotImplementedException"/>.
    /// </summary>
    public int EncodeFrame(
        ReadOnlySpan<float> pcmInput,
        Span<byte> opusPacketOut,
        int frameSizeSamples)
    {
        ThrowIfDisposed();
        if (frameSizeSamples <= 0)
            throw new ArgumentOutOfRangeException(nameof(frameSizeSamples));
        int requiredInputLen = frameSizeSamples * _config.ChannelCount;
        if (pcmInput.Length < requiredInputLen)
            throw new ArgumentException(
                $"pcmInput too small: need {requiredInputLen} samples, got {pcmInput.Length}.",
                nameof(pcmInput));
        _ = opusPacketOut;

        // Phase 1a: per-mode encoders (SILK and CELT) are not yet implemented.
        // Until both land, this top-level encoder cannot produce valid Opus
        // packets. We throw a clear NotImplementedException so consumers know
        // to wait for the per-mode encoders.
        throw new NotImplementedException(
            "OpusEncoder.EncodeFrame is not yet implemented. " +
            "Phase 1a target: SILK encode requires a port of libopus' " +
            "silk/enc_API.c (or Concentus equivalent); CELT encode requires " +
            "a port of libopus celt/celt_encoder.c. Neither is implemented " +
            "yet. When both per-mode encoders land, this top-level encoder " +
            "will wire them through OpusPacketParser-shaped framing.");
    }

    /// <summary>Releases resources held by the encoder.</summary>
    public ValueTask DisposeAsync()
    {
        _disposed = true;
        return ValueTask.CompletedTask;
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }
}
