// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
// See NOTICE.md for upstream attributions.
//
// Top-level Opus encoder. Routes per-frame encode requests to the SILK or
// CELT subsystem based on the configured mode. Mirrors the structure of
// OpusDecoder so consumers can encode/decode through symmetric APIs.
//
// Per the user's instruction to mirror the CELT decoder pattern (see
// Audio/Opus/Celt/CeltDecoder.cs and the CELT decoder commit message), the
// encode path currently delegates the per-frame work to the BSD-3 Concentus
// pure-C# port of libopus. This delivers a working encoder today against the
// libopus reference and gives a verifiable backbone for a future hand-port
// that will live behind the same public API in Audio/Opus/Silk/ and
// Audio/Opus/Celt/. The Concentus runtime dependency is documented in
// NOTICE.md and SpawnDev.Codecs.csproj.

using Concentus;
using Concentus.Enums;

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
/// Top-level Opus encoder. Encodes float PCM frames into RFC 6716 Opus
/// packets. The current implementation delegates the per-frame work to the
/// BSD-3 Concentus pure-C# port of libopus so the encoder is fully working
/// today; the SILK and CELT scaffolding in <c>Audio/Opus/Silk/</c> and
/// <c>Audio/Opus/Celt/</c> stays in place for the future hand-port. The
/// public surface here is stable: replacing the Concentus dependency with the
/// hand-port internally is an implementation detail.
///
/// Stateful: maintains the encoder's per-stream rate-control history,
/// adaptive prediction state, and (for SILK) LSF/LPC predictors across
/// frames. NOT thread-safe; one encoder per stream.
/// </summary>
public sealed class OpusEncoder : IDisposable
{
    private readonly OpusEncoderConfig _config;
    private IOpusEncoder? _concentusEnc;
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

    /// <summary>The application hint configured for this encoder.</summary>
    public OpusEncoderApplication Application => _config.Application;

    /// <summary>
    /// Encode one PCM frame into a complete Opus packet. The frame size in
    /// samples per channel must be one of the Opus-legal values
    /// (2.5/5/10/20/40/60 ms at the configured sample rate).
    /// </summary>
    /// <param name="pcmInput">
    /// Interleaved float PCM in [-1, +1]. Length must be at least
    /// <paramref name="frameSizeSamples"/> * channels. Values outside [-1, +1]
    /// are clipped before being passed to the underlying encoder.
    /// </param>
    /// <param name="opusPacketOut">
    /// Destination buffer for the produced Opus packet. Should be at least
    /// 1275 bytes (the maximum Opus packet size per RFC 6716 sec 3.2.1) to
    /// guarantee that any encoder decision fits.
    /// </param>
    /// <param name="frameSizeSamples">Frame size in samples per channel.</param>
    /// <returns>The number of bytes written to <paramref name="opusPacketOut"/>.</returns>
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
        if (opusPacketOut.IsEmpty)
            throw new ArgumentException("opusPacketOut must not be empty.", nameof(opusPacketOut));

        EnsureConcentusEncoder();

        // Concentus accepts the float overload directly and clips internally,
        // but to make the contract explicit at our boundary we present the
        // exact requested-length slice (no trailing junk past the frame).
        var trimmed = pcmInput.Slice(0, requiredInputLen);
        int bytes = _concentusEnc!.Encode(trimmed, frameSizeSamples, opusPacketOut, opusPacketOut.Length);
        if (bytes <= 0)
        {
            throw new InvalidOperationException(
                $"Opus encoder returned {bytes} bytes for a {frameSizeSamples}-sample frame at " +
                $"{_config.SampleRateHz} Hz, {_config.ChannelCount} ch. Negative values match libopus " +
                "OPUS_* error codes; 0 means 'frame fully consumed by DTX'.");
        }
        return bytes;
    }

    /// <summary>
    /// Reset the encoder's inter-frame state. Call after a stream restart so
    /// the next packet does not depend on prior history. Mirrors libopus
    /// <c>opus_encoder_ctl(OPUS_RESET_STATE)</c>.
    /// </summary>
    public void ResetState()
    {
        ThrowIfDisposed();
        _concentusEnc?.ResetState();
    }

    /// <summary>Releases resources held by the encoder.</summary>
    public ValueTask DisposeAsync()
    {
        Dispose();
        return ValueTask.CompletedTask;
    }

    /// <summary>Releases resources held by the encoder.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _concentusEnc?.Dispose();
        _concentusEnc = null;
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    private void EnsureConcentusEncoder()
    {
        if (_concentusEnc is not null) return;

        // Force pure-managed Concentus path (no native libopus probe). Matches
        // the policy used by CeltDecoder + ReferenceOracle so every code path
        // (encode, decode, oracle) goes through the same pure-managed Opus
        // implementation across desktop AND Blazor WASM.
        OpusCodecFactory.AttemptToUseNativeLibrary = false;

        var application = _config.Application switch
        {
            OpusEncoderApplication.Voip => OpusApplication.OPUS_APPLICATION_VOIP,
            OpusEncoderApplication.Audio => OpusApplication.OPUS_APPLICATION_AUDIO,
            OpusEncoderApplication.RestrictedLowDelay => OpusApplication.OPUS_APPLICATION_RESTRICTED_LOWDELAY,
            _ => throw new InvalidOperationException(
                $"Unsupported OpusEncoderApplication '{_config.Application}'."),
        };

        _concentusEnc = OpusCodecFactory.CreateEncoder(
            _config.SampleRateHz, _config.ChannelCount, application);

        if (_config.BitrateBitsPerSecond > 0)
        {
            _concentusEnc.Bitrate = _config.BitrateBitsPerSecond;
        }
    }
}
