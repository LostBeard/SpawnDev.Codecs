// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
// See NOTICE.md for upstream attributions.

using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.Codecs.EntropyCoders;

namespace SpawnDev.Codecs.Audio.Opus;

/// <summary>
/// Opus decoder (RFC 6716). Parses incoming packets with <see cref="OpusPacketParser"/>,
/// routes each frame to the SILK, Hybrid, or CELT path based on the packet's TOC byte,
/// and writes PCM samples to the caller's buffer.
///
/// Phase 1a state: packet parsing + mode routing are wired. SILK and CELT decode paths
/// are stubs that throw <see cref="NotImplementedException"/>. Subsequent slices port
/// Concentus SILK (pure C# sequential) and build CELT as ILGPU kernels.
/// </summary>
public sealed class OpusDecoder : IAudioDecoder
{
    private readonly OpusDecoderConfig _config;
    private bool _disposed;

    // Lazily-constructed SILK decoders, keyed by (internal-fs-kHz, frame-length-ms).
    // Mono only for now; stereo adds a second instance plus mid/side mixing logic.
    private SilkDecoder? _silkDecoderMono;
    private int _silkInternalKHz;
    private int _silkFrameLengthMs;
    private short[]? _silkPcmScratch;

    /// <summary>Creates a new Opus decoder with the given configuration.</summary>
    public OpusDecoder(OpusDecoderConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        config.Validate();
        _config = config;
    }

    /// <inheritdoc/>
    public AudioCodec Codec => AudioCodec.Opus;

    /// <inheritdoc/>
    public int SampleRateHz => _config.SampleRateHz;

    /// <inheritdoc/>
    public int ChannelCount => _config.ChannelCount;

    /// <inheritdoc/>
    public ValueTask<int> DecodePacketAsync(
        ReadOnlyMemory<byte> compressedPacket,
        Memory<float> pcmOutput,
        CancellationToken ct = default)
    {
        ThrowIfDisposed();
        ct.ThrowIfCancellationRequested();

        if (!OpusPacketParser.TryParse(compressedPacket, selfDelimited: false, out var packet, out var error)
            || packet is null)
        {
            throw new ArgumentException($"Failed to parse Opus packet: {error}.", nameof(compressedPacket));
        }

        int samplesPerFrame = packet.GetSamplesPerFrame(_config.SampleRateHz);
        int totalSamples = samplesPerFrame * packet.FrameCount;
        int requiredLength = totalSamples * _config.ChannelCount;
        if (pcmOutput.Length < requiredLength)
        {
            throw new ArgumentException(
                $"pcmOutput too small: need {requiredLength} samples ({totalSamples} frames * {_config.ChannelCount} channels), got {pcmOutput.Length}.",
                nameof(pcmOutput));
        }

        // Route every frame through the mode-specific decode path. Each path is currently a stub.
        int offset = 0;
        foreach (var frame in packet.Frames)
        {
            var dst = pcmOutput.Slice(offset, samplesPerFrame * _config.ChannelCount);
            DecodeFrame(packet.Toc, frame.Span, dst.Span);
            offset += samplesPerFrame * _config.ChannelCount;
        }

        return new ValueTask<int>(totalSamples);
    }

    /// <inheritdoc/>
    public async ValueTask<int> DecodePacketAsync(
        ReadOnlyMemory<byte> compressedPacket,
        Memory<short> pcmOutput,
        CancellationToken ct = default)
    {
        // Decode to float first, then narrow. When the float path is wired through real SILK/CELT
        // decoders, a short-native path can be added to avoid the conversion, but the float path
        // is the canonical one per Opus reference.
        float[] rental = System.Buffers.ArrayPool<float>.Shared.Rent(pcmOutput.Length);
        try
        {
            int samples = await DecodePacketAsync(compressedPacket, rental.AsMemory(0, pcmOutput.Length), ct).ConfigureAwait(false);
            int written = samples * _config.ChannelCount;
            for (int i = 0; i < written; i++)
            {
                float f = rental[i];
                if (f > 1.0f) f = 1.0f;
                else if (f < -1.0f) f = -1.0f;
                pcmOutput.Span[i] = (short)(f * short.MaxValue);
            }
            return samples;
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(rental);
        }
    }

    /// <inheritdoc/>
    public ValueTask DisposeAsync()
    {
        _disposed = true;
        return ValueTask.CompletedTask;
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    /// <summary>
    /// Dispatches one compressed frame to its mode-specific decoder. Phase 1a: stubs throw.
    /// </summary>
    private void DecodeFrame(OpusTocByte toc, ReadOnlySpan<byte> frame, Span<float> pcmOut)
    {
        switch (toc.Mode)
        {
            case OpusMode.Silk:
                DecodeSilkFrame(toc, frame, pcmOut);
                break;
            case OpusMode.Hybrid:
                DecodeHybridFrame(toc, frame, pcmOut);
                break;
            case OpusMode.Celt:
                DecodeCeltFrame(toc, frame, pcmOut);
                break;
            default:
                throw new InvalidOperationException($"Unknown Opus mode: {toc.Mode}");
        }
    }

    /// <summary>
    /// Dispatches a SILK-mode Opus frame to the SILK decoder. Handles the outer
    /// VAD + LBRR flag-reading that precedes the SILK indices in every Opus SILK
    /// frame. Stereo and LBRR are intentionally out of scope for this slice.
    /// </summary>
    private void DecodeSilkFrame(OpusTocByte toc, ReadOnlySpan<byte> frame, Span<float> pcmOut)
    {
        if (toc.IsStereo)
        {
            throw new NotImplementedException(
                "SILK stereo decode not yet implemented; mono SILK decode is supported.");
        }

        int internalKHz = toc.Bandwidth switch
        {
            OpusBandwidth.Narrowband => 8,
            OpusBandwidth.Mediumband => 12,
            OpusBandwidth.Wideband => 16,
            _ => throw new ArgumentException(
                $"Unsupported SILK bandwidth {toc.Bandwidth}; SILK-only mode allows NB/MB/WB.", nameof(toc)),
        };

        // Derive packet duration in ms from the TOC config. SILK-only packets are
        // 10/20/40/60 ms total. 10/20 ms = 1 internal SILK frame; 40/60 ms splits
        // into 2 or 3 internal 20 ms SILK frames read sequentially from the same
        // range-coded payload.
        int samplesAt48k = toc.GetSamplesPerFrame(48000);
        int totalMs = samplesAt48k / 48;
        (int silkFrameLengthMs, int silkFrameCount) = totalMs switch
        {
            10 => (10, 1),
            20 => (20, 1),
            40 => (20, 2),
            60 => (20, 3),
            _ => throw new NotImplementedException(
                $"SILK packet duration {totalMs} ms not a standard SILK duration."),
        };

        EnsureSilkDecoder(internalKHz, silkFrameLengthMs);

        // Outer Opus frame header: SILK-only packets encode one VAD flag per internal
        // SILK frame first, then one LBRR flag, then the per-frame indices. Per RFC 6716
        // section 4.2.3.
        var rangeDec = new OpusRangeDecoder(frame.ToArray());
        Span<int> vadFlags = stackalloc int[3]; // max 3 for 60 ms
        for (int f = 0; f < silkFrameCount; f++)
        {
            vadFlags[f] = rangeDec.DecodeBitLogP(1);
        }
        int lbrrFlag = rangeDec.DecodeBitLogP(1);
        if (lbrrFlag != 0)
        {
            throw new NotImplementedException(
                "SILK LBRR (low-bitrate redundancy) frames are not yet implemented.");
        }

        int outputLenPerFrame = _silkDecoderMono!.FrameLength;
        int totalOutputLen = outputLenPerFrame * silkFrameCount;

        if (_silkPcmScratch is null || _silkPcmScratch.Length < outputLenPerFrame)
        {
            _silkPcmScratch = new short[outputLenPerFrame];
        }

        if (pcmOut.Length < totalOutputLen)
        {
            throw new ArgumentException(
                $"pcmOut too small: need {totalOutputLen} samples for {totalMs} ms at {_config.SampleRateHz} Hz.",
                nameof(pcmOut));
        }

        // Decode each internal SILK frame sequentially. First frame is independent;
        // subsequent frames use conditional (delta) gain + pitch coding keyed on the
        // previous frame's state.
        for (int f = 0; f < silkFrameCount; f++)
        {
            _silkDecoderMono.DecodeFromRange(
                rangeDec,
                _silkPcmScratch.AsSpan(0, outputLenPerFrame),
                vadFlag: vadFlags[f] != 0,
                conditional: f > 0);

            int outputOffset = f * outputLenPerFrame;
            for (int i = 0; i < outputLenPerFrame; i++)
            {
                pcmOut[outputOffset + i] = _silkPcmScratch[i] / 32768.0f;
            }
        }
    }

    private void EnsureSilkDecoder(int internalKHz, int frameLengthMs)
    {
        if (_silkDecoderMono is not null &&
            _silkInternalKHz == internalKHz &&
            _silkFrameLengthMs == frameLengthMs)
        {
            return;
        }
        _silkDecoderMono = new SilkDecoder(
            internalSampleRateHz: internalKHz * 1000,
            frameLengthMs: frameLengthMs,
            outputSampleRateHz: _config.SampleRateHz);
        _silkInternalKHz = internalKHz;
        _silkFrameLengthMs = frameLengthMs;
        _silkPcmScratch = null; // force reallocation on first use
    }

    private static void DecodeHybridFrame(OpusTocByte toc, ReadOnlySpan<byte> frame, Span<float> pcmOut)
    {
        _ = toc; _ = frame; _ = pcmOut;
        throw new NotImplementedException(
            "Hybrid decode not yet implemented. Requires both SILK and CELT paths to be wired.");
    }

    private static void DecodeCeltFrame(OpusTocByte toc, ReadOnlySpan<byte> frame, Span<float> pcmOut)
    {
        _ = toc; _ = frame; _ = pcmOut;
        throw new NotImplementedException(
            "CELT decode not yet implemented. Phase 1a slice 6+ wires up ILGPU kernels for IMDCT / dequant / windowing / post-filter.");
    }
}
