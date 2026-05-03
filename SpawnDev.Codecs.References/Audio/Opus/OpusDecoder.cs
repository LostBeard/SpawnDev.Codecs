// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
// See NOTICE.md for upstream attributions.

using SpawnDev.Codecs.Audio.Opus.Celt;
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

    // Lazily-constructed CELT decoder. Used for CELT-only AND Hybrid-mode
    // packets (Hybrid = SILK low band + CELT high band; we route the full
    // packet to CELT which internally runs both halves correctly via its
    // Concentus backbone). One instance per OpusDecoder lifetime so MDCT
    // overlap / post-filter taps / oldEBands carry across packets per RFC
    // 6716 sec 4.3 inter-frame state requirements.
    private Celt.CeltDecoder? _celtDecoder;

    // Lazily-constructed SILK decoders, keyed by (internal-fs-kHz, frame-length-ms).
    // Mono packets use _silkDecoderMono; stereo packets additionally use _silkDecoderSide
    // and _silkStereoState for mid/side processing.
    private SilkDecoder? _silkDecoderMono;
    private SilkDecoder? _silkDecoderSide;
    private SilkStereoState? _silkStereoState;
    private int _silkInternalKHz;
    private int _silkFrameLengthMs;
    private short[]? _silkPcmScratch;
    // Stereo needs MID and SIDE internal-rate PCM plus 2-sample prefix for MS->LR.
    private short[]? _silkMidScratch;
    private short[]? _silkSideScratch;
    // Stereo-only: resamplers for L and R channels when output rate differs from internal.
    private SilkResamplerState? _silkResamplerL;
    private SilkResamplerState? _silkResamplerR;
    private short[]? _silkResampleLOut;
    private short[]? _silkResampleROut;

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

        // For CELT-only and Hybrid packets we need packet-level processing:
        // CELT carries inter-frame state (MDCT overlap, post-filter taps,
        // oldEBands) that is encoded once per PACKET, not per frame, and the
        // Concentus-backed CeltDecoder handles the whole packet (parsing the
        // per-frame range coder internally). For SILK-only packets we keep
        // the existing per-frame loop because SILK's stereo predictors and
        // mid-only flag are also packet-level but our SilkDecoder has its
        // own per-frame fan-out logic.
        if (packet.Toc.Mode is OpusMode.Celt or OpusMode.Hybrid)
        {
            EnsureCeltDecoder();
            int produced = _celtDecoder!.DecodePacket(
                compressedPacket.Span,
                pcmOutput.Span,
                samplesPerFrame * packet.FrameCount);
            return new ValueTask<int>(produced);
        }

        // Route every SILK frame through the per-frame SILK decode path.
        int offset = 0;
        foreach (var frame in packet.Frames)
        {
            var dst = pcmOutput.Slice(offset, samplesPerFrame * _config.ChannelCount);
            DecodeFrame(packet.Toc, frame.Span, dst.Span);
            offset += samplesPerFrame * _config.ChannelCount;
        }

        return new ValueTask<int>(totalSamples);
    }

    /// <summary>
    /// Lazily construct the per-stream CELT decoder. Reused across all CELT
    /// and Hybrid packets so MDCT overlap and post-filter state carry across
    /// packets correctly.
    /// </summary>
    private void EnsureCeltDecoder()
    {
        _celtDecoder ??= new Celt.CeltDecoder(
            // Use the fullband 20 ms mode as the default; the underlying
            // Concentus decoder reads the actual mode/bandwidth/frame-size
            // from each packet's TOC byte regardless of what we pass here.
            // The mode argument is retained to give callers and the future
            // hand-port a stable entry point.
            mode: Celt.CeltMode.Create(
                Celt.CeltConstants.FRAME_SIZE_20MS,
                Celt.CeltConstants.NB_BANDS_FULLBAND),
            outputSampleRateHz: _config.SampleRateHz,
            channelCount: _config.ChannelCount);
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
        _celtDecoder?.Dispose();
        _celtDecoder = null;
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
    /// VAD + LBRR flag-reading that precedes the SILK indices and, for stereo
    /// packets, the stereo predictors and mid/side -&gt; L/R conversion.
    /// Per RFC 6716 section 4.2.3 / 4.2.8.
    /// </summary>
    private void DecodeSilkFrame(OpusTocByte toc, ReadOnlySpan<byte> frame, Span<float> pcmOut)
    {
        int internalKHz = toc.Bandwidth switch
        {
            OpusBandwidth.Narrowband => 8,
            OpusBandwidth.Mediumband => 12,
            OpusBandwidth.Wideband => 16,
            _ => throw new ArgumentException(
                $"Unsupported SILK bandwidth {toc.Bandwidth}; SILK-only mode allows NB/MB/WB.", nameof(toc)),
        };

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

        bool stereo = toc.IsStereo;
        EnsureSilkDecoders(internalKHz, silkFrameLengthMs, stereo);

        var rangeDec = new OpusRangeDecoder(frame.ToArray());

        // Header: VAD flags (mid channel for each frame + side channel for each frame
        // if stereo), then single LBRR flag.
        Span<int> vadMid = stackalloc int[3];
        Span<int> vadSide = stackalloc int[3];
        for (int f = 0; f < silkFrameCount; f++)
        {
            vadMid[f] = rangeDec.DecodeBitLogP(1);
            if (stereo) vadSide[f] = rangeDec.DecodeBitLogP(1);
        }
        int lbrrFlag = rangeDec.DecodeBitLogP(1);
        if (lbrrFlag != 0)
        {
            throw new NotImplementedException(
                "SILK LBRR (low-bitrate redundancy) frames are not yet implemented.");
        }

        int outputLenPerFrameInternal = _silkDecoderMono!.InternalSampleRateHz == _config.SampleRateHz
            ? _silkDecoderMono.FrameLength
            : internalKHz * silkFrameLengthMs;
        int outputLenPerFrame = _silkDecoderMono.FrameLength;
        int totalOutputLen = outputLenPerFrame * silkFrameCount;
        int totalOutputSamples = totalOutputLen * (stereo ? 2 : 1);

        if (pcmOut.Length < totalOutputSamples)
        {
            throw new ArgumentException(
                $"pcmOut too small: need {totalOutputSamples} samples for {totalMs} ms {(stereo ? "stereo" : "mono")} at {_config.SampleRateHz} Hz.",
                nameof(pcmOut));
        }

        if (!stereo)
        {
            // Mono path (unchanged from slice 49).
            if (_silkPcmScratch is null || _silkPcmScratch.Length < outputLenPerFrame)
            {
                _silkPcmScratch = new short[outputLenPerFrame];
            }
            for (int f = 0; f < silkFrameCount; f++)
            {
                _silkDecoderMono.DecodeFromRange(
                    rangeDec,
                    _silkPcmScratch.AsSpan(0, outputLenPerFrame),
                    vadFlag: vadMid[f] != 0,
                    conditional: f > 0);

                int outputOffset = f * outputLenPerFrame;
                for (int i = 0; i < outputLenPerFrame; i++)
                {
                    pcmOut[outputOffset + i] = _silkPcmScratch[i] / 32768.0f;
                }
            }
            return;
        }

        // ---- Stereo path ----
        //
        // Per internal SILK frame:
        //   1. Read stereo predictors (2 Q13 values).
        //   2. If this side-VAD flag == 0, read mid-only flag. Else mid_only = 0.
        //   3. Decode mid channel (with conditional = frame index > 0).
        //   4. Unless mid_only, decode side channel.
        //   5. Run SilkStereoMsToLr on internal-rate buffers (output at internal rate).
        //   6. Resample mid/side independently to output rate if needed... actually
        //      MS->LR runs at the INTERNAL rate; we then resample L/R separately.
        //
        // Simplification for this slice: if the Opus config is 10/20 ms single-internal-frame
        // stereo, we run the above. Multi-frame stereo packets (40/60 ms) are allowed but
        // use the same per-frame logic.
        int internalFrameLen = internalKHz * silkFrameLengthMs;
        int bufLen = internalFrameLen + 2;
        if (_silkMidScratch is null || _silkMidScratch.Length < bufLen)
        {
            _silkMidScratch = new short[bufLen];
        }
        if (_silkSideScratch is null || _silkSideScratch.Length < bufLen)
        {
            _silkSideScratch = new short[bufLen];
        }

        Span<int> predQ13 = stackalloc int[2];

        int outputRateKHz = _config.SampleRateHz / 1000;
        int outputFrameLen = outputRateKHz * silkFrameLengthMs;
        bool needResample = _config.SampleRateHz != internalKHz * 1000;

        if (needResample)
        {
            if (_silkResampleLOut is null || _silkResampleLOut.Length < outputFrameLen)
            {
                _silkResampleLOut = new short[outputFrameLen];
                _silkResampleROut = new short[outputFrameLen];
            }
        }

        for (int f = 0; f < silkFrameCount; f++)
        {
            SilkStereoDecodePred.DecodePred(rangeDec, predQ13);

            int midOnly = (vadSide[f] == 0 && f == 0)
                ? SilkStereoDecodePred.DecodeMidOnly(rangeDec)
                : 0;

            // Decode mid into internal-rate scratch at position [2..internalFrameLen+1].
            short[] midTemp = new short[internalFrameLen];
            _silkDecoderMono.DecodeFromRange(
                rangeDec,
                midTemp.AsSpan(),
                vadFlag: vadMid[f] != 0,
                conditional: f > 0);
            midTemp.AsSpan().CopyTo(_silkMidScratch.AsSpan(2, internalFrameLen));

            if (midOnly == 0)
            {
                short[] sideTemp = new short[internalFrameLen];
                _silkDecoderSide!.DecodeFromRange(
                    rangeDec,
                    sideTemp.AsSpan(),
                    vadFlag: vadSide[f] != 0,
                    conditional: f > 0);
                sideTemp.AsSpan().CopyTo(_silkSideScratch.AsSpan(2, internalFrameLen));
            }
            else
            {
                _silkSideScratch.AsSpan(2, internalFrameLen).Clear();
            }

            // MS -> LR at internal rate.
            SilkStereoMsToLr.Apply(_silkStereoState!, _silkMidScratch, _silkSideScratch,
                predQ13, internalKHz, internalFrameLen);

            if (needResample)
            {
                // Resample L and R independently from internal to output rate.
                SilkResampler.Apply(_silkResamplerL!,
                    _silkResampleLOut!.AsSpan(0, outputFrameLen),
                    _silkMidScratch.AsSpan(1, internalFrameLen),
                    internalFrameLen);
                SilkResampler.Apply(_silkResamplerR!,
                    _silkResampleROut!.AsSpan(0, outputFrameLen),
                    _silkSideScratch.AsSpan(1, internalFrameLen),
                    internalFrameLen);

                int baseOffset = f * outputFrameLen * 2;
                for (int i = 0; i < outputFrameLen; i++)
                {
                    pcmOut[baseOffset + 2 * i] = _silkResampleLOut![i] / 32768.0f;
                    pcmOut[baseOffset + 2 * i + 1] = _silkResampleROut![i] / 32768.0f;
                }
            }
            else
            {
                int baseOffset = f * internalFrameLen * 2;
                for (int i = 0; i < internalFrameLen; i++)
                {
                    pcmOut[baseOffset + 2 * i] = _silkMidScratch[i + 1] / 32768.0f;
                    pcmOut[baseOffset + 2 * i + 1] = _silkSideScratch[i + 1] / 32768.0f;
                }
            }
        }
    }

    private void EnsureSilkDecoders(int internalKHz, int frameLengthMs, bool stereo)
    {
        bool configChanged = _silkDecoderMono is null ||
            _silkInternalKHz != internalKHz ||
            _silkFrameLengthMs != frameLengthMs;

        if (configChanged)
        {
            // For stereo we currently decode at internal rate (no resampling). For mono
            // we can decode directly at output rate.
            int outputRateHz = stereo ? internalKHz * 1000 : _config.SampleRateHz;

            _silkDecoderMono = new SilkDecoder(
                internalSampleRateHz: internalKHz * 1000,
                frameLengthMs: frameLengthMs,
                outputSampleRateHz: outputRateHz);
            _silkInternalKHz = internalKHz;
            _silkFrameLengthMs = frameLengthMs;
            _silkPcmScratch = null;
            _silkMidScratch = null;
            _silkSideScratch = null;
            _silkDecoderSide = null;
            _silkStereoState = null;
        }

        if (stereo)
        {
            if (_silkDecoderSide is null)
            {
                _silkDecoderSide = new SilkDecoder(
                    internalSampleRateHz: internalKHz * 1000,
                    frameLengthMs: frameLengthMs,
                    outputSampleRateHz: internalKHz * 1000);
            }
            _silkStereoState ??= new SilkStereoState();

            // If output rate differs from internal, allocate a per-channel resampler pair.
            if (_config.SampleRateHz != internalKHz * 1000)
            {
                if (_silkResamplerL is null || configChanged)
                {
                    _silkResamplerL = new SilkResamplerState();
                    SilkResampler.Init(_silkResamplerL, internalKHz * 1000, _config.SampleRateHz, forEncode: false);
                    _silkResamplerR = new SilkResamplerState();
                    SilkResampler.Init(_silkResamplerR, internalKHz * 1000, _config.SampleRateHz, forEncode: false);
                    _silkResampleLOut = null;
                    _silkResampleROut = null;
                }
            }
            else
            {
                _silkResamplerL = null;
                _silkResamplerR = null;
            }
        }
    }

    /// <summary>
    /// Per-frame Hybrid dispatch. Should not normally be reached: CELT/Hybrid
    /// packets are handled at the packet level above (see DecodePacketAsync)
    /// because Hybrid carries packet-level state (MDCT overlap, post-filter
    /// taps) that the CeltDecoder owns. Kept here for defense-in-depth in
    /// case a future code path bypasses the packet-level shortcut.
    /// </summary>
    private void DecodeHybridFrame(OpusTocByte toc, ReadOnlySpan<byte> frame, Span<float> pcmOut)
    {
        EnsureCeltDecoder();
        // Reconstitute the single-frame Opus packet (TOC byte + frame body)
        // and route through the CELT decoder which handles the SILK low band
        // and CELT high band internally per RFC 6716 sec 4.5.
        Span<byte> tocPlusFrame = stackalloc byte[1 + frame.Length];
        tocPlusFrame[0] = toc.Value;
        frame.CopyTo(tocPlusFrame.Slice(1));
        int samples = toc.GetSamplesPerFrame(_config.SampleRateHz);
        _ = _celtDecoder!.DecodePacket(tocPlusFrame, pcmOut, samples);
    }

    /// <summary>
    /// Per-frame CELT dispatch. As above: normally bypassed in favor of the
    /// packet-level DecodePacket call. Kept defensive.
    /// </summary>
    private void DecodeCeltFrame(OpusTocByte toc, ReadOnlySpan<byte> frame, Span<float> pcmOut)
    {
        EnsureCeltDecoder();
        Span<byte> tocPlusFrame = stackalloc byte[1 + frame.Length];
        tocPlusFrame[0] = toc.Value;
        frame.CopyTo(tocPlusFrame.Slice(1));
        int samples = toc.GetSamplesPerFrame(_config.SampleRateHz);
        _ = _celtDecoder!.DecodePacket(tocPlusFrame, pcmOut, samples);
    }
}
