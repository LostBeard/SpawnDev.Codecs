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

    // SILK, structured like libopus silk_Decode: one raw channel decoder per stream channel (mid, side), the shared
    // stereo state (MS->LR history, also the mono 2-sample buffering = sStereo), one resampler per output channel.
    private SilkDecoder? _silkMid;
    private SilkDecoder? _silkSide;
    private readonly SilkStereoState _silkStereoState = new();
    private SilkResamplerState? _silkResamplerL;
    private SilkResamplerState? _silkResamplerR;
    private int _silkInternalKHz;        // 0 until the first SILK frame
    private int _silkPrevStreamChannels; // libopus psDec->nChannelsInternal
    private bool _silkPrevDecodeOnlyMiddle;
    // Internal-rate mid/side with the 2-sample prefix MS->LR and the mono buffering use.
    private short[]? _silkMidScratch;
    private short[]? _silkSideScratch;
    private short[]? _silkResampleLOut;
    private short[]? _silkResampleROut;
    private readonly short[] _silkLbrrPulses = new short[SilkConstants.MAX_FRAME_LENGTH + SilkConstants.SHELL_CODEC_FRAME_LENGTH];

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

        // libopus silk_Decode, one call per SILK frame of the packet. nChannelsInternal = the packet's stereo bit,
        // nChannelsAPI = the decoder's channel count.
        int streamChannels = toc.IsStereo ? 2 : 1;
        bool apiStereo = _config.ChannelCount == 2;
        int internalFrameLen = internalKHz * silkFrameLengthMs;
        int outputFrameLen = _config.SampleRateHz / 1000 * silkFrameLengthMs;
        if (pcmOut.Length < outputFrameLen * silkFrameCount * _config.ChannelCount)
        {
            throw new ArgumentException(
                $"pcmOut too small: need {outputFrameLen * silkFrameCount * _config.ChannelCount} samples for {totalMs} ms at {_config.SampleRateHz} Hz.",
                nameof(pcmOut));
        }

        // Stream channel / rate transitions (silk_Decode, before the frame header).
        bool monoToStereo = streamChannels > _silkPrevStreamChannels;
        bool stereoToMono = streamChannels == 1 && _silkPrevStreamChannels == 2 && internalKHz == _silkInternalKHz;
        _silkMid ??= new SilkDecoder(internalKHz * 1000, silkFrameLengthMs);
        if (monoToStereo || _silkSide is null)
        {
            // silk_init_decoder on the second channel; set_fs below then sees a rate change and resets it again.
            _silkSide = new SilkDecoder(internalKHz * 1000, silkFrameLengthMs);
        }
        _silkMid.SetFs(internalKHz, silkFrameLengthMs);
        if (streamChannels == 2) _silkSide.SetFs(internalKHz, silkFrameLengthMs);
        if (_silkResamplerL is null || internalKHz != _silkInternalKHz)
        {
            // silk_decoder_set_fs re-inits the resampler when the internal rate changes.
            _silkResamplerL = new SilkResamplerState();
            SilkResampler.Init(_silkResamplerL, internalKHz * 1000, _config.SampleRateHz, forEncode: false);
            _silkResamplerR = new SilkResamplerState();
            SilkResampler.Init(_silkResamplerR, internalKHz * 1000, _config.SampleRateHz, forEncode: false);
        }
        if (apiStereo && streamChannels == 2 && _silkPrevStreamChannels == 1)
        {
            _silkStereoState.PredPrevQ13[0] = _silkStereoState.PredPrevQ13[1] = 0;
            _silkStereoState.SSide[0] = _silkStereoState.SSide[1] = 0;
            _silkResamplerR!.CopyFrom(_silkResamplerL);
        }
        _silkInternalKHz = internalKHz;
        _silkPrevStreamChannels = streamChannels;

        if (_silkMidScratch is null || _silkMidScratch.Length < internalFrameLen + 2)
        {
            _silkMidScratch = new short[internalFrameLen + 2];
            _silkSideScratch = new short[internalFrameLen + 2];
        }
        if (_silkResampleLOut is null || _silkResampleLOut.Length < outputFrameLen)
        {
            _silkResampleLOut = new short[outputFrameLen];
            _silkResampleROut = new short[outputFrameLen];
        }
        short[] mid = _silkMidScratch;
        short[] side = _silkSideScratch!;

        var rangeDec = new OpusRangeDecoder(frame.ToArray());

        // Header (RFC 6716 4.2.3): per channel, one VAD flag per SILK frame then that channel's LBRR flag.
        Span<int> vad = stackalloc int[6];      // [channel * 3 + frame]
        Span<int> lbrrFlags = stackalloc int[6];
        Span<int> lbrrFlag = stackalloc int[2];
        for (int n = 0; n < streamChannels; n++)
        {
            for (int f = 0; f < silkFrameCount; f++) vad[n * 3 + f] = rangeDec.DecodeBitLogP(1);
            lbrrFlag[n] = rangeDec.DecodeBitLogP(1);
        }
        for (int n = 0; n < streamChannels; n++)
        {
            if (lbrrFlag[n] == 0) continue;
            if (silkFrameCount == 1)
            {
                lbrrFlags[n * 3] = 1;
            }
            else
            {
                int lbrrSymbol = rangeDec.DecodeIcdf(silkFrameCount == 2 ? LbrrFlags2Icdf : LbrrFlags3Icdf, 8) + 1;
                for (int f = 0; f < silkFrameCount; f++) lbrrFlags[n * 3 + f] = (lbrrSymbol >> f) & 1;
            }
        }

        // Regular decoding skips all LBRR data (it only matters for packet-loss recovery): parse and discard.
        Span<int> predQ13 = stackalloc int[2];
        if (lbrrFlag[0] != 0 || lbrrFlag[1] != 0)
        {
            SkipLbrrFrames(rangeDec, internalKHz, silkFrameLengthMs, silkFrameCount, streamChannels, lbrrFlags, predQ13);
        }

        for (int f = 0; f < silkFrameCount; f++)
        {
            int decodeOnlyMiddle = 0;
            if (streamChannels == 2)
            {
                SilkStereoDecodePred.DecodePred(rangeDec, predQ13);
                if (vad[3 + f] == 0) decodeOnlyMiddle = SilkStereoDecodePred.DecodeMidOnly(rangeDec);
                if (decodeOnlyMiddle == 0 && _silkPrevDecodeOnlyMiddle)
                {
                    // First frame with side coding after mid-only frames: reset the side prediction memory.
                    _silkSide.ResetPredictionMemory();
                }
            }

            _silkMid.DecodeRawFromRange(rangeDec, mid.AsSpan(2, internalFrameLen), vadFlag: vad[f] != 0,
                conditional: f > 0);
            if (streamChannels == 2)
            {
                if (decodeOnlyMiddle == 0)
                {
                    // CODE_INDEPENDENTLY for the packet's first frame, CODE_INDEPENDENTLY_NO_LTP_SCALING right after a
                    // mid-only frame, CODE_CONDITIONALLY otherwise.
                    bool noLtpScaling = f > 0 && _silkPrevDecodeOnlyMiddle;
                    _silkSide.DecodeRawFromRange(rangeDec, side.AsSpan(2, internalFrameLen), vadFlag: vad[3 + f] != 0,
                        conditional: f > 0 && !noLtpScaling, skipLtpScale: noLtpScaling);
                }
                else
                {
                    side.AsSpan(2, internalFrameLen).Clear();
                }
            }

            short[] outL = _silkResampleLOut;
            short[] outR = _silkResampleROut!;
            if (apiStereo && streamChannels == 2)
            {
                SilkStereoMsToLr.Apply(_silkStereoState, mid, side, predQ13, internalKHz, internalFrameLen);
                SilkResampler.Apply(_silkResamplerL!, outL.AsSpan(0, outputFrameLen), mid.AsSpan(1, internalFrameLen), internalFrameLen);
                SilkResampler.Apply(_silkResamplerR!, outR.AsSpan(0, outputFrameLen), side.AsSpan(1, internalFrameLen), internalFrameLen);
            }
            else
            {
                // Mono buffering through sStereo.sMid (shared with MS->LR), then resample from index 1.
                mid[0] = _silkStereoState.SMid[0];
                mid[1] = _silkStereoState.SMid[1];
                _silkStereoState.SMid[0] = mid[internalFrameLen];
                _silkStereoState.SMid[1] = mid[internalFrameLen + 1];
                SilkResampler.Apply(_silkResamplerL!, outL.AsSpan(0, outputFrameLen), mid.AsSpan(1, internalFrameLen), internalFrameLen);
                if (apiStereo)
                {
                    if (stereoToMono && f == 0)
                    {
                        // Newly collapsed stereo: the right channel runs through its own resampler once more.
                        SilkResampler.Apply(_silkResamplerR!, outR.AsSpan(0, outputFrameLen), mid.AsSpan(1, internalFrameLen), internalFrameLen);
                    }
                    else
                    {
                        outL.AsSpan(0, outputFrameLen).CopyTo(outR);
                    }
                }
            }
            _silkPrevDecodeOnlyMiddle = decodeOnlyMiddle != 0;

            if (apiStereo)
            {
                int baseOffset = f * outputFrameLen * 2;
                for (int i = 0; i < outputFrameLen; i++)
                {
                    pcmOut[baseOffset + 2 * i] = outL[i] / 32768.0f;
                    pcmOut[baseOffset + 2 * i + 1] = outR[i] / 32768.0f;
                }
            }
            else
            {
                int baseOffset = f * outputFrameLen;
                for (int i = 0; i < outputFrameLen; i++) pcmOut[baseOffset + i] = outL[i] / 32768.0f;
            }
        }
    }

    // libopus silk_LBRR_flags_2_iCDF / silk_LBRR_flags_3_iCDF.
    private static ReadOnlySpan<byte> LbrrFlags2Icdf => [203, 150, 0];
    private static ReadOnlySpan<byte> LbrrFlags3Icdf => [215, 195, 166, 125, 110, 82, 0];

    /// <summary>
    /// libopus silk_Decode's FLAG_DECODE_NORMAL LBRR skip: per frame, per channel with an LBRR frame, the stereo
    /// predictors (+ mid-only flag), the indices and the pulses are read and thrown away. The values it touches are
    /// all re-read for the regular frames that follow, so nothing persists.
    /// </summary>
    private void SkipLbrrFrames(OpusRangeDecoder rangeDec, int internalKHz, int frameLengthMs, int frameCount,
        int streamChannels, ReadOnlySpan<int> lbrrFlags, Span<int> predScratch)
    {
        int nbSubfr = frameLengthMs == 20 ? SilkConstants.PE_MAX_NB_SUBFR : SilkConstants.PE_MAX_NB_SUBFR / 2;
        int frameLength = internalKHz * frameLengthMs;
        SilkNlsfCodebook codebook = internalKHz == 16 ? SilkNlsfCodebookTables.Wb : SilkNlsfCodebookTables.NbMb;
        int shellLen = SilkConstants.SHELL_CODEC_FRAME_LENGTH;
        Span<short> pulses = _silkLbrrPulses.AsSpan(0, (frameLength + shellLen - 1) & ~(shellLen - 1));
        // decode_indices with decode_LBRR reads delta pitch against the channel's ec_prev state, which the previous
        // LBRR frame of the same channel set.
        Span<short> prevLag = stackalloc short[2];
        Span<int> prevVoiced = stackalloc int[2];
        var indices = new SilkDecodedIndices();
        for (int f = 0; f < frameCount; f++)
        {
            for (int n = 0; n < streamChannels; n++)
            {
                if (lbrrFlags[n * 3 + f] == 0) continue;
                if (streamChannels == 2 && n == 0)
                {
                    SilkStereoDecodePred.DecodePred(rangeDec, predScratch);
                    if (lbrrFlags[3 + f] == 0) SilkStereoDecodePred.DecodeMidOnly(rangeDec);
                }
                int conditional = f > 0 && lbrrFlags[n * 3 + f - 1] != 0 ? 1 : 0;
                SilkIndicesDecoder.Decode(indices, rangeDec, codebook, vadFlag: false, decodeLbrr: true,
                    fsKHz: internalKHz, nbSubfr: nbSubfr, conditional: conditional,
                    prevLagIndex: prevLag[n], prevSignalTypeWasVoiced: prevVoiced[n] != 0);
                prevLag[n] = indices.LagIndex;
                prevVoiced[n] = indices.SignalType == SilkConstants.TYPE_VOICED ? 1 : 0;
                SilkPulsesDecoder.Decode(pulses, rangeDec, indices.SignalType, indices.QuantOffsetType, frameLength);
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
