// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
// See NOTICE.md for upstream attributions.

using SpawnDev.Codecs.EntropyCoders;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Public SILK decoder. Wraps the internal per-frame decode pipeline
/// (<see cref="SilkChannelDecoderState"/> + <see cref="SilkDecodeFrame"/>) behind a
/// stable API, with an optional resampler that converts the internal SILK sample
/// rate to the caller-specified output rate (8/12/16/24/48 kHz).
/// </summary>
public sealed class SilkDecoder
{
    private readonly SilkChannelDecoderState _state;
    private readonly SilkResamplerState? _resamplerState;
    private readonly short[]? _internalPcmBuf;
    private readonly short[] _sMid = new short[2];
    private readonly int _internalFrameLength;
    private readonly int _outputFrameLength;

    /// <summary>
    /// Create a SILK decoder for the given internal SILK rate. Output is produced at the
    /// same rate.
    /// </summary>
    /// <param name="internalSampleRateHz">SILK internal sample rate (8000, 12000, or 16000).</param>
    /// <param name="frameLengthMs">Frame duration in milliseconds (10 or 20).</param>
    public SilkDecoder(int internalSampleRateHz, int frameLengthMs = 20)
        : this(internalSampleRateHz, frameLengthMs, internalSampleRateHz) { }

    /// <summary>
    /// Create a SILK decoder that resamples the internal output to <paramref name="outputSampleRateHz"/>.
    /// </summary>
    /// <param name="internalSampleRateHz">SILK internal sample rate (8000, 12000, or 16000).</param>
    /// <param name="frameLengthMs">Frame duration in milliseconds (10 or 20).</param>
    /// <param name="outputSampleRateHz">Desired output sample rate (8000, 12000, 16000, 24000, or 48000).
    /// If equal to <paramref name="internalSampleRateHz"/>, no resampler is allocated.</param>
    public SilkDecoder(int internalSampleRateHz, int frameLengthMs, int outputSampleRateHz)
    {
        int fsKHz = internalSampleRateHz switch
        {
            8000 => 8,
            12000 => 12,
            16000 => 16,
            _ => throw new ArgumentException(
                $"internalSampleRateHz must be 8000, 12000, or 16000 (got {internalSampleRateHz}).",
                nameof(internalSampleRateHz)),
        };
        if (frameLengthMs != 10 && frameLengthMs != 20)
            throw new ArgumentException(
                $"frameLengthMs must be 10 or 20 (got {frameLengthMs}).", nameof(frameLengthMs));
        if (outputSampleRateHz != 8000 && outputSampleRateHz != 12000 &&
            outputSampleRateHz != 16000 && outputSampleRateHz != 24000 && outputSampleRateHz != 48000)
        {
            throw new ArgumentException(
                $"outputSampleRateHz must be 8000, 12000, 16000, 24000, or 48000 (got {outputSampleRateHz}).",
                nameof(outputSampleRateHz));
        }

        int nbSubfr = frameLengthMs == 20 ? SilkConstants.PE_MAX_NB_SUBFR : SilkConstants.PE_MAX_NB_SUBFR / 2;
        int lpcOrder = fsKHz == 16 ? 16 : 10;

        _state = new SilkChannelDecoderState();
        _state.Configure(fsKHz, nbSubfr, lpcOrder);
        _state.Reset();

        _internalFrameLength = _state.FrameLength;
        _outputFrameLength = outputSampleRateHz / 1000 * frameLengthMs;

        // ALWAYS through the resampler, like libopus silk_Decode: with equal rates it is the copy resampler, which still
        // applies delay_matrix_dec's input delay (4 samples at 8 kHz, 9 at 12, 12 at 16). Skipping it made every SILK
        // output start that many samples early (CrossValidate_Silk* vs Concentus failed since they were written).
        _resamplerState = new SilkResamplerState();
        SilkResampler.Init(_resamplerState, internalSampleRateHz, outputSampleRateHz, forEncode: false);
        // 2 prefix samples: silk_Decode's mono "Buffering" (sStereo.sMid) feeds the resampler from index 1, one more
        // sample of delay on top of the resampler's.
        _internalPcmBuf = new short[_internalFrameLength + 2];

        InternalSampleRateHz = internalSampleRateHz;
        OutputSampleRateHz = outputSampleRateHz;
        FrameLengthMs = frameLengthMs;
    }

    /// <summary>Internal SILK sample rate in Hz (8000, 12000, or 16000).</summary>
    public int InternalSampleRateHz { get; }

    /// <summary>Output sample rate in Hz (matches internal if no resampler is configured).</summary>
    public int OutputSampleRateHz { get; }

    /// <summary>Frame duration in milliseconds (10 or 20).</summary>
    public int FrameLengthMs { get; }

    /// <summary>LPC filter order (10 for NB/MB, 16 for WB).</summary>
    public int LpcOrder => _state.LpcOrder;

    /// <summary>Subframe count per frame (2 for 10 ms, 4 for 20 ms).</summary>
    public int NbSubfr => _state.NbSubfr;

    /// <summary>Frame length in samples at the output rate.</summary>
    public int FrameLength => _outputFrameLength;

    /// <summary>
    /// Reset the decoder state to first-frame defaults. Also reinitialises the resampler
    /// if one is in use.
    /// </summary>
    public void Reset()
    {
        _state.Reset();
        _sMid[0] = _sMid[1] = 0;
        if (_resamplerState is not null)
        {
            SilkResampler.Init(_resamplerState, InternalSampleRateHz, OutputSampleRateHz, forEncode: false);
        }
    }

    /// <summary>
    /// Decode one SILK frame from a range-coded byte payload into 16-bit PCM at the output rate.
    /// Convenience wrapper around <see cref="DecodeFromRange"/> that constructs the range decoder
    /// from the payload bytes.
    /// </summary>
    /// <param name="payload">Range-coded SILK frame bytes.</param>
    /// <param name="pcmOut">Output PCM buffer. Length &gt;= <see cref="FrameLength"/>.</param>
    /// <param name="vadFlag">VAD flag for the frame. Controls which signal-type iCDF is read.</param>
    /// <param name="conditional">True for conditional / delta coding, false for independent.</param>
    /// <returns>Number of PCM samples written (equal to <see cref="FrameLength"/>).</returns>
    public int DecodeFrame(ReadOnlySpan<byte> payload, Span<short> pcmOut, bool vadFlag, bool conditional)
    {
        if (payload.Length == 0) throw new ArgumentException("Empty payload.", nameof(payload));
        var rangeDec = new OpusRangeDecoder(payload.ToArray());
        return DecodeFromRange(rangeDec, pcmOut, vadFlag, conditional);
    }

    /// <summary>
    /// Decode one SILK frame from a pre-positioned range decoder. This is the primitive
    /// used by the Opus-level decoder, which reads VAD + LBRR flags from the range decoder
    /// before calling this method. End users who have raw SILK frame bytes should prefer
    /// <see cref="DecodeFrame"/> for convenience.
    /// </summary>
    /// <param name="rangeDec">Range decoder already positioned at the start of the SILK indices.</param>
    /// <param name="pcmOut">Output PCM buffer. Length &gt;= <see cref="FrameLength"/>.</param>
    /// <param name="vadFlag">VAD flag (as read from the outer Opus frame header).</param>
    /// <param name="conditional">True for conditional / delta coding, false for independent.</param>
    /// <returns>Number of PCM samples written.</returns>
    public int DecodeFromRange(OpusRangeDecoder rangeDec, Span<short> pcmOut, bool vadFlag, bool conditional)
    {
        if (rangeDec is null) throw new ArgumentNullException(nameof(rangeDec));
        if (pcmOut.Length < _outputFrameLength)
            throw new ArgumentException(
                $"pcmOut too small (need {_outputFrameLength}).", nameof(pcmOut));

        var buf = _internalPcmBuf!;
        SilkDecodeFrame.Decode(_state, rangeDec, buf.AsSpan(2, _internalFrameLength), vadFlag, conditional ? 1 : 0);
        // silk_Decode mono branch: buf[0..1] = last frame's tail, then resample from buf[1].
        buf[0] = _sMid[0];
        buf[1] = _sMid[1];
        _sMid[0] = buf[_internalFrameLength];
        _sMid[1] = buf[_internalFrameLength + 1];
        SilkResampler.Apply(_resamplerState!, pcmOut, buf.AsSpan(1, _internalFrameLength), _internalFrameLength);
        return _outputFrameLength;
    }

    /// <summary>
    /// Decode one SILK frame at the internal rate with no buffering and no resampling: the per-channel output that
    /// silk_Decode hands to silk_stereo_MS_to_LR. The Opus stereo path uses it; it does its own MS-to-LR and resampling.
    /// </summary>
    internal void DecodeRawFromRange(OpusRangeDecoder rangeDec, Span<short> internalPcmOut, bool vadFlag, bool conditional,
        bool skipLtpScale = false)
    {
        int frameLength = _state.FrameLength;
        if (internalPcmOut.Length < frameLength)
            throw new ArgumentException($"internalPcmOut too small (need {frameLength}).", nameof(internalPcmOut));
        SilkDecodeFrame.Decode(_state, rangeDec, internalPcmOut.Slice(0, frameLength), vadFlag, conditional ? 1 : 0,
            skipLtpScale);
    }

    /// <summary>
    /// libopus silk_decoder_set_fs for the raw path (<see cref="DecodeRawFromRange"/> only; the public
    /// buffering/resampling API keeps its constructor geometry): a frame-length change keeps all state, an
    /// internal-rate change also resets the prediction memory.
    /// </summary>
    internal void SetFs(int fsKHz, int frameLengthMs)
    {
        int nbSubfr = frameLengthMs == 20 ? SilkConstants.PE_MAX_NB_SUBFR : SilkConstants.PE_MAX_NB_SUBFR / 2;
        bool fsChanged = _state.FsKHz != fsKHz;
        _state.Configure(fsKHz, nbSubfr, fsKHz == 16 ? 16 : 10);
        if (fsChanged) _state.ResetPredictionMemory();
    }

    /// <summary>The side channel's prediction-memory reset when it is coded again after mid-only frames.</summary>
    internal void ResetPredictionMemory() => _state.ResetPredictionMemory();
}
