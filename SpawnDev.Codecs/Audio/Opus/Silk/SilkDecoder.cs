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

        if (outputSampleRateHz != internalSampleRateHz)
        {
            _resamplerState = new SilkResamplerState();
            SilkResampler.Init(_resamplerState, internalSampleRateHz, outputSampleRateHz, forEncode: false);
            _internalPcmBuf = new short[_internalFrameLength];
        }

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
        if (_resamplerState is not null)
        {
            SilkResampler.Init(_resamplerState, InternalSampleRateHz, OutputSampleRateHz, forEncode: false);
        }
    }

    /// <summary>
    /// Decode one SILK frame from a range-coded byte payload into 16-bit PCM at the output rate.
    /// </summary>
    /// <param name="payload">Range-coded SILK frame bytes.</param>
    /// <param name="pcmOut">Output PCM buffer. Length &gt;= <see cref="FrameLength"/>.</param>
    /// <param name="vadFlag">VAD flag for the frame. Controls which signal-type iCDF is read.</param>
    /// <param name="conditional">True for conditional / delta coding, false for independent.</param>
    /// <returns>Number of PCM samples written (equal to <see cref="FrameLength"/>).</returns>
    public int DecodeFrame(ReadOnlySpan<byte> payload, Span<short> pcmOut, bool vadFlag, bool conditional)
    {
        if (payload.Length == 0) throw new ArgumentException("Empty payload.", nameof(payload));
        if (pcmOut.Length < _outputFrameLength)
            throw new ArgumentException(
                $"pcmOut too small (need {_outputFrameLength}).", nameof(pcmOut));

        var rangeDec = new OpusRangeDecoder(payload.ToArray());

        if (_resamplerState is null)
        {
            // No resampling: decode directly into the caller's buffer.
            SilkDecodeFrame.Decode(_state, rangeDec, pcmOut, vadFlag, conditional ? 1 : 0);
            return _internalFrameLength;
        }

        // Resampling path: decode to internal buffer, then resample to caller's buffer.
        SilkDecodeFrame.Decode(_state, rangeDec, _internalPcmBuf!.AsSpan(0, _internalFrameLength),
            vadFlag, conditional ? 1 : 0);
        SilkResampler.Apply(_resamplerState, pcmOut, _internalPcmBuf.AsSpan(0, _internalFrameLength),
            _internalFrameLength);
        return _outputFrameLength;
    }
}
