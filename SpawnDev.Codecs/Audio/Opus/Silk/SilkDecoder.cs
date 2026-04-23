// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
// See NOTICE.md for upstream attributions.

using SpawnDev.Codecs.EntropyCoders;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Public SILK decoder. Wraps the internal per-frame decode pipeline
/// (<see cref="SilkChannelDecoderState"/> + <see cref="SilkDecodeFrame"/>) behind a
/// stable API. Produces PCM samples at the decoder's internal SILK sample rate
/// (8, 12, or 16 kHz); callers requiring a different output rate should feed the
/// result through an external resampler.
/// </summary>
public sealed class SilkDecoder
{
    private readonly SilkChannelDecoderState _state;

    /// <summary>
    /// Create a SILK decoder for the given configuration.
    /// </summary>
    /// <param name="internalSampleRateHz">SILK internal sample rate (8000, 12000, or 16000).</param>
    /// <param name="frameLengthMs">SILK frame length in milliseconds (10 or 20).</param>
    public SilkDecoder(int internalSampleRateHz, int frameLengthMs = 20)
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

        int nbSubfr = frameLengthMs == 20 ? SilkConstants.PE_MAX_NB_SUBFR : SilkConstants.PE_MAX_NB_SUBFR / 2;
        int lpcOrder = fsKHz == 16 ? 16 : 10;

        _state = new SilkChannelDecoderState();
        _state.Configure(fsKHz, nbSubfr, lpcOrder);
        _state.Reset();
    }

    /// <summary>Internal sample rate in Hz (8000, 12000, or 16000).</summary>
    public int InternalSampleRateHz => _state.FsKHz * 1000;

    /// <summary>LPC filter order (10 for NB/MB, 16 for WB).</summary>
    public int LpcOrder => _state.LpcOrder;

    /// <summary>Subframe count per frame (2 for 10 ms, 4 for 20 ms).</summary>
    public int NbSubfr => _state.NbSubfr;

    /// <summary>Frame length in samples (<see cref="InternalSampleRateHz"/> * frame duration).</summary>
    public int FrameLength => _state.FrameLength;

    /// <summary>
    /// Reset the decoder state to first-frame defaults. Call after a decoder reset event
    /// or between independent streams.
    /// </summary>
    public void Reset() => _state.Reset();

    /// <summary>
    /// Decode one SILK frame from a range-coded byte payload into 16-bit PCM.
    /// </summary>
    /// <param name="payload">Range-coded SILK frame bytes.</param>
    /// <param name="pcmOut">Output PCM buffer. Length &gt;= <see cref="FrameLength"/>.</param>
    /// <param name="vadFlag">VAD flag for the frame. Controls which signal-type iCDF is read.</param>
    /// <param name="conditional">True for conditional / delta coding (gains, NLSFs), false for independent.</param>
    /// <returns>Number of PCM samples written (always equal to <see cref="FrameLength"/>).</returns>
    public int DecodeFrame(ReadOnlySpan<byte> payload, Span<short> pcmOut, bool vadFlag, bool conditional)
    {
        if (payload.Length == 0) throw new ArgumentException("Empty payload.", nameof(payload));
        if (pcmOut.Length < _state.FrameLength)
            throw new ArgumentException(
                $"pcmOut too small (need {_state.FrameLength}).", nameof(pcmOut));

        var rangeDec = new OpusRangeDecoder(payload.ToArray());
        SilkDecodeFrame.Decode(_state, rangeDec, pcmOut, vadFlag, conditional ? 1 : 0);
        return _state.FrameLength;
    }
}
