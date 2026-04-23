// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Persistent SILK channel-decoder state carried between per-frame decodes.
// Corresponds to the subset of libopus silk/structs.h::silk_decoder_state that
// the decode pipeline currently needs: side-info scalars, NLSF history, and
// the long-form buffers used during decode_core synthesis.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Inter-frame state for a single SILK channel (mono or one side of stereo).
/// Created once per stream and updated in place by each frame's decode call.
/// Exposes scalar state as public fields so callers can pass them by reference
/// into the per-sub-system decoder methods, matching how libopus uses its
/// <c>silk_decoder_state</c> struct.
///
/// The buffer fields (<see cref="OutBuf"/>, <see cref="SLtpQ15"/>, <see cref="SLpcQ14Buf"/>,
/// <see cref="ExcQ14"/>) are sized to worst-case <c>MAX_FS_KHZ</c> dimensions. Call
/// <see cref="Configure"/> to set the active frame geometry; the buffers are reused across
/// frames so allocations happen once per stream.
/// </summary>
internal sealed class SilkChannelDecoderState
{
    // -------- Side-info / index-level state --------

    /// <summary>
    /// Previous frame's last gain index. Used by the conditional gain decoder for
    /// delta coding of the first subframe. Starts at 0 for the first frame.
    /// </summary>
    public sbyte LastGainIndex;

    /// <summary>
    /// Previous frame's NLSFs in Q15. Used by the NLSF interpolation step for the
    /// first half of the next frame. Buffered at <see cref="SilkConstants.MAX_LPC_ORDER"/>
    /// values; callers populate only the first <c>order</c> per the active codebook.
    /// </summary>
    public readonly short[] PrevNlsfQ15 = new short[SilkConstants.MAX_LPC_ORDER];

    /// <summary>Previous frame's pitch lag index (for delta pitch coding).</summary>
    public short PrevLagIndex;

    /// <summary>Whether the previous frame was voiced. Gates delta pitch-lag coding.</summary>
    public bool PrevSignalTypeWasVoiced;

    // -------- Decode_core synthesis state --------

    /// <summary>Internal SILK sample rate in kHz (8, 12, or 16). 0 means unconfigured.</summary>
    public int FsKHz;

    /// <summary>LPC filter order (10 for NB/MB, 16 for WB). 0 means unconfigured.</summary>
    public int LpcOrder;

    /// <summary>Subframe count (2 or 4). 0 means unconfigured.</summary>
    public int NbSubfr;

    /// <summary>Subframe length in samples (<c>SUB_FRAME_LENGTH_MS * fs_kHz</c>).</summary>
    public int SubfrLength;

    /// <summary>Frame length in samples (<c>NbSubfr * SubfrLength</c>).</summary>
    public int FrameLength;

    /// <summary>LTP buffer length in samples (<c>LTP_MEM_LENGTH_MS * fs_kHz</c>).</summary>
    public int LtpMemLength;

    /// <summary>
    /// LTP output buffer: holds the previous frame's output PCM concatenated with
    /// scratch space for the current frame. Length = <see cref="LtpMemLength"/> + <see cref="FrameLength"/>.
    /// Allocated at <see cref="SilkConstants.MAX_LTP_MEM_LENGTH"/> + <see cref="SilkConstants.MAX_FRAME_LENGTH"/>.
    /// </summary>
    public readonly short[] OutBuf = new short[SilkConstants.MAX_LTP_MEM_LENGTH + SilkConstants.MAX_FRAME_LENGTH];

    /// <summary>
    /// LTP Q15 state buffer. Allocated at <see cref="SilkConstants.MAX_LTP_MEM_LENGTH"/> + <see cref="SilkConstants.MAX_FRAME_LENGTH"/>.
    /// </summary>
    public readonly int[] SLtpQ15 = new int[SilkConstants.MAX_LTP_MEM_LENGTH + SilkConstants.MAX_FRAME_LENGTH];

    /// <summary>
    /// LPC filter state carried between subframes / frames. Length =
    /// <see cref="SilkConstants.MAX_LPC_ORDER"/>. Values are in Q14.
    /// </summary>
    public readonly int[] SLpcQ14Buf = new int[SilkConstants.MAX_LPC_ORDER];

    /// <summary>
    /// Excitation (pulse) buffer in Q14 for the current frame. Allocated at the worst
    /// case <see cref="SilkConstants.MAX_FRAME_LENGTH"/>.
    /// </summary>
    public readonly int[] ExcQ14 = new int[SilkConstants.MAX_FRAME_LENGTH];

    /// <summary>Previous frame's last-subframe gain in Q16. Used to scale LTP state across gain changes.</summary>
    public int PrevGainQ16 = 65536; // 1.0 Q16 = libopus init value per silk_init_decoder

    /// <summary>
    /// Packet-loss counter. Incremented on each lost packet (not yet wired up by this code);
    /// consumers use it to trigger BWE of LPC coefficients after losses.
    /// </summary>
    public int LossCnt;

    /// <summary>
    /// Signal type carried from the previous frame (for packet-loss handling in decode_core).
    /// </summary>
    public int PrevSignalType;

    /// <summary>
    /// True when the current frame is the first after a codec reset. Used to suppress NLSF
    /// interpolation on the first frame (otherwise the prev NLSFs would be all zeros).
    /// </summary>
    public bool FirstFrameAfterReset;

    /// <summary>Previous frame's last pitch lag value. Used by decode_core PLC only.</summary>
    public int LagPrev;

    /// <summary>
    /// Set the active frame geometry. Safe to call on every frame; only recomputes derived
    /// sizes. Does NOT clear the buffers - <see cref="Reset"/> does that.
    /// </summary>
    public void Configure(int fsKHz, int nbSubfr, int lpcOrder)
    {
        if (fsKHz != 8 && fsKHz != 12 && fsKHz != 16)
            throw new ArgumentException($"Unsupported fs_kHz: {fsKHz}.", nameof(fsKHz));
        if (nbSubfr != 2 && nbSubfr != 4)
            throw new ArgumentException($"nbSubfr must be 2 or 4, got {nbSubfr}.", nameof(nbSubfr));
        if (lpcOrder != 10 && lpcOrder != 16)
            throw new ArgumentException($"lpcOrder must be 10 or 16, got {lpcOrder}.", nameof(lpcOrder));

        FsKHz = fsKHz;
        NbSubfr = nbSubfr;
        LpcOrder = lpcOrder;
        SubfrLength = SilkConstants.SUB_FRAME_LENGTH_MS * fsKHz;
        FrameLength = nbSubfr * SubfrLength;
        LtpMemLength = SilkConstants.LTP_MEM_LENGTH_MS * fsKHz;
    }

    /// <summary>
    /// Reset the state to first-frame defaults. Called on codec initialization and after
    /// a decoder-reset event. Does NOT touch <see cref="FsKHz"/> / <see cref="NbSubfr"/>; those are set via <see cref="Configure"/>.
    /// </summary>
    public void Reset()
    {
        LastGainIndex = 0;
        PrevLagIndex = 0;
        PrevSignalTypeWasVoiced = false;
        Array.Clear(PrevNlsfQ15, 0, PrevNlsfQ15.Length);

        Array.Clear(OutBuf, 0, OutBuf.Length);
        Array.Clear(SLtpQ15, 0, SLtpQ15.Length);
        Array.Clear(SLpcQ14Buf, 0, SLpcQ14Buf.Length);
        Array.Clear(ExcQ14, 0, ExcQ14.Length);

        PrevGainQ16 = 65536; // 1.0 Q16, libopus silk_init_decoder default
        LossCnt = 0;
        PrevSignalType = SilkConstants.TYPE_NO_VOICE_ACTIVITY;
        FirstFrameAfterReset = true;
        LagPrev = 100; // libopus init value
    }
}
