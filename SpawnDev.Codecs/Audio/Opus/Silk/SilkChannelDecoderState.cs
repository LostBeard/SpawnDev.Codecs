// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Persistent SILK channel-decoder state carried between per-frame decodes.
// Corresponds to the subset of libopus silk/structs.h::silk_decoder_state that
// the current decoder slices need: per-frame gain index, NLSFs, previous lag
// index, previous signal type.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Inter-frame state for a single SILK channel (mono or one side of stereo).
/// Created once per stream and updated in place by each frame's decode call.
/// Exposes the two ref-taken scalars (<see cref="LastGainIndex"/>, <see cref="PrevLagIndex"/>)
/// as public fields so callers can pass them by reference into the decoder methods.
/// </summary>
internal sealed class SilkChannelDecoderState
{
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

    /// <summary>
    /// Previous frame's pitch lag index (for delta pitch coding when the current frame
    /// is voiced and conditional + the previous frame was voiced).
    /// </summary>
    public short PrevLagIndex;

    /// <summary>
    /// Whether the previous frame was voiced. Gates delta pitch-lag coding.
    /// </summary>
    public bool PrevSignalTypeWasVoiced;

    /// <summary>
    /// Reset the state to first-frame defaults. Called on codec init / reset events.
    /// </summary>
    public void Reset()
    {
        LastGainIndex = 0;
        PrevLagIndex = 0;
        PrevSignalTypeWasVoiced = false;
        Array.Clear(PrevNlsfQ15, 0, PrevNlsfQ15.Length);
    }
}
