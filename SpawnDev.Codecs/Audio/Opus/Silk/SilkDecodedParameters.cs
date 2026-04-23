// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// C# equivalent of libopus silk/structs.h::silk_decoder_control dequantized
// parameter set. Holds one frame worth of decoded Q-format values produced by
// silk_decode_parameters.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Per-frame dequantized SILK parameters as produced by <see cref="SilkParametersDecoder"/>.
/// Mirrors the subset of <c>silk_decoder_control</c> the decoder uses in decode_core:
/// gains, NLSFs, two halves' worth of LPC coefficients, pitch lags, LTP filter taps,
/// and LTP scale factor.
/// </summary>
internal sealed class SilkDecodedParameters
{
    /// <summary>Per-subframe linear gains in Q16. Length = <see cref="SilkConstants.MAX_NB_SUBFR"/>.</summary>
    public int[] GainsQ16 { get; } = new int[SilkConstants.MAX_NB_SUBFR];

    /// <summary>Current-frame NLSFs in Q15. Length = <see cref="SilkConstants.MAX_LPC_ORDER"/>.</summary>
    public short[] NlsfQ15 { get; } = new short[SilkConstants.MAX_LPC_ORDER];

    /// <summary>
    /// LPC coefficients in Q12 for the two halves of the frame. Flat layout:
    /// index <c>half * MAX_LPC_ORDER + k</c> where <c>half</c> is 0 (first half) or 1
    /// (second half) and <c>k</c> is the coefficient index in <c>[0, order)</c>.
    /// </summary>
    public short[] PredCoefQ12 { get; } = new short[2 * SilkConstants.MAX_LPC_ORDER];

    /// <summary>Per-subframe pitch lags in samples. Length = <see cref="SilkConstants.MAX_NB_SUBFR"/>.</summary>
    public int[] PitchL { get; } = new int[SilkConstants.MAX_NB_SUBFR];

    /// <summary>
    /// Per-subframe LTP filter taps in Q14. Flat layout: index
    /// <c>k * 5 + i</c> where <c>k</c> is the subframe and <c>i</c> is the tap (0..4).
    /// </summary>
    public short[] LtpCoefQ14 { get; } = new short[SilkConstants.MAX_NB_SUBFR * SilkLtpGainTables.LtpVecSize];

    /// <summary>LTP scale factor in Q14. 0 for non-voiced frames.</summary>
    public int LtpScaleQ14 { get; set; }
}
