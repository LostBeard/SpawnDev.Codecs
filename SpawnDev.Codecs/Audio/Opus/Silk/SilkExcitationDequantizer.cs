// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of the excitation-dequantization block inside libopus
// silk/decode_core.c. Converts decoded pulse magnitudes into a Q14 excitation
// signal with signal-type-dependent offset applied and PRNG-driven sign
// scrambling. Output feeds the LTP/LPC synthesis stages of decode_core.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Excitation dequantizer. Converts a frame of decoded pulses (from
/// <see cref="SilkPulsesDecoder"/>) into a Q14 excitation signal ready to drive the
/// SILK LTP + LPC synthesis filter chain.
/// <para>
/// Per-sample flow (matches libopus decode_core's excitation loop):
/// step the PRNG, left-shift the pulse by 14, apply the QUANT_LEVEL_ADJUST_Q10
/// nudge towards zero, add the signalType+offset-dependent offsetQ14, then flip
/// sign when the PRNG output is negative. The PRNG is re-seeded each step by
/// overflow-adding the pulse value, so runs of non-zero pulses scramble differently
/// from runs of zeros.
/// </para>
/// </summary>
internal static class SilkExcitationDequantizer
{
    /// <summary>
    /// Dequantize <paramref name="frameLength"/> pulse samples into the Q14 excitation
    /// buffer <paramref name="excQ14"/>.
    /// </summary>
    /// <param name="excQ14">Output: excitation in Q14, length &gt;= <paramref name="frameLength"/>.</param>
    /// <param name="pulses">Decoded pulse magnitudes (signed), length &gt;= <paramref name="frameLength"/>.</param>
    /// <param name="signalType">SILK signal type (0 inactive, 1 unvoiced, 2 voiced).</param>
    /// <param name="quantOffsetType">SILK quantizer offset type (0 low, 1 high).</param>
    /// <param name="seed">Initial PRNG seed (from <see cref="SilkDecodedIndices.Seed"/>).</param>
    /// <param name="frameLength">Frame length in samples.</param>
    internal static void Dequantize(
        Span<int> excQ14,
        ReadOnlySpan<short> pulses,
        int signalType,
        int quantOffsetType,
        int seed,
        int frameLength)
    {
        if ((uint)signalType > 2) throw new ArgumentOutOfRangeException(nameof(signalType));
        if ((uint)quantOffsetType > 1) throw new ArgumentOutOfRangeException(nameof(quantOffsetType));
        if (frameLength <= 0) throw new ArgumentOutOfRangeException(nameof(frameLength));
        if (excQ14.Length < frameLength) throw new ArgumentException($"excQ14 too small (need {frameLength}).", nameof(excQ14));
        if (pulses.Length < frameLength) throw new ArgumentException($"pulses too small (need {frameLength}).", nameof(pulses));

        int offsetQ10 = SilkConstants.QUANTIZATION_OFFSETS_Q10[signalType >> 1, quantOffsetType];
        int quantAdjustQ14 = SilkConstants.QUANT_LEVEL_ADJUST_Q10 << 4;
        int offsetQ14 = offsetQ10 << 4;

        int randSeed = seed;
        for (int i = 0; i < frameLength; i++)
        {
            randSeed = silk_RAND(randSeed);

            int exc = silk_LSHIFT((int)pulses[i], 14);
            if (exc > 0) exc -= quantAdjustQ14;
            else if (exc < 0) exc += quantAdjustQ14;
            exc += offsetQ14;

            if (randSeed < 0) exc = -exc;

            excQ14[i] = exc;

            randSeed = silk_ADD32_ovflw(randSeed, pulses[i]);
        }
    }
}
