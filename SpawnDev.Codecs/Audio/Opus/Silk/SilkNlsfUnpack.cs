// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/NLSF_unpack.c to clean C#.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.
//
// For a given first-stage NLSF codebook index, produces the per-coefficient
// entropy-table indices and residual predictor values. Called as part of the
// per-frame NLSF decode path. Reads the packed <c>ec_sel</c> byte stream one
// byte per coefficient pair (two coefficients at a time) and decodes two
// 3-bit entropy-table indices plus two 1-bit predictor-variant selectors.

using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// NLSF table-index and predictor unpacker for a given first-stage codebook index.
/// </summary>
internal static class SilkNlsfUnpack
{
    /// <summary>
    /// Unpack predictor values and entropy-coding indices for the given first-stage
    /// codebook entry.
    /// </summary>
    /// <param name="ecIx">Output: indices into entropy tables. Length = <c>codebook.Order</c>.</param>
    /// <param name="predQ8">Output: NLSF residual predictor in Q8. Length = <c>codebook.Order</c>.</param>
    /// <param name="codebook">Codebook containing the <c>ec_sel</c> and <c>pred_Q8</c> backing arrays.</param>
    /// <param name="cb1Index">First-stage NLSF codebook index (in <c>[0, codebook.NVectors)</c>).</param>
    internal static void Unpack(
        Span<short> ecIx,
        Span<byte> predQ8,
        SilkNlsfCodebook codebook,
        int cb1Index)
    {
        int order = codebook.Order;
        if (ecIx.Length < order) throw new ArgumentException($"ecIx too small (need {order}).", nameof(ecIx));
        if (predQ8.Length < order) throw new ArgumentException($"predQ8 too small (need {order}).", nameof(predQ8));
        if ((uint)cb1Index >= (uint)codebook.NVectors)
            throw new ArgumentOutOfRangeException(nameof(cb1Index), "cb1Index out of range.");

        int ecSelBase = cb1Index * order / 2;
        int bound = 2 * SilkConstants.NLSF_QUANT_MAX_AMPLITUDE + 1;

        for (int i = 0; i < order; i += 2)
        {
            byte entry = codebook.EcSel[ecSelBase++];

            ecIx[i] = (short)silk_SMULBB(silk_RSHIFT(entry, 1) & 7, bound);
            predQ8[i] = codebook.PredQ8[i + (entry & 1) * (order - 1)];

            ecIx[i + 1] = (short)silk_SMULBB(silk_RSHIFT(entry, 5) & 7, bound);
            predQ8[i + 1] = codebook.PredQ8[i + (silk_RSHIFT(entry, 4) & 1) * (order - 1) + 1];
        }
    }
}
