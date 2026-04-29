// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable SILK NLSF table-index + predictor unpacker. Mirror of
// SilkNlsfUnpack.Unpack (libopus silk/NLSF_unpack.c). Decodes the
// packed ec_sel byte stream for a given first-stage codebook index
// into per-coefficient entropy-table indices + Q8 predictor values.
//
// Per-coefficient-pair parallel: each pair (i, i+1) reads one ec_sel
// byte and produces 2 ecIx + 2 predQ8 values independently. Caller
// dispatches order/2 threads.
//
// All silk macros (RSHIFT, SMULBB) inlined here.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK NLSF unpacker. Mirror of
/// <see cref="SilkNlsfUnpack"/>.Unpack.
/// </summary>
public static class SilkNlsfUnpackGpu
{
    /// <summary>
    /// 2 * NLSF_QUANT_MAX_AMPLITUDE + 1 = 9. Used as the ec_ix scale.
    /// </summary>
    private const int BOUND = 9;

    /// <summary>
    /// Unpack one coefficient pair at index <paramref name="pairIdx"/>.
    /// Writes ecIx[2*pairIdx] / [2*pairIdx+1] and predQ8[2*pairIdx] / [2*pairIdx+1].
    /// </summary>
    /// <param name="ecIx">Output: indices into entropy tables (length order).</param>
    /// <param name="ecIxBase">Base offset.</param>
    /// <param name="predQ8Out">Output: Q8 predictor values (length order).</param>
    /// <param name="predQ8OutBase">Base offset.</param>
    /// <param name="ecSel">Codebook EcSel bytes (length nVectors * order / 2).</param>
    /// <param name="ecSelBase">Base offset for the cb1Index entry: cb1Index * order/2.</param>
    /// <param name="predQ8Source">Codebook PredQ8 array (length 2 * (order - 1)).</param>
    /// <param name="predQ8SrcBase">Base offset.</param>
    /// <param name="order">Filter order (10 or 16).</param>
    /// <param name="pairIdx">Index of the coefficient pair (0..order/2 - 1).</param>
    public static void UnpackPairAt(
        ArrayView<short> ecIx, long ecIxBase,
        ArrayView<byte> predQ8Out, long predQ8OutBase,
        ArrayView<byte> ecSel, long ecSelBase,
        ArrayView<byte> predQ8Source, long predQ8SrcBase,
        int order, int pairIdx)
    {
        int i = 2 * pairIdx;
        byte entry = ecSel[ecSelBase + pairIdx];

        // ecIx[i]   = SMULBB((entry >> 1) & 7, BOUND)
        // ecIx[i+1] = SMULBB((entry >> 5) & 7, BOUND)
        int low3 = (entry >> 1) & 7;
        int high3 = (entry >> 5) & 7;
        ecIx[ecIxBase + i] = (short)(low3 * BOUND);
        ecIx[ecIxBase + i + 1] = (short)(high3 * BOUND);

        // predQ8[i]   = predQ8Source[i + (entry & 1) * (order - 1)]
        // predQ8[i+1] = predQ8Source[i + ((entry >> 4) & 1) * (order - 1) + 1]
        int variant0 = entry & 1;
        int variant1 = (entry >> 4) & 1;
        predQ8Out[predQ8OutBase + i] =
            predQ8Source[predQ8SrcBase + i + variant0 * (order - 1)];
        predQ8Out[predQ8OutBase + i + 1] =
            predQ8Source[predQ8SrcBase + i + variant1 * (order - 1) + 1];
    }
}
