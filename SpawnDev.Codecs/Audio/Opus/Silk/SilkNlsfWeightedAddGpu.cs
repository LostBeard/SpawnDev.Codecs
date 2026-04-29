// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable SILK NLSF weighted-add stage. Mirror of the per-coefficient
// inverse-weight + first-stage-add loop inside SilkNlsfDecoder.Decode
// (libopus silk/NLSF_decode.c). Combines the residual stream produced by
// SilkNlsfResidualDequantGpu with the codebook's first-stage NLSF +
// inverse weights to produce the Q15 NLSF vector.
//
// Per-coefficient parallel: each thread reads one resQ10 + one cb1 entry +
// one cbWght entry, divides, adds the lifted first-stage value, and clamps
// to [0, 32767]. True parallel-per-coefficient across all 6 ILGPU backends.
//
// All silk macros (LSHIFT, DIV32_16, ADD_LSHIFT32, LIMIT_32) inlined.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK NLSF weighted-add stage. Mirror of the per-coefficient
/// loop inside <see cref="SilkNlsfDecoder"/>.Decode that turns
/// (resQ10, cb1, cbWght) into pNlsfQ15.
/// </summary>
public static class SilkNlsfWeightedAddGpu
{
    /// <summary>
    /// Compute one NLSF coefficient at index <paramref name="i"/>:
    /// <c>pNlsfQ15[i] = LIMIT_32((resQ10[i] &lt;&lt; 14) / cbWght[cbBase + i] +
    /// (cb1[cbBase + i] &lt;&lt; 7), 0, 32767)</c>.
    /// </summary>
    /// <param name="pNlsfQ15">Output: NLSF in Q15 (length order).</param>
    /// <param name="nlsfBase">Base offset.</param>
    /// <param name="resQ10">Residuals in Q10 from <see cref="SilkNlsfResidualDequantGpu"/> (length order).</param>
    /// <param name="resBase">Base offset.</param>
    /// <param name="cb1">Codebook first-stage NLSFs in Q8 (length nVectors * order).</param>
    /// <param name="cb1Base">Base offset for the selected first-stage entry: cb1Index * order.</param>
    /// <param name="cbWght">Codebook inverse weights in Q9 (length nVectors * order).</param>
    /// <param name="cbWghtBase">Base offset for the selected first-stage entry: cb1Index * order.</param>
    /// <param name="i">Coefficient index in [0, order).</param>
    public static void ApplyAt(
        ArrayView<short> pNlsfQ15, long nlsfBase,
        ArrayView<short> resQ10, long resBase,
        ArrayView<byte> cb1, long cb1Base,
        ArrayView<short> cbWght, long cbWghtBase,
        int i)
    {
        // residual = resQ10[i] << 14
        int residual = (int)resQ10[resBase + i] << 14;

        // silk_DIV32_16: int / short. cbWght entries are positive shorts.
        int weight = cbWght[cbWghtBase + i];
        int weightedResidual = residual / weight;

        // silk_ADD_LSHIFT32(weightedResidual, (short)cb1[i], 7) =
        //   weightedResidual + ((int)cb1[i] << 7)
        int cb1Val = cb1[cb1Base + i]; // byte 0..255 promotes to int 0..255
        int nlsfQ15Tmp = weightedResidual + (cb1Val << 7);

        // silk_LIMIT_32(x, 0, 32767)
        if (nlsfQ15Tmp < 0) nlsfQ15Tmp = 0;
        else if (nlsfQ15Tmp > 32767) nlsfQ15Tmp = 32767;

        pNlsfQ15[nlsfBase + i] = (short)nlsfQ15Tmp;
    }
}
