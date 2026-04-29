// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable SILK NLSF inter-frame interpolation. Mirror of the
// per-coefficient interpolation block inside SilkParametersDecoder.Decode
// (libopus silk/decode_parameters.c). Generates the first-half NLSF vector
// for a frame as a Q2-scaled mix of the previous and current quantized
// NLSFs:
//
//     nlsf0Q15[i] = prevNlsfQ15[i] + ((interpCoefQ2 * (nlsfQ15[i] - prevNlsfQ15[i])) >> 2)
//
// Per-coefficient parallel: each thread reads prev[i] + cur[i], computes
// delta, scales by interpCoefQ2, writes nlsf0[i]. True parallel-per-
// coefficient across all 6 ILGPU backends.
//
// Caller dispatches order threads (10 for NB/MB, 16 for WB) and calls this
// only when interpCoefQ2 < 4 (interpCoefQ2 == 4 means "use current as-is",
// in which case the caller copies cur into nlsf0 directly without invoking
// this primitive).

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK NLSF interpolation. Mirror of the per-coefficient
/// interpolation block inside <see cref="SilkParametersDecoder"/>.Decode.
/// </summary>
public static class SilkNlsfInterpolateGpu
{
    /// <summary>
    /// Compute one interpolated NLSF coefficient at index <paramref name="i"/>.
    /// Bit-exact vs the CPU SilkParametersDecoder.Decode interpolation.
    /// </summary>
    /// <param name="nlsf0Q15">Output: interpolated NLSF (length order).</param>
    /// <param name="nlsf0Base">Base offset.</param>
    /// <param name="prevNlsfQ15">Previous frame's NLSFs (length order).</param>
    /// <param name="prevBase">Base offset.</param>
    /// <param name="curNlsfQ15">Current frame's NLSFs (length order).</param>
    /// <param name="curBase">Base offset.</param>
    /// <param name="interpCoefQ2">Interpolation coefficient in Q2, in [0, 4).</param>
    /// <param name="i">Coefficient index.</param>
    public static void InterpolateAt(
        ArrayView<short> nlsf0Q15, long nlsf0Base,
        ArrayView<short> prevNlsfQ15, long prevBase,
        ArrayView<short> curNlsfQ15, long curBase,
        int interpCoefQ2, int i)
    {
        int prev = prevNlsfQ15[prevBase + i];
        int cur = curNlsfQ15[curBase + i];
        int delta = cur - prev;

        // silk_RSHIFT(silk_MUL(interpCoefQ2, delta), 2) = (interpCoefQ2 * delta) >> 2
        int scaled = (interpCoefQ2 * delta) >> 2;
        nlsf0Q15[nlsf0Base + i] = (short)(prev + scaled);
    }
}
