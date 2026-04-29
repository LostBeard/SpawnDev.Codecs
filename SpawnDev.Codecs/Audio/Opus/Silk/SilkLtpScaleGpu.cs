// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable SILK LTP scale lookup. Mirror of the LTP scale lookup
// inside SilkParametersDecoder.Decode (libopus silk/decode_parameters.c).
// Resolves the 3-entry silk_LTPScales_table_Q14 (= { 15565, 12288, 8192 })
// for a given index and writes the scalar Q14 value.
//
// Single-thread on the GPU since the work is one table read. Useful as a
// composable primitive in the per-frame parameter decode pipeline so the
// host doesn't need to do the lookup itself.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK LTP scale lookup. Mirror of the LTP scale step in
/// <see cref="SilkParametersDecoder"/>.Decode.
/// </summary>
public static class SilkLtpScaleGpu
{
    /// <summary>libopus silk_LTPScales_table_Q14 entry 0.</summary>
    private const short LTP_SCALE_0 = 15565;
    /// <summary>libopus silk_LTPScales_table_Q14 entry 1.</summary>
    private const short LTP_SCALE_1 = 12288;
    /// <summary>libopus silk_LTPScales_table_Q14 entry 2.</summary>
    private const short LTP_SCALE_2 = 8192;

    /// <summary>
    /// Look up the LTP scale Q14 for <paramref name="ltpScaleIndex"/> in [0, 3) and
    /// write it to <paramref name="ltpScaleQ14Out"/> at <paramref name="outBase"/>.
    /// Bit-exact vs SilkParametersDecoder.Decode's LtpScalesQ14[ltpScaleIndex].
    /// </summary>
    /// <param name="ltpScaleQ14Out">Output: 1-int buffer for the Q14 scale.</param>
    /// <param name="outBase">Base offset.</param>
    /// <param name="ltpScaleIndex">LTP scale index (0..2 for voiced; out-of-range
    /// is permitted and yields 0 to mirror the unvoiced fallback).</param>
    public static void LookupAt(
        ArrayView<int> ltpScaleQ14Out, long outBase,
        int ltpScaleIndex)
    {
        int v;
        if (ltpScaleIndex == 0) v = LTP_SCALE_0;
        else if (ltpScaleIndex == 1) v = LTP_SCALE_1;
        else if (ltpScaleIndex == 2) v = LTP_SCALE_2;
        else v = 0;

        ltpScaleQ14Out[outBase] = v;
    }
}
