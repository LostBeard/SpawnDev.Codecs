// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable SILK gain dequantizer. Mirror of
// SilkGainDecoder.Dequantize (libopus silk/gain_quant.c::silk_gains_dequant).
// Converts per-subframe gain indices recovered from the range-coded
// bitstream into linear gains in Q16 format.
//
// Sequential per-stream because each subframe's prevInd updates from the
// previous one. One-thread-per-stream on the GPU. Multiple independent
// SILK streams (multi-channel decode) parallelize cleanly across threads.
//
// Composes SilkLog2Gpu.Log2Lin internally for the final lin scaling.
// All silk macros (max_int, LSHIFT, LIMIT_int, SMULWB, min_32) inlined.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK gain dequantizer. Mirror of
/// <see cref="SilkGainDecoder"/>.Dequantize.
/// </summary>
public static class SilkGainDecoderGpu
{
    private const int N_LEVELS_QGAIN = 64;
    private const int MAX_DELTA_GAIN_QUANT = 36;
    private const int MIN_DELTA_GAIN_QUANT = -4;
    private const int GAIN_OFFSET_Q7 = 2090;
    private const int GAIN_INV_SCALE_Q16 = 1907825;
    private const int GAIN_LOG_CLAMP_HIGH_Q7 = 3967;

    /// <summary>
    /// Dequantize <paramref name="nbSubfr"/> gain indices into linear gains in Q16.
    /// Bit-exact vs the CPU SilkGainDecoder.Dequantize.
    /// </summary>
    /// <param name="gainQ16">Output: per-subframe gains in Q16. Length &gt;= nbSubfr.</param>
    /// <param name="gainBase">Base offset.</param>
    /// <param name="ind">Input: gain indices (sbyte). Length &gt;= nbSubfr.</param>
    /// <param name="indBase">Base offset.</param>
    /// <param name="prevIndOut">In/out: 1-int buffer holding prevInd at [0]. Updated to the
    /// last index produced by this call.</param>
    /// <param name="prevIndBase">Base offset.</param>
    /// <param name="conditional">If 1, the first gain is delta-coded from prevInd; if 0,
    /// it is a full index.</param>
    /// <param name="nbSubfr">Number of subframes to dequantize (2 or 4).</param>
    public static void DequantizeAt(
        ArrayView<int> gainQ16, long gainBase,
        ArrayView<sbyte> ind, long indBase,
        ArrayView<int> prevIndOut, long prevIndBase,
        int conditional, int nbSubfr)
    {
        int prevInd = prevIndOut[prevIndBase];

        for (int k = 0; k < nbSubfr; k++)
        {
            int curInd;
            if (k == 0 && conditional == 0)
            {
                // Gain index not allowed to drop more than 16 steps (~21.8 dB).
                int candidate = ind[indBase + k];
                int floor = prevInd - 16;
                curInd = candidate > floor ? candidate : floor;
            }
            else
            {
                int indTmp = ind[indBase + k] + MIN_DELTA_GAIN_QUANT;
                int doubleStepThreshold =
                    2 * MAX_DELTA_GAIN_QUANT - N_LEVELS_QGAIN + prevInd;

                if (indTmp > doubleStepThreshold)
                {
                    curInd = prevInd + (indTmp << 1) - doubleStepThreshold;
                }
                else
                {
                    curInd = prevInd + indTmp;
                }
            }

            // silk_LIMIT_int(curInd, 0, N_LEVELS_QGAIN - 1)
            if (curInd < 0) curInd = 0;
            else if (curInd > N_LEVELS_QGAIN - 1) curInd = N_LEVELS_QGAIN - 1;
            prevInd = curInd;

            // silk_SMULWB(GAIN_INV_SCALE_Q16, prevInd) + GAIN_OFFSET_Q7, clamped to GAIN_LOG_CLAMP_HIGH_Q7
            int smulwb = (int)((long)GAIN_INV_SCALE_Q16 * (short)prevInd >> 16);
            int inLogQ7 = smulwb + GAIN_OFFSET_Q7;
            if (inLogQ7 > GAIN_LOG_CLAMP_HIGH_Q7) inLogQ7 = GAIN_LOG_CLAMP_HIGH_Q7;

            gainQ16[gainBase + k] = SilkLog2Gpu.Log2LinQ7(inLogQ7);
        }

        prevIndOut[prevIndBase] = prevInd;
    }
}
