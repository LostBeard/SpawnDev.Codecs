// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable SILK 2x high-quality upsampler. Mirror of the private
// silk_resampler_private_up2_HQ inside libopus silk/resampler_private_up2_HQ.c.
// Doubles input sample count via 6 cascaded all-pass sections (3 even, 3 odd).
//
// Sequential per-stream: each output pair depends on the prior IIR state in
// S[0..5]. Per the cardinal rule, one-thread-per-stream on the GPU. Multiple
// independent streams (multi-channel decode) parallelize cleanly across threads.
//
// All silk macros (LSHIFT, SUB32, ADD32, SMULWB, SMLAWB, RSHIFT_ROUND, SAT16)
// are inlined here.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK 2x high-quality upsampler. Mirror of libopus
/// <c>silk_resampler_private_up2_HQ</c>.
/// </summary>
public static class SilkResamplerUp2HqGpu
{
    // Even-sample all-pass coefficients (Q15, mirrors libopus Up2Hq0).
    private const short COEF_EVEN_0 = 1746;
    private const short COEF_EVEN_1 = 14986;
    private const short COEF_EVEN_2 = unchecked((short)(39083 - 65536));

    // Odd-sample all-pass coefficients (Q15, mirrors libopus Up2Hq1).
    private const short COEF_ODD_0 = 6854;
    private const short COEF_ODD_1 = 25769;
    private const short COEF_ODD_2 = unchecked((short)(55542 - 65536));

    /// <summary>
    /// Run the 2x HQ upsampler over <paramref name="len"/> input samples,
    /// producing 2*len output samples. State buffer S holds 6 Q10 ints
    /// ([0..2] even branch, [3..5] odd branch); persisted across calls.
    /// Bit-exact vs the CPU SilkResampler.Up2Hq.
    /// </summary>
    public static void ApplyAt(
        ArrayView<int> state, long stateBase,
        ArrayView<short> output, long outBase,
        ArrayView<short> input, long inBase,
        int len)
    {
        int s0 = state[stateBase + 0];
        int s1 = state[stateBase + 1];
        int s2 = state[stateBase + 2];
        int s3 = state[stateBase + 3];
        int s4 = state[stateBase + 4];
        int s5 = state[stateBase + 5];

        for (int k = 0; k < len; k++)
        {
            int in32 = (int)input[inBase + k] << 10;

            // Even-sample branch: three all-pass sections using even coefs.
            int Y = in32 - s0;
            int X = SmulWB(Y, COEF_EVEN_0);
            int out32_1 = s0 + X;
            s0 = in32 + X;

            Y = out32_1 - s1;
            X = SmulWB(Y, COEF_EVEN_1);
            int out32_2 = s1 + X;
            s1 = out32_1 + X;

            Y = out32_2 - s2;
            X = SmlaWB(Y, Y, COEF_EVEN_2);
            out32_1 = s2 + X;
            s2 = out32_2 + X;

            output[outBase + 2 * k] = Sat16(RShiftRound(out32_1, 10));

            // Odd-sample branch: three all-pass sections using odd coefs.
            Y = in32 - s3;
            X = SmulWB(Y, COEF_ODD_0);
            out32_1 = s3 + X;
            s3 = in32 + X;

            Y = out32_1 - s4;
            X = SmulWB(Y, COEF_ODD_1);
            out32_2 = s4 + X;
            s4 = out32_1 + X;

            Y = out32_2 - s5;
            X = SmlaWB(Y, Y, COEF_ODD_2);
            out32_1 = s5 + X;
            s5 = out32_2 + X;

            output[outBase + 2 * k + 1] = Sat16(RShiftRound(out32_1, 10));
        }

        state[stateBase + 0] = s0;
        state[stateBase + 1] = s1;
        state[stateBase + 2] = s2;
        state[stateBase + 3] = s3;
        state[stateBase + 4] = s4;
        state[stateBase + 5] = s5;
    }

    /// <summary>silk_SMULWB(a, b) = (int)((long)a * (short)b >> 16).</summary>
    private static int SmulWB(int a32, short b16) =>
        (int)((long)a32 * b16 >> 16);

    /// <summary>silk_SMLAWB(c, a, b) = c + (int)((long)a * (short)b >> 16).</summary>
    private static int SmlaWB(int c32, int a32, short b16) =>
        c32 + (int)((long)a32 * b16 >> 16);

    /// <summary>silk_RSHIFT_ROUND for shift &gt;= 1.</summary>
    private static int RShiftRound(int a, int shift) =>
        ((a >> (shift - 1)) + 1) >> 1;

    /// <summary>silk_SAT16: saturate int to int16.</summary>
    private static short Sat16(int v)
    {
        if (v > short.MaxValue) return short.MaxValue;
        if (v < short.MinValue) return short.MinValue;
        return (short)v;
    }
}
