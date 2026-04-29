// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable SILK LPC coefficient fit. Mirror of SilkLpcFit.Fit
// (libopus silk/LPC_fit.c). Quantises higher-Q int32 LPC coefficients
// down to int16 in qOut format with iterative bandwidth-expansion on
// overflow.
//
// Sequential per-stream because each iteration of the bwexpand loop
// reads + writes the entire aQIn[0..d) coefficient array, and the
// next iteration depends on the previous expansion result. One-thread-
// per-stream on the GPU.
//
// Calls SilkBwexpanderGpu.Expand32 inline. All silk macros (abs,
// RSHIFT_ROUND, LSHIFT, RSHIFT32, MUL, DIV32, min, SAT16) inlined.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// GPU-callable SILK LPC coefficient fit. Mirror of
/// <see cref="SilkLpcFit"/>.Fit.
/// </summary>
public static class SilkLpcFitGpu
{
    private const int CHIRP_INITIAL_Q16 = 65470;
    private const int MAXABS_CAP = 163838;
    private const int INT16_MAX = 32767;

    /// <summary>
    /// Quantise int32 LPC coefficients in Q(<paramref name="qIn"/>) down to
    /// int16 in Q(<paramref name="qOut"/>), iterating bandwidth expansion
    /// up to 10 times when the maximum-magnitude tap overflows int16. Bit-exact
    /// vs the CPU SilkLpcFit.Fit.
    /// </summary>
    /// <param name="aQOut">Output int16 LPC coefs (length d).</param>
    /// <param name="aOutBase">Base offset.</param>
    /// <param name="aQIn">In/out int32 LPC coefs (length d). May be expanded in place.</param>
    /// <param name="aInBase">Base offset.</param>
    /// <param name="qOut">Output Q format (typically 12).</param>
    /// <param name="qIn">Input Q format.</param>
    /// <param name="d">Filter order.</param>
    public static void FitAt(
        ArrayView<short> aQOut, long aOutBase,
        ArrayView<int> aQIn, long aInBase,
        int qOut, int qIn, int d)
    {
        int idx = 0;
        int i;
        for (i = 0; i < 10; i++)
        {
            int maxabs = 0;
            for (int k = 0; k < d; k++)
            {
                int v = aQIn[aInBase + k];
                int absval = v < 0 ? unchecked(-v) : v;
                if (absval > maxabs)
                {
                    maxabs = absval;
                    idx = k;
                }
            }
            maxabs = RShiftRound(maxabs, qIn - qOut);

            if (maxabs > INT16_MAX)
            {
                int capped = maxabs < MAXABS_CAP ? maxabs : MAXABS_CAP;
                int numerator = (capped - INT16_MAX) << 14;
                int denominator = (capped * (idx + 1)) >> 2;
                int chirpQ16 = CHIRP_INITIAL_Q16 - numerator / denominator;
                SilkBwexpanderGpu.Expand32(aQIn, aInBase, d, chirpQ16);
            }
            else
            {
                break;
            }
        }

        if (i == 10)
        {
            // Last iteration: clip + write both aQOut and aQIn.
            for (int k = 0; k < d; k++)
            {
                int rounded = RShiftRound(aQIn[aInBase + k], qIn - qOut);
                if (rounded > INT16_MAX) rounded = INT16_MAX;
                else if (rounded < -INT16_MAX - 1) rounded = -INT16_MAX - 1;
                short shortVal = (short)rounded;
                aQOut[aOutBase + k] = shortVal;
                aQIn[aInBase + k] = (int)shortVal << (qIn - qOut);
            }
        }
        else
        {
            for (int k = 0; k < d; k++)
            {
                aQOut[aOutBase + k] = (short)RShiftRound(aQIn[aInBase + k], qIn - qOut);
            }
        }
    }

    /// <summary>silk_RSHIFT_ROUND for arbitrary shift &gt;= 0.</summary>
    private static int RShiftRound(int a, int shift)
    {
        if (shift <= 0) return a;
        if (shift == 1) return (a >> 1) + (a & 1);
        return ((a >> (shift - 1)) + 1) >> 1;
    }
}
