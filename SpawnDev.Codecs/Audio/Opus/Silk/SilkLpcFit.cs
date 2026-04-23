// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/LPC_fit.c to clean C#. Converts int32 LPC
// prediction coefficients to int16 with no overflow, applying bandwidth
// expansion iteratively if needed to shrink coefficients that would saturate.
//
// Upstream Copyright (c) 2013 Koen Vos. BSD 3-Clause. See NOTICE.md.
//
// Per upstream comment: this logic is also used by CELT's _celt_lpc() - any
// bug fix here needs to be mirrored when we port CELT.

using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Quantizes int32 LPC coefficients to int16 while preserving filter stability.
/// If any coefficient overflows int16 range at the target Q format, applies
/// bandwidth expansion to all coefficients and retries (up to 10 iterations).
/// If still overflowing after 10 attempts, saturates the output and writes the
/// saturated values back into the int32 input buffer too (so subsequent
/// stability checks see a consistent state).
/// </summary>
internal static class SilkLpcFit
{
    /// <summary>SILK_FIX_CONST(0.999, 16) = (int)(0.999 * 65536 + 0.5) = 65470.</summary>
    private const int CHIRP_INITIAL_Q16 = 65470;

    /// <summary>Upper bound on maxabs per libopus comment: (silk_int32_MAX &gt;&gt; 14) + silk_int16_MAX.</summary>
    private const int MAXABS_CAP = 163838;

    /// <summary>
    /// Fit <paramref name="aQIn"/> (in Q <paramref name="qIn"/> format) into
    /// <paramref name="aQOut"/> (in Q <paramref name="qOut"/> format) as int16
    /// with iterative bandwidth expansion on overflow.
    /// </summary>
    /// <param name="aQOut">Output int16 LPC coefficients in Q<paramref name="qOut"/>. Length <paramref name="d"/>.</param>
    /// <param name="aQIn">In/out int32 LPC coefficients in Q<paramref name="qIn"/>. Length <paramref name="d"/>.
    /// May be modified (bwexpand / saturation) during the fit.</param>
    /// <param name="qOut">Output Q format (typically 12).</param>
    /// <param name="qIn">Input Q format.</param>
    /// <param name="d">Filter order.</param>
    internal static void Fit(Span<short> aQOut, Span<int> aQIn, int qOut, int qIn, int d)
    {
        if (aQOut.Length < d) throw new ArgumentException($"aQOut too small (need {d}).", nameof(aQOut));
        if (aQIn.Length < d) throw new ArgumentException($"aQIn too small (need {d}).", nameof(aQIn));

        int idx = 0;
        int i;
        for (i = 0; i < 10; i++)
        {
            // Find the maximum absolute value and its index.
            int maxabs = 0;
            for (int k = 0; k < d; k++)
            {
                int absval = silk_abs(aQIn[k]);
                if (absval > maxabs)
                {
                    maxabs = absval;
                    idx = k;
                }
            }
            maxabs = silk_RSHIFT_ROUND(maxabs, qIn - qOut);

            if (maxabs > silk_int16_MAX)
            {
                // Reduce magnitude of prediction coefficients via bandwidth expansion.
                maxabs = silk_min(maxabs, MAXABS_CAP);
                int chirpQ16 = CHIRP_INITIAL_Q16 - silk_DIV32(
                    silk_LSHIFT(maxabs - silk_int16_MAX, 14),
                    silk_RSHIFT32(silk_MUL(maxabs, idx + 1), 2));
                SilkBwexpander.Expand32(aQIn.Slice(0, d), chirpQ16);
            }
            else
            {
                break;
            }
        }

        if (i == 10)
        {
            // Reached the last iteration - clip the coefficients and write back into aQIn.
            for (int k = 0; k < d; k++)
            {
                aQOut[k] = silk_SAT16(silk_RSHIFT_ROUND(aQIn[k], qIn - qOut));
                aQIn[k] = silk_LSHIFT((int)aQOut[k], qIn - qOut);
            }
        }
        else
        {
            for (int k = 0; k < d; k++)
            {
                aQOut[k] = (short)silk_RSHIFT_ROUND(aQIn[k], qIn - qOut);
            }
        }
    }
}
