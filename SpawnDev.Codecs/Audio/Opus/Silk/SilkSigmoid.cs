// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/sigm_Q15.c to clean C#.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.
//
// Approximate sigmoid (logistic) function with Q15 output and Q5 input. Used
// in various SILK estimation paths (VAD, speech activity). Six-entry LUTs plus
// linear interpolation produce ~16-bit accurate output with trivial compute.

using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Fast table-driven approximation of <c>1 / (1 + exp(-x))</c> in Q15 output.
/// Input is in Q5 format; output is in <c>[0, 32767]</c>. Linear interpolation
/// between LUT entries gives ~16-bit accuracy with two multiplies and an add.
/// </summary>
internal static class SilkSigmoid
{
    // From libopus sigm_Q15.c.
    //
    // sigm_LUT_slope_Q10[i] = round(1024 * (sigmoid(i+1) - sigmoid(i))) for i in [0, 5]
    // sigm_LUT_pos_Q15[i]   = round(32767 * sigmoid(i))                 for i in [0, 5]
    // sigm_LUT_neg_Q15[i]   = round(32767 * sigmoid(-i))                for i in [0, 5]

    private static readonly int[] sigm_LUT_slope_Q10 =
    {
        237, 153, 73, 30, 12, 7
    };

    private static readonly int[] sigm_LUT_pos_Q15 =
    {
        16384, 23955, 28861, 31213, 32178, 32548
    };

    private static readonly int[] sigm_LUT_neg_Q15 =
    {
        16384, 8812, 3906, 1554, 589, 219
    };

    /// <summary>
    /// Compute <c>sigmoid(in)</c> in Q15 for an input in Q5.
    /// </summary>
    /// <param name="inQ5">Input in Q5 format. Clipped to <c>[-6*32, 6*32)</c>.</param>
    /// <returns>Sigmoid output in Q15 in <c>[0, 32767]</c>.</returns>
    internal static int silk_sigm_Q15(int inQ5)
    {
        int ind;

        if (inQ5 < 0)
        {
            // Negative input.
            inQ5 = -inQ5;
            if (inQ5 >= 6 * 32)
            {
                return 0; // Clip.
            }
            else
            {
                ind = silk_RSHIFT(inQ5, 5);
                return sigm_LUT_neg_Q15[ind] - silk_SMULBB(sigm_LUT_slope_Q10[ind], inQ5 & 0x1F);
            }
        }
        else
        {
            // Positive input.
            if (inQ5 >= 6 * 32)
            {
                return 32767; // Clip.
            }
            else
            {
                ind = silk_RSHIFT(inQ5, 5);
                return sigm_LUT_pos_Q15[ind] + silk_SMULBB(sigm_LUT_slope_Q10[ind], inQ5 & 0x1F);
            }
        }
    }
}
