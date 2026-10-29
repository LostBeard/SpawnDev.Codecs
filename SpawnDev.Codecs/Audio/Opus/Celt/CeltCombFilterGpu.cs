// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// The CELT pitch comb filter on the accelerator (libopus celt/celt.c comb_filter + comb_filter_const), fixed-point build: the
// decoder's post-filter and the encoder's pre-filter (the encoder runs it with negated gains). Over the first `overlap` samples
// it cross-fades from the previous frame's (period T0, gain g0, tapset0) to this frame's (T1, g1, tapset1) with the squared
// MDCT window, then runs the constant filter. Ported statement for statement from Concentus 2.2.2 (BSD-3) so the results are
// bit-exact with it, including in place (y and x the same view and offset, as the decoder calls it).

using ILGPU;
using static SpawnDev.Codecs.Audio.Opus.Celt.CeltFixed;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>CELT pitch pre / post comb filter (GPU-callable).</summary>
public static class CeltCombFilterGpu
{
    /// <summary>The three-tap gain sets (libopus comb_filter gains[tapset][tap], Q15; a short[][] in libopus / Concentus).</summary>
    private static short Gain(int tapset, int tap)
    {
        if (tapset == 0) return tap == 0 ? (short)10048 : tap == 1 ? (short)7112 : (short)4248;
        if (tapset == 1) return tap == 0 ? (short)15200 : tap == 1 ? (short)8784 : (short)0;
        return tap == 0 ? (short)26208 : tap == 1 ? (short)3280 : (short)0;
    }

    /// <summary>The constant-period part of the filter (libopus comb_filter_const). Reads x from <c>xAt - T - 2</c>.</summary>
    public static void CombFilterConst(ArrayView<int> y, int yAt, ArrayView<int> x, int xAt, int T, int N, int g10, int g11, int g12)
    {
        int num = xAt - T;
        int b = x[num - 2];
        int num2 = x[num - 1];
        int num3 = x[num];
        int num4 = x[num + 1];
        for (int i = 0; i < N; i++)
        {
            int num5 = x[num + i + 2];
            y[yAt + i] = x[xAt + i] + MULT16_32_Q15(g10, num3) + MULT16_32_Q15(g11, ADD32(num4, num2)) + MULT16_32_Q15(g12, ADD32(num5, b));
            b = num2;
            num2 = num3;
            num3 = num4;
            num4 = num5;
        }
    }

    /// <summary>
    /// The comb filter (libopus comb_filter): N samples at <paramref name="yAt"/> from x at <paramref name="xAt"/> (x must hold
    /// max(T0, T1) + 2 samples of history before <paramref name="xAt"/>). Nothing is written when both gains are 0.
    /// <paramref name="overlap"/> is the cross-fade length (the mode's overlap, 120); the window is the mode's
    /// (<see cref="CeltModeTables.Window"/>).
    /// </summary>
    public static void CombFilter(CeltModeGpu m, ArrayView<int> y, int yAt, ArrayView<int> x, int xAt, int T0, int T1, int N,
        int g0, int g1, int tapset0, int tapset1, int overlap)
    {
        if (g0 != 0 || g1 != 0)
        {
            int b = MULT16_16_P15(g0, Gain(tapset0, 0));
            int b2 = MULT16_16_P15(g0, Gain(tapset0, 1));
            int b3 = MULT16_16_P15(g0, Gain(tapset0, 2));
            int num = MULT16_16_P15(g1, Gain(tapset1, 0));
            int num2 = MULT16_16_P15(g1, Gain(tapset1, 1));
            int num3 = MULT16_16_P15(g1, Gain(tapset1, 2));
            int num4 = x[xAt - T1 + 1];
            int num5 = x[xAt - T1];
            int num6 = x[xAt - T1 - 1];
            int b4 = x[xAt - T1 - 2];
            if (g0 == g1 && T0 == T1 && tapset0 == tapset1)
            {
                overlap = 0;
            }
            int i;
            for (i = 0; i < overlap; i++)
            {
                int num7 = x[xAt + i - T1 + 2];
                int num8 = MULT16_16_Q15(m.WindowOf(i), m.WindowOf(i));
                y[yAt + i] = x[xAt + i] + MULT16_32_Q15(MULT16_16_Q15((short)(32767 - num8), b), x[xAt + i - T0])
                    + MULT16_32_Q15(MULT16_16_Q15((short)(32767 - num8), b2), ADD32(x[xAt + i - T0 + 1], x[xAt + i - T0 - 1]))
                    + MULT16_32_Q15(MULT16_16_Q15((short)(32767 - num8), b3), ADD32(x[xAt + i - T0 + 2], x[xAt + i - T0 - 2]))
                    + MULT16_32_Q15(MULT16_16_Q15(num8, num), num5)
                    + MULT16_32_Q15(MULT16_16_Q15(num8, num2), ADD32(num4, num6))
                    + MULT16_32_Q15(MULT16_16_Q15(num8, num3), ADD32(num7, b4));
                b4 = num6;
                num6 = num5;
                num5 = num4;
                num4 = num7;
            }
            if (g1 != 0)
            {
                CombFilterConst(y, yAt + i, x, xAt + i, T1, N - i, num, num2, num3);
            }
        }
    }
}
