// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 4-point forward Asymmetric DST, GPU-callable form for in-kernel
// reuse. Bit-exact mirror of Vp9ForwardAdst4.Transform (libvpx
// vp9/encoder/vp9_dct.c fadst4 port).
//
// Pairs with the existing Vp9Iadst4x4Gpu (decoder side, just shipped) -
// now both directions of the VP9 4-point ADST have GPU primitives.
//
// Uses 64-bit intermediates because per-stage magnitudes can exceed
// 32-bit range before the final round_shift. Inlines the 4 sinpi
// constants from vpx_dsp/txfm_common.h.

using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// GPU-callable VP9 4-point forward ADST helper. Bit-exact mirror of
/// <see cref="Vp9ForwardAdst4"/> for in-kernel use.
/// </summary>
public static class Vp9ForwardAdst4Gpu
{
    private const int Sinpi1_9 = 5283;
    private const int Sinpi2_9 = 9929;
    private const int Sinpi3_9 = 13377;
    private const int Sinpi4_9 = 15212;
    private const int DctConstBits = 14;
    private const int DctConstRounding = 1 << (DctConstBits - 1);

    /// <summary>
    /// Apply the 4-point forward ADST to one 4-element 1D row/column.
    /// </summary>
    public static void Forward4(
        ArrayView<int> input, long inBase,
        ArrayView<int> output, long outBase)
    {
        int x0 = input[inBase + 0];
        int x1 = input[inBase + 1];
        int x2 = input[inBase + 2];
        int x3 = input[inBase + 3];

        if ((x0 | x1 | x2 | x3) == 0)
        {
            output[outBase + 0] = 0;
            output[outBase + 1] = 0;
            output[outBase + 2] = 0;
            output[outBase + 3] = 0;
            return;
        }

        long s0 = (long)Sinpi1_9 * x0;
        long s1 = (long)Sinpi4_9 * x0;
        long s2 = (long)Sinpi2_9 * x1;
        long s3 = (long)Sinpi1_9 * x1;
        long s4 = (long)Sinpi3_9 * x2;
        long s5 = (long)Sinpi4_9 * x3;
        long s6 = (long)Sinpi2_9 * x3;
        long s7 = (long)x0 + x1 - x3;

        long y0 = s0 + s2 + s5;
        long y1 = (long)Sinpi3_9 * s7;
        long y2 = s1 - s3 + s6;
        long y3 = s4;

        long t0 = y0 + y3;
        long t1 = y1;
        long t2 = y2 - y3;
        long t3 = y2 - y0 + y3;

        output[outBase + 0] = RoundShift(t0);
        output[outBase + 1] = RoundShift(t1);
        output[outBase + 2] = RoundShift(t2);
        output[outBase + 3] = RoundShift(t3);
    }

    /// <summary>libvpx round_shift: (input + 1 &lt;&lt; 13) &gt;&gt; 14.</summary>
    private static int RoundShift(long input) =>
        (int)((input + DctConstRounding) >> DctConstBits);
}
