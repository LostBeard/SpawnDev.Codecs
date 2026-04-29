// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 4-point forward DCT (1D), GPU-callable form for in-kernel reuse.
// Bit-exact mirror of Av1ForwardDct4.Transform (libaom
// av1/encoder/av1_fwd_txfm1d.c av1_fdct4 port).
//
// Pairs with Av1InverseDct4Gpu (just shipped) - now AV1 4-point DCT
// has GPU primitives in both directions. Combined with the existing
// 8-point and 16-point forward DCT GPU helpers this completes the AV1
// forward DCT family at all 3 small-block sizes.
//
// 3 stages: butterfly -> cospi rotations -> interleave.

using ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// GPU-callable AV1 4-point forward DCT helper. Bit-exact mirror of
/// <see cref="Av1ForwardDct4"/> for in-kernel use.
/// </summary>
public static class Av1ForwardDct4Gpu
{
    /// <summary>Default cosine-precision bits (libaom).</summary>
    public const int DefaultCosBit = Av1ForwardDct4.DefaultCosBit;

    /// <summary>
    /// Apply the 4-point forward DCT to one 4-element 1D row/column.
    /// </summary>
    public static void Forward4(
        ArrayView<int> input, long inBase,
        ArrayView<int> output, long outBase,
        int cosBit)
    {
        ResolveCospi(cosBit, out int c16, out int c32, out int c48);

        // Stage 1: butterfly.
        int s10 = input[inBase + 0] + input[inBase + 3];
        int s11 = input[inBase + 1] + input[inBase + 2];
        int s12 = -input[inBase + 2] + input[inBase + 1];
        int s13 = -input[inBase + 3] + input[inBase + 0];

        // Stage 2: cospi rotations.
        int s20 = HalfBtf(c32, s10, c32, s11, cosBit);
        int s21 = HalfBtf(-c32, s11, c32, s10, cosBit);
        int s22 = HalfBtf(c48, s12, c16, s13, cosBit);
        int s23 = HalfBtf(c48, s13, -c16, s12, cosBit);

        // Stage 3: interleave.
        output[outBase + 0] = s20;
        output[outBase + 1] = s22;
        output[outBase + 2] = s21;
        output[outBase + 3] = s23;
    }

    /// <summary>libaom <c>half_btf</c>: kernel-safe variant.</summary>
    private static int HalfBtf(int w0, int in0, int w1, int in1, int bit)
    {
        long result = (long)w0 * in0 + (long)w1 * in1;
        result += 1L << (bit - 1);
        return (int)(result >> bit);
    }

    /// <summary>Resolves the 3 cospi entries fdct4 needs.</summary>
    private static void ResolveCospi(int cosBit, out int c16, out int c32, out int c48)
    {
        if (cosBit == 13)
        {
            c16 = 7568; c32 = 5793; c48 = 3135;
        }
        else if (cosBit == 12)
        {
            c16 = 3784; c32 = 2896; c48 = 1567;
        }
        else if (cosBit == 11)
        {
            c16 = 1892; c32 = 1448; c48 = 784;
        }
        else
        {
            c16 = 946; c32 = 724; c48 = 392;
        }
    }
}
