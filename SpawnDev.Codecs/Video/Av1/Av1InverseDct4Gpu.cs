// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 4-point inverse DCT (1D), GPU-callable form for in-kernel reuse.
// Bit-exact mirror of Av1InverseDct4.Transform (libaom
// av1/common/av1_inv_txfm1d.c av1_idct4 port).
//
// Pairs with the existing Av1ForwardDct8Gpu / Av1InverseDct8Gpu /
// Av1InverseDct16Gpu - completes the AV1 inverse DCT family for the
// smallest 4-point block size.
//
// 3 stages with cospi-driven half_btf rotations + final butterfly.

using ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// GPU-callable AV1 4-point inverse DCT helper. Bit-exact mirror of
/// <see cref="Av1InverseDct4"/> for in-kernel use.
/// </summary>
public static class Av1InverseDct4Gpu
{
    /// <summary>Default cosine-precision bits (libaom).</summary>
    public const int DefaultCosBit = Av1InverseDct4.DefaultCosBit;

    /// <summary>
    /// Apply the 4-point inverse DCT to one 4-element 1D row/column.
    /// </summary>
    public static void Inverse4(
        ArrayView<int> input, long inBase,
        ArrayView<int> output, long outBase,
        int cosBit)
    {
        ResolveCospi(cosBit, out int c16, out int c32, out int c48);

        // Stage 1: re-permute input.
        int b0 = input[inBase + 0];
        int b1 = input[inBase + 2];
        int b2 = input[inBase + 1];
        int b3 = input[inBase + 3];

        // Stage 2: cospi butterflies.
        int s0 = HalfBtf(c32, b0, c32, b1, cosBit);
        int s1 = HalfBtf(c32, b0, -c32, b1, cosBit);
        int s2 = HalfBtf(c48, b2, -c16, b3, cosBit);
        int s3 = HalfBtf(c16, b2, c48, b3, cosBit);

        // Stage 3: outer butterfly.
        output[outBase + 0] = s0 + s3;
        output[outBase + 1] = s1 + s2;
        output[outBase + 2] = s1 - s2;
        output[outBase + 3] = s0 - s3;
    }

    /// <summary>libaom <c>half_btf</c>: kernel-safe variant.</summary>
    private static int HalfBtf(int w0, int in0, int w1, int in1, int bit)
    {
        long result = (long)w0 * in0 + (long)w1 * in1;
        result += 1L << (bit - 1);
        return (int)(result >> bit);
    }

    /// <summary>Resolves the 3 cospi entries idct4 needs.</summary>
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
