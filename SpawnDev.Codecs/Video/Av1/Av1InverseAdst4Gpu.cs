// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 4-point inverse Asymmetric DST (1D), GPU-callable form for
// in-kernel reuse. Bit-exact mirror of Av1InverseAdst4.Transform
// (libaom av1/common/av1_inv_txfm1d.c av1_iadst4 port).
//
// Pairs with Av1ForwardAdst4 (CPU-only) and the existing
// Av1InverseAdst8Gpu / Av1InverseAdst16Gpu (just shipped this session) -
// completes the AV1 inverse ADST family at 4-point.
//
// Uses 64-bit intermediates because per-stage magnitudes can exceed
// 32-bit range before the final round_shift. Inlines the 5-entry
// sinpi[i] = sin(pi*i/9) * 2^cosBit table for i in 0..4 across all
// 4 valid cosBit settings.

using ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// GPU-callable AV1 4-point inverse ADST helper. Bit-exact mirror of
/// <see cref="Av1InverseAdst4"/> for in-kernel use.
/// </summary>
public static class Av1InverseAdst4Gpu
{
    /// <summary>Default cosine-precision bits (libaom).</summary>
    public const int DefaultCosBit = Av1InverseAdst4.DefaultCosBit;

    /// <summary>
    /// Apply the 4-point inverse ADST to one 4-element 1D row/column.
    /// </summary>
    public static void Inverse4(
        ArrayView<int> input, long inBase,
        ArrayView<int> output, long outBase,
        int cosBit)
    {
        ResolveSinpi(cosBit, out int sin1, out int sin2, out int sin3, out int sin4);

        long x0 = input[inBase + 0];
        long x1 = input[inBase + 1];
        long x2 = input[inBase + 2];
        long x3 = input[inBase + 3];

        if ((x0 | x1 | x2 | x3) == 0)
        {
            output[outBase + 0] = 0;
            output[outBase + 1] = 0;
            output[outBase + 2] = 0;
            output[outBase + 3] = 0;
            return;
        }

        // Stage 1.
        long s0 = (long)sin1 * x0;
        long s1 = (long)sin2 * x0;
        long s2 = (long)sin3 * x1;
        long s3 = (long)sin4 * x2;
        long s4 = (long)sin1 * x2;
        long s5 = (long)sin2 * x3;
        long s6 = (long)sin4 * x3;

        // Stage 2.
        long s7 = (x0 - x2) + x3;

        // Stage 3.
        s0 = s0 + s3;
        s1 = s1 - s4;
        long sNew3 = s2;
        long sNew2 = (long)sin3 * s7;

        // Stage 4.
        s0 = s0 + s5;
        s1 = s1 - s6;

        // Stage 5.
        long y0 = s0 + sNew3;
        long y1 = s1 + sNew3;
        long y2 = sNew2;
        long y3 = s0 + s1;

        // Stage 6.
        y3 = y3 - sNew3;

        output[outBase + 0] = RoundShift(y0, cosBit);
        output[outBase + 1] = RoundShift(y1, cosBit);
        output[outBase + 2] = RoundShift(y2, cosBit);
        output[outBase + 3] = RoundShift(y3, cosBit);
    }

    /// <summary>libaom <c>round_shift</c>: arithmetic round-half-up by bit.</summary>
    private static int RoundShift(long value, int bit)
    {
        return (int)((value + (1L << (bit - 1))) >> bit);
    }

    /// <summary>Resolves sinpi[1..4] = sin(pi*i/9) * 2^cosBit per libaom sinpi_arr_data.</summary>
    private static void ResolveSinpi(int cosBit,
        out int sin1, out int sin2, out int sin3, out int sin4)
    {
        if (cosBit == 13)
        {
            sin1 = 2642; sin2 = 4964; sin3 = 6688; sin4 = 7606;
        }
        else if (cosBit == 12)
        {
            sin1 = 1321; sin2 = 2482; sin3 = 3344; sin4 = 3803;
        }
        else if (cosBit == 11)
        {
            sin1 = 660; sin2 = 1241; sin3 = 1672; sin4 = 1901;
        }
        else
        {
            sin1 = 330; sin2 = 621; sin3 = 836; sin4 = 951;
        }
    }
}
