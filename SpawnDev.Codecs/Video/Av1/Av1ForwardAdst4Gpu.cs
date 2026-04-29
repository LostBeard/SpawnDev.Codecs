// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 4-point forward Asymmetric DST (1D), GPU-callable form for
// in-kernel reuse. Bit-exact mirror of Av1ForwardAdst4.Transform
// (libaom av1/encoder/av1_fwd_txfm1d.c av1_fadst4 port).
//
// Pairs with Av1InverseAdst4Gpu (just shipped) - now AV1 4-point ADST
// has GPU primitives in both directions. Combined with the existing
// 8-point and 16-point forward ADST GPU helpers this completes the
// AV1 forward ADST family at all 3 small-block sizes.
//
// Uses 64-bit intermediates because per-stage magnitudes can exceed
// 32-bit range before the final round_shift. Inlines the 5-entry
// sinpi[i] = sin(pi*i/9) * 2^cosBit table for i in 0..4 across all
// 4 valid cosBit settings.

using ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// GPU-callable AV1 4-point forward ADST helper. Bit-exact mirror of
/// <see cref="Av1ForwardAdst4"/> for in-kernel use.
/// </summary>
public static class Av1ForwardAdst4Gpu
{
    /// <summary>Default cosine-precision bits (libaom).</summary>
    public const int DefaultCosBit = Av1ForwardAdst4.DefaultCosBit;

    /// <summary>
    /// Apply the 4-point forward ADST to one 4-element 1D row/column.
    /// </summary>
    public static void Forward4(
        ArrayView<int> input, long inBase,
        ArrayView<int> output, long outBase,
        int cosBit)
    {
        ResolveSinpi(cosBit, out int sin1, out int sin2, out int sin3, out int sin4);

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

        // Stage 1.
        long s0 = (long)sin1 * x0;
        long s1 = (long)sin4 * x0;
        long s2 = (long)sin2 * x1;
        long s3 = (long)sin1 * x1;
        long s4 = (long)sin3 * x2;
        long s5 = (long)sin4 * x3;
        long s6 = (long)sin2 * x3;
        long s7 = (long)x0 + x1;

        // Stage 2.
        s7 = s7 - x3;

        // Stage 3.
        long y0 = s0 + s2;
        long y1 = (long)sin3 * s7;
        long y2 = s1 - s3;
        long y3 = s4;

        // Stage 4.
        y0 += s5;
        y2 += s6;

        // Stage 5.
        long t0 = y0 + y3;
        long t1 = y1;
        long t2 = y2 - y3;
        long t3 = y2 - y0;

        // Stage 6.
        t3 += y3;

        // Final round_shift by cosBit.
        output[outBase + 0] = RoundShift(t0, cosBit);
        output[outBase + 1] = RoundShift(t1, cosBit);
        output[outBase + 2] = RoundShift(t2, cosBit);
        output[outBase + 3] = RoundShift(t3, cosBit);
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
