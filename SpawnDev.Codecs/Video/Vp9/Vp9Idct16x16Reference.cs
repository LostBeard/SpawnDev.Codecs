// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 inverse DCT 16x16 - bit-exact CPU reference port of
// libvpx idct16_c + vp9_idct16x16_256_add_c.
//
// Spec reference: VP9 Bitstream Specification sec 8.7.1.4 "Inverse 16x16 DCT"
// libvpx reference: vpx_dsp/inv_txfm.c (idct16_c) +
// vp9/common/vp9_idct.c (vp9_idct16x16_256_add_c).
//
// Structure
//   - 7-stage butterfly (8x8 was 4 stages). The 16-point transform is
//     built as two 8-point transforms on re-ordered input positions
//     combined through the usual DCT merging pattern (Chen-Wang style).
//   - Additional Q14 cosine constants cospi_{2,6,10,14,18,22,26,30}_64
//     on top of the {4,8,12,16,20,24,28}_64 set the 8x8 kernel uses.
//   - Final round is (x + 32) >> 6 (vs >>5 for 8x8, >>4 for 4x4).
//   - Input reads use a fixed bit-reversal-style reordering on stage 1
//     (input[0], input[8], input[4], input[12], input[2], input[10],
//     input[6], input[14], input[1], input[9], input[5], input[13],
//     input[3], input[11], input[7], input[15]). Mirrored from libvpx.
//
// Short-circuit: the reference exists to validate correctness, not
// speed, so we don't skip the transform when the input is all zero -
// any caller that cares about that fast path can check ahead of time.
// The deterministic-result property matters most here.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// CPU reference for VP9 inverse DCT 16x16 (spec sec 8.7.1.4). Bit-exact
/// against libvpx idct16_c / vp9_idct16x16_256_add_c.
/// </summary>
public static class Vp9Idct16x16Reference
{
    // Q14 fixed-point cosine constants. First 7 are shared with 4x4/8x8;
    // the remaining 8 are new to the 16-point transform.
    private const int CosPi16_64 = 11585;
    private const int CosPi8_64 = 15137;
    private const int CosPi24_64 = 6270;
    private const int CosPi4_64 = 16069;
    private const int CosPi12_64 = 13623;
    private const int CosPi20_64 = 9102;
    private const int CosPi28_64 = 3196;
    private const int CosPi2_64 = 16305;
    private const int CosPi6_64 = 15679;
    private const int CosPi10_64 = 14449;
    private const int CosPi14_64 = 12665;
    private const int CosPi18_64 = 10394;
    private const int CosPi22_64 = 7723;
    private const int CosPi26_64 = 4756;
    private const int CosPi30_64 = 1606;

    /// <summary>
    /// Apply <paramref name="input"/> (256 coefficients, row-major 16x16)
    /// as a residual to <paramref name="dest"/> (16x16 block of 8-bit
    /// pixels with <paramref name="stride"/> bytes per row). Matches
    /// libvpx <c>vp9_idct16x16_256_add</c> bit-exactly.
    /// </summary>
    public static void Idct16x16_256_Add(
        ReadOnlySpan<short> input, Span<byte> dest, int stride)
    {
        if (input.Length < 256)
            throw new ArgumentException("input must have >= 256 coefficients", nameof(input));
        if (stride < 16)
            throw new ArgumentException("stride must be >= 16", nameof(stride));
        if (dest.Length < 15 * stride + 16)
            throw new ArgumentException("dest too small for 16 rows at the given stride", nameof(dest));

        Span<short> tmp = stackalloc short[256];

        for (int row = 0; row < 16; row++)
        {
            Idct16_1d(
                input.Slice(row * 16, 16),
                tmp.Slice(row * 16, 16));
        }

        Span<short> colIn = stackalloc short[16];
        Span<short> colOut = stackalloc short[16];
        for (int col = 0; col < 16; col++)
        {
            for (int j = 0; j < 16; j++) colIn[j] = tmp[j * 16 + col];
            Idct16_1d(colIn, colOut);
            for (int j = 0; j < 16; j++)
            {
                // ROUND_POWER_OF_TWO(x, 6) = (x + 32) >> 6
                int residual = (colOut[j] + 32) >> 6;
                int predictor = dest[j * stride + col];
                int sum = predictor + residual;
                if (sum < 0) sum = 0;
                else if (sum > 255) sum = 255;
                dest[j * stride + col] = (byte)sum;
            }
        }
    }

    /// <summary>
    /// One-dimensional 16-point iDCT butterfly, bit-exact against
    /// libvpx <c>idct16_c</c>. 7 stages, int32 intermediates for the
    /// cospi multiplies, int16 narrowing between stages (WRAPLOW).
    /// </summary>
    private static void Idct16_1d(ReadOnlySpan<short> input, Span<short> output)
    {
        Span<short> step1 = stackalloc short[16];
        Span<short> step2 = stackalloc short[16];

        // Stage 1: bit-reversal-style input reordering (even-indexed
        // inputs to the even slots, odd-indexed inputs to the odd slots
        // of step1, with an internal shuffle inside each half).
        step1[0] = input[0];
        step1[1] = input[8];
        step1[2] = input[4];
        step1[3] = input[12];
        step1[4] = input[2];
        step1[5] = input[10];
        step1[6] = input[6];
        step1[7] = input[14];
        step1[8] = input[1];
        step1[9] = input[9];
        step1[10] = input[5];
        step1[11] = input[13];
        step1[12] = input[3];
        step1[13] = input[11];
        step1[14] = input[7];
        step1[15] = input[15];

        // Stage 2: pass through 0..7, four rotation butterflies on 8..15.
        for (int i = 0; i < 8; i++) step2[i] = step1[i];
        {
            int t1 = step1[8] * CosPi30_64 - step1[15] * CosPi2_64;
            int t2 = step1[8] * CosPi2_64 + step1[15] * CosPi30_64;
            step2[8] = Rs14(t1);
            step2[15] = Rs14(t2);
        }
        {
            int t1 = step1[9] * CosPi14_64 - step1[14] * CosPi18_64;
            int t2 = step1[9] * CosPi18_64 + step1[14] * CosPi14_64;
            step2[9] = Rs14(t1);
            step2[14] = Rs14(t2);
        }
        {
            int t1 = step1[10] * CosPi22_64 - step1[13] * CosPi10_64;
            int t2 = step1[10] * CosPi10_64 + step1[13] * CosPi22_64;
            step2[10] = Rs14(t1);
            step2[13] = Rs14(t2);
        }
        {
            int t1 = step1[11] * CosPi6_64 - step1[12] * CosPi26_64;
            int t2 = step1[11] * CosPi26_64 + step1[12] * CosPi6_64;
            step2[11] = Rs14(t1);
            step2[12] = Rs14(t2);
        }

        // Stage 3: 0..3 passthrough, two rotations on 4..7, add/sub on 8..15.
        for (int i = 0; i < 4; i++) step1[i] = step2[i];
        {
            int t1 = step2[4] * CosPi28_64 - step2[7] * CosPi4_64;
            int t2 = step2[4] * CosPi4_64 + step2[7] * CosPi28_64;
            step1[4] = Rs14(t1);
            step1[7] = Rs14(t2);
        }
        {
            int t1 = step2[5] * CosPi12_64 - step2[6] * CosPi20_64;
            int t2 = step2[5] * CosPi20_64 + step2[6] * CosPi12_64;
            step1[5] = Rs14(t1);
            step1[6] = Rs14(t2);
        }
        step1[8] = (short)(step2[8] + step2[9]);
        step1[9] = (short)(step2[8] - step2[9]);
        step1[10] = (short)(-step2[10] + step2[11]);
        step1[11] = (short)(step2[10] + step2[11]);
        step1[12] = (short)(step2[12] + step2[13]);
        step1[13] = (short)(step2[12] - step2[13]);
        step1[14] = (short)(-step2[14] + step2[15]);
        step1[15] = (short)(step2[14] + step2[15]);

        // Stage 4: DC butterfly on 0-1, rotation on 2-3, add/sub on 4-7,
        // passthrough + rotations on 8..15.
        {
            int t1 = (step1[0] + step1[1]) * CosPi16_64;
            int t2 = (step1[0] - step1[1]) * CosPi16_64;
            step2[0] = Rs14(t1);
            step2[1] = Rs14(t2);
        }
        {
            int t1 = step1[2] * CosPi24_64 - step1[3] * CosPi8_64;
            int t2 = step1[2] * CosPi8_64 + step1[3] * CosPi24_64;
            step2[2] = Rs14(t1);
            step2[3] = Rs14(t2);
        }
        step2[4] = (short)(step1[4] + step1[5]);
        step2[5] = (short)(step1[4] - step1[5]);
        step2[6] = (short)(-step1[6] + step1[7]);
        step2[7] = (short)(step1[6] + step1[7]);

        step2[8] = step1[8];
        step2[15] = step1[15];
        {
            int t1 = -step1[9] * CosPi8_64 + step1[14] * CosPi24_64;
            int t2 = step1[9] * CosPi24_64 + step1[14] * CosPi8_64;
            step2[9] = Rs14(t1);
            step2[14] = Rs14(t2);
        }
        {
            int t1 = -step1[10] * CosPi24_64 - step1[13] * CosPi8_64;
            int t2 = -step1[10] * CosPi8_64 + step1[13] * CosPi24_64;
            step2[10] = Rs14(t1);
            step2[13] = Rs14(t2);
        }
        step2[11] = step1[11];
        step2[12] = step1[12];

        // Stage 5.
        step1[0] = (short)(step2[0] + step2[3]);
        step1[1] = (short)(step2[1] + step2[2]);
        step1[2] = (short)(step2[1] - step2[2]);
        step1[3] = (short)(step2[0] - step2[3]);
        step1[4] = step2[4];
        {
            int t1 = (step2[6] - step2[5]) * CosPi16_64;
            int t2 = (step2[5] + step2[6]) * CosPi16_64;
            step1[5] = Rs14(t1);
            step1[6] = Rs14(t2);
        }
        step1[7] = step2[7];

        step1[8] = (short)(step2[8] + step2[11]);
        step1[9] = (short)(step2[9] + step2[10]);
        step1[10] = (short)(step2[9] - step2[10]);
        step1[11] = (short)(step2[8] - step2[11]);
        step1[12] = (short)(-step2[12] + step2[15]);
        step1[13] = (short)(-step2[13] + step2[14]);
        step1[14] = (short)(step2[13] + step2[14]);
        step1[15] = (short)(step2[12] + step2[15]);

        // Stage 6.
        step2[0] = (short)(step1[0] + step1[7]);
        step2[1] = (short)(step1[1] + step1[6]);
        step2[2] = (short)(step1[2] + step1[5]);
        step2[3] = (short)(step1[3] + step1[4]);
        step2[4] = (short)(step1[3] - step1[4]);
        step2[5] = (short)(step1[2] - step1[5]);
        step2[6] = (short)(step1[1] - step1[6]);
        step2[7] = (short)(step1[0] - step1[7]);
        step2[8] = step1[8];
        step2[9] = step1[9];
        {
            int t1 = (-step1[10] + step1[13]) * CosPi16_64;
            int t2 = (step1[10] + step1[13]) * CosPi16_64;
            step2[10] = Rs14(t1);
            step2[13] = Rs14(t2);
        }
        {
            int t1 = (-step1[11] + step1[12]) * CosPi16_64;
            int t2 = (step1[11] + step1[12]) * CosPi16_64;
            step2[11] = Rs14(t1);
            step2[12] = Rs14(t2);
        }
        step2[14] = step1[14];
        step2[15] = step1[15];

        // Stage 7: final combining butterfly.
        output[0] = (short)(step2[0] + step2[15]);
        output[1] = (short)(step2[1] + step2[14]);
        output[2] = (short)(step2[2] + step2[13]);
        output[3] = (short)(step2[3] + step2[12]);
        output[4] = (short)(step2[4] + step2[11]);
        output[5] = (short)(step2[5] + step2[10]);
        output[6] = (short)(step2[6] + step2[9]);
        output[7] = (short)(step2[7] + step2[8]);
        output[8] = (short)(step2[7] - step2[8]);
        output[9] = (short)(step2[6] - step2[9]);
        output[10] = (short)(step2[5] - step2[10]);
        output[11] = (short)(step2[4] - step2[11]);
        output[12] = (short)(step2[3] - step2[12]);
        output[13] = (short)(step2[2] - step2[13]);
        output[14] = (short)(step2[1] - step2[14]);
        output[15] = (short)(step2[0] - step2[15]);
    }

    /// <summary>VP9 normative Q14 rounding.</summary>
    private static short Rs14(int value) => (short)((value + (1 << 13)) >> 14);
}
