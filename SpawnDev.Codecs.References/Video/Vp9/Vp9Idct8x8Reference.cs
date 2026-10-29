// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 inverse DCT 8x8 - bit-exact CPU reference port of
// libvpx vp9_idct8x8_64_add.
//
// Spec reference: VP9 Bitstream Specification sec 8.7.1.3 "Inverse 8x8 DCT"
// libvpx reference: vp9/common/vp9_idct.c (vp9_idct8x8_64_add_c /
// vp9_idct8_c) - https://github.com/webmproject/libvpx
//
// Structure
//   - 4-stage butterfly (4x4 was 2-stage). Additional Q14 cosine
//     constants cospi_{4,12,20,28}_64 for the 8-point transform.
//   - Even-half of the butterfly reuses the 4-point iDCT pattern;
//     odd-half is new, with two more rounded multiplies in stage 3.
//   - Final round is (x + 16) >> 5 (vs (x + 8) >> 4 for 4x4). The
//     8-point DCT has a different scale factor.
//
// The implementation uses `int` throughout for butterfly state - matches
// libvpx's WRAPLOW()/check_range() in 8-bit-depth mode where intermediate
// values are stored as int16 but intermediates are promoted to int32 for
// the multiply + add chain.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// CPU reference for VP9 inverse DCT 8x8 (spec §8.7.1.3). Bit-exact
/// against libvpx. Used to validate the ILGPU kernel that follows.
/// </summary>
public static class Vp9Idct8x8Reference
{
    // Q14 fixed-point cosine constants. The first three are shared with
    // the 4x4 transform; the last four are new.
    private const int CosPi16_64 = 11585;
    private const int CosPi8_64 = 15137;
    private const int CosPi24_64 = 6270;
    private const int CosPi4_64 = 16069;
    private const int CosPi12_64 = 13623;
    private const int CosPi20_64 = 9102;
    private const int CosPi28_64 = 3196;

    /// <summary>
    /// Apply <paramref name="input"/> (64 coefficients, row-major 8x8) as a
    /// residual to <paramref name="dest"/> (8x8 block of 8-bit pixels with
    /// <paramref name="stride"/> bytes per row). Matches libvpx
    /// <c>vp9_idct8x8_64_add</c> bit-exactly.
    /// </summary>
    public static void Idct8x8_64_Add(
        ReadOnlySpan<short> input, Span<byte> dest, int stride)
    {
        if (input.Length < 64)
            throw new ArgumentException("input must have >= 64 coefficients", nameof(input));
        if (stride < 8)
            throw new ArgumentException("stride must be >= 8", nameof(stride));
        if (dest.Length < 7 * stride + 8)
            throw new ArgumentException("dest too small for 8 rows at the given stride", nameof(dest));

        // Row-pass intermediates: 8x8 int16.
        Span<short> tmp = stackalloc short[64];

        // Row pass: 8 input coefficients -> 8 int16 intermediates per row.
        for (int row = 0; row < 8; row++)
        {
            Idct8_1d(
                input.Slice(row * 8, 8),
                tmp.Slice(row * 8, 8));
        }

        // Column pass + final round + residual-add + pixel clip.
        Span<short> colIn = stackalloc short[8];
        Span<short> colOut = stackalloc short[8];
        for (int col = 0; col < 8; col++)
        {
            for (int j = 0; j < 8; j++) colIn[j] = tmp[j * 8 + col];
            Idct8_1d(colIn, colOut);
            for (int j = 0; j < 8; j++)
            {
                // ROUND_POWER_OF_TWO(x, 5) = (x + 16) >> 5 - note the >>5
                // here vs >>4 in the 4x4 case. The 8-point DCT scales
                // differently.
                int residual = (colOut[j] + 16) >> 5;
                int predictor = dest[j * stride + col];
                int sum = predictor + residual;
                if (sum < 0) sum = 0;
                else if (sum > 255) sum = 255;
                dest[j * stride + col] = (byte)sum;
            }
        }
    }

    /// <summary>
    /// One-dimensional 8-point iDCT butterfly per libvpx <c>vp9_idct8_c</c>.
    /// Four stages; even half reuses the 4-point pattern, odd half is
    /// additional. Internal so the iHT 8x8 tx_type dispatcher can share it.
    /// </summary>
    internal static void Idct8_1d(ReadOnlySpan<short> input, Span<short> output)
    {
        // Intermediate arrays mirror libvpx's step1[0..7] / step2[0..7].
        // Using individual locals keeps register-allocation friendly.
        short s1_0 = input[0];
        short s1_1 = input[2];
        short s1_2 = input[4];
        short s1_3 = input[6];

        // Stage 1 - odd half
        int t_a = input[1] * CosPi28_64 - input[7] * CosPi4_64;
        int t_b = input[1] * CosPi4_64 + input[7] * CosPi28_64;
        short s1_4 = RoundShift14(t_a);
        short s1_7 = RoundShift14(t_b);
        int t_c = input[5] * CosPi12_64 - input[3] * CosPi20_64;
        int t_d = input[5] * CosPi20_64 + input[3] * CosPi12_64;
        short s1_5 = RoundShift14(t_c);
        short s1_6 = RoundShift14(t_d);

        // Stage 2 - even half
        // Reorder: step1 indices reorganised into the 4-pt iDCT pattern.
        // libvpx: step1 slots [0..3] hold DC/AC evens, [4..7] hold odd terms.
        int t_e = (s1_0 + s1_2) * CosPi16_64;
        int t_f = (s1_0 - s1_2) * CosPi16_64;
        short s2_0 = RoundShift14(t_e);
        short s2_1 = RoundShift14(t_f);
        int t_g = s1_1 * CosPi24_64 - s1_3 * CosPi8_64;
        int t_h = s1_1 * CosPi8_64 + s1_3 * CosPi24_64;
        short s2_2 = RoundShift14(t_g);
        short s2_3 = RoundShift14(t_h);
        short s2_4 = (short)(s1_4 + s1_5);
        short s2_5 = (short)(s1_4 - s1_5);
        short s2_6 = (short)(-s1_6 + s1_7);
        short s2_7 = (short)(s1_6 + s1_7);

        // Stage 3
        short e1_0 = (short)(s2_0 + s2_3);
        short e1_1 = (short)(s2_1 + s2_2);
        short e1_2 = (short)(s2_1 - s2_2);
        short e1_3 = (short)(s2_0 - s2_3);
        short e1_4 = s2_4;
        int t_i = (s2_6 - s2_5) * CosPi16_64;
        int t_j = (s2_5 + s2_6) * CosPi16_64;
        short e1_5 = RoundShift14(t_i);
        short e1_6 = RoundShift14(t_j);
        short e1_7 = s2_7;

        // Stage 4 - final output butterfly
        output[0] = (short)(e1_0 + e1_7);
        output[1] = (short)(e1_1 + e1_6);
        output[2] = (short)(e1_2 + e1_5);
        output[3] = (short)(e1_3 + e1_4);
        output[4] = (short)(e1_3 - e1_4);
        output[5] = (short)(e1_2 - e1_5);
        output[6] = (short)(e1_1 - e1_6);
        output[7] = (short)(e1_0 - e1_7);
    }

    /// <summary>VP9 normative Q14 rounding: <c>(x + (1 &lt;&lt; 13)) &gt;&gt; 14</c>.</summary>
    private static short RoundShift14(int value) => (short)((value + (1 << 13)) >> 14);
}
