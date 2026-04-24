// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 inverse DCT 4x4 - bit-exact CPU reference port of
// libvpx vp9_idct4x4_16_add.
//
// Spec reference: VP9 Bitstream Specification sec 8.7.1.2 "Inverse DCT".
// libvpx reference: vp9/common/vp9_idct.c (vp9_idct4x4_16_add /
// vp9_idct4_c) - https://github.com/webmproject/libvpx
//
// Arithmetic notes
//   - Coefficients are int16. Row + column butterflies run in int32 with
//     rounded right-shifts to keep intermediate magnitude bounded.
//   - The Q14 cosine constants (cospi_{8,16,24}_64) are the EXACT integer
//     values from the spec. Do not replace with floats - the normative
//     bitstream requires bit-for-bit match and any fp rounding would
//     desynchronise the decoder against the reference tables.
//   - dct_const_round_shift(x) = (x + (1 << 13)) >> 14    - post-butterfly
//   - ROUND_POWER_OF_TWO(x, 4) = (x + 8) >> 4             - final output
//
// This slice intentionally ships the REFERENCE ONLY. A cross-backend
// ILGPU kernel lands in slice 117 and is validated bit-exactly against
// this reference across CPU / CUDA / OpenCL / WebGPU / WebGL / Wasm.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// CPU reference for VP9 inverse DCT 4x4 (spec §8.7.1.2). Bit-exact
/// against libvpx. Used to validate the ILGPU kernel that follows.
/// </summary>
public static class Vp9Idct4x4Reference
{
    // Q14 fixed-point cosine constants per VP9 spec §8.7.1.2.
    private const int CosPi16_64 = 11585; // cos(16/64 * pi) * 2^14
    private const int CosPi8_64 = 15137;  // cos(8/64 * pi)  * 2^14
    private const int CosPi24_64 = 6270;  // cos(24/64 * pi) * 2^14

    /// <summary>
    /// Apply <paramref name="input"/> (16 coefficients, row-major 4x4) as a
    /// residual to <paramref name="dest"/> (4x4 block of 8-bit pixels with
    /// <paramref name="stride"/> bytes per row). Matches libvpx
    /// <c>vp9_idct4x4_16_add</c> bit-exactly.
    /// </summary>
    /// <param name="input">Length 16 coefficient buffer in row-major order.</param>
    /// <param name="dest">
    /// Destination pixel block. The function OVERWRITES the 4x4 pixels at
    /// offset 0..3 on rows 0..3 (each row <paramref name="stride"/> bytes
    /// apart). Input dest[y*stride + x] is treated as the predictor; output
    /// is clip(predictor + residual) in the same location.
    /// </param>
    /// <param name="stride">Bytes per row of <paramref name="dest"/>.</param>
    public static void Idct4x4_16_Add(
        ReadOnlySpan<short> input, Span<byte> dest, int stride)
    {
        if (input.Length < 16)
            throw new ArgumentException("input must have >= 16 coefficients", nameof(input));
        if (stride < 4)
            throw new ArgumentException("stride must be >= 4", nameof(stride));
        if (dest.Length < 3 * stride + 4)
            throw new ArgumentException("dest too small for 4 rows at the given stride", nameof(dest));

        // Intermediate buffer: 4x4 int16 holding the row-transformed values
        // to feed into the column transform.
        Span<short> tmp = stackalloc short[16];

        // Row pass - each row of 4 input coefficients -> 4 int16s.
        for (int row = 0; row < 4; row++)
        {
            Idct4_1d(
                input.Slice(row * 4, 4),
                tmp.Slice(row * 4, 4));
        }

        // Column pass + final round + residual-add + pixel clip.
        Span<short> colIn = stackalloc short[4];
        Span<short> colOut = stackalloc short[4];
        for (int col = 0; col < 4; col++)
        {
            for (int j = 0; j < 4; j++) colIn[j] = tmp[j * 4 + col];
            Idct4_1d(colIn, colOut);
            for (int j = 0; j < 4; j++)
            {
                // ROUND_POWER_OF_TWO(x, 4) = (x + 8) >> 4
                int residual = (colOut[j] + 8) >> 4;
                int predictor = dest[j * stride + col];
                int sum = predictor + residual;
                // clip_pixel(x) = x < 0 ? 0 : x > 255 ? 255 : x
                if (sum < 0) sum = 0;
                else if (sum > 255) sum = 255;
                dest[j * stride + col] = (byte)sum;
            }
        }
    }

    /// <summary>
    /// One-dimensional 4-point iDCT butterfly, per libvpx <c>idct4_c</c>.
    /// <paramref name="input"/> and <paramref name="output"/> are 4-length
    /// int16 slices. Distinct buffers are allowed (and expected).
    /// </summary>
    private static void Idct4_1d(ReadOnlySpan<short> input, Span<short> output)
    {
        // Butterfly stage 1 (int32 intermediates to avoid overflow).
        int t1 = (input[0] + input[2]) * CosPi16_64;
        int t2 = (input[0] - input[2]) * CosPi16_64;
        short step0 = DctConstRoundShift(t1);
        short step1 = DctConstRoundShift(t2);
        int t3 = input[1] * CosPi24_64 - input[3] * CosPi8_64;
        int t4 = input[1] * CosPi8_64 + input[3] * CosPi24_64;
        short step2 = DctConstRoundShift(t3);
        short step3 = DctConstRoundShift(t4);

        // Butterfly stage 2 (simple add/sub - stays in int16 range for
        // legitimate VP9 coefficients; clamping mirrors libvpx which
        // produces int16 outputs that may overflow briefly on pathological
        // inputs but matches the normative bitstream).
        output[0] = (short)(step0 + step3);
        output[1] = (short)(step1 + step2);
        output[2] = (short)(step1 - step2);
        output[3] = (short)(step0 - step3);
    }

    /// <summary>
    /// VP9 normative rounding: <c>(x + (1 &lt;&lt; 13)) &gt;&gt; 14</c> with
    /// the result narrowed to int16. Must round-to-nearest with ties away
    /// from zero? No - ties go up (plain add-half-and-shift).
    /// </summary>
    private static short DctConstRoundShift(int value)
    {
        // Add rounding bias then arithmetic shift. Result fits in int16 for
        // any legitimate VP9 coefficient magnitude.
        int rounded = (value + (1 << 13)) >> 14;
        return (short)rounded;
    }
}
