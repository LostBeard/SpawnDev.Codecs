// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 8x8 inverse DCT, GPU-callable form for in-kernel reuse.
// Bit-exact mirror of Vp9Idct8x8Reference (the libvpx vp9_idct8x8_64_add
// port).
//
// Vp9Idct8x8Kernel already wraps this math as a standalone batched
// dispatch (one thread per 8x8 block). This helper is the in-kernel
// companion for the v3 sequential decoder/encoder path: the per-frame
// kernel iterates blocks sequentially and adds residual to recon
// inline.
//
// Two-pass shape:
//   Row pass: 8 row-1D iDCTs, intermediate stored as int (the int16
//             narrowing at each butterfly sub-step reproduces libvpx
//             WRAPLOW() semantics; storing as int dodges the WebGPU
//             packed-sub-word path that broke the int16-typed local
//             buffer in the standalone kernel's first version).
//   Column pass: 8 column-1D iDCTs that add `(colOut + 16) >> 5` to
//                the predictor pixel + clip to [0, 255].
//
// Caller supplies a 64-int scratch view for the inter-pass
// intermediate buffer.

using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// GPU-callable VP9 8x8 inverse DCT helper. Bit-exact mirror of
/// <see cref="Vp9Idct8x8Reference"/> for in-kernel use.
/// </summary>
public static class Vp9Idct8x8Gpu
{
    private const int CosPi16_64 = 11585;
    private const int CosPi8_64 = 15137;
    private const int CosPi24_64 = 6270;
    private const int CosPi4_64 = 16069;
    private const int CosPi12_64 = 13623;
    private const int CosPi20_64 = 9102;
    private const int CosPi28_64 = 3196;

    /// <summary>
    /// Inverse-DCT one 8x8 block and add the residual to
    /// <paramref name="dest"/> in place. Reads
    /// <paramref name="coefs"/> starting at <paramref name="coefBase"/>
    /// (64 contiguous shorts, row-major); writes back to
    /// <paramref name="dest"/> starting at <paramref name="destBase"/>
    /// with row stride <paramref name="destStride"/>. Each output
    /// pixel is <c>clip3(0, 255, dest + (residual + 16) &gt;&gt; 5)</c>.
    ///
    /// <paramref name="scratch"/> must hold at least 64 ints for the
    /// inter-pass intermediate buffer.
    /// </summary>
    public static void Idct8x8(
        ArrayView<short> coefs, long coefBase,
        ArrayView<byte> dest, long destBase, int destStride,
        ArrayView<int> scratch)
    {
        // Row pass.
        for (int row = 0; row < 8; row++)
        {
            long rBase = coefBase + row * 8;
            Idct8Row(
                coefs[rBase + 0], coefs[rBase + 1], coefs[rBase + 2], coefs[rBase + 3],
                coefs[rBase + 4], coefs[rBase + 5], coefs[rBase + 6], coefs[rBase + 7],
                out int o0, out int o1, out int o2, out int o3,
                out int o4, out int o5, out int o6, out int o7);
            int rb = row * 8;
            scratch[rb + 0] = o0;
            scratch[rb + 1] = o1;
            scratch[rb + 2] = o2;
            scratch[rb + 3] = o3;
            scratch[rb + 4] = o4;
            scratch[rb + 5] = o5;
            scratch[rb + 6] = o6;
            scratch[rb + 7] = o7;
        }

        // Column pass + residual add + clip.
        for (int col = 0; col < 8; col++)
        {
            Idct8Row(
                (short)scratch[0 * 8 + col], (short)scratch[1 * 8 + col],
                (short)scratch[2 * 8 + col], (short)scratch[3 * 8 + col],
                (short)scratch[4 * 8 + col], (short)scratch[5 * 8 + col],
                (short)scratch[6 * 8 + col], (short)scratch[7 * 8 + col],
                out int co0, out int co1, out int co2, out int co3,
                out int co4, out int co5, out int co6, out int co7);
            ApplyResidualAndClip(dest, destBase + 0L * destStride + col, co0);
            ApplyResidualAndClip(dest, destBase + 1L * destStride + col, co1);
            ApplyResidualAndClip(dest, destBase + 2L * destStride + col, co2);
            ApplyResidualAndClip(dest, destBase + 3L * destStride + col, co3);
            ApplyResidualAndClip(dest, destBase + 4L * destStride + col, co4);
            ApplyResidualAndClip(dest, destBase + 5L * destStride + col, co5);
            ApplyResidualAndClip(dest, destBase + 6L * destStride + col, co6);
            ApplyResidualAndClip(dest, destBase + 7L * destStride + col, co7);
        }
    }

    /// <summary>
    /// 8-point 1D iDCT butterfly. Mirrors libvpx vp9_idct8_1d_c
    /// bit-exactly; int16 narrowing via (short) cast at each
    /// butterfly sub-step reproduces libvpx WRAPLOW() semantics.
    /// </summary>
    private static void Idct8Row(
        short i0, short i1, short i2, short i3, short i4, short i5, short i6, short i7,
        out int o0, out int o1, out int o2, out int o3,
        out int o4, out int o5, out int o6, out int o7)
    {
        short s1_0 = i0;
        short s1_1 = i2;
        short s1_2 = i4;
        short s1_3 = i6;

        int t_a = i1 * CosPi28_64 - i7 * CosPi4_64;
        int t_b = i1 * CosPi4_64 + i7 * CosPi28_64;
        short s1_4 = (short)((t_a + (1 << 13)) >> 14);
        short s1_7 = (short)((t_b + (1 << 13)) >> 14);
        int t_c = i5 * CosPi12_64 - i3 * CosPi20_64;
        int t_d = i5 * CosPi20_64 + i3 * CosPi12_64;
        short s1_5 = (short)((t_c + (1 << 13)) >> 14);
        short s1_6 = (short)((t_d + (1 << 13)) >> 14);

        int t_e = (s1_0 + s1_2) * CosPi16_64;
        int t_f = (s1_0 - s1_2) * CosPi16_64;
        short s2_0 = (short)((t_e + (1 << 13)) >> 14);
        short s2_1 = (short)((t_f + (1 << 13)) >> 14);
        int t_g = s1_1 * CosPi24_64 - s1_3 * CosPi8_64;
        int t_h = s1_1 * CosPi8_64 + s1_3 * CosPi24_64;
        short s2_2 = (short)((t_g + (1 << 13)) >> 14);
        short s2_3 = (short)((t_h + (1 << 13)) >> 14);
        short s2_4 = (short)(s1_4 + s1_5);
        short s2_5 = (short)(s1_4 - s1_5);
        short s2_6 = (short)(-s1_6 + s1_7);
        short s2_7 = (short)(s1_6 + s1_7);

        short e1_0 = (short)(s2_0 + s2_3);
        short e1_1 = (short)(s2_1 + s2_2);
        short e1_2 = (short)(s2_1 - s2_2);
        short e1_3 = (short)(s2_0 - s2_3);
        short e1_4 = s2_4;
        int t_i = (s2_6 - s2_5) * CosPi16_64;
        int t_j = (s2_5 + s2_6) * CosPi16_64;
        short e1_5 = (short)((t_i + (1 << 13)) >> 14);
        short e1_6 = (short)((t_j + (1 << 13)) >> 14);
        short e1_7 = s2_7;

        o0 = (short)(e1_0 + e1_7);
        o1 = (short)(e1_1 + e1_6);
        o2 = (short)(e1_2 + e1_5);
        o3 = (short)(e1_3 + e1_4);
        o4 = (short)(e1_3 - e1_4);
        o5 = (short)(e1_2 - e1_5);
        o6 = (short)(e1_1 - e1_6);
        o7 = (short)(e1_0 - e1_7);
    }

    private static void ApplyResidualAndClip(ArrayView<byte> dest, long offset, int colOut)
    {
        int residual = (colOut + 16) >> 5;
        int sum = dest[offset] + residual;
        if (sum < 0) sum = 0;
        else if (sum > 255) sum = 255;
        dest[offset] = (byte)sum;
    }
}
