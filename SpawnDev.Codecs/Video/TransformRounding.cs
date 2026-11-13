// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Rounded fixed-point products for the transform butterflies (AV1 half_btf, VP9 dct_const_round_shift), computed in
// 32-bit arithmetic with the SAME result as the 64-bit formula for every int input.
//
// Why: the browser backends have no 64-bit integers. A "(long)w0 * in0 + (long)w1 * in1" there is two emulated
// 64-bit multiplies, two emulated adds and an emulated shift per butterfly (about 60 lines of shader each once the
// D3D compiler has inlined them; 18,159 of the 46,123 inlined WGSL lines of the AV1 encode kernel were emulated
// 64-bit arithmetic, 2026-11-12), and every one of them is slow at run time too. The transforms call this hundreds
// of times per block.
//
// How: split each input into its high and low 16 bits, in = hi * 65536 + lo with lo in 0..65535 (hi = in >> 16 is the
// floor, so this holds for negative values). Then
//     w0 * in0 + w1 * in1 + round = H * 65536 + L,   H = w0 * hi0 + w1 * hi1,   L = w0 * lo0 + w1 * lo1 + round
// and because 65536 is a multiple of 2^bit (bit <= 16)
//     floor((H * 65536 + L) / 2^bit) = H * 2^(16 - bit) + floor(L / 2^bit)          (exactly, no carry between the two).
// H and L fit in 32 bits when |w0| + |w1| <= 32767:  |H| <= 32767 * 32768 < 2^30,  |L| <= 32767 * 65535 + 32768 < 2^31.
// The final sum is taken modulo 2^32, which is what "(int)(value64 >> bit)" does.
//
// The weights are the codecs' cosine / sine constants: AV1 at most 8192 each (cos_bit <= 13), VP9 at most 16384 each
// and at most 23170 for the two of a rotation (cos + sin <= sqrt(2) * 16384). The host test
// TransformRounding_EqualsThe64BitFormula checks the bound with the real constant pairs and compares against the
// 64-bit formula on extreme and random inputs. The CPU reference transforms keep the 64-bit formula (the oracle).

using System.Runtime.CompilerServices;

namespace SpawnDev.Codecs.Video;

/// <summary>Rounded fixed-point products of the transform butterflies, exact in 32-bit arithmetic.</summary>
public static class TransformRounding
{
    /// <summary>Largest <c>|w0| + |w1|</c> (or single <c>|w|</c>) the 32-bit forms are exact for.</summary>
    public const int MaxWeightSum = 32767;

    /// <summary>
    /// <c>(int)(((long)w0 * in0 + (long)w1 * in1 + (1L &lt;&lt; (bit - 1))) &gt;&gt; bit)</c> for every <c>int</c> input,
    /// without a 64-bit intermediate. Requires <c>|w0| + |w1| &lt;= <see cref="MaxWeightSum"/></c> and
    /// <c>1 &lt;= bit &lt;= 16</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int RoundedDot(int w0, int in0, int w1, int in1, int bit)
    {
        int hi = w0 * (in0 >> 16) + w1 * (in1 >> 16);
        int lo = w0 * (in0 & 0xFFFF) + w1 * (in1 & 0xFFFF) + (1 << (bit - 1));
        return (hi << (16 - bit)) + (lo >> bit);
    }

    /// <summary>
    /// <c>(int)(((long)w * x + (1L &lt;&lt; (bit - 1))) &gt;&gt; bit)</c> for every <c>int</c> input, without a 64-bit
    /// intermediate. Requires <c>|w| &lt;= <see cref="MaxWeightSum"/></c> and <c>1 &lt;= bit &lt;= 16</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int RoundedProduct(int w, int x, int bit)
    {
        int hi = w * (x >> 16);
        int lo = w * (x & 0xFFFF) + (1 << (bit - 1));
        return (hi << (16 - bit)) + (lo >> bit);
    }
}
