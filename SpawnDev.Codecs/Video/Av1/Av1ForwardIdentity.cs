// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 identity transforms (forward direction). Bit-exact ports of
// libaom <c>av1_fidentity{4,8,16,32}_c</c> from
// av1/encoder/av1_fwd_txfm1d.c.
//
// Identity transforms multiply each input by a per-size constant and
// (for sizes 4 and 16) apply a libaom round_shift. They are the
// "no-frequency-transform" basis used in IDTX / V_DCT / H_DCT etc.
//
// Per-size scaling (libaom):
//   size 4 / 16:  output[i] = round_shift(input[i] * NewSqrt2, NewSqrt2Bits)
//                 with NewSqrt2 = 5793, NewSqrt2Bits = 12
//   size 8:       output[i] = input[i] * 2   (sqrt(2) folded into the
//                                             between-pass shift)
//   size 32:      output[i] = input[i] * 4   (sqrt(8) folded similarly)

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// AV1 forward identity transforms (1D building blocks). Bit-exact mirror
/// of libaom <c>av1_fidentity{4,8,16,32}_c</c>.
/// </summary>
public static class Av1ForwardIdentity
{
    /// <summary>libaom <c>NewSqrt2</c>: sqrt(2) * 2^12 rounded.</summary>
    public const int NewSqrt2 = 5793;

    /// <summary>libaom <c>NewSqrt2Bits</c>: shift count paired with <see cref="NewSqrt2"/>.</summary>
    public const int NewSqrt2Bits = 12;

    /// <summary>4-point forward identity. <c>output[i] = round_shift(input[i] * NewSqrt2, NewSqrt2Bits)</c>.</summary>
    public static void Transform4(ReadOnlySpan<int> input, Span<int> output)
    {
        if (input.Length < 4) throw new ArgumentException("input must have 4 entries", nameof(input));
        if (output.Length < 4) throw new ArgumentException("output must have 4 entries", nameof(output));
        for (int i = 0; i < 4; i++)
            output[i] = RoundShift((long)input[i] * NewSqrt2, NewSqrt2Bits);
    }

    /// <summary>8-point forward identity. <c>output[i] = input[i] * 2</c>.</summary>
    public static void Transform8(ReadOnlySpan<int> input, Span<int> output)
    {
        if (input.Length < 8) throw new ArgumentException("input must have 8 entries", nameof(input));
        if (output.Length < 8) throw new ArgumentException("output must have 8 entries", nameof(output));
        for (int i = 0; i < 8; i++) output[i] = input[i] * 2;
    }

    /// <summary>16-point forward identity. <c>output[i] = round_shift(input[i] * NewSqrt2, NewSqrt2Bits)</c>.</summary>
    public static void Transform16(ReadOnlySpan<int> input, Span<int> output)
    {
        if (input.Length < 16) throw new ArgumentException("input must have 16 entries", nameof(input));
        if (output.Length < 16) throw new ArgumentException("output must have 16 entries", nameof(output));
        for (int i = 0; i < 16; i++)
            output[i] = RoundShift((long)input[i] * NewSqrt2, NewSqrt2Bits);
    }

    /// <summary>32-point forward identity. <c>output[i] = input[i] * 4</c>.</summary>
    public static void Transform32(ReadOnlySpan<int> input, Span<int> output)
    {
        if (input.Length < 32) throw new ArgumentException("input must have 32 entries", nameof(input));
        if (output.Length < 32) throw new ArgumentException("output must have 32 entries", nameof(output));
        for (int i = 0; i < 32; i++) output[i] = input[i] * 4;
    }

    /// <summary>libaom <c>round_shift</c>: arithmetic round-half-up by <paramref name="bit"/> bits.</summary>
    private static int RoundShift(long value, int bit)
    {
        return (int)((value + (1L << (bit - 1))) >> bit);
    }
}
