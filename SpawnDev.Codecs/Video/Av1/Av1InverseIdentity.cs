// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 inverse identity transform (IDTX) - 1D building block for the
// non-trigonometric "transform-skip" transform used in AV1's 16-way
// per-block tx-type selection.
//
// Bit-exact port of libaom av1/common/av1_inv_txfm1d.c
// av1_iidentity{4,8,16,32}_c.
//
// Upstream Copyright (c) 2016, Alliance for Open Media.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.
//
// IDTX semantics: scale input by sqrt(2 * size / 4). For size=4 the
// scaling is sqrt(2) (NewSqrt2 / 2^NewSqrt2Bits). For size=8 the
// scaling is exactly 2. For size=16 the scaling is 2*sqrt(2). For
// size=32 the scaling is exactly 4.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 inverse identity transform (IDTX) 1D building block.</summary>
public static class Av1InverseIdentity
{
    /// <summary>libaom <c>NewSqrt2Bits</c>.</summary>
    public const int NewSqrt2Bits = 12;

    /// <summary>libaom <c>NewSqrt2</c> = round(sqrt(2) * 2^12).</summary>
    public const int NewSqrt2 = 5793;

    /// <summary>libaom <c>round_shift</c> for 64-bit values.</summary>
    public static int RoundShift(long value, int bit)
        => (int)((value + (1L << (bit - 1))) >> bit);

    /// <summary>4-point inverse identity. Mirrors libaom <c>av1_iidentity4_c</c>.</summary>
    public static void Transform4(ReadOnlySpan<int> input, Span<int> output)
    {
        if (input.Length < 4) throw new ArgumentException("input must have 4 entries", nameof(input));
        if (output.Length < 4) throw new ArgumentException("output must have 4 entries", nameof(output));
        for (int i = 0; i < 4; i++)
            output[i] = RoundShift((long)NewSqrt2 * input[i], NewSqrt2Bits);
    }

    /// <summary>8-point inverse identity. Mirrors libaom <c>av1_iidentity8_c</c>.</summary>
    public static void Transform8(ReadOnlySpan<int> input, Span<int> output)
    {
        if (input.Length < 8) throw new ArgumentException("input must have 8 entries", nameof(input));
        if (output.Length < 8) throw new ArgumentException("output must have 8 entries", nameof(output));
        for (int i = 0; i < 8; i++)
            output[i] = (int)((long)input[i] * 2);
    }

    /// <summary>16-point inverse identity. Mirrors libaom <c>av1_iidentity16_c</c>.</summary>
    public static void Transform16(ReadOnlySpan<int> input, Span<int> output)
    {
        if (input.Length < 16) throw new ArgumentException("input must have 16 entries", nameof(input));
        if (output.Length < 16) throw new ArgumentException("output must have 16 entries", nameof(output));
        for (int i = 0; i < 16; i++)
            output[i] = RoundShift((long)NewSqrt2 * 2 * input[i], NewSqrt2Bits);
    }

    /// <summary>32-point inverse identity. Mirrors libaom <c>av1_iidentity32_c</c>.</summary>
    public static void Transform32(ReadOnlySpan<int> input, Span<int> output)
    {
        if (input.Length < 32) throw new ArgumentException("input must have 32 entries", nameof(input));
        if (output.Length < 32) throw new ArgumentException("output must have 32 entries", nameof(output));
        for (int i = 0; i < 32; i++)
            output[i] = (int)((long)input[i] * 4);
    }
}
