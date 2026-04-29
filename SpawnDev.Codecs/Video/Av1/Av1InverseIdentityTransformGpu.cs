// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable AV1 inverse identity transforms (IDTX direction), 1D
// building blocks for the V_DCT / H_DCT / IDTX inverse paths. Bit-exact
// mirror of Av1InverseIdentity.{Transform4, Transform8, Transform16,
// Transform32} (libaom av1_iidentity{4,8,16,32}_c).
//
// Per-element parallel: each thread reads one input value, applies the
// per-size inverse scale, and writes the output. True parallel-per-
// element across all 6 ILGPU backends.
//
// Per-size scaling (libaom):
//   size 4:  output[i] = round_shift(NewSqrt2 * input[i], NewSqrt2Bits)   (same as fwd)
//   size 8:  output[i] = input[i] * 2                                     (same as fwd)
//   size 16: output[i] = round_shift(NewSqrt2 * 2 * input[i], NewSqrt2Bits) (2x of fwd)
//   size 32: output[i] = input[i] * 4                                     (same as fwd)
//
// Note: libaom inverse identity at size 16 multiplies by 2x compared
// to the forward variant. Sizes 4 / 8 / 32 are identical to forward.

using ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// GPU-callable AV1 inverse identity transforms. Mirror of
/// <see cref="Av1InverseIdentity"/>.
/// </summary>
public static class Av1InverseIdentityTransformGpu
{
    private const int NewSqrt2 = 5793;
    private const int NewSqrt2Bits = 12;

    /// <summary>
    /// Compute one element of the 4-point inverse identity transform:
    /// <c>output[i] = round_shift(NewSqrt2 * input[i], NewSqrt2Bits)</c>.
    /// </summary>
    public static void Inverse4At(
        ArrayView<int> input, long inBase,
        ArrayView<int> output, long outBase,
        int i)
    {
        long scaled = (long)NewSqrt2 * input[inBase + i];
        output[outBase + i] = (int)((scaled + (1L << (NewSqrt2Bits - 1))) >> NewSqrt2Bits);
    }

    /// <summary>
    /// Compute one element of the 8-point inverse identity transform:
    /// <c>output[i] = input[i] * 2</c>.
    /// </summary>
    public static void Inverse8At(
        ArrayView<int> input, long inBase,
        ArrayView<int> output, long outBase,
        int i)
    {
        output[outBase + i] = input[inBase + i] * 2;
    }

    /// <summary>
    /// Compute one element of the 16-point inverse identity transform:
    /// <c>output[i] = round_shift(NewSqrt2 * 2 * input[i], NewSqrt2Bits)</c>
    /// (note the 2x scale compared to forward 16-point identity).
    /// </summary>
    public static void Inverse16At(
        ArrayView<int> input, long inBase,
        ArrayView<int> output, long outBase,
        int i)
    {
        long scaled = (long)(NewSqrt2 * 2) * input[inBase + i];
        output[outBase + i] = (int)((scaled + (1L << (NewSqrt2Bits - 1))) >> NewSqrt2Bits);
    }

    /// <summary>
    /// Compute one element of the 32-point inverse identity transform:
    /// <c>output[i] = input[i] * 4</c>.
    /// </summary>
    public static void Inverse32At(
        ArrayView<int> input, long inBase,
        ArrayView<int> output, long outBase,
        int i)
    {
        output[outBase + i] = input[inBase + i] * 4;
    }
}
