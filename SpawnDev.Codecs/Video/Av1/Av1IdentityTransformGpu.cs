// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable AV1 identity transforms (forward direction), 1D
// building blocks for the V_DCT / H_DCT / IDTX paths. Bit-exact mirror
// of Av1ForwardIdentity.{Transform4, Transform8, Transform16,
// Transform32} (libaom av1_fidentity{4,8,16,32}_c).
//
// Per-element parallel: each thread reads one input value, applies the
// per-size scale, and writes the output. True parallel-per-element
// across all 6 ILGPU backends. Caller dispatches `size` threads
// (4 / 8 / 16 / 32 depending on which Apply method is used).
//
// Per-size scaling:
//   size 4 / 16: output[i] = round_shift(input[i] * NewSqrt2, NewSqrt2Bits)
//                            (NewSqrt2 = 5793, NewSqrt2Bits = 12)
//   size 8:      output[i] = input[i] * 2
//   size 32:     output[i] = input[i] * 4

using ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// GPU-callable AV1 forward identity transforms. Mirror of
/// <see cref="Av1ForwardIdentity"/>.
/// </summary>
public static class Av1IdentityTransformGpu
{
    private const int NewSqrt2 = 5793;
    private const int NewSqrt2Bits = 12;

    /// <summary>
    /// Compute one element of the 4-point forward identity transform:
    /// <c>output[i] = round_shift(input[i] * NewSqrt2, NewSqrt2Bits)</c>.
    /// </summary>
    public static void Forward4At(
        ArrayView<int> input, long inBase,
        ArrayView<int> output, long outBase,
        int i)
    {
        long scaled = (long)input[inBase + i] * NewSqrt2;
        output[outBase + i] = (int)((scaled + (1L << (NewSqrt2Bits - 1))) >> NewSqrt2Bits);
    }

    /// <summary>
    /// Compute one element of the 8-point forward identity transform:
    /// <c>output[i] = input[i] * 2</c>.
    /// </summary>
    public static void Forward8At(
        ArrayView<int> input, long inBase,
        ArrayView<int> output, long outBase,
        int i)
    {
        output[outBase + i] = input[inBase + i] * 2;
    }

    /// <summary>
    /// Compute one element of the 16-point forward identity transform:
    /// <c>output[i] = round_shift(input[i] * NewSqrt2, NewSqrt2Bits)</c>.
    /// </summary>
    public static void Forward16At(
        ArrayView<int> input, long inBase,
        ArrayView<int> output, long outBase,
        int i)
    {
        long scaled = (long)input[inBase + i] * NewSqrt2;
        output[outBase + i] = (int)((scaled + (1L << (NewSqrt2Bits - 1))) >> NewSqrt2Bits);
    }

    /// <summary>
    /// Compute one element of the 32-point forward identity transform:
    /// <c>output[i] = input[i] * 4</c>.
    /// </summary>
    public static void Forward32At(
        ArrayView<int> input, long inBase,
        ArrayView<int> output, long outBase,
        int i)
    {
        output[outBase + i] = input[inBase + i] * 4;
    }
}
