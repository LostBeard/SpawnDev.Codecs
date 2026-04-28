// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the AV1 4-point forward identity transform (1D).
// Bit-exact mirror of Av1ForwardIdentity.Transform4. One thread per
// 4-element 1D transform.
//
// libaom <c>av1_fidentity4_c</c>:
//   output[i] = round_shift(input[i] * NewSqrt2, NewSqrt2Bits)
//   NewSqrt2 = 5793, NewSqrt2Bits = 12.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// Batched ILGPU kernel for the AV1 4-point forward identity transform.
/// Bit-exact mirror of <see cref="Av1ForwardIdentity.Transform4"/>.
/// </summary>
public sealed class Av1ForwardIdentity4Kernel : IDisposable
{
    private const int NewSqrt2 = Av1ForwardIdentity.NewSqrt2;
    private const int NewSqrt2Bits = Av1ForwardIdentity.NewSqrt2Bits;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Av1ForwardIdentity4Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, int>(IdentityKernel);
    }

    /// <summary>
    /// Run the identity transform on <paramref name="transformCount"/>
    /// independent 4-element transforms.
    /// </summary>
    public void Run(ArrayView<int> input, ArrayView<int> output, int transformCount)
    {
        if (transformCount < 0) throw new ArgumentOutOfRangeException(nameof(transformCount));
        if (transformCount == 0) return;
        if (input.Length < transformCount * 4L)
            throw new ArgumentException($"input must hold at least transformCount*4 ints (got {input.Length}).", nameof(input));
        if (output.Length < transformCount * 4L)
            throw new ArgumentException($"output must hold at least transformCount*4 ints (got {output.Length}).", nameof(output));
        _kernel(transformCount, input, output, transformCount);
    }

    /// <summary>Kernel body. One thread per 4-element transform.</summary>
    private static void IdentityKernel(
        Index1D transformIdx,
        ArrayView<int> input,
        ArrayView<int> output,
        int transformCount)
    {
        int idx = transformIdx;
        if (idx >= transformCount) return;
        long inBase = (long)idx * 4;
        long outBase = (long)idx * 4;

        // round_shift((long)input[i] * NewSqrt2, NewSqrt2Bits)
        const long Round = 1L << (NewSqrt2Bits - 1);
        output[outBase + 0] = (int)(((long)input[inBase + 0] * NewSqrt2 + Round) >> NewSqrt2Bits);
        output[outBase + 1] = (int)(((long)input[inBase + 1] * NewSqrt2 + Round) >> NewSqrt2Bits);
        output[outBase + 2] = (int)(((long)input[inBase + 2] * NewSqrt2 + Round) >> NewSqrt2Bits);
        output[outBase + 3] = (int)(((long)input[inBase + 3] * NewSqrt2 + Round) >> NewSqrt2Bits);
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped */ }
}
