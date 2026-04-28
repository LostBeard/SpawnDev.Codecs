// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the AV1 8-point forward identity transform (1D).
// Bit-exact mirror of Av1ForwardIdentity.Transform8.
//
// libaom <c>av1_fidentity8_c</c>: output[i] = input[i] * 2.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// Batched ILGPU kernel for the AV1 8-point forward identity transform.
/// Bit-exact mirror of <see cref="Av1ForwardIdentity.Transform8"/>.
/// </summary>
public sealed class Av1ForwardIdentity8Kernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Av1ForwardIdentity8Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, int>(IdentityKernel);
    }

    /// <summary>
    /// Run the identity transform on <paramref name="transformCount"/>
    /// independent 8-element transforms.
    /// </summary>
    public void Run(ArrayView<int> input, ArrayView<int> output, int transformCount)
    {
        if (transformCount < 0) throw new ArgumentOutOfRangeException(nameof(transformCount));
        if (transformCount == 0) return;
        if (input.Length < transformCount * 8L)
            throw new ArgumentException($"input must hold at least transformCount*8 ints (got {input.Length}).", nameof(input));
        if (output.Length < transformCount * 8L)
            throw new ArgumentException($"output must hold at least transformCount*8 ints (got {output.Length}).", nameof(output));
        _kernel(transformCount, input, output, transformCount);
    }

    /// <summary>Kernel body. One thread per 8-element transform.</summary>
    private static void IdentityKernel(
        Index1D transformIdx,
        ArrayView<int> input,
        ArrayView<int> output,
        int transformCount)
    {
        int idx = transformIdx;
        if (idx >= transformCount) return;
        long inBase = (long)idx * 8;
        long outBase = (long)idx * 8;
        output[outBase + 0] = input[inBase + 0] * 2;
        output[outBase + 1] = input[inBase + 1] * 2;
        output[outBase + 2] = input[inBase + 2] * 2;
        output[outBase + 3] = input[inBase + 3] * 2;
        output[outBase + 4] = input[inBase + 4] * 2;
        output[outBase + 5] = input[inBase + 5] * 2;
        output[outBase + 6] = input[inBase + 6] * 2;
        output[outBase + 7] = input[inBase + 7] * 2;
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped */ }
}
