// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the AV1 32-point forward identity transform (1D).
// Bit-exact mirror of Av1ForwardIdentity.Transform32.
//
// libaom <c>av1_fidentity32_c</c>: output[i] = input[i] * 4.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// Batched ILGPU kernel for the AV1 32-point forward identity transform.
/// Bit-exact mirror of <see cref="Av1ForwardIdentity.Transform32"/>.
/// </summary>
public sealed class Av1ForwardIdentity32Kernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Av1ForwardIdentity32Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, int>(IdentityKernel);
    }

    /// <summary>
    /// Run the identity transform on <paramref name="transformCount"/>
    /// independent 32-element transforms.
    /// </summary>
    public void Run(ArrayView<int> input, ArrayView<int> output, int transformCount)
    {
        if (transformCount < 0) throw new ArgumentOutOfRangeException(nameof(transformCount));
        if (transformCount == 0) return;
        if (input.Length < transformCount * 32L)
            throw new ArgumentException($"input must hold at least transformCount*32 ints (got {input.Length}).", nameof(input));
        if (output.Length < transformCount * 32L)
            throw new ArgumentException($"output must hold at least transformCount*32 ints (got {output.Length}).", nameof(output));
        _kernel(transformCount, input, output, transformCount);
    }

    /// <summary>Kernel body. One thread per 32-element transform.</summary>
    private static void IdentityKernel(
        Index1D transformIdx,
        ArrayView<int> input,
        ArrayView<int> output,
        int transformCount)
    {
        int idx = transformIdx;
        if (idx >= transformCount) return;
        long inBase = (long)idx * 32;
        long outBase = (long)idx * 32;
        for (int i = 0; i < 32; i++) output[outBase + i] = input[inBase + i] * 4;
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped */ }
}
