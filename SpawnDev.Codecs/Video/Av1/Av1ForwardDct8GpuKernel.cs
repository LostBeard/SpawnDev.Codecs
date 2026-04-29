// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that drives the Av1ForwardDct8Gpu.Forward8
// static helper through ILGPU. Exists so we can verify the helper
// composes correctly when called from inside a kernel - the same
// call shape Av1FrameSequentialEncodeKernel will use later.
//
// One thread per 8-element 1D transform. Bit-exact mirror of
// Av1ForwardDct8.Transform CPU reference.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// Batched ILGPU kernel that drives the
/// <see cref="Av1ForwardDct8Gpu.Forward8"/> static helper. Used to
/// verify the helper composes inside a kernel.
/// </summary>
public sealed class Av1ForwardDct8GpuKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int, int> _kernel;

    /// <summary>Compile.</summary>
    public Av1ForwardDct8GpuKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, int, int>(FdctKernel);
    }

    /// <summary>
    /// Run the FDCT helper on <paramref name="transformCount"/>
    /// independent 8-element transforms.
    /// </summary>
    public void Run(ArrayView<int> input, ArrayView<int> output, int transformCount, int cosBit = Av1ForwardDct8Gpu.DefaultCosBit)
    {
        if (transformCount < 0) throw new ArgumentOutOfRangeException(nameof(transformCount));
        if (transformCount == 0) return;
        if (cosBit < 10 || cosBit > 13) throw new ArgumentOutOfRangeException(nameof(cosBit), "must be in [10, 13]");
        if (input.Length < transformCount * 8L)
            throw new ArgumentException($"input must hold at least transformCount*8 ints (got {input.Length}).", nameof(input));
        if (output.Length < transformCount * 8L)
            throw new ArgumentException($"output must hold at least transformCount*8 ints (got {output.Length}).", nameof(output));
        _kernel(transformCount, input, output, transformCount, cosBit);
    }

    private static void FdctKernel(
        Index1D transformIdx,
        ArrayView<int> input,
        ArrayView<int> output,
        int transformCount,
        int cosBit)
    {
        int idx = transformIdx;
        if (idx >= transformCount) return;
        long inBase = (long)idx * 8;
        long outBase = (long)idx * 8;
        Av1ForwardDct8Gpu.Forward8(input, inBase, output, outBase, cosBit);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
