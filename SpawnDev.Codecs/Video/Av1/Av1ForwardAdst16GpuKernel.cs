// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that drives the Av1ForwardAdst16Gpu.Forward16
// static helper through ILGPU. Verifies the helper composes correctly
// when called from inside a kernel.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// Batched ILGPU kernel that drives
/// <see cref="Av1ForwardAdst16Gpu.Forward16"/>.
/// </summary>
public sealed class Av1ForwardAdst16GpuKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int, int> _kernel;

    /// <summary>Compile.</summary>
    public Av1ForwardAdst16GpuKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, int, int>(FadstKernel);
    }

    /// <summary>Run on <paramref name="transformCount"/> 16-element transforms.</summary>
    public void Run(ArrayView<int> input, ArrayView<int> output, int transformCount, int cosBit = Av1ForwardAdst16Gpu.DefaultCosBit)
    {
        if (transformCount < 0) throw new ArgumentOutOfRangeException(nameof(transformCount));
        if (transformCount == 0) return;
        if (cosBit < 10 || cosBit > 13) throw new ArgumentOutOfRangeException(nameof(cosBit));
        if (input.Length < transformCount * 16L)
            throw new ArgumentException($"input must hold at least transformCount*16 ints (got {input.Length}).", nameof(input));
        if (output.Length < transformCount * 16L)
            throw new ArgumentException($"output must hold at least transformCount*16 ints (got {output.Length}).", nameof(output));
        _kernel(transformCount, input, output, transformCount, cosBit);
    }

    private static void FadstKernel(
        Index1D transformIdx,
        ArrayView<int> input,
        ArrayView<int> output,
        int transformCount,
        int cosBit)
    {
        int idx = transformIdx;
        if (idx >= transformCount) return;
        long inBase = (long)idx * 16;
        long outBase = (long)idx * 16;
        Av1ForwardAdst16Gpu.Forward16(input, inBase, output, outBase, cosBit);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
