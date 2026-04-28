// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that exercises Vp9ForwardDct8x8Gpu by
// running a single-block FDCT 8x8 on the accelerator. Single-thread
// dispatch; the entire two-pass DCT runs in one GPU thread.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Drives <see cref="Vp9ForwardDct8x8Gpu.Forward8x8"/> on the
/// accelerator for one 8x8 block per dispatch.
/// </summary>
public sealed class Vp9ForwardDct8x8GpuTestKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<int>, ArrayView<int>, int> _kernel;

    /// <summary>Compile.</summary>
    public Vp9ForwardDct8x8GpuTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<int>, ArrayView<int>, int>(FdctKernel);
    }

    /// <summary>Run FDCT on a single 8x8 block. Output is row-major (stride = 8).</summary>
    public void Run(
        ArrayView<short> input,
        ArrayView<int> output,
        ArrayView<int> scratch,
        int inStride)
    {
        if (input.Length < 8L * inStride) throw new ArgumentException("input too short.", nameof(input));
        if (output.Length < 64) throw new ArgumentException("output must hold 64 ints.", nameof(output));
        if (scratch.Length < 64) throw new ArgumentException("scratch must hold 64 ints.", nameof(scratch));
        _kernel(1, input, output, scratch, inStride);
    }

    private static void FdctKernel(
        Index1D _,
        ArrayView<short> input,
        ArrayView<int> output,
        ArrayView<int> scratch,
        int inStride)
    {
        Vp9ForwardDct8x8Gpu.Forward8x8(input, 0, inStride, output, 0, scratch);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
