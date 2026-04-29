// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernel that drives FlacCrcGpu through ILGPU.
// Computes CRC-8 + CRC-16 over a single byte range; writes the
// 8-bit + 16-bit results into outputs.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// Single-shot ILGPU kernel that drives <see cref="FlacCrcGpu"/>.
/// Computes both CRC-8 and CRC-16 over the given byte range.
/// </summary>
public sealed class FlacCrcGpuKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<ushort>, int> _kernel;

    /// <summary>Compile.</summary>
    public FlacCrcGpuKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<ushort>, int>(CrcKernel);
    }

    /// <summary>Compute CRC-8 (out8[0]) and CRC-16 (out16[0]) over data[0..length).</summary>
    public void Run(ArrayView<byte> data, ArrayView<byte> out8, ArrayView<ushort> out16, int length)
    {
        _kernel(1, data, out8, out16, length);
    }

    private static void CrcKernel(
        Index1D _,
        ArrayView<byte> data,
        ArrayView<byte> out8,
        ArrayView<ushort> out16,
        int length)
    {
        out8[0] = FlacCrcGpu.Compute8(data, 0, length);
        out16[0] = FlacCrcGpu.Compute16(data, 0, length);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
