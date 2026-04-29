// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Single-shot wrapper kernel for SilkBwexpanderGpu. Both Expand16
// and Expand32 are sequential per-coefficient (chirpQ16 update
// dependency), so this runs as a single-thread dispatch.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Single-shot ILGPU kernel for SILK bandwidth expansion. Mode 16 =
/// Expand16 (short coefficients), mode 32 = Expand32 (int
/// coefficients). The two modes ship as separate kernels to keep
/// the type signatures simple.
/// </summary>
public sealed class SilkBwexpanderGpuKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<short>, int, int> _kernel16;
    private readonly Action<Index1D, ArrayView<int>, int, int> _kernel32;

    /// <summary>Compile both kernels.</summary>
    public SilkBwexpanderGpuKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel16 = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, int, int>(Expand16Kernel);
        _kernel32 = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, int, int>(Expand32Kernel);
    }

    /// <summary>Expand a 16-bit AR filter in place.</summary>
    public void Run16(ArrayView<short> ar, int d, int chirpQ16)
    {
        if (d <= 0) return;
        if (ar.Length < d) throw new ArgumentException("ar too short.", nameof(ar));
        _kernel16(1, ar, d, chirpQ16);
    }

    /// <summary>Expand a 32-bit AR filter in place.</summary>
    public void Run32(ArrayView<int> ar, int d, int chirpQ16)
    {
        if (d <= 0) return;
        if (ar.Length < d) throw new ArgumentException("ar too short.", nameof(ar));
        _kernel32(1, ar, d, chirpQ16);
    }

    private static void Expand16Kernel(Index1D _, ArrayView<short> ar, int d, int chirpQ16)
        => SilkBwexpanderGpu.Expand16(ar, 0, d, chirpQ16);

    private static void Expand32Kernel(Index1D _, ArrayView<int> ar, int d, int chirpQ16)
        => SilkBwexpanderGpu.Expand32(ar, 0, d, chirpQ16);

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
