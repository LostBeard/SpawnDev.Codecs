// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test kernel that drives `SilkPitchComputeLagsGpu.ComputeLags`.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Drives `SilkPitchComputeLagsGpu.ComputeLags` on the accelerator.
/// </summary>
public sealed class SilkPitchComputeLagsGpuTestKernel : IDisposable
{
    private readonly Action<
        Index1D,
        ArrayView<sbyte>, int,
        int, int, int, int,
        ArrayView<int>> _kernel;

    /// <summary>Compile.</summary>
    public SilkPitchComputeLagsGpuTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<sbyte>, int,
            int, int, int, int,
            ArrayView<int>>(ComputeLagsKernel);
    }

    /// <summary>Run the kernel.</summary>
    public void Run(
        ArrayView<sbyte> contourCb, int cbSize,
        int lagIndex, int contourIndex, int fsKHz, int nbSubfr,
        ArrayView<int> pitchLagsOut)
    {
        if (pitchLagsOut.Length < nbSubfr)
            throw new ArgumentException("pitchLagsOut too short.", nameof(pitchLagsOut));
        _kernel(1, contourCb, cbSize, lagIndex, contourIndex, fsKHz, nbSubfr, pitchLagsOut);
    }

    private static void ComputeLagsKernel(
        Index1D _,
        ArrayView<sbyte> contourCb, int cbSize,
        int lagIndex, int contourIndex, int fsKHz, int nbSubfr,
        ArrayView<int> pitchLagsOut)
    {
        SilkPitchComputeLagsGpu.ComputeLags(
            contourCb, 0, cbSize,
            lagIndex, contourIndex, fsKHz, nbSubfr,
            pitchLagsOut, 0);
    }

    /// <summary>Release.</summary>
    public void Dispose() { /* auto-grouped */ }
}
