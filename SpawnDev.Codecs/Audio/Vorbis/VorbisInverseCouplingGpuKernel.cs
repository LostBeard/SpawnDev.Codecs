// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Wrapper kernel for VorbisInverseCouplingGpu. Each thread processes
// one coefficient of one (mag, ang) channel pair. Caller dispatches
// per coupling step in REVERSE order of the encoder's coupling steps.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// Per-coefficient inverse coupling kernel. One thread per coefficient
/// of one (mag, ang) channel pair.
/// </summary>
public sealed class VorbisInverseCouplingGpuKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, int> _kernel;

    /// <summary>Compile.</summary>
    public VorbisInverseCouplingGpuKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int>(CouplingKernel);
    }

    /// <summary>
    /// Run inverse coupling on one channel pair. <paramref name="magBuf"/>
    /// and <paramref name="angBuf"/> are mutated in place; both must
    /// hold at least <paramref name="coefCount"/> floats.
    /// </summary>
    public void Run(ArrayView<float> magBuf, ArrayView<float> angBuf, int coefCount)
    {
        if (coefCount < 0) throw new ArgumentOutOfRangeException(nameof(coefCount));
        if (coefCount == 0) return;
        if (magBuf.Length < coefCount) throw new ArgumentException("magBuf too short.", nameof(magBuf));
        if (angBuf.Length < coefCount) throw new ArgumentException("angBuf too short.", nameof(angBuf));
        _kernel(coefCount, magBuf, angBuf, coefCount);
    }

    private static void CouplingKernel(
        Index1D threadIdx,
        ArrayView<float> magBuf,
        ArrayView<float> angBuf,
        int coefCount)
    {
        int i = threadIdx;
        if (i >= coefCount) return;
        VorbisInverseCouplingGpu.ApplyAtCoefficient(magBuf, 0, angBuf, 0, i);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
