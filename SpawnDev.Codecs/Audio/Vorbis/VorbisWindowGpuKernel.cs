// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Test/integration kernels that drive VorbisWindowGpu primitives:
//   - VorbisCanonicalWindowGpuKernel: generate the canonical synthesis
//     window of length N, one thread per sample.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// Generate the Vorbis canonical synthesis window of length N on the
/// accelerator (one thread per sample).
/// </summary>
public sealed class VorbisCanonicalWindowGpuKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<float>, int> _kernel;

    /// <summary>Compile.</summary>
    public VorbisCanonicalWindowGpuKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, int>(WindowKernel);
    }

    /// <summary>Generate window of length n into <paramref name="output"/>.</summary>
    public void Run(ArrayView<float> output, int n)
    {
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (output.Length < n)
            throw new ArgumentException("output too short.", nameof(output));
        _kernel(n, output, n);
    }

    private static void WindowKernel(Index1D threadIdx, ArrayView<float> output, int n)
    {
        int i = threadIdx;
        if (i >= n) return;
        output[i] = VorbisWindowGpu.CanonicalSample(i, n);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
