// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Wrapper kernel for VorbisFloor1InverseDbGpu.ApplyAtBin. One thread
// per output bin computes residue[i] * InverseDbTable[clamp(curveIdx[i])].

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// Per-bin parallel kernel that applies the Vorbis I floor1 curve to
/// a residue spectrum via inverse-dB table lookup + multiply.
/// </summary>
public sealed class VorbisFloor1ApplyGpuKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, ArrayView<float>, int> _kernel;

    /// <summary>Compile.</summary>
    public VorbisFloor1ApplyGpuKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, ArrayView<float>, int>(ApplyKernel);
    }

    /// <summary>
    /// Apply: <c>output[i] = residue[i] * InverseDbTable[clamp(curveIdx[i])]</c>
    /// for i in [0, halfBlock).
    /// </summary>
    public void Run(
        ArrayView<float> residue, ArrayView<int> curveIdx,
        ArrayView<float> table, ArrayView<float> output, int halfBlock)
    {
        if (halfBlock < 0) throw new ArgumentOutOfRangeException(nameof(halfBlock));
        if (halfBlock == 0) return;
        if (residue.Length < halfBlock) throw new ArgumentException("residue too short.", nameof(residue));
        if (curveIdx.Length < halfBlock) throw new ArgumentException("curveIdx too short.", nameof(curveIdx));
        if (output.Length < halfBlock) throw new ArgumentException("output too short.", nameof(output));
        if (table.Length < 256) throw new ArgumentException("table must hold 256 entries.", nameof(table));
        _kernel(halfBlock, residue, curveIdx, table, output, halfBlock);
    }

    private static void ApplyKernel(
        Index1D threadIdx,
        ArrayView<float> residue,
        ArrayView<int> curveIdx,
        ArrayView<float> table,
        ArrayView<float> output,
        int halfBlock)
    {
        int i = threadIdx;
        if (i >= halfBlock) return;
        VorbisFloor1InverseDbGpu.ApplyAtBin(residue, 0, curveIdx, 0, table, 0, output, 0, i);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
