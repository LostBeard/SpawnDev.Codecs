// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>One test case of <see cref="CeltMdctGpuTestKernel"/>: one channel of one frame.</summary>
public struct CeltMdctCase
{
    /// <summary>MDCT shift: 3 - LM for a long block, 3 for short blocks.</summary>
    public int Shift;
    /// <summary>Blocks in the frame (1, or 1 &lt;&lt; LM short blocks), which is also the coefficient stride.</summary>
    public int Blocks;
}

/// <summary>
/// Runs <see cref="CeltMdctGpu.Backward"/> as celt_synthesis does (each block's N / 2 + overlap samples at N / 2 * block, the
/// coefficients interleaved by block) or <see cref="CeltMdctGpu.Forward"/> as the encoder's compute_mdcts does, one thread per
/// <see cref="CeltMdctCase"/>. Each case's block (<see cref="BlockInts"/> ints) holds the input at <see cref="InAt"/>, the
/// output at <see cref="OutAt"/> and the forward MDCT's scratch.
/// </summary>
public sealed class CeltMdctGpuTestKernel : IDisposable
{
    /// <summary>Block layout (ints from the case's block start).</summary>
    public const int InAt = 0, OutAt = 1088, ScratchAt = 2176, BlockInts = 4096;
    /// <summary>The overlap of the static mode.</summary>
    public const int Overlap = 120;

    private readonly Action<Index1D, ArrayView<CeltMdctCase>, CeltModeGpu, ArrayView<int>> _backward, _forward;

    /// <summary>Compile.</summary>
    public CeltMdctGpuTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _backward = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<CeltMdctCase>, CeltModeGpu, ArrayView<int>>(BackwardKernel);
        _forward = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<CeltMdctCase>, CeltModeGpu, ArrayView<int>>(ForwardKernel);
    }

    /// <summary>Run the inverse (<paramref name="backward"/>) or the forward MDCT over every case.</summary>
    public void Run(bool backward, ArrayView<CeltMdctCase> cases, CeltModeGpu mode, ArrayView<int> blocks)
        => (backward ? _backward : _forward)((int)cases.Length, cases, mode, blocks);

    private static void BackwardKernel(Index1D c, ArrayView<CeltMdctCase> cases, CeltModeGpu mode, ArrayView<int> blocks)
    {
        CeltMdctCase k = cases[c];
        int o = c * BlockInts;
        int half = (CeltMdctGpu.MdctN >> k.Shift) >> 1;
        for (int b = 0; b < k.Blocks; b++)
            CeltMdctGpu.Backward(mode, blocks, o + InAt + b, blocks, o + OutAt + half * b, Overlap, k.Shift, k.Blocks);
    }

    private static void ForwardKernel(Index1D c, ArrayView<CeltMdctCase> cases, CeltModeGpu mode, ArrayView<int> blocks)
    {
        CeltMdctCase k = cases[c];
        int o = c * BlockInts;
        int half = (CeltMdctGpu.MdctN >> k.Shift) >> 1;
        for (int b = 0; b < k.Blocks; b++)
            CeltMdctGpu.Forward(mode, blocks, o + InAt + half * b, blocks, o + OutAt + b, Overlap, k.Shift, k.Blocks, blocks, o + ScratchAt);
    }

    /// <summary>Release.</summary>
    public void Dispose() { /* auto-grouped */ }
}
