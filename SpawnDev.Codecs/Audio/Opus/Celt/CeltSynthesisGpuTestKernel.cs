// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>One test case of <see cref="CeltSynthesisGpuTestKernel"/>.</summary>
public struct CeltSynthesisCase
{
    /// <summary>Frame size log2 over 120 samples, 0..3.</summary>
    public int LM;
    /// <summary>Channels, 1 or 2 (de-emphasis).</summary>
    public int C;
    /// <summary>First coded band (denormalise).</summary>
    public int Start;
    /// <summary>One past the last coded band (denormalise).</summary>
    public int End;
    /// <summary>Output decimation 1, 2, 3, 4 or 6.</summary>
    public int Downsample;
    /// <summary>Silence frame 0 / 1 (denormalise).</summary>
    public int Silence;
    /// <summary>Add into the PCM already there 0 / 1 (de-emphasis).</summary>
    public int Accum;
    /// <summary>De-emphasis coefficient (Q15).</summary>
    public int Coef0;
}

/// <summary>
/// Runs <see cref="CeltSynthesisGpu.DenormaliseBands"/> or <see cref="CeltSynthesisGpu.Deemphasis"/>, one thread per
/// <see cref="CeltSynthesisCase"/>. Each case's int block (<see cref="BlockInts"/>) holds, at the offsets below: X (the normalised
/// spectrum) or the two channels' de-emphasis input, the band energies, the output spectrum, the filter memory and the decimation
/// scratch; its short block (<see cref="PcmShorts"/>) is the interleaved PCM.
/// </summary>
public sealed class CeltSynthesisGpuTestKernel : IDisposable
{
    /// <summary>Block layout (ints from the case's block start). De-emphasis channel c starts at InAt + c * InStride.</summary>
    public const int XAt = 0, InAt = 0, InStride = 960, EAt = 1920, FreqAt = 1941, MemAt = 2901, ScratchAt = 2903, BlockInts = 3872;
    /// <summary>PCM shorts per case.</summary>
    public const int PcmShorts = 1920;

    /// <summary>Which stage to run.</summary>
    public enum Op
    {
        /// <summary>denormalise_bands.</summary>
        Denormalise,
        /// <summary>deemphasis.</summary>
        Deemphasis,
    }

    private readonly Action<Index1D, ArrayView<CeltSynthesisCase>, CeltModeGpu, ArrayView<int>, ArrayView<short>> _denormalise, _deemphasis;

    /// <summary>Compile.</summary>
    public CeltSynthesisGpuTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _denormalise = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<CeltSynthesisCase>, CeltModeGpu, ArrayView<int>,
            ArrayView<short>>(DenormaliseKernel);
        _deemphasis = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<CeltSynthesisCase>, CeltModeGpu, ArrayView<int>,
            ArrayView<short>>(DeemphasisKernel);
    }

    /// <summary>Run <paramref name="op"/> over every case.</summary>
    public void Run(Op op, ArrayView<CeltSynthesisCase> cases, CeltModeGpu mode, ArrayView<int> blocks, ArrayView<short> pcm)
        => (op == Op.Denormalise ? _denormalise : _deemphasis)((int)cases.Length, cases, mode, blocks, pcm);

    private static void DenormaliseKernel(Index1D c, ArrayView<CeltSynthesisCase> cases, CeltModeGpu mode, ArrayView<int> blocks,
        ArrayView<short> pcm)
    {
        CeltSynthesisCase k = cases[c];
        int o = c * BlockInts;
        CeltSynthesisGpu.DenormaliseBands(mode, blocks, o + XAt, blocks, o + FreqAt, blocks, o + EAt, k.Start, k.End, 1 << k.LM,
            k.Downsample, k.Silence);
    }

    private static void DeemphasisKernel(Index1D c, ArrayView<CeltSynthesisCase> cases, CeltModeGpu mode, ArrayView<int> blocks,
        ArrayView<short> pcm)
    {
        CeltSynthesisCase k = cases[c];
        int o = c * BlockInts;
        CeltSynthesisGpu.Deemphasis(blocks, o + InAt, InStride, pcm, c * PcmShorts, 120 << k.LM, k.C, k.Downsample, k.Coef0,
            blocks, o + MemAt, k.Accum, blocks, o + ScratchAt);
    }

    /// <summary>Release.</summary>
    public void Dispose() { /* auto-grouped */ }
}
