// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>One test case of <see cref="CeltEnergyGpuTestKernel"/>: a CELT frame's energy and TF decode.</summary>
public struct CeltEnergyCase
{
    /// <summary>Channels, 1 or 2.</summary>
    public int C;
    /// <summary>Frame size log2 over 120 samples, 0..3.</summary>
    public int LM;
    /// <summary>First coded band.</summary>
    public int Start;
    /// <summary>One past the last coded band.</summary>
    public int End;
    /// <summary>Intra (no prediction from the previous frame) 0 / 1.</summary>
    public int Intra;
    /// <summary>Transient frame 0 / 1.</summary>
    public int IsTransient;
    /// <summary>Whole bits left for unquant_energy_finalise.</summary>
    public int BitsLeft;
    /// <summary>Byte offset of the packet.</summary>
    public int PacketAt;
    /// <summary>Byte length of the packet.</summary>
    public int PacketLength;
}

/// <summary>
/// Runs <see cref="CeltEnergyGpu.UnquantCoarseEnergy"/>, <see cref="CeltSynthesisGpu.TfDecode"/>,
/// <see cref="CeltEnergyGpu.UnquantFineEnergy"/> and <see cref="CeltEnergyGpu.UnquantEnergyFinalise"/> on one range decoder, in
/// celt_decode_with_ec's order, one thread per <see cref="CeltEnergyCase"/>. Each case's block (<see cref="BlockInts"/> ints)
/// holds, at the offsets below: the band energies (in / out, 2 x NbEBands), fine_quant and fine_priority (in), tf_res (out) and
/// the results.
/// </summary>
public sealed class CeltEnergyGpuTestKernel : IDisposable
{
    /// <summary>Block layout (ints from the case's block start).</summary>
    public const int OldEAt = 0, FineAt = 42, PrioAt = 63, TfAt = 84, ResultsAt = 105, BlockInts = 112;
    /// <summary>Results: tell_frac after coarse, after tf, after fine, after finalise; the final range.</summary>
    public const int ResultCount = 5;

    private readonly Action<Index1D, ArrayView<CeltEnergyCase>, CeltModeGpu, ArrayView<int>, ArrayView<byte>> _kernel;

    /// <summary>Compile.</summary>
    public CeltEnergyGpuTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<CeltEnergyCase>, CeltModeGpu, ArrayView<int>,
            ArrayView<byte>>(TestKernel);
    }

    /// <summary>Run every case: <paramref name="blocks"/> = BlockInts per case (energies, fine_quant, fine_priority filled in).</summary>
    public void Run(ArrayView<CeltEnergyCase> cases, CeltModeGpu mode, ArrayView<int> blocks, ArrayView<byte> packets)
        => _kernel((int)cases.Length, cases, mode, blocks, packets);

    private static void TestKernel(Index1D c, ArrayView<CeltEnergyCase> cases, CeltModeGpu mode, ArrayView<int> blocks,
        ArrayView<byte> packets)
    {
        CeltEnergyCase k = cases[c];
        int o = c * BlockInts;
        uint storage = (uint)k.PacketLength;
        var dec = OpusRangeDecoderGpu.Init(packets, k.PacketAt, storage);
        CeltEnergyGpu.UnquantCoarseEnergy(mode, k.Start, k.End, blocks, o + OldEAt, k.Intra, ref dec, packets, k.PacketAt, storage, k.C, k.LM);
        blocks[o + ResultsAt] = (int)CeltEntropyGpu.TellFrac(ref dec);
        CeltSynthesisGpu.TfDecode(mode, k.Start, k.End, k.IsTransient, blocks, o + TfAt, k.LM, ref dec, packets, k.PacketAt, storage);
        blocks[o + ResultsAt + 1] = (int)CeltEntropyGpu.TellFrac(ref dec);
        CeltEnergyGpu.UnquantFineEnergy(k.Start, k.End, blocks, o + OldEAt, blocks, o + FineAt, ref dec, packets, k.PacketAt, storage, k.C);
        blocks[o + ResultsAt + 2] = (int)CeltEntropyGpu.TellFrac(ref dec);
        CeltEnergyGpu.UnquantEnergyFinalise(k.Start, k.End, blocks, o + OldEAt, blocks, o + FineAt, blocks, o + PrioAt, k.BitsLeft,
            ref dec, packets, k.PacketAt, storage, k.C);
        blocks[o + ResultsAt + 3] = (int)CeltEntropyGpu.TellFrac(ref dec);
        blocks[o + ResultsAt + 4] = (int)dec.Rng;
    }

    /// <summary>Release.</summary>
    public void Dispose() { /* auto-grouped */ }
}
