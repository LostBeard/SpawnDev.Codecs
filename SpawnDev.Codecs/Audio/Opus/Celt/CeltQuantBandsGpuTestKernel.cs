// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>One test case of <see cref="CeltQuantBandsGpuTestKernel"/>: a CELT frame's allocation + band decode.</summary>
public struct CeltQuantBandsCase
{
    /// <summary>Channels, 1 or 2.</summary>
    public int C;
    /// <summary>Frame size log2 over 120 samples, 0..3.</summary>
    public int LM;
    /// <summary>First coded band.</summary>
    public int Start;
    /// <summary>One past the last coded band.</summary>
    public int End;
    /// <summary>Allocation trim 0..10.</summary>
    public int AllocTrim;
    /// <summary>Spread decision 0..3.</summary>
    public int Spread;
    /// <summary>Short blocks (1 << LM when the frame is transient, else 0).</summary>
    public int ShortBlocks;
    /// <summary>Folding noise seed.</summary>
    public uint Seed;
    /// <summary>Bits for the allocation, 1/8 bit.</summary>
    public int Total;
    /// <summary>Bits for the bands (quant_all_bands total_bits), 1/8 bit.</summary>
    public int TotalBits;
    /// <summary>The previous frame's coded band count.</summary>
    public int Prev;
    /// <summary>Highest band with signal.</summary>
    public int SignalBandwidth;
    /// <summary>Byte offset of the packet.</summary>
    public int PacketAt;
    /// <summary>Byte length of the packet.</summary>
    public int PacketLength;
}

/// <summary>
/// Drives <see cref="CeltRateGpu.ComputeAllocationDecode"/> then <see cref="CeltQuantBandsGpu.QuantAllBandsDecode"/> on one range
/// decoder, one thread per <see cref="CeltQuantBandsCase"/>. Each case's block (<see cref="BlockInts"/> ints) holds, at the
/// offsets below: X, Y, the folding history, then the per-band inputs / outputs and the results.
/// </summary>
public sealed class CeltQuantBandsGpuTestKernel : IDisposable
{
    /// <summary>Block layout (ints from the case's block start).</summary>
    public const int XAt = 0, YAt = 800, NormAt = 1600, OffsetsAt = 2848, TfAt = 2869, CapAt = 2890, PulsesAt = 2911, EbitsAt = 2932,
        FineAt = 2953, MasksAt = 2974, ResultsAt = 3016, BlockInts = 3072;
    /// <summary>Results: coded bands, balance, intensity, dual stereo, seed after, tell_frac after.</summary>
    public const int ResultCount = 6;

    private readonly Action<Index1D, ArrayView<CeltQuantBandsCase>, CeltModeGpu, ArrayView<int>, ArrayView<int>, ArrayView<byte>,
        ArrayView<uint>> _kernel;

    /// <summary>Compile.</summary>
    public CeltQuantBandsGpuTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<CeltQuantBandsCase>, CeltModeGpu, ArrayView<int>,
            ArrayView<int>, ArrayView<byte>, ArrayView<uint>>(TestKernel);
    }

    /// <summary>Run every case: <paramref name="blocks"/> = BlockInts per case (offsets and tf_res filled in),
    /// <paramref name="scratch"/> = (4 * NbEBands + CeltQuantBandsGpu.ScratchInts) per case.</summary>
    public void Run(ArrayView<CeltQuantBandsCase> cases, CeltModeGpu mode, ArrayView<int> blocks, ArrayView<int> scratch,
        ArrayView<byte> packets, ArrayView<uint> pvqU)
        => _kernel((int)cases.Length, cases, mode, blocks, scratch, packets, pvqU);

    /// <summary>Scratch ints per case.</summary>
    public const int ScratchPerCase = 4 * CeltModeTables.NbEBands + CeltQuantBandsGpu.ScratchInts;

    private static void TestKernel(Index1D c, ArrayView<CeltQuantBandsCase> cases, CeltModeGpu mode, ArrayView<int> blocks,
        ArrayView<int> scratch, ArrayView<byte> packets, ArrayView<uint> pvqU)
    {
        CeltQuantBandsCase k = cases[c];
        int o = c * BlockInts;
        int s = c * ScratchPerCase;
        var offsets = blocks.SubView(OffsetsAt, blocks.Length - OffsetsAt);
        var cap = blocks.SubView(CapAt, blocks.Length - CapAt);
        var pulses = blocks.SubView(PulsesAt, blocks.Length - PulsesAt);
        var ebits = blocks.SubView(EbitsAt, blocks.Length - EbitsAt);
        var fine = blocks.SubView(FineAt, blocks.Length - FineAt);
        CeltRateGpu.InitCaps(mode, cap, o, k.LM, k.C);
        var dec = OpusRangeDecoderGpu.Init(packets, k.PacketAt, (uint)k.PacketLength);
        int intensity = 0, dualStereo = 0;
        int coded = CeltRateGpu.ComputeAllocationDecode(mode, k.Start, k.End, offsets, cap, k.AllocTrim, ref intensity, ref dualStereo,
            k.Total, out int balance, pulses, ebits, fine, o, scratch, s, k.C, k.LM, ref dec, packets, k.PacketAt, (uint)k.PacketLength,
            k.Prev, k.SignalBandwidth);
        uint seed = k.Seed;
        CeltQuantBandsGpu.QuantAllBandsDecode(mode, k.Start, k.End, blocks, o + XAt, k.C == 2 ? o + YAt : -1, o + NormAt,
            blocks, o + MasksAt, blocks, o + PulsesAt, k.ShortBlocks, k.Spread, dualStereo, intensity, blocks, o + TfAt,
            k.TotalBits, balance, ref dec, packets, k.PacketAt, (uint)k.PacketLength, k.LM, coded, ref seed,
            scratch, s + 4 * CeltModeTables.NbEBands, pvqU, 0);
        blocks[o + ResultsAt] = coded;
        blocks[o + ResultsAt + 1] = balance;
        blocks[o + ResultsAt + 2] = intensity;
        blocks[o + ResultsAt + 3] = dualStereo;
        blocks[o + ResultsAt + 4] = (int)seed;
        blocks[o + ResultsAt + 5] = (int)CeltEntropyGpu.TellFrac(ref dec);
    }

    /// <summary>Release.</summary>
    public void Dispose() { /* auto-grouped */ }
}
