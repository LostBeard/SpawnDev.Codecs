// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>One test case of <see cref="CeltRateGpuTestKernel"/>: the arguments of one compute_allocation call.</summary>
public struct CeltRateCase
{
    /// <summary>First coded band.</summary>
    public int Start;
    /// <summary>One past the last coded band.</summary>
    public int End;
    /// <summary>Allocation trim 0..10.</summary>
    public int AllocTrim;
    /// <summary>Bits available, 1/8 bit.</summary>
    public int Total;
    /// <summary>Channels, 1 or 2.</summary>
    public int C;
    /// <summary>Frame size log2 over 120 samples, 0..3.</summary>
    public int LM;
    /// <summary>The previous frame's coded band count (skip hysteresis).</summary>
    public int Prev;
    /// <summary>The highest band with signal (skip decision).</summary>
    public int SignalBandwidth;
    /// <summary>Byte offset of this case's range-coded packet.</summary>
    public int PacketAt;
    /// <summary>Byte length of this case's packet.</summary>
    public int PacketLength;
}

/// <summary>
/// Drives <see cref="CeltRateGpu"/> on the accelerator, one thread per <see cref="CeltRateCase"/>: InitCaps, then
/// ComputeAllocationDecode. Each case's block (<see cref="OutInts"/> ints at <c>case * OutInts</c>) holds offsets (input, the
/// dynalloc boosts), then cap, pulses, ebits and fine_priority (NbEBands ints each, in that order), then coded bands, balance,
/// intensity, dual stereo. Every per-band array is the same view shifted by a multiple of NbEBands, so all of them index at the
/// case's block start, as CeltRateGpu expects (one shared offset for its per-band views).
/// </summary>
public sealed class CeltRateGpuTestKernel : IDisposable
{
    /// <summary>Ints per case in the output view.</summary>
    public const int OutInts = 5 * CeltModeTables.NbEBands + 4;

    private readonly Action<Index1D, ArrayView<CeltRateCase>, CeltModeGpu, ArrayView<int>, ArrayView<int>, ArrayView<byte>> _kernel;

    /// <summary>Compile.</summary>
    public CeltRateGpuTestKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<CeltRateCase>, CeltModeGpu, ArrayView<int>,
            ArrayView<int>, ArrayView<byte>>(TestKernel);
    }

    /// <summary>
    /// Run every case. <paramref name="blocks"/> holds <see cref="OutInts"/> ints per case (offsets filled in, the rest output),
    /// <paramref name="scratch"/> 4 * NbEBands per case.
    /// </summary>
    public void Run(ArrayView<CeltRateCase> cases, CeltModeGpu mode, ArrayView<int> blocks, ArrayView<int> scratch, ArrayView<byte> packets)
        => _kernel((int)cases.Length, cases, mode, blocks, scratch, packets);

    private static void TestKernel(Index1D c, ArrayView<CeltRateCase> cases, CeltModeGpu mode, ArrayView<int> blocks,
        ArrayView<int> scratch, ArrayView<byte> packets)
    {
        const int nb = CeltModeTables.NbEBands;
        CeltRateCase k = cases[c];
        int o = c * OutInts;
        var offsets = blocks;
        var cap = blocks.SubView(nb, blocks.Length - nb);
        var pulses = blocks.SubView(2 * nb, blocks.Length - 2 * nb);
        var ebits = blocks.SubView(3 * nb, blocks.Length - 3 * nb);
        var fine = blocks.SubView(4 * nb, blocks.Length - 4 * nb);
        CeltRateGpu.InitCaps(mode, cap, o, k.LM, k.C);
        var dec = OpusRangeDecoderGpu.Init(packets, k.PacketAt, (uint)k.PacketLength);
        int intensity = 0, dualStereo = 0;
        int coded = CeltRateGpu.ComputeAllocationDecode(mode, k.Start, k.End, offsets, cap, k.AllocTrim, ref intensity, ref dualStereo,
            k.Total, out int balance, pulses, ebits, fine, o, scratch, c * 4 * nb, k.C, k.LM,
            ref dec, packets, k.PacketAt, (uint)k.PacketLength, k.Prev, k.SignalBandwidth);
        blocks[o + 5 * nb] = coded;
        blocks[o + 5 * nb + 1] = balance;
        blocks[o + 5 * nb + 2] = intensity;
        blocks[o + 5 * nb + 3] = dualStereo;
    }

    /// <summary>Release.</summary>
    public void Dispose() { /* auto-grouped */ }
}
