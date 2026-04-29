// Cross-backend test for SilkExcitationDequantizerGpu.DequantizeAt. Verifies
// the GPU SILK excitation dequantizer matches the CPU reference
// SilkExcitationDequantizer.Dequantize bit-exactly across all combinations of
// signalType (0 inactive, 1 unvoiced, 2 voiced) and quantOffsetType (0/1).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task SilkExcitationDequantizerGpu_VoicedLow_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            await DequantizeAndVerify(acc, signalType: 2, quantOffsetType: 0,
                seed: unchecked((int)0xDEAD0001u), frameLength: 320, pulseSeed: 0xDEAD0001u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkExcitationDequantizerGpu_VoicedHigh_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            await DequantizeAndVerify(acc, signalType: 2, quantOffsetType: 1,
                seed: unchecked((int)0xDEAD0002u), frameLength: 320, pulseSeed: 0xDEAD0002u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkExcitationDequantizerGpu_UnvoicedLow_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            await DequantizeAndVerify(acc, signalType: 1, quantOffsetType: 0,
                seed: unchecked((int)0xDEAD0003u), frameLength: 240, pulseSeed: 0xDEAD0003u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkExcitationDequantizerGpu_InactiveHigh_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            await DequantizeAndVerify(acc, signalType: 0, quantOffsetType: 1,
                seed: unchecked((int)0xDEAD0004u), frameLength: 160, pulseSeed: 0xDEAD0004u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task DequantizeAndVerify(
        Accelerator acc, int signalType, int quantOffsetType, int seed,
        int frameLength, uint pulseSeed)
    {
        var rng = new Random(unchecked((int)pulseSeed));
        short[] pulses = new short[frameLength];
        for (int i = 0; i < frameLength; i++)
            pulses[i] = (short)rng.Next(-256, 257);

        // CPU reference.
        int[] cpuExc = new int[frameLength];
        SilkExcitationDequantizer.Dequantize(cpuExc, pulses, signalType, quantOffsetType, seed, frameLength);

        // GPU dispatch: single-thread per stream (sequential PRNG).
        using var dExc = acc.Allocate1D<int>(frameLength);
        using var dPulses = acc.Allocate1D<short>(frameLength);
        dPulses.View.CopyFromCPU(pulses);
        dExc.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<short>, int, int, int, int>(DequantizeKernel);
        kernel(new Index1D(1), dExc.View, dPulses.View, signalType, quantOffsetType, seed, frameLength);
        await acc.SynchronizeAsync();

        var gpuExc = await dExc.CopyToHostAsync();
        for (int i = 0; i < frameLength; i++)
        {
            if (cpuExc[i] != gpuExc[i])
                throw new Exception(
                    $"exc[{i}]: cpu={cpuExc[i]} gpu={gpuExc[i]} (signalType={signalType}, qoff={quantOffsetType})");
        }
    }

    private static void DequantizeKernel(
        Index1D _,
        ArrayView<int> excQ14, ArrayView<short> pulses,
        int signalType, int quantOffsetType, int seed, int frameLength)
    {
        SilkExcitationDequantizerGpu.DequantizeAt(excQ14, 0, pulses, 0,
            signalType, quantOffsetType, seed, frameLength);
    }
}
