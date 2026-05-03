// Cross-backend test for SilkLpcAnalysisFilterGpu.ApplyAt. Verifies
// the GPU per-sample LPC analysis filter matches the CPU reference
// SilkLpcAnalysisFilter.Apply bit-exactly across representative LPC
// orders (10 NB, 16 WB) on random Q12 inputs.

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
    public async Task SilkLpcAnalysisFilterGpu_Order10_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Order 10 NB SILK filter. Coefficient pattern is a small set of
            // signed Q12 values, exercise full-scale int16 input.
            short[] bQ12 = { 600, -300, 100, -50, 25, -12, 6, -3, 1, -1 };
            var rng = new Random(unchecked((int)0xDEADBEEFu));
            short[] inSignal = new short[320];
            for (int i = 0; i < inSignal.Length; i++)
                inSignal[i] = (short)rng.Next(-32768, 32768);
            await ApplyAndVerify(acc, inSignal, bQ12);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkLpcAnalysisFilterGpu_Order16_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Order 16 WB SILK filter, larger frame.
            short[] bQ12 = { 800, -400, 200, -100, 50, -25, 12, -6,
                             3, -1, 1, -1, 1, -1, 1, -1 };
            var rng = new Random(unchecked((int)0xCAFEBABEu));
            short[] inSignal = new short[640];
            for (int i = 0; i < inSignal.Length; i++)
                inSignal[i] = (short)rng.Next(-32768, 32768);
            await ApplyAndVerify(acc, inSignal, bQ12);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkLpcAnalysisFilterGpu_FullScaleStress_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Stress the SAT16 saturation path with full-scale coefficients +
            // alternating-sign input to maximize accumulator magnitude.
            short[] bQ12 = { 4095, -4095, 4095, -4095, 4095, -4095, 4095, -4095, 4095, -4095 };
            short[] inSignal = new short[256];
            for (int i = 0; i < inSignal.Length; i++)
                inSignal[i] = (short)((i & 1) == 0 ? short.MaxValue : short.MinValue);
            await ApplyAndVerify(acc, inSignal, bQ12);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task ApplyAndVerify(Accelerator acc, short[] inSignal, short[] bQ12)
    {
        int len = inSignal.Length;
        int d = bQ12.Length;

        // CPU reference.
        short[] cpuOut = new short[len];
        SilkLpcAnalysisFilter.Apply(cpuOut, inSignal, bQ12, len, d);

        // GPU dispatch: one thread per output sample in [d, len). Threads in
        // [0, d) leave outSignal pre-zeroed (we initialize the buffer to zero).
        using var dIn = acc.Allocate1D<short>(len);
        using var dB = acc.Allocate1D<short>(d);
        using var dOut = acc.Allocate1D<short>(len);
        dIn.View.CopyFromCPU(inSignal);
        dB.View.CopyFromCPU(bQ12);
        dOut.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<short>, ArrayView<short>, int>(
            ApplyKernel);
        kernel(new Index1D(len - d), dIn.View, dB.View, dOut.View, d);
        await acc.SynchronizeAsync();

        var gpuOut = await dOut.CopyToHostAsync();
        for (int i = 0; i < len; i++)
        {
            if (cpuOut[i] != gpuOut[i])
                throw new Exception($"sample[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]} (len={len}, d={d})");
        }
    }

    private static void ApplyKernel(
        Index1D index,
        ArrayView<short> inSignal, ArrayView<short> bQ12, ArrayView<short> outSignal,
        int d)
    {
        int ix = index.X + d;
        SilkLpcAnalysisFilterGpu.ApplyAt(inSignal, 0, bQ12, 0, outSignal, 0, d, ix);
    }
}
