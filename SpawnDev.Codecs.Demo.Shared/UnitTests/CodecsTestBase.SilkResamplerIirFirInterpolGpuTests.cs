// Cross-backend test for SilkResamplerIirFirInterpolGpu.ApplyAt. Verifies
// the GPU 12-phase fractional FIR matches the CPU reference
// SilkResampler.IirFirInterpol bit-exactly across representative
// upsample ratios.

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
    public async Task SilkResamplerIirFirInterpolGpu_ThreeQuarterIncrement_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // 3/4 ratio (after 2x up = 3/8 of input rate per output): index increments
            // by 3/4 * 65536 = 49152.
            int indexIncrement = 49152;
            int numOutputs = 240;
            await ApplyAndVerify(acc, indexIncrement, numOutputs, 0xC001D00Du);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkResamplerIirFirInterpolGpu_NonIntegerIncrement_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Non-integer ratio (e.g. 5/12 step) - exercises the full 12-phase table.
            int indexIncrement = (5 << 16) / 12;
            int numOutputs = 360;
            await ApplyAndVerify(acc, indexIncrement, numOutputs, 0x12FACE12u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkResamplerIirFirInterpolGpu_TightIncrement_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Tight increment (lots of outputs per input) - exercises low-tableIdx rows.
            int indexIncrement = 8192;
            int numOutputs = 600;
            await ApplyAndVerify(acc, indexIncrement, numOutputs, 0xBEEFCAFEu);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task ApplyAndVerify(
        Accelerator acc, int indexIncrement, int numOutputs, uint seed)
    {
        long maxIndexQ16 = (long)numOutputs * indexIncrement;
        int bufLen = (int)(maxIndexQ16 >> 16) + 8 + 1;

        var rng = new Random(unchecked((int)seed));
        short[] buf = new short[bufLen];
        for (int i = 0; i < bufLen; i++)
            buf[i] = (short)rng.Next(short.MinValue, short.MaxValue + 1);

        // CPU reference.
        short[] cpuOut = new short[numOutputs];
        SilkResampler.IirFirInterpol(cpuOut, 0, buf, maxIndexQ16, indexIncrement);

        // GPU dispatch: per-output-sample parallel.
        using var dBuf = acc.Allocate1D<short>(bufLen);
        using var dFrac = acc.Allocate1D<short>(SilkResamplerTables.FracFir12.Length);
        using var dOut = acc.Allocate1D<short>(numOutputs);
        dBuf.View.CopyFromCPU(buf);
        dFrac.View.CopyFromCPU(SilkResamplerTables.FracFir12);
        dOut.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<short>, ArrayView<short>, int>(
            IirFirInterpolKernel);
        kernel(new Index1D(numOutputs), dBuf.View, dFrac.View, dOut.View, indexIncrement);
        await acc.SynchronizeAsync();

        var gpuOut = await dOut.CopyToHostAsync();
        for (int i = 0; i < numOutputs; i++)
        {
            if (cpuOut[i] != gpuOut[i])
                throw new Exception($"sample[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]} (inc={indexIncrement})");
        }
    }

    private static void IirFirInterpolKernel(
        Index1D index,
        ArrayView<short> buf, ArrayView<short> fracFir12, ArrayView<short> output,
        int indexIncrement)
    {
        SilkResamplerIirFirInterpolGpu.ApplyAt(buf, 0, fracFir12, 0, output, 0, indexIncrement, index.X);
    }
}
