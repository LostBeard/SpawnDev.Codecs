// Cross-backend test for SilkResamplerDownFirInterpolGpu. Verifies the GPU
// polyphase FIR downsampler matches the CPU reference SilkResampler.DownFirInterpol
// bit-exactly across all 3 FIR orders (18/24/36) using the actual libopus
// coefficient tables.

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
    public async Task SilkResamplerDownFirInterpolGpu_Fir1_Half_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // 1/2 downsample (16 -> 8 kHz). Order-24 symmetric.
            short[] coefs = SilkResamplerTables.Coefs1To2;
            // indexIncrementQ16 doubles the rate (2 input samples per 1 output).
            // libopus computes invRatioQ16 such that index increments yield correct mapping.
            // For 1/2 down: invRatioQ16 ~ (16000 << 14 / 8000) << 2 = 2*65536 = 131072.
            int indexIncrement = 131072;
            int firOrder = 24;
            await TestFirVariant(acc, coefs, firOrder, firFracs: 1, indexIncrement, FirVariant.Fir1);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkResamplerDownFirInterpolGpu_Fir2_Third_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // 1/3 downsample. Order-36 symmetric.
            short[] coefs = SilkResamplerTables.Coefs1To3;
            int indexIncrement = 3 * 65536; // 3x stride per output for 1/3 down
            int firOrder = 36;
            await TestFirVariant(acc, coefs, firOrder, firFracs: 1, indexIncrement, FirVariant.Fir2);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkResamplerDownFirInterpolGpu_Fir0_ThreeQuarters_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // 3/4 downsample (16 -> 12 kHz). Order-18 polyphase (3 fractions).
            short[] coefs = SilkResamplerTables.Coefs3To4;
            // 3/4 ratio: each output advances indexQ16 by (4/3) * 65536 ~= 87381.
            int indexIncrement = (int)((4L << 16) / 3L);
            int firOrder = 18;
            int firFracs = 3;
            await TestFirVariant(acc, coefs, firOrder, firFracs, indexIncrement, FirVariant.Fir0);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private enum FirVariant { Fir0, Fir1, Fir2 }

    private static async Task TestFirVariant(
        Accelerator acc, short[] coefsTable, int firOrder, int firFracs,
        int indexIncrement, FirVariant variant)
    {
        // Layout: [AR2_Q14[2], FIR_Coefs[...]]. Slice off the AR2 head.
        short[] firCoefs = new short[coefsTable.Length - 2];
        Array.Copy(coefsTable, 2, firCoefs, 0, firCoefs.Length);

        // Build a long enough buf so output range is meaningful.
        // For numOutputs N, indexQ16 reaches N * indexIncrement. bufStart can reach
        // N*indexIncrement >> 16 = N*indexIncrement/65536. So buf must be at least that
        // plus firOrder samples wide.
        int numOutputs = 200;
        long maxIndexQ16 = (long)numOutputs * indexIncrement;
        int bufLen = (int)(maxIndexQ16 >> 16) + firOrder + 1;

        var rng = new Random(unchecked((int)0xBADF00Du));
        int[] buf = new int[bufLen];
        for (int i = 0; i < bufLen; i++)
            buf[i] = rng.Next(int.MinValue / 4, int.MaxValue / 4);

        // CPU reference.
        short[] cpuOut = new short[numOutputs];
        SilkResampler.DownFirInterpol(cpuOut, 0, buf, firCoefs, firOrder, firFracs, maxIndexQ16, indexIncrement);

        // GPU dispatch: per-output-sample parallel.
        using var dBuf = acc.Allocate1D<int>(bufLen);
        using var dCoefs = acc.Allocate1D<short>(firCoefs.Length);
        using var dOut = acc.Allocate1D<short>(numOutputs);
        dBuf.View.CopyFromCPU(buf);
        dCoefs.View.CopyFromCPU(firCoefs);
        dOut.MemSetToZero();

        switch (variant)
        {
            case FirVariant.Fir1:
            {
                var k = acc.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<int>, ArrayView<short>, ArrayView<short>, int>(Fir1Kernel);
                k(new Index1D(numOutputs), dBuf.View, dCoefs.View, dOut.View, indexIncrement);
                break;
            }
            case FirVariant.Fir2:
            {
                var k = acc.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<int>, ArrayView<short>, ArrayView<short>, int>(Fir2Kernel);
                k(new Index1D(numOutputs), dBuf.View, dCoefs.View, dOut.View, indexIncrement);
                break;
            }
            case FirVariant.Fir0:
            {
                var k = acc.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<int>, ArrayView<short>, ArrayView<short>, int, int>(Fir0Kernel);
                k(new Index1D(numOutputs), dBuf.View, dCoefs.View, dOut.View, indexIncrement, firFracs);
                break;
            }
        }
        await acc.SynchronizeAsync();

        var gpuOut = await dOut.CopyToHostAsync();
        for (int i = 0; i < numOutputs; i++)
        {
            if (cpuOut[i] != gpuOut[i])
                throw new Exception($"{variant} sample[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]}");
        }
    }

    private static void Fir1Kernel(
        Index1D index,
        ArrayView<int> buf, ArrayView<short> firCoefs, ArrayView<short> output,
        int indexIncrement)
    {
        SilkResamplerDownFirInterpolGpu.ApplyFir1At(buf, 0, firCoefs, 0, output, 0, indexIncrement, index.X);
    }

    private static void Fir2Kernel(
        Index1D index,
        ArrayView<int> buf, ArrayView<short> firCoefs, ArrayView<short> output,
        int indexIncrement)
    {
        SilkResamplerDownFirInterpolGpu.ApplyFir2At(buf, 0, firCoefs, 0, output, 0, indexIncrement, index.X);
    }

    private static void Fir0Kernel(
        Index1D index,
        ArrayView<int> buf, ArrayView<short> firCoefs, ArrayView<short> output,
        int indexIncrement, int firFracs)
    {
        SilkResamplerDownFirInterpolGpu.ApplyFir0At(buf, 0, firCoefs, 0, output, 0, indexIncrement, firFracs, index.X);
    }
}
