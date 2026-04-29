// Cross-backend test for SilkLtpScaleGpu.LookupAt. Verifies the GPU LTP
// scale Q14 lookup matches the libopus silk_LTPScales_table_Q14 values
// bit-exactly across all 3 valid indices plus an out-of-range fallback.

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
    public async Task SilkLtpScaleGpu_AllValidIndices_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Reference values: silk_LTPScales_table_Q14 = { 15565, 12288, 8192 }
            await LookupAndVerify(acc, ltpScaleIndex: 0, expected: 15565);
            await LookupAndVerify(acc, ltpScaleIndex: 1, expected: 12288);
            await LookupAndVerify(acc, ltpScaleIndex: 2, expected: 8192);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkLtpScaleGpu_OutOfRangeFallback_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Out-of-range yields 0 (mirrors the unvoiced fallback).
            await LookupAndVerify(acc, ltpScaleIndex: 3, expected: 0);
            await LookupAndVerify(acc, ltpScaleIndex: -1, expected: 0);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task LookupAndVerify(Accelerator acc, int ltpScaleIndex, int expected)
    {
        using var dOut = acc.Allocate1D<int>(1);
        dOut.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, int>(LtpScaleKernel);
        kernel(new Index1D(1), dOut.View, ltpScaleIndex);
        await acc.SynchronizeAsync();

        int gpuVal = (await dOut.CopyToHostAsync())[0];
        if (gpuVal != expected)
            throw new Exception($"ltpScaleQ14: expected={expected} gpu={gpuVal} (idx={ltpScaleIndex})");
    }

    private static void LtpScaleKernel(
        Index1D _,
        ArrayView<int> ltpScaleQ14Out, int ltpScaleIndex)
    {
        SilkLtpScaleGpu.LookupAt(ltpScaleQ14Out, 0, ltpScaleIndex);
    }
}
