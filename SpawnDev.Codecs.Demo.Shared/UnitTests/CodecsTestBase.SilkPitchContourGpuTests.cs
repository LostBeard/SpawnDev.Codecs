// Cross-backend test for SilkPitchContourGpu.ComputeLagAt. Verifies the
// GPU pitch contour expansion matches the CPU reference
// SilkPitchDecoder.ComputeLags bit-exactly across all 4 codebook selections
// (NB/WB x 10ms/20ms).

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
    public async Task SilkPitchContourGpu_NB_20ms_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // 8 kHz, 4 subframes -> Stage2, cbSize 11.
            await ContourAndVerify(acc, fsKHz: 8, nbSubfr: 4,
                cb: SilkPitchContourTables.Stage2, cbSize: 11,
                lagIndex: 50, contourIndex: 5);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkPitchContourGpu_NB_10ms_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // 8 kHz, 2 subframes -> Stage2_10Ms, cbSize 3.
            await ContourAndVerify(acc, fsKHz: 8, nbSubfr: 2,
                cb: SilkPitchContourTables.Stage210Ms, cbSize: 3,
                lagIndex: 30, contourIndex: 1);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkPitchContourGpu_WB_20ms_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // 16 kHz, 4 subframes -> Stage3, cbSize 34.
            await ContourAndVerify(acc, fsKHz: 16, nbSubfr: 4,
                cb: SilkPitchContourTables.Stage3, cbSize: 34,
                lagIndex: 100, contourIndex: 15);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkPitchContourGpu_WB_10ms_ClampPath_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Force clamp paths: small lagIndex (clamps low) and large lagIndex (clamps high).
            await ContourAndVerify(acc, fsKHz: 16, nbSubfr: 2,
                cb: SilkPitchContourTables.Stage310Ms, cbSize: 12,
                lagIndex: 0, contourIndex: 0);
            await ContourAndVerify(acc, fsKHz: 16, nbSubfr: 2,
                cb: SilkPitchContourTables.Stage310Ms, cbSize: 12,
                lagIndex: 250, contourIndex: 11);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task ContourAndVerify(
        Accelerator acc, int fsKHz, int nbSubfr, sbyte[] cb, int cbSize,
        short lagIndex, sbyte contourIndex)
    {
        // CPU reference.
        int[] cpuLags = new int[nbSubfr];
        SilkPitchDecoder.ComputeLags(cpuLags, lagIndex, contourIndex, fsKHz, nbSubfr);

        // GPU dispatch: per-subframe parallel.
        using var dCb = acc.Allocate1D<sbyte>(cb.Length);
        using var dLags = acc.Allocate1D<int>(nbSubfr);
        dCb.View.CopyFromCPU(cb);
        dLags.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<sbyte>,
            int, int, int, int>(ContourKernel);
        kernel(new Index1D(nbSubfr), dLags.View, dCb.View, lagIndex, contourIndex, cbSize, fsKHz);
        await acc.SynchronizeAsync();

        var gpuLags = await dLags.CopyToHostAsync();

        for (int i = 0; i < nbSubfr; i++)
        {
            if (cpuLags[i] != gpuLags[i])
                throw new Exception($"pitchLags[{i}]: cpu={cpuLags[i]} gpu={gpuLags[i]} (fsKHz={fsKHz}, nbSubfr={nbSubfr})");
        }
    }

    private static void ContourKernel(
        Index1D index,
        ArrayView<int> pitchLags, ArrayView<sbyte> contourCb,
        int lagIndex, int contourIndex, int cbSize, int fsKHz)
    {
        SilkPitchContourGpu.ComputeLagAt(pitchLags, 0, contourCb, 0,
            lagIndex, contourIndex, cbSize, fsKHz, index.X);
    }
}
