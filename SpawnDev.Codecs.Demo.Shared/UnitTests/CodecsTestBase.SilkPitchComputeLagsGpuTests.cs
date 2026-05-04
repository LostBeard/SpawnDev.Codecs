// Cross-backend tests for SilkPitchComputeLagsGpu - GPU port of
// SilkPitchDecoder.ComputeLags. Compares per-subframe pitch lags
// against the CPU reference.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static (sbyte[] cb, int cbSize) SilkPitchTest_SelectContourCb(int fsKHz, int nbSubfr)
    {
        if (fsKHz == 8)
        {
            if (nbSubfr == 4) return (SilkPitchContourTables.Stage2, 11);
            return (SilkPitchContourTables.Stage210Ms, 3);
        }
        if (nbSubfr == 4) return (SilkPitchContourTables.Stage3, 34);
        return (SilkPitchContourTables.Stage310Ms, 12);
    }

    /// <summary>CPU reference - bit-exact mirror of SilkPitchDecoder.ComputeLags.</summary>
    private static int[] SilkPitchComputeLagsCpu(
        sbyte[] cb, int cbSize, int lagIndex, int contourIndex, int fsKHz, int nbSubfr)
    {
        const int peMinLagMs = 2, peMaxLagMs = 18;
        int minLag = peMinLagMs * fsKHz;
        int maxLag = peMaxLagMs * fsKHz;
        int baseLag = minLag + lagIndex;

        var lags = new int[nbSubfr];
        for (int k = 0; k < nbSubfr; k++)
        {
            int lag = baseLag + cb[k * cbSize + contourIndex];
            if (lag < minLag) lag = minLag;
            else if (lag > maxLag) lag = maxLag;
            lags[k] = lag;
        }
        return lags;
    }

    private static async Task<int[]> SilkPitchComputeLagsGpuAsync(
        Accelerator acc,
        sbyte[] cb, int cbSize, int lagIndex, int contourIndex, int fsKHz, int nbSubfr)
    {
        using var dCb = acc.Allocate1D<sbyte>(cb.Length);
        using var dOut = acc.Allocate1D<int>(nbSubfr);

        dCb.View.CopyFromCPU(cb);

        using var kernel = new SilkPitchComputeLagsGpuTestKernel(acc);
        kernel.Run(dCb.View, cbSize, lagIndex, contourIndex, fsKHz, nbSubfr, dOut.View);
        await acc.SynchronizeAsync();

        var output = await dOut.CopyToHostAsync();
        var slice = new int[nbSubfr];
        Array.Copy(output, slice, nbSubfr);
        return slice;
    }

    private static async Task SilkPitchTest_AssertMatchesCpu(
        Accelerator acc, int fsKHz, int nbSubfr, int lagIndex, int contourIndex)
    {
        var (cb, cbSize) = SilkPitchTest_SelectContourCb(fsKHz, nbSubfr);
        int[] cpu = SilkPitchComputeLagsCpu(cb, cbSize, lagIndex, contourIndex, fsKHz, nbSubfr);
        int[] gpu = await SilkPitchComputeLagsGpuAsync(
            acc, cb, cbSize, lagIndex, contourIndex, fsKHz, nbSubfr);

        for (int k = 0; k < nbSubfr; k++)
            if (gpu[k] != cpu[k])
                throw new Exception(
                    $"pitchLag[{k}] mismatch (fs={fsKHz}, nbSubfr={nbSubfr}, " +
                    $"lag={lagIndex}, contour={contourIndex}): cpu={cpu[k]} gpu={gpu[k]}");
    }

    [TestMethod]
    public async Task SilkPitchComputeLagsGpu_Wb20ms_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // WB (16kHz) 20ms - Stage3 codebook, 34 contours, 4 subframes.
            await SilkPitchTest_AssertMatchesCpu(acc, fsKHz: 16, nbSubfr: 4, lagIndex: 50, contourIndex: 17);
            await SilkPitchTest_AssertMatchesCpu(acc, fsKHz: 16, nbSubfr: 4, lagIndex: 200, contourIndex: 0);
            await SilkPitchTest_AssertMatchesCpu(acc, fsKHz: 16, nbSubfr: 4, lagIndex: 100, contourIndex: 33);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkPitchComputeLagsGpu_Nb20ms_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // NB (8kHz) 20ms - Stage2 codebook, 11 contours, 4 subframes.
            await SilkPitchTest_AssertMatchesCpu(acc, fsKHz: 8, nbSubfr: 4, lagIndex: 30, contourIndex: 5);
            await SilkPitchTest_AssertMatchesCpu(acc, fsKHz: 8, nbSubfr: 4, lagIndex: 100, contourIndex: 0);
            await SilkPitchTest_AssertMatchesCpu(acc, fsKHz: 8, nbSubfr: 4, lagIndex: 60, contourIndex: 10);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkPitchComputeLagsGpu_Mb10ms_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // MB (12kHz) 10ms - Stage310Ms codebook, 12 contours, 2 subframes.
            await SilkPitchTest_AssertMatchesCpu(acc, fsKHz: 12, nbSubfr: 2, lagIndex: 80, contourIndex: 6);
            await SilkPitchTest_AssertMatchesCpu(acc, fsKHz: 12, nbSubfr: 2, lagIndex: 150, contourIndex: 11);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkPitchComputeLagsGpu_ClampingHigh_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Force lag clamping at maxLag = 18 * fsKHz = 288 for WB.
            // Pick a large lagIndex that pushes past the bound.
            await SilkPitchTest_AssertMatchesCpu(acc, fsKHz: 16, nbSubfr: 4, lagIndex: 286, contourIndex: 0);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
