// Cross-backend test for SilkStereoMsToLrGpu.{ApplySideAt, ApplyMixAt}.
// Verifies the GPU side reconstruction + M/S -> L/R conversion match the
// CPU reference SilkStereoMsToLr.Apply bit-exactly across NB/MB/WB rates.

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
    public async Task SilkStereoMsToLrGpu_NarrowBand_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            await StereoVerify(acc, fsKHz: 8, frameLength: 160,
                predPrev: new[] { 1024, -512 }, predCur: new[] { 2048, 256 }, seed: 0xC0DEF00Du);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkStereoMsToLrGpu_MediumBand_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            await StereoVerify(acc, fsKHz: 12, frameLength: 240,
                predPrev: new[] { -2000, 1500 }, predCur: new[] { 3000, -1000 }, seed: 0xCABBA6E0u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkStereoMsToLrGpu_WideBand_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            await StereoVerify(acc, fsKHz: 16, frameLength: 320,
                predPrev: new[] { 0, 0 }, predCur: new[] { 4096, -2048 }, seed: 0x12345AAAu);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task StereoVerify(
        Accelerator acc, int fsKHz, int frameLength,
        int[] predPrev, int[] predCur, uint seed)
    {
        var rng = new Random(unchecked((int)seed));

        // Random mid + side input. x1/x2 each have frameLength + 2 entries
        // (2 prefix slots seeded by the state, then frameLength samples).
        short[] x1 = new short[frameLength + 2];
        short[] x2 = new short[frameLength + 2];
        for (int i = 0; i < frameLength + 2; i++)
        {
            x1[i] = (short)rng.Next(short.MinValue, short.MaxValue + 1);
            x2[i] = (short)rng.Next(short.MinValue, short.MaxValue + 1);
        }
        // Stereo state with non-zero history.
        var state = new SilkStereoState();
        state.SMid[0] = (short)rng.Next(short.MinValue, short.MaxValue + 1);
        state.SMid[1] = (short)rng.Next(short.MinValue, short.MaxValue + 1);
        state.SSide[0] = (short)rng.Next(short.MinValue, short.MaxValue + 1);
        state.SSide[1] = (short)rng.Next(short.MinValue, short.MaxValue + 1);
        state.PredPrevQ13[0] = predPrev[0];
        state.PredPrevQ13[1] = predPrev[1];

        // CPU reference (mutates x1/x2 + state).
        short[] cpuX1 = (short[])x1.Clone();
        short[] cpuX2 = (short[])x2.Clone();
        var cpuState = new SilkStereoState();
        cpuState.SMid[0] = state.SMid[0];
        cpuState.SMid[1] = state.SMid[1];
        cpuState.SSide[0] = state.SSide[0];
        cpuState.SSide[1] = state.SSide[1];
        cpuState.PredPrevQ13[0] = state.PredPrevQ13[0];
        cpuState.PredPrevQ13[1] = state.PredPrevQ13[1];
        SilkStereoMsToLr.Apply(cpuState, cpuX1, cpuX2, predCur, fsKHz, frameLength);

        // GPU prep: mirror the CPU's prefix + trailing state housekeeping.
        short[] gpuX1 = (short[])x1.Clone();
        short[] gpuX2 = (short[])x2.Clone();
        gpuX1[0] = state.SMid[0]; gpuX1[1] = state.SMid[1];
        gpuX2[0] = state.SSide[0]; gpuX2[1] = state.SSide[1];

        // Compute the predictor delta + interpolation count.
        int interpLen = SilkStereoMsToLr.StereoInterpLenMs * fsKHz;
        int denomQ16 = (1 << 16) / interpLen;
        int delta0Q13 = ComputeDeltaQ13(predCur[0] - state.PredPrevQ13[0], denomQ16);
        int delta1Q13 = ComputeDeltaQ13(predCur[1] - state.PredPrevQ13[1], denomQ16);

        using var dX1 = acc.Allocate1D<short>(gpuX1.Length);
        using var dX2 = acc.Allocate1D<short>(gpuX2.Length);
        dX1.View.CopyFromCPU(gpuX1);
        dX2.View.CopyFromCPU(gpuX2);

        // Side reconstruction: per-sample parallel.
        var sideKernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<short>,
            int, int, int, int, int, int, int>(SideKernel);
        sideKernel(new Index1D(frameLength), dX1.View, dX2.View,
            state.PredPrevQ13[0], delta0Q13, predCur[0],
            state.PredPrevQ13[1], delta1Q13, predCur[1],
            interpLen);

        // Mix M/S -> L/R: per-sample parallel (after side reconstruction).
        var mixKernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<short>>(MixKernel);
        mixKernel(new Index1D(frameLength), dX1.View, dX2.View);
        await acc.SynchronizeAsync();

        var gotX1 = await dX1.CopyToHostAsync();
        var gotX2 = await dX2.CopyToHostAsync();

        for (int i = 0; i < frameLength + 2; i++)
        {
            if (cpuX1[i] != gotX1[i])
                throw new Exception($"x1[{i}]: cpu={cpuX1[i]} gpu={gotX1[i]} (fsKHz={fsKHz})");
            if (cpuX2[i] != gotX2[i])
                throw new Exception($"x2[{i}]: cpu={cpuX2[i]} gpu={gotX2[i]} (fsKHz={fsKHz})");
        }
    }

    private static int ComputeDeltaQ13(int diff, int denomQ16)
    {
        // silk_RSHIFT_ROUND(silk_SMULBB(diff, denomQ16), 16)
        int prod = (short)diff * (short)denomQ16;
        return (prod + (1 << 15)) >> 16;
    }

    private static void SideKernel(
        Index1D index, ArrayView<short> x1, ArrayView<short> x2,
        int predPrev0, int delta0, int pred0Final,
        int predPrev1, int delta1, int pred1Final,
        int interpLen)
    {
        SilkStereoMsToLrGpu.ApplySideAt(x1, 0, x2, 0,
            predPrev0, delta0, pred0Final,
            predPrev1, delta1, pred1Final,
            interpLen, index.X);
    }

    private static void MixKernel(
        Index1D index, ArrayView<short> x1, ArrayView<short> x2)
    {
        SilkStereoMsToLrGpu.ApplyMixAt(x1, 0, x2, 0, index.X);
    }
}
