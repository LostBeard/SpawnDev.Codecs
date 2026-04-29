// Cross-backend test for VorbisOverlapAddGpu.AddAt. Verifies the GPU
// post-IMDCT overlap-add matches the CPU reference VorbisWindow.OverlapAdd
// bit-exactly across representative half-block sizes.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task VorbisOverlapAddGpu_Short128_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Short Vorbis block half = 128.
            await OverlapAndVerify(acc, halfBlockSize: 128, seed: 0xBADC0DEDu);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task VorbisOverlapAddGpu_Long512_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Long Vorbis block half = 512.
            await OverlapAndVerify(acc, halfBlockSize: 512, seed: 0xC0FFEE00u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task VorbisOverlapAddGpu_Long1024_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Long Vorbis block half = 1024 (max for 2048-sample block).
            await OverlapAndVerify(acc, halfBlockSize: 1024, seed: 0xFEED1024u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task OverlapAndVerify(Accelerator acc, int halfBlockSize, uint seed)
    {
        var rng = new Random(unchecked((int)seed));
        float[] prev = new float[halfBlockSize];
        float[] cur = new float[halfBlockSize];
        for (int i = 0; i < halfBlockSize; i++)
        {
            prev[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            cur[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        // CPU reference.
        float[] cpuOut = new float[halfBlockSize];
        VorbisWindow.OverlapAdd(prev, cur, cpuOut);

        // GPU dispatch: per-sample parallel.
        using var dPrev = acc.Allocate1D<float>(halfBlockSize);
        using var dCur = acc.Allocate1D<float>(halfBlockSize);
        using var dOut = acc.Allocate1D<float>(halfBlockSize);
        dPrev.View.CopyFromCPU(prev);
        dCur.View.CopyFromCPU(cur);
        dOut.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(OverlapKernel);
        kernel(new Index1D(halfBlockSize), dPrev.View, dCur.View, dOut.View);
        await acc.SynchronizeAsync();

        var gpuOut = await dOut.CopyToHostAsync();

        for (int i = 0; i < halfBlockSize; i++)
        {
            if (cpuOut[i] != gpuOut[i])
                throw new Exception($"output[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]} (half={halfBlockSize})");
        }
    }

    private static void OverlapKernel(
        Index1D index,
        ArrayView<float> prev, ArrayView<float> cur, ArrayView<float> output)
    {
        VorbisOverlapAddGpu.AddAt(prev, 0, cur, 0, output, 0, index.X);
    }
}
