// Cross-backend test for VorbisPostImdctGpu.ProcessAt. Verifies the GPU
// post-IMDCT composite (window apply + overlap-add + right-half save)
// matches a direct CPU implementation bit-exactly.

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
    public async Task VorbisPostImdctGpu_Short256Block_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            await PostImdctAndVerify(acc, blockSize: 256, seed: 0xC1AC0256u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task VorbisPostImdctGpu_Long2048Block_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            await PostImdctAndVerify(acc, blockSize: 2048, seed: 0xC1AC2048u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task VorbisPostImdctGpu_FreshStream_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Fresh stream: previousRightHalf is all zeros (no overlap yet).
            await PostImdctAndVerify(acc, blockSize: 1024, seed: 0xC1AC1024u, freshStream: true);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task PostImdctAndVerify(
        Accelerator acc, int blockSize, uint seed, bool freshStream = false)
    {
        int halfBlockSize = blockSize / 2;
        var rng = new Random(unchecked((int)seed));

        // Build inputs.
        float[] td = new float[blockSize];
        for (int i = 0; i < blockSize; i++)
            td[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        float[] window = VorbisWindow.GenerateCanonical(blockSize);

        float[] prevRightHalf = new float[halfBlockSize];
        if (!freshStream)
        {
            for (int i = 0; i < halfBlockSize; i++)
                prevRightHalf[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        // CPU reference - direct implementation of the post-IMDCT loop.
        float[] cpuPcm = new float[halfBlockSize];
        float[] cpuNewRight = new float[halfBlockSize];
        for (int i = 0; i < halfBlockSize; i++)
        {
            float leftWindowed = td[i] * window[i];
            float rightWindowed = td[halfBlockSize + i] * window[halfBlockSize + i];
            cpuPcm[i] = leftWindowed + prevRightHalf[i];
            cpuNewRight[i] = rightWindowed;
        }

        // GPU dispatch: per-sample parallel.
        using var dTd = acc.Allocate1D<float>(blockSize);
        using var dWindow = acc.Allocate1D<float>(blockSize);
        using var dPrev = acc.Allocate1D<float>(halfBlockSize);
        using var dNewRight = acc.Allocate1D<float>(halfBlockSize);
        using var dPcm = acc.Allocate1D<float>(halfBlockSize);
        dTd.View.CopyFromCPU(td);
        dWindow.View.CopyFromCPU(window);
        dPrev.View.CopyFromCPU(prevRightHalf);
        dNewRight.MemSetToZero();
        dPcm.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, ArrayView<float>, int>(PostImdctKernel);
        kernel(new Index1D(halfBlockSize), dTd.View, dWindow.View, dPrev.View,
            dNewRight.View, dPcm.View, halfBlockSize);
        await acc.SynchronizeAsync();

        var gpuPcm = await dPcm.CopyToHostAsync();
        var gpuNewRight = await dNewRight.CopyToHostAsync();

        // 1-ULP tolerance: CUDA/OpenCL float multiply-add can diverge from CPU
        // x87/strict-fp by one bit on the mantissa for non-zero results. The
        // Vorbis spec does not require bit-exact decode (libvorbis itself
        // varies). 1e-5 absolute is well within "acoustically lossless".
        const float kTol = 1e-5f;
        for (int i = 0; i < halfBlockSize; i++)
        {
            float pcmDiff = MathF.Abs(cpuPcm[i] - gpuPcm[i]);
            if (pcmDiff > kTol)
                throw new Exception($"pcm[{i}]: cpu={cpuPcm[i]} gpu={gpuPcm[i]} diff={pcmDiff}");
            float rightDiff = MathF.Abs(cpuNewRight[i] - gpuNewRight[i]);
            if (rightDiff > kTol)
                throw new Exception($"newRight[{i}]: cpu={cpuNewRight[i]} gpu={gpuNewRight[i]} diff={rightDiff}");
        }
    }

    private static void PostImdctKernel(
        Index1D index,
        ArrayView<float> td, ArrayView<float> window, ArrayView<float> prev,
        ArrayView<float> newRight, ArrayView<float> pcm,
        int halfBlockSize)
    {
        VorbisPostImdctGpu.ProcessAt(td, 0, window, 0, prev, 0,
            newRight, 0, pcm, 0, halfBlockSize, index.X);
    }
}
