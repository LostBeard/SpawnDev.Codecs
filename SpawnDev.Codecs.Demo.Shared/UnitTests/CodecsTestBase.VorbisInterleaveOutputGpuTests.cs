// Cross-backend test for VorbisInterleaveOutputGpu.InterleaveAt. Verifies
// the GPU per-element interleave converts per-channel PCM to the standard
// interleaved layout bit-exactly across mono, stereo, and 5.1 channel
// configurations.

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
    public async Task VorbisInterleaveOutputGpu_Mono_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            await InterleaveAndVerify(acc, channels: 1, numFrames: 1024, seed: 0xC1AC0001u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task VorbisInterleaveOutputGpu_Stereo_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            await InterleaveAndVerify(acc, channels: 2, numFrames: 1024, seed: 0xC1AC0002u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task VorbisInterleaveOutputGpu_5_1Surround_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            await InterleaveAndVerify(acc, channels: 6, numFrames: 1024, seed: 0xC1AC0006u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task VorbisInterleaveOutputGpu_Stereo_LongFrame_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Long Vorbis frame half = 1024 (2048-block).
            await InterleaveAndVerify(acc, channels: 2, numFrames: 2048, seed: 0xC1AC2048u);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task InterleaveAndVerify(
        Accelerator acc, int channels, int numFrames, uint seed)
    {
        var rng = new Random(unchecked((int)seed));

        // Build per-channel input.
        float[] channelMajor = new float[channels * numFrames];
        for (int ch = 0; ch < channels; ch++)
        {
            for (int n = 0; n < numFrames; n++)
            {
                channelMajor[ch * numFrames + n] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
        }

        // CPU reference.
        float[] cpuOut = new float[numFrames * channels];
        for (int n = 0; n < numFrames; n++)
        {
            for (int ch = 0; ch < channels; ch++)
            {
                cpuOut[n * channels + ch] = channelMajor[ch * numFrames + n];
            }
        }

        // GPU dispatch: per-element parallel.
        using var dCm = acc.Allocate1D<float>(channels * numFrames);
        using var dOut = acc.Allocate1D<float>(numFrames * channels);
        dCm.View.CopyFromCPU(channelMajor);
        dOut.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int, int>(InterleaveKernel);
        kernel(new Index1D(numFrames * channels), dCm.View, dOut.View, channels, numFrames);
        await acc.SynchronizeAsync();

        var gpuOut = await dOut.CopyToHostAsync();

        for (int i = 0; i < cpuOut.Length; i++)
        {
            if (cpuOut[i] != gpuOut[i])
                throw new Exception($"output[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]} (channels={channels}, frames={numFrames})");
        }
    }

    private static void InterleaveKernel(
        Index1D index,
        ArrayView<float> channelMajor, ArrayView<float> interleavedOut,
        int channels, int numFrames)
    {
        VorbisInterleaveOutputGpu.InterleaveAt(channelMajor, 0, interleavedOut, 0,
            channels, numFrames, index.X);
    }
}
