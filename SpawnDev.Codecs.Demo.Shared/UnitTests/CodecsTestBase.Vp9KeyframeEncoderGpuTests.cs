// Cross-backend tests for Vp9KeyframeEncoderGpu - the v3 100% ILGPU
// VP9 v1 keyframe encoder. Verifies the GPU-produced frame bytes
// match Vp9KeyframeEncoder.EncodeKeyFrame byte-for-byte.
//
// V1 GPU encoder caps frame width + height at multiples of 64 (so
// the SB grid is integer; entropy kernel max width = 512).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static async Task AssertVp9KeyframeEncoderGpuMatchesCpuAsync(
        Accelerator acc,
        byte[] yPlane, byte[] uPlane, byte[] vPlane,
        int width, int height, int baseQIndex)
    {
        // CPU oracle - the existing Vp9KeyframeEncoder produces a
        // complete VP9 keyframe (uncompressed + compressed + tile).
        var cpuBytes = Vp9KeyframeEncoder.EncodeKeyFrame(
            yPlane, ySrcStride: width,
            uPlane, uvSrcStride: width / 2,
            vPlane,
            width, height,
            baseQIndex);

        // GPU encoder.
        using var enc = new Vp9KeyframeEncoderGpu(acc);
        var gpuBytes = await enc.EncodeKeyFrameAsync(yPlane, uPlane, vPlane, width, height, baseQIndex);

        if (cpuBytes.Length != gpuBytes.Length)
            throw new Exception(
                $"frame length mismatch: cpu={cpuBytes.Length} gpu={gpuBytes.Length}");
        for (int i = 0; i < cpuBytes.Length; i++)
        {
            if (cpuBytes[i] != gpuBytes[i])
                throw new Exception(
                    $"byte mismatch at offset {i}: cpu=0x{cpuBytes[i]:X2} gpu=0x{gpuBytes[i]:X2}");
        }
    }

    [TestMethod]
    public async Task Vp9KeyframeEncoderGpu_64x64_FlatGray_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            int width = 64, height = 64;
            int yLen = width * height;
            int uvLen = yLen / 4;
            var y = new byte[yLen];
            var u = new byte[uvLen];
            var v = new byte[uvLen];
            for (int i = 0; i < yLen; i++) y[i] = 128;
            for (int i = 0; i < uvLen; i++) { u[i] = 128; v[i] = 128; }

            await AssertVp9KeyframeEncoderGpuMatchesCpuAsync(acc, y, u, v, width, height, baseQIndex: 30);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9KeyframeEncoderGpu_64x64_RandomContent_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            int width = 64, height = 64;
            int yLen = width * height;
            int uvLen = yLen / 4;
            var rng = new Random(unchecked((int)0xF9E2A001u));
            var y = new byte[yLen];
            var u = new byte[uvLen];
            var v = new byte[uvLen];
            rng.NextBytes(y);
            rng.NextBytes(u);
            rng.NextBytes(v);

            await AssertVp9KeyframeEncoderGpuMatchesCpuAsync(acc, y, u, v, width, height, baseQIndex: 30);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9KeyframeEncoderGpu_64x64_BaseQSweep_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            int width = 64, height = 64;
            int yLen = width * height;
            int uvLen = yLen / 4;
            var rng = new Random(unchecked((int)0xF9E2BB02u));
            var y = new byte[yLen];
            var u = new byte[uvLen];
            var v = new byte[uvLen];
            rng.NextBytes(y);
            rng.NextBytes(u);
            rng.NextBytes(v);

            int[] baseQs = { 1, 30, 64, 128, 200 };
            foreach (var q in baseQs)
                await AssertVp9KeyframeEncoderGpuMatchesCpuAsync(acc, y, u, v, width, height, baseQIndex: q);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9KeyframeEncoderGpu_128x128_MatchesCpu()
    {
        // Multi-SB frame (4 SBs total) - exercises the SB row-major
        // walk + edge propagation across SB boundaries.
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            int width = 128, height = 128;
            int yLen = width * height;
            int uvLen = yLen / 4;
            var rng = new Random(unchecked((int)0xF9E2C128u));
            var y = new byte[yLen];
            var u = new byte[uvLen];
            var v = new byte[uvLen];
            rng.NextBytes(y);
            rng.NextBytes(u);
            rng.NextBytes(v);

            await AssertVp9KeyframeEncoderGpuMatchesCpuAsync(acc, y, u, v, width, height, baseQIndex: 30);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
