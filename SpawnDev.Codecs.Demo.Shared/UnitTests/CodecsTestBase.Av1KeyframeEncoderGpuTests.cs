// Cross-backend tests for Av1KeyframeEncoderGpu - the v3 100% ILGPU
// keyframe encoder integration class. Verifies the GPU walker
// (Av1FrameSequentialEncodeKernel + helpers) produces byte-exact tile
// output to the CPU Av1KeyframeEncoder.EncodeSingleTile reference.
//
// Test surface:
//   - Constant gray YUV 64x64 (smallest meaningful frame, 1 SB).
//   - Random YUV 64x64 (exercises every coef cat + entropy ctx update).
//   - Constant gray YUV 64x128 (multi-SB walker, vertical).
//
// Each test:
//   1. Generates a YUV 4:2:0 source.
//   2. Runs the CPU encoder's EncodeSingleTile to get reference bytes.
//   3. Runs Av1KeyframeEncoderGpu.EncodeSingleTileAsync on GPU.
//   4. Compares byte-by-byte.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Av1KeyframeEncoderGpu_ConstGray64x64_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            const int width = 64;
            const int height = 64;
            const int qIdx = 32;

            var (yPlane, uPlane, vPlane) = MakeConstYuv420(width, height, 128, 128, 128);
            await EncodeAndCompare(acc, yPlane, uPlane, vPlane, width, height, qIdx);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1KeyframeEncoderGpu_Random64x64_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            const int width = 64;
            const int height = 64;
            const int qIdx = 32;

            var rng = new Random(unchecked((int)0xA1F11164u));
            var (yPlane, uPlane, vPlane) = MakeRandomYuv420(width, height, rng);
            await EncodeAndCompare(acc, yPlane, uPlane, vPlane, width, height, qIdx);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1KeyframeEncoderGpu_ConstGray64x128_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            const int width = 64;
            const int height = 128;
            const int qIdx = 64;

            var (yPlane, uPlane, vPlane) = MakeConstYuv420(width, height, 100, 120, 140);
            await EncodeAndCompare(acc, yPlane, uPlane, vPlane, width, height, qIdx);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1KeyframeEncoderGpu_FullKeyFrame_Random64x64_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            const int width = 64;
            const int height = 64;
            const int qIdx = 32;

            var rng = new Random(unchecked((int)0xA1FA22EBu));
            var (yPlane, uPlane, vPlane) = MakeRandomYuv420(width, height, rng);

            // CPU: full TD + SH + Frame OBU stream.
            byte[] cpuFull = Av1KeyframeEncoder.EncodeKeyFrame(
                yPlane, width, uPlane, width >> 1, vPlane, width, height, qIdx);

            // GPU: full TD + SH + Frame OBU stream (tile bytes from GPU walker,
            // OBU framing via shared CPU helpers).
            using var gpu = new Av1KeyframeEncoderGpu(acc);
            byte[] gpuFull = await gpu.EncodeKeyFrameAsync(yPlane, uPlane, vPlane, width, height, qIdx);

            if (cpuFull.Length != gpuFull.Length)
                throw new Exception($"keyframe len mismatch: cpu={cpuFull.Length} gpu={gpuFull.Length}");
            for (int i = 0; i < cpuFull.Length; i++)
                if (cpuFull[i] != gpuFull[i])
                    throw new Exception($"keyframe byte {i}: cpu={cpuFull[i]:X2} gpu={gpuFull[i]:X2}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task EncodeAndCompare(
        Accelerator acc,
        byte[] yPlane, byte[] uPlane, byte[] vPlane,
        int width, int height, int qIdx)
    {
        // ---- CPU reference ----
        // The CPU EncodeSingleTile takes spans + strides. yStride = width;
        // uvStride = width / 2 for 4:2:0.
        byte[] cpuTile = Av1KeyframeEncoder.EncodeSingleTile(
            yPlane, width,
            uPlane, width >> 1,
            vPlane,
            width, height, qIdx);

        // ---- GPU encoder ----
        using var gpu = new Av1KeyframeEncoderGpu(acc);
        byte[] gpuTile = await gpu.EncodeSingleTileAsync(yPlane, uPlane, vPlane, width, height, qIdx);

        // ---- Compare ----
        if (cpuTile.Length != gpuTile.Length)
        {
            var cpuFull = string.Join(" ", cpuTile.Select(b => b.ToString("X2")));
            var gpuFull = string.Join(" ", gpuTile.Select(b => b.ToString("X2")));
            throw new Exception(
                $"tile len mismatch: cpu={cpuTile.Length} gpu={gpuTile.Length}\n" +
                $"cpu: {cpuFull}\n" +
                $"gpu: {gpuFull}");
        }
        for (int i = 0; i < cpuTile.Length; i++)
        {
            if (cpuTile[i] != gpuTile[i])
            {
                int show = Math.Min(8, cpuTile.Length - i);
                var cpuHex = string.Join(" ", cpuTile.AsSpan(i, show).ToArray().Select(b => b.ToString("X2")));
                var gpuHex = string.Join(" ", gpuTile.AsSpan(i, show).ToArray().Select(b => b.ToString("X2")));
                throw new Exception($"byte {i}: cpu={cpuTile[i]:X2} gpu={gpuTile[i]:X2}; ctx cpu={cpuHex} gpu={gpuHex}");
            }
        }
    }

    private static (byte[] y, byte[] u, byte[] v) MakeConstYuv420(int w, int h, byte yV, byte uV, byte vV)
    {
        int yLen = w * h;
        int uvLen = yLen / 4;
        var y = new byte[yLen];
        var u = new byte[uvLen];
        var v = new byte[uvLen];
        for (int i = 0; i < yLen; i++) y[i] = yV;
        for (int i = 0; i < uvLen; i++) { u[i] = uV; v[i] = vV; }
        return (y, u, v);
    }

    private static (byte[] y, byte[] u, byte[] v) MakeRandomYuv420(int w, int h, Random rng)
    {
        int yLen = w * h;
        int uvLen = yLen / 4;
        var y = new byte[yLen];
        var u = new byte[uvLen];
        var v = new byte[uvLen];
        rng.NextBytes(y);
        rng.NextBytes(u);
        rng.NextBytes(v);
        return (y, u, v);
    }
}
