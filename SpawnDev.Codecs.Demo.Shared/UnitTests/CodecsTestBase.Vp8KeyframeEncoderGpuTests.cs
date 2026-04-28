// Tests for Vp8KeyframeEncoderGpu - end-to-end GPU encoder integration.
// Critical assertion: GPU-encoded keyframe bytes MUST match the
// CPU-encoded keyframe bytes for the same input. This proves the
// pipeline produces a valid VP8 bitstream that decodes correctly
// (CPU output is already verified via Vp8KeyframeWalker round-trip
// tests in this repo).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp8KeyframeEncoderGpu_SingleMbFrame_MatchesCpuEncoder()
    {
        await RunEncoderMatchesCpuTest(width: 16, height: 16, seed: 0x60D6);
    }

    [TestMethod]
    public async Task Vp8KeyframeEncoderGpu_MultiMb_2x2_MatchesCpuEncoder()
    {
        await RunEncoderMatchesCpuTest(width: 32, height: 32, seed: 0x60D7);
    }

    [TestMethod]
    public async Task Vp8KeyframeEncoderGpu_MultiMb_4x4_MatchesCpuEncoder()
    {
        await RunEncoderMatchesCpuTest(width: 64, height: 64, seed: 0x60D8);
    }

    [TestMethod]
    public async Task Vp8KeyframeEncoderGpu_MultiMb_8x8_MatchesCpuEncoder()
    {
        await RunEncoderMatchesCpuTest(width: 128, height: 128, seed: 0x60D9);
    }

    private async Task RunEncoderMatchesCpuTest(int width, int height, int seed)
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var enc = new Vp8KeyframeEncoderGpu(acc);
            const int baseQIndex = 30;

            var rng = new Random(seed);
            var ySrc = new byte[width * height];
            var uSrc = new byte[(width / 2) * (height / 2)];
            var vSrc = new byte[(width / 2) * (height / 2)];
            for (int i = 0; i < ySrc.Length; i++) ySrc[i] = (byte)rng.Next(0, 256);
            for (int i = 0; i < uSrc.Length; i++) uSrc[i] = (byte)rng.Next(0, 256);
            for (int i = 0; i < vSrc.Length; i++) vSrc[i] = (byte)rng.Next(0, 256);

            // CPU reference output.
            byte[] cpuBytes = Vp8KeyframeEncoder.EncodeKeyFrame(
                ySrc, ySrcStride: width,
                uSrc, uvSrcStride: width / 2,
                vSrc,
                width, height, baseQIndex);

            // GPU output.
            byte[] gpuBytes = enc.EncodeKeyFrame(
                ySrc, ySrcStride: width,
                uSrc, uvSrcStride: width / 2,
                vSrc,
                width, height, baseQIndex);

            Equal(cpuBytes.Length, gpuBytes.Length, $"{width}x{height} keyframe byte length");
            int mismatches = 0;
            int firstBad = -1;
            for (int i = 0; i < cpuBytes.Length && i < gpuBytes.Length; i++)
            {
                if (cpuBytes[i] != gpuBytes[i])
                {
                    if (firstBad < 0) firstBad = i;
                    mismatches++;
                }
            }
            Equal(0, mismatches, $"{width}x{height} first byte mismatch i={firstBad}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
