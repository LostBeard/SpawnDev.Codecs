// Cross-backend tests for Vp9FrameUncompressedHeaderKernel. The
// GPU-emitted uncompressed header must be byte-for-byte identical to
// what Vp9KeyframeEncoder.BuildUncompressedHeader produces on the CPU
// for every (width, height, baseQIndex, firstPartitionSize) tuple
// the v1 keyframe encoder can hit.
//
// Width/height are restricted to multiples of 16 per the v1
// keyframe encoder contract; baseQIndex is in [1, 255]; first-
// partition size is the byte length of the compressed header (held
// in 16 bits).

using System.Reflection;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] BuildVp9UncompressedHeaderCpu(
        int width, int height, int baseQIndex, int firstPartitionSize)
    {
        // Vp9KeyframeEncoder.BuildUncompressedHeader is private; reach
        // it via reflection so the test pins the GPU output to the
        // canonical CPU emit without exposing internals to the public
        // surface. If this method ever moves to public, the reflection
        // hop can drop.
        var mi = typeof(Vp9KeyframeEncoder).GetMethod(
            "BuildUncompressedHeader",
            BindingFlags.Static | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException(
                "Vp9KeyframeEncoder.BuildUncompressedHeader not found.");
        return (byte[])mi.Invoke(null,
            new object[] { width, height, baseQIndex, firstPartitionSize })!;
    }

    private static async Task<byte[]> BuildVp9UncompressedHeaderGpuAsync(
        Accelerator acc,
        int width, int height, int baseQIndex, int firstPartitionSize)
    {
        using var kernel = new Vp9FrameUncompressedHeaderKernel(acc);
        // 32 bytes is comfortably larger than the v1 header's worst case
        // (~17 bytes for typical parameters; tile-info increment bit +
        // partial-byte rounding can push it up by 1).
        using var dOutBuf = acc.Allocate1D<byte>(32);
        using var dOutLen = acc.Allocate1D<long>(1);
        dOutBuf.View.CopyFromCPU(new byte[32]); // pre-zero
        kernel.Run(dOutBuf.View, dOutLen.View, width, height, baseQIndex, firstPartitionSize);
        await acc.SynchronizeAsync();

        long outLen = (await dOutLen.CopyToHostAsync())[0];
        var bytes = await dOutBuf.CopyToHostAsync();
        var result = new byte[outLen];
        Array.Copy(bytes, result, outLen);
        return result;
    }

    private static async Task AssertVp9UncompressedHeaderGpuMatchesCpuAsync(
        Accelerator acc,
        int width, int height, int baseQIndex, int firstPartitionSize)
    {
        var cpu = BuildVp9UncompressedHeaderCpu(width, height, baseQIndex, firstPartitionSize);
        var gpu = await BuildVp9UncompressedHeaderGpuAsync(
            acc, width, height, baseQIndex, firstPartitionSize);

        Equal(cpu.Length, gpu.Length);
        for (int i = 0; i < cpu.Length; i++)
        {
            if (cpu[i] != gpu[i])
                throw new Exception(
                    $"byte mismatch at offset {i}: cpu=0x{cpu[i]:X2} gpu=0x{gpu[i]:X2} " +
                    $"(width={width}, height={height}, baseQ={baseQIndex}, fps={firstPartitionSize})");
        }
    }

    [TestMethod]
    public async Task Vp9FrameUncompressedHeaderKernel_TypicalKeyframe_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Mirror the v1 encoder default: 64x64 frame, baseQ=30, small fps.
            await AssertVp9UncompressedHeaderGpuMatchesCpuAsync(
                acc, width: 64, height: 64, baseQIndex: 30, firstPartitionSize: 16);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9FrameUncompressedHeaderKernel_BaseQSweep_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Exercise the base_q_idx(8) field with values that span the
            // full byte range (1, 64, 128, 192, 255). 0 is excluded
            // because the v1 encoder rejects it (lossless not supported).
            int[] qIndices = { 1, 64, 128, 192, 255 };
            foreach (var q in qIndices)
            {
                await AssertVp9UncompressedHeaderGpuMatchesCpuAsync(
                    acc, width: 128, height: 64, baseQIndex: q, firstPartitionSize: 24);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9FrameUncompressedHeaderKernel_FrameSizeSweep_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // (width, height) - all multiples of 16. Includes the
            // single-tile widths (<= 4096) where minLog2 stays 0.
            (int w, int h)[] sizes =
            {
                (16, 16),
                (64, 64),
                (320, 240),
                (640, 480),
                (1920, 1088),  // multiple of 16; 1080 isn't.
            };
            foreach (var (w, h) in sizes)
            {
                await AssertVp9UncompressedHeaderGpuMatchesCpuAsync(
                    acc, w, h, baseQIndex: 30, firstPartitionSize: 16);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9FrameUncompressedHeaderKernel_FirstPartitionSizeSweep_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            int[] sizes = { 0, 1, 16, 256, 4096, 32768, 65535 };
            foreach (var fps in sizes)
            {
                await AssertVp9UncompressedHeaderGpuMatchesCpuAsync(
                    acc, width: 64, height: 64, baseQIndex: 30, firstPartitionSize: fps);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
