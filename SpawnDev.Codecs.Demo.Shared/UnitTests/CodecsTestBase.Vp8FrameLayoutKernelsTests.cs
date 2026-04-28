// Tests for Vp8FrameLayoutKernels gather + scatter pairs.
// Round-trip: gather -> scatter -> the frame plane is exactly as it was.

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
    public async Task Vp8FrameLayoutKernels_GatherScatterY16_RoundTrip()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var k = new Vp8FrameLayoutKernels(acc);
            const int mbCols = 5, mbRows = 3;
            int width = mbCols * 16, height = mbRows * 16;
            int yStride = width;
            int frameSize = yStride * height;
            int packedSize = mbCols * mbRows * 256;

            var rng = new Random(0xF12);
            var yIn = new byte[frameSize];
            for (int i = 0; i < yIn.Length; i++) yIn[i] = (byte)rng.Next(0, 256);

            using var dY = acc.Allocate1D<byte>(frameSize);
            using var dY16 = acc.Allocate1D<byte>(packedSize);
            using var dYOut = acc.Allocate1D<byte>(frameSize);
            dY.View.CopyFromCPU(yIn);

            k.GatherY16(dY.View, dY16.View, mbCols, mbRows, yStride);
            // Re-scatter back; should reproduce the original plane.
            k.ScatterY16(dY16.View, dYOut.View, mbCols, mbRows, yStride);
            await acc.SynchronizeAsync();

            int mismatches = await GpuTestVerifyCodecs.CountByteMismatches(
                acc, dYOut.View, yIn, yIn.Length);
            Equal(0, mismatches, "Y16 gather->scatter round-trip");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp8FrameLayoutKernels_GatherScatterUv8_RoundTrip()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var k = new Vp8FrameLayoutKernels(acc);
            const int mbCols = 5, mbRows = 3;
            int uvWidth = mbCols * 8, uvHeight = mbRows * 8;
            int uvStride = uvWidth;
            int frameSize = uvStride * uvHeight;
            int packedSize = mbCols * mbRows * 64;

            var rng = new Random(0xF13);
            var uIn = new byte[frameSize];
            for (int i = 0; i < uIn.Length; i++) uIn[i] = (byte)rng.Next(0, 256);

            using var dU = acc.Allocate1D<byte>(frameSize);
            using var dU8 = acc.Allocate1D<byte>(packedSize);
            using var dUOut = acc.Allocate1D<byte>(frameSize);
            dU.View.CopyFromCPU(uIn);

            k.GatherUv8(dU.View, dU8.View, mbCols, mbRows, uvStride);
            k.ScatterUv8(dU8.View, dUOut.View, mbCols, mbRows, uvStride);
            await acc.SynchronizeAsync();

            int mismatches = await GpuTestVerifyCodecs.CountByteMismatches(
                acc, dUOut.View, uIn, uIn.Length);
            Equal(0, mismatches, "UV8 gather->scatter round-trip");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp8FrameLayoutKernels_GatherY16_MatchesCpuLayout()
    {
        // Verify the GPU gather produces the exact byte layout the FDCT
        // kernel would expect: per-MB 256 bytes, row-major within the MB.
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var k = new Vp8FrameLayoutKernels(acc);
            const int mbCols = 4, mbRows = 2;
            int yStride = mbCols * 16;
            int frameSize = yStride * mbRows * 16;
            int packedSize = mbCols * mbRows * 256;

            var rng = new Random(0xF14);
            var yIn = new byte[frameSize];
            for (int i = 0; i < yIn.Length; i++) yIn[i] = (byte)rng.Next(0, 256);

            // CPU reference: explicitly extract MBs.
            var cpuPacked = new byte[packedSize];
            for (int mbRow = 0; mbRow < mbRows; mbRow++)
            {
                for (int mbCol = 0; mbCol < mbCols; mbCol++)
                {
                    int mbIdx = mbRow * mbCols + mbCol;
                    int pBase = mbIdx * 256;
                    int fBase = mbRow * 16 * yStride + mbCol * 16;
                    for (int r = 0; r < 16; r++)
                    {
                        int fRow = fBase + r * yStride;
                        int pRow = pBase + r * 16;
                        for (int c = 0; c < 16; c++) cpuPacked[pRow + c] = yIn[fRow + c];
                    }
                }
            }

            using var dY = acc.Allocate1D<byte>(frameSize);
            using var dY16 = acc.Allocate1D<byte>(packedSize);
            dY.View.CopyFromCPU(yIn);
            k.GatherY16(dY.View, dY16.View, mbCols, mbRows, yStride);
            await acc.SynchronizeAsync();

            int mismatches = await GpuTestVerifyCodecs.CountByteMismatches(
                acc, dY16.View, cpuPacked, cpuPacked.Length);
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
