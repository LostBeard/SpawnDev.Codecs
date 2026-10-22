// Tests for the three-overload boundary of Vp9KeyframeEncoderGpu / Vp9KeyframeDecoderGpu
// (managed / TypedArray / ArrayView1D) and for the on-accelerator plane padding that replaced the
// host row loops.
//
// Oracles: the CPU Vp9KeyframeEncoder (bytes) and the CPU Vp9KeyframeWalker (planes) on 64-aligned
// pictures; the single-frame encode for the batch path. What each test can fail on:
//  - ArrayView1D_*  : output / planes are written into the CALLER's views (pre-filled with 0xAB, so
//                     a kernel that wrote elsewhere leaves 0xAB and mismatches the CPU bytes).
//  - Batch_*        : the frame-major batch path (device pad kernel per frame, caller-owned outputs and
//                     lengths) must equal N single-frame encodes, aligned and not.
//  - TypedArray_*   : browser only; JS bytes in, JS bytes out, equal to the managed result.
// The unaligned VP9 pad itself shares PlanePadKernel with VP8, whose odd-size tests have an
// independent oracle (Vp8KeyframeEncoderGpu_OddDimensions_*).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp9KeyframeEncoderGpu_ArrayViewOverload_WritesCallerBufferAndMatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new Vp9KeyframeEncoderGpu(acc);
            const int width = 64, height = 64, baseQIndex = 30;
            var rng = new Random(0x71D1);
            var y = RandomPlane(rng, width * height);
            var u = RandomPlane(rng, width * height / 4);
            var v = RandomPlane(rng, width * height / 4);
            byte[] cpu = Vp9KeyframeEncoder.EncodeKeyFrame(y, width, u, width / 2, v, width, height, baseQIndex);

            using var dY = acc.Allocate1D<byte>(y.Length);
            using var dU = acc.Allocate1D<byte>(u.Length);
            using var dV = acc.Allocate1D<byte>(v.Length);
            DeviceTransfer.Upload(dY.View, y);
            DeviceTransfer.Upload(dU.View, u);
            DeviceTransfer.Upload(dV.View, v);
            int cap = Vp9KeyframeEncoderGpu.GetMaxEncodedSize(width, height);
            using var dOut = acc.Allocate1D<byte>(cap);
            var fill = new byte[cap];
            System.Array.Fill(fill, (byte)0xAB);
            DeviceTransfer.Upload(dOut.View, fill);

            int len = await enc.EncodeKeyFrameAsync(dY.View, dU.View, dV.View, width, height, dOut.View, baseQIndex);
            byte[] gpu = await DeviceTransfer.DownloadAsync(dOut.View, len);

            Equal(cpu.Length, len, "VP9 ArrayView1D overload encoded length");
            EqualBytes(cpu, gpu, "VP9 ArrayView1D overload bytes vs CPU reference");

            byte[] managed = await enc.EncodeKeyFrameAsync(y, u, v, width, height, baseQIndex);
            EqualBytes(cpu, managed, "VP9 managed overload bytes vs CPU reference");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9KeyframeEncoderGpu_Batch_ArrayViewOverload_EqualsSingleFrameEncodes()
    {
        await RunVp9BatchMatchesSingles(frameCount: 3, width: 64, height: 64, seed: 0x71D2);
    }

    [TestMethod]
    public async Task Vp9KeyframeEncoderGpu_Batch_Unaligned_EqualsSingleFrameEncodes()
    {
        await RunVp9BatchMatchesSingles(frameCount: 2, width: 70, height: 50, seed: 0x71D3);
    }

    private async Task RunVp9BatchMatchesSingles(int frameCount, int width, int height, int seed)
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new Vp9KeyframeEncoderGpu(acc);
            const int baseQIndex = 30;
            int uvW = (width + 1) / 2, uvH = (height + 1) / 2;
            var rng = new Random(seed);
            var ys = new byte[frameCount][];
            var us = new byte[frameCount][];
            var vs = new byte[frameCount][];
            var singles = new byte[frameCount][];
            for (int f = 0; f < frameCount; f++)
            {
                ys[f] = RandomPlane(rng, width * height);
                us[f] = RandomPlane(rng, uvW * uvH);
                vs[f] = RandomPlane(rng, uvW * uvH);
                singles[f] = await enc.EncodeKeyFrameAsync(ys[f], us[f], vs[f], width, height, baseQIndex);
            }

            using var dY = acc.Allocate1D<byte>((long)frameCount * width * height);
            using var dU = acc.Allocate1D<byte>((long)frameCount * uvW * uvH);
            using var dV = acc.Allocate1D<byte>((long)frameCount * uvW * uvH);
            for (int f = 0; f < frameCount; f++)
            {
                DeviceTransfer.Upload(dY.View.SubView((long)f * width * height, width * height), ys[f]);
                DeviceTransfer.Upload(dU.View.SubView((long)f * uvW * uvH, uvW * uvH), us[f]);
                DeviceTransfer.Upload(dV.View.SubView((long)f * uvW * uvH, uvW * uvH), vs[f]);
            }
            int cap = Vp9KeyframeEncoderGpu.GetMaxEncodedSize(width, height);
            using var dOut = acc.Allocate1D<byte>((long)frameCount * cap);
            using var dLens = acc.Allocate1D<long>(frameCount);

            await enc.EncodeKeyFramesBatchAsync(
                dY.View, dU.View, dV.View, frameCount, width, height, dOut.View, dLens.View, baseQIndex);

            var lens = await dLens.View.CopyToHostAsync();
            for (int f = 0; f < frameCount; f++)
            {
                Equal(singles[f].Length, (int)lens[f], $"VP9 frame {f} batch length");
                byte[] got = await DeviceTransfer.DownloadAsync(dOut.View.SubView((long)f * cap, lens[f]), lens[f]);
                EqualBytes(singles[f], got, $"VP9 frame {f} batch bytes vs single-frame encode");
            }

            var managed = await enc.EncodeKeyFramesBatchAsync(
                System.Array.ConvertAll(ys, p => (ReadOnlyMemory<byte>)p),
                System.Array.ConvertAll(us, p => (ReadOnlyMemory<byte>)p),
                System.Array.ConvertAll(vs, p => (ReadOnlyMemory<byte>)p),
                width, height, baseQIndex);
            for (int f = 0; f < frameCount; f++)
                EqualBytes(singles[f], managed[f], $"VP9 frame {f} managed batch bytes vs single-frame encode");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9KeyframeDecoderGpu_ArrayViewOverload_WritesCallerPlanesAndMatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var dec = new Vp9KeyframeDecoderGpu(acc);
            const int width = 64, height = 64, baseQIndex = 30;
            var rng = new Random(0x71D4);
            var y = RandomPlane(rng, width * height);
            var u = RandomPlane(rng, width * height / 4);
            var v = RandomPlane(rng, width * height / 4);
            byte[] encoded = Vp9KeyframeEncoder.EncodeKeyFrame(y, width, u, width / 2, v, width, height, baseQIndex);
            var (cpuY, cpuU, cpuV, _, _) = Vp9KeyframeDecodeCpu(encoded);

            using var dEnc = acc.Allocate1D<byte>(encoded.Length + 37);
            DeviceTransfer.Upload(dEnc.View, encoded);
            using var dY = acc.Allocate1D<byte>(Vp9KeyframeDecoderGpu.GetLumaPlaneSize(width, height) + 5);
            using var dU = acc.Allocate1D<byte>(Vp9KeyframeDecoderGpu.GetChromaPlaneSize(width, height) + 5);
            using var dV = acc.Allocate1D<byte>(Vp9KeyframeDecoderGpu.GetChromaPlaneSize(width, height) + 5);
            foreach (var b in new[] { dY, dU, dV })
            {
                var fill = new byte[b.Length];
                System.Array.Fill(fill, (byte)0xAB);
                DeviceTransfer.Upload(b.View, fill);
            }

            var header = await dec.DecodeKeyFrameAsync(dEnc.View, encoded.Length, dY.View, dU.View, dV.View);
            Equal(width, header.FrameHeader.FrameWidth, "VP9 header width");
            Equal(height, header.FrameHeader.FrameHeight, "VP9 header height");
            EqualPlanes(cpuY, await DeviceTransfer.DownloadAsync(dY.View, cpuY.Length), "VP9 Y plane vs CPU walker");
            EqualPlanes(cpuU, await DeviceTransfer.DownloadAsync(dU.View, cpuU.Length), "VP9 U plane vs CPU walker");
            EqualPlanes(cpuV, await DeviceTransfer.DownloadAsync(dV.View, cpuV.Length), "VP9 V plane vs CPU walker");

            var managed = await dec.DecodeKeyFrameAsync(encoded);
            EqualPlanes(cpuY, managed.YPlane, "VP9 managed Y plane vs CPU walker");
            EqualPlanes(cpuU, managed.UPlane, "VP9 managed U plane vs CPU walker");
            EqualPlanes(cpuV, managed.VPlane, "VP9 managed V plane vs CPU walker");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9KeyframeGpu_TypedArrayOverloads_MatchManaged()
    {
        if (!OperatingSystem.IsBrowser())
            throw new UnsupportedTestException("TypedArray overloads need a browser accelerator.");
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new Vp9KeyframeEncoderGpu(acc);
            using var dec = new Vp9KeyframeDecoderGpu(acc);
            await RunVp9TypedArrayMatchesManaged(enc, dec);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("browser")]
    private static async Task RunVp9TypedArrayMatchesManaged(Vp9KeyframeEncoderGpu enc, Vp9KeyframeDecoderGpu dec)
    {
        const int width = 64, height = 64, baseQIndex = 30;
        var rng = new Random(0x71D5);
        var y = RandomPlane(rng, width * height);
        var u = RandomPlane(rng, width * height / 4);
        var v = RandomPlane(rng, width * height / 4);
        byte[] managed = await enc.EncodeKeyFrameAsync(y, u, v, width, height, baseQIndex);

        using var jy = new Uint8Array(y.Length); jy.Set(y);
        using var ju = new Uint8Array(u.Length); ju.Set(u);
        using var jv = new Uint8Array(v.Length); jv.Set(v);
        using var encoded = await enc.EncodeKeyFrameAsync(jy, ju, jv, width, height, baseQIndex);
        // Reading back to .NET is the TEST's terminal compare, not the product path.
        EqualBytes(managed, encoded.ReadBytes(), "VP9 TypedArray encode bytes vs managed");

        var managedFrame = await dec.DecodeKeyFrameAsync(managed);
        using var frame = await dec.DecodeKeyFrameAsync(encoded);
        Equal(width, frame.Width, "VP9 TypedArray decode width");
        EqualPlanes(managedFrame.YPlane, frame.YPlane.ReadBytes(), "VP9 TypedArray Y plane vs managed");
        EqualPlanes(managedFrame.UPlane, frame.UPlane.ReadBytes(), "VP9 TypedArray U plane vs managed");
        EqualPlanes(managedFrame.VPlane, frame.VPlane.ReadBytes(), "VP9 TypedArray V plane vs managed");
    }
}
