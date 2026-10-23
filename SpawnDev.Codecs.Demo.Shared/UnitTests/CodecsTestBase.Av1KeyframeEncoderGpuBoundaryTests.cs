// Tests for the three-overload boundary of Av1KeyframeEncoderGpu (managed / TypedArray / ArrayView1D) and
// for the on-accelerator OBU framing (Av1ObuAssembleKernel: header blob from the host, leb128 + concat on the
// accelerator).
//
// Oracle: the CPU Av1KeyframeEncoder.EncodeKeyFrame temporal unit (TD + SH + Frame OBU), byte for byte, on
// 64-aligned pictures (the encoder's current size limit); the single-frame encode for the batch path.
// What each test can fail on:
//  - ArrayView1D_*  : the temporal unit is written into the CALLER's view (pre-filled with 0xAB, so a kernel
//                     that wrote elsewhere or stopped short leaves 0xAB and mismatches the CPU bytes).
//  - Batch_*        : frame-major batch (N frames in one dispatch, caller-owned outputs and lengths) must equal
//                     N single-frame encodes.
//  - Rejects_*      : unsupported sizes / quantizers throw before any dispatch.
//  - TypedArray_*   : browser only; JS bytes in, JS bytes out, equal to the managed result.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Timeout = 600000)]
    public async Task Av1KeyframeEncoderGpu_ArrayViewOverload_WritesCallerBufferAndMatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new Av1KeyframeEncoderGpu(acc);
            const int width = 64, height = 64, baseQIndex = 32;
            var rng = new Random(0xA1B1);
            var y = RandomPlane(rng, width * height);
            var u = RandomPlane(rng, width * height / 4);
            var v = RandomPlane(rng, width * height / 4);
            byte[] cpu = Av1KeyframeEncoder.EncodeKeyFrame(y, width, u, width / 2, v, width, height, baseQIndex);

            using var dY = acc.Allocate1D<byte>(y.Length);
            using var dU = acc.Allocate1D<byte>(u.Length);
            using var dV = acc.Allocate1D<byte>(v.Length);
            DeviceTransfer.Upload(dY.View, y);
            DeviceTransfer.Upload(dU.View, u);
            DeviceTransfer.Upload(dV.View, v);
            int cap = Av1KeyframeEncoderGpu.GetMaxEncodedSize(width, height);
            using var dOut = acc.Allocate1D<byte>(cap);
            var fill = new byte[cap];
            System.Array.Fill(fill, (byte)0xAB);
            DeviceTransfer.Upload(dOut.View, fill);

            int len = await enc.EncodeKeyFrameAsync(dY.View, dU.View, dV.View, width, height, dOut.View, baseQIndex);
            byte[] gpu = await DeviceTransfer.DownloadAsync(dOut.View, len);

            Equal(cpu.Length, len, "AV1 ArrayView1D overload encoded length");
            EqualBytes(cpu, gpu, "AV1 ArrayView1D overload bytes vs CPU reference");

            byte[] managed = await enc.EncodeKeyFrameAsync(y, u, v, width, height, baseQIndex);
            EqualBytes(cpu, managed, "AV1 managed overload bytes vs CPU reference");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod(Timeout = 600000)]
    public async Task Av1KeyframeEncoderGpu_Batch_ArrayViewOverload_EqualsSingleFrameEncodes()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new Av1KeyframeEncoderGpu(acc);
            const int frameCount = 3, width = 64, height = 128, baseQIndex = 40;
            int uvLen = width * height / 4;
            var rng = new Random(0xA1B2);
            var ys = new byte[frameCount][];
            var us = new byte[frameCount][];
            var vs = new byte[frameCount][];
            var cpu = new byte[frameCount][];
            for (int f = 0; f < frameCount; f++)
            {
                ys[f] = RandomPlane(rng, width * height);
                us[f] = RandomPlane(rng, uvLen);
                vs[f] = RandomPlane(rng, uvLen);
                cpu[f] = Av1KeyframeEncoder.EncodeKeyFrame(ys[f], width, us[f], width / 2, vs[f], width, height, baseQIndex);
            }

            using var dY = acc.Allocate1D<byte>((long)frameCount * width * height);
            using var dU = acc.Allocate1D<byte>((long)frameCount * uvLen);
            using var dV = acc.Allocate1D<byte>((long)frameCount * uvLen);
            for (int f = 0; f < frameCount; f++)
            {
                DeviceTransfer.Upload(dY.View.SubView((long)f * width * height, width * height), ys[f]);
                DeviceTransfer.Upload(dU.View.SubView((long)f * uvLen, uvLen), us[f]);
                DeviceTransfer.Upload(dV.View.SubView((long)f * uvLen, uvLen), vs[f]);
            }
            int cap = Av1KeyframeEncoderGpu.GetMaxEncodedSize(width, height);
            using var dOut = acc.Allocate1D<byte>((long)frameCount * cap);
            using var dLens = acc.Allocate1D<int>(frameCount);

            await enc.EncodeKeyFramesBatchAsync(
                dY.View, dU.View, dV.View, frameCount, width, height, dOut.View, dLens.View, baseQIndex);

            var lens = await dLens.View.CopyToHostAsync();
            for (int f = 0; f < frameCount; f++)
            {
                Equal(cpu[f].Length, lens[f], $"AV1 frame {f} batch length");
                byte[] got = await DeviceTransfer.DownloadAsync(dOut.View.SubView((long)f * cap, lens[f]), lens[f]);
                EqualBytes(cpu[f], got, $"AV1 frame {f} batch bytes vs CPU reference");
            }

            var managed = await enc.EncodeKeyFramesBatchAsync(
                System.Array.ConvertAll(ys, p => (ReadOnlyMemory<byte>)p),
                System.Array.ConvertAll(us, p => (ReadOnlyMemory<byte>)p),
                System.Array.ConvertAll(vs, p => (ReadOnlyMemory<byte>)p),
                width, height, baseQIndex);
            for (int f = 0; f < frameCount; f++)
                EqualBytes(cpu[f], managed[f], $"AV1 frame {f} managed batch bytes vs CPU reference");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod(Timeout = 600000)]
    public async Task Av1KeyframeEncoderGpu_Rejects_UnalignedSizeAndBadQuantizer()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new Av1KeyframeEncoderGpu(acc);
            var plane = new byte[64 * 64];
            var chroma = new byte[64 * 64 / 4];
            bool threwSize = false, threwQ = false;
            try { await enc.EncodeKeyFrameAsync(plane, chroma, chroma, 70, 64, 32); }
            catch (NotSupportedException) { threwSize = true; }
            try { await enc.EncodeKeyFrameAsync(plane, chroma, chroma, 64, 64, 0); }
            catch (ArgumentOutOfRangeException) { threwQ = true; }
            True(threwSize, "a width that is not a multiple of 64 must throw NotSupportedException");
            True(threwQ, "baseQIndex 0 must throw ArgumentOutOfRangeException");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1KeyframeEncoderGpu_TypedArrayOverloads_MatchManaged()
    {
        if (!OperatingSystem.IsBrowser())
            throw new UnsupportedTestException("TypedArray overloads need a browser accelerator.");
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new Av1KeyframeEncoderGpu(acc);
            await RunAv1TypedArrayMatchesManaged(enc);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("browser")]
    private static async Task RunAv1TypedArrayMatchesManaged(Av1KeyframeEncoderGpu enc)
    {
        const int width = 64, height = 64, baseQIndex = 32;
        var rng = new Random(0xA1B3);
        var y = RandomPlane(rng, width * height);
        var u = RandomPlane(rng, width * height / 4);
        var v = RandomPlane(rng, width * height / 4);
        byte[] managed = await enc.EncodeKeyFrameAsync(y, u, v, width, height, baseQIndex);

        using var jy = new Uint8Array(y.Length); jy.Set(y);
        using var ju = new Uint8Array(u.Length); ju.Set(u);
        using var jv = new Uint8Array(v.Length); jv.Set(v);
        using var encoded = await enc.EncodeKeyFrameAsync(jy, ju, jv, width, height, baseQIndex);
        // Reading back to .NET is the TEST's terminal compare, not the product path.
        EqualBytes(managed, encoded.ReadBytes(), "AV1 TypedArray encode bytes vs managed");
    }
}
