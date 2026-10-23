// Tests for the three-overload boundary of Vp8KeyframeEncoderGpu (managed / TypedArray /
// ArrayView1D) and for the on-accelerator plane padding that replaced the host row loops.
//
// What each test can fail on:
//  - ArrayView1D_*      : the encoder writes into the CALLER's output buffer (pre-filled with 0xAB, so
//                         an encoder that wrote anywhere else leaves 0xAB and mismatches the CPU bytes).
//  - OddDimensions_*    : a picture whose width/height are odd (chroma = ceil(dim/2), not dim/2) must
//                         encode exactly like the same picture explicitly zero-padded to the
//                         macroblock grid. Dropping the last chroma column/row changes the token bytes.
//  - Batch_*            : the frame-major batch path (device pad kernel per frame, caller-owned outputs
//                         and lengths) must equal N single-frame encodes.
//  - TypedArray_*       : browser only; bulk bytes go JS -> GPU -> JS and the result equals the managed one.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    // VP8 keyframe header: 3-byte frame tag, 3-byte start code, then 14-bit width + 2-bit scale and
    // 14-bit height + 2-bit scale. Bytes 6..9 are the only ones that depend on the signalled picture
    // size rather than on the coded macroblock grid.
    private const int Vp8TagDimsStart = 6;
    private const int Vp8TagDimsEnd = 10;

    private static byte[] RandomPlane(Random rng, int length)
    {
        var plane = new byte[length];
        rng.NextBytes(plane);
        return plane;
    }

    private static byte[] ZeroPad(byte[] src, int w, int h, int destW, int destH)
    {
        var dst = new byte[destW * destH];
        for (int r = 0; r < h; r++)
            Buffer.BlockCopy(src, r * w, dst, r * destW, w);
        return dst;
    }

    [TestMethod]
    public async Task Vp8KeyframeEncoderGpu_ArrayViewOverload_WritesCallerBufferAndMatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new Vp8KeyframeEncoderGpu(acc);
            const int width = 64, height = 48, baseQIndex = 30;
            var rng = new Random(0x71B1);
            var y = RandomPlane(rng, width * height);
            var u = RandomPlane(rng, (width / 2) * (height / 2));
            var v = RandomPlane(rng, (width / 2) * (height / 2));
            byte[] cpu = Vp8KeyframeEncoder.EncodeKeyFrame(y, width, u, width / 2, v, width, height, baseQIndex);

            using var dY = acc.Allocate1D<byte>(y.Length);
            using var dU = acc.Allocate1D<byte>(u.Length);
            using var dV = acc.Allocate1D<byte>(v.Length);
            DeviceTransfer.Upload(dY.View, y);
            DeviceTransfer.Upload(dU.View, u);
            DeviceTransfer.Upload(dV.View, v);
            int cap = Vp8KeyframeEncoderGpu.GetMaxEncodedSize(width, height);
            using var dOut = acc.Allocate1D<byte>(cap);
            var fill = new byte[cap];
            System.Array.Fill(fill, (byte)0xAB);
            DeviceTransfer.Upload(dOut.View, fill);

            int len = await enc.EncodeKeyFrameAsync(
                dY.View, width, dU.View, width / 2, dV.View, width, height, dOut.View, baseQIndex);
            byte[] gpu = await DeviceTransfer.DownloadAsync(dOut.View, len);

            Equal(cpu.Length, len, "ArrayView1D overload encoded length");
            EqualBytes(cpu, gpu, "ArrayView1D overload bytes vs CPU reference");

            // Same bytes through the managed overload (the funnel through the staging buffers).
            byte[] managed = await enc.EncodeKeyFrameAsync(y, width, u, width / 2, v, width, height, baseQIndex);
            EqualBytes(cpu, managed, "managed overload bytes vs CPU reference");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task Vp8KeyframeEncoderGpu_OddDimensions_EqualsExplicitlyZeroPaddedPicture()
    {
        await RunOddDimensionsMatchesPaddedPicture(width: 33, height: 17, seed: 0x71B2);
    }

    [TestMethod]
    public async Task Vp8KeyframeEncoderGpu_OddWidthEvenHeight_EqualsExplicitlyZeroPaddedPicture()
    {
        await RunOddDimensionsMatchesPaddedPicture(width: 47, height: 32, seed: 0x71B3);
    }

    private async Task RunOddDimensionsMatchesPaddedPicture(int width, int height, int seed)
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new Vp8KeyframeEncoderGpu(acc);
            const int baseQIndex = 30;
            int uvW = (width + 1) / 2, uvH = (height + 1) / 2;
            int workW = (width + 15) & ~15, workH = (height + 15) & ~15;
            var rng = new Random(seed);
            var y = RandomPlane(rng, width * height);
            var u = RandomPlane(rng, uvW * uvH);
            var v = RandomPlane(rng, uvW * uvH);

            // Reference: the same picture, zero-padded by the TEST to the macroblock grid, encoded by
            // the CPU reference at the padded size. Only the signalled-size tag bytes may differ.
            var yPad = ZeroPad(y, width, height, workW, workH);
            var uPad = ZeroPad(u, uvW, uvH, workW / 2, workH / 2);
            var vPad = ZeroPad(v, uvW, uvH, workW / 2, workH / 2);
            byte[] cpu = Vp8KeyframeEncoder.EncodeKeyFrame(
                yPad, workW, uPad, workW / 2, vPad, workW, workH, baseQIndex);

            byte[] gpu = await enc.EncodeKeyFrameAsync(y, width, u, uvW, v, width, height, baseQIndex);

            Equal(cpu.Length, gpu.Length, $"{width}x{height} encoded length");
            int mismatches = 0, firstBad = -1;
            for (int i = 0; i < cpu.Length; i++)
            {
                if (i >= Vp8TagDimsStart && i < Vp8TagDimsEnd) continue;
                if (cpu[i] != gpu[i]) { if (firstBad < 0) firstBad = i; mismatches++; }
            }
            Equal(0, mismatches, $"{width}x{height} vs zero-padded {workW}x{workH}: first mismatch at byte {firstBad}");
            // The signalled size must be the real picture size, not the padded one.
            int signalledW = (gpu[6] | (gpu[7] << 8)) & 0x3FFF;
            int signalledH = (gpu[8] | (gpu[9] << 8)) & 0x3FFF;
            Equal(width, signalledW, "frame tag width");
            Equal(height, signalledH, "frame tag height");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task Vp8KeyframeEncoderGpu_Batch_ArrayViewOverload_EqualsSingleFrameEncodes()
    {
        await RunBatchMatchesSingles(frameCount: 3, width: 48, height: 32, seed: 0x71B4);
    }

    [TestMethod]
    public async Task Vp8KeyframeEncoderGpu_Batch_OddDimensions_EqualsSingleFrameEncodes()
    {
        await RunBatchMatchesSingles(frameCount: 2, width: 33, height: 17, seed: 0x71B5);
    }

    private async Task RunBatchMatchesSingles(int frameCount, int width, int height, int seed)
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new Vp8KeyframeEncoderGpu(acc);
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
                singles[f] = await enc.EncodeKeyFrameAsync(ys[f], width, us[f], uvW, vs[f], width, height, baseQIndex);
            }

            // Frame-major, tightly packed device planes; caller-owned outputs + lengths.
            using var dY = acc.Allocate1D<byte>((long)frameCount * width * height);
            using var dU = acc.Allocate1D<byte>((long)frameCount * uvW * uvH);
            using var dV = acc.Allocate1D<byte>((long)frameCount * uvW * uvH);
            for (int f = 0; f < frameCount; f++)
            {
                DeviceTransfer.Upload(dY.View.SubView((long)f * width * height, width * height), ys[f]);
                DeviceTransfer.Upload(dU.View.SubView((long)f * uvW * uvH, uvW * uvH), us[f]);
                DeviceTransfer.Upload(dV.View.SubView((long)f * uvW * uvH, uvW * uvH), vs[f]);
            }
            int cap = Vp8KeyframeEncoderGpu.GetMaxEncodedSize(width, height);
            using var dOut = acc.Allocate1D<byte>((long)frameCount * cap);
            using var dLens = acc.Allocate1D<int>(frameCount);

            await enc.EncodeKeyFramesBatchAsync(
                dY.View, dU.View, dV.View, frameCount, width, height, dOut.View, dLens.View, baseQIndex);

            var lens = await dLens.View.CopyToHostAsync();
            for (int f = 0; f < frameCount; f++)
            {
                Equal(singles[f].Length, lens[f], $"frame {f} batch length");
                byte[] got = await DeviceTransfer.DownloadAsync(dOut.View.SubView((long)f * cap, lens[f]), lens[f]);
                EqualBytes(singles[f], got, $"frame {f} batch bytes vs single-frame encode");
            }

            // The managed batch overload funnels into the same implementation.
            var managed = await enc.EncodeKeyFramesBatchAsync(
                System.Array.ConvertAll(ys, p => (ReadOnlyMemory<byte>)p),
                System.Array.ConvertAll(us, p => (ReadOnlyMemory<byte>)p),
                System.Array.ConvertAll(vs, p => (ReadOnlyMemory<byte>)p),
                width, height, baseQIndex);
            for (int f = 0; f < frameCount; f++)
                EqualBytes(singles[f], managed[f], $"frame {f} managed batch bytes vs single-frame encode");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task Vp8KeyframeEncoderGpu_TypedArrayOverload_MatchesManaged()
    {
        if (!OperatingSystem.IsBrowser())
            throw new UnsupportedTestException("TypedArray overloads need a browser accelerator.");
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new Vp8KeyframeEncoderGpu(acc);
            await RunTypedArrayMatchesManaged(acc, enc);
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("browser")]
    private static async Task RunTypedArrayMatchesManaged(Accelerator acc, Vp8KeyframeEncoderGpu enc)
    {
        const int width = 33, height = 17, baseQIndex = 30;
        int uvW = (width + 1) / 2, uvH = (height + 1) / 2;
        var rng = new Random(0x71B6);
        var y = RandomPlane(rng, width * height);
        var u = RandomPlane(rng, uvW * uvH);
        var v = RandomPlane(rng, uvW * uvH);
        byte[] managed = await enc.EncodeKeyFrameAsync(y, width, u, uvW, v, width, height, baseQIndex);

        using var jy = new Uint8Array(y.Length); jy.Set(y);
        using var ju = new Uint8Array(u.Length); ju.Set(u);
        using var jv = new Uint8Array(v.Length); jv.Set(v);
        using var result = await enc.EncodeKeyFrameAsync(jy, width, ju, uvW, jv, width, height, baseQIndex);
        // Reading the result back to .NET is the TEST's terminal compare, not the product path.
        EqualBytes(managed, result.ReadBytes(), "TypedArray overload bytes vs managed overload");
    }
}
