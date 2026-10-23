// Tests for the three-overload boundary of Vp8KeyframeDecoderGpu and for decoding pictures whose
// size is not a multiple of 16 (the encoder has always emitted those; the decoder used to throw).
//
// Oracle: the CPU Vp8KeyframeWalker decoding the SAME bytes, cropped to the signalled picture size
// with chroma = ceil(dim/2). What each test can fail on:
//  - ArrayView1D_*   : the decoder writes into the CALLER's plane views (pre-filled with 0xAB, so a
//                      decoder that wrote elsewhere leaves 0xAB and mismatches the CPU planes).
//  - Unaligned_*     : the on-accelerator crop of the macroblock grid. A wrong stride or an off-by-one
//                      chroma size mismatches the cropped CPU planes.
//  - TypedArray_*    : browser only; JS bytes in, JS planes out, equal to the managed result.

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
    private static (byte[] Y, byte[] U, byte[] V, int W, int H) CpuDecodeVp8Cropped(byte[] encoded)
    {
        var tag = Vp8FrameTagParser.Parse(encoded.AsSpan());
        int width = tag.Width!.Value, height = tag.Height!.Value;
        const int firstPartOffset = 10;
        var firstPart = encoded.AsSpan(firstPartOffset, tag.FirstPartitionSize).ToArray();
        var modeReader = new Vp8BoolDecoder(firstPart);
        var hdr = Vp8FrameHeaderParser.ParseKeyFrameHeader(modeReader);
        int tokenOffset = firstPartOffset + tag.FirstPartitionSize;
        var tokenPart = encoded.AsSpan(tokenOffset, encoded.Length - tokenOffset).ToArray();
        var fb = new Vp8FrameBuffer(width, height);
        var ec = new Vp8EntropyContexts(fb.MbCols);
        Vp8KeyframeWalker.Decode(tag, hdr, modeReader, tokenPart, fb, ec);

        int uvW = (width + 1) / 2, uvH = (height + 1) / 2;
        var y = new byte[width * height];
        var u = new byte[uvW * uvH];
        var v = new byte[uvW * uvH];
        for (int r = 0; r < height; r++)
            Buffer.BlockCopy(fb.YPlane, r * fb.YStride, y, r * width, width);
        for (int r = 0; r < uvH; r++)
        {
            Buffer.BlockCopy(fb.UPlane, r * fb.UvStride, u, r * uvW, uvW);
            Buffer.BlockCopy(fb.VPlane, r * fb.UvStride, v, r * uvW, uvW);
        }
        return (y, u, v, width, height);
    }

    private static void EqualPlanes(byte[] expected, byte[] actual, string what)
    {
        Equal(expected.Length, actual.Length, $"{what} length");
        int mismatches = 0, firstBad = -1;
        for (int i = 0; i < expected.Length; i++)
            if (expected[i] != actual[i]) { if (firstBad < 0) firstBad = i; mismatches++; }
        Equal(0, mismatches, $"{what}: first mismatch at {firstBad}");
    }

    [TestMethod(Timeout = 600000)]
    public async Task Vp8KeyframeDecoderGpu_ArrayViewOverload_WritesCallerPlanesAndMatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new Vp8KeyframeEncoderGpu(acc);
            using var dec = new Vp8KeyframeDecoderGpu(acc);
            const int width = 64, height = 48, baseQIndex = 30;
            var rng = new Random(0x71C1);
            var y = RandomPlane(rng, width * height);
            var u = RandomPlane(rng, (width / 2) * (height / 2));
            var v = RandomPlane(rng, (width / 2) * (height / 2));
            byte[] encoded = await enc.EncodeKeyFrameAsync(y, width, u, width / 2, v, width, height, baseQIndex);
            var cpu = CpuDecodeVp8Cropped(encoded);

            // Encoded frame on the accelerator, in a buffer LONGER than the frame (the length is explicit).
            using var dEnc = acc.Allocate1D<byte>(encoded.Length + 37);
            DeviceTransfer.Upload(dEnc.View, encoded);
            using var dY = acc.Allocate1D<byte>(Vp8KeyframeDecoderGpu.GetLumaPlaneSize(width, height) + 5);
            using var dU = acc.Allocate1D<byte>(Vp8KeyframeDecoderGpu.GetChromaPlaneSize(width, height) + 5);
            using var dV = acc.Allocate1D<byte>(Vp8KeyframeDecoderGpu.GetChromaPlaneSize(width, height) + 5);
            foreach (var b in new[] { dY, dU, dV })
            {
                var fill = new byte[b.Length];
                System.Array.Fill(fill, (byte)0xAB);
                DeviceTransfer.Upload(b.View, fill);
            }

            var tag = await dec.DecodeKeyFrameAsync(dEnc.View, encoded.Length, baseQIndex, dY.View, dU.View, dV.View);
            Equal(width, tag.Width!.Value, "tag width");
            Equal(height, tag.Height!.Value, "tag height");
            EqualPlanes(cpu.Y, await DeviceTransfer.DownloadAsync(dY.View, cpu.Y.Length), "Y plane vs CPU walker");
            EqualPlanes(cpu.U, await DeviceTransfer.DownloadAsync(dU.View, cpu.U.Length), "U plane vs CPU walker");
            EqualPlanes(cpu.V, await DeviceTransfer.DownloadAsync(dV.View, cpu.V.Length), "V plane vs CPU walker");

            // The managed overload funnels into the same implementation.
            var managed = await dec.DecodeKeyFrameAsync(encoded, baseQIndex);
            EqualPlanes(cpu.Y, managed.YPlane, "managed Y plane vs CPU walker");
            EqualPlanes(cpu.U, managed.UPlane, "managed U plane vs CPU walker");
            EqualPlanes(cpu.V, managed.VPlane, "managed V plane vs CPU walker");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod(Timeout = 600000)]
    public async Task Vp8KeyframeDecoderGpu_Unaligned_33x17_MatchesCroppedCpuWalker()
    {
        await RunUnalignedDecodeMatchesCpu(width: 33, height: 17, seed: 0x71C2);
    }

    [TestMethod(Timeout = 600000)]
    public async Task Vp8KeyframeDecoderGpu_Unaligned_47x32_MatchesCroppedCpuWalker()
    {
        await RunUnalignedDecodeMatchesCpu(width: 47, height: 32, seed: 0x71C3);
    }

    private async Task RunUnalignedDecodeMatchesCpu(int width, int height, int seed)
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new Vp8KeyframeEncoderGpu(acc);
            using var dec = new Vp8KeyframeDecoderGpu(acc);
            const int baseQIndex = 30;
            int uvW = (width + 1) / 2, uvH = (height + 1) / 2;
            var rng = new Random(seed);
            var y = RandomPlane(rng, width * height);
            var u = RandomPlane(rng, uvW * uvH);
            var v = RandomPlane(rng, uvW * uvH);
            byte[] encoded = await enc.EncodeKeyFrameAsync(y, width, u, uvW, v, width, height, baseQIndex);
            var cpu = CpuDecodeVp8Cropped(encoded);

            var gpu = await dec.DecodeKeyFrameAsync(encoded, baseQIndex);
            Equal(width, gpu.Width, "decoded width");
            Equal(height, gpu.Height, "decoded height");
            EqualPlanes(cpu.Y, gpu.YPlane, $"{width}x{height} Y plane vs cropped CPU walker");
            EqualPlanes(cpu.U, gpu.UPlane, $"{width}x{height} U plane vs cropped CPU walker");
            EqualPlanes(cpu.V, gpu.VPlane, $"{width}x{height} V plane vs cropped CPU walker");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp8KeyframeDecoderGpu_TypedArrayOverload_MatchesManaged()
    {
        if (!OperatingSystem.IsBrowser())
            throw new UnsupportedTestException("TypedArray overloads need a browser accelerator.");
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new Vp8KeyframeEncoderGpu(acc);
            using var dec = new Vp8KeyframeDecoderGpu(acc);
            await RunTypedArrayDecodeMatchesManaged(enc, dec);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("browser")]
    private static async Task RunTypedArrayDecodeMatchesManaged(Vp8KeyframeEncoderGpu enc, Vp8KeyframeDecoderGpu dec)
    {
        const int width = 33, height = 17, baseQIndex = 30;
        int uvW = (width + 1) / 2, uvH = (height + 1) / 2;
        var rng = new Random(0x71C4);
        var y = RandomPlane(rng, width * height);
        var u = RandomPlane(rng, uvW * uvH);
        var v = RandomPlane(rng, uvW * uvH);
        byte[] encoded = await enc.EncodeKeyFrameAsync(y, width, u, uvW, v, width, height, baseQIndex);
        var managed = await dec.DecodeKeyFrameAsync(encoded, baseQIndex);

        using var jEnc = new Uint8Array(encoded.Length); jEnc.Set(encoded);
        using var frame = await dec.DecodeKeyFrameAsync(jEnc, baseQIndex);
        Equal(width, frame.Width, "TypedArray decode width");
        Equal(height, frame.Height, "TypedArray decode height");
        // Reading back to .NET is the TEST's terminal compare, not the product path.
        EqualPlanes(managed.YPlane, frame.YPlane.ReadBytes(), "TypedArray Y plane vs managed");
        EqualPlanes(managed.UPlane, frame.UPlane.ReadBytes(), "TypedArray U plane vs managed");
        EqualPlanes(managed.VPlane, frame.VPlane.ReadBytes(), "TypedArray V plane vs managed");
    }
}
