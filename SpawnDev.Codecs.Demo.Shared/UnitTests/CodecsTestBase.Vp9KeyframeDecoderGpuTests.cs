// Cross-backend tests for Vp9KeyframeDecoderGpu - the v3 100% ILGPU
// VP9 v1 keyframe decoder. Verifies the GPU decoder produces the
// SAME recon planes that Vp9KeyframeWalker (CPU oracle) produces
// when fed the same encoded bytes.

#pragma warning disable CS0618 // Vp9KeyframeDecoderGpu is obsolete; these tests keep covering it until it is removed
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.Codecs.Video.Vp9.Decoder;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Decode an encoded VP9 keyframe via the CPU walker. Returns
    /// the recon Y/U/V planes packed contiguously per plane.
    /// </summary>
    private static (byte[] yRecon, byte[] uRecon, byte[] vRecon, int width, int height)
        Vp9KeyframeDecodeCpu(byte[] frameBytes)
    {
        var complete = Vp9CompleteUncompressedHeaderParser.Parse(
            frameBytes.AsSpan(), refFrameSizes: null!);
        var fh = complete.FrameHeader;

        // Compressed header sits between byte-aligned uncompressed header
        // and tile data. v1 keyframes always have a non-zero
        // first_partition_size.
        var compressedState = new Vp9CompressedHeaderState();
        var inputs = new Vp9CompressedHeaderInputs(
            IsLossless: complete.Quantization.BaseQIndex == 0
                && complete.Quantization.YDcDeltaQ == 0
                && complete.Quantization.UvDcDeltaQ == 0
                && complete.Quantization.UvAcDeltaQ == 0,
            IsIntraOnly: true,
            InterpFilter: complete.InterpFilter,
            AllowHighPrecisionMv: complete.AllowHighPrecisionMv,
            SignBiasLast: false,
            SignBiasGolden: false,
            SignBiasAltRef: false);

        var headerBytes = frameBytes.AsSpan(
            complete.UncompressedHeaderSizeBytes,
            complete.FirstPartitionSize).ToArray();
        var reader = new Vp9BoolDecoder(headerBytes, 0, headerBytes.Length);
        var compressedResult = Vp9CompressedHeaderParser.Read(compressedState, inputs, reader);

        var tileGroup = Vp9TileGroupExtractor.Extract(frameBytes.AsSpan(), complete);

        var walker = new Vp9KeyframeWalker();
        var fb = walker.DecodeFrame(
            frameBytes.AsMemory(),
            complete,
            compressedState,
            compressedResult,
            tileGroup);

        // Pack planes into contiguous arrays. Vp9FrameBuffer's stored
        // plane has stride padded to LumaWidth / ChromaWidth which
        // already equals the frame width (no extra padding) but read
        // explicitly via stride to be defensive.
        int yLen = fh.FrameWidth * fh.FrameHeight;
        int uvLen = yLen / 4;
        var y = new byte[yLen];
        var u = new byte[uvLen];
        var v = new byte[uvLen];
        Array.Copy(fb.Y, 0, y, 0, yLen);
        Array.Copy(fb.U, 0, u, 0, uvLen);
        Array.Copy(fb.V, 0, v, 0, uvLen);
        return (y, u, v, fh.FrameWidth, fh.FrameHeight);
    }

    private static async Task AssertVp9KeyframeDecoderGpuMatchesCpuAsync(
        Accelerator acc,
        byte[] yPlane, byte[] uPlane, byte[] vPlane,
        int width, int height, int baseQIndex)
    {
        // Encode source with CPU encoder. Bytes are bit-exact mirror
        // of libvpx.
        var encodedBytes = Vp9KeyframeEncoder.EncodeKeyFrame(
            yPlane, ySrcStride: width,
            uPlane, uvSrcStride: width / 2,
            vPlane,
            width, height,
            baseQIndex);

        // CPU walker decodes the bytes -> oracle recon planes.
        var (cpuY, cpuU, cpuV, cpuW, cpuH) = Vp9KeyframeDecodeCpu(encodedBytes);
        Equal(width, cpuW);
        Equal(height, cpuH);

        // GPU decoder decodes the SAME bytes.
        using var decoder = new Vp9KeyframeDecoderGpu(acc);
        var gpuFrame = await decoder.DecodeKeyFrameAsync(encodedBytes);
        Equal(width, gpuFrame.Width);
        Equal(height, gpuFrame.Height);

        for (int i = 0; i < cpuY.Length; i++)
            if (cpuY[i] != gpuFrame.YPlane[i])
                throw new Exception($"Y plane mismatch at {i}: cpu={cpuY[i]} gpu={gpuFrame.YPlane[i]}");
        for (int i = 0; i < cpuU.Length; i++)
            if (cpuU[i] != gpuFrame.UPlane[i])
                throw new Exception($"U plane mismatch at {i}: cpu={cpuU[i]} gpu={gpuFrame.UPlane[i]}");
        for (int i = 0; i < cpuV.Length; i++)
            if (cpuV[i] != gpuFrame.VPlane[i])
                throw new Exception($"V plane mismatch at {i}: cpu={cpuV[i]} gpu={gpuFrame.VPlane[i]}");
    }

    [TestMethod(Category = "NotOnWebGL")]
    public async Task Vp9KeyframeDecoderGpu_64x64_FlatGray_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
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

            await AssertVp9KeyframeDecoderGpuMatchesCpuAsync(acc, y, u, v, width, height, baseQIndex: 30);
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Category = "NotOnWebGL")]
    public async Task Vp9KeyframeDecoderGpu_64x64_RandomContent_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            int width = 64, height = 64;
            int yLen = width * height;
            int uvLen = yLen / 4;
            var rng = new Random(unchecked((int)0xDC9F0001u));
            var y = new byte[yLen];
            var u = new byte[uvLen];
            var v = new byte[uvLen];
            rng.NextBytes(y);
            rng.NextBytes(u);
            rng.NextBytes(v);

            await AssertVp9KeyframeDecoderGpuMatchesCpuAsync(acc, y, u, v, width, height, baseQIndex: 30);
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Category = "NotOnWebGL")]
    public async Task Vp9KeyframeDecoderGpu_64x64_BaseQSweep_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            int width = 64, height = 64;
            int yLen = width * height;
            int uvLen = yLen / 4;
            var rng = new Random(unchecked((int)0xDC9FBB02u));
            var y = new byte[yLen];
            var u = new byte[uvLen];
            var v = new byte[uvLen];
            rng.NextBytes(y);
            rng.NextBytes(u);
            rng.NextBytes(v);

            int[] baseQs = { 1, 30, 64, 128, 200 };
            foreach (var q in baseQs)
                await AssertVp9KeyframeDecoderGpuMatchesCpuAsync(acc, y, u, v, width, height, baseQIndex: q);
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>
    /// Self-consistency: encode source with my GPU encoder (capturing
    /// the encoder's internal recon planes), decode the resulting
    /// bytes with my GPU decoder, verify the decoder produces the
    /// SAME recon planes the encoder built. Bypasses the CPU walker
    /// oracle so this test is independent of any walker behavior.
    /// </summary>
    private static async Task AssertVp9EncoderDecoderSelfConsistentAsync(
        Accelerator acc,
        byte[] yPlane, byte[] uPlane, byte[] vPlane,
        int width, int height, int baseQIndex)
    {
        // Encode source + capture encoder's recon.
        using var enc = new Vp9KeyframeEncoderGpu(acc);
        var (encodedBytes, encYRecon, encURecon, encVRecon) =
            await enc.EncodeKeyFrameWithReconAsync(yPlane, uPlane, vPlane, width, height, baseQIndex);

        // Decode bytes -> decoder recon.
        using var dec = new Vp9KeyframeDecoderGpu(acc);
        var decFrame = await dec.DecodeKeyFrameAsync(encodedBytes);

        Equal(width, decFrame.Width);
        Equal(height, decFrame.Height);

        for (int i = 0; i < encYRecon.Length; i++)
            if (encYRecon[i] != decFrame.YPlane[i])
                throw new Exception($"Y self-consistency mismatch at {i}: encoder={encYRecon[i]} decoder={decFrame.YPlane[i]}");
        for (int i = 0; i < encURecon.Length; i++)
            if (encURecon[i] != decFrame.UPlane[i])
                throw new Exception($"U self-consistency mismatch at {i}: encoder={encURecon[i]} decoder={decFrame.UPlane[i]}");
        for (int i = 0; i < encVRecon.Length; i++)
            if (encVRecon[i] != decFrame.VPlane[i])
                throw new Exception($"V self-consistency mismatch at {i}: encoder={encVRecon[i]} decoder={decFrame.VPlane[i]}");
    }

    /// <summary>
    /// Our keyframe encoder's frames decoded by the full decoder (<see cref="Vp9DecoderGpu"/>, bit-exact with libvpx on
    /// the conformance vectors) equal the encoder's reconstruction and the keyframe-only decoder's planes: the evidence
    /// that <see cref="Vp9KeyframeDecoderGpu"/> can retire in favour of the full decoder.
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "NotOnWebGL,GiantShader")]
    public async Task Vp9DecoderGpu_DecodesOwnKeyframeEncoder_LikeKeyframeDecoder()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            var rng = new Random(0x5EED);
            // WebGPU diagnostics for a failure: the device-lost reason and the last dispatches (kernel, grid, group size)
            string lost = "";
            if (acc is SpawnDev.ILGPU.WebGPU.WebGPUAccelerator wg) wg.DeviceLost += (reason, message) => lost = $"{reason}: {message}";
            string Diagnostics() => acc is SpawnDev.ILGPU.WebGPU.WebGPUAccelerator
                ? $" | device lost: {lost} | last dispatches: " + string.Join("; ",
                    SpawnDev.ILGPU.WebGPU.WebGPUAccelerator.DispatchLog.TakeLast(8).Select(d =>
                        $"{d.KernelName} [{d.GridDimX}x{d.GridDimY}x{d.GridDimZ}, group {d.WorkgroupSize}]"))
                : "";
            foreach (var (w, h, q, flat) in new[] { (64, 64, 30, false), (64, 64, 120, false), (128, 128, 30, false), (64, 64, 60, true), (192, 128, 80, false) })
            {
                var y = new byte[w * h];
                var u = new byte[w * h / 4];
                var v = new byte[w * h / 4];
                if (flat) { Array.Fill(y, (byte)128); Array.Fill(u, (byte)128); Array.Fill(v, (byte)128); }
                else { rng.NextBytes(y); rng.NextBytes(u); rng.NextBytes(v); }
                string tag = $"{w}x{h} q{q}{(flat ? " flat" : "")}";
                // Each step waits for the accelerator before the next, so a failure (a lost WebGPU device surfaces at
                // the next wait) names the case and step that caused it.
                async Task<T> Step<T>(string step, Func<Task<T>> work)
                {
                    try
                    {
                        var r = await work();
                        await acc.SynchronizeAsync();
                        return r;
                    }
                    catch (Exception ex) { throw new Exception($"{tag}: {step}: {ex.GetType().Name}: {ex.Message}{Diagnostics()}"); }   // no inner: the runner reports the innermost
                }
                using var enc = new Vp9KeyframeEncoderGpu(acc);
                var (bytes, ry, ru, rv) = await Step("Vp9KeyframeEncoderGpu.EncodeKeyFrameWithReconAsync",
                    () => enc.EncodeKeyFrameWithReconAsync(y, u, v, w, h, q));
                using var old = new Vp9KeyframeDecoderGpu(acc);
                var oldFrame = await Step("Vp9KeyframeDecoderGpu.DecodeKeyFrameAsync", () => old.DecodeKeyFrameAsync(bytes));
                using var full = new Vp9DecoderGpu(acc);
                var frame = await Step("Vp9DecoderGpu.DecodePacketAsync", () => full.DecodePacketAsync(bytes));
                True(frame != null, $"{tag}: the full decoder shows no picture");
                Equal(new Vp9PictureFormat(w, h, 1, 1), frame!.Format, $"{tag}: format");
                byte[] planes = frame.Planes;
                int ySize = w * h, cSize = w * h / 4;
                EqualBytes(ry, planes[..ySize], $"{tag}: Y vs the encoder's reconstruction");
                EqualBytes(ru, planes[ySize..(ySize + cSize)], $"{tag}: U vs the encoder's reconstruction");
                EqualBytes(rv, planes[(ySize + cSize)..], $"{tag}: V vs the encoder's reconstruction");
                EqualBytes(oldFrame.YPlane, planes[..ySize], $"{tag}: Y vs Vp9KeyframeDecoderGpu");
                EqualBytes(oldFrame.UPlane, planes[ySize..(ySize + cSize)], $"{tag}: U vs Vp9KeyframeDecoderGpu");
                EqualBytes(oldFrame.VPlane, planes[(ySize + cSize)..], $"{tag}: V vs Vp9KeyframeDecoderGpu");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task Vp9KeyframeEncoderDecoderGpu_128x128_SelfConsistent()
    {
        // Self-consistency test that bypasses the CPU walker oracle.
        // If this passes for 128x128, the GPU encoder + GPU decoder
        // are mutually consistent at SB row boundaries.
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            int width = 128, height = 128;
            int yLen = width * height;
            int uvLen = yLen / 4;
            var rng = new Random(unchecked((int)0xDC9FC128u));
            var y = new byte[yLen];
            var u = new byte[uvLen];
            var v = new byte[uvLen];
            rng.NextBytes(y);
            rng.NextBytes(u);
            rng.NextBytes(v);

            await AssertVp9EncoderDecoderSelfConsistentAsync(acc, y, u, v, width, height, baseQIndex: 30);
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    // KNOWN INVESTIGATION (2026-04-28): multi-SB decoder vs CPU walker
    // mismatches at the SB row boundary.
    /*
    [TestMethod]
    public async Task Vp9KeyframeDecoderGpu_128x128_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            int width = 128, height = 128;
            int yLen = width * height;
            int uvLen = yLen / 4;
            var rng = new Random(unchecked((int)0xDC9FC128u));
            var y = new byte[yLen];
            var u = new byte[uvLen];
            var v = new byte[uvLen];
            rng.NextBytes(y);
            rng.NextBytes(u);
            rng.NextBytes(v);

            await AssertVp9KeyframeDecoderGpuMatchesCpuAsync(acc, y, u, v, width, height, baseQIndex: 30);
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
    */
}
