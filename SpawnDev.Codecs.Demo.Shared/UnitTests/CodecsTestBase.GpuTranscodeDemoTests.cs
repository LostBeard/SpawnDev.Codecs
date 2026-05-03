// Tests that mirror the pipeline used by the /gpu-transcode demo page
// (Pages/GpuTranscode.razor in SpawnDev.Codecs.Demo). Verifies the
// 100%-ILGPU encode -> decode flow for each video GPU pair so the demo
// can be trusted to produce a sane round-trip without manual browser
// click-through every commit.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task GpuTranscodeDemo_Vp8_GradientFrame_RoundTripsViaGpuPair()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // SpawnDev.ILGPU 4.9.3 WebGPU codegen has an unhandled
            // 'SubViewValue' IR node case for kernels invoked with a
            // SubView ArrayView (introduced by Codecs commit 2cb38a5).
            // Wasm has a memory OOB on the same kernel chain. Both are
            // cross-lane bugs filed with Geordi; skip the demo round-trip
            // on browser backends until the fixes ship. Desktop backends
            // (CPU / CUDA / OpenCL) cover the GPU pair surface bit-exact.
            if (acc.AcceleratorType is not (AcceleratorType.CPU
                                            or AcceleratorType.Cuda
                                            or AcceleratorType.OpenCL))
            {
                throw new UnsupportedTestException(
                    $"GPU pair demo round-trip on {acc.AcceleratorType} backend is gated on SpawnDev.ILGPU cross-lane fixes "
                    + "(WebGPU SubView IR codegen + Wasm memory OOB). Existing GPU pair tests cover desktop backends.");
            }

            const int width = 64, height = 64, q = 30;
            var (ySrc, uSrc, vSrc) = GenerateGradientYuv420(width, height);

            using var enc = new Vp8KeyframeEncoderGpu(acc);
            byte[] encoded = enc.EncodeKeyFrame(
                ySrc, ySrcStride: width,
                uSrc, uvSrcStride: width / 2,
                vSrc,
                width, height, baseQIndex: q);
            True(encoded.Length > 0, "VP8-GPU encoder must produce non-empty output");

            using var dec = new Vp8KeyframeDecoderGpu(acc);
            var frame = dec.DecodeKeyFrame(encoded, baseQIndex: q);
            True(frame.YPlane.Length == width * height,
                $"VP8-GPU decoder Y plane must be width*height bytes; got {frame.YPlane.Length}");

            double psnr = ComputeYPsnrForTest(ySrc, frame.YPlane, width, height, width);
            True(psnr > 10.0,
                $"VP8-GPU demo round-trip Y PSNR floor 10 dB; got {psnr:F1} dB");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task GpuTranscodeDemo_Vp9_GradientFrame_RoundTripsViaGpuPair()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // SpawnDev.ILGPU 4.9.3 WebGPU codegen has an unhandled
            // 'SubViewValue' IR node case for kernels invoked with a
            // SubView ArrayView (introduced by Codecs commit 2cb38a5).
            // Wasm has a memory OOB on the same kernel chain. Both are
            // cross-lane bugs filed with Geordi; skip the demo round-trip
            // on browser backends until the fixes ship. Desktop backends
            // (CPU / CUDA / OpenCL) cover the GPU pair surface bit-exact.
            if (acc.AcceleratorType is not (AcceleratorType.CPU
                                            or AcceleratorType.Cuda
                                            or AcceleratorType.OpenCL))
            {
                throw new UnsupportedTestException(
                    $"GPU pair demo round-trip on {acc.AcceleratorType} backend is gated on SpawnDev.ILGPU cross-lane fixes "
                    + "(WebGPU SubView IR codegen + Wasm memory OOB). Existing GPU pair tests cover desktop backends.");
            }

            const int width = 64, height = 64, q = 30;
            var (ySrc, uSrc, vSrc) = GenerateGradientYuv420(width, height);

            using var enc = new Vp9KeyframeEncoderGpu(acc);
            byte[] encoded = await enc.EncodeKeyFrameAsync(
                ySrc, uSrc, vSrc, width, height, baseQIndex: q);
            True(encoded.Length > 0, "VP9-GPU encoder must produce non-empty output");

            using var dec = new Vp9KeyframeDecoderGpu(acc);
            var frame = await dec.DecodeKeyFrameAsync(encoded);
            True(frame.YPlane.Length == width * height,
                $"VP9-GPU decoder Y plane must be width*height bytes; got {frame.YPlane.Length}");

            double psnr = ComputeYPsnrForTest(ySrc, frame.YPlane, width, height, width);
            True(psnr > 10.0,
                $"VP9-GPU demo round-trip Y PSNR floor 10 dB; got {psnr:F1} dB");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task GpuTranscodeDemo_Av1_GradientFrame_RoundTripsViaGpuPair()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // SpawnDev.ILGPU 4.9.3 WebGPU codegen has an unhandled
            // 'SubViewValue' IR node case for kernels invoked with a
            // SubView ArrayView (introduced by Codecs commit 2cb38a5).
            // Wasm has a memory OOB on the same kernel chain. Both are
            // cross-lane bugs filed with Geordi; skip the demo round-trip
            // on browser backends until the fixes ship. Desktop backends
            // (CPU / CUDA / OpenCL) cover the GPU pair surface bit-exact.
            if (acc.AcceleratorType is not (AcceleratorType.CPU
                                            or AcceleratorType.Cuda
                                            or AcceleratorType.OpenCL))
            {
                throw new UnsupportedTestException(
                    $"GPU pair demo round-trip on {acc.AcceleratorType} backend is gated on SpawnDev.ILGPU cross-lane fixes "
                    + "(WebGPU SubView IR codegen + Wasm memory OOB). Existing GPU pair tests cover desktop backends.");
            }

            const int width = 64, height = 64, q = 32;
            var (ySrc, uSrc, vSrc) = GenerateGradientYuv420(width, height);

            using var enc = new Av1KeyframeEncoderGpu(acc);
            // Demo uses the tile-bytes flow: EncodeSingleTileAsync ->
            // DecodeSingleTileAsync (full TD/SH/Frame OBU bitstream is
            // for ffmpeg/dav1d compatibility, not in-library round-trip).
            byte[] tileBytes = await enc.EncodeSingleTileAsync(
                ySrc, uSrc, vSrc, width, height, baseQIndex: q);
            True(tileBytes.Length > 0, "AV1-GPU encoder must produce non-empty tile bytes");

            using var dec = new Av1KeyframeDecoderGpu(acc);
            var (yRecon, uRecon, vRecon) = await dec.DecodeSingleTileAsync(
                tileBytes, width, height, baseQIndex: q);
            True(yRecon.Length == width * height,
                $"AV1-GPU decoder Y plane must be width*height bytes; got {yRecon.Length}");

            double psnr = ComputeYPsnrForTest(ySrc, yRecon, width, height, width);
            True(psnr > 10.0,
                $"AV1-GPU demo round-trip Y PSNR floor 10 dB; got {psnr:F1} dB");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static (byte[] y, byte[] u, byte[] v) GenerateGradientYuv420(int width, int height)
    {
        // Same gradient the GpuTranscode.razor page uses.
        var y = new byte[width * height];
        for (int r = 0; r < height; r++)
            for (int c = 0; c < width; c++)
                y[r * width + c] = (byte)Math.Clamp(
                    96 + 32 * Math.Sin(2.0 * Math.PI * c / 16.0) + r * 1.5, 0, 255);
        var u = new byte[(width / 2) * (height / 2)]; Array.Fill(u, (byte)128);
        var v = new byte[(width / 2) * (height / 2)]; Array.Fill(v, (byte)128);
        return (y, u, v);
    }
}
