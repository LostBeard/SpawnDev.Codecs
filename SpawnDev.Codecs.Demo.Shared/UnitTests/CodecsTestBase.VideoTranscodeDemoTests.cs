// Tests that mirror the pipeline used by the /transcode demo page
// (Pages/Transcode.razor in SpawnDev.Codecs.Demo). Verifies the exact
// CPU-public-API encode -> decode flow the page wires together so the
// demo can be trusted to produce a sane round-trip without manual
// browser testing every commit.

using SpawnDev.Codecs.Video;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private sealed class VideoCaptureSink : IVideoFrameSink
    {
        public byte[]? Y;
        public byte[]? U;
        public byte[]? V;
        public int YStride;
        public int UStride;
        public int VStride;
        public ValueTask OnFrameAsync(
            ReadOnlyMemory<byte> y, int ys,
            ReadOnlyMemory<byte> u, int us,
            ReadOnlyMemory<byte> v, int vs,
            long pts)
        {
            Y = y.ToArray(); YStride = ys;
            U = u.ToArray(); UStride = us;
            V = v.ToArray(); VStride = vs;
            return ValueTask.CompletedTask;
        }
    }

    [TestMethod]
    public async Task VideoTranscodeDemo_Vp8_GradientFrame_RoundTripsViaCpuApi()
    {
        await VideoTranscodeDemoRoundTripAsync(
            "VP8", width: 64, height: 64, q: 30,
            (y, u, v, w, h, q) => Vp8KeyframeEncoder.EncodeKeyFrame(
                y, w, u, w / 2, v, w, h, baseQIndex: q),
            () => new Vp8Decoder());
    }

    [TestMethod]
    public async Task VideoTranscodeDemo_Vp9_GradientFrame_RoundTripsViaCpuApi()
    {
        await VideoTranscodeDemoRoundTripAsync(
            "VP9", width: 64, height: 64, q: 30,
            (y, u, v, w, h, q) => Vp9KeyframeEncoder.EncodeKeyFrame(
                y, w, u, w / 2, v, w, h, baseQIndex: q),
            () => new Vp9Decoder());
    }

    [TestMethod]
    public async Task VideoTranscodeDemo_Av1_GradientFrame_RoundTripsViaCpuApi()
    {
        await VideoTranscodeDemoRoundTripAsync(
            "AV1", width: 64, height: 64, q: 32,
            (y, u, v, w, h, q) => Av1KeyframeEncoder.EncodeKeyFrame(
                y, w, u, w / 2, v, w, h, baseQIndex: q),
            () => new Av1Decoder());
    }

    private async Task VideoTranscodeDemoRoundTripAsync(
        string codec, int width, int height, int q,
        Func<byte[], byte[], byte[], int, int, int, byte[]> encode,
        Func<IVideoDecoder> newDecoder)
    {
        // Build a synthetic gradient frame the same way Transcode.razor does.
        var ySrc = new byte[width * height];
        for (int r = 0; r < height; r++)
            for (int c = 0; c < width; c++)
                ySrc[r * width + c] = (byte)Math.Clamp(
                    96 + 32 * Math.Sin(2.0 * Math.PI * c / 16.0) + r * 2, 0, 255);
        var uSrc = new byte[(width / 2) * (height / 2)]; Array.Fill(uSrc, (byte)128);
        var vSrc = new byte[(width / 2) * (height / 2)]; Array.Fill(vSrc, (byte)128);

        // Encode through the same public API the page uses.
        byte[] encoded = encode(ySrc, uSrc, vSrc, width, height, q);
        True(encoded.Length > 0, $"{codec} encoder must produce non-empty output");

        // Decode back via the matching public IVideoDecoder.
        var sink = new VideoCaptureSink();
        var dec = newDecoder();
        try
        {
            await dec.DecodeFrameAsync(encoded, sink);
        }
        finally
        {
            await dec.DisposeAsync();
        }
        True(sink.Y is not null, $"{codec} decoder must hand back at least one frame");
        True(sink.Y!.Length >= width * height,
            $"{codec} decoder Y plane must be at least width*height bytes; got {sink.Y.Length}");

        // Sanity check: PSNR vs source must be above a low floor (the
        // same metric Transcode.razor reports). Lossy keyframes at q=30
        // recover comfortably above 15 dB; we set 10 dB as the floor so
        // codec-specific tweaks don't trigger false alarms while still
        // catching outright decode breakage.
        double psnr = ComputeYPsnrForTest(ySrc, sink.Y, width, height, sink.YStride);
        True(psnr > 10.0,
            $"{codec} demo round-trip Y PSNR floor 10 dB; got {psnr:F1} dB");
    }

    private static double ComputeYPsnrForTest(
        byte[] sourceY, byte[] decodedY, int width, int height, int decodedYStride)
    {
        if (decodedYStride < width) return 0;
        long sumSq = 0; long n = 0;
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                int s = sourceY[row * width + col];
                int d = decodedY[row * decodedYStride + col];
                int e = s - d;
                sumSq += e * e;
                n++;
            }
        }
        if (sumSq == 0) return double.PositiveInfinity;
        if (n == 0) return 0;
        double mse = sumSq / (double)n;
        return 10.0 * Math.Log10(255.0 * 255.0 / mse);
    }
}
