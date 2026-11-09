// Kernel loading of the codecs (FLAC, MP3, Opus, VP9 decode, and the still-image encoders and decoders: BMP, GIF, JPEG,
// PNG, WebP lossless / lossy): constructors compile nothing, WarmUpAsync compiles every kernel the codec uses (awaited
// loads, so a browser page keeps running), and work after the warm-up compiles nothing more.
//
// Lore (SpawnGen) measured the page frozen 30-104 s while Codecs compiled its kernels synchronously in constructors and
// on first use. ILGPU 5.4.19's LoadAutoGroupedStreamKernelAsync yields to the event loop before each compile; the
// codecs now load through it, lazily, from their async entry points. Uses a FRESH accelerator: the shared one's kernel
// cache would hide a constructor compile (a cache hit does not count in CompiledKernelCount).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.Codecs.Audio.Mp3;
using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.Codecs.Images.Bmp;
using SpawnDev.Codecs.Images.Gif;
using SpawnDev.Codecs.Images.Jpeg;
using SpawnDev.Codecs.Images.Png;
using SpawnDev.Codecs.Images.WebP;
using SpawnDev.Codecs.Images.WebP.Lossy;
using SpawnDev.Codecs.Video.Vp9.Decoder;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Codecs_ConstructorsCompileNothing_WarmUpCompilesWhatTheyUse()
    {
        (Context ctx, Accelerator acc) fresh;
        try { fresh = await CreateKernelAcceleratorAsync(); }
        catch (NotSupportedException ex) { throw new UnsupportedTestException(ex.Message); }
        var (ctx, acc) = fresh;
        try
        {
            long Count() => acc.CompiledKernelCount;
            long c0 = Count();
            using var flacDec = new FlacDecoderGpu(acc);
            using var flacEnc = new FlacEncoderGpu(acc);
            using var mp3 = new Mp3DecoderGpu(acc);
            using var opus = new OpusEncoderGpu(acc, 48000, 2, OpusEncoderApplication.Audio);
            using var vp9 = new Vp9DecoderGpu(acc);
            Equal(c0, Count(), "kernels compiled by the constructors (FlacDecoderGpu, FlacEncoderGpu, Mp3DecoderGpu, OpusEncoderGpu, Vp9DecoderGpu)");

            // the still-image codecs, also before anything of theirs is loaded
            long ci = Count();
            using var bmpDec = new BmpDecoderGpu(acc);
            using var bmpEnc = new BmpEncoderGpu(acc);
            using var gifDec = new GifDecoderGpu(acc);
            using var gifEnc = new GifEncoderGpu(acc);
            using var jpegDec = new JpegDecoderGpu(acc);
            using var jpegEnc = new JpegEncoderGpu(acc);
            using var pngDec = new PngDecoderGpu(acc);
            using var pngEnc = new PngEncoderGpu(acc);
            using var webpDec = new WebPDecoderGpu(acc);
            using var webpLossless = new WebPLosslessEncoderGpu(acc);
            using var webpLossy = new WebPLossyEncoderGpu(acc);
            Equal(ci, Count(), "kernels compiled by the image codec constructors (BMP, GIF, JPEG, PNG decoders and encoders, " +
                "WebPDecoderGpu, WebPLosslessEncoderGpu, WebPLossyEncoderGpu)");

            async Task<long> Compiled(Func<Task> work)
            {
                long before = Count();
                await work();
                return Count() - before;
            }
            long flacDecWarm = await Compiled(flacDec.WarmUpAsync);
            long flacEncWarm = await Compiled(flacEnc.WarmUpAsync);
            long mp3Warm = await Compiled(mp3.WarmUpAsync);
            long opusWarm = await Compiled(() => OggOpusEncoder.WarmUpAsync(acc));
            True(flacDecWarm > 0, "FlacDecoderGpu.WarmUpAsync compiled no kernel");
            True(flacEncWarm > 0, "FlacEncoderGpu.WarmUpAsync compiled no kernel");
            True(mp3Warm > 0, "Mp3DecoderGpu.WarmUpAsync compiled no kernel");
            True(opusWarm > 0, "OggOpusEncoder.WarmUpAsync compiled no kernel");
            long vp9Warm = await Compiled(vp9.WarmUpAsync);
            True(vp9Warm >= 10, $"Vp9DecoderGpu.WarmUpAsync compiled {vp9Warm} kernels; the decode uses 10");

            // real work after the warm-ups: a FLAC encode at another level + decode both ways, an Opus encode from int16
            // and float with the already constructed encoder; all kernels were loaded by the warm-ups
            const int n = 9000;
            var pcm = new int[n * 2];
            var s16 = new short[960 * 2];
            for (int i = 0; i < n; i++)
            {
                pcm[2 * i] = (int)(9000 * Math.Sin(i * 0.031));
                pcm[2 * i + 1] = (int)(7000 * Math.Sin(i * 0.017 + 2));
            }
            for (int i = 0; i < s16.Length; i++) s16[i] = (short)pcm[i];
            // The accelerator's kernel cache holds compiled kernels weakly: a kernel nothing keeps alive (OggOpusEncoder
            // makes an encoder per call) would be collected here and compiled again by the work below.
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            long after = await Compiled(async () =>
            {
                var file = await flacEnc.EncodeAsync(pcm, 2, 44100, 16, new FlacEncodeOptions { CompressionLevel = 5 });
                var r = await flacDec.DecodeAsync(file);
                True(r.InterleavedSamples.AsSpan().SequenceEqual(pcm), "FLAC round trip after the warm-ups is not lossless");
                await flacDec.DecodeFloatAsync(file);
                await opus.EncodePacketAsync(s16, 960);
                await OggOpusEncoder.EncodeAsync(acc, new float[4800 * 2], 48000, 2);
            });
            Equal(0L, after, "kernels compiled by work after the warm-ups");

            // The image codecs' tests need scatter stores (they are NotOnWebGL): the constructor check above runs on every
            // backend, the warm-ups and the work below where the codecs run.
            if (acc.Device.Satisfies(SerialStreamKernels))
            {
                var imageWarmUps = new (string Name, Func<Task> WarmUp)[]
                {
                    ("BmpDecoderGpu", bmpDec.WarmUpAsync), ("BmpEncoderGpu", bmpEnc.WarmUpAsync),
                    ("GifDecoderGpu", gifDec.WarmUpAsync), ("GifEncoderGpu", gifEnc.WarmUpAsync),
                    ("JpegDecoderGpu", jpegDec.WarmUpAsync), ("JpegEncoderGpu", jpegEnc.WarmUpAsync),
                    ("PngDecoderGpu", pngDec.WarmUpAsync), ("PngEncoderGpu", pngEnc.WarmUpAsync),
                    ("WebPDecoderGpu", webpDec.WarmUpAsync), ("WebPLosslessEncoderGpu", webpLossless.WarmUpAsync),
                    ("WebPLossyEncoderGpu", webpLossy.WarmUpAsync),
                };
                // Each warm-up loads at least one kernel no earlier one shares (the shared ones are cache hits, not counted).
                foreach (var (name, warmUp) in imageWarmUps)
                    True(await Compiled(warmUp) > 0, $"{name}.WarmUpAsync compiled no kernel");

                // real work after the warm-ups: encode a picture with every encoder and decode it again
                const int w = 24, h = 20;
                var rgba = new byte[w * h * 4];
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        int p = (y * w + x) * 4;
                        rgba[p] = (byte)(x * 10);
                        rgba[p + 1] = (byte)(y * 12);
                        rgba[p + 2] = (byte)((x * y * 7) & 255);
                        rgba[p + 3] = 255;
                    }
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
                long imageAfter = await Compiled(async () =>
                {
                    var bmp = await bmpEnc.EncodeAsync(rgba, w, h);
                    True((await bmpDec.DecodeAsync(bmp)).Pixels.AsSpan().SequenceEqual(rgba), "BMP round trip after the warm-ups is not lossless");
                    var gif = await gifEnc.EncodeAsync(rgba, w, h);
                    await gifDec.DecodeAsync(gif);
                    var jpeg = await jpegEnc.EncodeAsync(rgba, w, h);
                    await jpegDec.DecodeAsync(jpeg);
                    var png = await pngEnc.EncodeAsync(rgba, w, h);
                    True((await pngDec.DecodeAsync(png)).Pixels.AsSpan().SequenceEqual(rgba), "PNG round trip after the warm-ups is not lossless");
                    var lossless = await webpLossless.EncodeAsync(rgba, w, h);
                    True((await webpDec.DecodeAsync(lossless)).Pixels.AsSpan().SequenceEqual(rgba), "WebP lossless round trip after the warm-ups is not lossless");
                    var lossy = await webpLossy.EncodeAsync(rgba, w, h);
                    await webpDec.DecodeAsync(lossy);
                });
                Equal(0L, imageAfter, "kernels compiled by image work after the warm-ups");
            }
        }
        finally
        {
            acc.Dispose();
            ctx.Dispose();
        }
    }
}
