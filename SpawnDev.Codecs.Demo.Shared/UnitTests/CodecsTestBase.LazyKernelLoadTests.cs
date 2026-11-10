// Kernel loading of the codecs (FLAC, MP3 decode / encode, Opus encode / decode with its SILK and CELT layers, Vorbis
// encode / decode, VP9 decode, the still-image encoders and decoders: BMP, GIF, JPEG, PNG, WebP lossless / lossy,
// BCn, and the compressors: shuffle filters, WeightPack, GDeflate, LZ4 frame, Snappy): constructors compile nothing, WarmUpAsync compiles every kernel the codec uses (awaited loads, so a browser page
// keeps running), and work after the warm-up compiles nothing more.
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
using SpawnDev.Codecs.Audio.Opus.Celt;
using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.Codecs.Compression;
using SpawnDev.Codecs.Compression.GDeflate;
using SpawnDev.Codecs.Compression.Lz4;
using SpawnDev.Codecs.Compression.Snappy;
using SpawnDev.Codecs.Images.Bmp;
using SpawnDev.Codecs.Images.Gif;
using SpawnDev.Codecs.Images.Jpeg;
using SpawnDev.Codecs.Images.Png;
using SpawnDev.Codecs.Images.WebP;
using SpawnDev.Codecs.Images.WebP.Lossy;
using SpawnDev.Codecs.Images.Texture;
using SpawnDev.Codecs.Video.Vp9.Decoder;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Timeout = 1800000)]
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

            // the other audio codecs: the MP3 encoder, the Opus decoder and its SILK / CELT layers (also on their own), the
            // SILK frame orchestrator, the Vorbis encoder and decoder
            long ca = Count();
            using var mp3Enc = new Mp3EncoderGpu(acc);
            using var opusDec = new OpusDecoderGpu(acc, 48000, 2);
            using var silk = new SilkPacketDecoderGpu(acc, 48000, 2);
            using var celt = new CeltDecoderGpu(acc, 48000, 2);
            using var silkFrames = new SilkDecodeFrameGpuOrchestrator(acc);
            var vorbisOptions = new VorbisAudioEncoderOptions { SampleRateHz = 44100, Channels = 1, BlockSize = 1024 };
            using var vorbisEnc = new VorbisAudioEncoderGpu(acc, vorbisOptions);
            using var vorbisDec = new VorbisAudioDecoderGpu(acc, vorbisEnc.Identification, vorbisEnc.Setup);
            Equal(ca, Count(), "kernels compiled by the constructors (Mp3EncoderGpu, OpusDecoderGpu, SilkPacketDecoderGpu, " +
                "CeltDecoderGpu, SilkDecodeFrameGpuOrchestrator, VorbisAudioEncoderGpu, VorbisAudioDecoderGpu)");

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
            using var bcnEnc = new BcnEncoderGpu(acc);
            using var bcnDec = new BcnDecoderGpu(acc);
            Equal(ci, Count(), "kernels compiled by the image codec constructors (BMP, GIF, JPEG, PNG decoders and encoders, " +
                "WebPDecoderGpu, WebPLosslessEncoderGpu, WebPLossyEncoderGpu, BcnEncoderGpu, BcnDecoderGpu)");

            // the compressors (small waves: the working memory, not the kernels, scales with them)
            long cz = Count();
            using var shuffle = new ShuffleFilterGpu(acc);
            using var weightPack = new WeightPackGpu(acc, decompressWave: 16);
            using var gdeflateEnc = new GDeflateCompressorGpu(acc, wave: 4);
            using var gdeflateDec = new GDeflateDecompressorGpu(acc, wave: 4);
            using var lz4Enc = new Lz4FrameCompressorGpu(acc, waveBlocks: 4);
            using var lz4Dec = new Lz4FrameDecompressorGpu(acc);
            using var snappyEnc = new SnappyCompressorGpu(acc, waveBlocks: 4);
            using var snappyDec = new SnappyDecompressorGpu(acc);
            Equal(cz, Count(), "kernels compiled by the compressor constructors (ShuffleFilterGpu, WeightPackGpu, GDeflate, LZ4 frame " +
                "and Snappy compressors and decompressors)");

            // The codecs need scatter stores (their tests are NotOnWebGL): the constructor checks above run on every
            // backend, the warm-ups and the work below where the codecs run.
            if (!acc.Device.Satisfies(SerialStreamKernels)) return;

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
            string warmCounts = $"{acc.AcceleratorType}: compiled by warm-ups: FLAC decoder {flacDecWarm}, FLAC encoder {flacEncWarm}, " +
                $"MP3 decoder {mp3Warm}, Ogg Opus {opusWarm}; constructors {Count() - c0 - flacDecWarm - flacEncWarm - mp3Warm - opusWarm}";
            Console.WriteLine(warmCounts);
            True(flacDecWarm > 0, "FlacDecoderGpu.WarmUpAsync compiled no kernel. " + warmCounts);
            True(flacEncWarm > 0, "FlacEncoderGpu.WarmUpAsync compiled no kernel. " + warmCounts);
            True(mp3Warm > 0, "Mp3DecoderGpu.WarmUpAsync compiled no kernel. " + warmCounts);
            True(opusWarm > 0, "OggOpusEncoder.WarmUpAsync compiled no kernel. " + warmCounts);
            // The VP9 decoder's warm-up compiles its kernels without running them. On WebGPU they compile in Chrome in at
            // most a few seconds and under 0.6 GB each since 2026-11-10 (call-site rewrites); WebGL still cannot hold
            // them (GLSL). Its constructor check above runs everywhere.
            if (acc.AcceleratorType is not AcceleratorType.WebGL)
            {
                long vp9Warm = await Compiled(vp9.WarmUpAsync);
                True(vp9Warm >= 10, $"Vp9DecoderGpu.WarmUpAsync compiled {vp9Warm} kernels; the decode uses 10");
            }
            // the SILK and CELT layers on their own first: the Opus decoder's warm-up loads theirs too (cache hits then,
            // not counted), and its own kernels on top
            var audioWarmUps = new (string Name, Func<Task> WarmUp)[]
            {
                ("Mp3EncoderGpu", mp3Enc.WarmUpAsync), ("SilkPacketDecoderGpu", silk.WarmUpAsync),
                ("CeltDecoderGpu", celt.WarmUpAsync), ("SilkDecodeFrameGpuOrchestrator", silkFrames.WarmUpAsync),
                ("OpusDecoderGpu", opusDec.WarmUpAsync), ("VorbisAudioEncoderGpu", vorbisEnc.WarmUpAsync),
                ("VorbisAudioDecoderGpu", vorbisDec.WarmUpAsync),
            };
            foreach (var (name, warmUp) in audioWarmUps)
                True(await Compiled(warmUp) > 0, $"{name}.WarmUpAsync compiled no kernel");
            var opusFixture = LoadOpusFixture("switch_modes_48k_stereo_loss");

            // real work after the warm-ups: a FLAC encode at another level + decode both ways, an Opus encode from int16
            // and float with the already constructed encoder, an MP3 encode (CBR from float, VBR from int32) decoded again,
            // a libopus stream through every Opus mode switch and loss (checked against libopus), a Vorbis encode (single,
            // batch, stream) decoded again; all kernels were loaded by the warm-ups
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

                var pcmF = new float[pcm.Length];
                for (int i = 0; i < pcm.Length; i++) pcmF[i] = pcm[i] / 32768f;
                await mp3.DecodeAsync(await mp3Enc.EncodeAsync(pcmF, 2, 48000));
                var pcm32 = new int[pcm.Length];
                for (int i = 0; i < pcm.Length; i++) pcm32[i] = pcm[i] << 16;
                await mp3.DecodeAsync(await mp3Enc.EncodeAsync(pcm32, 2, 22050, new Mp3EncodeOptions { VbrQuality = 4 }));

                int at = 0;
                for (int p = 0; p < opusFixture.Packets.Count; p++)
                {
                    var pkt = opusFixture.Packets[p];
                    var got = pkt is null
                        ? await opusDec.DecodeLostAsync((int)((long)opusFixture.ApiRate * opusFixture.PacketUs / 1_000_000))
                        : await opusDec.DecodePacketAsync(pkt);
                    for (int i = 0; i < got.Length; i++)
                        if (got[i] != opusFixture.Pcm[at + i] / 32768.0f)
                            throw new Exception($"Opus packet {p} after the warm-ups: sample {i} differs from libopus");
                    at += got.Length;
                    Equal(opusFixture.Ranges[p], await opusDec.GetFinalRangeAsync(), $"Opus packet {p} final range");
                }
                Equal(opusFixture.Pcm.Length, at, "Opus samples after the warm-ups");

                var mono = new float[n];
                for (int i = 0; i < n; i++) mono[i] = pcmF[2 * i];
                ReadOnlyMemory<float> block = mono.AsMemory(0, vorbisOptions.BlockSize);
                var vorbisPackets = await vorbisEnc.EncodeAudioPacketsBatchAsync(new[] { block, block, block });
                await vorbisEnc.EncodeAudioPacketAsync(block);
                await vorbisEnc.EncodeStreamAsync(mono);
                foreach (var packet in vorbisPackets) await vorbisDec.DecodePacketAsync(packet);
            });
            Equal(0L, after, "kernels compiled by work after the warm-ups");

            var imageWarmUps = new (string Name, Func<Task> WarmUp)[]
            {
                ("BmpDecoderGpu", bmpDec.WarmUpAsync), ("BmpEncoderGpu", bmpEnc.WarmUpAsync),
                ("GifDecoderGpu", gifDec.WarmUpAsync), ("GifEncoderGpu", gifEnc.WarmUpAsync),
                ("JpegDecoderGpu", jpegDec.WarmUpAsync), ("JpegEncoderGpu", jpegEnc.WarmUpAsync),
                ("PngDecoderGpu", pngDec.WarmUpAsync), ("PngEncoderGpu", pngEnc.WarmUpAsync),
                ("WebPDecoderGpu", webpDec.WarmUpAsync), ("WebPLosslessEncoderGpu", webpLossless.WarmUpAsync),
                ("WebPLossyEncoderGpu", webpLossy.WarmUpAsync),
                ("BcnEncoderGpu", () => bcnEnc.WarmUpAsync()), ("BcnDecoderGpu", bcnDec.WarmUpAsync),
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
            // per step, so a failure names the codec whose warm-up missed a kernel
            var imageSteps = new List<string>();
            async Task Step(string name, Func<Task> work)
            {
                long n = await Compiled(work);
                if (n != 0) imageSteps.Add($"{name}: {n}");
            }
            byte[] bmp = null!, gif = null!, jpeg = null!, png = null!, lossless = null!, lossy = null!;
            await Step("BmpEncoderGpu.EncodeAsync", async () => bmp = await bmpEnc.EncodeAsync(rgba, w, h));
            await Step("BmpDecoderGpu.DecodeAsync", async () =>
                True((await bmpDec.DecodeAsync(bmp)).Pixels.AsSpan().SequenceEqual(rgba), "BMP round trip after the warm-ups is not lossless"));
            await Step("GifEncoderGpu.EncodeAsync", async () => gif = await gifEnc.EncodeAsync(rgba, w, h));
            await Step("GifDecoderGpu.DecodeAsync", async () => await gifDec.DecodeAsync(gif));
            await Step("JpegEncoderGpu.EncodeAsync", async () => jpeg = await jpegEnc.EncodeAsync(rgba, w, h));
            await Step("JpegDecoderGpu.DecodeAsync", async () => await jpegDec.DecodeAsync(jpeg));
            await Step("PngEncoderGpu.EncodeAsync", async () => png = await pngEnc.EncodeAsync(rgba, w, h));
            await Step("PngDecoderGpu.DecodeAsync", async () =>
                True((await pngDec.DecodeAsync(png)).Pixels.AsSpan().SequenceEqual(rgba), "PNG round trip after the warm-ups is not lossless"));
            await Step("WebPLosslessEncoderGpu.EncodeAsync", async () => lossless = await webpLossless.EncodeAsync(rgba, w, h));
            await Step("WebPDecoderGpu.DecodeAsync (lossless)", async () =>
                True((await webpDec.DecodeAsync(lossless)).Pixels.AsSpan().SequenceEqual(rgba), "WebP lossless round trip after the warm-ups is not lossless"));
            await Step("WebPLossyEncoderGpu.EncodeAsync", async () => lossy = await webpLossy.EncodeAsync(rgba, w, h));
            await Step("WebPDecoderGpu.DecodeAsync (lossy)", async () => await webpDec.DecodeAsync(lossy));
            foreach (var format in new[] { BcnFormat.Bc1, BcnFormat.Bc7 })
            {
                byte[] blocks = null!;
                await Step($"BcnEncoderGpu.EncodeAsync ({format})", async () => blocks = await bcnEnc.EncodeAsync(rgba, w, h, format));
                await Step($"BcnDecoderGpu.DecodeAsync ({format})", async () => await bcnDec.DecodeAsync(blocks, format, w, h));
            }
            True(imageSteps.Count == 0, $"{acc.AcceleratorType}: kernels compiled by image work after the warm-ups: " + string.Join(", ", imageSteps));

            // ShuffleFilterGpu first: WeightPackGpu's warm-up loads the shuffle kernels too (cache hits then)
            var compressorWarmUps = new (string Name, Func<Task> WarmUp)[]
            {
                ("ShuffleFilterGpu", shuffle.WarmUpAsync), ("WeightPackGpu", weightPack.WarmUpAsync),
                ("GDeflateCompressorGpu", gdeflateEnc.WarmUpAsync), ("GDeflateDecompressorGpu", gdeflateDec.WarmUpAsync),
                ("Lz4FrameCompressorGpu", lz4Enc.WarmUpAsync), ("Lz4FrameDecompressorGpu", lz4Dec.WarmUpAsync),
                ("SnappyCompressorGpu", snappyEnc.WarmUpAsync), ("SnappyDecompressorGpu", snappyDec.WarmUpAsync),
            };
            foreach (var (name, warmUp) in compressorWarmUps)
                True(await Compiled(warmUp) > 0, $"{name}.WarmUpAsync compiled no kernel");

            // real work after the warm-ups: every filter and compressor both ways on 100 KB of float-like data
            var data = new byte[100_000];
            for (int i = 0; i < data.Length; i += 4)
                BitConverter.GetBytes((float)Math.Sin(i * 0.0013) * 3f).CopyTo(data, i);
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            var zipSteps = new List<string>();
            async Task Zip(string name, Func<Task<byte[]>> roundTrip)
            {
                byte[] back = null!;
                long n = await Compiled(async () => back = await roundTrip());
                if (n != 0) zipSteps.Add($"{name}: {n}");
                True(back.AsSpan().SequenceEqual(data), $"{name} round trip after the warm-ups is not lossless");
            }
            await Zip("ShuffleFilterGpu (byte)", async () => await shuffle.UnshuffleAsync(await shuffle.ShuffleAsync(data, 4), 4));
            await Zip("ShuffleFilterGpu (bit)", async () => await shuffle.BitUnshuffleAsync(await shuffle.BitShuffleAsync(data, 4), 4));
            await Zip("WeightPackGpu", async () => await weightPack.UnpackAsync(await weightPack.PackAsync(data, 4)));
            await Zip("GDeflate", async () => await gdeflateDec.DecompressAsync(await gdeflateEnc.CompressAsync(data)));
            await Zip("LZ4 frame", async () => await lz4Dec.DecompressAsync(await lz4Enc.CompressAsync(data)));
            foreach (var format in new[] { SnappyFormat.Raw, SnappyFormat.Framed })
                await Zip($"Snappy ({format})", async () => await snappyDec.DecompressAsync(await snappyEnc.CompressAsync(data, format), format));
            True(zipSteps.Count == 0, $"{acc.AcceleratorType}: kernels compiled by compressor work after the warm-ups: " + string.Join(", ", zipSteps));
        }
        finally
        {
            acc.Dispose();
            ctx.Dispose();
        }
    }
}
