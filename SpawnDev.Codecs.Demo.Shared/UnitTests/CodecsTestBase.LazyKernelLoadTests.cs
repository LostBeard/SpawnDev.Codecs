// Kernel loading of the codecs (FLAC, MP3, Opus, VP9 decode): constructors compile nothing, WarmUpAsync compiles every
// kernel the codec uses (awaited loads, so a browser page keeps running), and work after the warm-up compiles nothing more.
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
        }
        finally
        {
            acc.Dispose();
            ctx.Dispose();
        }
    }
}
