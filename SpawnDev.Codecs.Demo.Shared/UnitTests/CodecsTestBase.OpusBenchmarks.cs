using Concentus.Enums;
using System.Diagnostics;
using SpawnDev.UnitTesting;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Cross-backend Opus encode/decode benchmarks using Concentus as the reference codec.
/// Measures throughput in packets/sec and samples/sec for representative scenarios
/// (SILK-NB 20ms, Hybrid-FB 20ms stereo, CELT-FB 20ms stereo) across every backend
/// test class the suite runs on.
///
/// Phase 1a state: these benchmarks measure CONCENTUS, not SpawnDev.Codecs, because
/// SpawnDev.Codecs' decode path is still being built out. When our decoder is complete,
/// we'll add "SpawnDev.Codecs decode" time alongside the Concentus number so both run
/// and get reported side-by-side in the same run.
///
/// Results are printed via Console.WriteLine; PlaywrightMultiTest captures that output
/// per-test and routes it to the latest.json / test-run-*.json sidecar files.
/// </summary>
public abstract partial class CodecsTestBase
{
    /// <summary>Runs the given action <paramref name="iterations"/> times after a warmup pass and reports elapsed ms.</summary>
    private static BenchmarkResult RunBenchmark(string label, int iterations, Action work)
    {
        // Warmup: ensure any JIT / cache / allocation cost is out of the measured window.
        for (int i = 0; i < 10; i++) work();

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++) work();
        sw.Stop();

        double totalMs = sw.Elapsed.TotalMilliseconds;
        double opsPerSec = iterations * 1000.0 / Math.Max(totalMs, 0.0001);
        var result = new BenchmarkResult
        {
            Label = label,
            Iterations = iterations,
            TotalMs = totalMs,
            OpsPerSec = opsPerSec,
            UsPerOp = totalMs * 1000.0 / iterations,
        };

        Console.WriteLine(
            $"[BENCH] {label}: {iterations} iters in {totalMs:F2} ms " +
            $"({opsPerSec:F0} ops/sec, {result.UsPerOp:F2} us/op)");
        return result;
    }

    private sealed class BenchmarkResult
    {
        public required string Label { get; init; }
        public required int Iterations { get; init; }
        public required double TotalMs { get; init; }
        public required double OpsPerSec { get; init; }
        public required double UsPerOp { get; init; }
    }

    // -------- Concentus encode benchmarks --------

    [TestMethod]
    public void Benchmark_Concentus_Encode_SilkNb20ms_Mono()
    {
        var pcm = ReferenceOracle.GenerateSineWave(440, 8000, 1, 160);
        RunBenchmark("Concentus encode SILK-NB-20ms-mono", 200, () =>
        {
            var bytes = ReferenceOracle.EncodeFrame(pcm, 8000, 1, 160, OpusApplication.OPUS_APPLICATION_VOIP);
            _ = bytes.Length; // prevent DCE
        });
    }

    [TestMethod]
    public void Benchmark_Concentus_Encode_SilkWb20ms_Mono()
    {
        var pcm = ReferenceOracle.GenerateSineWave(440, 16000, 1, 320);
        RunBenchmark("Concentus encode SILK-WB-20ms-mono", 200, () =>
        {
            var bytes = ReferenceOracle.EncodeFrame(pcm, 16000, 1, 320, OpusApplication.OPUS_APPLICATION_VOIP);
            _ = bytes.Length;
        });
    }

    [TestMethod]
    public void Benchmark_Concentus_Encode_HybridFb20ms_Stereo()
    {
        var pcm = ReferenceOracle.GenerateSineWave(440, 48000, 2, 960);
        RunBenchmark("Concentus encode Hybrid-FB-20ms-stereo", 100, () =>
        {
            var bytes = ReferenceOracle.EncodeFrame(pcm, 48000, 2, 960, OpusApplication.OPUS_APPLICATION_VOIP);
            _ = bytes.Length;
        });
    }

    [TestMethod]
    public void Benchmark_Concentus_Encode_CeltFb20ms_Stereo()
    {
        var pcm = ReferenceOracle.GenerateSineWave(440, 48000, 2, 960);
        RunBenchmark("Concentus encode CELT-FB-20ms-stereo", 100, () =>
        {
            var bytes = ReferenceOracle.EncodeFrame(pcm, 48000, 2, 960, OpusApplication.OPUS_APPLICATION_RESTRICTED_LOWDELAY);
            _ = bytes.Length;
        });
    }

    // -------- Concentus decode benchmarks --------

    [TestMethod]
    public void Benchmark_Concentus_Decode_SilkNb20ms_Mono()
    {
        var pcm = ReferenceOracle.GenerateSineWave(440, 8000, 1, 160);
        var opusBytes = ReferenceOracle.EncodeFrame(pcm, 8000, 1, 160, OpusApplication.OPUS_APPLICATION_VOIP);

        RunBenchmark("Concentus decode SILK-NB-20ms-mono", 500, () =>
        {
            var decoded = ReferenceOracle.DecodePacket(opusBytes, 8000, 1);
            _ = decoded.Length;
        });
    }

    [TestMethod]
    public void Benchmark_Concentus_Decode_SilkWb20ms_Mono()
    {
        var pcm = ReferenceOracle.GenerateSineWave(440, 16000, 1, 320);
        var opusBytes = ReferenceOracle.EncodeFrame(pcm, 16000, 1, 320, OpusApplication.OPUS_APPLICATION_VOIP);

        RunBenchmark("Concentus decode SILK-WB-20ms-mono", 500, () =>
        {
            var decoded = ReferenceOracle.DecodePacket(opusBytes, 16000, 1);
            _ = decoded.Length;
        });
    }

    [TestMethod]
    public void Benchmark_Concentus_Decode_HybridFb20ms_Stereo()
    {
        var pcm = ReferenceOracle.GenerateSineWave(440, 48000, 2, 960);
        var opusBytes = ReferenceOracle.EncodeFrame(pcm, 48000, 2, 960, OpusApplication.OPUS_APPLICATION_VOIP);

        RunBenchmark("Concentus decode Hybrid-FB-20ms-stereo", 300, () =>
        {
            var decoded = ReferenceOracle.DecodePacket(opusBytes, 48000, 2);
            _ = decoded.Length;
        });
    }

    [TestMethod]
    public void Benchmark_Concentus_Decode_CeltFb20ms_Stereo()
    {
        var pcm = ReferenceOracle.GenerateSineWave(440, 48000, 2, 960);
        var opusBytes = ReferenceOracle.EncodeFrame(pcm, 48000, 2, 960, OpusApplication.OPUS_APPLICATION_RESTRICTED_LOWDELAY);

        RunBenchmark("Concentus decode CELT-FB-20ms-stereo", 300, () =>
        {
            var decoded = ReferenceOracle.DecodePacket(opusBytes, 48000, 2);
            _ = decoded.Length;
        });
    }

    // -------- SpawnDev.Codecs benchmarks for paths that ARE complete --------

    [TestMethod]
    public void Benchmark_SpawnDevCodecs_RangeCoder_RoundTrip_16Symbols()
    {
        // Range coder IS complete; benchmark 16 encode-then-decode ops.
        var icdf = new byte[] { 200, 150, 100, 50, 0 };
        int[] symbols = { 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1 };

        RunBenchmark("SpawnDev.Codecs range-coder 16-symbol encode+decode", 1000, () =>
        {
            var enc = new SpawnDev.Codecs.EntropyCoders.OpusRangeEncoder(64);
            foreach (int s in symbols) enc.EncodeIcdf(s, icdf, 8);
            enc.Done();
            var dec = new SpawnDev.Codecs.EntropyCoders.OpusRangeDecoder(enc.ToArray());
            for (int i = 0; i < symbols.Length; i++) _ = dec.DecodeIcdf(icdf, 8);
        });
    }

    [TestMethod]
    public void Benchmark_SpawnDevCodecs_OpusPacketParser_SimplePacket()
    {
        // Packet parsing IS complete; benchmark a simple count-code-0 packet.
        byte[] packet = new byte[] { 0x00, 0x11, 0x22, 0x33, 0x44, 0x55 }; // 5-byte SILK-NB-10ms frame
        var memory = packet.AsMemory();

        RunBenchmark("SpawnDev.Codecs OpusPacketParser count-0 5-byte payload", 10000, () =>
        {
            var parsed = SpawnDev.Codecs.Audio.Opus.OpusPacketParser.Parse(memory);
            _ = parsed.FrameCount;
        });
    }
}
