// Unified benchmark for every SpawnDev.Codecs encoder + decoder.
// Runs synthetic workloads through each codec and reports throughput
// (samples/s, frames/s, MB/s) + output size + (where applicable)
// reference-encoder comparison via ffmpeg.
//
// Usage: dotnet run benchmark_all_codecs.cs [iterations]

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.Codecs.Video.Vp9;

int iterations = args.Length >= 1 && int.TryParse(args[0], out int n) ? n : 50;
string ffmpeg = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
if (!File.Exists(ffmpeg)) ffmpeg = "ffmpeg";

var report = new StringBuilder();
report.AppendLine("============================================================");
report.AppendLine("  SpawnDev.Codecs - Unified Benchmark Suite");
report.AppendLine("============================================================");
report.AppendLine($"  Iterations per measurement: {iterations}");
report.AppendLine($"  Date: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss}Z");
report.AppendLine();

// =================================================================
// VIDEO ENCODERS - throughput at multiple frame sizes + Q levels
// =================================================================
report.AppendLine("VIDEO ENCODERS (synthetic gradient YUV420)");
report.AppendLine("------------------------------------------");
report.AppendLine($"{"Codec",-10}{"WxH",-10}{"Q",-5}{"Frames/s",-12}{"avg B",-10}{"MB/s in",-10}");
report.AppendLine($"{new string('-', 10)}{new string('-', 10)}{new string('-', 5)}{new string('-', 12)}{new string('-', 10)}{new string('-', 10)}");

foreach (var (W, H) in new[] { (16, 16), (32, 32), (64, 64) })
{
    var ySrc = new byte[W * H];
    var uSrc = new byte[(W / 2) * (H / 2)];
    var vSrc = new byte[(W / 2) * (H / 2)];
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            ySrc[r * W + c] = (byte)Math.Clamp(96 + 32 * Math.Sin(2.0 * Math.PI * c / 16.0) + r * 2, 0, 255);
    for (int i = 0; i < uSrc.Length; i++) { uSrc[i] = (byte)(120 + i % 16); vSrc[i] = (byte)(136 - i % 16); }
    int rawBytes = W * H + 2 * (W / 2) * (H / 2);

    foreach (int q in new[] { 30, 100, 200 })
    {
        // VP8
        Vp8KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: q); // warmup
        var sw = Stopwatch.StartNew();
        long total = 0;
        for (int i = 0; i < iterations; i++)
            total += Vp8KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: q).Length;
        sw.Stop();
        Append("VP8", W, H, q, iterations, sw.Elapsed.TotalSeconds, total, rawBytes);

        // VP9
        Vp9KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: q);
        sw.Restart();
        total = 0;
        for (int i = 0; i < iterations; i++)
            total += Vp9KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: q).Length;
        sw.Stop();
        Append("VP9", W, H, q, iterations, sw.Elapsed.TotalSeconds, total, rawBytes);

        // AV1 (uses qindex 32-200; clamp)
        int av1Q = Math.Clamp(q, 1, 255);
        Av1KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: av1Q);
        sw.Restart();
        total = 0;
        for (int i = 0; i < iterations; i++)
            total += Av1KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: av1Q).Length;
        sw.Stop();
        Append("AV1", W, H, av1Q, iterations, sw.Elapsed.TotalSeconds, total, rawBytes);
    }
    report.AppendLine();
}

// =================================================================
// AUDIO ENCODERS - throughput + compression
// =================================================================
report.AppendLine("AUDIO ENCODERS");
report.AppendLine("--------------");
report.AppendLine($"{"Codec",-10}{"Format",-15}{"Realtime",-12}{"Ratio",-8}");
report.AppendLine($"{new string('-', 10)}{new string('-', 15)}{new string('-', 12)}{new string('-', 8)}");

// FLAC: 30 seconds of stereo 16-bit @ 44.1 kHz
{
    int sr = 44100, ch = 2, secs = 30;
    int totalSamples = sr * secs;
    var samples = new int[totalSamples * ch];
    var rng = new Random(42);
    for (int i = 0; i < totalSamples; i++)
    {
        samples[i * 2 + 0] = (int)(0.4 * 32767 * Math.Sin(2.0 * Math.PI * 440 * i / sr)) + rng.Next(-50, 50);
        samples[i * 2 + 1] = (int)(0.4 * 32767 * Math.Sin(2.0 * Math.PI * 880 * i / sr)) + rng.Next(-50, 50);
    }
    int rawBytes = totalSamples * ch * 2;
    var sw = Stopwatch.StartNew();
    var enc = FlacEncoder.EncodeStream(samples, sr, ch, 16, blockSize: 4096);
    sw.Stop();
    double rt = secs / sw.Elapsed.TotalSeconds;
    double ratio = (double)enc.Length / rawBytes;
    report.AppendLine($"{"FLAC",-10}{"30s 44k st",-15}{rt,-12:F1}{ratio,-8:F3}");
}

// Opus: 30 seconds of mono 48 kHz
{
    int sr = 48000, secs = 30;
    int frameSize = sr / 50; // 20ms
    var pcm = new float[sr * secs];
    for (int i = 0; i < pcm.Length; i++)
        pcm[i] = 0.4f * (float)Math.Sin(2.0 * Math.PI * 440 * i / sr);
    var enc = new OpusEncoder(new OpusEncoderConfig { SampleRateHz = sr, ChannelCount = 1, Application = OpusEncoderApplication.Audio });
    var packet = new byte[1275];
    int frameCount = pcm.Length / frameSize;
    long totalEncoded = 0;
    var sw = Stopwatch.StartNew();
    for (int f = 0; f < frameCount; f++)
        totalEncoded += enc.EncodeFrame(pcm.AsSpan(f * frameSize, frameSize), packet, frameSize);
    sw.Stop();
    enc.Dispose();
    double rt = secs / sw.Elapsed.TotalSeconds;
    double ratio = (double)totalEncoded / (pcm.Length * 4); // float input
    report.AppendLine($"{"Opus",-10}{"30s 48k mn",-15}{rt,-12:F1}{ratio,-8:F4}");
}

// Vorbis: 30 seconds of mono 44.1 kHz
{
    int sr = 44100, secs = 30;
    var pcm = new float[sr * secs];
    for (int i = 0; i < pcm.Length; i++)
        pcm[i] = 0.4f * (float)Math.Sin(2.0 * Math.PI * 440 * i / sr);
    var enc = new VorbisAudioEncoder(new VorbisAudioEncoderOptions { SampleRateHz = sr, Channels = 1 });
    var sw = Stopwatch.StartNew();
    var ogg = enc.EncodeStream(pcm);
    sw.Stop();
    double rt = secs / sw.Elapsed.TotalSeconds;
    double ratio = (double)ogg.Length / (pcm.Length * 4);
    report.AppendLine($"{"Vorbis",-10}{"30s 44k mn",-15}{rt,-12:F1}{ratio,-8:F4}");
}
report.AppendLine();

// =================================================================
// VIDEO DECODERS - throughput on encoded keyframes
// =================================================================
report.AppendLine("VIDEO DECODERS (decode our own encoder output)");
report.AppendLine("----------------------------------------------");
report.AppendLine($"{"Codec",-10}{"WxH",-10}{"Frames/s",-12}{"MB/s out",-10}");
report.AppendLine($"{new string('-', 10)}{new string('-', 10)}{new string('-', 12)}{new string('-', 10)}");

foreach (var (W, H) in new[] { (16, 16), (32, 32), (64, 64) })
{
    var ySrc = new byte[W * H]; Array.Fill(ySrc, (byte)128);
    var uSrc = new byte[(W / 2) * (H / 2)]; Array.Fill(uSrc, (byte)128);
    var vSrc = new byte[(W / 2) * (H / 2)]; Array.Fill(vSrc, (byte)128);
    int outBytes = W * H + 2 * (W / 2) * (H / 2);

    // VP8
    var vp8Frame = Vp8KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 30);
    var vp8Sink = new NoopSink();
    { var d = new Vp8Decoder(); d.DecodeFrameAsync(vp8Frame, vp8Sink).GetAwaiter().GetResult(); d.DisposeAsync().GetAwaiter().GetResult(); } // warmup
    var sw = Stopwatch.StartNew();
    for (int i = 0; i < iterations; i++)
    {
        var d = new Vp8Decoder();
        d.DecodeFrameAsync(vp8Frame, vp8Sink).GetAwaiter().GetResult();
        d.DisposeAsync().GetAwaiter().GetResult();
    }
    sw.Stop();
    AppendDec("VP8", W, H, iterations, sw.Elapsed.TotalSeconds, outBytes);

    // VP9
    var vp9Frame = Vp9KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 30);
    var vp9Sink = new NoopSink();
    { var d = new Vp9Decoder(); d.DecodeFrameAsync(vp9Frame, vp9Sink).GetAwaiter().GetResult(); d.DisposeAsync().GetAwaiter().GetResult(); } // warmup
    sw.Restart();
    for (int i = 0; i < iterations; i++)
    {
        var d = new Vp9Decoder();
        d.DecodeFrameAsync(vp9Frame, vp9Sink).GetAwaiter().GetResult();
        d.DisposeAsync().GetAwaiter().GetResult();
    }
    sw.Stop();
    AppendDec("VP9", W, H, iterations, sw.Elapsed.TotalSeconds, outBytes);

    // AV1
    var av1Frame = Av1KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 32);
    var av1Sink = new NoopSink();
    { var d = new Av1Decoder(); d.DecodeFrameAsync(av1Frame, av1Sink).GetAwaiter().GetResult(); d.DisposeAsync().GetAwaiter().GetResult(); } // warmup
    sw.Restart();
    for (int i = 0; i < iterations; i++)
    {
        var d = new Av1Decoder();
        d.DecodeFrameAsync(av1Frame, av1Sink).GetAwaiter().GetResult();
        d.DisposeAsync().GetAwaiter().GetResult();
    }
    sw.Stop();
    AppendDec("AV1", W, H, iterations, sw.Elapsed.TotalSeconds, outBytes);

    report.AppendLine();
}

string output = report.ToString();
Console.Write(output);
string reportPath = Path.Combine(Path.GetTempPath(), "spawndev_codecs_benchmark.txt");
File.WriteAllText(reportPath, output);
Console.WriteLine($"Report saved: {reportPath}");

void Append(string codec, int W, int H, int q, int iters, double secs, long total, int rawIn)
{
    double fps = iters / secs;
    double avgB = total / (double)iters;
    double mbs = (rawIn * (double)iters) / secs / 1_000_000.0;
    report.AppendLine($"{codec,-10}{$"{W}x{H}",-10}{q,-5}{fps,-12:F1}{avgB,-10:F0}{mbs,-10:F2}");
}

void AppendDec(string codec, int W, int H, int iters, double secs, int outBytes)
{
    double fps = iters / secs;
    double mbs = (outBytes * (double)iters) / secs / 1_000_000.0;
    report.AppendLine($"{codec,-10}{$"{W}x{H}",-10}{fps,-12:F1}{mbs,-10:F2}");
}

sealed class NoopSink : SpawnDev.Codecs.Video.IVideoFrameSink
{
    public ValueTask OnFrameAsync(
        ReadOnlyMemory<byte> y, int ys,
        ReadOnlyMemory<byte> u, int us,
        ReadOnlyMemory<byte> v, int vs,
        long pts) => ValueTask.CompletedTask;
}
