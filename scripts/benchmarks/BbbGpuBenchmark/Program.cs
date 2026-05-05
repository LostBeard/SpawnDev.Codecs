// GPU-pair version of benchmark_bbb_full_transcode.cs - drives the BBB
// clip through SpawnDev.Codecs' ILGPU GPU encoders + ffmpeg side-by-side.
// Uses CUDA backend (GPU 0) by default; falls back to CPU backend if CUDA
// init fails. Same source clip, same frame count, same audio rate.
//
// Usage: dotnet run --project scripts/benchmarks/BbbGpuBenchmark -- [seconds=1]

using System.Diagnostics;
using System.Text;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.Codecs.Video.Vp9;

int seconds = args.Length >= 1 && int.TryParse(args[0], out int s) ? s : 1;
// GPU v1 encoder dimension caps (current state of the library, 2026-05-04):
//   VP8 GPU: width + height multiples of 16. No row/col cap.
//   VP9 GPU: width + height multiples of 64. mbCols ≤ 32 (width ≤ 512).
//   AV1 GPU: width + height multiples of 64. No row/col cap exposed.
// Pick the biggest resolution all three accept: 512x256 (VP9-cap-bound).
// ffmpeg references run at the same size for apples-to-apples timing.
// Lifting Vp9FrameEntropyKernel.MaxMiColsAligned would unblock larger
// frames but increases per-thread local memory; deferred to v2.
const int W = 512, H = 256, Fps = 60;
int frameCount = seconds * Fps;
string source = "V:\\Video\\Big Buck Bunny - FULL HD 60FPS.mp4";
string ffmpeg = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
string outDir = Path.Combine(Path.GetTempPath(), "spawndev_gpu_transcode");
Directory.CreateDirectory(outDir);

if (!File.Exists(source))
{
    Console.WriteLine($"Source not found: {source}");
    Environment.Exit(1);
}

// ---- Acquire accelerator (CUDA preferred, CPU fallback) ----
Context ctx;
Accelerator acc;
string backend;
try
{
    ctx = Context.Create(b => b.Cuda());
    acc = ctx.CreateCudaAccelerator(0);
    backend = "CUDA";
}
catch (Exception ex)
{
    Console.WriteLine($"CUDA init failed ({ex.Message.Split('\n')[0]}), falling back to CPU backend");
    ctx = Context.Create(b => b.CPU());
    acc = ctx.CreateCPUAccelerator(0);
    backend = "CPU (ILGPU)";
}
Console.WriteLine($"ILGPU backend: {backend} ({acc.Device.Name})");

var report = new StringBuilder();
report.AppendLine("============================================================");
report.AppendLine($"  BBB GPU Transcode Benchmark - {seconds}s ({frameCount} frames @ {W}x{H} + audio)");
report.AppendLine($"  ILGPU backend: {backend} ({acc.Device.Name})");
report.AppendLine($"  {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss}Z");
report.AppendLine("============================================================");
report.AppendLine();

// ----- Extract YUV video + raw audio (same source as CPU benchmark) -----
string yuvPath = Path.Combine(outDir, "src.yuv");
string srcStereoPcm44 = Path.Combine(outDir, "src44s.pcm");
string srcMono44 = Path.Combine(outDir, "src44m.pcm");
RunFfmpeg($"-y -i \"{source}\" -vf crop={W}:{H}:0:0 -frames:v {frameCount} -f rawvideo -pix_fmt yuv420p \"{yuvPath}\"");
RunFfmpeg($"-y -i \"{source}\" -t {seconds} -f s16le -ac 2 -ar 44100 \"{srcStereoPcm44}\"");
RunFfmpeg($"-y -i \"{source}\" -t {seconds} -f s16le -ac 1 -ar 44100 \"{srcMono44}\"");
int frameSize = W * H + 2 * (W / 2) * (H / 2);
var allFrames = File.ReadAllBytes(yuvPath);
var srcStereoBytes = File.ReadAllBytes(srcStereoPcm44);
var srcMonoBytes = File.ReadAllBytes(srcMono44);
Console.WriteLine($"Source: {frameCount} frames {W}x{H} ({allFrames.Length / 1024 / 1024}MB), audio stereo44k {srcStereoBytes.Length}B + mono44k {srcMonoBytes.Length}B");
Console.WriteLine();

report.AppendLine("VIDEO ENCODERS (full BBB clip)");
report.AppendLine("  'frame 0 ms' = first-frame cost (kernel JIT + buffer alloc + dispatch + readback)");
report.AppendLine("  'steady ms/frame' = avg of frames 1..N-1 (no JIT cost; pure encode + transfer)");
report.AppendLine();
report.AppendLine($"{"Codec",-26}{"Total ms",-10}{"frame0 ms",-11}{"steady ms/f",-13}{"steady fps",-12}{"KB",-8}{"kbps",-8}");
report.AppendLine($"{new string('-', 26)}{new string('-', 10)}{new string('-', 11)}{new string('-', 13)}{new string('-', 12)}{new string('-', 8)}{new string('-', 8)}");

// ---- VP8 GPU (per-frame) ----
Console.WriteLine("Encoding VP8 GPU (per-frame)...");
{
    using var enc = new Vp8KeyframeEncoderGpu(acc);
    long total = 0;
    double frame0Ms = 0;
    var swTotal = Stopwatch.StartNew();
    for (int f = 0; f < frameCount; f++)
    {
        int yOff = f * frameSize;
        int uOff = yOff + W * H;
        int vOff = uOff + (W / 2) * (H / 2);
        var ySpan = new ReadOnlySpan<byte>(allFrames, yOff, W * H);
        var uSpan = new ReadOnlySpan<byte>(allFrames, uOff, (W / 2) * (H / 2));
        var vSpan = new ReadOnlySpan<byte>(allFrames, vOff, (W / 2) * (H / 2));
        var swFrame = Stopwatch.StartNew();
        var bytes = enc.EncodeKeyFrame(ySpan, W, uSpan, W / 2, vSpan, W, H, baseQIndex: 30);
        swFrame.Stop();
        if (f == 0) frame0Ms = swFrame.Elapsed.TotalMilliseconds;
        total += bytes.Length;
    }
    swTotal.Stop();
    AddVideo($"VP8 GPU ({backend})", swTotal.Elapsed.TotalMilliseconds, frame0Ms, total);
}

// ---- VP8 GPU (batch - submits all frames as one stream chain) ----
Console.WriteLine("Encoding VP8 GPU (batch)...");
{
    using var enc = new Vp8KeyframeEncoderGpu(acc);
    var yPlanes = new ReadOnlyMemory<byte>[frameCount];
    var uPlanes = new ReadOnlyMemory<byte>[frameCount];
    var vPlanes = new ReadOnlyMemory<byte>[frameCount];
    for (int f = 0; f < frameCount; f++)
    {
        int yOff = f * frameSize;
        int uOff = yOff + W * H;
        int vOff = uOff + (W / 2) * (H / 2);
        yPlanes[f] = new ReadOnlyMemory<byte>(allFrames, yOff, W * H);
        uPlanes[f] = new ReadOnlyMemory<byte>(allFrames, uOff, (W / 2) * (H / 2));
        vPlanes[f] = new ReadOnlyMemory<byte>(allFrames, vOff, (W / 2) * (H / 2));
    }
    var swTotal = Stopwatch.StartNew();
    var results = enc.EncodeKeyFramesBatch(yPlanes, uPlanes, vPlanes, W, H, baseQIndex: 30);
    swTotal.Stop();
    long total = 0;
    foreach (var r in results) total += r.Length;
    AddVideo($"VP8 GPU batch ({backend})", swTotal.Elapsed.TotalMilliseconds, -1, total);
}

// ---- VP9 GPU (async) ----
Console.WriteLine("Encoding VP9 GPU...");
{
    using var enc = new Vp9KeyframeEncoderGpu(acc);
    long total = 0;
    double frame0Ms = 0;
    var swTotal = Stopwatch.StartNew();
    for (int f = 0; f < frameCount; f++)
    {
        int yOff = f * frameSize;
        int uOff = yOff + W * H;
        int vOff = uOff + (W / 2) * (H / 2);
        var y = allFrames[yOff..uOff];
        var u = allFrames[uOff..vOff];
        var v = allFrames[vOff..(vOff + (W / 2) * (H / 2))];
        var swFrame = Stopwatch.StartNew();
        var bytes = await enc.EncodeKeyFrameAsync(y, u, v, W, H, baseQIndex: 30);
        swFrame.Stop();
        if (f == 0) frame0Ms = swFrame.Elapsed.TotalMilliseconds;
        total += bytes.Length;
    }
    swTotal.Stop();
    AddVideo($"VP9 GPU ({backend})", swTotal.Elapsed.TotalMilliseconds, frame0Ms, total);
}

// ---- AV1 GPU (async) ----
Console.WriteLine("Encoding AV1 GPU...");
{
    using var enc = new Av1KeyframeEncoderGpu(acc);
    long total = 0;
    double frame0Ms = 0;
    var swTotal = Stopwatch.StartNew();
    for (int f = 0; f < frameCount; f++)
    {
        int yOff = f * frameSize;
        int uOff = yOff + W * H;
        int vOff = uOff + (W / 2) * (H / 2);
        var y = allFrames[yOff..uOff];
        var u = allFrames[uOff..vOff];
        var v = allFrames[vOff..(vOff + (W / 2) * (H / 2))];
        var swFrame = Stopwatch.StartNew();
        var bytes = await enc.EncodeKeyFrameAsync(y, u, v, W, H, baseQIndex: 32);
        swFrame.Stop();
        if (f == 0) frame0Ms = swFrame.Elapsed.TotalMilliseconds;
        total += bytes.Length;
    }
    swTotal.Stop();
    AddVideo($"AV1 GPU ({backend})", swTotal.Elapsed.TotalMilliseconds, frame0Ms, total);
}

// ---- ffmpeg references on the same YUV ----
Console.WriteLine("Encoding ffmpeg refs...");
// ffmpeg encodes the whole clip in one process invocation, so frame0/steady
// breakdown isn't directly comparable. Report total ms only for ffmpeg rows.
TimeFfmpegVideo("VP8 (ffmpeg)", $"-c:v libvpx -keyint_min 1 -g 1 -auto-alt-ref 0 -frames:v {frameCount}", "ff_vp8.ivf");
TimeFfmpegVideo("VP9 (ffmpeg)", $"-c:v libvpx-vp9 -keyint_min 1 -g 1 -frames:v {frameCount}", "ff_vp9.ivf");
TimeFfmpegVideo("AV1 (ffmpeg)", $"-c:v libaom-av1 -cpu-used 8 -keyint_min 1 -g 1 -frames:v {frameCount}", "ff_av1.ivf");

report.AppendLine();
report.AppendLine("AUDIO ENCODERS (same source clip)");
report.AppendLine($"{"Codec",-26}{"Encode ms",-12}{"Output KB",-12}{"Realtime",-10}");
report.AppendLine($"{new string('-', 26)}{new string('-', 12)}{new string('-', 12)}{new string('-', 10)}");

// ---- FLAC GPU (44k stereo, interleaved int) ----
Console.WriteLine("Encoding FLAC GPU...");
{
    int totalPerCh = srcStereoBytes.Length / 4; // 2 ch × 2 B/sample
    int blockSize = 4096;
    int padded = ((totalPerCh + blockSize - 1) / blockSize) * blockSize;
    var samples = new int[padded * 2];
    for (int i = 0; i < totalPerCh; i++)
    {
        samples[i * 2] = (short)(srcStereoBytes[i * 4] | (srcStereoBytes[i * 4 + 1] << 8));
        samples[i * 2 + 1] = (short)(srcStereoBytes[i * 4 + 2] | (srcStereoBytes[i * 4 + 3] << 8));
    }
    using var enc = new FlacEncoderGpu(acc);
    var sw = Stopwatch.StartNew();
    var bytes = await enc.EncodeStreamAsync(samples, channels: 2);
    sw.Stop();
    AddAudio($"FLAC GPU ({backend})", sw.Elapsed.TotalMilliseconds, bytes.Length, seconds);
}
TimeFfmpegAudio("FLAC (ffmpeg)", srcStereoPcm44, 44100, 2, "-c:a flac", "ff.flac");

// ---- Vorbis GPU (mono 44k float) ----
Console.WriteLine("Encoding Vorbis GPU...");
{
    var pcm = new float[srcMonoBytes.Length / 2];
    for (int i = 0; i < pcm.Length; i++) pcm[i] = ((short)(srcMonoBytes[i * 2] | (srcMonoBytes[i * 2 + 1] << 8))) / 32768f;
    using var enc = new VorbisAudioEncoderGpu(acc, new VorbisAudioEncoderOptions { SampleRateHz = 44100, Channels = 1 });
    var sw = Stopwatch.StartNew();
    var ogg = await enc.EncodeStreamAsync(pcm);
    sw.Stop();
    AddAudio($"Vorbis GPU ({backend})", sw.Elapsed.TotalMilliseconds, ogg.Length, seconds);
}
TimeFfmpegAudio("Vorbis (ffmpeg)", srcMono44, 44100, 1, "-c:a libvorbis", "ff.ogg");

acc.Dispose();
ctx.Dispose();

string output = report.ToString();
Console.Write(output);
string reportPath = Path.Combine(outDir, "report.txt");
File.WriteAllText(reportPath, output);
Console.WriteLine();
Console.WriteLine($"Report: {reportPath}");

void TimeFfmpegVideo(string label, string args, string outName)
{
    string outPath = Path.Combine(outDir, outName);
    var sw = Stopwatch.StartNew();
    RunFfmpeg($"-y -f rawvideo -pix_fmt yuv420p -s {W}x{H} -i \"{yuvPath}\" {args} \"{outPath}\"");
    sw.Stop();
    // ffmpeg: no frame0/steady split available - report total only.
    AddVideo(label, sw.Elapsed.TotalMilliseconds, -1, new FileInfo(outPath).Length);
}

void TimeFfmpegAudio(string label, string srcPcm, int rate, int ch, string codecArgs, string outName)
{
    string outPath = Path.Combine(outDir, outName);
    var sw = Stopwatch.StartNew();
    RunFfmpeg($"-y -f s16le -ar {rate} -ac {ch} -i \"{srcPcm}\" {codecArgs} \"{outPath}\"");
    sw.Stop();
    AddAudio(label, sw.Elapsed.TotalMilliseconds, new FileInfo(outPath).Length, seconds);
}

void AddVideo(string label, double totalMs, double frame0Ms, long sz)
{
    double kbps = sz * 8.0 / 1000.0 / seconds;
    string frame0Str = frame0Ms < 0 ? "n/a" : $"{frame0Ms:F0}";
    string steadyStr = "n/a", steadyFpsStr = "n/a";
    if (frame0Ms >= 0 && frameCount > 1)
    {
        double steadyMsTotal = totalMs - frame0Ms;
        double steadyMsPerFrame = steadyMsTotal / (frameCount - 1);
        double steadyFps = 1000.0 / steadyMsPerFrame;
        steadyStr = $"{steadyMsPerFrame:F1}";
        steadyFpsStr = $"{steadyFps:F1}";
    }
    report.AppendLine($"{label,-26}{totalMs,-10:F0}{frame0Str,-11}{steadyStr,-13}{steadyFpsStr,-12}{sz / 1024.0,-8:F1}{kbps,-8:F0}");
}

void AddAudio(string label, double encMs, long sz, int durSec)
{
    double rt = durSec * 1000.0 / encMs;
    report.AppendLine($"{label,-26}{encMs,-12:F0}{sz / 1024.0,-12:F1}{rt,-10:F1}x");
}

void RunFfmpeg(string args)
{
    var p = Process.Start(new ProcessStartInfo(ffmpeg, args) { RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true })!;
    p.StandardError.ReadToEnd();
    p.WaitForExit();
    if (p.ExitCode != 0) throw new Exception($"ffmpeg failed: {args}");
}
