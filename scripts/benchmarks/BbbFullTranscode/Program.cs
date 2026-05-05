// BBB full-clip transcode test: source MP4 -> VP8 / VP9 / AV1 with FLAC
// audio, both via SpawnDev.Codecs GPU encoders and ffmpeg side-by-side
// at the same display resolution.
//
// Output dir: V:\Video\_CodecsTest
//   bbb_vp8_gpu.mkv     (VP8 GPU + FLAC GPU, muxed)
//   bbb_vp9_gpu.mkv     (VP9 GPU + FLAC GPU, muxed)
//   bbb_av1_gpu.mkv     (AV1 GPU + FLAC GPU, muxed)
//   bbb_vp8_ffmpeg.mkv  (libvpx + flac)
//   bbb_vp9_ffmpeg.mkv  (libvpx-vp9 + flac)
//   bbb_av1_ffmpeg.mkv  (libaom-av1 + flac)
//   timings.txt         (per-codec wall-clock breakdown)

using System.Diagnostics;
using System.Text;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.Codecs.Video.Vp9;

string source = args.Length >= 1 ? args[0] : @"V:\Video\Big Buck Bunny - FULL HD 60FPS.mkv";
string outDir = args.Length >= 2 ? args[1] : @"V:\Video\_CodecsTest";
int chunkFrames = args.Length >= 3 && int.TryParse(args[2], out var cf) ? cf : 60;
int maxFrames = args.Length >= 4 && int.TryParse(args[3], out var mf) ? mf : int.MaxValue;
// Optional "WxH" override - forces ffmpeg pipe to scale source to that
// resolution before feeding the GPU encoders. Useful for isolating
// pad-related vs cap-related issues at non-aligned dims.
string? overrideRes = args.Length >= 5 ? args[4] : null;

string ffmpeg = @"C:\Users\TJ\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe";
string ffprobe = @"C:\Users\TJ\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffprobe.exe";

// Auto-flush stdout so progress appears immediately when piped/captured.
Console.SetOut(new System.IO.StreamWriter(Console.OpenStandardOutput()) { AutoFlush = true });

if (!File.Exists(source)) { Console.Error.WriteLine($"Source not found: {source}"); return 1; }
Directory.CreateDirectory(outDir);

Console.WriteLine($"Source: {source}");
Console.WriteLine($"Output: {outDir}");
Console.WriteLine($"Chunk:  {chunkFrames} frames per GPU batch");

// === 1. Probe ===
var probe = ProbeVideo(source);
int width = probe.Width, height = probe.Height, frameCount = Math.Min(probe.FrameCount, maxFrames), fps = probe.Fps;
int sampleRate = probe.SampleRate, channels = probe.Channels;
// Override target resolution if requested (forces ffmpeg pipe -s WxH).
if (!string.IsNullOrEmpty(overrideRes))
{
    var parts = overrideRes.Split('x');
    if (parts.Length == 2 && int.TryParse(parts[0], out var ow) && int.TryParse(parts[1], out var oh))
    {
        width = ow; height = oh;
        Console.WriteLine($"Resolution override: {width}x{height}");
    }
}
Console.WriteLine($"Video: {width}x{height} @ {fps}fps, {frameCount} frames");
Console.WriteLine($"Audio: {sampleRate}Hz x{channels}");

var report = new StringBuilder();
report.AppendLine("BBB full-clip transcode timings");
report.AppendLine($"Source: {source}");
report.AppendLine($"Video: {width}x{height} @ {fps}fps, {frameCount} frames, {(double)frameCount / fps:F1}s");
report.AppendLine($"Audio: {sampleRate}Hz x{channels}");
report.AppendLine($"Chunk: {chunkFrames} frames per GPU batch");
report.AppendLine();

// === 2. Set up accelerator ===
var swCtx = Stopwatch.StartNew();
using var ctx = Context.Create(b => b.AllAccelerators().EnableAlgorithms());
Accelerator? acc = null;
foreach (var d in ctx.Devices)
{
    if (d.AcceleratorType == AcceleratorType.Cuda)
    {
        acc = d.CreateAccelerator(ctx);
        break;
    }
}
if (acc == null)
{
    foreach (var d in ctx.Devices)
    {
        if (d.AcceleratorType == AcceleratorType.CPU)
        {
            acc = d.CreateAccelerator(ctx);
            break;
        }
    }
}
if (acc == null) { Console.Error.WriteLine("No accelerator found."); return 1; }
swCtx.Stop();
string backend = acc is CudaAccelerator ca ? $"CUDA ({ca.Name})" : "CPU";
Console.WriteLine($"Backend: {backend}  (init {swCtx.Elapsed.TotalSeconds:F2}s)");
report.AppendLine($"Backend: {backend}");
report.AppendLine();

// === 3. Encode audio via GPU FLAC (lossless) ===
var swAudio = Stopwatch.StartNew();
byte[] flacBytes;
{
    Console.WriteLine("Extracting + GPU-encoding FLAC audio...");
    int totalSamplesPerChannel = (int)Math.Floor((double)frameCount * sampleRate / fps);
    // Round to FLAC v1 block size (4096) - drop trailing samples that don't fit a full block.
    int roundedPerChannel = (totalSamplesPerChannel / FlacEncoderGpu.BlockSize) * FlacEncoderGpu.BlockSize;
    int interleavedLen = roundedPerChannel * channels;
    var pcm = new int[interleavedLen];

    var swDecode = Stopwatch.StartNew();
    using (var ff = StartFfmpegPipe(ffmpeg,
        $"-i \"{source}\" -ac {channels} -ar {sampleRate} -t {(double)roundedPerChannel / sampleRate} -f s16le -acodec pcm_s16le -"))
    {
        // Read interleaved s16, convert to int.
        var buf = new byte[Math.Min(1 << 20, interleavedLen * 2)];
        int wIdx = 0;
        while (wIdx < interleavedLen)
        {
            int want = Math.Min(buf.Length, (interleavedLen - wIdx) * 2);
            int got = ReadFully(ff.StandardOutput.BaseStream, buf, 0, want);
            if (got == 0) break;
            for (int i = 0; i + 1 < got; i += 2)
            {
                short s = (short)(buf[i] | (buf[i + 1] << 8));
                pcm[wIdx++] = s;
            }
        }
        ff.WaitForExit();
    }
    swDecode.Stop();

    var enc = new FlacEncoderGpu(acc);
    var swFlac = Stopwatch.StartNew();
    flacBytes = await enc.EncodeStreamAsync(pcm, channels);
    swFlac.Stop();
    enc.Dispose();
    string flacPath = Path.Combine(outDir, "audio.flac");
    File.WriteAllBytes(flacPath, flacBytes);
    Console.WriteLine($"FLAC GPU: extract {swDecode.Elapsed.TotalSeconds:F2}s + encode {swFlac.Elapsed.TotalSeconds:F2}s -> {flacBytes.Length / 1024.0 / 1024.0:F1} MB");
    report.AppendLine($"FLAC GPU audio: extract {swDecode.Elapsed.TotalSeconds:F2}s + encode {swFlac.Elapsed.TotalSeconds:F2}s, {flacBytes.Length / 1024.0 / 1024.0:F1} MB");
}
swAudio.Stop();

// === 4. Encode each video codec via GPU at full resolution ===
TranscodeVideoGpu("VP8 GPU", "vp8.ivf", "VP80", encVp8: true);
TranscodeVideoGpu("VP9 GPU", "vp9.ivf", "VP90", encVp8: false, encVp9: true);
try
{
    TranscodeVideoGpu("AV1 GPU", "av1.ivf", "AV01", encVp8: false, encVp9: false, encAv1: true);
}
catch (NotSupportedException ex)
{
    Console.WriteLine($"AV1 GPU SKIPPED: {ex.Message}");
    report.AppendLine($"AV1 GPU SKIPPED: non-64-aligned dim. Walker boundary work tracked.");
}

// === 5. Mux GPU outputs into MKV ===
MuxToMkv("vp8.ivf", "audio.flac", "bbb_vp8_gpu.mkv", "VP8");
MuxToMkv("vp9.ivf", "audio.flac", "bbb_vp9_gpu.mkv", "VP9");
if (File.Exists(Path.Combine(outDir, "av1.ivf")) && new FileInfo(Path.Combine(outDir, "av1.ivf")).Length > 32)
    MuxToMkv("av1.ivf", "audio.flac", "bbb_av1_gpu.mkv", "AV1");

// === 6. ffmpeg comparable encodes (lossless / near-lossless, same frame count) ===
double clipSec = (double)frameCount / fps;
string clipDur = clipSec.ToString("F4", System.Globalization.CultureInfo.InvariantCulture);
string ffScale = (probe.Width == width && probe.Height == height) ? "" : $"-vf scale={width}:{height} ";
// VP8 has no true lossless in libvpx (-lossless 1 is VP9-only). Force low Q
// range with high target bitrate for near-source quality.
EncodeFfmpegRef("VP8 ffmpeg", "bbb_vp8_ffmpeg.mkv",
    $"-t {clipDur} {ffScale}-c:v libvpx -qmin 0 -qmax 8 -b:v 50M -auto-alt-ref 0 -quality best -c:a flac");
EncodeFfmpegRef("VP9 ffmpeg", "bbb_vp9_ffmpeg.mkv",
    $"-t {clipDur} {ffScale}-c:v libvpx-vp9 -lossless 1 -c:a flac");
EncodeFfmpegRef("AV1 ffmpeg", "bbb_av1_ffmpeg.mkv",
    $"-t {clipDur} {ffScale}-c:v libaom-av1 -crf 0 -b:v 0 -cpu-used 8 -c:a flac");

// === 7. Write report ===
File.WriteAllText(Path.Combine(outDir, "timings.txt"), report.ToString());
Console.WriteLine();
Console.WriteLine(report.ToString());
acc.Dispose();
return 0;

void TranscodeVideoGpu(string label, string outName, string fourCc,
    bool encVp8 = false, bool encVp9 = false, bool encAv1 = false)
{
    Console.WriteLine($"=== {label} @ {width}x{height} ===");
    string outPath = Path.Combine(outDir, outName);
    var swTotal = Stopwatch.StartNew();
    var swExtract = new Stopwatch();
    var swEncode = new Stopwatch();
    var swMux = new Stopwatch();
    long totalBytes = 0;

    using var fs = File.Create(outPath);
    var ivf = new IvfWriter(fs, fourCc, width, height, frameRate: (uint)fps, timeScale: 1);

    // Allocate one set of encoders and reuse across chunks.
    Vp8KeyframeEncoderGpu? vp8 = encVp8 ? new Vp8KeyframeEncoderGpu(acc) : null;
    Vp9KeyframeEncoderGpu? vp9 = encVp9 ? new Vp9KeyframeEncoderGpu(acc) : null;
    Av1KeyframeEncoderGpu? av1 = encAv1 ? new Av1KeyframeEncoderGpu(acc) : null;

    int yLen = width * height;
    int uvLen = yLen / 4;
    int frameSize = yLen + uvLen + uvLen;

    int processed = 0;
    swExtract.Start();
    string scaleArg = (probe.Width == width && probe.Height == height)
        ? ""
        : $"-vf scale={width}:{height} ";
    using var ff = StartFfmpegPipe(ffmpeg,
        $"-i \"{source}\" {scaleArg}-f rawvideo -pix_fmt yuv420p -");
    var stdout = ff.StandardOutput.BaseStream;
    swExtract.Stop();

    while (processed < frameCount)
    {
        int batch = Math.Min(chunkFrames, frameCount - processed);
        var yArr = new byte[batch][];
        var uArr = new byte[batch][];
        var vArr = new byte[batch][];
        var yMem = new ReadOnlyMemory<byte>[batch];
        var uMem = new ReadOnlyMemory<byte>[batch];
        var vMem = new ReadOnlyMemory<byte>[batch];

        swExtract.Start();
        for (int i = 0; i < batch; i++)
        {
            var frameBuf = new byte[frameSize];
            int got = ReadFully(stdout, frameBuf, 0, frameSize);
            if (got != frameSize)
            {
                batch = i;
                break;
            }
            var y = new byte[yLen];
            var u = new byte[uvLen];
            var v = new byte[uvLen];
            Array.Copy(frameBuf, 0, y, 0, yLen);
            Array.Copy(frameBuf, yLen, u, 0, uvLen);
            Array.Copy(frameBuf, yLen + uvLen, v, 0, uvLen);
            yArr[i] = y; uArr[i] = u; vArr[i] = v;
            yMem[i] = y; uMem[i] = u; vMem[i] = v;
        }
        swExtract.Stop();
        if (batch == 0) break;

        if (yMem.Length != batch)
        {
            Array.Resize(ref yMem, batch);
            Array.Resize(ref uMem, batch);
            Array.Resize(ref vMem, batch);
        }

        swEncode.Start();
        byte[][] encoded;
        if (vp8 != null) encoded = vp8.EncodeKeyFramesBatch(yMem, uMem, vMem, width, height, baseQIndex: 4);
        else if (vp9 != null) encoded = vp9.EncodeKeyFramesBatchAsync(yMem, uMem, vMem, width, height, baseQIndex: 4).GetAwaiter().GetResult();
        else encoded = av1!.EncodeKeyFramesBatchAsync(yMem, uMem, vMem, width, height, baseQIndex: 4).GetAwaiter().GetResult();
        swEncode.Stop();

        swMux.Start();
        for (int i = 0; i < batch; i++)
        {
            ivf.WriteFrame(encoded[i], processed + i);
            totalBytes += encoded[i].Length;
        }
        swMux.Stop();

        processed += batch;
        if (processed % (chunkFrames * 10) == 0 || processed == frameCount)
        {
            double progress = 100.0 * processed / frameCount;
            double encFps = processed / swEncode.Elapsed.TotalSeconds;
            Console.WriteLine($"  {processed}/{frameCount} ({progress:F1}%)  enc {encFps:F0} fps");
        }
    }

    ivf.Finish();
    fs.Close();
    vp8?.Dispose();
    vp9?.Dispose();
    av1?.Dispose();

    if (!ff.HasExited) { try { ff.Kill(); } catch { } }
    swTotal.Stop();
    Console.WriteLine($"  TOTAL  extract {swExtract.Elapsed.TotalSeconds:F2}s  encode {swEncode.Elapsed.TotalSeconds:F2}s  mux {swMux.Elapsed.TotalSeconds:F2}s  WALL {swTotal.Elapsed.TotalSeconds:F2}s  size {totalBytes / 1024.0 / 1024.0:F1} MB");
    report.AppendLine($"{label}: extract {swExtract.Elapsed.TotalSeconds:F2}s + encode {swEncode.Elapsed.TotalSeconds:F2}s + mux {swMux.Elapsed.TotalSeconds:F2}s = WALL {swTotal.Elapsed.TotalSeconds:F2}s, {totalBytes / 1024.0 / 1024.0:F1} MB");
}

void MuxToMkv(string videoIvf, string audioFile, string outMkv, string codecLabel)
{
    string vp = Path.Combine(outDir, videoIvf);
    string ap = Path.Combine(outDir, audioFile);
    string op = Path.Combine(outDir, outMkv);
    var sw = Stopwatch.StartNew();
    int rc = RunFfmpeg($"-y -i \"{vp}\" -i \"{ap}\" -c copy \"{op}\"");
    sw.Stop();
    long sz = File.Exists(op) ? new FileInfo(op).Length : 0;
    Console.WriteLine($"Mux {codecLabel} -> {outMkv}: {sw.Elapsed.TotalSeconds:F2}s ({sz / 1024.0 / 1024.0:F1} MB) rc={rc}");
    report.AppendLine($"Mux {codecLabel} -> {outMkv}: {sw.Elapsed.TotalSeconds:F2}s, {sz / 1024.0 / 1024.0:F1} MB");
}

void EncodeFfmpegRef(string label, string outName, string codecArgs)
{
    Console.WriteLine($"=== {label} (lossless) ===");
    string op = Path.Combine(outDir, outName);
    var sw = Stopwatch.StartNew();
    int rc = RunFfmpeg($"-y -i \"{source}\" {codecArgs} \"{op}\"");
    sw.Stop();
    long sz = File.Exists(op) ? new FileInfo(op).Length : 0;
    Console.WriteLine($"  WALL {sw.Elapsed.TotalSeconds:F2}s  size {sz / 1024.0 / 1024.0:F1} MB  rc={rc}");
    report.AppendLine($"{label}: WALL {sw.Elapsed.TotalSeconds:F2}s, {sz / 1024.0 / 1024.0:F1} MB");
}

(int Width, int Height, int FrameCount, int Fps, int SampleRate, int Channels) ProbeVideo(string path)
{
    var psi = new ProcessStartInfo(ffprobe, $"-v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames -of default=noprint_wrappers=1:nokey=1 \"{path}\"")
    { RedirectStandardOutput = true, UseShellExecute = false };
    using var p = Process.Start(psi)!;
    var lines = p.StandardOutput.ReadToEnd().Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
    p.WaitForExit();
    int w = int.Parse(lines[0]);
    int h = int.Parse(lines[1]);
    int fps;
    {
        var fr = lines[2].Split('/');
        fps = (int)Math.Round(double.Parse(fr[0]) / double.Parse(fr[1]));
    }
    int frames = int.Parse(lines[3]);

    var psi2 = new ProcessStartInfo(ffprobe, $"-v error -select_streams a:0 -show_entries stream=sample_rate,channels -of default=noprint_wrappers=1:nokey=1 \"{path}\"")
    { RedirectStandardOutput = true, UseShellExecute = false };
    using var p2 = Process.Start(psi2)!;
    var aLines = p2.StandardOutput.ReadToEnd().Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
    p2.WaitForExit();
    int sr = int.Parse(aLines[0]);
    int ch = int.Parse(aLines[1]);
    return (w, h, frames, fps, sr, ch);
}

Process StartFfmpegPipe(string exe, string args)
{
    var psi = new ProcessStartInfo(exe, args)
    {
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        UseShellExecute = false,
        CreateNoWindow = true,
    };
    var p = Process.Start(psi)!;
    // Drain stderr async so the pipe doesn't block.
    _ = Task.Run(() => p.StandardError.ReadToEnd());
    return p;
}

int RunFfmpeg(string args)
{
    var psi = new ProcessStartInfo(ffmpeg, args)
    {
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        UseShellExecute = false,
        CreateNoWindow = true,
    };
    using var p = Process.Start(psi)!;
    _ = Task.Run(() => p.StandardOutput.ReadToEnd());
    _ = Task.Run(() => p.StandardError.ReadToEnd());
    p.WaitForExit();
    return p.ExitCode;
}

int ReadFully(Stream s, byte[] buf, int off, int n)
{
    int total = 0;
    while (total < n)
    {
        int got = s.Read(buf, off + total, n - total);
        if (got <= 0) break;
        total += got;
    }
    return total;
}
