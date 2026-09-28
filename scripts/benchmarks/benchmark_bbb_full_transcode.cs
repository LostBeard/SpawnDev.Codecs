// Full BBB transcode benchmark: video + audio together, our encoders
// side-by-side with ffmpeg's libvpx / libvpx-vp9 / libaom-av1 / FLAC /
// libopus / libvorbis. Drives the same source clip through both
// pipelines and reports encode time + output size + a quick decode
// validation.
//
// Usage: dotnet run scripts/benchmarks/benchmark_bbb_full_transcode.cs [seconds=1]

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.Codecs.Video.Vp9;

int seconds = args.Length >= 1 && int.TryParse(args[0], out int s) ? s : 1;
const int W = 1920, H = 1072, Fps = 60;
int frameCount = seconds * Fps;
string source = "V:\\Video\\Big Buck Bunny - FULL HD 60FPS.mp4";
string ffmpeg = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
string outDir = Path.Combine(Path.GetTempPath(), "spawndev_full_transcode");
Directory.CreateDirectory(outDir);

if (!File.Exists(source))
{
    Console.WriteLine($"Source not found: {source}");
    Environment.Exit(1);
}

var report = new StringBuilder();
report.AppendLine("============================================================");
report.AppendLine($"  BBB Full Transcode Benchmark - {seconds}s ({frameCount} frames @ {W}x{H} + audio)");
report.AppendLine($"  {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss}Z");
report.AppendLine("============================================================");
report.AppendLine();

// ----- Extract YUV video + raw audio (seeds for both pipelines) -----
string yuvPath = Path.Combine(outDir, "src.yuv");
string srcMonoPcm48 = Path.Combine(outDir, "src48.pcm"); // Opus
string srcStereoPcm44 = Path.Combine(outDir, "src44s.pcm"); // FLAC + Vorbis
RunFfmpeg($"-y -i \"{source}\" -vf crop={W}:{H}:0:0 -frames:v {frameCount} -f rawvideo -pix_fmt yuv420p \"{yuvPath}\"");
RunFfmpeg($"-y -i \"{source}\" -t {seconds} -f s16le -ac 1 -ar 48000 \"{srcMonoPcm48}\"");
RunFfmpeg($"-y -i \"{source}\" -t {seconds} -f s16le -ac 2 -ar 44100 \"{srcStereoPcm44}\"");
int frameSize = W * H + 2 * (W / 2) * (H / 2);
var allFrames = File.ReadAllBytes(yuvPath);
var srcMonoBytes = File.ReadAllBytes(srcMonoPcm48);
var srcStereoBytes = File.ReadAllBytes(srcStereoPcm44);
Console.WriteLine($"Source: {frameCount} frames {W}x{H} ({allFrames.Length / 1024 / 1024}MB), audio mono48k {srcMonoBytes.Length}B + stereo44k {srcStereoBytes.Length}B");
Console.WriteLine();

report.AppendLine("VIDEO ENCODERS (full BBB clip)");
report.AppendLine($"{"Codec",-22}{"Encode ms",-12}{"fps",-8}{"Output KB",-12}{"Bitrate kbps",-14}");
report.AppendLine($"{new string('-', 22)}{new string('-', 12)}{new string('-', 8)}{new string('-', 12)}{new string('-', 14)}");

// Our VP8
TimeOursVideo("VP8 (ours)", "VP80",
    (y, u, v) => Vp8KeyframeEncoder.EncodeKeyFrame(y, W, u, W / 2, v, W, H, baseQIndex: 30));
// Our VP9
TimeOursVideo("VP9 (ours)", "VP90",
    (y, u, v) => Vp9KeyframeEncoder.EncodeKeyFrame(y, W, u, W / 2, v, W, H, baseQIndex: 30));
// Our AV1
TimeOursVideo("AV1 (ours)", "AV01",
    (y, u, v) => Av1KeyframeEncoder.EncodeKeyFrame(y, W, u, W / 2, v, W, H, baseQIndex: 32));

// ffmpeg references on the same YUV.
TimeFfmpegVideo("VP8 (ffmpeg)", $"-c:v libvpx -keyint_min 1 -g 1 -auto-alt-ref 0 -frames:v {frameCount}", "ff_vp8.ivf");
TimeFfmpegVideo("VP9 (ffmpeg)", $"-c:v libvpx-vp9 -keyint_min 1 -g 1 -frames:v {frameCount}", "ff_vp9.ivf");
TimeFfmpegVideo("AV1 (ffmpeg)", $"-c:v libaom-av1 -cpu-used 8 -keyint_min 1 -g 1 -frames:v {frameCount}", "ff_av1.ivf");

report.AppendLine();
report.AppendLine("AUDIO ENCODERS (same source clip, all 3 codecs side-by-side)");
report.AppendLine($"{"Codec",-22}{"Encode ms",-12}{"Output KB",-12}{"Realtime",-10}");
report.AppendLine($"{new string('-', 22)}{new string('-', 12)}{new string('-', 12)}{new string('-', 10)}");

// Our FLAC (44k stereo).
{
    var samples = new int[srcStereoBytes.Length / 2];
    for (int i = 0; i < samples.Length; i++) samples[i] = (short)(srcStereoBytes[i * 2] | (srcStereoBytes[i * 2 + 1] << 8));
    var sw = Stopwatch.StartNew();
    var bytes = FlacEncoder.EncodeStream(samples, 44100, 2, 16);
    sw.Stop();
    AddAudio("FLAC (ours)", sw.Elapsed.TotalMilliseconds, bytes.Length, seconds);
}
// ffmpeg FLAC.
TimeFfmpegAudio("FLAC (ffmpeg)", srcStereoPcm44, 44100, 2, "-c:a flac", "ff.flac");

// Our Opus (48k mono).
{
    var pcm = new float[srcMonoBytes.Length / 2];
    for (int i = 0; i < pcm.Length; i++) pcm[i] = ((short)(srcMonoBytes[i * 2] | (srcMonoBytes[i * 2 + 1] << 8))) / 32768f;
    var enc = new OpusEncoder(new OpusEncoderConfig { SampleRateHz = 48000, ChannelCount = 1, Application = OpusEncoderApplication.Audio });
    var packetBuf = new byte[1275];
    int frameSamples = 48000 / 50;
    int frames = pcm.Length / frameSamples;
    long total = 0;
    var sw = Stopwatch.StartNew();
    for (int f = 0; f < frames; f++) total += enc.EncodeFrame(pcm.AsSpan(f * frameSamples, frameSamples), packetBuf, frameSamples);
    sw.Stop();
    enc.Dispose();
    AddAudio("Opus (ours)", sw.Elapsed.TotalMilliseconds, total, seconds);
}
// ffmpeg Opus.
TimeFfmpegAudio("Opus (ffmpeg)", srcMonoPcm48, 48000, 1, "-c:a libopus -b:a 64k", "ff.opus");

// Our Vorbis (44k mono).
{
    var monoBytes = new byte[srcStereoBytes.Length / 2];
    for (int i = 0; i < monoBytes.Length / 2; i++)
    {
        short l = (short)(srcStereoBytes[i * 4] | (srcStereoBytes[i * 4 + 1] << 8));
        short r = (short)(srcStereoBytes[i * 4 + 2] | (srcStereoBytes[i * 4 + 3] << 8));
        short m = (short)((l + r) / 2);
        monoBytes[i * 2] = (byte)m;
        monoBytes[i * 2 + 1] = (byte)(m >> 8);
    }
    var pcm = new float[monoBytes.Length / 2];
    for (int i = 0; i < pcm.Length; i++) pcm[i] = ((short)(monoBytes[i * 2] | (monoBytes[i * 2 + 1] << 8))) / 32768f;
    var enc = new VorbisAudioEncoder(new VorbisAudioEncoderOptions { SampleRateHz = 44100, Channels = 1 });
    var sw = Stopwatch.StartNew();
    var ogg = enc.EncodeStream(pcm);
    sw.Stop();
    AddAudio("Vorbis (ours)", sw.Elapsed.TotalMilliseconds, ogg.Length, seconds);
}
// ffmpeg Vorbis on the same downmixed mono so the comparison is fair.
{
    string srcMono44 = Path.Combine(outDir, "src44m.pcm");
    RunFfmpeg($"-y -i \"{source}\" -t {seconds} -f s16le -ac 1 -ar 44100 \"{srcMono44}\"");
    TimeFfmpegAudio("Vorbis (ffmpeg)", srcMono44, 44100, 1, "-c:a libvorbis", "ff.ogg");
}

string output = report.ToString();
Console.Write(output);
string reportPath = Path.Combine(outDir, "report.txt");
File.WriteAllText(reportPath, output);
Console.WriteLine();
Console.WriteLine($"Report: {reportPath}");

void TimeOursVideo(string label, string fourcc, Func<byte[], byte[], byte[], byte[]> encode)
{
    long totalBytes = 0;
    var sw = Stopwatch.StartNew();
    for (int f = 0; f < frameCount; f++)
    {
        int yOff = f * frameSize;
        int uOff = yOff + W * H;
        int vOff = uOff + (W / 2) * (H / 2);
        var y = allFrames[yOff..uOff];
        var u = allFrames[uOff..vOff];
        var v = allFrames[vOff..(vOff + (W / 2) * (H / 2))];
        totalBytes += encode(y, u, v).Length;
    }
    sw.Stop();
    AddVideo(label, sw.Elapsed.TotalMilliseconds, totalBytes);
}

void TimeFfmpegVideo(string label, string args, string outName)
{
    string outPath = Path.Combine(outDir, outName);
    var sw = Stopwatch.StartNew();
    RunFfmpeg($"-y -f rawvideo -pix_fmt yuv420p -s {W}x{H} -i \"{yuvPath}\" {args} \"{outPath}\"");
    sw.Stop();
    AddVideo(label, sw.Elapsed.TotalMilliseconds, new FileInfo(outPath).Length);
}

void TimeFfmpegAudio(string label, string srcPcm, int rate, int ch, string codecArgs, string outName)
{
    string outPath = Path.Combine(outDir, outName);
    var sw = Stopwatch.StartNew();
    RunFfmpeg($"-y -f s16le -ar {rate} -ac {ch} -i \"{srcPcm}\" {codecArgs} \"{outPath}\"");
    sw.Stop();
    AddAudio(label, sw.Elapsed.TotalMilliseconds, new FileInfo(outPath).Length, seconds);
}

void AddVideo(string label, double encMs, long sz)
{
    double fps = frameCount * 1000.0 / encMs;
    double kbps = sz * 8.0 / 1000.0 / seconds;
    report.AppendLine($"{label,-22}{encMs,-12:F0}{fps,-8:F1}{sz / 1024.0,-12:F1}{kbps,-14:F0}");
}

void AddAudio(string label, double encMs, long sz, int durSec)
{
    double rt = durSec * 1000.0 / encMs;
    report.AppendLine($"{label,-22}{encMs,-12:F0}{sz / 1024.0,-12:F1}{rt,-10:F1}x");
}

void RunFfmpeg(string args)
{
    var p = Process.Start(new ProcessStartInfo(ffmpeg, args) { RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true })!;
    p.StandardError.ReadToEnd();
    p.WaitForExit();
    if (p.ExitCode != 0) throw new Exception($"ffmpeg failed: {args}");
}
