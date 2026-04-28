// Real-world transcoding benchmark using TJ's Big Buck Bunny FullHD source.
// Extracts N frames at 1920x1072 (cropped from 1920x1080 to satisfy our
// encoders' multiple-of-16 constraint), runs each frame through VP8/VP9/AV1
// keyframe encoders + ffmpeg references, and reports throughput +
// compression vs libvpx/libaom on the same data.
//
// Outputs a playable .mp4 for VLC visual verification (h264-remuxed from
// our IVF) so TJ can eyeball quality.
//
// Usage: dotnet run benchmark_bbb_transcode.cs [frameCount=10]

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.Codecs.Video.Vp9;

int frameCount = args.Length >= 1 && int.TryParse(args[0], out int n) ? n : 10;
const int W = 1920;
const int H = 1072; // 1080 cropped to 16-multiple
string source = "V:\\Video\\Big Buck Bunny - FULL HD 60FPS.mp4";
string ffmpeg = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
string outDir = Path.Combine(Path.GetTempPath(), "spawndev_bbb_transcode");
Directory.CreateDirectory(outDir);

if (!File.Exists(source))
{
    Console.WriteLine($"Source not found: {source}");
    Environment.Exit(1);
}

// Extract first N frames as raw YUV420p, cropped to 1920x1072.
string yuvPath = Path.Combine(outDir, "bbb_input.yuv");
Console.WriteLine($"Extracting {frameCount} frames from BBB at {W}x{H}...");
{
    var sw = Stopwatch.StartNew();
    var psi = new ProcessStartInfo(ffmpeg,
        $"-y -i \"{source}\" -vf crop={W}:{H}:0:0 -frames:v {frameCount} -f rawvideo -pix_fmt yuv420p \"{yuvPath}\"")
    { RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true };
    var p = Process.Start(psi)!;
    p.StandardError.ReadToEnd();
    p.WaitForExit();
    sw.Stop();
    if (p.ExitCode != 0) { Console.WriteLine("ffmpeg extract failed"); Environment.Exit(1); }
    Console.WriteLine($"  extracted in {sw.Elapsed.TotalSeconds:F1}s, file={new FileInfo(yuvPath).Length / 1024 / 1024}MB");
}

int frameSize = W * H + 2 * (W / 2) * (H / 2);
var allFrames = File.ReadAllBytes(yuvPath);
if (allFrames.Length < frameCount * frameSize)
{
    Console.WriteLine($"Got {allFrames.Length} bytes, expected {(long)frameCount * frameSize}");
    Environment.Exit(1);
}
Console.WriteLine($"  YUV bytes per frame: {frameSize:N0}");
Console.WriteLine();

var report = new StringBuilder();
report.AppendLine("============================================================");
report.AppendLine($"  BBB transcode benchmark - {frameCount} frames @ {W}x{H}");
report.AppendLine("============================================================");
report.AppendLine($"  Source: V:\\Video\\Big Buck Bunny - FULL HD 60FPS.mp4");
report.AppendLine($"  Raw YUV per frame: {frameSize:N0} bytes");
report.AppendLine();
report.AppendLine($"{"Codec",-15}{"Frames/s",-12}{"avg KB/frame",-15}{"ratio",-10}{"total MB out",-14}");
report.AppendLine($"{new string('-', 15)}{new string('-', 12)}{new string('-', 15)}{new string('-', 10)}{new string('-', 14)}");

void Run(string codec, Func<byte[], byte[], byte[], byte[]> encode)
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
    double fps = frameCount / sw.Elapsed.TotalSeconds;
    double avgKB = totalBytes / 1024.0 / frameCount;
    double ratio = totalBytes / (double)(frameCount * (long)frameSize);
    double totalMB = totalBytes / 1024.0 / 1024.0;
    report.AppendLine($"{codec,-15}{fps,-12:F2}{avgKB,-15:F2}{ratio,-10:F4}{totalMB,-14:F2}");
}

// VP8 ours
Run("VP8 (ours)", (y, u, v) =>
    Vp8KeyframeEncoder.EncodeKeyFrame(y, W, u, W / 2, v, W, H, baseQIndex: 30));

// VP9 ours
Run("VP9 (ours)", (y, u, v) =>
    Vp9KeyframeEncoder.EncodeKeyFrame(y, W, u, W / 2, v, W, H, baseQIndex: 30));

// AV1 ours
Run("AV1 (ours)", (y, u, v) =>
    Av1KeyframeEncoder.EncodeKeyFrame(y, W, u, W / 2, v, W, H, baseQIndex: 32));

// ffmpeg references on the same input.
void RunFfmpeg(string codec, string args, string outName)
{
    string outPath = Path.Combine(outDir, outName);
    var sw = Stopwatch.StartNew();
    var psi = new ProcessStartInfo(ffmpeg,
        $"-y -f rawvideo -pix_fmt yuv420p -s {W}x{H} -i \"{yuvPath}\" {args} \"{outPath}\"")
    { RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true };
    var p = Process.Start(psi)!;
    p.StandardError.ReadToEnd();
    p.WaitForExit();
    sw.Stop();
    if (p.ExitCode != 0) { report.AppendLine($"{codec,-15}FAILED ffmpeg"); return; }
    long sz = new FileInfo(outPath).Length;
    double fps = frameCount / sw.Elapsed.TotalSeconds;
    double avgKB = sz / 1024.0 / frameCount;
    double ratio = sz / (double)(frameCount * (long)frameSize);
    double totalMB = sz / 1024.0 / 1024.0;
    report.AppendLine($"{codec,-15}{fps,-12:F2}{avgKB,-15:F2}{ratio,-10:F4}{totalMB,-14:F2}");
}

RunFfmpeg("VP8 (ffmpeg)", $"-c:v libvpx -keyint_min 1 -g 1 -auto-alt-ref 0 -frames:v {frameCount}", "ff_vp8.ivf");
RunFfmpeg("VP9 (ffmpeg)", $"-c:v libvpx-vp9 -keyint_min 1 -g 1 -frames:v {frameCount}", "ff_vp9.ivf");
RunFfmpeg("AV1 (ffmpeg)", $"-c:v libaom-av1 -cpu-used 8 -keyint_min 1 -g 1 -frames:v {frameCount}", "ff_av1.ivf");

// Build a playable VP8 .mp4 from our encoder output for VLC eyeball check.
string oursVp8Ivf = Path.Combine(outDir, "ours_vp8.ivf");
string oursVp9Ivf = Path.Combine(outDir, "ours_vp9.ivf");
string oursVp8Mp4 = Path.Combine(outDir, "ours_vp8.mp4");
string oursVp9Mp4 = Path.Combine(outDir, "ours_vp9.mp4");
{
    using var fs = File.Create(oursVp8Ivf);
    var w = new IvfWriter(fs, "VP80", W, H, frameRate: 60, timeScale: 1, numFrames: 0, leaveOpen: true);
    for (int f = 0; f < frameCount; f++)
    {
        int yOff = f * frameSize;
        int uOff = yOff + W * H;
        int vOff = uOff + (W / 2) * (H / 2);
        var y = allFrames[yOff..uOff];
        var u = allFrames[uOff..vOff];
        var v = allFrames[vOff..(vOff + (W / 2) * (H / 2))];
        w.WriteFrame(Vp8KeyframeEncoder.EncodeKeyFrame(y, W, u, W / 2, v, W, H, baseQIndex: 30), f);
    }
    w.Finish();
}
{
    using var fs = File.Create(oursVp9Ivf);
    var w = new IvfWriter(fs, "VP90", W, H, frameRate: 60, timeScale: 1, numFrames: 0, leaveOpen: true);
    for (int f = 0; f < frameCount; f++)
    {
        int yOff = f * frameSize;
        int uOff = yOff + W * H;
        int vOff = uOff + (W / 2) * (H / 2);
        var y = allFrames[yOff..uOff];
        var u = allFrames[uOff..vOff];
        var v = allFrames[vOff..(vOff + (W / 2) * (H / 2))];
        w.WriteFrame(Vp9KeyframeEncoder.EncodeKeyFrame(y, W, u, W / 2, v, W, H, baseQIndex: 30), f);
    }
    w.Finish();
}

void Remux(string ivf, string mp4)
{
    var psi = new ProcessStartInfo(ffmpeg, $"-y -i \"{ivf}\" -c:v libx264 -pix_fmt yuv420p \"{mp4}\"")
    { RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true };
    var p = Process.Start(psi)!;
    p.StandardError.ReadToEnd();
    p.WaitForExit();
}
Remux(oursVp8Ivf, oursVp8Mp4);
Remux(oursVp9Ivf, oursVp9Mp4);

string output = report.ToString();
Console.Write(output);
string reportPath = Path.Combine(outDir, "report.txt");
File.WriteAllText(reportPath, output);
Console.WriteLine();
Console.WriteLine($"Output dir: {outDir}");
Console.WriteLine($"  ours VP8 .mp4 (VLC playable): {oursVp8Mp4}");
Console.WriteLine($"  ours VP9 .mp4 (VLC playable): {oursVp9Mp4}");
Console.WriteLine($"  ffmpeg references: ff_vp8.ivf, ff_vp9.ivf, ff_av1.ivf");
Console.WriteLine($"  Report: {reportPath}");
