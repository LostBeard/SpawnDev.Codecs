// Multi-frame VP9 encode -> IVF -> ffmpeg verify -> visual MP4.
//
// Produces a 60-frame VP9 keyframe-only animation (rotating gradient)
// wrapped in an IVF container, then runs it through ffmpeg to:
//   1. Verify ffmpeg accepts every frame (single-block 16x16 path is
//      ffmpeg-compatible per Vp9KeyframeEncoder agent's report).
//   2. Convert to MP4 for VLC playback.
//
// Output:
//   D:\Temp\spawndev_vp9_animation.ivf  - raw VP9 in IVF
//   D:\Temp\spawndev_vp9_animation.mp4  - MP4 for VLC
//
// Usage: dotnet run vp9_encode_animation.cs

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Vp9;

const int W = 16, H = 16;          // single-block size (ffmpeg-clean per Vp9KeyframeEncoder agent)
const int Frames = 60;
const int Q = 30;

string outDir = Path.Combine(Path.GetTempPath(), "spawndev_vp9_anim");
Directory.CreateDirectory(outDir);
string ivfPath = Path.Combine(outDir, "spawndev_vp9_animation.ivf");
string mp4Path = Path.Combine(outDir, "spawndev_vp9_animation.mp4");

// === Encode ===
long totalBytes = 0;
var sw = Stopwatch.StartNew();
using (var fs = File.Create(ivfPath))
{
    var writer = new IvfWriter(fs, "VP90", W, H, frameRate: 30, timeScale: 1, numFrames: 0, leaveOpen: true);
    var ySrc = new byte[W * H];
    var uSrc = new byte[(W / 2) * (H / 2)];
    var vSrc = new byte[(W / 2) * (H / 2)];

    for (int f = 0; f < Frames; f++)
    {
        // Rotating gradient: per-frame phase shift produces moving sine pattern.
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
            {
                double phase = 2.0 * Math.PI * (c + f) / W;
                ySrc[r * W + c] = (byte)Math.Clamp(80 + 40 * Math.Sin(phase) + r * 4, 0, 255);
            }
        for (int r = 0; r < H / 2; r++)
            for (int c = 0; c < W / 2; c++)
            {
                uSrc[r * (W / 2) + c] = (byte)(128 + (f - Frames / 2));
                vSrc[r * (W / 2) + c] = (byte)(128 - (f - Frames / 2));
            }

        var frameBytes = Vp9KeyframeEncoder.EncodeKeyFrame(
            ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: Q);
        writer.WriteFrame(frameBytes, f);
        totalBytes += frameBytes.Length;
    }
    writer.Finish();
}
sw.Stop();

double encFps = Frames / sw.Elapsed.TotalSeconds;
double avgFrameBytes = totalBytes / (double)Frames;
Console.WriteLine($"VP9 keyframe-only animation:");
Console.WriteLine($"  {Frames} x {W}x{H} frames @ Q={Q}");
Console.WriteLine($"  Encode time:  {sw.Elapsed.TotalMilliseconds:F1}ms ({encFps:F1} fps)");
Console.WriteLine($"  Avg frame:    {avgFrameBytes:F0} bytes");
Console.WriteLine($"  IVF written:  {new FileInfo(ivfPath).Length:N0} bytes -> {ivfPath}");

// === Verify with ffmpeg + convert to MP4 ===
string ffmpeg = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
if (!File.Exists(ffmpeg)) ffmpeg = "ffmpeg";

var p = Process.Start(new ProcessStartInfo(ffmpeg, $"-y -i \"{ivfPath}\" -c:v libx264 -pix_fmt yuv420p \"{mp4Path}\"")
{
    RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true,
})!;
string err = p.StandardError.ReadToEnd();
p.WaitForExit();

if (p.ExitCode != 0)
{
    Console.WriteLine();
    Console.WriteLine($"FAIL: ffmpeg failed to decode our VP9 IVF.");
    Console.Error.WriteLine(err);
    Environment.Exit(1);
    return;
}

long mp4Size = File.Exists(mp4Path) ? new FileInfo(mp4Path).Length : 0;
Console.WriteLine();
Console.WriteLine($"PASS: ffmpeg decoded all {Frames} VP9 frames + remuxed to MP4");
Console.WriteLine($"  MP4 written:  {mp4Size:N0} bytes -> {mp4Path}");
Console.WriteLine($"  Open in VLC for visual playback.");
