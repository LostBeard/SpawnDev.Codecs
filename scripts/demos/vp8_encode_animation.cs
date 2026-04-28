// Multi-frame VP8 encode -> IVF -> ffmpeg verify -> visual MP4.
// Mirrors vp9_encode_animation.cs for the VP8 encoder.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Vp8;

const int W = 32, H = 32;          // VP8 needs multiples of 16; 32x32 is the smoke-tested size
const int Frames = 60;
const int Q = 30;

string outDir = Path.Combine(Path.GetTempPath(), "spawndev_vp8_anim");
Directory.CreateDirectory(outDir);
string ivfPath = Path.Combine(outDir, "spawndev_vp8_animation.ivf");
string mp4Path = Path.Combine(outDir, "spawndev_vp8_animation.mp4");

long totalBytes = 0;
var sw = Stopwatch.StartNew();
using (var fs = File.Create(ivfPath))
{
    var writer = new IvfWriter(fs, "VP80", W, H, frameRate: 30, timeScale: 1, numFrames: 0, leaveOpen: true);
    var ySrc = new byte[W * H];
    var uSrc = new byte[(W / 2) * (H / 2)];
    var vSrc = new byte[(W / 2) * (H / 2)];

    for (int f = 0; f < Frames; f++)
    {
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
            {
                double phase = 2.0 * Math.PI * (c + f) / W;
                ySrc[r * W + c] = (byte)Math.Clamp(80 + 40 * Math.Sin(phase) + r * 2, 0, 255);
            }
        for (int r = 0; r < H / 2; r++)
            for (int c = 0; c < W / 2; c++)
            {
                uSrc[r * (W / 2) + c] = (byte)(128 + (f - Frames / 2));
                vSrc[r * (W / 2) + c] = (byte)(128 - (f - Frames / 2));
            }

        var frameBytes = Vp8KeyframeEncoder.EncodeKeyFrame(
            ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: Q);
        writer.WriteFrame(frameBytes, f);
        totalBytes += frameBytes.Length;
    }
    writer.Finish();
}
sw.Stop();

double encFps = Frames / sw.Elapsed.TotalSeconds;
double avgFrameBytes = totalBytes / (double)Frames;
Console.WriteLine($"VP8 keyframe-only animation:");
Console.WriteLine($"  {Frames} x {W}x{H} frames @ Q={Q}");
Console.WriteLine($"  Encode time:  {sw.Elapsed.TotalMilliseconds:F1}ms ({encFps:F1} fps)");
Console.WriteLine($"  Avg frame:    {avgFrameBytes:F0} bytes");
Console.WriteLine($"  IVF written:  {new FileInfo(ivfPath).Length:N0} bytes -> {ivfPath}");

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
    Console.WriteLine("FAIL: ffmpeg failed to decode our VP8 IVF.");
    Console.Error.WriteLine(err);
    Environment.Exit(1);
    return;
}

long mp4Size = File.Exists(mp4Path) ? new FileInfo(mp4Path).Length : 0;
Console.WriteLine();
Console.WriteLine($"PASS: ffmpeg decoded all {Frames} VP8 frames + remuxed to MP4");
Console.WriteLine($"  MP4 written:  {mp4Size:N0} bytes -> {mp4Path}");
Console.WriteLine($"  Open in VLC for visual playback.");
