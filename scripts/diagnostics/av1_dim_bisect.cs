// AV1 encoder dimension bisection: encode flat-Y=128 frames at multiple
// widths/heights and report which sizes libdav1d (via ffmpeg) accepts.

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Av1;

string ffmpeg = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
string outDir = Path.Combine(Path.GetTempPath(), "av1_dim_bisect");
Directory.CreateDirectory(outDir);

(int W, int H)[] sizes = {
    (16, 16), (32, 32), (64, 64), (96, 96), (128, 128), (256, 256),
    (320, 240), (640, 480), (1280, 720), (1920, 1072),
};

Console.WriteLine("AV1 dimension bisection (flat Y=128, libdav1d decoder):");
Console.WriteLine($"  {"WxH",-12}{"frame B",-10}{"dav1d",-10}{"detail"}");

foreach (var (W, H) in sizes)
{
    var ySrc = new byte[W * H];
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            ySrc[r * W + c] = (byte)Math.Clamp(96 + (r + c) % 64, 0, 255);
    var uSrc = new byte[(W / 2) * (H / 2)]; Array.Fill(uSrc, (byte)128);
    var vSrc = new byte[(W / 2) * (H / 2)]; Array.Fill(vSrc, (byte)128);
    byte[] frame;
    try
    {
        frame = Av1KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 32);
    }
    catch (Exception ex)
    {
        Console.WriteLine($"  {$"{W}x{H}",-12}encoder threw: {ex.Message}");
        continue;
    }
    string ivf = Path.Combine(outDir, $"f_{W}x{H}.ivf");
    string yuv = Path.Combine(outDir, $"f_{W}x{H}.yuv");
    using (var fs = File.Create(ivf))
    {
        var w = new IvfWriter(fs, "AV01", W, H, frameRate: 1, timeScale: 30, numFrames: 0, leaveOpen: true);
        w.WriteFrame(frame, 0); w.Finish();
    }
    var psi = new ProcessStartInfo(ffmpeg, $"-y -c:v libdav1d -i \"{ivf}\" -f rawvideo -pix_fmt yuv420p \"{yuv}\"")
    { RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true };
    var p = Process.Start(psi)!;
    string err = p.StandardError.ReadToEnd();
    p.WaitForExit();
    string verdict = p.ExitCode == 0 ? "OK" : "REJECT";
    string detail = "";
    if (p.ExitCode == 0)
    {
        var dec = File.ReadAllBytes(yuv);
        if (dec.Length >= W * H)
        {
            int min = 255, max = 0;
            for (int i = 0; i < W * H; i++) { if (dec[i] < min) min = dec[i]; if (dec[i] > max) max = dec[i]; }
            detail = $"Y range=[{min},{max}]";
        }
    }
    else
    {
        var lines = err.Split('\n');
        foreach (var line in lines)
        {
            if (line.Contains("Error", StringComparison.OrdinalIgnoreCase) ||
                line.Contains("Invalid", StringComparison.OrdinalIgnoreCase) ||
                line.Contains("[dav1d", StringComparison.OrdinalIgnoreCase))
            {
                detail = line.Trim();
                if (detail.Length > 60) detail = detail[..60] + "...";
                break;
            }
        }
    }
    Console.WriteLine($"  {$"{W}x{H}",-12}{frame.Length,-10}{verdict,-10}{detail}");
}
