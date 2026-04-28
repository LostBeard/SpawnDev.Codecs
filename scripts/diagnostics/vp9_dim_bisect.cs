// VP9 encoder dimension bisection: encode flat-Y=128 frames at multiple
// widths/heights and report which sizes ffmpeg's VP9 decoder accepts vs
// rejects. Localizes the FullHD failure surface.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Vp9;

string ffmpeg = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
string outDir = Path.Combine(Path.GetTempPath(), "vp9_dim_bisect");
Directory.CreateDirectory(outDir);

(int W, int H)[] sizes = {
    (16, 16), (32, 32), (64, 64), (96, 96), (128, 128), (160, 160),
    (192, 192), (256, 256), (320, 240), (640, 480), (1280, 720),
    (1920, 1072), (1920, 1088),
};

Console.WriteLine("VP9 dimension bisection (flat Y=128/U=128/V=128):");
Console.WriteLine($"  {"WxH",-12}{"frame B",-10}{"ffmpeg",-10}{"detail"}");

foreach (var (W, H) in sizes)
{
    var ySrc = new byte[W * H]; Array.Fill(ySrc, (byte)128);
    var uSrc = new byte[(W / 2) * (H / 2)]; Array.Fill(uSrc, (byte)128);
    var vSrc = new byte[(W / 2) * (H / 2)]; Array.Fill(vSrc, (byte)128);
    var frame = Vp9KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 30);
    string ivf = Path.Combine(outDir, $"f_{W}x{H}.ivf");
    string yuv = Path.Combine(outDir, $"f_{W}x{H}.yuv");
    using (var fs = File.Create(ivf))
    {
        var w = new IvfWriter(fs, "VP90", W, H, frameRate: 1, timeScale: 30, numFrames: 0, leaveOpen: true);
        w.WriteFrame(frame, 0); w.Finish();
    }
    var psi = new ProcessStartInfo(ffmpeg, $"-y -i \"{ivf}\" -f rawvideo -pix_fmt yuv420p \"{yuv}\"")
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
        // Extract first error line.
        var lines = err.Split('\n');
        foreach (var line in lines)
        {
            if (line.Contains("Error", StringComparison.OrdinalIgnoreCase) ||
                line.Contains("Invalid", StringComparison.OrdinalIgnoreCase))
            {
                detail = line.Trim();
                if (detail.Length > 60) detail = detail[..60] + "...";
                break;
            }
        }
    }
    Console.WriteLine($"  {$"{W}x{H}",-12}{frame.Length,-10}{verdict,-10}{detail}");
}
