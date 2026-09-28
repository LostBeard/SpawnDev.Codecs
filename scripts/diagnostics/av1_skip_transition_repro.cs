// AV1 skip-transition reproducer. Builds tiny frames with specific block
// content patterns to isolate the "non-skip block followed by skip block"
// bug. Reports libdav1d acceptance per pattern.
//
// Pattern key:
//   "G" - gradient block (gradient content; produces non-zero coefs)
//   "F" - flat block (Y=128; quantizes to zero coefs -> skip block)
//
// We test 32x32 frames (4 BLOCK_16X16 leaves) with various combinations.
// Scan order is: top-left, top-right, bottom-left, bottom-right.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Av1;

string ffmpeg = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
string outDir = Path.Combine(Path.GetTempPath(), "av1_skip_transition_repro");
Directory.CreateDirectory(outDir);

// Block patterns to test on 32x32 frame (4 BLOCK_16X16 in raster order):
// "G" = gradient (non-skip), "F" = flat (skip)
string[] patterns = {
    "GGGG", // all non-skip
    "FFFF", // all skip
    "GGGF", // skip is LAST
    "FGGG", // skip is FIRST
    "GFGG", // skip in middle
    "GGFG",
    "FGFG",
    "GFGF",
    "GFFF", // 1 non-skip, then skips
    "FGGF",
};

int W = 32, H = 32;
Console.WriteLine($"AV1 skip-transition repro on {W}x{H} frames (4 BLOCK_16X16 each):");
Console.WriteLine($"  {"Pattern",-10}{"frame B",-10}{"dav1d",-10}{"detail"}");

foreach (var pattern in patterns)
{
    if (pattern.Length != 4) continue;
    var ySrc = new byte[W * H];
    // Fill all 128 (flat).
    Array.Fill(ySrc, (byte)128);
    // Add gradient to specified blocks.
    for (int b = 0; b < 4; b++)
    {
        if (pattern[b] != 'G') continue;
        int br = b / 2;       // block row 0/1
        int bc = b % 2;       // block col 0/1
        int xOff = bc * 16;
        int yOff = br * 16;
        for (int r = 0; r < 16; r++)
            for (int c = 0; c < 16; c++)
                ySrc[(yOff + r) * W + xOff + c] = (byte)Math.Clamp(64 + (r + c) * 4, 0, 255);
    }
    var uSrc = new byte[(W / 2) * (H / 2)]; Array.Fill(uSrc, (byte)128);
    var vSrc = new byte[(W / 2) * (H / 2)]; Array.Fill(vSrc, (byte)128);

    byte[] frame;
    try
    {
        frame = Av1KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 32);
    }
    catch (Exception ex)
    {
        Console.WriteLine($"  {pattern,-10}encoder threw: {ex.Message}");
        continue;
    }
    string ivf = Path.Combine(outDir, $"p_{pattern}.ivf");
    string yuv = Path.Combine(outDir, $"p_{pattern}.yuv");
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
    if (p.ExitCode != 0)
    {
        var lines = err.Split('\n');
        foreach (var line in lines)
        {
            if (line.Contains("Error", StringComparison.OrdinalIgnoreCase) ||
                line.Contains("Invalid", StringComparison.OrdinalIgnoreCase) ||
                line.Contains("Failed", StringComparison.OrdinalIgnoreCase) ||
                line.Contains("[dav1d", StringComparison.OrdinalIgnoreCase))
            {
                detail = line.Trim();
                if (detail.Length > 70) detail = detail[..70] + "...";
                break;
            }
        }
    }
    Console.WriteLine($"  {pattern,-10}{frame.Length,-10}{verdict,-10}{detail}");
}
