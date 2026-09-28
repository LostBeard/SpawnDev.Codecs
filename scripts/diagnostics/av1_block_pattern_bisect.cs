// AV1 block pattern bisect: encodes various block contents at multiple
// frame sizes and tests against libdav1d. Helps isolate whether the bug
// depends on:
//  - Block content (skip vs non-skip mix)
//  - Frame size (32x32, 64x64, 80x64, etc)
//  - Partition tree (sub-superblock vs full superblock)

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Av1;

string ffmpeg = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
string outDir = Path.Combine(Path.GetTempPath(), "av1_block_pattern_bisect");
Directory.CreateDirectory(outDir);

(int W, int H, string contentDesc, Func<int, int, byte> yFunc)[] tests = {
    // 16x16 frame: 1 block. Must be GG.
    (16, 16, "16x16 flat", (r, c) => 128),
    (16, 16, "16x16 grad", (r, c) => (byte)Math.Clamp(64 + (r + c) * 4, 0, 255)),

    // 32x32: 4 BLOCK_16X16. Test all combos.
    (32, 32, "32x32 all flat (FFFF)", (r, c) => 128),
    (32, 32, "32x32 all grad (GGGG)", (r, c) => (byte)Math.Clamp(64 + (r + c) * 4, 0, 255)),

    // 48x48: not a multiple of 16x... wait it is. But not multiple of 32. Sub-SB partition.
    (48, 48, "48x48 flat", (r, c) => 128),
    (48, 48, "48x48 grad", (r, c) => (byte)Math.Clamp(64 + (r + c) * 4, 0, 255)),

    (64, 64, "64x64 flat", (r, c) => 128),
    (64, 64, "64x64 grad", (r, c) => (byte)Math.Clamp(64 + (r + c) * 4, 0, 255)),

    // 80x80: not multiple of 64. 2x2 SBs but each only partial.
    (80, 80, "80x80 flat", (r, c) => 128),
    (80, 80, "80x80 grad", (r, c) => (byte)Math.Clamp(64 + (r + c) * 4, 0, 255)),

    (96, 96, "96x96 flat", (r, c) => 128),
    (96, 96, "96x96 grad", (r, c) => (byte)Math.Clamp(64 + (r + c) * 4, 0, 255)),

    // 128x64: 2x1 SBs.
    (128, 64, "128x64 flat", (r, c) => 128),
    (128, 64, "128x64 grad", (r, c) => (byte)Math.Clamp(64 + (r + c) * 4, 0, 255)),

    (128, 128, "128x128 flat", (r, c) => 128),
    (128, 128, "128x128 grad", (r, c) => (byte)Math.Clamp(64 + (r + c) * 4, 0, 255)),
};

Console.WriteLine($"AV1 block pattern bisect (libdav1d):");
Console.WriteLine($"  {"Test",-30}{"frame B",-10}{"verdict",-10}{"detail"}");

foreach (var (W, H, desc, yFunc) in tests)
{
    var ySrc = new byte[W * H];
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            ySrc[r * W + c] = yFunc(r, c);
    var uSrc = new byte[(W / 2) * (H / 2)]; Array.Fill(uSrc, (byte)128);
    var vSrc = new byte[(W / 2) * (H / 2)]; Array.Fill(vSrc, (byte)128);

    byte[] frame;
    try
    {
        frame = Av1KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 32);
    }
    catch (Exception ex)
    {
        Console.WriteLine($"  {desc,-30}encoder threw: {ex.Message}");
        continue;
    }
    string ivf = Path.Combine(outDir, $"f_{W}x{H}_{desc.Replace(" ", "_").Replace("(", "").Replace(")", "")}.ivf");
    string yuv = Path.Combine(outDir, $"f_{W}x{H}_{desc.Replace(" ", "_").Replace("(", "").Replace(")", "")}.yuv");
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
    Console.WriteLine($"  {desc,-30}{frame.Length,-10}{verdict,-10}{detail}");
}
