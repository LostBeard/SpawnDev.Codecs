// YUV to visual file converter. Takes a raw YUV420 plane buffer
// (Y followed by U followed by V) and writes a PNG (single frame)
// or MP4 (animated) via ffmpeg for visual verification.
//
// Usage:
//   dotnet run yuv_to_visual.cs <input.yuv> <width>x<height> [out.png]
//   dotnet run yuv_to_visual.cs SpawnDev.Codecs.Demo.Shared/TestData/bbb_first_frame.yuv 320x180
//
// If no out path is given, writes to ./<input>_visual.png next to the input.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;

if (args.Length < 2)
{
    Console.Error.WriteLine("Usage: yuv_to_visual.cs <input.yuv> <width>x<height> [out.png|out.mp4]");
    Environment.Exit(1);
}

string yuvPath = args[0];
string sizeArg = args[1];
string outPath = args.Length >= 3
    ? args[2]
    : Path.Combine(Path.GetDirectoryName(yuvPath) ?? ".",
                   Path.GetFileNameWithoutExtension(yuvPath) + "_visual.png");

// Parse "WIDTHxHEIGHT".
int width = 0, height = 0;
{
    var parts = sizeArg.Split('x', StringSplitOptions.RemoveEmptyEntries);
    if (parts.Length != 2 || !int.TryParse(parts[0], out width) || !int.TryParse(parts[1], out height))
    {
        Console.Error.WriteLine($"Bad size '{sizeArg}'; expected '<width>x<height>'.");
        Environment.Exit(1);
        return;
    }
}

if (!File.Exists(yuvPath))
{
    Console.Error.WriteLine($"YUV file not found: {yuvPath}");
    Environment.Exit(1);
}

long expected = (long)width * height + 2L * (width / 2) * (height / 2);
long actual = new FileInfo(yuvPath).Length;
if (actual != expected)
{
    Console.Error.WriteLine($"YUV size mismatch: file={actual}B, expected for {width}x{height} 4:2:0 = {expected}B.");
    Environment.Exit(1);
}

string ffmpeg = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
if (!File.Exists(ffmpeg))
{
    // Try PATH lookup.
    ffmpeg = "ffmpeg";
}

string ext = Path.GetExtension(outPath).ToLowerInvariant();
string ffmpegArgs = "";
if (ext == ".png" || ext == ".jpg" || ext == ".jpeg")
{
    ffmpegArgs = $"-y -f rawvideo -pix_fmt yuv420p -s {width}x{height} -i \"{yuvPath}\" -frames:v 1 \"{outPath}\"";
}
else if (ext == ".mp4" || ext == ".mkv" || ext == ".webm")
{
    ffmpegArgs = $"-y -f rawvideo -pix_fmt yuv420p -s {width}x{height} -r 30 -i \"{yuvPath}\" -c:v libx264 -pix_fmt yuv420p \"{outPath}\"";
}
else
{
    Console.Error.WriteLine($"Unknown output extension '{ext}'. Use .png/.jpg/.mp4/.mkv/.webm.");
    Environment.Exit(1);
    return;
}

var p = Process.Start(new ProcessStartInfo(ffmpeg, ffmpegArgs)
{
    RedirectStandardError = true,
    UseShellExecute = false,
    CreateNoWindow = true,
})!;
string err = p.StandardError.ReadToEnd();
p.WaitForExit();
if (p.ExitCode != 0)
{
    Console.Error.WriteLine("ffmpeg failed:");
    Console.Error.WriteLine(err);
    Environment.Exit(1);
}

long outSize = File.Exists(outPath) ? new FileInfo(outPath).Length : 0;
Console.WriteLine($"OK  {yuvPath} -> {outPath} ({outSize:N0} B). Open in VLC / image viewer for visual check.");
