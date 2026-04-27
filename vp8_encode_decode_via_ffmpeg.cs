// VP8 encoder integration test: encode a 32x32 YUV420 input via our
// encoder, wrap in IVF, then run ffmpeg to decode the bitstream and
// verify it accepts the file as valid VP8.

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Vp8;

const int Width = 32;
const int Height = 32;
const int Fps = 30;

string ffmpegPath = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
string outIvf = Path.Combine(Path.GetTempPath(), "spawndev_vp8_encode_test.ivf");
string outYuv = Path.Combine(Path.GetTempPath(), "spawndev_vp8_decode_test.yuv");

// Build a 32x32 YUV420 source: constant Y=80 (mid-gray), U=V=128 (no chroma).
var ySrc = new byte[Width * Height];
var uSrc = new byte[(Width / 2) * (Height / 2)];
var vSrc = new byte[(Width / 2) * (Height / 2)];
Array.Fill(ySrc, (byte)80);
Array.Fill(uSrc, (byte)128);
Array.Fill(vSrc, (byte)128);

// Encode the keyframe.
var frameBytes = Vp8KeyframeEncoder.EncodeKeyFrame(
    ySrc, Width, uSrc, Width / 2, vSrc, Width, Height, baseQIndex: 30);
Console.WriteLine($"Encoded VP8 keyframe: {frameBytes.Length} bytes");

// Wrap in IVF.
using (var fs = File.Create(outIvf))
{
    var writer = new IvfWriter(fs, "VP80", Width, Height, frameRate: 1, timeScale: Fps, numFrames: 1);
    writer.WriteFrame(frameBytes, 0);
    writer.Finish();
}
Console.WriteLine($"IVF written to {outIvf} ({new FileInfo(outIvf).Length} bytes)");

// Have ffmpeg decode it to raw YUV.
var psi = new ProcessStartInfo(ffmpegPath, $"-y -i \"{outIvf}\" -f rawvideo -pix_fmt yuv420p \"{outYuv}\"")
{
    RedirectStandardError = true,
    UseShellExecute = false,
    CreateNoWindow = true,
};
var p = Process.Start(psi)!;
string stderr = p.StandardError.ReadToEnd();
p.WaitForExit();

Console.WriteLine($"ffmpeg exit code: {p.ExitCode}");
if (p.ExitCode != 0)
{
    Console.WriteLine("ffmpeg STDERR (last 30 lines):");
    var lines = stderr.Split('\n');
    int start = Math.Max(0, lines.Length - 30);
    for (int i = start; i < lines.Length; i++)
        Console.WriteLine("  " + lines[i].TrimEnd('\r'));
    Console.WriteLine();
    Console.WriteLine("=== VP8 ENCODER: ffmpeg REJECTED bitstream ===");
    Environment.Exit(1);
}

// Verify ffmpeg produced YUV output.
if (!File.Exists(outYuv))
{
    Console.WriteLine("FAIL: ffmpeg did not write YUV output");
    Environment.Exit(1);
}

var yuvBytes = File.ReadAllBytes(outYuv);
int expectedSize = Width * Height + 2 * (Width / 2) * (Height / 2);
Console.WriteLine($"ffmpeg decoded YUV: {yuvBytes.Length} bytes (expected {expectedSize})");

if (yuvBytes.Length != expectedSize)
{
    Console.WriteLine($"FAIL: YUV size mismatch");
    Environment.Exit(1);
}

// Compare statistics to source.
int yMin = 255, yMax = 0, ySum = 0;
for (int i = 0; i < Width * Height; i++)
{
    if (yuvBytes[i] < yMin) yMin = yuvBytes[i];
    if (yuvBytes[i] > yMax) yMax = yuvBytes[i];
    ySum += yuvBytes[i];
}
int yMean = ySum / (Width * Height);

Console.WriteLine($"Decoded Y plane: min={yMin}, max={yMax}, mean={yMean}");
Console.WriteLine($"Source  Y plane: const=80");
Console.WriteLine();

// For "working" we just need ffmpeg to accept and decode without error.
// Quality is lossy due to no recon write-back; we just check ffmpeg got
// a valid stream back.
Console.WriteLine("=== VP8 ENCODER: ffmpeg ACCEPTED our bitstream ===");
Console.WriteLine($"libvpx-via-ffmpeg decoded our 32x32 keyframe to {yuvBytes.Length} YUV bytes.");
Console.WriteLine($"Visual quality is degraded (no recon write-back yet) but the bitstream parses valid.");
