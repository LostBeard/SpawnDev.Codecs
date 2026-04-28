// VP8 pixel-fidelity check: encode a known gray frame, decode via
// ffmpeg, and report the actual decoded pixel values. Diagnoses
// the "black output" issue TJ sees.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Vp8;

const int W = 16, H = 16;
string outDir = Path.Combine(Path.GetTempPath(), "vp8_pixel_check");
Directory.CreateDirectory(outDir);
string ivfPath = Path.Combine(outDir, "gray.ivf");
string yuvPath = Path.Combine(outDir, "gray_decoded.yuv");

string ffmpeg = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
if (!File.Exists(ffmpeg)) ffmpeg = "ffmpeg";

// Sweep flat luma values to find which encode/decode correctly.
Console.WriteLine($"Flat-luma sweep at Q=30 (predictor=128):");
Console.WriteLine($"  {"input",-6}  {"decoded mean",-14}  {"min",-5}  {"max",-5}  {"verdict",-10}");
foreach (int luma in new int[] { 64, 96, 110, 128, 144, 160, 192 })
{
    var ySrc2 = new byte[W * H];
    Array.Fill(ySrc2, (byte)luma);
    var uSrc2 = new byte[(W / 2) * (H / 2)]; Array.Fill(uSrc2, (byte)128);
    var vSrc2 = new byte[(W / 2) * (H / 2)]; Array.Fill(vSrc2, (byte)128);
    var frame2 = Vp8KeyframeEncoder.EncodeKeyFrame(ySrc2, W, uSrc2, W / 2, vSrc2, W, H, baseQIndex: 30);
    string ivf2 = Path.Combine(outDir, $"flat_{luma}.ivf");
    string yuv2 = Path.Combine(outDir, $"flat_{luma}.yuv");
    using (var fs = File.Create(ivf2))
    {
        var w = new IvfWriter(fs, "VP80", W, H, frameRate: 1, timeScale: 30, numFrames: 0, leaveOpen: true);
        w.WriteFrame(frame2, 0); w.Finish();
    }
    var p2 = Process.Start(new ProcessStartInfo(ffmpeg, $"-y -i \"{ivf2}\" -f rawvideo -pix_fmt yuv420p \"{yuv2}\"") { RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true })!;
    p2.StandardError.ReadToEnd();
    p2.WaitForExit();
    if (!File.Exists(yuv2)) { Console.WriteLine($"  {luma,-6}  ffmpeg failed"); continue; }
    var dec2 = File.ReadAllBytes(yuv2);
    int yMin2 = 255, yMax2 = 0, ySum2 = 0;
    for (int i = 0; i < W * H; i++) { yMin2 = Math.Min(yMin2, dec2[i]); yMax2 = Math.Max(yMax2, dec2[i]); ySum2 += dec2[i]; }
    int mean = ySum2 / (W * H);
    string verdict = Math.Abs(mean - luma) <= 4 ? "OK" : Math.Abs(mean - luma) <= 16 ? "drifted" : "BROKEN";
    Console.WriteLine($"  {luma,-6}  {mean,-14}  {yMin2,-5}  {yMax2,-5}  {verdict,-10}");
}
Console.WriteLine();

// Original gradient test for reference.
var ySrc = new byte[W * H];
for (int r = 0; r < H; r++)
    for (int c = 0; c < W; c++)
        ySrc[r * W + c] = (byte)Math.Clamp(80 + 40 * Math.Sin(2.0 * Math.PI * c / W) + r * 4, 0, 255);
var uSrc = new byte[(W / 2) * (H / 2)]; Array.Fill(uSrc, (byte)128);
var vSrc = new byte[(W / 2) * (H / 2)]; Array.Fill(vSrc, (byte)128);

int srcMean = 0; foreach (var b in ySrc) srcMean += b;
Console.WriteLine($"Gradient Y plane: min={Min(ySrc)}, max={Max(ySrc)}, mean={srcMean / (W * H)}");
static byte Min(byte[] a) { byte m = 255; foreach (var b in a) if (b < m) m = b; return m; }
static byte Max(byte[] a) { byte m = 0; foreach (var b in a) if (b > m) m = b; return m; }

var frame = Vp8KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 30);
Console.WriteLine($"Encoded {frame.Length} bytes for 16x16 Y=128/UV=128.");

using (var fs = File.Create(ivfPath))
{
    var w = new IvfWriter(fs, "VP80", W, H, frameRate: 1, timeScale: 30, numFrames: 0, leaveOpen: true);
    w.WriteFrame(frame, 0);
    w.Finish();
}

var p = Process.Start(new ProcessStartInfo(ffmpeg, $"-y -i \"{ivfPath}\" -f rawvideo -pix_fmt yuv420p \"{yuvPath}\"")
{
    RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true,
})!;
string err = p.StandardError.ReadToEnd();
p.WaitForExit();

if (p.ExitCode != 0)
{
    Console.Error.WriteLine("ffmpeg failed:");
    Console.Error.WriteLine(err);
    Environment.Exit(1);
    return;
}

var decoded = File.ReadAllBytes(yuvPath);
int expectedSize = W * H + 2 * (W / 2) * (H / 2);
Console.WriteLine($"Decoded YUV size: {decoded.Length} bytes (expected {expectedSize})");

// Y plane is the first W*H bytes.
int yOff = 0, uOff = W * H, vOff = W * H + (W / 2) * (H / 2);
int yMin = 255, yMax = 0, ySum = 0;
for (int i = 0; i < W * H; i++) { yMin = Math.Min(yMin, decoded[yOff + i]); yMax = Math.Max(yMax, decoded[yOff + i]); ySum += decoded[yOff + i]; }
int uMin = 255, uMax = 0, uSum = 0;
for (int i = 0; i < (W / 2) * (H / 2); i++) { uMin = Math.Min(uMin, decoded[uOff + i]); uMax = Math.Max(uMax, decoded[uOff + i]); uSum += decoded[uOff + i]; }
int vMin = 255, vMax = 0, vSum = 0;
for (int i = 0; i < (W / 2) * (H / 2); i++) { vMin = Math.Min(vMin, decoded[vOff + i]); vMax = Math.Max(vMax, decoded[vOff + i]); vSum += decoded[vOff + i]; }

Console.WriteLine($"Y plane: min={yMin}, max={yMax}, mean={ySum / (W * H):F1} (expected ~128)");
Console.WriteLine($"U plane: min={uMin}, max={uMax}, mean={uSum / ((W / 2) * (H / 2)):F1} (expected ~128)");
Console.WriteLine($"V plane: min={vMin}, max={vMax}, mean={vSum / ((W / 2) * (H / 2)):F1} (expected ~128)");

if (yMax < 64) Console.WriteLine("DIAGNOSIS: Y plane is BLACK (mean < 64). Encoder is dropping luma data.");
else if (Math.Abs(ySum / (W * H) - 128) > 16) Console.WriteLine($"DIAGNOSIS: Y plane far from 128 - encoder/decoder pixel mismatch.");
else Console.WriteLine("DIAGNOSIS: Y plane is in the right range.");
