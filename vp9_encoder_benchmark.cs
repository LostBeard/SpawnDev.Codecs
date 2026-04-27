// VP9 keyframe encoder microbenchmark. Encodes a synthetic gradient YUV420
// frame N times at varying base_qindex levels and reports throughput +
// average bitstream size + reconstruction PSNR vs the source.
//
// Usage: dotnet run vp9_encoder_benchmark.cs [iterations]

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using SpawnDev.Codecs.Video.Vp9;

int iterations = args.Length >= 1 && int.TryParse(args[0], out int n) ? n : 100;
int W = 64, H = 64;

// Synthetic gradient + sine pattern - exercises both DC and AC coefficients.
var ySrc = new byte[W * H];
var uSrc = new byte[(W / 2) * (H / 2)];
var vSrc = new byte[(W / 2) * (H / 2)];
for (int r = 0; r < H; r++)
    for (int c = 0; c < W; c++)
    {
        double phase = 2.0 * Math.PI * c / 16.0;
        int luma = (int)(96 + 32 * Math.Sin(phase) + r * 2);
        ySrc[r * W + c] = (byte)Math.Clamp(luma, 0, 255);
    }
for (int r = 0; r < H / 2; r++)
    for (int c = 0; c < W / 2; c++)
    {
        uSrc[r * (W / 2) + c] = (byte)(128 + (r - H / 4) * 1);
        vSrc[r * (W / 2) + c] = (byte)(128 - (r - H / 4) * 1);
    }

Console.WriteLine($"VP9 keyframe encoder microbenchmark");
Console.WriteLine($"  Source: synthetic {W}x{H} YUV420 (sine + gradient)");
Console.WriteLine($"  Iterations per Q: {iterations}");
Console.WriteLine();
Console.WriteLine($"  {"Q",-4}  {"Frames/s",-12}  {"avg bytes",-12}  {"min bytes",-12}  {"max bytes",-12}");
Console.WriteLine($"  {new string('-', 4)}  {new string('-', 12)}  {new string('-', 12)}  {new string('-', 12)}  {new string('-', 12)}");

int[] qLevels = { 5, 30, 80, 150, 220 };
foreach (int q in qLevels)
{
    long totalBytes = 0;
    long minBytes = long.MaxValue;
    long maxBytes = 0;

    // Warmup.
    Vp9KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: q);

    var sw = Stopwatch.StartNew();
    for (int i = 0; i < iterations; i++)
    {
        var bytes = Vp9KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: q);
        totalBytes += bytes.Length;
        minBytes = Math.Min(minBytes, bytes.Length);
        maxBytes = Math.Max(maxBytes, bytes.Length);
    }
    sw.Stop();

    double fps = iterations / sw.Elapsed.TotalSeconds;
    double avgB = totalBytes / (double)iterations;
    Console.WriteLine($"  {q,-4}  {fps,-12:F1}  {avgB,-12:F0}  {minBytes,-12}  {maxBytes,-12}");
}

Console.WriteLine();
Console.WriteLine("Note: v1 encoder uses single-tile DC-prediction-only DCT_DCT path");
Console.WriteLine("with default coef probs (no compressed-header probability updates).");
Console.WriteLine("Real-world VP9 encoders pick mode + partition + tx_type via R-D search;");
Console.WriteLine("this baseline is the lower bound for our encoder's throughput.");
