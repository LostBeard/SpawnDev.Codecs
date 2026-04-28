// VP9 walker coverage check on BBB - measure per-plane zero%, mean,
// range to localize chroma drift bugs.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.IO;
using System.Linq;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.Codecs.Container.Matroska;

string webmPath = "SpawnDev.Codecs.Demo.Shared/TestData/Big_Buck_Bunny_180_10s.webm";
if (!File.Exists(webmPath))
{
    Console.WriteLine($"Missing fixture: {webmPath}");
    Environment.Exit(1);
}

// Locate first VP9 keyframe in BBB.webm via the EBML container parser.
var webmBytes = File.ReadAllBytes(webmPath);
Console.WriteLine($"BBB.webm size: {webmBytes.Length}B");

// Use the existing Vp9Decoder + a capture sink to drive the walker.
var sink = new CaptureSink();
var dec = new Vp9Decoder();

// Find the first VP9 packet via the MatroskaContainer reader.
using var ms = new MemoryStream(webmBytes);
var container = new MatroskaContainer(ms);
var videoTrack = container.Tracks.First(t => t.IsVideo);
var firstFrame = container.Frames.First(f => f.TrackNumber == videoTrack.TrackNumber).Data;
Console.WriteLine($"First VP9 packet: {firstFrame.Length}B");

int n = dec.DecodeFrameAsync(firstFrame, sink).GetAwaiter().GetResult();
Console.WriteLine($"DecodeFrameAsync returned {n}; W={dec.Width} H={dec.Height}");

if (sink.Y is null) { Console.WriteLine("FAIL: no frame captured"); Environment.Exit(1); }

PrintCoverage("Y", sink.Y);
PrintCoverage("U", sink.U!);
PrintCoverage("V", sink.V!);

// To get exact ffmpeg ground truth means, run:
//   ffmpeg -i Big_Buck_Bunny_180_10s.webm -frames:v 1 -f rawvideo -pix_fmt yuv420p out.yuv
// then sum each plane.

void PrintCoverage(string name, byte[] plane)
{
    int zeroCount = 0;
    int min = 255, max = 0;
    long sum = 0;
    var hist = new int[256];
    foreach (var b in plane)
    {
        if (b == 0) zeroCount++;
        if (b < min) min = b;
        if (b > max) max = b;
        sum += b;
        hist[b]++;
    }
    double mean = sum / (double)plane.Length;
    double zeroPct = 100.0 * zeroCount / plane.Length;
    Console.WriteLine($"{name}: mean={mean:F2} min={min} max={max} zero%={zeroPct:F1} ({zeroCount}/{plane.Length})");
    var topValues = hist
        .Select((cnt, val) => (val, cnt))
        .OrderByDescending(t => t.cnt)
        .Take(5)
        .ToArray();
    Console.WriteLine($"   top 5: {string.Join(", ", topValues.Select(t => $"{t.val}={t.cnt}"))}");
}

sealed class CaptureSink : SpawnDev.Codecs.Video.IVideoFrameSink
{
    public byte[]? Y;
    public byte[]? U;
    public byte[]? V;
    public ValueTask OnFrameAsync(
        ReadOnlyMemory<byte> y, int ys,
        ReadOnlyMemory<byte> u, int us,
        ReadOnlyMemory<byte> v, int vs,
        long pts)
    {
        Y = y.ToArray();
        U = u.ToArray();
        V = v.ToArray();
        return ValueTask.CompletedTask;
    }
}
