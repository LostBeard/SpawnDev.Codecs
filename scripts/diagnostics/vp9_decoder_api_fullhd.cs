// Verify Vp9Decoder.DecodeFrameAsync (public API surface) handles
// FullHD content end-to-end. Encoder produces 1920x1072 frame; the
// decoder consumes it through the IVideoDecoder interface and emits
// pixels to a sink.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using SpawnDev.Codecs.Video;
using SpawnDev.Codecs.Video.Vp9;

const int W = 1920, H = 1072;

// Simple flat frame to keep encode time bounded.
var ySrc = new byte[W * H]; Array.Fill(ySrc, (byte)128);
var uSrc = new byte[(W / 2) * (H / 2)]; Array.Fill(uSrc, (byte)128);
var vSrc = new byte[(W / 2) * (H / 2)]; Array.Fill(vSrc, (byte)128);

var swEnc = Stopwatch.StartNew();
var frame = Vp9KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 30);
swEnc.Stop();
Console.WriteLine($"VP9 encoded {W}x{H}: {frame.Length}B in {swEnc.Elapsed.TotalSeconds:F2}s");

var sink = new CaptureSink();
var dec = new Vp9Decoder();
var swDec = Stopwatch.StartNew();
int n = dec.DecodeFrameAsync(frame, sink).GetAwaiter().GetResult();
swDec.Stop();
Console.WriteLine($"Vp9Decoder API decoded: {n} frame(s) in {swDec.Elapsed.TotalSeconds:F2}s");
Console.WriteLine($"Decoder reports W={dec.Width} H={dec.Height}");

if (sink.Y is null) { Console.WriteLine("FAIL: no frame"); Environment.Exit(1); }
if (sink.Y.Length != W * H) { Console.WriteLine($"FAIL: Y plane {sink.Y.Length}, expected {W * H}"); Environment.Exit(1); }

long sum = 0; int min = 255, max = 0;
foreach (var b in sink.Y) { sum += b; if (b < min) min = b; if (b > max) max = b; }
int mean = (int)(sum / sink.Y.Length);
Console.WriteLine($"Y plane: mean={mean} range=[{min},{max}]");

if (Math.Abs(mean - 128) > 4)
{
    Console.WriteLine($"FAIL: mean {mean} drifted from 128 (expected flat input)");
    Environment.Exit(1);
}
Console.WriteLine();
Console.WriteLine("=== Vp9Decoder API FullHD round-trip OK ===");

sealed class CaptureSink : IVideoFrameSink
{
    public byte[]? Y;
    public ValueTask OnFrameAsync(
        ReadOnlyMemory<byte> y, int ys, ReadOnlyMemory<byte> u, int us,
        ReadOnlyMemory<byte> v, int vs, long pts)
    { Y = y.ToArray(); return ValueTask.CompletedTask; }
}
