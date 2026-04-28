// VP8 decoder API smoke: encode a flat-Y=128 keyframe with our encoder,
// decode it through the public Vp8Decoder.DecodeFrameAsync, capture the
// frame in a sink, and report mean luma. Verifies the keyframe walker is
// now wired into the IVideoDecoder surface.

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Threading.Tasks;
using SpawnDev.Codecs.Video;
using SpawnDev.Codecs.Video.Vp8;

const int W = 16, H = 16;

var ySrc = new byte[W * H]; Array.Fill(ySrc, (byte)128);
var uSrc = new byte[(W / 2) * (H / 2)]; Array.Fill(uSrc, (byte)128);
var vSrc = new byte[(W / 2) * (H / 2)]; Array.Fill(vSrc, (byte)128);

var frame = Vp8KeyframeEncoder.EncodeKeyFrame(
    ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 30);
Console.WriteLine($"Encoded {frame.Length} bytes for {W}x{H} flat Y=128 / UV=128.");

var sink = new CaptureSink();
await using var dec = new Vp8Decoder();
int frames = await dec.DecodeFrameAsync(frame, sink);
Console.WriteLine($"DecodeFrameAsync returned {frames}; sink received {sink.FrameCount} frame(s).");
Console.WriteLine($"Decoder reports Width={dec.Width}, Height={dec.Height}, KeyFrameCount={dec.KeyFrameCount}.");

if (sink.LastY is null) { Console.WriteLine("FAIL: no Y plane captured"); Environment.Exit(1); }
long sumY = 0; int minY = 255, maxY = 0;
foreach (var b in sink.LastY) { sumY += b; if (b < minY) minY = b; if (b > maxY) maxY = b; }
int meanY = (int)(sumY / sink.LastY.Length);
Console.WriteLine($"Y plane: mean={meanY}, range=[{minY}, {maxY}] (expected ~128)");

if (Math.Abs(meanY - 128) > 8)
{
    Console.WriteLine("FAIL: Y mean too far from 128");
    Environment.Exit(1);
}

// Verify inter-frame rejection.
try
{
    await dec.DecodeFrameAsync(new byte[] { 0x01, 0x00, 0x00 }, sink);
    Console.WriteLine("FAIL: inter frame should have thrown");
    Environment.Exit(1);
}
catch (NotImplementedException ex)
{
    Console.WriteLine($"PASS: inter-frame rejected with descriptive message ({ex.Message[..Math.Min(60, ex.Message.Length)]}...)");
}

Console.WriteLine();
Console.WriteLine("=== Vp8Decoder API smoke OK ===");

sealed class CaptureSink : IVideoFrameSink
{
    public int FrameCount { get; private set; }
    public byte[]? LastY { get; private set; }
    public byte[]? LastU { get; private set; }
    public byte[]? LastV { get; private set; }

    public ValueTask OnFrameAsync(
        ReadOnlyMemory<byte> y, int ys,
        ReadOnlyMemory<byte> u, int us,
        ReadOnlyMemory<byte> v, int vs,
        long pts)
    {
        FrameCount++;
        LastY = y.ToArray();
        LastU = u.ToArray();
        LastV = v.ToArray();
        return ValueTask.CompletedTask;
    }
}
