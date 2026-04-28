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

// === Multi-MB exercise (32x32 = 4 MBs) ===
// Confirms the decoder works past a single MB. Uses the same gradient
// pattern the verify harness produces.
{
    const int W2 = 32, H2 = 32;
    var ySrc2 = new byte[W2 * H2];
    for (int r = 0; r < H2; r++)
        for (int c = 0; c < W2; c++)
            ySrc2[r * W2 + c] = (byte)Math.Clamp(80 + 40 * Math.Sin(2.0 * Math.PI * c / W2) + r * 2, 0, 255);
    var uSrc2 = new byte[(W2 / 2) * (H2 / 2)]; Array.Fill(uSrc2, (byte)128);
    var vSrc2 = new byte[(W2 / 2) * (H2 / 2)]; Array.Fill(vSrc2, (byte)128);
    var frame2 = Vp8KeyframeEncoder.EncodeKeyFrame(ySrc2, W2, uSrc2, W2 / 2, vSrc2, W2, H2, baseQIndex: 30);
    var sink2 = new CaptureSink();
    var dec2 = new Vp8Decoder();
    int n2 = await dec2.DecodeFrameAsync(frame2, sink2);
    if (n2 != 1 || sink2.LastY!.Length != W2 * H2) { Console.WriteLine($"FAIL: multi-MB frame, n={n2} len={sink2.LastY?.Length}"); Environment.Exit(1); }
    int srcMin = 255, srcMax = 0; foreach (var b in ySrc2) { if (b < srcMin) srcMin = b; if (b > srcMax) srcMax = b; }
    int outMin = 255, outMax = 0; foreach (var b in sink2.LastY!) { if (b < outMin) outMin = b; if (b > outMax) outMax = b; }
    Console.WriteLine($"PASS: 32x32 multi-MB frame decoded; source Y range=[{srcMin}, {srcMax}], decoded Y range=[{outMin}, {outMax}]");
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
