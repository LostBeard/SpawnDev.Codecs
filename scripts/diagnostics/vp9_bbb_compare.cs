// Compare our VP9 walker output to ffmpeg native VP9 decoder output on
// BBB.webm first frame. Measures per-plane delta + zero%.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.IO;
using System.Linq;
using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.Codecs.Video.Vp9;

const int W = 320, H = 180;
string ffmpegYuv = Path.Combine(Path.GetTempPath(), "bbb_ffmpeg_y.yuv");
string webmPath = "SpawnDev.Codecs.Demo.Shared/TestData/Big_Buck_Bunny_180_10s.webm";

if (!File.Exists(webmPath)) { Console.WriteLine($"Missing {webmPath}"); Environment.Exit(1); }
if (!File.Exists(ffmpegYuv)) { Console.WriteLine($"Missing ground truth: {ffmpegYuv} - run ffmpeg first"); Environment.Exit(1); }

var ff = File.ReadAllBytes(ffmpegYuv);
int yLen = W * H;
int uvLen = (W / 2) * (H / 2);
var ffY = ff[0..yLen];
var ffU = ff[yLen..(yLen + uvLen)];
var ffV = ff[(yLen + uvLen)..(yLen + 2 * uvLen)];

// Walk our decoder.
var webmBytes = File.ReadAllBytes(webmPath);
using var ms = new MemoryStream(webmBytes);
var container = new MatroskaContainer(ms);
var videoTrack = container.Tracks.First(t => t.IsVideo);
var firstFrame = container.Frames.First(f => f.TrackNumber == videoTrack.TrackNumber).Data;

var sink = new CaptureSink();
var dec = new Vp9Decoder();
dec.DecodeFrameAsync(firstFrame, sink).GetAwaiter().GetResult();

if (sink.Y is null) { Console.WriteLine("FAIL"); Environment.Exit(1); }

Compare("Y", sink.Y, ffY, W, H);
Compare("U", sink.U!, ffU, W / 2, H / 2);
Compare("V", sink.V!, ffV, W / 2, H / 2);

void Compare(string name, byte[] ours, byte[] ref_, int width, int height)
{
    long oursSum = 0, refSum = 0;
    long absErr = 0, maxAbs = 0;
    int oursMin = 255, oursMax = 0, refMin = 255, refMax = 0;
    int oursZero = 0;
    for (int i = 0; i < ours.Length; i++)
    {
        oursSum += ours[i];
        refSum += ref_[i];
        if (ours[i] < oursMin) oursMin = ours[i];
        if (ours[i] > oursMax) oursMax = ours[i];
        if (ref_[i] < refMin) refMin = ref_[i];
        if (ref_[i] > refMax) refMax = ref_[i];
        if (ours[i] == 0) oursZero++;
        int e = Math.Abs(ours[i] - ref_[i]);
        absErr += e;
        if (e > maxAbs) maxAbs = e;
    }
    double oursMean = oursSum / (double)ours.Length;
    double refMean = refSum / (double)ours.Length;
    double mae = absErr / (double)ours.Length;
    Console.WriteLine($"{name}: ours mean={oursMean,6:F2} range=[{oursMin},{oursMax}]  ffmpeg mean={refMean,6:F2} range=[{refMin},{refMax}]  delta={oursMean - refMean,6:F2} MAE={mae,5:F2} maxAbs={maxAbs} zero%={100.0 * oursZero / ours.Length:F1}");
}

sealed class CaptureSink : SpawnDev.Codecs.Video.IVideoFrameSink
{
    public byte[]? Y, U, V;
    public ValueTask OnFrameAsync(
        ReadOnlyMemory<byte> y, int ys, ReadOnlyMemory<byte> u, int us,
        ReadOnlyMemory<byte> v, int vs, long pts)
    { Y = y.ToArray(); U = u.ToArray(); V = v.ToArray(); return ValueTask.CompletedTask; }
}
