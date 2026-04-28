// Quick check: what fraction of the chroma plane is left at zero (init
// value) after the AV1 walker decodes the BBB first keyframe? If many
// pixels are 0, chroma decode is missing them. If all are non-zero but
// values are wrong, decode logic is buggy.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.IO;
using System.Linq;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Av1;

string ivfPath = "SpawnDev.Codecs.Demo.Shared/TestData/bbb_180_2s.ivf";
if (!File.Exists(ivfPath))
{
    Console.WriteLine($"Missing fixture: {ivfPath}");
    Environment.Exit(1);
}
var ivfBytes = File.ReadAllBytes(ivfPath);
var firstFrame = IvfReader.EnumerateFrames(ivfBytes).First();

Av1SequenceHeader? sh = null;
Av1Obu? frameObu = null;
foreach (var obu in Av1ObuParser.EnumerateObus(firstFrame.Data))
{
    if (obu.Type == Av1ObuType.SequenceHeader)
        sh = Av1SequenceHeaderParser.Parse(firstFrame.Data.Span.Slice(obu.PayloadOffset, obu.PayloadLength));
    else if (obu.Type == Av1ObuType.Frame)
        frameObu = obu;
}
if (sh is null || !frameObu.HasValue) { Console.WriteLine("missing SH or Frame OBU"); Environment.Exit(1); }
var payload = firstFrame.Data.Slice(frameObu.Value.PayloadOffset, frameObu.Value.PayloadLength);
var complete = Av1CompleteFrameHeaderParser.Parse(payload.Span, sh);
var tg = Av1TileGroupExtractor.Extract(payload.Span, complete);

var walker = new Av1KeyframeWalker();
var fb = walker.DecodeFrame(payload, sh, complete, tg);

Console.WriteLine($"Frame: {fb.LumaWidth}x{fb.LumaHeight}, chroma {fb.ChromaWidth}x{fb.ChromaHeight}");
Console.WriteLine();
PrintCoverage("Y", fb.Y);
PrintCoverage("U", fb.U);
PrintCoverage("V", fb.V);

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
    // Top 5 most common values.
    var topValues = hist
        .Select((cnt, val) => (val, cnt))
        .OrderByDescending(t => t.cnt)
        .Take(5)
        .ToArray();
    Console.WriteLine($"   top 5: {string.Join(", ", topValues.Select(t => $"{t.val}={t.cnt}"))}");
}
