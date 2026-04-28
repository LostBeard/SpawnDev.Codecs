// VP9 frame timeline demo: per-frame metadata report on a real VP9
// stream using the SpawnDev.Codecs Vp9 decoder pipeline. Prints
// frame_type, show_frame, dimensions, profile, subsampling, bit
// depth, tx_mode, ref_mode, tile count for every coded frame in
// Big_Buck_Bunny_180_10s.webm.
//
// Mirrors av1_frame_timeline.cs for VP9.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.IO;
using System.Linq;
using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.Codecs.Video;
using SpawnDev.Codecs.Video.Vp9;

string webmPath = "SpawnDev.Codecs.Demo.Shared/TestData/Big_Buck_Bunny_180_10s.webm";
var bytes = File.ReadAllBytes(webmPath);
using var stream = new MemoryStream(bytes);
var container = new MatroskaContainer(stream);
var video = container.Tracks.First(t => t.IsVideo);

Console.WriteLine();
Console.WriteLine("============================================================");
Console.WriteLine("  VP9 frame timeline (per-frame metadata)");
Console.WriteLine("============================================================");
Console.WriteLine();
Console.WriteLine($"Source: {webmPath}");
Console.WriteLine($"Container: {container.DocType} (WebM)");
Console.WriteLine($"Video codec: {video.CodecId}");
Console.WriteLine();

// Use the high-level analyzer for the summary header.
var packetEnumerable = container.Frames
    .Where(f => f.TrackNumber == video.TrackNumber)
    .Select(f => (ReadOnlyMemory<byte>)f.Data);
var summary = Vp9StreamAnalyzer.Analyze(packetEnumerable);
Console.WriteLine(summary.ToReport());
Console.WriteLine();

// Re-enumerate for the per-packet timeline (analyzer consumed once).
var decoder = new Vp9Decoder();
var sink = new CountingSink();
int packetIdx = 0;

Console.WriteLine($"{"Pkt",-4} {"Type",-8} {"Show",-5} {"Size",-9} {"Profile",-7} {"Sub",-5} {"Depth",-5} {"TxMode",-12} {"RefMode",-15} {"Tiles",-5}");
Console.WriteLine($"{new string('-', 90)}");

int kfRows = 0, frameRows = 0;
foreach (var frame in container.Frames.Where(f => f.TrackNumber == video.TrackNumber))
{
    packetIdx++;
    await decoder.DecodeFrameAsync(frame.Data, sink);
    var fh = decoder.LastFrameHeader;
    if (fh is null) continue;

    string type = fh.FrameType.ToString();
    string size = $"{fh.FrameWidth}x{fh.FrameHeight}";
    string sub = fh.SubsamplingX && fh.SubsamplingY ? "4:2:0"
        : !fh.SubsamplingX && !fh.SubsamplingY ? "4:4:4"
        : "?";
    string depth = $"{fh.BitDepth}";
    string txMode = decoder.LastCompressedResult?.TxMode.ToString() ?? "-";
    string refMode = decoder.LastCompressedResult?.ReferenceMode.ToString() ?? "-";
    int tileCount = decoder.LastTileGroup?.Tiles.Count ?? 0;

    if (fh.FrameType == Vp9FrameType.Key) kfRows++;
    frameRows++;

    // Only print first ~30 + every 30th to keep output manageable for 300 frames.
    if (frameRows <= 30 || frameRows % 30 == 0)
    {
        Console.WriteLine($"{packetIdx,-4} {type,-8} {fh.ShowFrame,-5} {size,-9} {fh.Profile,-7} {sub,-5} {depth,-5} {txMode,-12} {refMode,-15} {tileCount,-5}");
    }
}

Console.WriteLine();
Console.WriteLine($"Packets processed:        {packetIdx}");
Console.WriteLine($"Coded frames:             {decoder.TotalCodedFrames}");
Console.WriteLine($"Visible frames emitted:   {decoder.TotalVisibleFrames}");
Console.WriteLine($"Cumulative frame types:   "
    + string.Join(", ", decoder.CumulativeFrameTypeCounts.OrderByDescending(kv => kv.Value)
        .Select(kv => $"{kv.Key}={kv.Value}")));
Console.WriteLine($"ShowExistingFrame count:  {decoder.ShowExistingFrameCount}");

internal sealed class CountingSink : IVideoFrameSink
{
    public int Count;
    public ValueTask OnFrameAsync(
        ReadOnlyMemory<byte> y, int ys,
        ReadOnlyMemory<byte> u, int us,
        ReadOnlyMemory<byte> v, int vs,
        long pts)
    {
        Count++;
        return ValueTask.CompletedTask;
    }
}
