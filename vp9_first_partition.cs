// Demo: read the first partition decision from BBB.webm's first frame.
// Composes Vp9BoolDecoder + Vp9PartitionProbs + Vp9PartitionTree on
// real VP9 tile bytes to show the entropy decode path producing a
// meaningful symbol.

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.IO;
using System.Linq;
using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.Codecs.Video;
using SpawnDev.Codecs.Video.Vp9;

string webmPath = "SpawnDev.Codecs.Demo.Shared/TestData/Big_Buck_Bunny_180_10s.webm";
using var stream = File.OpenRead(webmPath);
var container = new MatroskaContainer(stream);
var video = container.Tracks.First(t => t.IsVideo);
var first = container.Frames.First(f => f.TrackNumber == video.TrackNumber);

var decoder = new Vp9Decoder();
var sink = new IgnoreSink();
await decoder.DecodeFrameAsync(first.Data, sink);

Console.WriteLine();
Console.WriteLine("============================================================");
Console.WriteLine("  VP9 first-partition decode demo");
Console.WriteLine("============================================================");
Console.WriteLine();
Console.WriteLine($"Source: {webmPath}");
Console.WriteLine($"Frame: KeyFrame {decoder.LastFrameHeader?.FrameWidth}x{decoder.LastFrameHeader?.FrameHeight}, "
    + $"{decoder.LastTileGroup?.Tiles.Count} tile(s)");
Console.WriteLine();
Console.WriteLine($"Compressed header parsed:");
Console.WriteLine($"  tx_mode  = {decoder.LastCompressedResult?.TxMode}");
Console.WriteLine($"  ref_mode = {decoder.LastCompressedResult?.ReferenceMode}");
Console.WriteLine();

// Initialize bool decoder over tile 0 bytes.
var tile0 = decoder.LastTileGroup!.Tiles[0];
var data = first.Data.ToArray();
var tileBytes = new byte[tile0.Length];
Buffer.BlockCopy(data, tile0.Offset, tileBytes, 0, tile0.Length);

Console.WriteLine($"Tile 0: offset={tile0.Offset}, length={tile0.Length}");

// First superblock at top-left: 64x64 (sizeIdx=3), both above + left
// out of frame so splitState=0.
var br = new Vp9BoolDecoder(tileBytes, 0, tileBytes.Length);
var probs = Vp9PartitionProbs.DefaultProbs(sizeIdx: 3, splitState: 0);
Console.WriteLine($"Partition probs (64x64, both unsplit context): "
    + $"None_vs_else={probs[0]}, Horz_vs_else={probs[1]}, Vert_vs_Split={probs[2]}");

var partition = Vp9PartitionTree.Decode(p => br.Read(p), probs);

Console.WriteLine();
Console.WriteLine("============================================================");
Console.WriteLine($"  FIRST PARTITION DECISION (top-left 64x64): {partition}");
Console.WriteLine("============================================================");
Console.WriteLine();
Console.WriteLine("Interpretation:");
switch (partition)
{
    case Vp9PartitionType.None:
        Console.WriteLine("  -> Block is decoded as one 64x64 transform block.");
        break;
    case Vp9PartitionType.Horz:
        Console.WriteLine("  -> Block split horizontally: 64x32 top + 64x32 bottom.");
        break;
    case Vp9PartitionType.Vert:
        Console.WriteLine("  -> Block split vertically: 32x64 left + 32x64 right.");
        break;
    case Vp9PartitionType.Split:
        Console.WriteLine("  -> Block split into 4 quarter-sized 32x32 sub-blocks; recurse.");
        break;
}

internal sealed class IgnoreSink : IVideoFrameSink
{
    public ValueTask OnFrameAsync(
        ReadOnlyMemory<byte> y, int ys,
        ReadOnlyMemory<byte> u, int us,
        ReadOnlyMemory<byte> v, int vs,
        long pts) => ValueTask.CompletedTask;
}
