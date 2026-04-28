// Demo: decode the FIRST FULL KEYFRAME of BBB.webm via the VP9
// keyframe walker and compare against ffmpeg ground truth.
//
// Composes Vp9SuperframeParser + Vp9CompleteUncompressedHeaderParser
// + Vp9CompressedHeaderParser + Vp9TileGroupExtractor + the new
// Vp9KeyframeWalker. Loop filter is OUT OF SCOPE - output will be
// blocky vs ffmpeg's loop-filtered reference but should be the
// recognizable BBB scene at correct mean / variance.

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.IO;
using System.Linq;
using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.Codecs.Video;
using SpawnDev.Codecs.Video.Vp9;

string webmPath = "SpawnDev.Codecs.Demo.Shared/TestData/Big_Buck_Bunny_180_10s.webm";
string ffmpegYuvPath = "SpawnDev.Codecs.Demo.Shared/TestData/bbb_first_frame.yuv";

using var stream = File.OpenRead(webmPath);
var container = new MatroskaContainer(stream);
var video = container.Tracks.First(t => t.IsVideo);
var first = container.Frames.First(f => f.TrackNumber == video.TrackNumber);

// 1. Drive the existing decoder pipeline to extract the parsed
//    header, compressed-header state, and tile group for this frame.
var decoder = new Vp9Decoder();
await decoder.DecodeFrameAsync(first.Data, new IgnoreSink());

var header = decoder.LastCompleteHeader!;
var state = decoder.LastCompressedState!;
var result = decoder.LastCompressedResult!;
var tileGroup = decoder.LastTileGroup!;

Console.WriteLine();
Console.WriteLine("============================================================");
Console.WriteLine("  VP9 FULL FRAME DECODE - BBB first keyframe");
Console.WriteLine("============================================================");
Console.WriteLine();
Console.WriteLine($"Source: {webmPath}");
Console.WriteLine($"Frame size: {header.FrameHeader.FrameWidth}x{header.FrameHeader.FrameHeight}");
Console.WriteLine($"Subsampling: ssX={(header.FrameHeader.SubsamplingX ? 1 : 0)} ssY={(header.FrameHeader.SubsamplingY ? 1 : 0)}");
Console.WriteLine($"Bit depth: {header.FrameHeader.BitDepth}");
Console.WriteLine($"Tile cols: {header.TileInfo.TileCols}, Tile rows: {header.TileInfo.TileRows}");
Console.WriteLine($"Compressed header tx_mode = {result.TxMode}, ref_mode = {result.ReferenceMode}");
Console.WriteLine($"Quantization base_q_idx = {header.Quantization.BaseQIndex}, lossless = {header.Quantization.Lossless}");
Console.WriteLine($"Segmentation enabled: {header.Segmentation.Enabled}");
Console.WriteLine();

// 2. Run the full-frame walker.
Console.WriteLine("Decoding full keyframe via Vp9KeyframeWalker...");
var walker = new Vp9KeyframeWalker { Trace = new List<Vp9KeyframeWalker.DecodedBlockTrace>() };
Vp9FrameBuffer fb;
try
{
    fb = walker.DecodeFrame(first.Data, header, state, result, tileGroup);
}
catch (Exception ex)
{
    Console.WriteLine();
    Console.WriteLine($"DECODE THREW: {ex.GetType().Name}: {ex.Message}");
    Console.WriteLine();
    if (ex.StackTrace is { } st)
    {
        var lines = st.Split('\n');
        for (int i = 0; i < Math.Min(15, lines.Length); i++)
            Console.WriteLine("  " + lines[i].Trim());
    }
    return;
}

Console.WriteLine($"Decoded successfully.");
Console.WriteLine($"  Total leaf blocks decoded: {walker.Trace?.Count ?? -1}");
if (walker.Trace is { } tr)
{
    Console.WriteLine("  First 16 blocks decoded:");
    for (int i = 0; i < Math.Min(16, tr.Count); i++)
    {
        var b = tr[i];
        Console.WriteLine($"    {i,3}: mi=({b.MiRow,2},{b.MiCol,2}) bsize={b.Bsize} tx={b.TxSize} y={b.YMode} uv={b.UvMode} skip={b.Skip} skipCtx={b.SkipContext} txCtx={b.TxSizeContext}");
    }
}
Console.WriteLine($"  Y plane: {fb.LumaWidth}x{fb.LumaHeight} = {fb.Y.Length} bytes");
Console.WriteLine($"  U plane: {fb.ChromaWidth}x{fb.ChromaHeight} = {fb.U.Length} bytes");
Console.WriteLine($"  V plane: {fb.ChromaWidth}x{fb.ChromaHeight} = {fb.V.Length} bytes");
Console.WriteLine();

// 3. Statistics on our decoded Y plane.
PrintPlaneStats("Our Y", fb.Y);
PrintPlaneStats("Our U", fb.U);
PrintPlaneStats("Our V", fb.V);
// Coverage: count pixels that are non-zero (a good proxy for "decoded vs left blank").
int coverNz = 0;
for (int i = 0; i < fb.Y.Length; i++) if (fb.Y[i] != 0) coverNz++;
Console.WriteLine($"  Coverage Y (non-zero px): {coverNz}/{fb.Y.Length} ({100.0 * coverNz / fb.Y.Length:F1}%)");
// Per-row analysis: is the decoder just running out at some Y row?
Console.WriteLine($"  Per-row mean (every 16 rows):");
for (int row = 0; row < fb.LumaHeight; row += 16)
{
    long sum = 0;
    int min = 255, max = 0;
    for (int c = 0; c < fb.LumaWidth; c++)
    {
        var v = fb.Y[row * fb.LumaWidth + c];
        sum += v;
        if (v < min) min = v;
        if (v > max) max = v;
    }
    Console.WriteLine($"    row {row,3}: mean={sum / fb.LumaWidth,3} min={min,3} max={max,3}");
}
Console.WriteLine();

// 4. Load ffmpeg ground truth for the same frame.
var gtBytes = File.ReadAllBytes(ffmpegYuvPath);
int yLen = fb.LumaWidth * fb.LumaHeight;
int uvLen = fb.ChromaWidth * fb.ChromaHeight;
if (gtBytes.Length != yLen + 2 * uvLen)
{
    Console.WriteLine($"WARN: ground truth size {gtBytes.Length} != expected {yLen + 2 * uvLen}");
}
var gtY = gtBytes.AsSpan(0, yLen).ToArray();
var gtU = gtBytes.AsSpan(yLen, uvLen).ToArray();
var gtV = gtBytes.AsSpan(yLen + uvLen, uvLen).ToArray();

PrintPlaneStats("ffmpeg Y", gtY);
PrintPlaneStats("ffmpeg U", gtU);
PrintPlaneStats("ffmpeg V", gtV);
Console.WriteLine();

// 5. Mean absolute difference per plane.
Console.WriteLine("--- Pixel difference (our - ffmpeg) ---");
PrintDiffStats("Y", fb.Y, gtY);
PrintDiffStats("U", fb.U, gtU);
PrintDiffStats("V", fb.V, gtV);
Console.WriteLine();

// 6b. Per-16x16 block diff for Y plane (find where it goes off the rails).
Console.WriteLine("--- Per-16x16 Y block MAE (rough heatmap) ---");
int blocksW = fb.LumaWidth / 16;
int blocksH = fb.LumaHeight / 16;
for (int by = 0; by < blocksH; by++)
{
    Console.Write($"  row{by * 16,3}:");
    for (int bx = 0; bx < blocksW; bx++)
    {
        long sum = 0;
        for (int r = 0; r < 16; r++)
        {
            for (int c = 0; c < 16; c++)
            {
                int p = (by * 16 + r) * fb.LumaWidth + (bx * 16 + c);
                int d = fb.Y[p] - gtY[p];
                sum += d < 0 ? -d : d;
            }
        }
        int mae = (int)(sum / 256);
        Console.Write($" {mae,3}");
    }
    Console.WriteLine();
}
Console.WriteLine();

// 6. First 16 pixels of top row + left col, side by side.
Console.WriteLine("--- First 16 px of Y top row ---");
Console.Write("  Ours  : ");
for (int i = 0; i < 16; i++) Console.Write($"{fb.Y[i],4}");
Console.WriteLine();
Console.Write("  ffmpeg: ");
for (int i = 0; i < 16; i++) Console.Write($"{gtY[i],4}");
Console.WriteLine();

Console.WriteLine();
Console.WriteLine("--- First 16 px of Y left col ---");
Console.Write("  Ours  : ");
for (int i = 0; i < 16; i++) Console.Write($"{fb.Y[i * fb.LumaWidth],4}");
Console.WriteLine();
Console.Write("  ffmpeg: ");
for (int i = 0; i < 16; i++) Console.Write($"{gtY[i * fb.LumaWidth],4}");
Console.WriteLine();

Console.WriteLine();
Console.WriteLine("--- First 16x16 Y block (top-left) - ours ---");
for (int r = 0; r < 4; r++)
{
    Console.Write("  ");
    for (int c = 0; c < 16; c++)
        Console.Write($"{fb.Y[r * fb.LumaWidth + c],4}");
    Console.WriteLine();
}
Console.WriteLine();
Console.WriteLine("--- First 16x16 Y block (top-left) - ffmpeg ---");
for (int r = 0; r < 4; r++)
{
    Console.Write("  ");
    for (int c = 0; c < 16; c++)
        Console.Write($"{gtY[r * fb.LumaWidth + c],4}");
    Console.WriteLine();
}

Console.WriteLine();
Console.WriteLine("Loop filter is OUT OF SCOPE for this slice; expect blocky output");
Console.WriteLine("vs ffmpeg's loop-filtered reference. The scene content should be");
Console.WriteLine("recognizable (similar mean / variance / range).");

static void PrintPlaneStats(string name, ReadOnlySpan<byte> plane)
{
    int min = 255, max = 0;
    long sum = 0;
    for (int i = 0; i < plane.Length; i++)
    {
        var v = plane[i];
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
    }
    double mean = (double)sum / plane.Length;
    // Variance.
    double varSum = 0;
    for (int i = 0; i < plane.Length; i++)
    {
        double d = plane[i] - mean;
        varSum += d * d;
    }
    double variance = varSum / plane.Length;
    Console.WriteLine($"  {name,-10}: min={min,3} max={max,3} mean={mean:F2} var={variance:F1}");
}

static void PrintDiffStats(string name, ReadOnlySpan<byte> ours, ReadOnlySpan<byte> truth)
{
    int n = Math.Min(ours.Length, truth.Length);
    long absSum = 0;
    int maxAbs = 0;
    int exactMatches = 0;
    for (int i = 0; i < n; i++)
    {
        int d = ours[i] - truth[i];
        int abs = d < 0 ? -d : d;
        absSum += abs;
        if (abs > maxAbs) maxAbs = abs;
        if (abs == 0) exactMatches++;
    }
    double mae = (double)absSum / n;
    double exactPct = 100.0 * exactMatches / n;
    Console.WriteLine($"  {name}: MAE={mae:F2} max_abs={maxAbs} exact={exactMatches}/{n} ({exactPct:F1}%)");
}

internal sealed class IgnoreSink : IVideoFrameSink
{
    public ValueTask OnFrameAsync(
        ReadOnlyMemory<byte> y, int ys,
        ReadOnlyMemory<byte> u, int us,
        ReadOnlyMemory<byte> v, int vs,
        long pts) => ValueTask.CompletedTask;
}
