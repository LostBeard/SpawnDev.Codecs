// AV1 keyframe decode demo - demonstrates the AV1 decoder pipeline
// against the BBB bbb_180_2s.ivf fixture and compares the (currently
// placeholder) output against the libdav1d/ffmpeg ground truth.
//
// What this demo shows TODAY:
//   - End-to-end decode pipeline from IVF -> OBU enumeration ->
//     SequenceHeader parse -> CompleteFrameHeader parse (with all
//     of tile_info / quant / segmentation / lf / cdef / lr) ->
//     TileGroup byte-range extraction -> KeyframeWalker.
//   - The walker correctly walks superblocks within the parsed
//     tile geometry until the partition decode point, then throws
//     NotImplementedException because the AV1 partition CDF tables
//     are still pending (see Av1KeyframeWalker for the gap detail).
//   - Pixel comparison shows the placeholder mid-gray output vs
//     ffmpeg ground truth: Y mean=97.40, U mean=109, V mean=125 from
//     ffmpeg vs 128/128/128 placeholder from us.
//
// Once the partition + mode info + coefficient + intra prediction +
// reconstruction pieces land (the partition-tree CDF tables are the
// gating dependency), the walker will produce real pixels and the
// comparison will close to within 5% as required by the integration
// target.

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.IO;
using System.Linq;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Av1;

Console.WriteLine();
Console.WriteLine("==========================================================");
Console.WriteLine("  AV1 keyframe decode demo (BBB 320x180 fixture)");
Console.WriteLine("==========================================================");

string testDataDir = "SpawnDev.Codecs.Demo.Shared/TestData";
string ivfPath = Path.Combine(testDataDir, "bbb_180_2s.ivf");
string gtPath = Path.Combine(testDataDir, "bbb_av1_first_frame.yuv");

if (!File.Exists(ivfPath))
{
    Console.WriteLine($"ERROR: Missing AV1 fixture {ivfPath}");
    return 1;
}
if (!File.Exists(gtPath))
{
    Console.WriteLine($"WARN: Missing ffmpeg ground truth {gtPath} (will skip comparison)");
}

// Load IVF + first frame
var ivfBytes = File.ReadAllBytes(ivfPath);
var firstIvfFrame = IvfReader.EnumerateFrames(ivfBytes).First();
Console.WriteLine($"\nIVF first frame: {firstIvfFrame.Data.Length} bytes, pts={firstIvfFrame.Pts}");

// Walk the OBUs
Av1SequenceHeader? sh = null;
Av1Obu? frameObu = null;
foreach (var obu in Av1ObuParser.EnumerateObus(firstIvfFrame.Data))
{
    Console.WriteLine($"  OBU {obu.Type} payload @ {obu.PayloadOffset} length {obu.PayloadLength}");
    if (obu.Type == Av1ObuType.SequenceHeader)
    {
        sh = Av1SequenceHeaderParser.Parse(
            firstIvfFrame.Data.Span.Slice(obu.PayloadOffset, obu.PayloadLength));
    }
    else if (obu.Type == Av1ObuType.Frame)
    {
        frameObu = obu;
    }
}

if (sh is null || !frameObu.HasValue)
{
    Console.WriteLine("ERROR: missing SH or Frame OBU in first IVF packet");
    return 1;
}

Console.WriteLine($"\nSequenceHeader: profile={sh.SeqProfile} {sh.MaxFrameWidth}x{sh.MaxFrameHeight}");
Console.WriteLine($"  bit_depth={sh.BitDepth} mono={sh.Monochrome} subX={sh.SubsamplingX} subY={sh.SubsamplingY}");
Console.WriteLine($"  use128SB={sh.Use128x128Superblock} EnableCdef={sh.EnableCdef} EnableRestoration={sh.EnableRestoration}");
Console.WriteLine($"  EnableSuperres={sh.EnableSuperres} EnableOrderHint={sh.EnableOrderHint}");

// Parse the complete frame header
var framePayload = firstIvfFrame.Data.Slice(frameObu.Value.PayloadOffset, frameObu.Value.PayloadLength);
var complete = Av1CompleteFrameHeaderParser.Parse(framePayload.Span, sh);

Console.WriteLine($"\nCompleteFrameHeader:");
Console.WriteLine($"  FrameType={complete.Prefix.FrameType} {complete.Prefix.FrameWidth}x{complete.Prefix.FrameHeight}");
Console.WriteLine($"  AllowIntraBc={complete.Prefix.AllowIntraBc}");
Console.WriteLine($"  TileInfo: {complete.TileInfo.TileCols}x{complete.TileInfo.TileRows}, " +
    $"uniform={complete.TileInfo.UniformSpacing}, tileSizeBytes={complete.TileInfo.TileSizeBytes}");
Console.WriteLine($"  Quant: baseQindex={complete.Quant.BaseQindex} " +
    $"yDc={complete.Quant.YDcDeltaQ} uDc={complete.Quant.UDcDeltaQ} uAc={complete.Quant.UAcDeltaQ}");
Console.WriteLine($"  Segmentation: enabled={complete.Segmentation.Enabled}");
Console.WriteLine($"  LoopFilter: lf0={complete.LoopFilter.FilterLevel0} lf1={complete.LoopFilter.FilterLevel1} " +
    $"sharp={complete.LoopFilter.SharpnessLevel}");
if (complete.Cdef is not null)
{
    Console.WriteLine($"  CDEF: damping={complete.Cdef.Damping} bits={complete.Cdef.Bits}");
}
if (complete.Lr is not null)
{
    Console.WriteLine($"  LR: per_plane=[{string.Join(",", complete.Lr.PerPlane)}]");
}
Console.WriteLine($"  TxMode={complete.TxMode} ReducedTxSet={complete.ReducedTxSetUsed}");
Console.WriteLine($"  CodedLossless={complete.CodedLossless} AllLossless={complete.AllLossless}");
Console.WriteLine($"  HeaderSizeBytes={complete.HeaderSizeBytes}");

// Extract tile group byte ranges
Av1TileGroup tg;
try
{
    tg = Av1TileGroupExtractor.Extract(framePayload.Span, complete);
    Console.WriteLine($"\nTileGroup: tiles {tg.StartTile}..{tg.EndTile}");
    foreach (var t in tg.Tiles)
    {
        Console.WriteLine($"  tile [{t.TileRow},{t.TileCol}] @ offset {t.Offset} length {t.Length}");
    }
}
catch (Exception ex)
{
    Console.WriteLine($"\nTileGroup extraction failed: {ex.Message}");
    Console.WriteLine("(This is expected for multi-tile-group bitstreams which are not yet supported.)");
    return 0;
}

// Try the keyframe walker - will throw on partition decode boundary
Console.WriteLine($"\n========== KeyframeWalker ==========");
var walker = new Av1KeyframeWalker();
try
{
    var fb = walker.DecodeFrame(framePayload, sh, complete, tg);
    Console.WriteLine($"Decoded frame: {fb.LumaWidth}x{fb.LumaHeight}, chroma {fb.ChromaWidth}x{fb.ChromaHeight}");
    PrintStats("Decoded", fb.Y, fb.U, fb.V);
}
catch (NotImplementedException ex)
{
    Console.WriteLine($"Walker hit known boundary: {ex.Message}");
    Console.WriteLine("Falling back to PLACEHOLDER mid-gray output for comparison harness.");
    int yLen = complete.Prefix.FrameWidth * complete.Prefix.FrameHeight;
    int cLen = ((complete.Prefix.FrameWidth + 1) / 2) * ((complete.Prefix.FrameHeight + 1) / 2);
    var pY = new byte[yLen]; Array.Fill(pY, (byte)128);
    var pU = new byte[cLen]; Array.Fill(pU, (byte)128);
    var pV = new byte[cLen]; Array.Fill(pV, (byte)128);
    PrintStats("PLACEHOLDER", pY, pU, pV);
}

// ffmpeg comparison
Console.WriteLine($"\n========== ffmpeg ground truth comparison ==========");
if (File.Exists(gtPath))
{
    var gt = File.ReadAllBytes(gtPath);
    int yLen = complete.Prefix.FrameWidth * complete.Prefix.FrameHeight;
    int cW = (complete.Prefix.FrameWidth + 1) / 2;
    int cH = (complete.Prefix.FrameHeight + 1) / 2;
    int cLen = cW * cH;
    if (gt.Length != yLen + 2 * cLen)
    {
        Console.WriteLine($"WARN: ground truth size {gt.Length} != expected {yLen + 2 * cLen}");
    }
    else
    {
        var gtY = gt.AsSpan(0, yLen).ToArray();
        var gtU = gt.AsSpan(yLen, cLen).ToArray();
        var gtV = gt.AsSpan(yLen + cLen, cLen).ToArray();
        PrintStats("ffmpeg ground truth", gtY, gtU, gtV);
    }
}
else
{
    Console.WriteLine($"(no ground truth file at {gtPath})");
}

Console.WriteLine();
Console.WriteLine("==========================================================");
Console.WriteLine("  Demo complete. Decoder pipeline runs from IVF parse all");
Console.WriteLine("  the way through complete frame header + tile group");
Console.WriteLine("  extraction + walker entry. Pixel decode requires the");
Console.WriteLine("  partition CDF tables to land before producing real");
Console.WriteLine("  pixel output.");
Console.WriteLine("==========================================================");
return 0;

static void PrintStats(string label, byte[] y, byte[] u, byte[] v)
{
    Console.WriteLine($"{label}:");
    Print("  Y", y);
    Print("  U", u);
    Print("  V", v);

    static void Print(string name, byte[] data)
    {
        long sum = 0;
        int min = 255, max = 0;
        for (int i = 0; i < data.Length; i++)
        {
            byte b = data[i];
            sum += b;
            if (b < min) min = b;
            if (b > max) max = b;
        }
        double mean = (double)sum / data.Length;
        Console.WriteLine($"  {name}: mean={mean:F2} min={min} max={max} len={data.Length}");
    }
}
