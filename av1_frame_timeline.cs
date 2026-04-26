// AV1 frame timeline demo: per-frame metadata report on a real AV1
// stream using the SpawnDev.Codecs Av1StreamAnalyzer high-level API.
// Prints type, show, allow_scc, force_int_mv, order_hint,
// refresh_frame_flags, and frame size for every coded frame in
// bbb_180_2s.ivf.

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.IO;
using System.Linq;
using SpawnDev.Codecs.Video.Av1;

string ivfPath = "SpawnDev.Codecs.Demo.Shared/TestData/bbb_180_2s.ivf";
var bytes = File.ReadAllBytes(ivfPath);
var summary = Av1StreamAnalyzer.Analyze(bytes);

Console.WriteLine();
Console.WriteLine("============================================================");
Console.WriteLine("  AV1 frame timeline (per-frame metadata)");
Console.WriteLine("============================================================");
Console.WriteLine();
Console.WriteLine($"Source: {ivfPath}");
Console.WriteLine($"IVF: {summary.IvfHeader.FourCc} {summary.IvfHeader.Width}x{summary.IvfHeader.Height} ({summary.IvfHeader.NumFrames} frames declared)");
Console.WriteLine($"SH: profile={summary.SequenceHeader?.SeqProfile}, "
    + $"bit_depth={summary.SequenceHeader?.BitDepth}, "
    + $"order_hint={summary.SequenceHeader?.EnableOrderHint}, "
    + $"cdef={summary.SequenceHeader?.EnableCdef}");
Console.WriteLine();
Console.WriteLine($"OBU counts: " + string.Join(", ",
    summary.ObuCounts.OrderByDescending(kv => kv.Value).Select(kv => $"{kv.Key}={kv.Value}")));
Console.WriteLine($"Frame types: " + string.Join(", ",
    summary.FrameTypeCounts.OrderByDescending(kv => kv.Value).Select(kv => $"{kv.Key}={kv.Value}"))
    + $", ShowExist={summary.ShowExistingFrames.Count}");
Console.WriteLine();

Console.WriteLine($"{"TU",-3} {"#",-3} {"Type",-15} {"Show",-5} {"AllowSCC",-9} {"FIntMV",-7} {"OrderHint",-9} {"Refresh",-9} {"Size",-9}");
Console.WriteLine($"{new string('-', 80)}");

// Merge coded + show-existing frames in TU order for the timeline.
var allFrames = summary.CodedFrames.Concat(summary.ShowExistingFrames)
    .OrderBy(f => f.TemporalUnit).ThenBy(f => f.IndexInTu);

foreach (var f in allFrames)
{
    var fh = f.Header;
    string typeLabel = fh.ShowExistingFrame
        ? $"ShowExist[{fh.FrameToShowMapIdx}]"
        : fh.FrameType.ToString();
    string size = fh.ShowExistingFrame ? "(replay)" : $"{fh.FrameWidth}x{fh.FrameHeight}";
    string refresh = fh.ShowExistingFrame ? "(replay)" : $"0x{fh.RefreshFrameFlags:X2}";
    Console.WriteLine(
        $"{f.TemporalUnit,-3} {f.IndexInTu,-3} {typeLabel,-15} {fh.ShowFrame,-5} {fh.AllowScreenContentTools,-9} {fh.ForceIntegerMv,-7} {fh.OrderHint,-9} {refresh,-9} {size}");
}

Console.WriteLine();
Console.WriteLine($"Total temporal units: {summary.TotalTemporalUnits}");
Console.WriteLine($"Coded frames:         {summary.CodedFrames.Count}");
Console.WriteLine($"ShowExist frames:     {summary.ShowExistingFrames.Count}");
