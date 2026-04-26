// AV1 frame timeline demo: per-frame metadata report on a real AV1
// stream using the SpawnDev.Codecs FrameHeader parser. Prints type,
// show, allow_scc, force_int_mv, order_hint, refresh_frame_flags, and
// frame size for every coded frame in bbb_180_2s.ivf.
//
// Useful for codec developers debugging GOP structure or verifying
// that the parser sees the same per-frame state as a libaom-encoded
// stream's actual structure.

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.IO;
using System.Linq;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Av1;

string ivfPath = "SpawnDev.Codecs.Demo.Shared/TestData/bbb_180_2s.ivf";
var bytes = File.ReadAllBytes(ivfPath);
var ivfHeader = IvfReader.ParseHeader(bytes);

Console.WriteLine();
Console.WriteLine("============================================================");
Console.WriteLine("  AV1 frame timeline (per-frame metadata)");
Console.WriteLine("============================================================");
Console.WriteLine();
Console.WriteLine($"Source: {ivfPath}");
Console.WriteLine($"IVF: {ivfHeader.FourCc} {ivfHeader.Width}x{ivfHeader.Height} ({ivfHeader.NumFrames} frames declared)");
Console.WriteLine();

Av1SequenceHeader? sh = null;
int tu = 0;
int totalFrames = 0;

Console.WriteLine($"{"TU",-3} {"#",-3} {"Type",-15} {"Show",-5} {"AllowSCC",-9} {"FIntMV",-7} {"OrderHint",-9} {"Refresh",-9} {"Size",-9}");
Console.WriteLine($"{new string('-', 80)}");

foreach (var ivfFrame in IvfReader.EnumerateFrames(bytes))
{
    tu++;
    int fhInTu = 0;
    foreach (var obu in Av1ObuParser.EnumerateObus(ivfFrame.Data))
    {
        if (obu.Type == Av1ObuType.SequenceHeader)
        {
            sh = Av1SequenceHeaderParser.Parse(
                ivfFrame.Data.Span.Slice(obu.PayloadOffset, obu.PayloadLength));
        }
        else if (obu.IsCodedFrameData && sh is not null)
        {
            var fh = Av1FrameHeaderParser.Parse(
                ivfFrame.Data.Span.Slice(obu.PayloadOffset, obu.PayloadLength), sh);
            fhInTu++;
            totalFrames++;
            string typeLabel = fh.ShowExistingFrame
                ? $"ShowExist[{fh.FrameToShowMapIdx}]"
                : fh.FrameType.ToString();
            string size = fh.ShowExistingFrame ? "(replay)" : $"{fh.FrameWidth}x{fh.FrameHeight}";
            string refresh = fh.ShowExistingFrame ? "(replay)" : $"0x{fh.RefreshFrameFlags:X2}";
            Console.WriteLine(
                $"{tu,-3} {fhInTu,-3} {typeLabel,-15} {fh.ShowFrame,-5} {fh.AllowScreenContentTools,-9} {fh.ForceIntegerMv,-7} {fh.OrderHint,-9} {refresh,-9} {size}");
        }
    }
}

Console.WriteLine();
Console.WriteLine($"Total temporal units: {tu}");
Console.WriteLine($"Total frame headers parsed: {totalFrames}");
