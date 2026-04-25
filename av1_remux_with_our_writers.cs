// SpawnDev.Codecs AV1 remux: writer-side end-to-end demo.
//
// Takes bbb_180_2s.ivf (libaom-encoded), parses every OBU, re-emits
// each through Av1ObuWriter, and writes the result using our IvfWriter.
// ffmpeg then decodes both files to YUV and we compare byte-identical.
//
// If the YUV outputs match: our OBU writer emits framing ffmpeg accepts,
// and our IVF writer emits a container ffmpeg accepts. End-to-end on
// the bytes-out side.
//
// Note: our SequenceHeaderWriter emits a minimal SH that lacks fields
// libaom's SH carries (which downstream frame parsing references), so
// substituting our SH is not a goal of this demo - we keep the source
// SH OBU verbatim and prove the framing + container.

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Av1;

string ffmpegPath = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
string testDataDir = "SpawnDev.Codecs.Demo.Shared/TestData";
string sourceIvf = Path.Combine(testDataDir, "bbb_180_2s.ivf");
string remuxIvf = Path.Combine(Path.GetTempPath(), "bbb_remuxed.ivf");
string sourceYuv = Path.Combine(Path.GetTempPath(), "bbb_source.yuv");
string remuxYuv = Path.Combine(Path.GetTempPath(), "bbb_remux.yuv");

Console.WriteLine();
Console.WriteLine("============================================================");
Console.WriteLine("  AV1 remux: writer-side end-to-end demo");
Console.WriteLine("============================================================");

// 1. Read the source IVF.
var sourceBytes = File.ReadAllBytes(sourceIvf);
var sourceHeader = IvfReader.ParseHeader(sourceBytes);
Console.WriteLine();
Console.WriteLine($"Source IVF: {sourceHeader.FourCc} {sourceHeader.Width}x{sourceHeader.Height} ({sourceHeader.NumFrames} frames declared)");

// 2. Walk every IVF frame; re-emit each OBU verbatim through Av1ObuWriter,
//    write the resulting frame through our IvfWriter. Source SH stays
//    intact so frames parse correctly.
int frameIdx = 0;
int totalObus = 0;
using (var outFs = new FileStream(remuxIvf, FileMode.Create, FileAccess.Write))
{
    var ivfOut = new IvfWriter(outFs, "AV01", sourceHeader.Width, sourceHeader.Height,
        frameRate: sourceHeader.FrameRate, timeScale: sourceHeader.TimeScale);

    foreach (var srcFrame in IvfReader.EnumerateFrames(sourceBytes))
    {
        using var frameMs = new MemoryStream();
        foreach (var obu in Av1ObuParser.EnumerateObus(srcFrame.Data))
        {
            var re = Av1ObuWriter.EmitObu(obu, srcFrame.Data);
            frameMs.Write(re, 0, re.Length);
            totalObus++;
        }
        ivfOut.WriteFrame(frameMs.ToArray(), srcFrame.Pts);
        frameIdx++;
    }
    ivfOut.Finish();
}
Console.WriteLine($"Remuxed: {frameIdx} frames written, {totalObus} OBUs re-emitted");
Console.WriteLine($"Output: {remuxIvf} ({new FileInfo(remuxIvf).Length} bytes; source = {sourceBytes.Length})");

// 4. ffmpeg decode source + remux to YUV.
RunFfmpeg(ffmpegPath, $"-y -i \"{sourceIvf}\" -f rawvideo -pix_fmt yuv420p \"{sourceYuv}\"");
RunFfmpeg(ffmpegPath, $"-y -i \"{remuxIvf}\"  -f rawvideo -pix_fmt yuv420p \"{remuxYuv}\"");
var srcYuv = File.ReadAllBytes(sourceYuv);
var rmxYuv = File.ReadAllBytes(remuxYuv);
Console.WriteLine($"Source YUV: {srcYuv.Length} bytes");
Console.WriteLine($"Remux  YUV: {rmxYuv.Length} bytes");

// 5. Compare.
if (srcYuv.Length != rmxYuv.Length)
{
    Console.WriteLine($"  Length mismatch (source vs remux).");
}
else
{
    int mismatches = 0;
    for (int i = 0; i < srcYuv.Length; i++)
        if (srcYuv[i] != rmxYuv[i]) mismatches++;
    Console.WriteLine($"  Byte mismatch: {mismatches}/{srcYuv.Length}");
    if (mismatches == 0)
    {
        Console.WriteLine();
        Console.WriteLine("============================================================");
        Console.WriteLine("  END-TO-END BIT-EXACT.");
        Console.WriteLine("  - Av1ObuWriter wrote framing ffmpeg + dav1d accept.");
        Console.WriteLine("  - IvfWriter wrote a container ffmpeg accepts.");
        Console.WriteLine("  - The decoded pixel result is identical to source.");
        Console.WriteLine("============================================================");
    }
}

File.Delete(remuxIvf);
File.Delete(sourceYuv);
File.Delete(remuxYuv);

static void RunFfmpeg(string path, string args)
{
    var psi = new ProcessStartInfo(path, args) { RedirectStandardError = true, UseShellExecute = false };
    var p = Process.Start(psi)!;
    p.WaitForExit();
    if (p.ExitCode != 0)
        throw new Exception($"ffmpeg failed (exit {p.ExitCode}):\n{p.StandardError.ReadToEnd()}");
}
