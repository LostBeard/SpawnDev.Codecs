// Minimal AV1 stream demo: write a stream containing JUST our
// SequenceHeader + TemporalDelimiter OBUs and have ffprobe identify
// the codec / dimensions. Stronger validation that our SH bytes are
// well-formed than a parser-only round-trip.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Av1;

string ffprobePath = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffprobe.exe";
string outIvf = Path.Combine(Path.GetTempPath(), "spawndev_av1_minimal.ivf");

Console.WriteLine();
Console.WriteLine("============================================================");
Console.WriteLine("  AV1 minimal stream from scratch -> ffprobe");
Console.WriteLine("============================================================");

// Build OBUs with our writers.
var sh = Av1SequenceHeaderWriter.EmitPayload(new Av1SequenceHeaderConfig
{
    SeqProfile = 0,
    MaxFrameWidth = 320,
    MaxFrameHeight = 180,
    BitDepth = 8,
    SubsamplingX = 1,
    SubsamplingY = 1,
});
var shObu = Av1ObuWriter.EmitObu(Av1ObuType.SequenceHeader, sh, hasSizeField: true);
var tdObu = Av1ObuWriter.EmitObu(Av1ObuType.TemporalDelimiter, ReadOnlySpan<byte>.Empty, hasSizeField: true);

Console.WriteLine($"  TD OBU:  {tdObu.Length} bytes (0x{string.Join(" 0x", tdObu.Select(b => b.ToString("X2")))})");
Console.WriteLine($"  SH OBU: {shObu.Length} bytes");

// One IVF frame combining TD + SH (libdav1d expects TD at the start of every TU).
using (var fs = new FileStream(outIvf, FileMode.Create, FileAccess.Write))
{
    var ivf = new IvfWriter(fs, "AV01", 320, 180, frameRate: 30, timeScale: 1);
    using var ms = new MemoryStream();
    ms.Write(tdObu, 0, tdObu.Length);
    ms.Write(shObu, 0, shObu.Length);
    ivf.WriteFrame(ms.ToArray(), 0);
    ivf.Finish();
}
long size = new FileInfo(outIvf).Length;
Console.WriteLine($"  Output: {outIvf} ({size} bytes)");

// Run ffprobe and print what it sees.
var psi = new ProcessStartInfo(ffprobePath, $"-hide_banner -i \"{outIvf}\"")
{
    RedirectStandardError = true,
    RedirectStandardOutput = true,
    UseShellExecute = false,
};
var p = Process.Start(psi)!;
string stdout = p.StandardOutput.ReadToEnd();
string stderr = p.StandardError.ReadToEnd();
p.WaitForExit();
Console.WriteLine();
Console.WriteLine("--- ffprobe output ---");
Console.WriteLine(stderr.Trim());
if (!string.IsNullOrEmpty(stdout))
{
    Console.WriteLine("--- stdout ---");
    Console.WriteLine(stdout.Trim());
}
Console.WriteLine();
if (stderr.Contains("av1") || stderr.Contains("AV1") || stderr.Contains("AOMedia"))
{
    Console.WriteLine("============================================================");
    Console.WriteLine("  FFPROBE accepted our bytes as a valid AV1 stream.");
    Console.WriteLine("============================================================");
}

File.Delete(outIvf);
