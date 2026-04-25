// AV1 single-keyframe demo: construct a one-frame .ivf using only our
// writers + the first BBB frame's OBU body. ffmpeg decodes the result
// to one YUV frame which is bit-identical to ffmpeg decoding the same
// frame out of the original stream.
//
// This is the "smallest writer-driven AV1 stream that produces a real
// pixel frame" demonstration. Strongest possible "encoder framing
// works" signal for AV1 keyframes.

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
string singleIvf = Path.Combine(Path.GetTempPath(), "spawndev_av1_single_kf.ivf");
string sourceYuv = Path.Combine(Path.GetTempPath(), "spawndev_av1_single_src.yuv");
string singleYuv = Path.Combine(Path.GetTempPath(), "spawndev_av1_single_out.yuv");

Console.WriteLine();
Console.WriteLine("============================================================");
Console.WriteLine("  AV1 single-keyframe demo: writer + 1 source frame OBU body");
Console.WriteLine("============================================================");

var sourceBytes = File.ReadAllBytes(sourceIvf);
var sourceHeader = IvfReader.ParseHeader(sourceBytes);
var firstFrame = IvfReader.EnumerateFrames(sourceBytes).First();

// Pull source OBUs but use writer-emitted SH from config.
byte[] sourceSh = Array.Empty<byte>();
foreach (var obu in Av1ObuParser.EnumerateObus(firstFrame.Data))
{
    if (obu.Type == Av1ObuType.SequenceHeader)
    {
        sourceSh = firstFrame.Data.Slice(obu.PayloadOffset, obu.PayloadLength).ToArray();
        break;
    }
}
var parsedSh = Av1SequenceHeaderParser.Parse(sourceSh);
var ourSh = Av1SequenceHeaderWriter.EmitPayload(Av1SequenceHeaderConfig.FromHeader(parsedSh));
var ourShObu = Av1ObuWriter.EmitObu(Av1ObuType.SequenceHeader, ourSh, hasSizeField: true);

// Build the single-frame stream's OBU list:
//   - Source TD OBU (re-emitted via Av1ObuWriter)
//   - Our SH OBU (writer-emitted, byte-identical to source)
//   - Source Frame OBU (re-emitted via Av1ObuWriter; entropy body comes
//     from libaom-encoded source since we don't yet have the daala
//     range coder to write our own)
using (var fs = new FileStream(singleIvf, FileMode.Create, FileAccess.Write))
{
    var ivf = new IvfWriter(fs, "AV01", sourceHeader.Width, sourceHeader.Height,
        frameRate: sourceHeader.FrameRate, timeScale: sourceHeader.TimeScale);
    using var ms = new MemoryStream();
    int substituted = 0, copied = 0;
    foreach (var obu in Av1ObuParser.EnumerateObus(firstFrame.Data))
    {
        if (obu.Type == Av1ObuType.SequenceHeader)
        {
            ms.Write(ourShObu, 0, ourShObu.Length);
            substituted++;
        }
        else
        {
            byte[] re = Av1ObuWriter.EmitObu(obu, firstFrame.Data);
            ms.Write(re, 0, re.Length);
            copied++;
        }
    }
    ivf.WriteFrame(ms.ToArray(), 0);
    ivf.Finish();
    Console.WriteLine();
    Console.WriteLine($"Single-frame stream: {substituted} writer-emitted SH OBU(s) + {copied} re-framed OBU(s)");
}
Console.WriteLine($"Output: {singleIvf} ({new FileInfo(singleIvf).Length} bytes)");

// ffmpeg decodes both the source (only frame 0) and the single-frame stream.
RunFfmpeg(ffmpegPath, $"-y -i \"{sourceIvf}\" -vframes 1 -f rawvideo -pix_fmt yuv420p \"{sourceYuv}\"");
RunFfmpeg(ffmpegPath, $"-y -i \"{singleIvf}\" -f rawvideo -pix_fmt yuv420p \"{singleYuv}\"");

var src = File.ReadAllBytes(sourceYuv);
var ours = File.ReadAllBytes(singleYuv);
Console.WriteLine($"Source YUV (frame 0): {src.Length} bytes");
Console.WriteLine($"Our    YUV (frame 0): {ours.Length} bytes");

if (src.Length != ours.Length)
{
    Console.WriteLine($"  Length mismatch.");
}
else
{
    int mismatches = 0;
    for (int i = 0; i < src.Length; i++) if (src[i] != ours[i]) mismatches++;
    Console.WriteLine($"  Source vs writer-built: {src.Length - mismatches}/{src.Length} BIT-EXACT");
    if (mismatches == 0)
    {
        Console.WriteLine();
        Console.WriteLine("============================================================");
        Console.WriteLine("  SINGLE-KEYFRAME WRITER STREAM BIT-EXACT.");
        Console.WriteLine($"  - {new FileInfo(singleIvf).Length}-byte writer-built .ivf");
        Console.WriteLine("  - ffmpeg+dav1d decode it to 1 frame of pixel-identical YUV");
        Console.WriteLine("  - SH bytes from our writer config; OBU framing from our writer;");
        Console.WriteLine("    container from our writer; only the entropy-coded body comes");
        Console.WriteLine("    from libaom (the gate to a complete pure-.NET AV1 encoder)");
        Console.WriteLine("============================================================");
    }
}

File.Delete(singleIvf); File.Delete(sourceYuv); File.Delete(singleYuv);

static void RunFfmpeg(string path, string args)
{
    var psi = new ProcessStartInfo(path, args) { RedirectStandardError = true, UseShellExecute = false };
    var p = Process.Start(psi)!;
    p.WaitForExit();
    if (p.ExitCode != 0)
        throw new Exception($"ffmpeg failed (exit {p.ExitCode}):\n{p.StandardError.ReadToEnd()}");
}
