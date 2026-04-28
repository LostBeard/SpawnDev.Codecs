// AV1 PURE WRITER remux: every byte of the output IVF except the
// entropy-coded frame OBU payloads is emitted by SpawnDev.Codecs writers.
//
// Pipeline:
//   1. Read source bbb_180_2s.ivf.
//   2. Parse the source SH into Av1SequenceHeader.
//   3. Convert to Av1SequenceHeaderConfig via FromHeader.
//   4. Re-emit SH bytes via Av1SequenceHeaderWriter (must equal source SH).
//   5. For each frame: re-emit every OBU through Av1ObuWriter (or
//      substitute our SH-from-config bytes for the original SH OBU).
//   6. Write the result via IvfWriter to a new .ivf file.
//   7. ffmpeg + dav1d decode source vs remux to YUV - byte-by-byte
//      comparison must be 0/N mismatches.
//
// This is the strongest "pure encoder framing" demonstration:
// SequenceHeader bytes from our config-driven writer (not byte-copied
// from source), OBU framing from our writer, container from our writer.
// Only the entropy-coded inner bodies of frames are pulled from source -
// because those require the Daala range coder we haven't built yet.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Av1;

string ffmpegPath = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
string testDataDir = "SpawnDev.Codecs.Demo.Shared/TestData";
string sourceIvf = Path.Combine(testDataDir, "bbb_180_2s.ivf");
string remuxIvf = Path.Combine(Path.GetTempPath(), "bbb_pure_writer_remux.ivf");
string sourceYuv = Path.Combine(Path.GetTempPath(), "bbb_pure_src.yuv");
string remuxYuv = Path.Combine(Path.GetTempPath(), "bbb_pure_rmx.yuv");

Console.WriteLine();
Console.WriteLine("============================================================");
Console.WriteLine("  AV1 PURE WRITER remux: emit framing from config, not bytes");
Console.WriteLine("============================================================");

var sourceBytes = File.ReadAllBytes(sourceIvf);
var sourceHeader = IvfReader.ParseHeader(sourceBytes);
Console.WriteLine();
Console.WriteLine($"Source IVF: {sourceHeader.FourCc} {sourceHeader.Width}x{sourceHeader.Height} ({sourceHeader.NumFrames} frames declared)");

// Step 2 + 3: parse source SH and convert to config.
var firstFrame = IvfReader.EnumerateFrames(sourceBytes).First();
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
var ourCfg = Av1SequenceHeaderConfig.FromHeader(parsedSh);
Console.WriteLine($"Parsed source SH: profile={parsedSh.SeqProfile}, {parsedSh.MaxFrameWidth}x{parsedSh.MaxFrameHeight}, "
    + $"order_hint={parsedSh.EnableOrderHint}, cdef={parsedSh.EnableCdef}, "
    + $"matrix={parsedSh.MatrixCoefficients}");

// Step 4: re-emit SH bytes from our config.
byte[] ourSh = Av1SequenceHeaderWriter.EmitPayload(ourCfg);
int shMatch = 0;
int shLen = Math.Min(sourceSh.Length, ourSh.Length);
for (int i = 0; i < shLen; i++) if (sourceSh[i] == ourSh[i]) shMatch++;
Console.WriteLine($"Our SH from-config bytes: {shMatch}/{sourceSh.Length} BIT-EXACT vs source");

byte[] ourShObu = Av1ObuWriter.EmitObu(Av1ObuType.SequenceHeader, ourSh, hasSizeField: true);

// Step 5 + 6: Walk every IVF frame, substitute writer-emitted SH for
// the source SH, re-emit every other OBU through our writer, write the
// resulting frames into a new IVF via our IvfWriter.
int frameIdx = 0;
int totalObus = 0;
int substitutedSh = 0;
using (var outFs = new FileStream(remuxIvf, FileMode.Create, FileAccess.Write))
{
    var ivfOut = new IvfWriter(outFs, "AV01", sourceHeader.Width, sourceHeader.Height,
        frameRate: sourceHeader.FrameRate, timeScale: sourceHeader.TimeScale);

    foreach (var srcFrame in IvfReader.EnumerateFrames(sourceBytes))
    {
        using var ms = new MemoryStream();
        foreach (var obu in Av1ObuParser.EnumerateObus(srcFrame.Data))
        {
            if (obu.Type == Av1ObuType.SequenceHeader)
            {
                ms.Write(ourShObu, 0, ourShObu.Length);
                substitutedSh++;
            }
            else
            {
                var re = Av1ObuWriter.EmitObu(obu, srcFrame.Data);
                ms.Write(re, 0, re.Length);
            }
            totalObus++;
        }
        ivfOut.WriteFrame(ms.ToArray(), srcFrame.Pts);
        frameIdx++;
    }
    ivfOut.Finish();
}
Console.WriteLine($"Remuxed {frameIdx} frames / {totalObus} OBUs (substituted {substitutedSh} writer-emitted SH OBU(s))");
Console.WriteLine($"Output: {remuxIvf} ({new FileInfo(remuxIvf).Length} bytes; source = {sourceBytes.Length})");

// Step 7: ffmpeg cross-validates.
RunFfmpeg(ffmpegPath, $"-y -i \"{sourceIvf}\" -f rawvideo -pix_fmt yuv420p \"{sourceYuv}\"");
RunFfmpeg(ffmpegPath, $"-y -i \"{remuxIvf}\"  -f rawvideo -pix_fmt yuv420p \"{remuxYuv}\"");
var srcYuv = File.ReadAllBytes(sourceYuv);
var rmxYuv = File.ReadAllBytes(remuxYuv);

if (srcYuv.Length != rmxYuv.Length)
{
    Console.WriteLine($"  Length mismatch: source {srcYuv.Length}, remux {rmxYuv.Length}");
}
else
{
    int mismatches = 0;
    for (int i = 0; i < srcYuv.Length; i++) if (srcYuv[i] != rmxYuv[i]) mismatches++;
    Console.WriteLine();
    Console.WriteLine($"  Source vs remux YUV: {srcYuv.Length - mismatches}/{srcYuv.Length} BIT-EXACT");
    if (mismatches == 0)
    {
        Console.WriteLine();
        Console.WriteLine("============================================================");
        Console.WriteLine("  PURE WRITER PIPELINE BIT-EXACT.");
        Console.WriteLine("  - SH bytes emitted from config, not byte-copied from source");
        Console.WriteLine("  - OBU framing emitted by Av1ObuWriter");
        Console.WriteLine("  - Container emitted by IvfWriter");
        Console.WriteLine("  - ffmpeg + dav1d decode our output PIXEL-IDENTICAL to source");
        Console.WriteLine("============================================================");
    }
}

File.Delete(remuxIvf); File.Delete(sourceYuv); File.Delete(remuxYuv);

static void RunFfmpeg(string path, string args)
{
    var psi = new ProcessStartInfo(path, args) { RedirectStandardError = true, UseShellExecute = false };
    var p = Process.Start(psi)!;
    p.WaitForExit();
    if (p.ExitCode != 0)
        throw new Exception($"ffmpeg failed (exit {p.ExitCode}):\n{p.StandardError.ReadToEnd()}");
}
