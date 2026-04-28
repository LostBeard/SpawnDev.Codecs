// SpawnDev.Codecs VP9 round-trip remux through ffmpeg.
//
// 1. Read BBB.webm via MatroskaContainer
// 2. Extract VP9 packets
// 3. For each packet, re-emit through Vp9SuperframeWriter (BBB has no
//    superframes so this is the verbatim path)
// 4. Pack into a new .ivf file via our IvfWriter (fourcc VP90)
// 5. ffmpeg decodes both source and our remux to YUV
// 6. Compare byte-by-byte
//
// If they match, the VP9 packet path round-trips through SpawnDev.Codecs
// without losing any bytes - same encoded body, same container output.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.Codecs.Video.Vp9;

string ffmpegPath = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
string testDataDir = "SpawnDev.Codecs.Demo.Shared/TestData";
string sourceWebm = Path.Combine(testDataDir, "Big_Buck_Bunny_180_10s.webm");
string remuxIvf = Path.Combine(Path.GetTempPath(), "vp9_remux.ivf");
string sourceYuv = Path.Combine(Path.GetTempPath(), "vp9_remux_src.yuv");
string remuxYuv = Path.Combine(Path.GetTempPath(), "vp9_remux_out.yuv");

Console.WriteLine();
Console.WriteLine("============================================================");
Console.WriteLine("  VP9 round-trip remux: WebM -> our writers -> IVF -> ffmpeg");
Console.WriteLine("============================================================");

using var stream = File.OpenRead(sourceWebm);
var container = new MatroskaContainer(stream);
var video = container.Tracks.First(t => t.IsVideo);
Console.WriteLine();
Console.WriteLine($"Source: {sourceWebm}");
Console.WriteLine($"Container: {container.DocType} (WebM); Codec: {video.CodecId}");

// Step 1-4: extract packets, re-emit through Vp9SuperframeWriter, pack into IVF.
int packetCount = 0;
int sliceCount = 0;
using (var outFs = new FileStream(remuxIvf, FileMode.Create, FileAccess.Write))
{
    var ivfOut = new IvfWriter(outFs, "VP90", 320, 180, frameRate: 30, timeScale: 1);
    long pts = 0;
    foreach (var pkt in container.Frames.Where(f => f.TrackNumber == video.TrackNumber))
    {
        packetCount++;
        var data = pkt.Data.ToArray();
        var parsed = Vp9SuperframeParser.Parse(data);
        var frames = new byte[parsed.Frames.Count][];
        for (int i = 0; i < parsed.Frames.Count; i++)
        {
            var slice = parsed.Frames[i];
            var fbytes = new byte[slice.Length];
            Buffer.BlockCopy(data, slice.Offset, fbytes, 0, slice.Length);
            frames[i] = fbytes;
            sliceCount++;
        }
        var reEmitted = Vp9SuperframeWriter.Emit(frames);
        ivfOut.WriteFrame(reEmitted, pts++);
    }
    ivfOut.Finish();
}
Console.WriteLine($"Remuxed {packetCount} packets ({sliceCount} slices) through Vp9SuperframeWriter + IvfWriter");
Console.WriteLine($"Output: {remuxIvf} ({new FileInfo(remuxIvf).Length} bytes)");

// Step 5: ffmpeg decodes both source and our remux.
RunFfmpeg(ffmpegPath, $"-y -i \"{sourceWebm}\" -f rawvideo -pix_fmt yuv420p \"{sourceYuv}\"");
RunFfmpeg(ffmpegPath, $"-y -i \"{remuxIvf}\" -f rawvideo -pix_fmt yuv420p \"{remuxYuv}\"");

var srcYuv = File.ReadAllBytes(sourceYuv);
var rmxYuv = File.ReadAllBytes(remuxYuv);
Console.WriteLine($"Source YUV: {srcYuv.Length} bytes");
Console.WriteLine($"Remux  YUV: {rmxYuv.Length} bytes");

// Step 6: compare.
if (srcYuv.Length != rmxYuv.Length)
{
    Console.WriteLine($"  Length mismatch.");
}
else
{
    int mismatches = 0;
    for (int i = 0; i < srcYuv.Length; i++)
        if (srcYuv[i] != rmxYuv[i]) mismatches++;
    Console.WriteLine($"  Source vs remux YUV: {srcYuv.Length - mismatches}/{srcYuv.Length} BIT-EXACT");
    if (mismatches == 0)
    {
        Console.WriteLine();
        Console.WriteLine("============================================================");
        Console.WriteLine("  VP9 ROUND-TRIP BIT-EXACT.");
        Console.WriteLine($"  - {packetCount} VP9 packets re-emitted via Vp9SuperframeWriter");
        Console.WriteLine($"  - Repackaged as .ivf (VP90) via our IvfWriter");
        Console.WriteLine($"  - ffmpeg decodes both source and remux pixel-identical");
        Console.WriteLine($"  - {srcYuv.Length} byte YUV output matches byte-for-byte");
        Console.WriteLine("============================================================");
    }
}

File.Delete(remuxIvf); File.Delete(sourceYuv); File.Delete(remuxYuv);

static void RunFfmpeg(string path, string args)
{
    // Suppress ffmpeg's verbose stderr by adding -loglevel quiet up
    // front - long stderr can deadlock when not drained asynchronously.
    var psi = new ProcessStartInfo(path, "-loglevel error " + args)
    {
        RedirectStandardError = true,
        UseShellExecute = false,
    };
    var p = Process.Start(psi)!;
    string stderr = p.StandardError.ReadToEnd(); // drains synchronously - OK with -loglevel error
    p.WaitForExit();
    if (p.ExitCode != 0)
        throw new Exception($"ffmpeg failed (exit {p.ExitCode}):\n{stderr}");
}
