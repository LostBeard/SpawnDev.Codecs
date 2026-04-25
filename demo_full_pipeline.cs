// SpawnDev.Codecs end-to-end demo: drives every working codec
// pipeline against real-world data and ffmpeg reference decoders.

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.Codecs.Video;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.Codecs.Video.Vp9;

string ffmpegPath = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
string testDataDir = "SpawnDev.Codecs.Demo.Shared/TestData";

Console.WriteLine();
Console.WriteLine("============================================================");
Console.WriteLine("  SpawnDev.Codecs end-to-end pipeline demo");
Console.WriteLine("============================================================");
Console.WriteLine();

// 1. FLAC encoder + decoder + ffmpeg round-trip.
Console.WriteLine("--- FLAC: encoder + decoder, ffmpeg cross-validated ---");
{
    int total = 44100;
    var input = new int[total];
    double a = 0.5 * 32767;
    for (int n = 0; n < total; n++)
        input[n] = (int)(Math.Sin(2.0 * Math.PI * 440 * n / 44100) * a);

    byte[] encoded = FlacEncoder.EncodeStream(input, 44100, 1, 16);
    var decoded = FlacDecoder.Decode(encoded);

    int matches = 0;
    for (int i = 0; i < input.Length; i++)
        if (decoded.InterleavedSamples[i] == input[i]) matches++;
    Console.WriteLine($"  Generated {input.Length} samples (1s @ 440Hz mono 16-bit)");
    Console.WriteLine($"  FlacEncoder produced {encoded.Length} bytes");
    Console.WriteLine($"  FlacDecoder round-trip: {matches}/{input.Length} BIT-EXACT");

    string flacFile = Path.Combine(Path.GetTempPath(), "demo_flac.flac");
    string pcmFile = Path.Combine(Path.GetTempPath(), "demo_flac.pcm");
    File.WriteAllBytes(flacFile, encoded);
    RunFfmpeg(ffmpegPath, $"-y -i \"{flacFile}\" -f s16le \"{pcmFile}\"");
    var ffPcm = File.ReadAllBytes(pcmFile);
    int ffMatches = 0;
    for (int i = 0; i < input.Length; i++)
    {
        short ffSample = (short)(ffPcm[i * 2] | (ffPcm[i * 2 + 1] << 8));
        if (ffSample == (short)input[i]) ffMatches++;
    }
    Console.WriteLine($"  ffmpeg decoded our FLAC: {ffMatches}/{input.Length} BIT-EXACT");
    File.Delete(flacFile); File.Delete(pcmFile);
}

// 2. VP9 decoder pipeline on BBB.webm.
Console.WriteLine();
Console.WriteLine("--- VP9: decoder pipeline on Big Buck Bunny WebM ---");
{
    var bbbBytes = File.ReadAllBytes(Path.Combine(testDataDir, "Big_Buck_Bunny_180_10s.webm"));
    using var bbbStream = new MemoryStream(bbbBytes);
    var container = new MatroskaContainer(bbbStream);
    var video = container.Tracks.First(t => t.IsVideo);

    var vp9 = new Vp9Decoder();
    var sink = new CountingFrameSink();
    int packets = 0;
    foreach (var f in container.Frames.Where(f => f.TrackNumber == video.TrackNumber))
    {
        await vp9.DecodeFrameAsync(f.Data, sink);
        packets++;
    }
    Console.WriteLine($"  Container: {container.DocType} (WebM)");
    Console.WriteLine($"  Codec: {video.CodecId}");
    Console.WriteLine($"  Vp9Decoder learned: {vp9.Width}x{vp9.Height}, {vp9.Subsampling}, {vp9.BitDepth}");
    Console.WriteLine($"  Packets decoded: {packets}");
    Console.WriteLine($"  Visible frames emitted: {sink.FrameCount}");
    Console.WriteLine($"  Last frame header: {vp9.LastFrameHeader?.FrameType}, {vp9.LastFrameHeader?.FrameWidth}x{vp9.LastFrameHeader?.FrameHeight}");
    Console.WriteLine($"  Last compressed header: tx_mode={vp9.LastCompressedResult?.TxMode}, ref_mode={vp9.LastCompressedResult?.ReferenceMode}");
    Console.WriteLine($"  Last tile group: {vp9.LastTileGroup?.Tiles.Count} tile(s)");
}

// 3. AV1 decoder pipeline on BBB-encoded AV1 IVF.
Console.WriteLine();
Console.WriteLine("--- AV1: decoder pipeline on bbb_180_2s.ivf ---");
{
    var av1Bytes = File.ReadAllBytes(Path.Combine(testDataDir, "bbb_180_2s.ivf"));
    var ivfHeader = IvfReader.ParseHeader(av1Bytes);
    Console.WriteLine($"  IVF: {ivfHeader.FourCc} {ivfHeader.Width}x{ivfHeader.Height} ({ivfHeader.NumFrames} frames declared)");

    var av1 = new Av1Decoder();
    var sink = new CountingFrameSink();
    int frameCount = 0;
    foreach (var ivfFrame in IvfReader.EnumerateFrames(av1Bytes))
    {
        await av1.DecodeFrameAsync(ivfFrame.Data, sink);
        frameCount++;
    }
    Console.WriteLine($"  IVF frames fed: {frameCount}");
    Console.WriteLine($"  Av1Decoder learned: {av1.Width}x{av1.Height}");
    Console.WriteLine($"  SequenceHeader: profile={av1.LastSequenceHeader?.SeqProfile}, " +
                      $"bit_depth={av1.LastSequenceHeader?.BitDepth}, " +
                      $"subsampling=({av1.LastSequenceHeader?.SubsamplingX},{av1.LastSequenceHeader?.SubsamplingY})");
    Console.WriteLine($"  Visible frames emitted: {sink.FrameCount}");

    // Per-frame metadata across all 60 frames.
    Av1SequenceHeader? sh = null;
    int kfCount = 0, interCount = 0, intraOnlyCount = 0;
    foreach (var ivfFrame in IvfReader.EnumerateFrames(av1Bytes))
    {
        foreach (var obu in Av1ObuParser.EnumerateObus(ivfFrame.Data))
        {
            if (obu.Type == Av1ObuType.SequenceHeader)
                sh = Av1SequenceHeaderParser.Parse(ivfFrame.Data.Span.Slice(obu.PayloadOffset, obu.PayloadLength));
            else if (obu.IsCodedFrameData && sh is not null)
            {
                var fh = Av1FrameHeaderParser.Parse(ivfFrame.Data.Span.Slice(obu.PayloadOffset, obu.PayloadLength), sh);
                if (fh.FrameType == Av1FrameType.KeyFrame) kfCount++;
                else if (fh.FrameType == Av1FrameType.InterFrame) interCount++;
                else if (fh.FrameType == Av1FrameType.IntraOnlyFrame) intraOnlyCount++;
                break;
            }
        }
    }
    Console.WriteLine($"  Frame type breakdown: {kfCount} key + {interCount} inter + {intraOnlyCount} intra-only");
}

// 4. AV1 OBU type distribution across BBB.
Console.WriteLine();
Console.WriteLine("--- AV1: OBU type distribution across all 60 frames ---");
{
    var av1Bytes = File.ReadAllBytes(Path.Combine(testDataDir, "bbb_180_2s.ivf"));
    var typeCounts = new System.Collections.Generic.Dictionary<Av1ObuType, int>();
    foreach (var ivfFrame in IvfReader.EnumerateFrames(av1Bytes))
    {
        foreach (var obu in Av1ObuParser.EnumerateObus(ivfFrame.Data))
        {
            typeCounts.TryGetValue(obu.Type, out int c);
            typeCounts[obu.Type] = c + 1;
        }
    }
    foreach (var kv in typeCounts.OrderByDescending(kv => kv.Value))
    {
        Console.WriteLine($"  {kv.Key,-22} {kv.Value} OBUs");
    }
}

Console.WriteLine();
Console.WriteLine("============================================================");
Console.WriteLine("  Working today, end-to-end:");
Console.WriteLine("    - FLAC encoder + decoder (ffmpeg cross-verified BIT-EXACT)");
Console.WriteLine("    - VP9 decoder pipeline through tile group extraction");
Console.WriteLine("    - AV1 decoder pipeline + SH + FrameHeader parsers");
Console.WriteLine("    - IVF / Matroska / Ogg container readers");
Console.WriteLine("  In progress (placeholder pixels for video):");
Console.WriteLine("    - VP9 / AV1 block decode walker");
Console.WriteLine("    - Opus CELT mode (SILK works)");
Console.WriteLine("============================================================");

static void RunFfmpeg(string path, string args)
{
    var psi = new ProcessStartInfo(path, args) { RedirectStandardError = true, UseShellExecute = false };
    var p = Process.Start(psi)!;
    p.WaitForExit();
    if (p.ExitCode != 0)
        throw new Exception($"ffmpeg failed (exit {p.ExitCode}):\n{p.StandardError.ReadToEnd()}");
}

internal sealed class CountingFrameSink : IVideoFrameSink
{
    public int FrameCount;
    public ValueTask OnFrameAsync(
        ReadOnlyMemory<byte> y, int ys, ReadOnlyMemory<byte> u, int us,
        ReadOnlyMemory<byte> v, int vs, long pts)
    {
        FrameCount++;
        return ValueTask.CompletedTask;
    }
}
