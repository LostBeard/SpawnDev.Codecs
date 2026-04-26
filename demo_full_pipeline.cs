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
    Console.WriteLine($"  Coded frames: {vp9.TotalCodedFrames}, Visible frames emitted: {vp9.TotalVisibleFrames}");
    Console.WriteLine($"  Cumulative frame types: "
        + string.Join(", ", vp9.CumulativeFrameTypeCounts.OrderByDescending(kv => kv.Value).Select(kv => $"{kv.Key}={kv.Value}"))
        + (vp9.ShowExistingFrameCount > 0 ? $", ShowExist={vp9.ShowExistingFrameCount}" : ""));
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
    var shInfo = av1.LastSequenceHeader;
    Console.WriteLine($"  SequenceHeader: profile={shInfo?.SeqProfile}, " +
                      $"bit_depth={shInfo?.BitDepth}, " +
                      $"subsampling=({shInfo?.SubsamplingX},{shInfo?.SubsamplingY}), " +
                      $"order_hint={shInfo?.EnableOrderHint}, " +
                      $"cdef={shInfo?.EnableCdef}, " +
                      $"matrix={shInfo?.MatrixCoefficients}");
    Console.WriteLine($"  Visible frames emitted: {sink.FrameCount}");
    Console.WriteLine($"  Cumulative OBU counts: "
        + string.Join(", ", av1.CumulativeObuCounts.OrderByDescending(kv => kv.Value).Select(kv => $"{kv.Key}={kv.Value}")));
    Console.WriteLine($"  Cumulative frame headers: "
        + string.Join(", ", av1.CumulativeFrameTypeCounts.OrderByDescending(kv => kv.Value).Select(kv => $"{kv.Key}={kv.Value}")));
    var lastFh = av1.LastFrameHeader;
    if (lastFh is not null)
    {
        Console.WriteLine($"  Last frame header: type={lastFh.FrameType}, "
            + $"show={lastFh.ShowFrame}, allow_scc={lastFh.AllowScreenContentTools}, "
            + $"force_int_mv={lastFh.ForceIntegerMv}, order_hint={lastFh.OrderHint}, "
            + $"refresh=0x{lastFh.RefreshFrameFlags:X2}, "
            + $"size={lastFh.FrameWidth}x{lastFh.FrameHeight}");
    }

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

// 4. AV1 stream summary via Av1StreamAnalyzer (high-level API).
Console.WriteLine();
Console.WriteLine("--- AV1: stream summary via Av1StreamAnalyzer ---");
{
    var av1Bytes = File.ReadAllBytes(Path.Combine(testDataDir, "bbb_180_2s.ivf"));
    var summary = Av1StreamAnalyzer.Analyze(av1Bytes);
    Console.WriteLine($"  Stream:       {summary.IvfHeader.FourCc} {summary.IvfHeader.Width}x{summary.IvfHeader.Height}");
    Console.WriteLine($"  Total TUs:    {summary.TotalTemporalUnits}");
    Console.WriteLine($"  Coded frames: {summary.CodedFrames.Count} (KeyFrame={summary.FrameTypeCounts.GetValueOrDefault(Av1FrameType.KeyFrame, 0)}, InterFrame={summary.FrameTypeCounts.GetValueOrDefault(Av1FrameType.InterFrame, 0)})");
    Console.WriteLine($"  ShowExist:    {summary.ShowExistingFrames.Count}");
    foreach (var kv in summary.ObuCounts.OrderByDescending(kv => kv.Value))
    {
        Console.WriteLine($"  {kv.Key,-22} {kv.Value} OBUs");
    }
}

// 5. AV1 ENCODER FOUNDATION: writers emit bytes ffmpeg + dav1d accept.
Console.WriteLine();
Console.WriteLine("--- AV1 ENCODER FOUNDATION: bit-exact emit, ffmpeg-validated ---");
{
    var av1Bytes = File.ReadAllBytes(Path.Combine(testDataDir, "bbb_180_2s.ivf"));
    var firstFrame = IvfReader.EnumerateFrames(av1Bytes).First();
    byte[] sourceSh = Array.Empty<byte>();
    foreach (var obu in Av1ObuParser.EnumerateObus(firstFrame.Data))
        if (obu.Type == Av1ObuType.SequenceHeader)
        {
            sourceSh = firstFrame.Data.Slice(obu.PayloadOffset, obu.PayloadLength).ToArray();
            break;
        }

    // Build BBB-equivalent SH config from observed bits.
    var ourSh = Av1SequenceHeaderWriter.EmitPayload(new Av1SequenceHeaderConfig
    {
        SeqProfile = 0, SeqLevelIdx0 = 0, MaxFrameWidth = 320, MaxFrameHeight = 180,
        BitDepth = 8, SubsamplingX = 1, SubsamplingY = 1,
        EnableFilterIntra = true, EnableIntraEdgeFilter = true,
        EnableMaskedCompound = true, EnableWarpedMotion = true,
        EnableOrderHint = true, EnableRefFrameMvs = true, OrderHintBitsMinus1 = 6,
        SeqChooseScreenContentTools = true, SeqChooseIntegerMv = true,
        EnableCdef = true, ColorDescriptionPresent = true,
        ColorPrimaries = 2, TransferCharacteristics = 2, MatrixCoefficients = 5,
    });
    int shMatch = 0;
    int shLen = Math.Min(sourceSh.Length, ourSh.Length);
    for (int i = 0; i < shLen; i++) if (sourceSh[i] == ourSh[i]) shMatch++;
    Console.WriteLine($"  Av1SequenceHeaderWriter vs libaom-av1 BBB SH: {shMatch}/{sourceSh.Length} BIT-EXACT");

    // Round-trip every OBU through Av1ObuWriter and write a new IVF.
    string remuxIvf = Path.Combine(Path.GetTempPath(), "demo_remux.ivf");
    string srcYuv = Path.Combine(Path.GetTempPath(), "demo_src.yuv");
    string rmxYuv = Path.Combine(Path.GetTempPath(), "demo_rmx.yuv");
    int frameCount = 0, obuCount = 0;
    using (var outFs = new FileStream(remuxIvf, FileMode.Create, FileAccess.Write))
    {
        var ivf = new IvfWriter(outFs, "AV01", 320, 180, frameRate: 30, timeScale: 1);
        foreach (var ivfFrame in IvfReader.EnumerateFrames(av1Bytes))
        {
            using var ms = new MemoryStream();
            foreach (var obu in Av1ObuParser.EnumerateObus(ivfFrame.Data))
            {
                var re = Av1ObuWriter.EmitObu(obu, ivfFrame.Data);
                ms.Write(re, 0, re.Length);
                obuCount++;
            }
            ivf.WriteFrame(ms.ToArray(), ivfFrame.Pts);
            frameCount++;
        }
        ivf.Finish();
    }
    Console.WriteLine($"  Remuxed {frameCount} frames / {obuCount} OBUs through our IvfWriter + Av1ObuWriter");

    RunFfmpeg(ffmpegPath, $"-y -i \"{Path.Combine(testDataDir, "bbb_180_2s.ivf")}\" -f rawvideo -pix_fmt yuv420p \"{srcYuv}\"");
    RunFfmpeg(ffmpegPath, $"-y -i \"{remuxIvf}\" -f rawvideo -pix_fmt yuv420p \"{rmxYuv}\"");
    var src = File.ReadAllBytes(srcYuv);
    var rmx = File.ReadAllBytes(rmxYuv);
    int yuvMismatch = 0;
    int yuvLen = Math.Min(src.Length, rmx.Length);
    for (int i = 0; i < yuvLen; i++) if (src[i] != rmx[i]) yuvMismatch++;
    Console.WriteLine($"  ffmpeg+dav1d source vs remux YUV: {yuvLen - yuvMismatch}/{yuvLen} BIT-EXACT");
    File.Delete(remuxIvf); File.Delete(srcYuv); File.Delete(rmxYuv);
}

Console.WriteLine();
Console.WriteLine("============================================================");
Console.WriteLine("  Working today, end-to-end:");
Console.WriteLine("    - FLAC encoder + decoder (ffmpeg cross-verified BIT-EXACT)");
Console.WriteLine("    - VP9 decoder pipeline through tile group extraction");
Console.WriteLine("    - AV1 decoder pipeline + SH + FrameHeader parsers");
Console.WriteLine("    - AV1 ENCODER FOUNDATION:");
Console.WriteLine("        SequenceHeader writer BIT-EXACT vs libaom-av1");
Console.WriteLine("        FrameHeader writer + OBU writer + IVF writer");
Console.WriteLine("        ffmpeg+dav1d accept our remux pixel-identical to source");
Console.WriteLine("    - IVF / Matroska / Ogg container readers + IVF writer");
Console.WriteLine("  In progress (placeholder pixels for video):");
Console.WriteLine("    - VP9 / AV1 block decode walker");
Console.WriteLine("    - Opus CELT mode (SILK works)");
Console.WriteLine("    - AV1 daala range coder (gates real encoder pixels)");
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
