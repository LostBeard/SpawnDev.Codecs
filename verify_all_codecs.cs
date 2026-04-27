// Master verification: runs every encoder + decoder smoke + writes
// every visual/audio artifact to %TEMP%\spawndev_verify\ for manual
// VLC / image-viewer eyeball check.
//
// Output structure:
//   %TEMP%\spawndev_verify\
//     vp8_animation.mp4         <- 60 frames VP8 encode -> ffmpeg remux
//     vp9_animation.mp4         <- 60 frames VP9 encode -> ffmpeg remux
//     vorbis_chord.ogg          <- 3-sec A-minor chord Vorbis encode
//     flac_chord.flac           <- 3-sec A-minor chord FLAC encode (lossless)
//     bbb_first_frame.png       <- VP9 BBB first frame visual (reference)
//     bbb_av1_first_frame.png   <- AV1 BBB first frame visual (reference)
//     report.txt                <- Summary of what each file demonstrates
//
// Usage: dotnet run verify_all_codecs.cs

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.Codecs.Video.Vp9;

string outDir = Path.Combine(Path.GetTempPath(), "spawndev_verify");
Directory.CreateDirectory(outDir);

string ffmpeg = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
if (!File.Exists(ffmpeg)) ffmpeg = "ffmpeg";

int passed = 0, failed = 0;
var summary = new System.Text.StringBuilder();
summary.AppendLine("SpawnDev.Codecs Verification Report");
summary.AppendLine("===================================");
summary.AppendLine();

void Section(string name, Action body)
{
    Console.WriteLine($"== {name} ==");
    summary.AppendLine($"== {name} ==");
    try { body(); passed++; }
    catch (Exception ex) { Console.WriteLine($"FAIL: {ex.Message}"); summary.AppendLine($"FAIL: {ex.Message}"); failed++; }
    Console.WriteLine();
    summary.AppendLine();
}

bool RunFfmpeg(string args)
{
    var p = Process.Start(new ProcessStartInfo(ffmpeg, args) { RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true })!;
    p.StandardError.ReadToEnd();
    p.WaitForExit();
    return p.ExitCode == 0;
}

// === VP9 keyframe encoder -> IVF -> MP4 ===
Section("VP9 encoder (60-frame animation)", () =>
{
    int W = 16, H = 16, Frames = 60;
    string ivf = Path.Combine(outDir, "vp9_animation.ivf");
    string mp4 = Path.Combine(outDir, "vp9_animation.mp4");
    using (var fs = File.Create(ivf))
    {
        var w = new IvfWriter(fs, "VP90", W, H, frameRate: 30, timeScale: 1, numFrames: 0, leaveOpen: true);
        var ySrc = new byte[W * H]; var uSrc = new byte[(W / 2) * (H / 2)]; var vSrc = new byte[(W / 2) * (H / 2)];
        for (int f = 0; f < Frames; f++)
        {
            for (int r = 0; r < H; r++) for (int c = 0; c < W; c++)
                ySrc[r * W + c] = (byte)Math.Clamp(80 + 40 * Math.Sin(2.0 * Math.PI * (c + f) / W) + r * 4, 0, 255);
            for (int r = 0; r < H / 2; r++) for (int c = 0; c < W / 2; c++) { uSrc[r * (W / 2) + c] = (byte)(128 + (f - 30)); vSrc[r * (W / 2) + c] = (byte)(128 - (f - 30)); }
            w.WriteFrame(Vp9KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 30), f);
        }
        w.Finish();
    }
    if (!RunFfmpeg($"-y -i \"{ivf}\" -c:v libx264 -pix_fmt yuv420p \"{mp4}\"")) throw new Exception("ffmpeg failed on VP9 IVF");
    long sz = new FileInfo(mp4).Length;
    Console.WriteLine($"  PASS: {Frames} frames -> {mp4} ({sz:N0}B)");
    summary.AppendLine($"  PASS: {Frames} frames -> {mp4} ({sz:N0}B) - PLAYABLE IN VLC");
});

// === VP8 keyframe encoder -> IVF -> MP4 ===
Section("VP8 encoder (60-frame animation)", () =>
{
    int W = 32, H = 32, Frames = 60;
    string ivf = Path.Combine(outDir, "vp8_animation.ivf");
    string mp4 = Path.Combine(outDir, "vp8_animation.mp4");
    using (var fs = File.Create(ivf))
    {
        var w = new IvfWriter(fs, "VP80", W, H, frameRate: 30, timeScale: 1, numFrames: 0, leaveOpen: true);
        var ySrc = new byte[W * H]; var uSrc = new byte[(W / 2) * (H / 2)]; var vSrc = new byte[(W / 2) * (H / 2)];
        for (int f = 0; f < Frames; f++)
        {
            for (int r = 0; r < H; r++) for (int c = 0; c < W; c++)
                ySrc[r * W + c] = (byte)Math.Clamp(80 + 40 * Math.Sin(2.0 * Math.PI * (c + f) / W) + r * 2, 0, 255);
            for (int r = 0; r < H / 2; r++) for (int c = 0; c < W / 2; c++) { uSrc[r * (W / 2) + c] = (byte)(128 + (f - 30)); vSrc[r * (W / 2) + c] = (byte)(128 - (f - 30)); }
            w.WriteFrame(Vp8KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 30), f);
        }
        w.Finish();
    }
    if (!RunFfmpeg($"-y -i \"{ivf}\" -c:v libx264 -pix_fmt yuv420p \"{mp4}\"")) throw new Exception("ffmpeg failed on VP8 IVF");
    long sz = new FileInfo(mp4).Length;
    Console.WriteLine($"  PASS: {Frames} frames -> {mp4} ({sz:N0}B)");
    summary.AppendLine($"  PASS: {Frames} frames -> {mp4} ({sz:N0}B) - PLAYABLE IN VLC");
});

// === Vorbis encoder -> ogg ===
Section("Vorbis encoder (3-sec A-minor chord)", () =>
{
    int sr = 44100; int n = sr * 3;
    var pcm = new float[n];
    double[] freqs = { 440, 523.25, 659.25 };
    for (int i = 0; i < n; i++)
    {
        double sample = 0, env = Math.Min(1.0, i / (0.05 * sr));
        foreach (var f in freqs) sample += Math.Sin(2.0 * Math.PI * f * i / sr);
        pcm[i] = (float)(0.3 * env * sample / freqs.Length);
    }
    var enc = new VorbisAudioEncoder(new VorbisAudioEncoderOptions { SampleRateHz = sr, Channels = 1 });
    var ogg = enc.EncodeStream(pcm);
    string oggPath = Path.Combine(outDir, "vorbis_chord.ogg");
    File.WriteAllBytes(oggPath, ogg);
    string wavPath = Path.Combine(outDir, "vorbis_chord_via_ffmpeg.wav");
    if (!RunFfmpeg($"-y -i \"{oggPath}\" -acodec pcm_s16le \"{wavPath}\"")) throw new Exception("ffmpeg failed on Vorbis ogg");
    Console.WriteLine($"  PASS: {ogg.Length:N0}B ogg -> {oggPath}");
    summary.AppendLine($"  PASS: {ogg.Length:N0}B ogg -> {oggPath} - PLAYABLE IN VLC (audio)");
});

// === FLAC encoder + decoder lossless round-trip ===
Section("FLAC encoder + bit-exact decoder", () =>
{
    int sr = 44100; int n = sr * 3;
    var samples = new int[n];
    double[] freqs = { 440, 523.25, 659.25 };
    for (int i = 0; i < n; i++)
    {
        double sample = 0, env = Math.Min(1.0, i / (0.05 * sr));
        foreach (var f in freqs) sample += Math.Sin(2.0 * Math.PI * f * i / sr);
        samples[i] = (int)(env * sample / freqs.Length * 16000);
    }
    string flacPath = Path.Combine(outDir, "flac_chord.flac");
    FlacEncoder.EncodeToFile(flacPath, samples, sr, channels: 1, bitsPerSample: 16);
    var dec = FlacDecoder.DecodeFile(flacPath);
    if (dec.InterleavedSamples.Length != n) throw new Exception($"length mismatch: {dec.InterleavedSamples.Length} vs {n}");
    for (int i = 0; i < n; i++)
        if (dec.InterleavedSamples[i] != samples[i]) throw new Exception($"bit-exact fail at {i}");
    long flacSize = new FileInfo(flacPath).Length;
    Console.WriteLine($"  PASS: {flacSize:N0}B FLAC, lossless round-trip -> {flacPath}");
    summary.AppendLine($"  PASS: {flacSize:N0}B FLAC, lossless round-trip -> {flacPath} - PLAYABLE IN VLC (audio)");
});

// === Visual reference frames ===
Section("Visual reference (BBB first frames)", () =>
{
    string vp9Y = "SpawnDev.Codecs.Demo.Shared/TestData/bbb_first_frame.yuv";
    string vp9Png = Path.Combine(outDir, "bbb_first_frame.png");
    if (File.Exists(vp9Y))
    {
        if (!RunFfmpeg($"-y -f rawvideo -pix_fmt yuv420p -s 320x180 -i \"{vp9Y}\" -frames:v 1 \"{vp9Png}\"")) throw new Exception("ffmpeg failed on VP9 YUV");
        Console.WriteLine($"  VP9 reference: {vp9Png}");
        summary.AppendLine($"  VP9 first frame -> {vp9Png}");
    }
    string av1Y = "SpawnDev.Codecs.Demo.Shared/TestData/bbb_av1_first_frame.yuv";
    string av1Png = Path.Combine(outDir, "bbb_av1_first_frame.png");
    if (File.Exists(av1Y))
    {
        if (!RunFfmpeg($"-y -f rawvideo -pix_fmt yuv420p -s 320x180 -i \"{av1Y}\" -frames:v 1 \"{av1Png}\"")) throw new Exception("ffmpeg failed on AV1 YUV");
        Console.WriteLine($"  AV1 reference: {av1Png}");
        summary.AppendLine($"  AV1 first frame -> {av1Png}");
    }
});

string reportPath = Path.Combine(outDir, "report.txt");
summary.AppendLine();
summary.AppendLine($"Result: {passed}/{passed + failed} sections passed.");
summary.AppendLine($"Open {outDir} and play the .mp4 / .ogg / .flac files in VLC.");
File.WriteAllText(reportPath, summary.ToString());

Console.WriteLine($"========================================================");
Console.WriteLine($"  {passed}/{passed + failed} sections passed.");
Console.WriteLine($"  Output: {outDir}");
Console.WriteLine($"  Report: {reportPath}");
Console.WriteLine($"========================================================");
if (failed != 0) Environment.Exit(1);
