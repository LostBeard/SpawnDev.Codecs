// Produce playable + listenable BBB transcode artifacts through every
// SpawnDev.Codecs encoder, so TJ can eyeball quality + audibility in
// VLC. Extracts a short clip from BBB (default 60 frames = 1 second at
// 60fps + 1 second audio), encodes through every codec, packages each
// for VLC playback.
//
// Usage: dotnet run bbb_transcode_artifacts.cs [seconds=1]

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.Codecs.Video.Vp9;

int seconds = args.Length >= 1 && int.TryParse(args[0], out int s) ? s : 1;
const int W = 1920;
const int H = 1072;
const int Fps = 60;
int frameCount = seconds * Fps;
string source = "V:\\Video\\Big Buck Bunny - FULL HD 60FPS.mp4";
string ffmpeg = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
string outDir = Path.Combine(Path.GetTempPath(), "spawndev_bbb_artifacts");
Directory.CreateDirectory(outDir);

if (!File.Exists(source))
{
    Console.WriteLine($"Source not found: {source}");
    Environment.Exit(1);
}

Console.WriteLine($"Producing {seconds}s ({frameCount} frames @ {W}x{H}) BBB transcode artifacts.");
Console.WriteLine($"Output dir: {outDir}");
Console.WriteLine();

// 1) Extract video frames as raw YUV420
string yuvPath = Path.Combine(outDir, "input.yuv");
RunFfmpeg($"-y -i \"{source}\" -vf crop={W}:{H}:0:0 -frames:v {frameCount} -f rawvideo -pix_fmt yuv420p \"{yuvPath}\"");
int frameSize = W * H + 2 * (W / 2) * (H / 2);
var allFrames = File.ReadAllBytes(yuvPath);
Console.WriteLine($"Extracted {allFrames.Length / frameSize} frames ({allFrames.Length / 1024 / 1024}MB raw).");

// 2) Extract audio at 48k mono (for Opus) and 44.1k stereo (for FLAC + Vorbis)
string audio48k = Path.Combine(outDir, "audio48k.pcm");
string audio44k = Path.Combine(outDir, "audio44k.pcm");
RunFfmpeg($"-y -i \"{source}\" -t {seconds} -f s16le -ac 1 -ar 48000 \"{audio48k}\"");
RunFfmpeg($"-y -i \"{source}\" -t {seconds} -f s16le -ac 2 -ar 44100 \"{audio44k}\"");
Console.WriteLine($"Extracted audio: 48k mono ({new FileInfo(audio48k).Length}B), 44.1k stereo ({new FileInfo(audio44k).Length}B).");
Console.WriteLine();

// === VIDEO: VP8 ===
{
    string ivf = Path.Combine(outDir, "vp8.ivf");
    string mp4 = Path.Combine(outDir, "vp8.mp4");
    var sw = Stopwatch.StartNew();
    using (var fs = File.Create(ivf))
    {
        var w = new IvfWriter(fs, "VP80", W, H, frameRate: Fps, timeScale: 1, numFrames: 0, leaveOpen: true);
        for (int f = 0; f < frameCount; f++)
        {
            var (y, u, v) = SliceFrame(allFrames, f);
            w.WriteFrame(Vp8KeyframeEncoder.EncodeKeyFrame(y, W, u, W / 2, v, W, H, baseQIndex: 30), f);
        }
        w.Finish();
    }
    sw.Stop();
    long ivfSz = new FileInfo(ivf).Length;
    bool remuxOk = TryRunFfmpeg($"-y -i \"{ivf}\" -c:v libx264 -pix_fmt yuv420p \"{mp4}\"");
    string mp4Status = remuxOk ? $"-> MP4 {new FileInfo(mp4).Length / 1024}KB" : "(MP4 remux failed; .ivf still playable in ffmpeg)";
    Console.WriteLine($"VP8: {frameCount} frames in {sw.Elapsed.TotalSeconds:F1}s ({frameCount / sw.Elapsed.TotalSeconds:F1} fps), IVF {ivfSz / 1024}KB {mp4Status}");
}

// === VIDEO: VP9 ===
{
    string ivf = Path.Combine(outDir, "vp9.ivf");
    string mp4 = Path.Combine(outDir, "vp9.mp4");
    var sw = Stopwatch.StartNew();
    using (var fs = File.Create(ivf))
    {
        var w = new IvfWriter(fs, "VP90", W, H, frameRate: Fps, timeScale: 1, numFrames: 0, leaveOpen: true);
        for (int f = 0; f < frameCount; f++)
        {
            var (y, u, v) = SliceFrame(allFrames, f);
            w.WriteFrame(Vp9KeyframeEncoder.EncodeKeyFrame(y, W, u, W / 2, v, W, H, baseQIndex: 30), f);
        }
        w.Finish();
    }
    sw.Stop();
    long ivfSz = new FileInfo(ivf).Length;
    bool remuxOk = TryRunFfmpeg($"-y -i \"{ivf}\" -c:v libx264 -pix_fmt yuv420p \"{mp4}\"");
    string mp4Status = remuxOk ? $"-> MP4 {new FileInfo(mp4).Length / 1024}KB" : "(MP4 remux failed; .ivf still playable in ffmpeg)";
    Console.WriteLine($"VP9: {frameCount} frames in {sw.Elapsed.TotalSeconds:F1}s ({frameCount / sw.Elapsed.TotalSeconds:F1} fps), IVF {ivfSz / 1024}KB {mp4Status}");
}

// === VIDEO: AV1 (decoded via libdav1d for VLC) ===
{
    string ivf = Path.Combine(outDir, "av1.ivf");
    string mp4 = Path.Combine(outDir, "av1.mp4");
    var sw = Stopwatch.StartNew();
    using (var fs = File.Create(ivf))
    {
        var w = new IvfWriter(fs, "AV01", W, H, frameRate: Fps, timeScale: 1, numFrames: 0, leaveOpen: true);
        for (int f = 0; f < frameCount; f++)
        {
            var (y, u, v) = SliceFrame(allFrames, f);
            w.WriteFrame(Av1KeyframeEncoder.EncodeKeyFrame(y, W, u, W / 2, v, W, H, baseQIndex: 32), f);
        }
        w.Finish();
    }
    sw.Stop();
    long ivfSz = new FileInfo(ivf).Length;
    bool remuxOk = TryRunFfmpeg($"-y -c:v libdav1d -i \"{ivf}\" -c:v libx264 -pix_fmt yuv420p \"{mp4}\"");
    string mp4Status = remuxOk ? $"-> MP4 {new FileInfo(mp4).Length / 1024}KB" : "(libdav1d remux failed; .ivf still readable via ffmpeg)";
    Console.WriteLine($"AV1: {frameCount} frames in {sw.Elapsed.TotalSeconds:F1}s ({frameCount / sw.Elapsed.TotalSeconds:F1} fps), IVF {ivfSz / 1024}KB {mp4Status}");
}

// === AUDIO: FLAC ===
{
    string flacPath = Path.Combine(outDir, "audio.flac");
    var pcmBytes = File.ReadAllBytes(audio44k);
    var samples = new int[pcmBytes.Length / 2];
    for (int i = 0; i < samples.Length; i++) samples[i] = (short)(pcmBytes[i * 2] | (pcmBytes[i * 2 + 1] << 8));
    var sw = Stopwatch.StartNew();
    var enc = FlacEncoder.EncodeStream(samples, sampleRateHz: 44100, channels: 2, bitsPerSample: 16);
    sw.Stop();
    File.WriteAllBytes(flacPath, enc);
    Console.WriteLine($"FLAC: {samples.Length} samples in {sw.Elapsed.TotalSeconds:F2}s -> {enc.Length / 1024}KB ({(double)enc.Length / pcmBytes.Length:P1} of raw)");
}

// === AUDIO: Opus ===
{
    string opusPath = Path.Combine(outDir, "audio.opus");
    var pcmBytes = File.ReadAllBytes(audio48k);
    var pcm = new float[pcmBytes.Length / 2];
    for (int i = 0; i < pcm.Length; i++) pcm[i] = ((short)(pcmBytes[i * 2] | (pcmBytes[i * 2 + 1] << 8))) / 32768f;
    var enc = new OpusEncoder(new OpusEncoderConfig { SampleRateHz = 48000, ChannelCount = 1, Application = OpusEncoderApplication.Audio });
    var packetBuf = new byte[1275];
    int frameSamples = 48000 / 50; // 20ms
    int packetCount = pcm.Length / frameSamples;
    var packets = new System.Collections.Generic.List<byte[]>(packetCount);
    var sw = Stopwatch.StartNew();
    for (int f = 0; f < packetCount; f++)
    {
        int n = enc.EncodeFrame(pcm.AsSpan(f * frameSamples, frameSamples), packetBuf, frameSamples);
        packets.Add(packetBuf.AsSpan(0, n).ToArray());
    }
    sw.Stop();
    enc.Dispose();
    var ogg = OpusOggEncoder.Encode(packets, new OpusOggEncoderOptions { OutputChannels = 1, InputSampleRateHz = 48000, PreSkip = 312 });
    File.WriteAllBytes(opusPath, ogg);
    Console.WriteLine($"Opus: {pcm.Length} samples in {sw.Elapsed.TotalSeconds:F2}s -> {ogg.Length / 1024}KB ({(double)ogg.Length / pcmBytes.Length:P1} of raw)");
}

// === AUDIO: Vorbis ===
{
    string oggPath = Path.Combine(outDir, "audio.ogg");
    var pcmBytes = File.ReadAllBytes(audio44k);
    // Reduce to mono by averaging channels (Vorbis encoder is mono-only here).
    var monoPcm = new float[pcmBytes.Length / 4];
    for (int i = 0; i < monoPcm.Length; i++)
    {
        short l = (short)(pcmBytes[i * 4] | (pcmBytes[i * 4 + 1] << 8));
        short r = (short)(pcmBytes[i * 4 + 2] | (pcmBytes[i * 4 + 3] << 8));
        monoPcm[i] = (l + r) / 2 / 32768f;
    }
    var enc = new VorbisAudioEncoder(new VorbisAudioEncoderOptions { SampleRateHz = 44100, Channels = 1 });
    var sw = Stopwatch.StartNew();
    var ogg = enc.EncodeStream(monoPcm);
    sw.Stop();
    File.WriteAllBytes(oggPath, ogg);
    Console.WriteLine($"Vorbis: {monoPcm.Length} samples in {sw.Elapsed.TotalSeconds:F2}s -> {ogg.Length / 1024}KB");
}

Console.WriteLine();
Console.WriteLine("=========================================================================");
Console.WriteLine($"  Open the artifacts in VLC to verify visually + audibly:");
Console.WriteLine($"  Video: {Path.Combine(outDir, "vp8.mp4")}");
Console.WriteLine($"  Video: {Path.Combine(outDir, "vp9.mp4")}");
Console.WriteLine($"  Video: {Path.Combine(outDir, "av1.mp4")}");
Console.WriteLine($"  Audio: {Path.Combine(outDir, "audio.flac")}");
Console.WriteLine($"  Audio: {Path.Combine(outDir, "audio.opus")}");
Console.WriteLine($"  Audio: {Path.Combine(outDir, "audio.ogg")}");
Console.WriteLine("=========================================================================");

(byte[] y, byte[] u, byte[] v) SliceFrame(byte[] allFrames, int frameIndex)
{
    int frameSize = W * H + 2 * (W / 2) * (H / 2);
    int yOff = frameIndex * frameSize;
    int uOff = yOff + W * H;
    int vOff = uOff + (W / 2) * (H / 2);
    return (
        allFrames[yOff..uOff],
        allFrames[uOff..vOff],
        allFrames[vOff..(vOff + (W / 2) * (H / 2))]
    );
}

void RunFfmpeg(string args)
{
    var p = Process.Start(new ProcessStartInfo(ffmpeg, args) { RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true })!;
    p.StandardError.ReadToEnd();
    p.WaitForExit();
    if (p.ExitCode != 0) throw new Exception($"ffmpeg failed: {args}");
}
bool TryRunFfmpeg(string args)
{
    var p = Process.Start(new ProcessStartInfo(ffmpeg, args) { RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true })!;
    p.StandardError.ReadToEnd();
    p.WaitForExit();
    return p.ExitCode == 0;
}
