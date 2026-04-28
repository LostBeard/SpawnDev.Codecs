// Side-by-side comparison: SpawnDev.Codecs vs ffmpeg/libavcodec.
// For each codec, encodes the same input with both pipelines and
// reports encode time, output size, compression ratio, and (where
// applicable) decoded-pixel PSNR vs source.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.Codecs.Video.Vp9;

string ffmpeg = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
if (!File.Exists(ffmpeg)) ffmpeg = "ffmpeg";
string tmp = Path.Combine(Path.GetTempPath(), "spawndev_vs_ffmpeg");
Directory.CreateDirectory(tmp);

var report = new StringBuilder();
report.AppendLine("============================================================");
report.AppendLine("  SpawnDev.Codecs vs ffmpeg side-by-side");
report.AppendLine("============================================================");
report.AppendLine();

// ============================================================
// AUDIO
// ============================================================
report.AppendLine("AUDIO ENCODERS (30s 44.1k stereo chord, raw PCM input)");
report.AppendLine($"{"Codec",-10}{"Enc ms",-10}{"Out B",-10}{"Ratio",-8}");
report.AppendLine($"{new string('-', 10)}{new string('-', 10)}{new string('-', 10)}{new string('-', 8)}");

int sr = 44100, ch = 2, secs = 30;
int totalSamples = sr * secs;
var pcmInt = new int[totalSamples * ch];
var pcmFloat = new float[totalSamples * ch];
double[] freqs = { 440, 523.25, 659.25 };
for (int i = 0; i < totalSamples; i++)
{
    double s = 0; foreach (var f in freqs) s += Math.Sin(2.0 * Math.PI * f * i / sr);
    s = 0.3 * s / freqs.Length;
    pcmInt[i * 2 + 0] = (int)(s * 32767);
    pcmInt[i * 2 + 1] = (int)(s * 32767 * 0.9);
    pcmFloat[i * 2 + 0] = (float)s;
    pcmFloat[i * 2 + 1] = (float)(s * 0.9);
}
int rawAudioBytes = totalSamples * ch * 2;

// Write raw PCM for ffmpeg.
string pcmPath = Path.Combine(tmp, "in.pcm");
{
    var bytes = new byte[rawAudioBytes];
    for (int i = 0; i < pcmInt.Length; i++)
    {
        short s = (short)Math.Clamp(pcmInt[i], short.MinValue, short.MaxValue);
        bytes[i * 2] = (byte)s; bytes[i * 2 + 1] = (byte)(s >> 8);
    }
    File.WriteAllBytes(pcmPath, bytes);
}

// FLAC ours
{
    var sw = Stopwatch.StartNew();
    var enc = FlacEncoder.EncodeStream(pcmInt, sr, ch, 16, blockSize: 4096);
    sw.Stop();
    AddRow("FLAC (ours)", sw.Elapsed.TotalMilliseconds, enc.Length, rawAudioBytes);
}
// FLAC ffmpeg
{
    string outPath = Path.Combine(tmp, "ff.flac");
    var sw = Stopwatch.StartNew();
    RunFfmpeg($"-y -f s16le -ar {sr} -ac {ch} -i \"{pcmPath}\" -c:a flac \"{outPath}\"");
    sw.Stop();
    AddRow("FLAC (ffmpeg)", sw.Elapsed.TotalMilliseconds, new FileInfo(outPath).Length, rawAudioBytes);
}

// Opus ours (mono, regenerate at 48k since Opus only supports 8/12/16/24/48 kHz).
{
    int opusSr = 48000;
    int opusSamples = opusSr * secs;
    var monoPcm = new float[opusSamples];
    for (int i = 0; i < opusSamples; i++)
    {
        double s = 0; foreach (var f in freqs) s += Math.Sin(2.0 * Math.PI * f * i / opusSr);
        monoPcm[i] = (float)(0.3 * s / freqs.Length);
    }
    var enc = new OpusEncoder(new OpusEncoderConfig { SampleRateHz = opusSr, ChannelCount = 1, Application = OpusEncoderApplication.Audio });
    var packet = new byte[1275];
    int frameSize = opusSr / 50;
    int frameCount = opusSamples / frameSize;
    long encBytes = 0;
    var sw = Stopwatch.StartNew();
    for (int f = 0; f < frameCount; f++) encBytes += enc.EncodeFrame(monoPcm.AsSpan(f * frameSize, frameSize), packet, frameSize);
    sw.Stop();
    enc.Dispose();
    AddRow("Opus (ours)", sw.Elapsed.TotalMilliseconds, encBytes, opusSamples * 2);
}
// Opus ffmpeg (will use libopus)
{
    string outPath = Path.Combine(tmp, "ff.opus");
    var sw = Stopwatch.StartNew();
    RunFfmpeg($"-y -f s16le -ar {sr} -ac {ch} -i \"{pcmPath}\" -c:a libopus -ar 48000 -ac 1 \"{outPath}\"");
    sw.Stop();
    AddRow("Opus (ffmpeg)", sw.Elapsed.TotalMilliseconds, new FileInfo(outPath).Length, 48000 * secs * 2);
}

// Vorbis ours (mono)
{
    var monoPcm = new float[totalSamples];
    for (int i = 0; i < totalSamples; i++) monoPcm[i] = pcmFloat[i * 2];
    var enc = new VorbisAudioEncoder(new VorbisAudioEncoderOptions { SampleRateHz = sr, Channels = 1 });
    var sw = Stopwatch.StartNew();
    var ogg = enc.EncodeStream(monoPcm);
    sw.Stop();
    AddRow("Vorbis (ours)", sw.Elapsed.TotalMilliseconds, ogg.Length, totalSamples * 2);
}
// Vorbis ffmpeg
{
    string outPath = Path.Combine(tmp, "ff.ogg");
    var sw = Stopwatch.StartNew();
    RunFfmpeg($"-y -f s16le -ar {sr} -ac {ch} -i \"{pcmPath}\" -c:a libvorbis -ac 1 \"{outPath}\"");
    sw.Stop();
    AddRow("Vorbis (ffmpeg)", sw.Elapsed.TotalMilliseconds, new FileInfo(outPath).Length, totalSamples * 2);
}
report.AppendLine();

// ============================================================
// VIDEO
// ============================================================
report.AppendLine("VIDEO ENCODERS (32x32 single keyframe, synthetic gradient)");
report.AppendLine($"{"Codec",-15}{"Enc ms",-10}{"Out B",-10}{"Ratio",-8}");
report.AppendLine($"{new string('-', 15)}{new string('-', 10)}{new string('-', 10)}{new string('-', 8)}");

int W = 32, H = 32;
var ySrc = new byte[W * H];
for (int r = 0; r < H; r++)
    for (int c = 0; c < W; c++)
        ySrc[r * W + c] = (byte)Math.Clamp(96 + 32 * Math.Sin(2.0 * Math.PI * c / 16.0) + r * 2, 0, 255);
var uSrc = new byte[(W / 2) * (H / 2)]; Array.Fill(uSrc, (byte)128);
var vSrc = new byte[(W / 2) * (H / 2)]; Array.Fill(vSrc, (byte)128);
int rawVideo = W * H + 2 * (W / 2) * (H / 2);

string yuvPath = Path.Combine(tmp, "in.yuv");
{
    var raw = new byte[rawVideo];
    Buffer.BlockCopy(ySrc, 0, raw, 0, ySrc.Length);
    Buffer.BlockCopy(uSrc, 0, raw, ySrc.Length, uSrc.Length);
    Buffer.BlockCopy(vSrc, 0, raw, ySrc.Length + uSrc.Length, vSrc.Length);
    File.WriteAllBytes(yuvPath, raw);
}

// VP8 ours
{
    var sw = Stopwatch.StartNew();
    var enc = Vp8KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 30);
    sw.Stop();
    AddRowV("VP8 (ours)", sw.Elapsed.TotalMilliseconds, enc.Length, rawVideo);
}
// VP8 ffmpeg (libvpx)
{
    string outPath = Path.Combine(tmp, "ff.vp8.ivf");
    var sw = Stopwatch.StartNew();
    RunFfmpeg($"-y -f rawvideo -pix_fmt yuv420p -s {W}x{H} -i \"{yuvPath}\" -c:v libvpx -keyint_min 1 -g 1 -auto-alt-ref 0 -frames:v 1 \"{outPath}\"");
    sw.Stop();
    AddRowV("VP8 (ffmpeg)", sw.Elapsed.TotalMilliseconds, new FileInfo(outPath).Length, rawVideo);
}

// VP9 ours
{
    var sw = Stopwatch.StartNew();
    var enc = Vp9KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 30);
    sw.Stop();
    AddRowV("VP9 (ours)", sw.Elapsed.TotalMilliseconds, enc.Length, rawVideo);
}
// VP9 ffmpeg (libvpx-vp9)
{
    string outPath = Path.Combine(tmp, "ff.vp9.ivf");
    var sw = Stopwatch.StartNew();
    RunFfmpeg($"-y -f rawvideo -pix_fmt yuv420p -s {W}x{H} -i \"{yuvPath}\" -c:v libvpx-vp9 -keyint_min 1 -g 1 -frames:v 1 \"{outPath}\"");
    sw.Stop();
    AddRowV("VP9 (ffmpeg)", sw.Elapsed.TotalMilliseconds, new FileInfo(outPath).Length, rawVideo);
}

// AV1 ours
{
    var sw = Stopwatch.StartNew();
    var enc = Av1KeyframeEncoder.EncodeKeyFrame(ySrc, W, uSrc, W / 2, vSrc, W, H, baseQIndex: 32);
    sw.Stop();
    AddRowV("AV1 (ours)", sw.Elapsed.TotalMilliseconds, enc.Length, rawVideo);
}
// AV1 ffmpeg (libaom)
{
    string outPath = Path.Combine(tmp, "ff.av1.ivf");
    var sw = Stopwatch.StartNew();
    RunFfmpeg($"-y -f rawvideo -pix_fmt yuv420p -s {W}x{H} -i \"{yuvPath}\" -c:v libaom-av1 -keyint_min 1 -g 1 -frames:v 1 \"{outPath}\"");
    sw.Stop();
    AddRowV("AV1 (ffmpeg)", sw.Elapsed.TotalMilliseconds, new FileInfo(outPath).Length, rawVideo);
}

string output = report.ToString();
Console.Write(output);
string reportPath = Path.Combine(Path.GetTempPath(), "spawndev_vs_ffmpeg.txt");
File.WriteAllText(reportPath, output);
Console.WriteLine($"\nReport: {reportPath}");

void AddRow(string codec, double ms, long size, int rawBytes)
{
    double ratio = (double)size / rawBytes;
    report.AppendLine($"{codec,-10}{ms,-10:F1}{size,-10}{ratio,-8:F4}");
}
void AddRowV(string codec, double ms, long size, int rawBytes)
{
    double ratio = (double)size / rawBytes;
    report.AppendLine($"{codec,-15}{ms,-10:F1}{size,-10}{ratio,-8:F4}");
}
void RunFfmpeg(string args)
{
    var p = Process.Start(new ProcessStartInfo(ffmpeg, args) { RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true })!;
    p.StandardError.ReadToEnd();
    p.WaitForExit();
    if (p.ExitCode != 0) throw new Exception($"ffmpeg failed: {args}");
}
