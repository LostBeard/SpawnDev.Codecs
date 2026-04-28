// Audio quality benchmark: encode the BBB audio track through our
// FLAC/Opus/Vorbis encoders, decode each output back to PCM via ffmpeg,
// and report SNR (RMS of error vs source) + size.
//
// Usage: dotnet run benchmark_audio_quality.cs [seconds=10]

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.Codecs.Audio.Vorbis;

int seconds = args.Length >= 1 && int.TryParse(args[0], out int s) ? s : 10;
string source = "V:\\Video\\Big Buck Bunny - FULL HD 60FPS.mp4";
string ffmpeg = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
string outDir = Path.Combine(Path.GetTempPath(), "spawndev_audio_psnr");
Directory.CreateDirectory(outDir);
if (!File.Exists(source)) { Console.WriteLine($"Source not found: {source}"); Environment.Exit(1); }

// Extract source audio: 44.1k stereo for FLAC + Vorbis, 48k mono for Opus.
string src44s = Path.Combine(outDir, "src_44k_st.pcm");
string src48m = Path.Combine(outDir, "src_48k_mn.pcm");
RunFf($"-y -i \"{source}\" -t {seconds} -f s16le -ac 2 -ar 44100 \"{src44s}\"");
RunFf($"-y -i \"{source}\" -t {seconds} -f s16le -ac 1 -ar 48000 \"{src48m}\"");
var src44sBytes = File.ReadAllBytes(src44s);
var src48mBytes = File.ReadAllBytes(src48m);
Console.WriteLine($"Source: {seconds}s, 44k stereo {src44sBytes.Length}B + 48k mono {src48mBytes.Length}B");
Console.WriteLine();

Console.WriteLine($"Audio quality vs source ({seconds}s of BBB):");
Console.WriteLine($"{"Codec",-15}{"size KB",-10}{"ratio",-10}{"SNR dB",-12}{"max diff",-12}{"enc ms",-10}");
Console.WriteLine($"{new string('-', 15)}{new string('-', 10)}{new string('-', 10)}{new string('-', 12)}{new string('-', 12)}{new string('-', 10)}");

// FLAC ours - lossless, expect SNR = inf
{
    var samples = new int[src44sBytes.Length / 2];
    for (int i = 0; i < samples.Length; i++) samples[i] = (short)(src44sBytes[i * 2] | (src44sBytes[i * 2 + 1] << 8));
    var sw = Stopwatch.StartNew();
    var enc = FlacEncoder.EncodeStream(samples, 44100, 2, 16);
    sw.Stop();
    string flacPath = Path.Combine(outDir, "ours.flac");
    File.WriteAllBytes(flacPath, enc);
    string decPath = Path.Combine(outDir, "ours_flac_dec.pcm");
    RunFf($"-y -i \"{flacPath}\" -f s16le \"{decPath}\"");
    var dec = File.ReadAllBytes(decPath);
    var (snr, maxDiff) = ComputeSnr16(src44sBytes, dec);
    PrintRow("FLAC (ours)", enc.Length, src44sBytes.Length, snr, maxDiff, sw.Elapsed.TotalMilliseconds);
}

// Opus ours
{
    var pcm = new float[src48mBytes.Length / 2];
    for (int i = 0; i < pcm.Length; i++) pcm[i] = ((short)(src48mBytes[i * 2] | (src48mBytes[i * 2 + 1] << 8))) / 32768f;
    var enc = new OpusEncoder(new OpusEncoderConfig { SampleRateHz = 48000, ChannelCount = 1, Application = OpusEncoderApplication.Audio });
    var packets = new System.Collections.Generic.List<byte[]>();
    var packetBuf = new byte[1275];
    int frameSamples = 48000 / 50;
    int frameCount = pcm.Length / frameSamples;
    long totalEnc = 0;
    var sw = Stopwatch.StartNew();
    for (int f = 0; f < frameCount; f++)
    {
        int n = enc.EncodeFrame(pcm.AsSpan(f * frameSamples, frameSamples), packetBuf, frameSamples);
        totalEnc += n;
        packets.Add(packetBuf.AsSpan(0, n).ToArray());
    }
    sw.Stop();
    enc.Dispose();
    var ogg = OpusOggEncoder.Encode(packets, new OpusOggEncoderOptions { OutputChannels = 1, InputSampleRateHz = 48000, PreSkip = 312 });
    string opusPath = Path.Combine(outDir, "ours.opus");
    File.WriteAllBytes(opusPath, ogg);
    string decPath = Path.Combine(outDir, "ours_opus_dec.pcm");
    RunFf($"-y -i \"{opusPath}\" -f s16le -ac 1 -ar 48000 \"{decPath}\"");
    var dec = File.ReadAllBytes(decPath);
    var (snr, maxDiff) = ComputeSnr16(src48mBytes, dec);
    PrintRow("Opus (ours)", ogg.Length, src48mBytes.Length, snr, maxDiff, sw.Elapsed.TotalMilliseconds);
}

// Vorbis ours (mono)
{
    var monoBytes = new byte[src44sBytes.Length / 2];
    for (int i = 0; i < monoBytes.Length / 2; i++)
    {
        short l = (short)(src44sBytes[i * 4] | (src44sBytes[i * 4 + 1] << 8));
        short r = (short)(src44sBytes[i * 4 + 2] | (src44sBytes[i * 4 + 3] << 8));
        short m = (short)((l + r) / 2);
        monoBytes[i * 2] = (byte)m;
        monoBytes[i * 2 + 1] = (byte)(m >> 8);
    }
    var pcm = new float[monoBytes.Length / 2];
    for (int i = 0; i < pcm.Length; i++) pcm[i] = ((short)(monoBytes[i * 2] | (monoBytes[i * 2 + 1] << 8))) / 32768f;

    var enc = new VorbisAudioEncoder(new VorbisAudioEncoderOptions { SampleRateHz = 44100, Channels = 1 });
    var sw = Stopwatch.StartNew();
    var ogg = enc.EncodeStream(pcm);
    sw.Stop();
    string oggPath = Path.Combine(outDir, "ours.ogg");
    File.WriteAllBytes(oggPath, ogg);
    string decPath = Path.Combine(outDir, "ours_vorbis_dec.pcm");
    RunFf($"-y -i \"{oggPath}\" -f s16le -ac 1 -ar 44100 \"{decPath}\"");
    var dec = File.ReadAllBytes(decPath);
    var (snr, maxDiff) = ComputeSnr16(monoBytes, dec);
    PrintRow("Vorbis (ours)", ogg.Length, monoBytes.Length, snr, maxDiff, sw.Elapsed.TotalMilliseconds);
}

Console.WriteLine();

// ffmpeg references
RunFf($"-y -f s16le -ar 44100 -ac 2 -i \"{src44s}\" -c:a flac \"{Path.Combine(outDir, "ff.flac")}\"");
EvalFfDec("FLAC (ffmpeg)", Path.Combine(outDir, "ff.flac"), src44sBytes, "-ac 2 -ar 44100");
RunFf($"-y -f s16le -ar 48000 -ac 1 -i \"{src48m}\" -c:a libopus -b:a 64k \"{Path.Combine(outDir, "ff.opus")}\"");
EvalFfDec("Opus (ffmpeg)", Path.Combine(outDir, "ff.opus"), src48mBytes, "-ac 1 -ar 48000");
RunFf($"-y -f s16le -ar 44100 -ac 1 -i \"{src44s}\" -ac 1 -c:a libvorbis \"{Path.Combine(outDir, "ff.ogg")}\"");
// For Vorbis ffmpeg, we used the stereo source downmixed by ffmpeg directly - reuse our mono extract instead.
{
    string srcMono = Path.Combine(outDir, "src_44k_mn.pcm");
    RunFf($"-y -i \"{source}\" -t {seconds} -f s16le -ac 1 -ar 44100 \"{srcMono}\"");
    var srcMonoBytes = File.ReadAllBytes(srcMono);
    string ffOgg = Path.Combine(outDir, "ff_mono.ogg");
    RunFf($"-y -f s16le -ar 44100 -ac 1 -i \"{srcMono}\" -c:a libvorbis \"{ffOgg}\"");
    EvalFfDec("Vorbis (ffmpeg)", ffOgg, srcMonoBytes, "-ac 1 -ar 44100");
}

void EvalFfDec(string name, string encPath, byte[] srcBytes, string decAr)
{
    string decPath = encPath + ".pcm";
    RunFf($"-y -i \"{encPath}\" {decAr} -f s16le \"{decPath}\"");
    var dec = File.ReadAllBytes(decPath);
    var (snr, maxDiff) = ComputeSnr16(srcBytes, dec);
    long sz = new FileInfo(encPath).Length;
    PrintRow(name, sz, srcBytes.Length, snr, maxDiff, 0);
}

void PrintRow(string codec, long size, long rawSize, double snr, int maxDiff, double encMs)
{
    double ratio = (double)size / rawSize;
    string snrStr = double.IsInfinity(snr) ? "inf" : snr.ToString("F2");
    Console.WriteLine($"{codec,-15}{size / 1024.0,-10:F1}{ratio,-10:F4}{snrStr,-12}{maxDiff,-12}{encMs,-10:F1}");
}

(double snr, int maxDiff) ComputeSnr16(byte[] src, byte[] dec)
{
    int compareLen = Math.Min(src.Length / 2, dec.Length / 2);
    if (compareLen < 100) return (0, 0);
    double sumSrcSq = 0;
    double sumErrSq = 0;
    int maxDiff = 0;
    for (int i = 0; i < compareLen; i++)
    {
        int s = (short)(src[i * 2] | (src[i * 2 + 1] << 8));
        int d = (short)(dec[i * 2] | (dec[i * 2 + 1] << 8));
        int e = s - d;
        if (Math.Abs(e) > maxDiff) maxDiff = Math.Abs(e);
        sumSrcSq += (double)s * s;
        sumErrSq += (double)e * e;
    }
    if (sumErrSq <= 0) return (double.PositiveInfinity, 0);
    return (10.0 * Math.Log10(sumSrcSq / sumErrSq), maxDiff);
}

void RunFf(string args)
{
    var p = Process.Start(new ProcessStartInfo(ffmpeg, args) { RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true })!;
    p.StandardError.ReadToEnd();
    p.WaitForExit();
    if (p.ExitCode != 0) throw new Exception($"ffmpeg failed: {args}");
}
