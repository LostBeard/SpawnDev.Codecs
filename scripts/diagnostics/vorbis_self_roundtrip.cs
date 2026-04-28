// Vorbis encoder + our own decoder round-trip on real BBB audio.
// Verifies the (corrected post-c67d8ec) encoder output is also accepted
// by our VorbisOggDecoder, not just by ffmpeg's libvorbis.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using SpawnDev.Codecs.Audio.Vorbis;

string source = "V:\\Video\\Big Buck Bunny - FULL HD 60FPS.mp4";
string ffmpeg = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
string outDir = Path.Combine(Path.GetTempPath(), "vorbis_self_rt");
Directory.CreateDirectory(outDir);
if (!File.Exists(source)) { Console.WriteLine($"Missing {source}"); Environment.Exit(1); }

string srcMonoPcm = Path.Combine(outDir, "src_mono.pcm");
RunFf($"-y -i \"{source}\" -t 5 -f s16le -ac 1 -ar 44100 \"{srcMonoPcm}\"");
var srcBytes = File.ReadAllBytes(srcMonoPcm);
var pcm = new float[srcBytes.Length / 2];
for (int i = 0; i < pcm.Length; i++) pcm[i] = ((short)(srcBytes[i * 2] | (srcBytes[i * 2 + 1] << 8))) / 32768f;
Console.WriteLine($"Source: {pcm.Length} samples (5s mono 44.1k)");

// Encode with our encoder.
var enc = new VorbisAudioEncoder(new VorbisAudioEncoderOptions { SampleRateHz = 44100, Channels = 1 });
var sw = Stopwatch.StartNew();
var ogg = enc.EncodeStream(pcm);
sw.Stop();
Console.WriteLine($"Encoded: {ogg.Length}B in {sw.Elapsed.TotalMilliseconds:F1}ms");

// Decode through our decoder.
sw.Restart();
var ourDec = VorbisOggDecoder.Decode(ogg);
sw.Stop();
Console.WriteLine($"Our decoder: {ourDec.InterleavedSamples.Length} samples in {sw.Elapsed.TotalMilliseconds:F1}ms");

// Decode through ffmpeg for comparison.
string oggPath = Path.Combine(outDir, "ours.ogg");
File.WriteAllBytes(oggPath, ogg);
string ffPcm = Path.Combine(outDir, "ff.pcm");
RunFf($"-y -i \"{oggPath}\" -f s16le -ac 1 -ar 44100 \"{ffPcm}\"");
var ffBytes = File.ReadAllBytes(ffPcm);
Console.WriteLine($"ffmpeg decoded: {ffBytes.Length / 2} samples");

// Compare power vs source.
double oursRms = ComputeRmsFloat(ourDec.InterleavedSamples);
double ffRms = ComputeRmsBytes(ffBytes);
double srcRms = 0;
foreach (var s in pcm) srcRms += s * (double)s;
srcRms = Math.Sqrt(srcRms / pcm.Length);

Console.WriteLine();
Console.WriteLine($"Source       RMS = {srcRms:F4}");
Console.WriteLine($"Our decoder  RMS = {oursRms:F4} (ratio {oursRms / srcRms:F2})");
Console.WriteLine($"ffmpeg decoder RMS = {ffRms:F4} (ratio {ffRms / srcRms:F2})");

double ComputeRmsFloat(float[] samples)
{
    double sumSq = 0;
    for (int i = 0; i < samples.Length; i++)
    {
        double s = samples[i];
        sumSq += s * s;
    }
    return Math.Sqrt(sumSq / samples.Length);
}

double ComputeRmsBytes(byte[] bytes)
{
    int count = bytes.Length / 2;
    double sumSq = 0;
    for (int i = 0; i < count; i++)
    {
        short v = (short)(bytes[i * 2] | (bytes[i * 2 + 1] << 8));
        double s = v / 32768.0;
        sumSq += s * s;
    }
    return Math.Sqrt(sumSq / count);
}

void RunFf(string args)
{
    var p = Process.Start(new ProcessStartInfo(ffmpeg, args) { RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true })!;
    p.StandardError.ReadToEnd();
    p.WaitForExit();
    if (p.ExitCode != 0) throw new Exception($"ffmpeg failed: {args}");
}
