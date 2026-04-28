// Vorbis amplitude regression test (our encoder side).
//
// Encodes a 1s 440Hz mono sine (amplitude 0.4) with VorbisAudioEncoder, then
// decodes the resulting .ogg with BOTH our decoder and ffmpeg, and verifies
// peak + RMS amplitude match the source. Catches the historical "ffmpeg
// decode is deafening" bug (encoder MDCT not normalised to libvorbis 4/N
// convention) and the "noise-gated bins quantise to ±half-step" bug
// (codebook entry N/2 not anchored at exactly 0).
//
// Pass criteria (post 2026-04-27 fix):
//   ffmpeg/source RMS ratio  ~= 1.00 (within 5%)
//   our/source RMS ratio     ~= 1.00 (within 5%)
//   ffmpeg/our   ratio       ~= 1.00 (encoder convention agreement)
//
// Usage: dotnet run vorbis_amplitude_diag.cs

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using SpawnDev.Codecs.Audio.Vorbis;

const int SR = 44100;
const int Seconds = 1;
const double Hz = 440.0;
const float Amp = 0.4f;
const int Total = SR * Seconds;

string ffmpeg = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
if (!File.Exists(ffmpeg)) ffmpeg = "ffmpeg";

string outDir = Path.Combine(Path.GetTempPath(), "spawndev_vorbis_diag");
Directory.CreateDirectory(outDir);
string oggPath = Path.Combine(outDir, "diag.ogg");
string ffmpegPcmPath = Path.Combine(outDir, "diag_ffmpeg.pcm");

var input = new float[Total];
for (int n = 0; n < Total; n++)
    input[n] = (float)(Amp * Math.Sin(2 * Math.PI * Hz * n / SR));

var (srcPeak, srcRms) = MeasureF(input);
Console.WriteLine($"Source 440Hz @ {Amp}: peak={srcPeak:F4} rms={srcRms:F4}");

// === Encode with our encoder ===
var enc = new VorbisAudioEncoder(new VorbisAudioEncoderOptions
{
    SampleRateHz = SR,
    Channels = 1,
});
var ogg = enc.EncodeStream(input);
File.WriteAllBytes(oggPath, ogg);
Console.WriteLine($"Encoded {ogg.Length} bytes -> {oggPath}");

// === Decode with our decoder ===
var ourDecoded = VorbisOggDecoder.Decode(ogg);
var ours = ourDecoded.InterleavedSamples;
var (ourPeak, ourRms) = MeasureF(ours);
Console.WriteLine($"Our decoder:    peak={ourPeak:F4} rms={ourRms:F4} samples={ours.Length}");

// === Decode with ffmpeg ===
var psi = new ProcessStartInfo(ffmpeg, $"-y -i \"{oggPath}\" -f s16le -ac 1 -ar {SR} \"{ffmpegPcmPath}\"")
{
    RedirectStandardError = true,
    UseShellExecute = false,
    CreateNoWindow = true,
};
var p = Process.Start(psi)!;
string err = p.StandardError.ReadToEnd();
p.WaitForExit();
if (p.ExitCode != 0)
{
    Console.WriteLine($"ffmpeg failed: {err.Substring(0, Math.Min(2000, err.Length))}");
    return;
}
var pcmBytes = File.ReadAllBytes(ffmpegPcmPath);
var ffm = new float[pcmBytes.Length / 2];
for (int i = 0; i < ffm.Length; i++)
{
    short v = (short)(pcmBytes[i * 2] | (pcmBytes[i * 2 + 1] << 8));
    ffm[i] = v / 32768f;
}
var (ffmPeak, ffmRms) = MeasureF(ffm);
Console.WriteLine($"ffmpeg decoder: peak={ffmPeak:F4} rms={ffmRms:F4} samples={ffm.Length}");

Console.WriteLine();
Console.WriteLine($"Peak ratio (ffmpeg/source): {ffmPeak/srcPeak:F3} (target ~1.0)");
Console.WriteLine($"Peak ratio (our/source):    {ourPeak/srcPeak:F3} (target ~1.0)");
Console.WriteLine($"RMS ratio (ffmpeg/source):  {ffmRms/srcRms:F3} (target ~1.0)");
Console.WriteLine($"RMS ratio (our/source):     {ourRms/srcRms:F3} (target ~1.0)");
Console.WriteLine();
Console.WriteLine($"Peak ratio ffmpeg/our:      {ffmPeak/ourPeak:F3}  -- bug indicator");
Console.WriteLine($"RMS ratio ffmpeg/our:       {ffmRms/ourRms:F3}");

static (float peak, float rms) MeasureF(ReadOnlySpan<float> v)
{
    float peak = 0; double sumSq = 0;
    for (int i = 0; i < v.Length; i++)
    {
        float a = Math.Abs(v[i]);
        if (a > peak) peak = a;
        sumSq += v[i] * (double)v[i];
    }
    float rms = (float)Math.Sqrt(sumSq / Math.Max(1, v.Length));
    return (peak, rms);
}
