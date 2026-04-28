// Vorbis amplitude regression test (our decoder side).
//
// Encodes a 1s 440Hz mono sine via ffmpeg/libvorbis, then decodes that .ogg
// with our decoder. Verifies that we follow the libvorbis MDCT normalisation
// convention (4/N on the encoder forward, unscaled inverse) so we produce
// full-amplitude output from third-party-encoded streams.
//
// Pass criteria (post 2026-04-27 fix):
//   ffmpeg->ffmpeg RMS ratio  ~= 1.00 (sanity check that ffmpeg pipeline works)
//   ffmpeg->ours   RMS ratio  ~= 1.00 (within 10%; lossy quantisation lives here)
//
// Pre-fix this script reported ratio = 0.002 (decoder was 600x too quiet).
//
// Usage: dotnet run vorbis_amplitude_diag2.cs

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
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

string outDir = Path.Combine(Path.GetTempPath(), "spawndev_vorbis_diag2");
Directory.CreateDirectory(outDir);
string srcPcm = Path.Combine(outDir, "src.pcm");
string ffEnc = Path.Combine(outDir, "ff_enc.ogg");
string ffEncFfDec = Path.Combine(outDir, "ff_enc_ff_dec.pcm");

// === Synthesize source at 0.4 amplitude ===
var input = new float[Total];
for (int n = 0; n < Total; n++)
    input[n] = (float)(Amp * Math.Sin(2 * Math.PI * Hz * n / SR));

var (srcPeak, srcRms) = MeasureF(input);
Console.WriteLine($"Source 440Hz @ {Amp}: peak={srcPeak:F4} rms={srcRms:F4}");

// Write source as float32 PCM (avoid quantization for the ffmpeg encode path)
var srcBytes = new byte[Total * 4];
Buffer.BlockCopy(input, 0, srcBytes, 0, srcBytes.Length);
File.WriteAllBytes(srcPcm, srcBytes);

// === Encode with ffmpeg/libvorbis ===
RunCmd(ffmpeg, $"-y -f f32le -ar {SR} -ac 1 -i \"{srcPcm}\" -c:a libvorbis -q:a 5 \"{ffEnc}\"");
long ffEncSize = new FileInfo(ffEnc).Length;
Console.WriteLine($"ffmpeg-encoded ogg: {ffEncSize} bytes");

// === Decode the ffmpeg-encoded ogg with ffmpeg ===
RunCmd(ffmpeg, $"-y -i \"{ffEnc}\" -f s16le -ac 1 -ar {SR} \"{ffEncFfDec}\"");
var ffPcm = File.ReadAllBytes(ffEncFfDec);
var ffSamples = new float[ffPcm.Length / 2];
for (int i = 0; i < ffSamples.Length; i++)
{
    short v = (short)(ffPcm[i * 2] | (ffPcm[i * 2 + 1] << 8));
    ffSamples[i] = v / 32768f;
}
var (ffPeak, ffRms) = MeasureF(ffSamples);
Console.WriteLine($"ffmpeg encode -> ffmpeg decode: peak={ffPeak:F4} rms={ffRms:F4}");

// === Decode the ffmpeg-encoded ogg with OUR decoder ===
var ffEncBytes = File.ReadAllBytes(ffEnc);
var ourDecoded = VorbisOggDecoder.Decode(ffEncBytes);
var (ourFFPeak, ourFFRms) = MeasureF(ourDecoded.InterleavedSamples);
Console.WriteLine($"ffmpeg encode -> OUR decode:    peak={ourFFPeak:F4} rms={ourFFRms:F4}");

Console.WriteLine();
Console.WriteLine($"Source RMS:                     {srcRms:F4}");
Console.WriteLine($"ffmpeg->ffmpeg RMS ratio:       {ffRms/srcRms:F3} (target ~1.0)");
Console.WriteLine($"ffmpeg->OUR RMS ratio:          {ourFFRms/srcRms:F3} (target ~1.0)");

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

static void RunCmd(string path, string args)
{
    var psi = new ProcessStartInfo(path, args)
    {
        RedirectStandardError = true,
        UseShellExecute = false,
        CreateNoWindow = true,
    };
    var p = Process.Start(psi)!;
    string err = p.StandardError.ReadToEnd();
    p.WaitForExit();
    if (p.ExitCode != 0) throw new Exception($"ffmpeg failed: {err.Substring(0, Math.Min(2000, err.Length))}");
}
