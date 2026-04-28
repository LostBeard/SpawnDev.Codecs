// FlacEncoder + ffmpeg round-trip verification.
//
// Generates a sine wave, encodes it with our FlacEncoder, writes to
// a .flac file, decodes the file with ffmpeg, and compares ffmpeg's
// output PCM bytes against our input PCM. Bit-exact match proves
// our encoder produces spec-compliant FLAC bytes that external
// decoders accept.
//
// Run with:
//   cd D:/users/tj/Projects/SpawnDev.Codecs && dotnet run flac_ffmpeg_roundtrip.cs

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using SpawnDev.Codecs.Audio.Flac;

const int SampleRate = 44100;
const int BitsPerSample = 16;
const int Channels = 1;
const int Seconds = 1;
const double Frequency = 440.0;

int totalSamples = SampleRate * Seconds;
var input = new int[totalSamples];
int amp = (1 << (BitsPerSample - 1)) - 1;
double a = amp * 0.5;
for (int n = 0; n < totalSamples; n++)
{
    double phase = 2.0 * Math.PI * Frequency * n / SampleRate;
    input[n] = (int)(Math.Sin(phase) * a);
}

Console.WriteLine($"Generated {totalSamples} samples ({Seconds}s of {Frequency}Hz at {SampleRate}Hz {Channels}ch {BitsPerSample}-bit)");

// Encode with our FlacEncoder.
byte[] encoded = FlacEncoder.EncodeStream(input, SampleRate, Channels, BitsPerSample, blockSize: 4096);
Console.WriteLine($"Our FlacEncoder produced {encoded.Length} bytes");

// Verify our FlacDecoder round-trip first.
var ourDecoded = FlacDecoder.Decode(encoded);
if (ourDecoded.InterleavedSamples.Length != input.Length)
    throw new Exception($"Our decoder length mismatch: input {input.Length}, decoded {ourDecoded.InterleavedSamples.Length}");
for (int i = 0; i < input.Length; i++)
    if (ourDecoded.InterleavedSamples[i] != input[i])
        throw new Exception($"Our decoder mismatch at sample {i}: input {input[i]}, decoded {ourDecoded.InterleavedSamples[i]}");
Console.WriteLine($"OUR round-trip: BIT-EXACT ({input.Length} samples)");

// Write the FLAC bytes to a temp file.
string tempFlac = Path.Combine(Path.GetTempPath(), "spawndev_flac_test.flac");
string tempPcm = Path.Combine(Path.GetTempPath(), "spawndev_flac_test.pcm");
File.WriteAllBytes(tempFlac, encoded);
Console.WriteLine($"Wrote {tempFlac}");

// Run ffmpeg to decode our .flac to raw PCM.
string ffmpegPath = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
var psi = new ProcessStartInfo(ffmpegPath,
    $"-y -i \"{tempFlac}\" -f s16le -acodec pcm_s16le \"{tempPcm}\"")
{
    RedirectStandardError = true,
    UseShellExecute = false,
};
var proc = Process.Start(psi)!;
proc.WaitForExit();
if (proc.ExitCode != 0)
    throw new Exception($"ffmpeg failed with exit {proc.ExitCode}:\n{proc.StandardError.ReadToEnd()}");
Console.WriteLine($"ffmpeg decoded {tempFlac} -> {tempPcm}");

// Read ffmpeg's PCM output + compare.
byte[] ffmpegPcm = File.ReadAllBytes(tempPcm);
int ffmpegSamples = ffmpegPcm.Length / 2; // 16-bit mono
Console.WriteLine($"ffmpeg produced {ffmpegPcm.Length} bytes = {ffmpegSamples} samples");

if (ffmpegSamples != input.Length)
    throw new Exception($"ffmpeg sample count mismatch: input {input.Length}, ffmpeg {ffmpegSamples}");

int mismatches = 0;
int firstMismatchIdx = -1;
for (int i = 0; i < input.Length; i++)
{
    short ffmpegSample = (short)(ffmpegPcm[i * 2] | (ffmpegPcm[i * 2 + 1] << 8));
    if (ffmpegSample != (short)input[i])
    {
        if (firstMismatchIdx < 0) firstMismatchIdx = i;
        mismatches++;
    }
}

if (mismatches == 0)
{
    Console.WriteLine($"ffmpeg <-> our FlacEncoder: BIT-EXACT ({input.Length} samples)");
    Console.WriteLine("");
    Console.WriteLine("=========================================");
    Console.WriteLine("SpawnDev.Codecs FlacEncoder produces");
    Console.WriteLine("VALID, SPEC-COMPLIANT FLAC bytes that");
    Console.WriteLine("ffmpeg decodes to the EXACT input PCM.");
    Console.WriteLine("=========================================");
}
else
{
    throw new Exception(
        $"ffmpeg decoded {mismatches}/{input.Length} samples differently. " +
        $"First mismatch at sample {firstMismatchIdx}: " +
        $"input={input[firstMismatchIdx]}, ffmpeg={(short)(ffmpegPcm[firstMismatchIdx*2] | (ffmpegPcm[firstMismatchIdx*2+1] << 8))}");
}

File.Delete(tempFlac);
File.Delete(tempPcm);
