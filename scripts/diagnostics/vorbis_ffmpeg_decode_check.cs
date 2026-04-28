// SpawnDev.Codecs Vorbis decoder ffmpeg cross-validation.

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using SpawnDev.Codecs.Audio.Vorbis;

const int SampleRate = 44100;
const int Channels = 1;
const int Seconds = 1;
const double Frequency = 440.0;

string ffmpegPath = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
string tempVorbis = Path.Combine(Path.GetTempPath(), "spawndev_vorbis_check.ogg");
string tempPcmRef = Path.Combine(Path.GetTempPath(), "spawndev_vorbis_ref.pcm");
string tempPcmIn = Path.Combine(Path.GetTempPath(), "spawndev_vorbis_input.pcm");

int total = SampleRate * Seconds;
var input = new short[total];
double a = 0.5 * 32767;
for (int n = 0; n < total; n++)
{
    double phase = 2.0 * Math.PI * Frequency * n / SampleRate;
    input[n] = (short)(Math.Sin(phase) * a);
}

var inputBytes = new byte[input.Length * 2];
Buffer.BlockCopy(input, 0, inputBytes, 0, inputBytes.Length);
File.WriteAllBytes(tempPcmIn, inputBytes);

RunFfmpeg(ffmpegPath, $"-y -f s16le -ar {SampleRate} -ac {Channels} -i \"{tempPcmIn}\" -c:a libvorbis -q:a 5 \"{tempVorbis}\"");
long fileSize = new FileInfo(tempVorbis).Length;
Console.WriteLine($"ffmpeg encoded {input.Length} samples as {fileSize}-byte .ogg vorbis file");

RunFfmpeg(ffmpegPath, $"-y -i \"{tempVorbis}\" -f s16le -ac {Channels} -ar {SampleRate} \"{tempPcmRef}\"");
var refBytes = File.ReadAllBytes(tempPcmRef);
int refSamples = refBytes.Length / 2;
Console.WriteLine($"ffmpeg reference decode: {refSamples} samples");

var oggBytes = File.ReadAllBytes(tempVorbis);
VorbisOggDecodeResult decoded;
try
{
    decoded = VorbisOggDecoder.Decode(oggBytes);
}
catch (Exception ex)
{
    Console.WriteLine($"VorbisOggDecoder failed: {ex.GetType().Name}: {ex.Message}");
    return;
}
int ourSamples = decoded.InterleavedSamples.Length / Channels;
Console.WriteLine($"SpawnDev.Codecs Vorbis decode: {ourSamples} samples per channel");

int compareLen = Math.Min(refSamples, ourSamples);
Console.WriteLine($"Comparing first {compareLen} samples...");

int exactMatches = 0;
int closeMatches = 0;
int maxDelta = 0;
for (int i = 0; i < compareLen; i++)
{
    short refSample = (short)(refBytes[i * 2] | (refBytes[i * 2 + 1] << 8));
    int ourInt16 = (int)Math.Clamp(decoded.InterleavedSamples[i] * 32768f, -32768f, 32767f);
    int delta = Math.Abs(refSample - ourInt16);
    if (delta == 0) exactMatches++;
    if (delta <= 16) closeMatches++;
    if (delta > maxDelta) maxDelta = delta;
}

Console.WriteLine($"Exact matches: {exactMatches}/{compareLen} ({100.0 * exactMatches / compareLen:F2}%)");
Console.WriteLine($"Within 16 LSB: {closeMatches}/{compareLen} ({100.0 * closeMatches / compareLen:F2}%)");
Console.WriteLine($"Max delta:    {maxDelta}");

if (closeMatches >= compareLen * 0.99)
{
    Console.WriteLine("");
    Console.WriteLine("=========================================");
    Console.WriteLine("SpawnDev.Codecs Vorbis decoder produces");
    Console.WriteLine("PCM that MATCHES ffmpeg's reference");
    Console.WriteLine("decode (>=99% within 16 LSB).");
    Console.WriteLine("=========================================");
}

File.Delete(tempPcmIn);
File.Delete(tempPcmRef);
File.Delete(tempVorbis);

static void RunFfmpeg(string path, string args)
{
    var psi = new ProcessStartInfo(path, args)
    {
        RedirectStandardError = true,
        UseShellExecute = false,
    };
    var p = Process.Start(psi)!;
    p.WaitForExit();
    if (p.ExitCode != 0)
        throw new Exception($"ffmpeg failed (exit {p.ExitCode}):\n{p.StandardError.ReadToEnd()}");
}
