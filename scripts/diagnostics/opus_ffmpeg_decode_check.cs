// SpawnDev.Codecs Opus decoder ffmpeg cross-validation.

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using SpawnDev.Codecs.Audio.Opus;

const int SampleRate = 48000;
const int Channels = 1;
const int Seconds = 1;
const double Frequency = 440.0;

string ffmpegPath = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
string tempOpus = Path.Combine(Path.GetTempPath(), "spawndev_opus_check.opus");
string tempPcmRef = Path.Combine(Path.GetTempPath(), "spawndev_opus_ref.pcm");
string tempPcmIn = Path.Combine(Path.GetTempPath(), "spawndev_opus_input.pcm");

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

RunFfmpeg(ffmpegPath, $"-y -f s16le -ar {SampleRate} -ac {Channels} -i \"{tempPcmIn}\" -c:a libopus -b:a 64k \"{tempOpus}\"");
long opusSize = new FileInfo(tempOpus).Length;
Console.WriteLine($"ffmpeg encoded {input.Length} samples as {opusSize}-byte .opus file");

RunFfmpeg(ffmpegPath, $"-y -i \"{tempOpus}\" -f s16le -ac {Channels} -ar {SampleRate} \"{tempPcmRef}\"");
var refBytes = File.ReadAllBytes(tempPcmRef);
int refSamples = refBytes.Length / 2;
Console.WriteLine($"ffmpeg reference decode: {refSamples} samples");

var opusBytes = File.ReadAllBytes(tempOpus);
OpusOggDecodeResult ourDecoded;
try
{
    ourDecoded = await OpusOggDecoder.DecodeAsync(opusBytes);
}
catch (Exception ex)
{
    Console.WriteLine($"OpusOggDecoder failed: {ex.GetType().Name}: {ex.Message}");
    Console.WriteLine("Opus decoder is not yet fully working for ffmpeg-encoded streams.");
    Console.WriteLine("Codec status: SILK is implemented; CELT integration is pending.");
    return;
}
int ourSamples = ourDecoded.InterleavedSamples48kHz.Length / Channels;
Console.WriteLine($"SpawnDev.Codecs Opus decode: {ourSamples} samples per channel");

int compareLen = Math.Min(refSamples, ourSamples);
Console.WriteLine($"Comparing first {compareLen} samples...");

int exactMatches = 0;
int closeMatches = 0;
int maxDelta = 0;
for (int i = 0; i < compareLen; i++)
{
    short refSample = (short)(refBytes[i * 2] | (refBytes[i * 2 + 1] << 8));
    int ourInt16 = (int)Math.Clamp(ourDecoded.InterleavedSamples48kHz[i] * 32768f, -32768f, 32767f);
    int delta = Math.Abs(refSample - ourInt16);
    if (delta == 0) exactMatches++;
    if (delta <= 16) closeMatches++;
    if (delta > maxDelta) maxDelta = delta;
}

Console.WriteLine($"Exact matches: {exactMatches}/{compareLen} ({100.0 * exactMatches / compareLen:F2}%)");
Console.WriteLine($"Within 16 LSB: {closeMatches}/{compareLen} ({100.0 * closeMatches / compareLen:F2}%)");
Console.WriteLine($"Max delta:    {maxDelta} (Opus is lossy; small drift expected at int16 quantization)");

File.Delete(tempPcmIn);
File.Delete(tempPcmRef);
File.Delete(tempOpus);

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
