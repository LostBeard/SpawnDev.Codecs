// FLAC encoder demo: encode a 3-second piano chord to .flac, verify
// ffmpeg accepts it, AND decode our own output and verify bit-exact
// round-trip (FLAC is lossless).
//
// Usage: dotnet run flac_encode_chord.cs

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using SpawnDev.Codecs.Audio.Flac;

const int SampleRate = 44100;
const int Seconds = 3;
const int Channels = 1;
const int Bits = 16;

double[] frequencies = { 440.0, 523.25, 659.25 };

string outDir = Path.Combine(Path.GetTempPath(), "spawndev_flac_chord");
Directory.CreateDirectory(outDir);
string flacPath = Path.Combine(outDir, "chord_a_minor.flac");
string wavPath = Path.Combine(outDir, "chord_a_minor_via_ffmpeg.wav");

int n = SampleRate * Seconds;
var samples = new int[n];
for (int i = 0; i < n; i++)
{
    double t = i / (double)SampleRate;
    double sample = 0;
    foreach (var f in frequencies) sample += Math.Sin(2.0 * Math.PI * f * t);
    double envelope = Math.Min(1.0, i / (0.05 * SampleRate));
    samples[i] = (int)(envelope * sample / frequencies.Length * 16000);
}

var sw = Stopwatch.StartNew();
FlacEncoder.EncodeToFile(flacPath, samples, SampleRate, Channels, Bits);
sw.Stop();

long flacSize = new FileInfo(flacPath).Length;
double encodeRate = Seconds / sw.Elapsed.TotalSeconds;
Console.WriteLine($"FLAC encode of A-minor chord:");
Console.WriteLine($"  Source:    {Seconds}s @ {SampleRate}Hz mono 16-bit = {n * 2} bytes raw");
Console.WriteLine($"  Encode:    {sw.Elapsed.TotalMilliseconds:F1}ms ({encodeRate:F1}x realtime)");
Console.WriteLine($"  FLAC size: {flacSize:N0} bytes (compression ratio {(n * 2.0 / flacSize):F2}x)");
Console.WriteLine($"  Out path:  {flacPath}");

// === Verify with our own decoder (bit-exact round-trip) ===
sw.Restart();
var decoded = FlacDecoder.DecodeFile(flacPath);
sw.Stop();

bool bitExact = decoded.InterleavedSamples.Length == n;
if (bitExact)
{
    for (int i = 0; i < n; i++)
        if (decoded.InterleavedSamples[i] != samples[i]) { bitExact = false; break; }
}
Console.WriteLine();
Console.WriteLine($"FLAC self round-trip:");
Console.WriteLine($"  Decode:    {sw.Elapsed.TotalMilliseconds:F1}ms");
Console.WriteLine($"  Bit-exact: {(bitExact ? "PASS" : "FAIL")}");

// === ffmpeg sanity (should be lossless too) ===
string ffmpeg = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
if (!File.Exists(ffmpeg)) ffmpeg = "ffmpeg";

var p = Process.Start(new ProcessStartInfo(ffmpeg, $"-y -i \"{flacPath}\" -acodec pcm_s16le \"{wavPath}\"")
{
    RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true,
})!;
string err = p.StandardError.ReadToEnd();
p.WaitForExit();

if (p.ExitCode != 0)
{
    Console.WriteLine();
    Console.WriteLine("FAIL: ffmpeg failed to decode our FLAC.");
    Console.Error.WriteLine(err);
    Environment.Exit(1);
    return;
}

long wavSize = File.Exists(wavPath) ? new FileInfo(wavPath).Length : 0;
Console.WriteLine();
Console.WriteLine($"PASS: ffmpeg decoded our FLAC to PCM");
Console.WriteLine($"  WAV path:  {wavPath} ({wavSize:N0} bytes)");
Console.WriteLine($"  Open the .flac or .wav in VLC for audio playback.");
