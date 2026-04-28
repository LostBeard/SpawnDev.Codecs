// Vorbis encoder demo: encode a 3-second piano chord (A4 + C5 + E5)
// to .ogg and verify ffmpeg accepts it. Mirrors the video animation
// harnesses on the audio side.
//
// Usage: dotnet run vorbis_encode_chord.cs

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using SpawnDev.Codecs.Audio.Vorbis;

const int SampleRate = 44100;
const int Seconds = 3;
const int Channels = 1;

// A4 = 440 Hz, C5 = 523.25 Hz, E5 = 659.25 Hz (A minor chord).
double[] frequencies = { 440.0, 523.25, 659.25 };

string outDir = Path.Combine(Path.GetTempPath(), "spawndev_vorbis_chord");
Directory.CreateDirectory(outDir);
string oggPath = Path.Combine(outDir, "chord_a_minor.ogg");
string wavPath = Path.Combine(outDir, "chord_a_minor_decoded.wav");

// === Synthesize ===
int n = SampleRate * Seconds;
var pcm = new float[n];
for (int i = 0; i < n; i++)
{
    double t = i / (double)SampleRate;
    double sample = 0;
    foreach (var f in frequencies) sample += Math.Sin(2.0 * Math.PI * f * t);
    // Apply a 50ms attack envelope to avoid pop.
    double envelope = Math.Min(1.0, i / (0.05 * SampleRate));
    pcm[i] = (float)(0.3 * envelope * sample / frequencies.Length);
}

// === Encode ===
var sw = Stopwatch.StartNew();
var encoder = new VorbisAudioEncoder(new VorbisAudioEncoderOptions
{
    SampleRateHz = SampleRate,
    Channels = Channels,
});
var oggBytes = encoder.EncodeStream(pcm);
sw.Stop();
File.WriteAllBytes(oggPath, oggBytes);

double encodeRate = (n / (double)SampleRate) / sw.Elapsed.TotalSeconds;
Console.WriteLine($"Vorbis encode of A-minor chord:");
Console.WriteLine($"  Source:    {Seconds}s @ {SampleRate}Hz mono = {n} samples");
Console.WriteLine($"  Encode:    {sw.Elapsed.TotalMilliseconds:F1}ms ({encodeRate:F1}x realtime)");
Console.WriteLine($"  Ogg size:  {oggBytes.Length:N0} bytes ({oggBytes.Length * 8.0 / Seconds / 1000:F1} kbps)");
Console.WriteLine($"  Out path:  {oggPath}");

// === Verify with ffmpeg + decode to WAV for VLC playback ===
string ffmpeg = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
if (!File.Exists(ffmpeg)) ffmpeg = "ffmpeg";

var p = Process.Start(new ProcessStartInfo(ffmpeg, $"-y -i \"{oggPath}\" -acodec pcm_s16le \"{wavPath}\"")
{
    RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true,
})!;
string err = p.StandardError.ReadToEnd();
p.WaitForExit();

if (p.ExitCode != 0)
{
    Console.WriteLine();
    Console.WriteLine("FAIL: ffmpeg failed to decode our Vorbis ogg.");
    Console.Error.WriteLine(err);
    Environment.Exit(1);
    return;
}

long wavSize = File.Exists(wavPath) ? new FileInfo(wavPath).Length : 0;
Console.WriteLine();
Console.WriteLine($"PASS: ffmpeg decoded our Vorbis ogg back to PCM");
Console.WriteLine($"  WAV path:  {wavPath} ({wavSize:N0} bytes)");
Console.WriteLine($"  Open the .ogg or .wav in VLC for audio playback.");
