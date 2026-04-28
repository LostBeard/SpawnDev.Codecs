// Verify chord WAV amplitude after ffmpeg decode.
//
// Usage: dotnet run vorbis_chord_amp_check.cs

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.IO;

string wavPath = Path.Combine(Path.GetTempPath(), "spawndev_vorbis_chord", "chord_a_minor_decoded.wav");
if (!File.Exists(wavPath)) { Console.WriteLine($"WAV not found at {wavPath}; run vorbis_encode_chord.cs first."); return; }

var bytes = File.ReadAllBytes(wavPath);
// Skip 44-byte standard WAV header
int dataStart = 44;
int len = (bytes.Length - dataStart) / 2;
var samples = new float[len];
for (int i = 0; i < len; i++)
{
    short v = (short)(bytes[dataStart + i*2] | (bytes[dataStart + i*2 + 1] << 8));
    samples[i] = v / 32768f;
}

// Skip first / last 100ms (startup + tail transients from envelope and overlap-add).
const int Skip = 4410;
int analyseLen = Math.Max(0, len - 2 * Skip);
float peak = 0; double sumSq = 0;
for (int i = Skip; i < Skip + analyseLen; i++)
{
    float a = MathF.Abs(samples[i]);
    if (a > peak) peak = a;
    sumSq += samples[i] * (double)samples[i];
}
float rms = (float)Math.Sqrt(sumSq / analyseLen);
Console.WriteLine($"Decoded chord WAV: {len} samples (analysing middle {analyseLen} samples, dropping {Skip} from each end)");
Console.WriteLine($"Peak: {peak:F4}");
Console.WriteLine($"RMS:  {rms:F4}");

// Source: 0.3 * sin(...)/3, three sines, max overlap ~0.3 amplitude
// So peak should be around 0.3, RMS around 0.3/sqrt(2)/sqrt(3) ~= 0.12
Console.WriteLine();
Console.WriteLine("Expected for the chord (3 sines * 0.3 amp / 3 = 0.1 each):");
Console.WriteLine("  Peak should be ~0.3 (when all 3 sines align)");
Console.WriteLine("  RMS  should be ~0.12 (sum of independent sines RMS)");
// Allow 50% peak headroom for quantization error.
if (peak < 0.45f) Console.WriteLine("PASS: Peak is within 50% of source, no audible distortion");
else Console.WriteLine($"FAIL: Peak {peak:F3} > 0.45 (more than 50% over source)");
