// Smoke test: encode 1s of 440Hz mono via SpawnDev.Codecs Vorbis encoder and
// decode via SpawnDev.Codecs Vorbis decoder. Verify round-trip basic
// statistics (samples count, dominant frequency via zero-crossing).

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.IO;
using SpawnDev.Codecs.Audio.Vorbis;

const int SampleRate = 44100;
const int Channels = 1;
const int Seconds = 1;
const double Frequency = 440.0;
const int BlockSize = 1024;

int total = SampleRate * Seconds;
var input = new float[total];
for (int n = 0; n < total; n++)
{
    double phase = 2.0 * Math.PI * Frequency * n / SampleRate;
    input[n] = (float)(0.5 * Math.Sin(phase));
}

var encoder = new VorbisAudioEncoder(new VorbisAudioEncoderOptions
{
    SampleRateHz = SampleRate,
    Channels = Channels,
    BlockSize = BlockSize,
});

byte[] oggBytes;
try
{
    oggBytes = encoder.EncodeStream(input);
    Console.WriteLine($"Encoded {input.Length} samples as {oggBytes.Length}-byte .ogg vorbis");
}
catch (Exception ex)
{
    Console.WriteLine($"Encode FAILED: {ex.GetType().Name}: {ex.Message}");
    Console.WriteLine(ex.StackTrace);
    return;
}

string tempOgg = Path.Combine(Path.GetTempPath(), "spawndev_vorbis_encsmoke.ogg");
File.WriteAllBytes(tempOgg, oggBytes);
Console.WriteLine($"Wrote test stream to {tempOgg}");

VorbisOggDecodeResult decoded;
try
{
    decoded = VorbisOggDecoder.Decode(oggBytes);
}
catch (Exception ex)
{
    Console.WriteLine($"Self-decode FAILED: {ex.GetType().Name}: {ex.Message}");
    Console.WriteLine(ex.StackTrace);
    return;
}
Console.WriteLine($"Self-decoded {decoded.InterleavedSamples.Length} samples (expected ~{total})");

if (decoded.InterleavedSamples.Length == 0)
{
    Console.WriteLine("FAIL: Self-decode produced 0 samples.");
    return;
}

float minOur = float.PositiveInfinity, maxOur = float.NegativeInfinity, sumAbsOur = 0f;
for (int i = 0; i < decoded.InterleavedSamples.Length; i++)
{
    float v = decoded.InterleavedSamples[i];
    if (v < minOur) minOur = v;
    if (v > maxOur) maxOur = v;
    sumAbsOur += Math.Abs(v);
}
Console.WriteLine($"Decoded range: [{minOur:F4}, {maxOur:F4}], mean|x|={sumAbsOur / decoded.InterleavedSamples.Length:F4}");

// DFT scan to find the actual dominant frequency in our decoded output.
int searchN = Math.Min(decoded.InterleavedSamples.Length, 8192);
double peakHz = 0; double peakMag = 0;
for (int hz = 50; hz <= 1000; hz++)
{
    System.Numerics.Complex sum = System.Numerics.Complex.Zero;
    for (int n = 0; n < searchN; n++)
    {
        double phase = -2 * Math.PI * hz * n / SampleRate;
        sum += new System.Numerics.Complex(decoded.InterleavedSamples[n] * Math.Cos(phase),
                                            decoded.InterleavedSamples[n] * Math.Sin(phase));
    }
    double mag = sum.Magnitude;
    if (mag > peakMag) { peakMag = mag; peakHz = hz; }
}
Console.WriteLine($"Decoded DFT peak: {peakHz:F0} Hz (expected {Frequency:F0} Hz)");

if (Math.Abs(peakHz - Frequency) <= 5.0)
{
    Console.WriteLine("");
    Console.WriteLine("=========================================");
    Console.WriteLine("VORBIS ENCODER ROUND-TRIPS WITH CORRECT");
    Console.WriteLine("DOMINANT FREQUENCY (within 5 Hz)");
    Console.WriteLine("=========================================");
}
else
{
    Console.WriteLine($"WARN: dominant freq off by {Math.Abs(peakHz - Frequency):F1} Hz");
}
