// Check what subframe types the FLAC encoder picks for various inputs.
// Smoke-decodes the encoded bytes via FlacDecoder + reports which
// subframe types appear.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using SpawnDev.Codecs.Audio.Flac;

string ffmpegPath = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
string tempPcm = Path.Combine(Path.GetTempPath(), "flac_check.pcm");
string tempFlac = Path.Combine(Path.GetTempPath(), "flac_check.flac");

const int SampleRate = 44100;
const int Channels = 1;
const int Seconds = 5;

void TestInput(string label, int[] input)
{
    Console.WriteLine();
    Console.WriteLine($"--- {label} ---");
    var sw = Stopwatch.StartNew();
    byte[] encoded = FlacEncoder.EncodeStream(input, SampleRate, Channels, 16, blockSize: 4096);
    sw.Stop();
    int rawBytes = input.Length * 2;
    double ratio = (double)encoded.Length / rawBytes;
    Console.WriteLine($"  Raw: {rawBytes} bytes; Encoded: {encoded.Length} bytes; Ratio: {ratio:F3}");
    Console.WriteLine($"  Encode time: {sw.Elapsed.TotalMilliseconds:F1} ms");

    // ffmpeg encode for comparison.
    var pcmBytes = new byte[input.Length * 2];
    for (int i = 0; i < input.Length; i++)
    {
        short s = (short)Math.Clamp(input[i], short.MinValue, short.MaxValue);
        pcmBytes[i * 2] = (byte)(s & 0xff);
        pcmBytes[i * 2 + 1] = (byte)((s >> 8) & 0xff);
    }
    File.WriteAllBytes(tempPcm, pcmBytes);
    var sw2 = Stopwatch.StartNew();
    RunFfmpeg(ffmpegPath, $"-y -f s16le -ar {SampleRate} -ac {Channels} -i \"{tempPcm}\" -c:a flac \"{tempFlac}\"");
    sw2.Stop();
    long ffSize = new FileInfo(tempFlac).Length;
    double ffRatio = (double)ffSize / rawBytes;
    Console.WriteLine($"  ffmpeg: {ffSize} bytes; Ratio: {ffRatio:F3} ({sw2.Elapsed.TotalMilliseconds:F1} ms)");

    // Verify our encoder's bytes round-trip via FlacDecoder.
    var dec = FlacDecoder.Decode(encoded);
    int matches = 0;
    int compareLen = Math.Min(dec.InterleavedSamples.Length, input.Length);
    for (int i = 0; i < compareLen; i++)
        if (dec.InterleavedSamples[i] == input[i]) matches++;
    Console.WriteLine($"  Decode round-trip: {matches}/{compareLen} BIT-EXACT");
}

// 1. Pure sine - very compressible (lots of structure)
{
    var sine = new int[SampleRate * Seconds];
    for (int n = 0; n < sine.Length; n++)
        sine[n] = (int)(Math.Sin(2 * Math.PI * 440 * n / SampleRate) * 0.5 * 32767);
    TestInput("Pure sine 440Hz", sine);
}

// 2. White noise - incompressible
{
    var rng = new Random(42);
    var noise = new int[SampleRate * Seconds];
    for (int n = 0; n < noise.Length; n++)
        noise[n] = (short)rng.Next(-32768, 32767);
    TestInput("White noise (incompressible)", noise);
}

// 3. Constant zero - maximum compression
{
    var zero = new int[SampleRate * Seconds];
    TestInput("Constant zero (CONSTANT subframe)", zero);
}

// 4. DC offset - constant non-zero
{
    var dc = new int[SampleRate * Seconds];
    Array.Fill(dc, 1234);
    TestInput("Constant 1234 (CONSTANT subframe)", dc);
}

// 5. Linear ramp - FIXED order 1 should compress well. Sawtooth in
//    [-32767, 32767] so all values fit in signed 16-bit.
{
    var ramp = new int[SampleRate * Seconds];
    for (int n = 0; n < ramp.Length; n++)
    {
        int wrapped = (n * 7) % 65535 - 32767;
        ramp[n] = wrapped;
    }
    TestInput("Linear ramp (signed 16-bit sawtooth)", ramp);
}

File.Delete(tempPcm);
File.Delete(tempFlac);

static void RunFfmpeg(string path, string args)
{
    var psi = new ProcessStartInfo(path, args) { RedirectStandardError = true, UseShellExecute = false };
    var p = Process.Start(psi)!;
    p.WaitForExit();
    if (p.ExitCode != 0)
        throw new Exception($"ffmpeg failed (exit {p.ExitCode}):\n{p.StandardError.ReadToEnd()}");
}
