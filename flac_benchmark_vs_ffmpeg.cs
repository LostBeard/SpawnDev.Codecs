// SpawnDev.Codecs FlacEncoder benchmark vs ffmpeg.
//
// Encodes the same PCM data with both encoders and reports:
//   - Encode time (wall-clock ms)
//   - Compression ratio (output size / raw PCM size)
//   - Decode time round-trip
//
// Note: SpawnDev.Codecs.FlacEncoder currently uses VERBATIM subframes
// (lossless framing without prediction). Compression will be lower
// than ffmpeg's full LPC + Rice. Speed and bit-exactness are the
// primary metrics.

#:project SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using SpawnDev.Codecs.Audio.Flac;

const int SampleRate = 44100;
const int Channels = 2;
const int BitsPerSample = 16;
const int Seconds = 30;

string ffmpegPath = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
string tempPcm = Path.Combine(Path.GetTempPath(), "bench_input.pcm");
string tempOursFlac = Path.Combine(Path.GetTempPath(), "bench_ours.flac");
string tempFfmpegFlac = Path.Combine(Path.GetTempPath(), "bench_ffmpeg.flac");
string tempOursDecoded = Path.Combine(Path.GetTempPath(), "bench_ours_decoded.pcm");
string tempFfmpegDecoded = Path.Combine(Path.GetTempPath(), "bench_ffmpeg_decoded.pcm");

// Generate complex audio: 440Hz left + 880Hz right + light noise.
int totalSamples = SampleRate * Seconds * Channels;
var input = new int[totalSamples];
var rng = new Random(42);
double a = 0.4 * 32767;
for (int n = 0; n < SampleRate * Seconds; n++)
{
    double phase440 = 2.0 * Math.PI * 440 * n / SampleRate;
    double phase880 = 2.0 * Math.PI * 880 * n / SampleRate;
    int noise = rng.Next(-50, 50);
    input[n * 2 + 0] = (int)(Math.Sin(phase440) * a) + noise;
    input[n * 2 + 1] = (int)(Math.Sin(phase880) * a) + noise;
}
int rawPcmSize = totalSamples * 2;
Console.WriteLine($"Input: {Seconds}s of {Channels}-channel {SampleRate}Hz {BitsPerSample}-bit = {rawPcmSize} bytes raw PCM");
Console.WriteLine();

// Write PCM for ffmpeg.
var inputBytes = new byte[rawPcmSize];
for (int i = 0; i < totalSamples; i++)
{
    short s = (short)Math.Clamp(input[i], short.MinValue, short.MaxValue);
    inputBytes[i * 2] = (byte)(s & 0xff);
    inputBytes[i * 2 + 1] = (byte)((s >> 8) & 0xff);
}
File.WriteAllBytes(tempPcm, inputBytes);

// === Encode benchmarks ===

// SpawnDev FlacEncoder.
var sw = Stopwatch.StartNew();
byte[] oursEncoded = FlacEncoder.EncodeStream(input, SampleRate, Channels, BitsPerSample, blockSize: 4096);
sw.Stop();
File.WriteAllBytes(tempOursFlac, oursEncoded);
double oursEncodeMs = sw.Elapsed.TotalMilliseconds;
double oursRatio = (double)oursEncoded.Length / rawPcmSize;
Console.WriteLine($"SpawnDev FlacEncoder (VERBATIM): {oursEncodeMs:F1} ms, {oursEncoded.Length} bytes, ratio {oursRatio:F3}");

// ffmpeg FLAC encoder (full LPC).
sw = Stopwatch.StartNew();
RunFfmpeg(ffmpegPath, $"-y -f s16le -ar {SampleRate} -ac {Channels} -i \"{tempPcm}\" -c:a flac \"{tempFfmpegFlac}\"");
sw.Stop();
long ffmpegSize = new FileInfo(tempFfmpegFlac).Length;
double ffmpegEncodeMs = sw.Elapsed.TotalMilliseconds;
double ffmpegRatio = (double)ffmpegSize / rawPcmSize;
Console.WriteLine($"ffmpeg FLAC (default LPC):       {ffmpegEncodeMs:F1} ms, {ffmpegSize} bytes, ratio {ffmpegRatio:F3}");
Console.WriteLine();

// === Decode benchmarks ===

// SpawnDev FlacDecoder reading our encoded file.
sw = Stopwatch.StartNew();
var oursDecoded = FlacDecoder.Decode(oursEncoded);
sw.Stop();
double oursDecodeMs = sw.Elapsed.TotalMilliseconds;
Console.WriteLine($"SpawnDev FlacDecoder (our file):    {oursDecodeMs:F1} ms");

// SpawnDev FlacDecoder reading ffmpeg's encoded file.
var ffmpegFlacBytes = File.ReadAllBytes(tempFfmpegFlac);
sw = Stopwatch.StartNew();
try
{
    var crossDecoded = FlacDecoder.Decode(ffmpegFlacBytes);
    sw.Stop();
    Console.WriteLine($"SpawnDev FlacDecoder (ffmpeg's):    {sw.Elapsed.TotalMilliseconds:F1} ms");
    int matches = 0;
    int compareLen = Math.Min(crossDecoded.InterleavedSamples.Length, input.Length);
    for (int i = 0; i < compareLen; i++)
    {
        if (crossDecoded.InterleavedSamples[i] == input[i]) matches++;
    }
    Console.WriteLine($"  Bit-exact match vs original: {matches}/{compareLen} samples");
}
catch (Exception ex)
{
    sw.Stop();
    Console.WriteLine($"SpawnDev FlacDecoder (ffmpeg's):    FAILED: {ex.GetType().Name}: {ex.Message}");
    Console.WriteLine($"  (ffmpeg's FLAC uses LPC subframes which our decoder may not yet handle)");
}

// ffmpeg decoding our FLAC.
sw = Stopwatch.StartNew();
RunFfmpeg(ffmpegPath, $"-y -i \"{tempOursFlac}\" -f s16le \"{tempOursDecoded}\"");
sw.Stop();
double ffmpegDecodeOursMs = sw.Elapsed.TotalMilliseconds;
Console.WriteLine($"ffmpeg decoding our FLAC:           {ffmpegDecodeOursMs:F1} ms");

// ffmpeg decoding its own FLAC.
sw = Stopwatch.StartNew();
RunFfmpeg(ffmpegPath, $"-y -i \"{tempFfmpegFlac}\" -f s16le \"{tempFfmpegDecoded}\"");
sw.Stop();
Console.WriteLine($"ffmpeg decoding ffmpeg's FLAC:      {sw.Elapsed.TotalMilliseconds:F1} ms");

// === Bit-exactness check on round-trip ===

var oursRtPcm = File.ReadAllBytes(tempOursDecoded);
int rtMatches = 0;
int rtCompareLen = Math.Min(oursRtPcm.Length, inputBytes.Length);
for (int i = 0; i < rtCompareLen; i++)
{
    if (oursRtPcm[i] == inputBytes[i]) rtMatches++;
}
Console.WriteLine();
Console.WriteLine($"Round-trip (us encode -> ffmpeg decode): {rtMatches}/{rtCompareLen} bytes BIT-EXACT");
if (rtMatches == rtCompareLen)
{
    Console.WriteLine("  -> SpawnDev FlacEncoder produces SPEC-COMPLIANT FLAC.");
}

Console.WriteLine();
Console.WriteLine("=========================================================");
Console.WriteLine($"Encode speed:  ours = {rawPcmSize / oursEncodeMs / 1000:F1} MB/s; ffmpeg = {rawPcmSize / ffmpegEncodeMs / 1000:F1} MB/s");
Console.WriteLine($"Compression:   ours = {oursRatio:F3} (VERBATIM); ffmpeg = {ffmpegRatio:F3} (full LPC)");
Console.WriteLine("=========================================================");

File.Delete(tempPcm);
File.Delete(tempOursFlac);
File.Delete(tempFfmpegFlac);
File.Delete(tempOursDecoded);
File.Delete(tempFfmpegDecoded);

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
