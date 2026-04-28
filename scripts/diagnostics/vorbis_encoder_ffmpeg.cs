// Encode 1s 440Hz mono with our Vorbis encoder, then decode with ffmpeg.
// Verify ffmpeg accepts our stream and produces sensible output.

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using System.Numerics;
using SpawnDev.Codecs.Audio.Vorbis;

const int SR = 44100;
const double Hz = 440.0;
const int Total = SR * 1;
const int BS = 1024;
string ffmpegPath = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
string tempOgg = Path.Combine(Path.GetTempPath(), "spawndev_vorbis_outFFm.ogg");
string tempPcm = Path.Combine(Path.GetTempPath(), "spawndev_vorbis_outFFm.pcm");

var input = new float[Total];
for (int n = 0; n < Total; n++)
    input[n] = (float)(0.5 * Math.Sin(2 * Math.PI * Hz * n / SR));

var enc = new VorbisAudioEncoder(new VorbisAudioEncoderOptions { SampleRateHz = SR, Channels = 1, BlockSize = BS });
var ogg = enc.EncodeStream(input);
File.WriteAllBytes(tempOgg, ogg);
Console.WriteLine($"Wrote {ogg.Length} bytes to {tempOgg}");

// Try ffmpeg ffprobe first
var probe = RunCmd(ffmpegPath, $"-i \"{tempOgg}\" -f null -");
Console.WriteLine($"ffmpeg probe (stderr) exit code {probe.exitCode}");
Console.WriteLine(probe.stderr.Substring(0, Math.Min(1500, probe.stderr.Length)));

// Try a real decode
var decResult = RunCmd(ffmpegPath, $"-y -i \"{tempOgg}\" -f s16le -ac 1 -ar {SR} \"{tempPcm}\"");
if (decResult.exitCode != 0)
{
    Console.WriteLine($"ffmpeg DECODE FAILED (exit {decResult.exitCode})");
    Console.WriteLine(decResult.stderr.Substring(0, Math.Min(2000, decResult.stderr.Length)));
    return;
}
var pcmBytes = File.ReadAllBytes(tempPcm);
int samples = pcmBytes.Length / 2;
Console.WriteLine($"ffmpeg decoded {samples} int16 samples");

// Find dominant frequency in ffmpeg's PCM output
int searchN = Math.Min(samples, 8192);
var sub = new float[searchN];
for (int i = 0; i < searchN; i++)
{
    short v = (short)(pcmBytes[i * 2] | (pcmBytes[i * 2 + 1] << 8));
    sub[i] = v / 32768f;
}
double peakHz = 0; double peakMag = 0;
for (int hz = 50; hz <= 1000; hz++)
{
    Complex sum = Complex.Zero;
    for (int n = 0; n < searchN; n++)
    {
        double phase = -2 * Math.PI * hz * n / SR;
        sum += new Complex(sub[n] * Math.Cos(phase), sub[n] * Math.Sin(phase));
    }
    double mag = sum.Magnitude;
    if (mag > peakMag) { peakMag = mag; peakHz = hz; }
}
Console.WriteLine($"ffmpeg-decoded DFT peak: {peakHz:F0} Hz (expected {Hz:F0} Hz)");
if (Math.Abs(peakHz - Hz) <= 5)
{
    Console.WriteLine("=========================================");
    Console.WriteLine("FFMPEG DECODES OUR VORBIS STREAM AT 440 Hz");
    Console.WriteLine("=========================================");
}

static (int exitCode, string stdout, string stderr) RunCmd(string path, string args)
{
    var p = new Process { StartInfo = new ProcessStartInfo(path, args)
    {
        RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true
    }};
    p.Start();
    string sout = p.StandardOutput.ReadToEnd();
    string serr = p.StandardError.ReadToEnd();
    p.WaitForExit();
    return (p.ExitCode, sout, serr);
}
