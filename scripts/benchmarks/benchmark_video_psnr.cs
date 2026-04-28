// Video quality benchmark: encode each frame from BBB through our VP8/VP9
// encoders + ffmpeg references, decode the output back to YUV via ffmpeg,
// and report PSNR (Y-plane) vs the source for each codec at multiple
// quantizer levels.
//
// Usage: dotnet run benchmark_video_psnr.cs [frameCount=5] [W=320] [H=240]

#:project ../../SpawnDev.Codecs/SpawnDev.Codecs.csproj
using System;
using System.Diagnostics;
using System.IO;
using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.Codecs.Video.Vp9;

int frameCount = args.Length >= 1 && int.TryParse(args[0], out int n) ? n : 5;
int W = args.Length >= 2 && int.TryParse(args[1], out int w) ? w : 320;
int H = args.Length >= 3 && int.TryParse(args[2], out int h) ? h : 240;
string source = "V:\\Video\\Big Buck Bunny - FULL HD 60FPS.mp4";
string ffmpeg = "C:\\Users\\TJ\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.1-full_build\\bin\\ffmpeg.exe";
string outDir = Path.Combine(Path.GetTempPath(), "spawndev_psnr");
Directory.CreateDirectory(outDir);

if (!File.Exists(source)) { Console.WriteLine($"Source not found: {source}"); Environment.Exit(1); }

Console.WriteLine($"Extracting {frameCount} frames at {W}x{H} from BBB...");
string srcYuv = Path.Combine(outDir, "src.yuv");
RunFf($"-y -i \"{source}\" -vf scale={W}:{H},crop={W}:{H}:0:0 -frames:v {frameCount} -f rawvideo -pix_fmt yuv420p \"{srcYuv}\"");
int frameSize = W * H + 2 * (W / 2) * (H / 2);
var src = File.ReadAllBytes(srcYuv);
Console.WriteLine($"  source: {src.Length} bytes ({frameCount} frames x {frameSize})");
Console.WriteLine();

Console.WriteLine($"Quality vs source ({frameCount}-frame avg, Y-plane PSNR):");
Console.WriteLine($"{"Codec",-15}{"Q",-5}{"Y PSNR dB",-12}{"avg KB/fr",-12}{"enc fps",-10}");
Console.WriteLine($"{new string('-', 15)}{new string('-', 5)}{new string('-', 12)}{new string('-', 12)}{new string('-', 10)}");

void Eval(string name, int q, Func<byte[], byte[], byte[], byte[]> encode, string ourFourcc, string decoder, string container)
{
    long totalBytes = 0;
    var ourIvf = Path.Combine(outDir, $"{name}_q{q}.ivf");
    var sw = Stopwatch.StartNew();
    using (var fs = File.Create(ourIvf))
    {
        var w = new IvfWriter(fs, ourFourcc, W, H, frameRate: 30, timeScale: 1, numFrames: 0, leaveOpen: true);
        for (int f = 0; f < frameCount; f++)
        {
            var (y, u, v) = SliceFrame(src, f);
            var bytes = encode(y, u, v);
            totalBytes += bytes.Length;
            w.WriteFrame(bytes, f);
        }
        w.Finish();
    }
    sw.Stop();
    double encFps = frameCount / sw.Elapsed.TotalSeconds;
    double avgKB = totalBytes / 1024.0 / frameCount;

    var decYuv = Path.Combine(outDir, $"{name}_q{q}.yuv");
    bool decOk = TryRunFf($"-y {decoder} -i \"{ourIvf}\" -f rawvideo -pix_fmt yuv420p \"{decYuv}\"");
    if (!decOk)
    {
        Console.WriteLine($"{name,-15}{q,-5}{"DECODE FAIL",-12}{avgKB,-12:F2}{encFps,-10:F2}");
        return;
    }
    var dec = File.ReadAllBytes(decYuv);
    if (dec.Length < frameCount * frameSize)
    {
        Console.WriteLine($"{name,-15}{q,-5}{"SHORT YUV",-12}{avgKB,-12:F2}{encFps,-10:F2}");
        return;
    }
    double psnr = ComputeYPsnr(src, dec, frameCount);
    Console.WriteLine($"{name,-15}{q,-5}{psnr,-12:F2}{avgKB,-12:F2}{encFps,-10:F2}");
}

void EvalFf(string name, int crf, string codec, string fourcc, string decArg)
{
    var ivf = Path.Combine(outDir, $"{name}_crf{crf}.ivf");
    var sw = Stopwatch.StartNew();
    RunFf($"-y -f rawvideo -pix_fmt yuv420p -s {W}x{H} -i \"{srcYuv}\" -c:v {codec} -crf {crf} -keyint_min 1 -g 1 -frames:v {frameCount} \"{ivf}\"");
    sw.Stop();
    double encFps = frameCount / sw.Elapsed.TotalSeconds;
    long sz = new FileInfo(ivf).Length;
    double avgKB = sz / 1024.0 / frameCount;

    var decYuv = Path.Combine(outDir, $"{name}_crf{crf}.yuv");
    if (!TryRunFf($"-y {decArg} -i \"{ivf}\" -f rawvideo -pix_fmt yuv420p \"{decYuv}\""))
    {
        Console.WriteLine($"{name,-15}{crf,-5}{"DECODE FAIL",-12}{avgKB,-12:F2}{encFps,-10:F2}");
        return;
    }
    var dec = File.ReadAllBytes(decYuv);
    double psnr = ComputeYPsnr(src, dec, frameCount);
    Console.WriteLine($"{name,-15}{crf,-5}{psnr,-12:F2}{avgKB,-12:F2}{encFps,-10:F2}");
}

// VP8 ours - BaseQIndex is a 7-bit field per spec (0..127).
foreach (int q in new[] { 5, 30, 80, 127 })
    Eval("VP8 (ours)", q, (y, u, v) => Vp8KeyframeEncoder.EncodeKeyFrame(y, W, u, W / 2, v, W, H, baseQIndex: q),
        "VP80", "", "ivf");

// VP9 ours - base_q_idx is 8-bit (0..255).
foreach (int q in new[] { 5, 30, 80, 200 })
    Eval("VP9 (ours)", q, (y, u, v) => Vp9KeyframeEncoder.EncodeKeyFrame(y, W, u, W / 2, v, W, H, baseQIndex: q),
        "VP90", "", "ivf");

Console.WriteLine();

// ffmpeg references for context
foreach (int crf in new[] { 4, 32, 63 })
    EvalFf("VP8 (ffmpeg)", crf, "libvpx", "VP80", "");
foreach (int crf in new[] { 4, 32, 63 })
    EvalFf("VP9 (ffmpeg)", crf, "libvpx-vp9", "VP90", "");

double ComputeYPsnr(byte[] src, byte[] dec, int frames)
{
    int yLen = W * H;
    double sumSq = 0;
    long n = 0;
    for (int f = 0; f < frames; f++)
    {
        int yOff = f * frameSize;
        for (int i = 0; i < yLen; i++)
        {
            int d = src[yOff + i] - dec[yOff + i];
            sumSq += d * d;
            n++;
        }
    }
    double mse = sumSq / n;
    if (mse <= 0) return 99.0;
    return 10.0 * Math.Log10(255.0 * 255.0 / mse);
}

(byte[] y, byte[] u, byte[] v) SliceFrame(byte[] src, int f)
{
    int yOff = f * frameSize;
    int uOff = yOff + W * H;
    int vOff = uOff + (W / 2) * (H / 2);
    return (src[yOff..uOff], src[uOff..vOff], src[vOff..(vOff + (W / 2) * (H / 2))]);
}

void RunFf(string args)
{
    var p = Process.Start(new ProcessStartInfo(ffmpeg, args) { RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true })!;
    p.StandardError.ReadToEnd();
    p.WaitForExit();
    if (p.ExitCode != 0) throw new Exception($"ffmpeg failed: {args}");
}
bool TryRunFf(string args)
{
    var p = Process.Start(new ProcessStartInfo(ffmpeg, args) { RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true })!;
    p.StandardError.ReadToEnd();
    p.WaitForExit();
    return p.ExitCode == 0;
}
