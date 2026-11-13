// vp9-encode-bench: decodes a WebM clip, re-encodes it with VideoFileEncoderGpu (VP9) for each q and key frame
// interval, and prints video bytes, key/inter frame sizes, PSNR against the source frames and encode time.
// Usage: vp9-encode-bench <clip.webm> [--backend cpu|velocity|cuda|opencl] [--frames N] [--q 20,40,60] [--intervals 1,128] [--out dir]

using System.Diagnostics;
using System.Globalization;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Velocity;
using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.Codecs.Video;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.DemoConsole;

internal static class Vp9EncodeBench
{
    private sealed class Result
    {
        public int Q;
        public int Interval;
        public long VideoBytes;
        public int KeyFrames;
    }

    private static List<int> ParseList(string text)
    {
        var list = new List<int>();
        foreach (var part in text.Split(',', StringSplitOptions.RemoveEmptyEntries))
            list.Add(int.Parse(part.Trim(), CultureInfo.InvariantCulture));
        return list;
    }

    private static string F(double v, string fmt) => v.ToString(fmt, CultureInfo.InvariantCulture);

    public static async Task<int> Run(string[] args)
    {
        if (args.Length < 2)
        {
            Console.WriteLine("usage: vp9-encode-bench <clip.webm> [--backend cpu|velocity|cuda|opencl] [--frames N] [--q 20,40,60] [--intervals 1,128] [--out dir]");
            return 2;
        }
        string clipPath = args[1];
        string backend = "velocity";
        int maxFrames = int.MaxValue;
        var qs = new List<int> { 20, 40, 60 };
        var intervals = new List<int> { 1, 128 };
        string? outDir = null;
        for (int i = 2; i < args.Length; i++)
        {
            string a = args[i];
            if (i + 1 >= args.Length)
            {
                Console.WriteLine($"missing value for {a}");
                return 2;
            }
            string v = args[++i];
            if (a == "--backend") backend = v;
            else if (a == "--frames") maxFrames = int.Parse(v, CultureInfo.InvariantCulture);
            else if (a == "--q") qs = ParseList(v);
            else if (a == "--intervals") intervals = ParseList(v);
            else if (a == "--out") outDir = v;
            else
            {
                Console.WriteLine($"unknown option {a}");
                return 2;
            }
        }
        string clipName = Path.GetFileName(clipPath);
        string clipBase = Path.GetFileNameWithoutExtension(clipPath);
        byte[] clipBytes = File.ReadAllBytes(clipPath);
        if (outDir != null) Directory.CreateDirectory(outDir);

        using var ctx = backend switch
        {
            "cuda" => Context.Create(b => b.Cuda()),
            "opencl" => Context.Create(b => b.OpenCL()),
            "velocity" => Context.Create(b => b.Velocity(VelocityDeviceType.Vector128)),
            _ => Context.Create(b => b.CPU()),
        };
        using Accelerator acc = backend switch
        {
            "cuda" => ctx.CreateCudaAccelerator(0),
            "opencl" => ctx.CreateCLAccelerator(0),
            "velocity" => ctx.CreateVelocityAccelerator(0),
            _ => ctx.CreateCPUAccelerator(0),
        };
        Console.WriteLine($"{clipName}: {clipBytes.Length / 1e6:F2} MB on {acc.Name} ({acc.AcceleratorType})");

        var loaded = await new WebMVideoLoaderGpu(acc).LoadAsync(clipBytes);
        int width = loaded.Width;
        int height = loaded.Height;
        var frames = new List<float[]>();
        foreach (var f in loaded.Frames)
        {
            if (frames.Count >= maxFrames) break;
            frames.Add(f);
        }
        int n = frames.Count;
        double fps = loaded.FrameRate;
        int fpsNum, fpsDen;
        if (Math.Abs(fps - Math.Round(fps)) < 1e-9) { fpsNum = (int)Math.Round(fps); fpsDen = 1; }
        else { fpsNum = (int)Math.Round(Math.Round(fps * 1000) ); fpsDen = 1000; }
        if (fpsNum <= 0) { fpsNum = 30; fpsDen = 1; }
        Console.WriteLine($"decoded {loaded.Frames.Count} frames {width}x{height} at {F(fps, "F3")} fps, using {n}");

        using var enc = new VideoFileEncoderGpu(acc);
        var warm = new List<float[]>();
        for (int i = 0; i < Math.Min(2, n); i++) warm.Add(frames[i]);
        await enc.EncodeAsync(warm, width, height, fpsNum, fpsDen, null,
            new VideoFileEncodeOptions { Container = VideoContainer.WebM, QIndex = qs[0], KeyFrameInterval = intervals[0] });

        var results = new List<Result>();
        var psnrMeans = new Dictionary<string, double>();
        var videoBytesBy = new Dictionary<string, long>();
        foreach (int q in qs)
        {
            foreach (int interval in intervals)
            {
                var sw = Stopwatch.StartNew();
                byte[] file = await enc.EncodeAsync(frames, width, height, fpsNum, fpsDen, null,
                    new VideoFileEncodeOptions { Container = VideoContainer.WebM, QIndex = q, KeyFrameInterval = interval });
                await acc.SynchronizeAsync();
                double encodeS = sw.Elapsed.TotalSeconds;

                long videoBytes = 0;
                int keyFrames = 0;
                long keyBytes = 0;
                long interBytes = 0;
                int interCount = 0;
                var mkv = new MatroskaContainer(new MemoryStream(file));
                foreach (var fr in mkv.Frames)
                {
                    if (fr.TrackNumber != 1) continue;
                    videoBytes += fr.Data.Length;
                    if (fr.IsKeyframe) { keyFrames++; keyBytes += fr.Data.Length; }
                    else { interCount++; interBytes += fr.Data.Length; }
                }
                double avgKey = keyFrames > 0 ? (double)keyBytes / keyFrames : 0;
                double avgInter = interCount > 0 ? (double)interBytes / interCount : 0;

                var decoded = await new WebMVideoLoaderGpu(acc).LoadAsync(file);
                int count = Math.Min(n, decoded.Frames.Count);
                double sum = 0;
                double min = double.MaxValue;
                for (int i = 0; i < count; i++)
                {
                    float[] s = frames[i];
                    float[] d = decoded.Frames[i];
                    double se = 0;
                    for (int k = 0; k < s.Length; k++)
                    {
                        double diff = (s[k] - d[k]) * 255.0;
                        se += diff * diff;
                    }
                    double mse = se / s.Length;
                    double psnr = mse <= 0 ? 99.0 : 10.0 * Math.Log10(255.0 * 255.0 / mse);
                    sum += psnr;
                    if (psnr < min) min = psnr;
                }
                double psnrMean = count > 0 ? sum / count : 0;
                if (count == 0) min = 0;

                Console.WriteLine($"BENCH vp9-encode clip={clipName} backend={backend} size={width}x{height} frames={n} q={q} interval={interval} " +
                    $"video_bytes={videoBytes} key_frames={keyFrames} avg_key_bytes={F(avgKey, "F0")} avg_inter_bytes={F(avgInter, "F0")} " +
                    $"psnr_mean={F(psnrMean, "F2")} psnr_min={F(min, "F2")} encode_s={F(encodeS, "F2")} fps={F(n / encodeS, "F2")}");
                if (decoded.Frames.Count != n)
                    Console.WriteLine($"WARN decoded frame count {decoded.Frames.Count} != source {n} (q={q} interval={interval})");

                if (outDir != null)
                    File.WriteAllBytes(Path.Combine(outDir, $"{clipBase}_q{q}_i{interval}.webm"), file);

                results.Add(new Result { Q = q, Interval = interval, VideoBytes = videoBytes, KeyFrames = keyFrames });
                psnrMeans[q + "/" + interval] = psnrMean;
                videoBytesBy[q + "/" + interval] = videoBytes;
            }
        }

        foreach (int q in qs)
        {
            if (!videoBytesBy.TryGetValue(q + "/1", out long baseBytes)) continue;
            double basePsnr = psnrMeans[q + "/1"];
            foreach (int interval in intervals)
            {
                if (interval == 1) continue;
                long bytes = videoBytesBy[q + "/" + interval];
                double delta = psnrMeans[q + "/" + interval] - basePsnr;
                string sign = delta >= 0 ? "+" : "-";
                Console.WriteLine($"RATIO q={q} interval={interval} inter/key bytes={F((double)bytes / baseBytes, "F3")} psnr_mean_delta={sign}{F(Math.Abs(delta), "F2")}");
            }
        }
        return 0;
    }
}
