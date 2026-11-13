// vp9-psnr: PSNR of one or more encoded WebM files against a source WebM clip, plus video bytes and key frame count.
// Usage: vp9-psnr <source.webm> <encoded1.webm> [encoded2.webm ...] [--backend cpu|velocity|cuda|opencl] [--frames N] [--label text]

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

internal static class Vp9PsnrCompare
{
    private static string F(double v, string fmt) => v.ToString(fmt, CultureInfo.InvariantCulture);

    public static async Task<int> Run(string[] args)
    {
        const string usage = "usage: vp9-psnr <source.webm> <encoded1.webm> [encoded2.webm ...] [--backend cpu|velocity|cuda|opencl] [--frames N] [--label text]";
        var files = new List<string>();
        string backend = "velocity";
        int maxFrames = int.MaxValue;
        string label = "-";
        int i = 1;
        while (i < args.Length && !args[i].StartsWith("--", StringComparison.Ordinal)) files.Add(args[i++]);
        for (; i < args.Length; i++)
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
            else if (a == "--label") label = v;
            else
            {
                Console.WriteLine($"unknown option {a}");
                return 2;
            }
        }
        if (files.Count < 2)
        {
            Console.WriteLine(usage);
            return 2;
        }
        string clipPath = files[0];
        string clipName = Path.GetFileName(clipPath);
        byte[] clipBytes = File.ReadAllBytes(clipPath);

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

        for (int fi = 1; fi < files.Count; fi++)
        {
            string path = files[fi];
            string fileName = Path.GetFileName(path);
            byte[] file = File.ReadAllBytes(path);

            var mkv = new MatroskaContainer(new MemoryStream(file));
            ulong videoTrack = 0;
            foreach (var t in mkv.Tracks)
            {
                if (t.CodecId == "V_VP9") { videoTrack = t.TrackNumber; break; }
            }
            if (videoTrack == 0)
            {
                Console.WriteLine($"ERROR no V_VP9 track in {fileName}");
                continue;
            }
            long videoBytes = 0;
            int keyFrames = 0;
            int videoFrames = 0;
            foreach (var fr in mkv.Frames)
            {
                if (fr.TrackNumber != videoTrack) continue;
                videoFrames++;
                videoBytes += fr.Data.Length;
                if (fr.IsKeyframe) keyFrames++;
            }

            var decoded = await new WebMVideoLoaderGpu(acc).LoadAsync(file);
            if (decoded.Width != width || decoded.Height != height)
            {
                Console.WriteLine($"ERROR size mismatch file={fileName} decoded={decoded.Width}x{decoded.Height} source={width}x{height}");
                continue;
            }
            int count = Math.Min(n, decoded.Frames.Count);
            if (decoded.Frames.Count != n)
                Console.WriteLine($"WARN decoded frame count {decoded.Frames.Count} != source {n} (file={fileName}), using {count}");

            double sum = 0;
            double min = double.MaxValue;
            double sumY = 0, minY = double.MaxValue, lastY = 0;
            int lastCount = 0;
            for (int k0 = 0; k0 < count; k0++)
            {
                float[] s = frames[k0];
                float[] d = decoded.Frames[k0];
                double se = 0;
                for (int k = 0; k < s.Length; k++)
                {
                    double diff = (s[k] - d[k]) * 255.0;
                    se += diff * diff;
                }
                double seY = 0;
                for (int k = 0; k < s.Length; k += 3)
                {
                    double diff = (0.299 * (s[k] - d[k]) + 0.587 * (s[k + 1] - d[k + 1]) + 0.114 * (s[k + 2] - d[k + 2])) * 255.0;
                    seY += diff * diff;
                }
                double mseY = seY / (s.Length / 3);
                double py = mseY <= 0 ? 99.0 : 10.0 * Math.Log10(255.0 * 255.0 / mseY);
                sumY += py;
                if (py < minY) minY = py;
                if (k0 >= count - 8) { lastY += py; lastCount++; }
                double mse = se / s.Length;
                double psnr = mse <= 0 ? 99.0 : 10.0 * Math.Log10(255.0 * 255.0 / mse);
                sum += psnr;
                if (psnr < min) min = psnr;
            }
            double psnrMean = count > 0 ? sum / count : 0;
            double psnrY = count > 0 ? sumY / count : 0;
            if (count == 0) min = 0;

            Console.WriteLine($"PSNR clip={clipName} file={fileName} label={label} size={width}x{height} frames={videoFrames} video_bytes={videoBytes} key_frames={keyFrames} " +
                $"psnr_mean={F(psnrMean, "F2")} psnr_min={F(min, "F2")} psnr_y={F(psnrY, "F3")} psnr_y_min={F(count > 0 ? minY : 0, "F2")} psnr_y_last8={F(lastCount > 0 ? lastY / lastCount : 0, "F2")}");
        }
        return 0;
    }
}
