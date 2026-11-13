// av1-encode-bench: Av1KeyframeEncoderGpu on a synthetic textured picture, planes already on the accelerator
// (the ArrayView overload: only the encoded length crosses to the host per frame). Prints the time per keyframe
// (median, mean, min over the timed runs, after warm-up encodes that pay for kernel compilation) and the encoded size.
// Usage: av1-encode-bench [--backend cpu|velocity|cuda|opencl] [--size 832x480] [--frames 12] [--warmup 2] [--q 40]
// For an A/B of two builds: run the two executables alternately, several times each, with nothing else on the GPU.

using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Velocity;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.DemoConsole;

internal static class Av1EncodeBench
{
    private static string F(double v, string fmt) => v.ToString(fmt, CultureInfo.InvariantCulture);

    public static async Task<int> Run(string[] args)
    {
        string backend = "velocity";
        int width = 832, height = 480, frames = 12, warmup = 2, q = 40;
        for (int i = 1; i < args.Length; i++)
        {
            string a = args[i];
            if (i + 1 >= args.Length) { Console.WriteLine($"missing value for {a}"); return 2; }
            string v = args[++i];
            if (a == "--backend") backend = v;
            else if (a == "--size")
            {
                var parts = v.Split('x');
                if (parts.Length != 2) { Console.WriteLine("--size takes WIDTHxHEIGHT"); return 2; }
                width = int.Parse(parts[0], CultureInfo.InvariantCulture);
                height = int.Parse(parts[1], CultureInfo.InvariantCulture);
            }
            else if (a == "--frames") frames = int.Parse(v, CultureInfo.InvariantCulture);
            else if (a == "--warmup") warmup = int.Parse(v, CultureInfo.InvariantCulture);
            else if (a == "--q") q = int.Parse(v, CultureInfo.InvariantCulture);
            else { Console.WriteLine($"unknown option {a}"); return 2; }
        }
        if (frames < 1 || warmup < 0) { Console.WriteLine("--frames must be >= 1 and --warmup >= 0"); return 2; }

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

        int cw = Av1KeyframeEncoderGpu.ChromaWidth(width), ch = Av1KeyframeEncoderGpu.ChromaHeight(height);
        var y = new byte[width * height]; var u = new byte[cw * ch]; var v2 = new byte[cw * ch];
        for (int r = 0; r < height; r++)
            for (int c = 0; c < width; c++) y[r * width + c] = Texel(c, r, 1);
        for (int r = 0; r < ch; r++)
            for (int c = 0; c < cw; c++)
            {
                u[r * cw + c] = Texel(2 * c + 301, 2 * r + 17, 2);
                v2[r * cw + c] = Texel(2 * c + 77, 2 * r + 409, 3);
            }

        using var enc = new Av1KeyframeEncoderGpu(acc);
        using var dY = acc.Allocate1D<byte>(y.Length);
        using var dU = acc.Allocate1D<byte>(u.Length);
        using var dV = acc.Allocate1D<byte>(v2.Length);
        dY.View.CopyFromCPU(y); dU.View.CopyFromCPU(u); dV.View.CopyFromCPU(v2);
        using var dOut = acc.Allocate1D<byte>(Av1KeyframeEncoderGpu.GetMaxEncodedSize(width, height));

        var compile = Stopwatch.StartNew();
        int len = 0;
        for (int i = 0; i < warmup; i++)
            len = await enc.EncodeKeyFrameAsync(dY.View, dU.View, dV.View, width, height, dOut.View, q);
        compile.Stop();

        var ms = new double[frames];
        for (int i = 0; i < frames; i++)
        {
            var sw = Stopwatch.StartNew();
            len = await enc.EncodeKeyFrameAsync(dY.View, dU.View, dV.View, width, height, dOut.View, q);
            sw.Stop();
            ms[i] = sw.Elapsed.TotalMilliseconds;
        }
        var sorted = (double[])ms.Clone();
        System.Array.Sort(sorted);
        double median = frames % 2 == 1 ? sorted[frames / 2] : (sorted[frames / 2 - 1] + sorted[frames / 2]) / 2;
        double mean = 0;
        foreach (double m in ms) mean += m;
        mean /= frames;
        string build = typeof(Av1KeyframeEncoderGpu).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "?";
        Console.WriteLine($"av1-encode-bench {acc.AcceleratorType} {width}x{height} q{q}: median {F(median, "F2")} ms, mean {F(mean, "F2")} ms, " +
            $"min {F(sorted[0], "F2")} ms, max {F(sorted[frames - 1], "F2")} ms per keyframe over {frames} frames " +
            $"({F(1000.0 / median, "F2")} fps); {len} bytes; warm-up {warmup} frames {F(compile.Elapsed.TotalMilliseconds, "F0")} ms; SpawnDev.Codecs {build}");
        return 0;
    }

    private static byte Texel(int x, int y, int seed)
    {
        int wave = (int)(52 * Math.Sin(x * 0.19 + seed) + 44 * Math.Cos(y * 0.27 - seed * 0.5));
        int detail = (((x * 7) ^ (y * 13) ^ (seed * 31)) & 31) - 16;
        int v = 128 + wave + detail + ((x + 2 * y) & 63) - 32;
        return (byte)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
}
