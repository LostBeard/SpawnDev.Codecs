// vp9-key-bench: Vp9KeyframeEncoderGpu on a synthetic textured picture, planes already on the accelerator (only the
// encoded length crosses to the host per frame). Prints the time per key frame (median, mean, min over the timed runs,
// after warm-up encodes that pay for kernel compilation) and the encoded size.
// Usage: vp9-key-bench [--backend cpu|velocity|cuda|opencl] [--size 512x512] [--frames 12] [--warmup 2] [--q 40]
//                      [--stages 1] [--inter N] [--out file.vp9frame]
// --inter N: Vp9EncoderGpu instead: a key frame, then N inter frames of a picture moving 3 pixels right and 1 down per
//            frame (motion search on); the times are of the inter frames only.
// --stages 1: measurements: synchronize after every launch of the encode (Vp9KeyframeEncoderGpu.Trace) and report the
//             time per stage, summed over a frame (the entropy steps are one stage). The waits add their own cost: the
//             per-stage times add up to more than an untimed frame; run without --stages for the real per-frame time.
// --serial 1|0: force the serial walkers (one thread) or the parallel entropy passes (the A/B); default: the encoder's choice.
// --out: write the last encoded frame (a raw VP9 frame) to a file.
// For an A/B of two builds: run the two executables alternately, several times each, with nothing else on the GPU.

using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Text.RegularExpressions;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Velocity;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.DemoConsole;

internal static class Vp9KeyBench
{
    private static string F(double v, string fmt) => v.ToString(fmt, CultureInfo.InvariantCulture);

    public static async Task<int> Run(string[] args)
    {
        string backend = "cuda";
        string? outPath = null;
        int width = 512, height = 512, frames = 12, warmup = 2, q = 40;
        bool stages = false;
        int inter = 0;
        bool? serial = null;
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
            else if (a == "--stages") stages = v == "1";
            else if (a == "--out") outPath = v;
            else if (a == "--inter") inter = int.Parse(v, CultureInfo.InvariantCulture);
            else if (a == "--serial") serial = v == "1";
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
        // CODECS_VELOCITY_HOST_COPY=0: Velocity host-path kernels as the original methods (SpawnDev.ILGPU 5.4.31's
        // behaviour) instead of 5.4.32's optimized copies, for an A/B (as VelocityCodecsTests)
        if (acc is VelocityAccelerator va && Environment.GetEnvironmentVariable("CODECS_VELOCITY_HOST_COPY") == "0") va.OptimizedHostKernels = false;

        int cw = (width + 1) / 2, ch = (height + 1) / 2;
        // picture f is the texture moved 3 pixels right and 1 down per frame (f = 0 for the key frame bench)
        int pictures = inter > 0 ? 1 + warmup + inter * frames : 1;
        if (inter > 0 && frames != 1) { Console.WriteLine("--inter N times N inter frames: leave --frames at 1"); return 2; }
        if (inter > 0) pictures = 1 + warmup + inter;
        var y = new byte[(long)pictures * width * height]; var u = new byte[(long)pictures * cw * ch]; var v2 = new byte[(long)pictures * cw * ch];
        for (int f = 0; f < pictures; f++)
        {
            long yAt = (long)f * width * height, cAt = (long)f * cw * ch;
            for (int r = 0; r < height; r++)
                for (int c = 0; c < width; c++) y[yAt + r * width + c] = Texel(c - 3 * f, r - f, 1);
            for (int r = 0; r < ch; r++)
                for (int c = 0; c < cw; c++)
                {
                    u[cAt + r * cw + c] = Texel(2 * c + 301 - 3 * f, 2 * r + 17 - f, 2);
                    v2[cAt + r * cw + c] = Texel(2 * c + 77 - 3 * f, 2 * r + 409 - f, 3);
                }
        }

        using var enc = new Vp9KeyframeEncoderGpu(acc);
        using var seq = new Vp9EncoderGpu(acc) { KeyFrameInterval = 0 };
        if (serial.HasValue) { enc.SerialEntropy = serial.Value; seq.SerialEntropy = serial.Value; }
        bool serialUsed = enc.SerialEntropy;
        int cap = Vp9KeyframeEncoderGpu.GetMaxEncodedSize(width, height);
        using var dY = acc.Allocate1D(y);
        using var dU = acc.Allocate1D(u);
        using var dV = acc.Allocate1D(v2);
        using var dOut = acc.Allocate1D<byte>(cap);
        int picture = 0;
        async Task<int> EncodeOnce()
        {
            if (inter == 0) return await enc.EncodeKeyFrameAsync(dY.View, dU.View, dV.View, width, height, dOut.View, q);
            long yLen = (long)width * height, cLen = (long)cw * ch;
            int f = picture++;
            var r = await seq.EncodeFrameAsync(dY.View.SubView(f * yLen, yLen), dU.View.SubView(f * cLen, cLen),
                dV.View.SubView(f * cLen, cLen), width, height, dOut.View, q);
            if (f > 0 && r.IsKeyFrame) throw new InvalidOperationException($"frame {f} is a key frame");
            return r.Length;
        }

        var compile = Stopwatch.StartNew();
        int len = 0;
        // inter: the key frame + warm-up inter frames
        for (int i = 0; i < warmup + (inter > 0 ? 1 : 0); i++) len = await EncodeOnce();
        compile.Stop();
        if (inter > 0) frames = inter;

        // per-stage sums: the time from the previous trace point to this one, by stage name ("step 3 of 128" -> "step")
        var stageMs = new Dictionary<string, double>();
        var order = new List<string>();
        var mark = Stopwatch.StartNew();
        // "wavefront single-dispatch diagonals 0..8 of 63 (1 frames x 32)" -> "wavefront single-dispatch diagonals"
        var stepNumber = new Regex(@"\s*[\d.]+ of \d+|\s*\([^)]*\)", RegexOptions.CultureInvariant);
        if (stages)
        {
            Vp9KeyframeEncoderGpu.Trace = what =>
            {
                string key = stepNumber.Replace(what, "");
                double ms = mark.Elapsed.TotalMilliseconds;
                if (!stageMs.ContainsKey(key)) order.Add(key);
                stageMs[key] = stageMs.GetValueOrDefault(key) + ms;
                mark.Restart();
            };
        }

        var times = new double[frames];
        try
        {
            for (int i = 0; i < frames; i++)
            {
                mark.Restart();
                var sw = Stopwatch.StartNew();
                len = await EncodeOnce();
                sw.Stop();
                if (stages)
                {
                    const string tail = "(after last trace point: headers, assembly, length readback)";
                    if (!stageMs.ContainsKey(tail)) order.Add(tail);
                    stageMs[tail] = stageMs.GetValueOrDefault(tail) + mark.Elapsed.TotalMilliseconds;
                }
                times[i] = sw.Elapsed.TotalMilliseconds;
            }
        }
        finally
        {
            Vp9KeyframeEncoderGpu.Trace = null;
        }
        var sorted = (double[])times.Clone();
        System.Array.Sort(sorted);
        double median = frames % 2 == 1 ? sorted[frames / 2] : (sorted[frames / 2 - 1] + sorted[frames / 2]) / 2;
        double mean = 0;
        foreach (double m in times) mean += m;
        mean /= frames;
        if (outPath != null) await File.WriteAllBytesAsync(outPath, await dOut.View.SubView(0, len).CopyToHostAsync());
        foreach (string key in order)
            Console.WriteLine($"stage {key}: {F(stageMs[key] / frames, "F2")} ms per frame");
        string build = typeof(Vp9KeyframeEncoderGpu).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "?";
        string kind = inter > 0 ? "inter" : "key";
        Console.WriteLine($"vp9-key-bench {acc.AcceleratorType} {width}x{height} q{q} {kind} frames, {(serialUsed ? "SERIAL" : "parallel")} entropy{(stages ? " (stages timed: every launch waited for)" : "")}: " +
            $"median {F(median, "F2")} ms, mean {F(mean, "F2")} ms, min {F(sorted[0], "F2")} ms, max {F(sorted[frames - 1], "F2")} ms " +
            $"per {kind} frame over {frames} frames ({F(1000.0 / median, "F2")} fps); {len} bytes; " +
            $"warm-up {warmup} frames {F(compile.Elapsed.TotalMilliseconds, "F0")} ms; SpawnDev.Codecs {build}");
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
