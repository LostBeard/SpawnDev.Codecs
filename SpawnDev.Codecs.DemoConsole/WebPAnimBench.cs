// webp-anim-bench: animated WebP encode of a folder of PNG frames (SpawnGen's Save Animated WEBP; Lore's report
// _DevComms/2026-11-14-0500-Lore-TO-Geordi-webp-gpu-encode-slow.md). Prints the encode time and the file size per mode.
// Usage: webp-anim-bench <frames folder> [--backend cpu|velocity|cuda|opencl] [--frames N] [--quality 80]
//                        [--mode rect|anim|lossy] [--runs 2] [--stages 1] [--out file.webp]
// --mode rect:  WebPLosslessEncoderGpu.EncodeAnimationAsync(frames, ImageFrameInfo per frame) - the 1.0.1 call: every
//               frame full size at (0, 0), duration 41, no disposal, blend Source.
// --mode anim:  WebPLosslessEncoderGpu.EncodeAnimationAsync(frames, w, h, durations, WebPEncodeOptions { Quality,
//               Exact = false }) - the local.18 call (libwebp WebPAnimEncoder decisions).
// --mode lossy: WebPLossyEncoderGpu.EncodeAnimationAsync(frames, w, h, durations, WebPLossyEncodeOptions { Quality }).
// --stages 1:   the lossless encoder's trace points (WebPLosslessEncoderGpu.Trace / WebPEncodeKernels.Trace): the time
//               from one point to the next, summed per name over the run (the trace waits for the device at each point).

using System.Diagnostics;
using System.Globalization;
using System.Text.RegularExpressions;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Velocity;
using SpawnDev.Codecs.Images;
using SpawnDev.Codecs.Images.Png;
using SpawnDev.Codecs.Images.WebP;
using SpawnDev.Codecs.Images.WebP.Lossy;

namespace SpawnDev.Codecs.DemoConsole;

internal static class WebPAnimBench
{
    private static string F(double v, string fmt) => v.ToString(fmt, CultureInfo.InvariantCulture);

    public static async Task<int> Run(string[] args)
    {
        if (args.Length < 2) { Console.WriteLine("usage: webp-anim-bench <frames folder> [--backend ...] [--frames N] [--quality 80] [--mode rect|anim|lossy] [--runs 2] [--stages 1] [--out file]"); return 2; }
        string dir = args[1];
        string backend = "cuda", mode = "anim";
        string? outPath = null;
        int maxFrames = int.MaxValue, quality = 80, runs = 2;
        bool stages = false;
        for (int i = 2; i < args.Length; i++)
        {
            string a = args[i];
            if (i + 1 >= args.Length) { Console.WriteLine($"missing value for {a}"); return 2; }
            string v = args[++i];
            if (a == "--backend") backend = v;
            else if (a == "--frames") maxFrames = int.Parse(v, CultureInfo.InvariantCulture);
            else if (a == "--quality") quality = int.Parse(v, CultureInfo.InvariantCulture);
            else if (a == "--mode") mode = v;
            else if (a == "--runs") runs = int.Parse(v, CultureInfo.InvariantCulture);
            else if (a == "--stages") stages = v == "1";
            else if (a == "--out") outPath = v;
            else { Console.WriteLine($"unknown option {a}"); return 2; }
        }

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

        // the frames (bench input setup: PNG files decoded on the accelerator, kept as managed RGBA like SpawnGen's)
        var files = Directory.GetFiles(dir, "*.png").OrderBy(f => f, StringComparer.Ordinal).Take(maxFrames).ToArray();
        if (files.Length == 0) { Console.WriteLine($"no .png files in {dir}"); return 2; }
        var png = new PngDecoderGpu(acc);
        var frames = new List<ReadOnlyMemory<byte>>();
        int w = 0, h = 0;
        foreach (var f in files)
        {
            var img = await png.DecodeAsync(await File.ReadAllBytesAsync(f), PixelFormat.Rgba8);
            w = img.Info.Width; h = img.Info.Height;
            frames.Add(img.Pixels);
        }
        var durations = Enumerable.Repeat(41, frames.Count).ToArray();
        Console.WriteLine($"{frames.Count} frames {w}x{h} from {dir}; {acc.Name} ({acc.AcceleratorType}); mode {mode}, quality {quality}");

        var stageMs = new Dictionary<string, double>();
        var order = new List<string>();
        var mark = Stopwatch.StartNew();
        var number = new Regex(@"\d+", RegexOptions.CultureInvariant);
        void Trace(string what)
        {
            string key = number.Replace(what, "#");
            if (!stageMs.ContainsKey(key)) order.Add(key);
            stageMs[key] = stageMs.GetValueOrDefault(key) + mark.Elapsed.TotalMilliseconds;
            mark.Restart();
        }

        using var lossless = new WebPLosslessEncoderGpu(acc);
        using var lossy = new WebPLossyEncoderGpu(acc);
        byte[] file = Array.Empty<byte>();
        for (int r = 0; r < runs; r++)
        {
            bool timedStages = stages && r == runs - 1;
            if (timedStages) { WebPLosslessEncoderGpu.Trace = Trace; WebPEncodeKernels.Trace = Trace; }
            mark.Restart();
            var sw = Stopwatch.StartNew();
            try
            {
                if (mode == "rect")
                {
                    var infos = new List<ImageFrameInfo>();
                    for (int k = 0; k < frames.Count; k++) infos.Add(new ImageFrameInfo(k, 0, 0, w, h, 41, FrameDispose.None, FrameBlend.Source));
                    file = await lossless.EncodeAnimationAsync(frames, infos, w, h, new WebPAnimationEncodeOptions { Quality = quality, LoopCount = 0 });
                }
                else if (mode == "anim")
                    file = await lossless.EncodeAnimationAsync(frames, w, h, durations, new WebPEncodeOptions { Quality = quality, Exact = false });
                else
                    file = await lossy.EncodeAnimationAsync(frames, w, h, durations, new WebPLossyEncodeOptions { Quality = quality });
            }
            finally { WebPLosslessEncoderGpu.Trace = null; WebPEncodeKernels.Trace = null; }
            sw.Stop();
            Console.WriteLine($"run {r + 1}: {F(sw.Elapsed.TotalSeconds, "F2")} s ({F(sw.Elapsed.TotalMilliseconds / frames.Count, "F0")} ms per frame), {file.Length} bytes");
        }
        foreach (string key in order.OrderByDescending(k => stageMs[k]).Take(25))
            Console.WriteLine($"stage {key}: {F(stageMs[key], "F0")} ms total");
        if (outPath != null) await File.WriteAllBytesAsync(outPath, file);
        Console.WriteLine($"webp-anim-bench {acc.AcceleratorType} {mode} q{quality}: {frames.Count} frames {w}x{h}, {file.Length} bytes");
        return 0;
    }
}
