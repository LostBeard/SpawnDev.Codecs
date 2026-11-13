// av1-encode-bench: Av1KeyframeEncoderGpu on a synthetic textured picture, planes already on the accelerator
// (the ArrayView overload: only the encoded length crosses to the host per frame). Prints the time per keyframe
// (median, mean, min over the timed runs, after warm-up encodes that pay for kernel compilation) and the encoded size.
// Usage: av1-encode-bench [--backend cpu|velocity|cuda|opencl] [--size 832x480] [--frames 12] [--warmup 2] [--q 40]
//                         [--tile N|max] [--maxtiles N] [--batch N] [--out file.obu] [--fps N[/D]] [--level X.Y|none]
//                         [--dump-kernel file]
// --tile: largest tile size in 64x64 superblocks per side (Av1KeyframeEncoderGpu.TileSizeSuperblocks; default 1 =
//         one GPU thread per superblock; max = the fewest tiles the picture allows, one thread per frame).
// --maxtiles: the most tiles per frame (Av1KeyframeEncoderGpu.MaxTiles; default 256, what hardware decoders take).
// --batch: frames per EncodeKeyFramesBatchAsync call (default 1 = EncodeKeyFrameAsync). A timed run is one call;
//          the time per keyframe is the call's time divided by the batch size.
// --fps: the rate the frames are shown at (Av1KeyframeEncoderGpu.SetFrameRate; default 30): it decides the level signalled.
// --level: keep the tile grid inside an AV1 level (Av1KeyframeEncoderGpu.LevelLimit), e.g. 4.0 or 5.1; none = no limit.
// --out: write the last encoded temporal unit to a file (an .obu file FFmpeg reads: ffmpeg -i file.obu ...).
// --dump-kernel: write the tile kernel as the backend compiles it (PTX on CUDA, OpenCL C on OpenCL) and exit.
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
        string? outPath = null, dumpPath = null;
        int width = 832, height = 480, frames = 12, warmup = 2, q = 40, tile = 1, batch = 1, maxTiles = Av1TileLayout.HardwareDecoderMaxTiles;
        int fpsNum = 30, fpsDen = 1, levelLimit = Av1Level.MaximumParameters;
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
            else if (a == "--tile") tile = v == "max" ? int.MaxValue : int.Parse(v, CultureInfo.InvariantCulture);
            else if (a == "--batch") batch = int.Parse(v, CultureInfo.InvariantCulture);
            else if (a == "--maxtiles") maxTiles = int.Parse(v, CultureInfo.InvariantCulture);
            else if (a == "--out") outPath = v;
            else if (a == "--fps")
            {
                var parts = v.Split('/');
                fpsNum = int.Parse(parts[0], CultureInfo.InvariantCulture);
                fpsDen = parts.Length > 1 ? int.Parse(parts[1], CultureInfo.InvariantCulture) : 1;
            }
            else if (a == "--level")
            {
                if (v == "none") levelLimit = Av1Level.MaximumParameters;
                else
                {
                    var parts = v.Split('.');
                    if (parts.Length != 2) { Console.WriteLine("--level takes X.Y (for example 5.1) or none"); return 2; }
                    levelLimit = ((int.Parse(parts[0], CultureInfo.InvariantCulture) - 2) << 2) | int.Parse(parts[1], CultureInfo.InvariantCulture);
                }
            }
            else if (a == "--dump-kernel") dumpPath = v;
            else { Console.WriteLine($"unknown option {a}"); return 2; }
        }
        if (frames < 1 || warmup < 0 || tile < 1 || batch < 1 || maxTiles < 1) { Console.WriteLine("--frames, --tile, --maxtiles and --batch must be >= 1 and --warmup >= 0"); return 2; }

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

        if (dumpPath != null)
        {
            var method = typeof(Av1FrameSequentialEncodeKernel).GetMethod("Av1EncodeTileStepKernel", BindingFlags.NonPublic | BindingFlags.Static)
                ?? throw new InvalidOperationException("Av1EncodeTileStepKernel not found");
            var compiled = acc.CompileKernel(global::ILGPU.Backends.EntryPoints.EntryPointDescription.FromImplicitlyGroupedKernel(method));
            string? text = compiled switch
            {
                global::ILGPU.Backends.PTX.PTXCompiledKernel ptx => ptx.PTXAssembly,
                global::ILGPU.Backends.OpenCL.CLCompiledKernel cl => cl.Source,
                _ => null,
            };
            if (text == null) { Console.WriteLine($"no source text for a {compiled.GetType().Name}"); return 2; }
            await File.WriteAllTextAsync(dumpPath, text);
            Console.WriteLine($"{acc.AcceleratorType} kernel: {text.Length} characters -> {dumpPath}");
            // OpenCL: also the driver's program binary (what its compiler made of the source)
            if (acc.LoadKernel(compiled) is global::ILGPU.Runtime.OpenCL.CLKernel clKernel)
            {
                byte[] binary = clKernel.LoadBinaryRepresentation();
                await File.WriteAllBytesAsync(dumpPath + ".bin", binary);
                Console.WriteLine($"OpenCL program binary: {binary.Length} bytes -> {dumpPath}.bin");
            }
            return 0;
        }

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

        using var enc = new Av1KeyframeEncoderGpu(acc) { TileSizeSuperblocks = tile, MaxTiles = maxTiles, LevelLimit = levelLimit };
        enc.SetFrameRate(fpsNum, fpsDen);
        var layout = enc.GetTileLayout(width, height);
        int cap = Av1KeyframeEncoderGpu.GetMaxEncodedSize(width, height);
        // batch frames of the same picture, frame-major (the batch overload's layout)
        using var dY = acc.Allocate1D<byte>((long)batch * y.Length);
        using var dU = acc.Allocate1D<byte>((long)batch * u.Length);
        using var dV = acc.Allocate1D<byte>((long)batch * v2.Length);
        for (int f = 0; f < batch; f++)
        {
            dY.View.SubView((long)f * y.Length, y.Length).CopyFromCPU(y);
            dU.View.SubView((long)f * u.Length, u.Length).CopyFromCPU(u);
            dV.View.SubView((long)f * v2.Length, v2.Length).CopyFromCPU(v2);
        }
        using var dOut = acc.Allocate1D<byte>((long)batch * cap);
        using var dLens = acc.Allocate1D<int>(batch);

        // One timed run = one call (a keyframe, or a batch of them); the length readback closes it.
        async Task<int> EncodeOnce()
        {
            if (batch == 1)
                return await enc.EncodeKeyFrameAsync(dY.View, dU.View, dV.View, width, height, dOut.View, q);
            await enc.EncodeKeyFramesBatchAsync(dY.View, dU.View, dV.View, batch, width, height, dOut.View, dLens.View, q);
            int[] lens = await dLens.CopyToHostAsync();
            for (int f = 1; f < batch; f++)
                if (lens[f] != lens[0]) throw new InvalidOperationException($"batch frame {f} is {lens[f]} bytes, frame 0 is {lens[0]} (the frames are the same picture)");
            return lens[0];
        }

        var compile = Stopwatch.StartNew();
        int len = 0;
        for (int i = 0; i < warmup; i++) len = await EncodeOnce();
        compile.Stop();

        var ms = new double[frames];
        for (int i = 0; i < frames; i++)
        {
            var sw = Stopwatch.StartNew();
            len = await EncodeOnce();
            sw.Stop();
            ms[i] = sw.Elapsed.TotalMilliseconds / batch;
        }
        var sorted = (double[])ms.Clone();
        System.Array.Sort(sorted);
        double median = frames % 2 == 1 ? sorted[frames / 2] : (sorted[frames / 2 - 1] + sorted[frames / 2]) / 2;
        double mean = 0;
        foreach (double m in ms) mean += m;
        mean /= frames;
        string build = typeof(Av1KeyframeEncoderGpu).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "?";
        // the first frame's temporal unit: its sequence header (after the 2-byte temporal delimiter) for the level, and the file
        byte[] unit = await dOut.View.SubView(0, len).CopyToHostAsync();
        var sh = Av1SequenceHeaderParser.Parse(unit.AsSpan(4, unit[3]));
        var level = new Av1LevelSignal(sh.SeqLevelIdx0, sh.SeqTier0);
        double mbps = len * 8.0 * fpsNum / fpsDen / 1e6;
        if (outPath != null) await File.WriteAllBytesAsync(outPath, unit);
        string tileText = tile == int.MaxValue ? "max" : tile.ToString(CultureInfo.InvariantCulture);
        Console.WriteLine($"av1-encode-bench {acc.AcceleratorType} {width}x{height} q{q} tile {tileText} " +
            $"({layout.TileCols} x {layout.TileRows} = {layout.TileCount} tiles of {layout.TileWidthSb} x {layout.TileHeightSb} superblocks) batch {batch}: " +
            $"median {F(median, "F2")} ms, mean {F(mean, "F2")} ms, " +
            $"min {F(sorted[0], "F2")} ms, max {F(sorted[frames - 1], "F2")} ms per keyframe over {frames} {(batch == 1 ? "frames" : "batches")} " +
            $"({F(1000.0 / median, "F2")} fps); {len} bytes = {F(mbps, "F1")} Mbit/s at {fpsNum}/{fpsDen} fps, level signalled: {level} (seq_level_idx {level.SeqLevelIdx}){(levelLimit == Av1Level.MaximumParameters ? "" : $", tile grid limited to seq_level_idx {levelLimit}")}; warm-up {warmup} frames {F(compile.Elapsed.TotalMilliseconds, "F0")} ms; SpawnDev.Codecs {build}");
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
