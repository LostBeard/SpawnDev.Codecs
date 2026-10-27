using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using ILGPU.Runtime.CPU;
using SpawnDev.Codecs.Images.Texture;

namespace SpawnDev.Codecs.DemoConsole;

/// <summary>
/// Encodes a raw RGBA8 picture to BCn blocks with <see cref="BcnEncoderGpu"/> and times it (the texture encode speed
/// bench; compare the blocks with the reference encoder's for the same picture).
///
/// Run: SpawnDev.Codecs.DemoConsole bcn-encode in.rgba width height bc1|bc3|bc4|bc5|bc7 out [cpu|cuda|opencl=cpu] [time]
/// </summary>
internal static class BcnEncode
{
    public static async Task<int> Run(string[] args)
    {
        if (args.Length < 6)
        {
            Console.WriteLine("usage: bcn-encode <in.rgba> <width> <height> <bc1|bc3|bc4|bc5|bc7> <out> [cpu|cuda|opencl] [time]");
            return 2;
        }
        byte[] rgba = File.ReadAllBytes(args[1]);
        int width = int.Parse(args[2]), height = int.Parse(args[3]);
        var format = args[4] switch
        {
            "bc1" => BcnFormat.Bc1,
            "bc3" => BcnFormat.Bc3,
            "bc4" => BcnFormat.Bc4,
            "bc5" => BcnFormat.Bc5,
            "bc7" => BcnFormat.Bc7,
            _ => throw new ArgumentException($"unknown format {args[4]}"),
        };
        string backend = args.Length > 6 ? args[6] : "cpu";
        using var ctx = backend switch
        {
            "cuda" => Context.Create(b => b.Cuda()),
            "opencl" => Context.Create(b => b.OpenCL()),
            _ => Context.Create(b => b.CPU()),
        };
        using Accelerator acc = backend switch
        {
            "cuda" => ctx.CreateCudaAccelerator(0),
            "opencl" => ctx.CreateCLAccelerator(0),
            _ => ctx.CreateCPUAccelerator(0),
        };
        using var enc = new BcnEncoderGpu(acc);
        byte[] blocks = await enc.EncodeAsync(rgba, width, height, format);
        File.WriteAllBytes(args[5], blocks);
        Console.WriteLine($"{width}x{height} -> {blocks.Length} bytes {format} ({acc.AcceleratorType})");
        if (args.Length > 7 && args[7] == "time")
        {
            // Warm (kernels compiled above): best of 3, device to device.
            using var dIn = acc.Allocate1D<byte>(rgba.Length);
            dIn.View.CopyFromCPU(rgba);
            using var dOut = acc.Allocate1D<byte>(blocks.Length);
            double best = double.MaxValue;
            for (int r = 0; r < 3; r++)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                enc.Encode(dIn.View, width, height, format, dOut.View);
                await acc.SynchronizeAsync();
                best = Math.Min(best, sw.Elapsed.TotalSeconds);
            }
            double mpix = (double)width * height / 1e6;
            Console.WriteLine($"encode best {best:F4} s = {mpix / best:F1} Mpixel/s");
        }
        return 0;
    }
}
