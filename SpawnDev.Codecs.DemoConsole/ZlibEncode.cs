using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using ILGPU.Runtime.CPU;
using SpawnDev.Codecs.Compression;

namespace SpawnDev.Codecs.DemoConsole;

/// <summary>
/// Writes a file as a zlib stream with <see cref="ZlibCompressorGpu"/> and times it (the zlib speed bench; the stream
/// is checked by inflating it with .NET's ZLibStream).
///
/// Run: SpawnDev.Codecs.DemoConsole zlib-encode in out.zz [level=2] [cpu|cuda|opencl=cpu] [time] [wave]
/// </summary>
internal static class ZlibEncode
{
    public static async Task<int> Run(string[] args)
    {
        if (args.Length < 3)
        {
            Console.WriteLine("usage: zlib-encode <in> <out.zz> [level] [cpu|cuda|opencl] [time] [wave]");
            return 2;
        }
        byte[] input = File.ReadAllBytes(args[1]);
        int level = args.Length > 3 ? int.Parse(args[3]) : 2;
        string backend = args.Length > 4 ? args[4] : "cpu";
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
        int wave = args.Length > 6 ? int.Parse(args[6]) : 0;
        using var comp = new ZlibCompressorGpu(acc, ZlibCompressorGpu.DefaultBlockSize, wave);
        byte[] stream = await comp.CompressAsync(input, level);
        File.WriteAllBytes(args[2], stream);
        using (var z = new System.IO.Compression.ZLibStream(new MemoryStream(stream), System.IO.Compression.CompressionMode.Decompress))
        using (var back = new MemoryStream())
        {
            z.CopyTo(back);
            bool exact = back.Length == input.Length && back.ToArray().AsSpan().SequenceEqual(input);
            Console.WriteLine($"{input.Length} -> {stream.Length} (level {level}, {acc.AcceleratorType}, wave {wave}), inflates exact: {exact}");
        }
        if (args.Length > 5 && args[5] == "time")
        {
            // Warm (kernels compiled above): best of 3, device to device.
            using var dIn = acc.Allocate1D<byte>(input.Length);
            dIn.View.CopyFromCPU(input);
            using var dOut = acc.Allocate1D<byte>(ZlibCompressorGpu.GetMaxCompressedSize(input.Length));
            double best = double.MaxValue;
            for (int r = 0; r < 3; r++)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                await comp.CompressAsync(dIn.View, input.Length, dOut.View, level);
                await acc.SynchronizeAsync();
                best = Math.Min(best, sw.Elapsed.TotalSeconds);
            }
            Console.WriteLine($"compress best {best:F3} s = {input.Length / best / 1e9:F3} GB/s");
        }
        return 0;
    }
}
