using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using ILGPU.Runtime.CPU;
using SpawnDev.Codecs.Compression.GDeflate;

namespace SpawnDev.Codecs.DemoConsole;

/// <summary>
/// Writes a file as a GDeflate stream with <see cref="GDeflateCompressorGpu"/>, for checking with Microsoft's reference
/// decoder (scripts/compression/check_gdeflate_encoder.py).
///
/// Run: SpawnDev.Codecs.DemoConsole gdeflate-encode in out.gdf [level=2] [cpu|cuda|opencl=cpu]
/// </summary>
internal static class GDeflateEncode
{
    public static async Task<int> Run(string[] args)
    {
        if (args.Length < 3)
        {
            Console.WriteLine("usage: gdeflate-encode <in> <out.gdf> [level] [cpu|cuda|opencl]");
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
        using var comp = new GDeflateCompressorGpu(acc);
        byte[] stream = await comp.CompressAsync(input, level);
        File.WriteAllBytes(args[2], stream);
        Console.WriteLine($"{input.Length} -> {stream.Length} (level {level}, {acc.AcceleratorType})");
        return 0;
    }
}
