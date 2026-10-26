using System.Diagnostics;
using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using ILGPU.Runtime.CPU;
using SpawnDev.Codecs.Compression.GDeflate;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.DemoConsole;

/// <summary>
/// Benchmark of <see cref="GDeflateDecompressorGpu"/> on a GDeflate stream written by Microsoft's reference (any file
/// compressed with the scratchpad gdcli driver): best of reps, decoded output checked against the original, one JSON
/// line (for _TestStats, kind "bench").
///
/// Run: SpawnDev.Codecs.DemoConsole gdeflate-bench file.gdf original.bin [cuda|opencl|cpu=cuda] [reps=3]
/// </summary>
internal static class GDeflateBench
{
    public static async Task<int> Run(string[] args)
    {
        if (args.Length < 3)
        {
            Console.WriteLine("usage: gdeflate-bench <file.gdf> <original> [cuda|opencl|cpu] [reps]");
            return 2;
        }
        byte[] stream = File.ReadAllBytes(args[1]);
        byte[] original = File.ReadAllBytes(args[2]);
        string backend = args.Length > 3 ? args[3] : "cuda";
        int reps = args.Length > 4 ? int.Parse(args[4]) : 3;

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
        Console.WriteLine($"{Path.GetFileName(args[1])}: {stream.Length / 1e6:F1} MB -> {original.Length / 1e6:F1} MB on {acc.Name} ({acc.AcceleratorType})");
        using var dec = new GDeflateDecompressorGpu(acc, 16384);
        using var dIn = acc.Allocate1D<byte>(stream.Length);
        dIn.View.CopyFromCPU(stream);
        using var dOut = acc.Allocate1D<byte>(original.Length);

        int n = await dec.DecompressAsync(dIn.View, stream.Length, dOut.View); // warm-up (kernel compile)
        double best = double.MaxValue;
        for (int r = 0; r < reps; r++)
        {
            var sw = Stopwatch.StartNew();
            n = await dec.DecompressAsync(dIn.View, stream.Length, dOut.View);
            await acc.SynchronizeAsync();
            best = Math.Min(best, sw.Elapsed.TotalSeconds);
        }
        var back = await dOut.View.CopyToHostAsync();
        bool exact = n == original.Length && back.AsSpan(0, n).SequenceEqual(original);
        var record = new
        {
            kind = "bench",
            repo = "SpawnDev.Codecs",
            name = "GDeflate decode throughput",
            value = Math.Round(original.Length / best / 1e9, 3),
            unit = "GB/s",
            config = $"gdeflate-{backend}-{Path.GetFileNameWithoutExtension(args[1])}",
            stream_bytes = stream.Length,
            output_bytes = original.Length,
            tiles = (original.Length + 65535) / 65536,
            decode_seconds = Math.Round(best, 4),
            exact,
            accelerator = acc.Name,
        };
        Console.WriteLine(JsonSerializer.Serialize(record));
        return exact ? 0 : 1;
    }
}
