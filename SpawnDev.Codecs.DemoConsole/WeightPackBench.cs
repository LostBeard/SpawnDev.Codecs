using System.Diagnostics;
using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using ILGPU.Runtime.CPU;
using SpawnDev.ILGPU;
using SpawnDev.Codecs.Compression;

namespace SpawnDev.Codecs.DemoConsole;

/// <summary>
/// Benchmark of <see cref="WeightPackGpu"/> on REAL model weights (a .safetensors file): the same sampling as the
/// Python measurement (scratchpad weights_entropy.py: tensors of the file's main dtype spread over the file, up to
/// maxMB), packed and unpacked on the accelerator, round trip verified, ratio and throughput printed as one JSON line
/// (for _TestStats, kind "bench").
///
/// Run: SpawnDev.Codecs.DemoConsole weightpack-bench file.safetensors [maxMB=512] [cuda|opencl|cpu=cuda] [reps=3]
/// </summary>
internal static class WeightPackBench
{
    public static async Task<int> Run(string[] args)
    {
        if (args.Length < 2)
        {
            Console.WriteLine("usage: weightpack-bench <file.safetensors> [maxMB] [cuda|opencl|cpu] [reps]");
            return 2;
        }
        string path = args[1];
        int maxMb = args.Length > 2 ? int.Parse(args[2]) : 512;
        string backend = args.Length > 3 ? args[3] : "cuda";
        int reps = args.Length > 4 ? int.Parse(args[4]) : 3;

        var (data, dtype, elementSize, tensors) = Sample(path, maxMb);
        Console.WriteLine($"{Path.GetFileName(path)}: sampled {data.Length / 1e6:F0} MB of {dtype} from {tensors} tensors");

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
        Console.WriteLine($"accelerator: {acc.Name} ({acc.AcceleratorType})");
        using var pack = new WeightPackGpu(acc);
        using var dIn = acc.Allocate1D<byte>(data.Length);
        dIn.View.CopyFromCPU(data);
        using var dPacked = acc.Allocate1D<byte>(WeightPackGpu.GetMaxPackedSize(data.Length, elementSize));
        using var dOut = acc.Allocate1D<byte>(data.Length);

        // Warm-up (kernel compile) then timed repetitions.
        int n = await pack.PackAsync(dIn.View, data.Length, elementSize, dPacked.View);
        await pack.UnpackAsync(dPacked.View, n, dOut.View);
        double packBest = double.MaxValue, unpackBest = double.MaxValue;
        for (int r = 0; r < reps; r++)
        {
            var sw = Stopwatch.StartNew();
            n = await pack.PackAsync(dIn.View, data.Length, elementSize, dPacked.View);
            await acc.SynchronizeAsync();
            packBest = Math.Min(packBest, sw.Elapsed.TotalSeconds);
            sw.Restart();
            await pack.UnpackAsync(dPacked.View, n, dOut.View);
            await acc.SynchronizeAsync();
            unpackBest = Math.Min(unpackBest, sw.Elapsed.TotalSeconds);
        }
        var back = await dOut.View.CopyToHostAsync();
        bool exact = back.AsSpan().SequenceEqual(data);
        double ratio = (double)n / data.Length;
        var record = new
        {
            kind = "bench",
            repo = "SpawnDev.Codecs",
            config = $"weightpack-{backend}-{Path.GetFileNameWithoutExtension(path)}",
            file = Path.GetFileName(path),
            dtype,
            sampled_bytes = data.Length,
            packed_bytes = n,
            ratio = Math.Round(ratio, 4),
            pack_seconds = Math.Round(packBest, 3),
            unpack_seconds = Math.Round(unpackBest, 3),
            pack_gbps = Math.Round(data.Length / packBest / 1e9, 3),
            unpack_gbps = Math.Round(data.Length / unpackBest / 1e9, 3),
            roundtrip_exact = exact,
            accelerator = acc.Name,
        };
        Console.WriteLine(JsonSerializer.Serialize(record));
        return exact ? 0 : 1;
    }

    /// <summary>Tensors of the file's main dtype, every (count / 64)th in file order, while the sample stays under maxMb.</summary>
    private static (byte[] data, string dtype, int elementSize, int tensors) Sample(string path, int maxMb)
    {
        using var f = File.OpenRead(path);
        var lenBytes = new byte[8];
        f.ReadExactly(lenBytes);
        long headerLength = BitConverter.ToInt64(lenBytes);
        var headerBytes = new byte[headerLength];
        f.ReadExactly(headerBytes);
        long dataBase = 8 + headerLength;
        using var doc = JsonDocument.Parse(headerBytes);
        var list = new List<(string dtype, long a, long b)>();
        foreach (var p in doc.RootElement.EnumerateObject())
        {
            if (p.Name == "__metadata__") continue;
            var offs = p.Value.GetProperty("data_offsets");
            list.Add((p.Value.GetProperty("dtype").GetString()!, offs[0].GetInt64(), offs[1].GetInt64()));
        }
        list.Sort((x, y) => x.a.CompareTo(y.a));
        var bytesByType = new Dictionary<string, long>();
        foreach (var t in list) bytesByType[t.dtype] = bytesByType.GetValueOrDefault(t.dtype) + (t.b - t.a);
        string main = bytesByType.MaxBy(kv => kv.Value).Key;
        int elementSize = main switch { "BF16" or "F16" => 2, "F32" => 4, "F64" => 8, "F8_E4M3" or "F8_E5M2" or "I8" or "U8" => 1, _ => 4 };
        var chosen = list.Where(t => t.dtype == main).ToList();
        int step = Math.Max(1, chosen.Count / 64);
        long cap = (long)maxMb * 1_000_000;
        var ms = new MemoryStream();
        int count = 0;
        for (int i = 0; i < chosen.Count; i += step)
        {
            var (_, a, b) = chosen[i];
            if (ms.Length + (b - a) > cap) continue;
            f.Seek(dataBase + a, SeekOrigin.Begin);
            var buf = new byte[b - a];
            f.ReadExactly(buf);
            ms.Write(buf);
            count++;
        }
        return (ms.ToArray(), main, elementSize, count);
    }
}
