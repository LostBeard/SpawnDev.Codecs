// vp9-lf-bench: where the VP9 loop filter's time goes on a backend. Decodes a VP9 file (Vp9DecodePipeline, the path of
// Vp9Decode_EveryFrame_MatchesLibvpx) and, after every decoded frame with a loop filter level above 0, runs the frame's
// loop filter again --reps times as it is (all superblocks filtered) and --reps times with every group inactive (the
// same launches, thread groups and barriers, no superblock filtered), each timed to a synchronize. The inactive time is
// the floor of the wavefront's launches and barriers. The DIFFERENCE is NOT the cost of the filtering on backends where a
// group's threads are real threads (the CPU accelerator): there the filtering overlaps the barrier waits, so active minus
// inactive understates it (2026-11-15: CPU accelerator 985 vs 973 ms/frame read as "12 ms of filtering" and "Velocity's
// filtering 53x slower"; wrong: end to end Velocity V256 379 ms/frame vs CPU accelerator 1287, and a sampling profile put
// ~60% of Velocity's loop filter in group-barrier waits behind lane 0's serial mask build). Compare ACTIVE times across
// backends; profile to attribute. The first frame's loop filter inside the decode is reported apart (first launch).
// Usage: vp9-lf-bench <file.webm|.ivf> [--backend cpu|velocity|velocity256|cuda|opencl] [--frames 20] [--reps 3]
//        [--skip-pass 0|1|2] (1: no vertical-edge pass, 2: no horizontal one - where the time goes; the picture is wrong)
// (velocity = Vector128, the VelocityCodecsTests lane's default; velocity256 = Vector256)

using System.Diagnostics;
using System.Globalization;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Velocity;
using SpawnDev.Codecs.Video.Vp9.Decoder;

namespace SpawnDev.Codecs.DemoConsole;

internal static class Vp9LfBench
{
    private static string F(double v) => v.ToString("F2", CultureInfo.InvariantCulture);

    public static async Task<int> Run(string[] args)
    {
        if (args.Length < 2) { Console.WriteLine("usage: vp9-lf-bench <file> [--backend ...] [--frames 20] [--reps 3]"); return 2; }
        string file = args[1];
        string backend = "velocity";
        int maxFrames = 20, reps = 3, skipPasses = 0;
        for (int i = 2; i < args.Length; i++)
        {
            string a = args[i];
            if (i + 1 >= args.Length) { Console.WriteLine($"missing value for {a}"); return 2; }
            string v = args[++i];
            if (a == "--backend") backend = v;
            else if (a == "--frames") maxFrames = int.Parse(v, CultureInfo.InvariantCulture);
            else if (a == "--reps") reps = int.Parse(v, CultureInfo.InvariantCulture);
            else if (a == "--skip-pass") skipPasses = int.Parse(v, CultureInfo.InvariantCulture);
            else { Console.WriteLine($"unknown option {a}"); return 2; }
        }
        if (maxFrames < 1 || reps < 1) { Console.WriteLine("--frames and --reps must be >= 1"); return 2; }

        using var ctx = backend switch
        {
            "cuda" => Context.Create(b => b.Cuda()),
            "opencl" => Context.Create(b => b.OpenCL()),
            "velocity" => Context.Create(b => b.Velocity(VelocityDeviceType.Vector128)),
            "velocity256" => Context.Create(b => b.Velocity(VelocityDeviceType.Vector256)),
            _ => Context.Create(b => b.CPU()),
        };
        using Accelerator acc = backend switch
        {
            "cuda" => ctx.CreateCudaAccelerator(0),
            "opencl" => ctx.CreateCLAccelerator(0),
            "velocity" or "velocity256" => ctx.CreateVelocityAccelerator(0),
            _ => ctx.CreateCPUAccelerator(0),
        };
        Console.WriteLine($"{acc.AcceleratorType} {acc.Name}: {Path.GetFileName(file)}, up to {maxFrames} filtered frames, {reps} reps each");

        using var pipe = new Vp9DecodePipeline(acc);
        byte[] bytes = File.ReadAllBytes(file);
        double active = 0, inactive = 0, activeMax = 0, inactiveMax = 0;
        int timed = 0, launches = 0;
        double? firstInDecode = null;
        var sw = new Stopwatch();
        foreach (byte[] packet in Vp9Packets.ReadFile(bytes))
        {
            if (timed >= maxFrames) break;
            using var buf = acc.Allocate1D(packet);
            foreach (var (offset, length) in Vp9Packets.Split(packet))
            {
                if (timed >= maxFrames) break;
                bool first = firstInDecode == null;
                if (first) pipe.StageTimes = new Dictionary<string, (double TotalMs, double MaxMs, int Frames)>();
                var h = await pipe.DecodeFrameAsync(packet.AsMemory(offset, length), buf.View, offset);
                await acc.SynchronizeAsync();
                if (first && pipe.StageTimes!.TryGetValue("loop filter", out var lf0)) firstInDecode = lf0.MaxMs;
                pipe.StageTimes = null;
                if (h.ShowExistingFrame || h.LoopFilter.Level == 0) continue;
                for (int pass = 0; pass < 2; pass++)
                {
                    bool off = pass == 1;
                    for (int r = 0; r < reps; r++)
                    {
                        sw.Restart();
                        pipe.RunLoopFilterForBench(h, off, skipPasses);
                        await acc.SynchronizeAsync();
                        double ms = sw.Elapsed.TotalMilliseconds;
                        if (off) { inactive += ms; inactiveMax = Math.Max(inactiveMax, ms); }
                        else { active += ms; activeMax = Math.Max(activeMax, ms); }
                    }
                }
                int sbRows = (h.MiRows + 7) >> 3, sbCols = (h.MiCols + 7) >> 3;
                launches += 2 * (sbRows - 1) + sbCols;
                if (timed == 0) Console.WriteLine($"frame size {h.Width}x{h.Height}: {sbRows} x {sbCols} superblocks, {2 * (sbRows - 1) + sbCols} wavefront launches per frame");
                timed++;
            }
        }
        if (timed == 0) { Console.WriteLine("no frame with a loop filter level above 0"); return 1; }
        int runs = timed * reps;
        double perLaunch = launches / (double)timed;
        Console.WriteLine($"first frame's loop filter inside the decode (first launch): {F(firstInDecode ?? double.NaN)} ms");
        Console.WriteLine($"filtered frames timed: {timed}, {reps} reps each");
        Console.WriteLine($"loop filter, all superblocks: mean {F(active / runs)} ms/frame (worst {F(activeMax)}), {F(active / runs / perLaunch)} ms/launch");
        Console.WriteLine($"loop filter, groups inactive: mean {F(inactive / runs)} ms/frame (worst {F(inactiveMax)}), {F(inactive / runs / perLaunch)} ms/launch");
        Console.WriteLine($"filtering itself (difference): {F((active - inactive) / runs)} ms/frame");
        return 0;
    }
}
