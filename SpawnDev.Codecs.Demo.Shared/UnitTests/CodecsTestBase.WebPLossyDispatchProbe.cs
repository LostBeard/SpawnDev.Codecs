// Probe (2026-11-23): what a lossy WebP frame costs in launches on a browser backend, and whether its bytes are stable.
// Lore measured Codecs 1.1.4's Save Animated WEBP (lossy q90, 512x352) at 1.28 s per frame in Chrome WebGPU on an RX 590
// against ~175 ms on CUDA here. The RD loop launches 5 small stage kernels per macroblock diagonal (~1,400 dispatches per
// frame), each paced by LaunchPacer (a submit every FlushEvery launches, a wait for the GPU every WaitEvery). For a
// 512x352 picture this reports, per configuration: the time of 3 warm encodes, the dispatches batched by SpawnDev.ILGPU
// (WebGPUBackend.ScalarArenaSlotsUsed grows by one per batched dispatch) and the size and hash of EVERY encode, then
// asserts they are all the same file. On the RX 590 (Lore, 7d8377a) repeated encodes at the same pacing differed, so the
// configurations include SerialWavefront (a diagonal's macroblocks one launch each): stable there but not in the parallel
// runs = a race between the work-groups of a diagonal; unstable in both = inside a group (or a miscompile).

using System.Diagnostics;
using SpawnDev.Codecs.Images.WebP.Lossy;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task Probe_WebPLossy_DispatchCost()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, AtomicKernels);
        const int w = 512, h = 352;
        var rgba = WebPTokenizeStressPicture(4, w, h);
        for (int i = 3; i < rgba.Length; i += 4) rgba[i] = 255;
        var options = new WebPLossyEncodeOptions { Quality = 90, Method = 4 };
        var reports = new List<string>();
        var files = new List<(string Config, byte[] File)>();
        try
        {
            using var enc = new WebPLossyEncoderGpu(acc);
            foreach (var (flush, wait, serial) in new (int, int, bool)[] { (8, 32, false), (64, 256, false), (128, 1024, false), (64, 256, true) })
            {
                string config = $"{flush}/{wait}{(serial ? " serial" : "")}";
                LaunchPacer.FlushEveryOverride = flush;
                LaunchPacer.WaitEveryOverride = wait;
                WebPLossyEncoderGpu.SerialWavefront = serial;
                try
                {
                    var hashes = new List<string>();
                    byte[] file = await enc.EncodeAsync(rgba, w, h, options);   // warm
                    files.Add((config + " warm", file));
                    hashes.Add(Describe(file));
                    long slots0 = ArenaSlots();
                    const int runs = 3;
                    var sw = Stopwatch.StartNew();
                    for (int r = 0; r < runs; r++)
                    {
                        file = await enc.EncodeAsync(rgba, w, h, options);
                        files.Add((config + " run " + (r + 1), file));
                        hashes.Add(Describe(file));
                    }
                    double ms = sw.Elapsed.TotalMilliseconds / runs;
                    long slots = (ArenaSlots() - slots0) / runs;
                    reports.Add($"{config}: {ms:F0} ms per encode, {slots} batched dispatches per encode, files {string.Join(" ", hashes)}");
                }
                finally
                {
                    LaunchPacer.FlushEveryOverride = null;
                    LaunchPacer.WaitEveryOverride = null;
                    WebPLossyEncoderGpu.SerialWavefront = false;
                }
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
        Console.WriteLine($"[report] {acc.AcceleratorType} lossy WebP 512x352 q90 m4: " + string.Join("; ", reports));
        var first = files[0];
        foreach (var (config, file) in files)
            True(file.AsSpan().SequenceEqual(first.File), $"'{config}' wrote a different file than '{first.Config}' ({Describe(file)} vs {Describe(first.File)})");

        long ArenaSlots() => acc is SpawnDev.ILGPU.WebGPU.WebGPUAccelerator ? SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.ScalarArenaSlotsUsed[0] : 0;

        static string Describe(byte[] f)
        {
            ulong x = 14695981039346656037UL;   // FNV-1a 64
            foreach (byte b in f) { x ^= b; x *= 1099511628211UL; }
            return $"{f.Length}B:{x:x16}";
        }
    }
}
