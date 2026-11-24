// Probe (2026-11-23): what a lossy WebP frame costs in launches on a browser backend. Lore measured Codecs 1.1.4's Save
// Animated WEBP (lossy q90, 512x352) at 1.28 s per frame in Chrome WebGPU on an RX 590 against ~175 ms on CUDA here. The
// RD loop launches 5 small stage kernels per macroblock diagonal (~1,800 dispatches per frame), each paced by LaunchPacer
// (a submit every FlushEvery launches, a wait for the GPU every WaitEvery). This reports, for one warm encode of a
// 512x352 picture: the time, the dispatches batched by SpawnDev.ILGPU (WebGPUBackend.ScalarArenaSlotsUsed grows by one
// per batched dispatch; 0 = every dispatch took the slow per-dispatch path) and the same with looser pacing, checking the
// bytes are the same in every configuration.

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
        byte[]? reference = null;
        try
        {
            using var enc = new WebPLossyEncoderGpu(acc);
            foreach (var (flush, wait) in new (int, int)[] { (8, 32), (64, 256), (128, 1024) })
            {
                LaunchPacer.FlushEveryOverride = flush;
                LaunchPacer.WaitEveryOverride = wait;
                try
                {
                    byte[] file = await enc.EncodeAsync(rgba, w, h, options);   // warm
                    reference ??= file;
                    True(file.AsSpan().SequenceEqual(reference), $"pacing {flush}/{wait} wrote different bytes");
                    long slots0 = ArenaSlots();
                    const int runs = 3;
                    var sw = Stopwatch.StartNew();
                    for (int r = 0; r < runs; r++) file = await enc.EncodeAsync(rgba, w, h, options);
                    double ms = sw.Elapsed.TotalMilliseconds / runs;
                    long slots = (ArenaSlots() - slots0) / runs;
                    True(file.AsSpan().SequenceEqual(reference), $"pacing {flush}/{wait} wrote different bytes");
                    reports.Add($"flush {flush} / wait {wait}: {ms:F0} ms per encode, {slots} batched dispatches per encode");
                }
                finally
                {
                    LaunchPacer.FlushEveryOverride = null;
                    LaunchPacer.WaitEveryOverride = null;
                }
            }
            Console.WriteLine($"[report] {acc.AcceleratorType} lossy WebP 512x352 q90 m4 ({reference!.Length} B): " + string.Join("; ", reports));
        }
        finally { ReleaseAccelerator(ctx, acc); }

        long ArenaSlots() => acc is SpawnDev.ILGPU.WebGPU.WebGPUAccelerator ? SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.ScalarArenaSlotsUsed[0] : 0;
    }
}
