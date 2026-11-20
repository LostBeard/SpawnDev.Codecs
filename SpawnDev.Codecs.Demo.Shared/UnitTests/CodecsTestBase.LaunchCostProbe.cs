// A measurement, not a check: what one kernel dispatch costs on this accelerator, next to what a lossy WebP frame costs.
// The lossy encoder's RD loop launches 5 one-macroblock kernels per macroblock (~17,600 dispatches for a 512x352 frame);
// SpawnGen's site measured ~4.8 s per frame in Chrome (WebGPU / D3D12) on an RX 590 (Lore, 2026-11-19), while CUDA is
// ~0.5 s. The report gives the dispatch cost (an empty one-thread kernel, and the lossy encoder's own RecordStage-sized
// launches are of the same kind) and a lossy frame's time, so the share of dispatch overhead can be read off. It passes
// whenever it runs: the numbers go to the report line (PlaywrightMultiTest / the desktop runner print it).

using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Images.WebP.Lossy;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static void LaunchProbeKernel(Index1D i, ArrayView<int> a, int v)
    {
        if (i == 0) a[0] = a[0] + v;
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task Probe_DispatchCost_AndLossyFrameTime()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, AtomicKernels);
        try
        {
            var k = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int>(LaunchProbeKernel);
            using var buf = acc.Allocate1D<int>(1);
            buf.View.MemSetToZero();
            k(1, buf.View, 1);
            await acc.SynchronizeAsync();   // compiled and warm
            const int launches = 4000;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < launches; i++) k(1, buf.View, 1);
            await acc.SynchronizeAsync();
            double perLaunchUs = sw.Elapsed.TotalMilliseconds * 1000.0 / launches;
            int got = (await buf.View.CopyToHostAsync())[0];

            // a lossy frame like SpawnGen's (512x352, q90, method 4), warm (second encode timed)
            const int w = 512, h = 352;
            var rgba = WebPTokenizeStressPicture(4, w, h);
            for (int i = 3; i < rgba.Length; i += 4) rgba[i] = 255;
            using var enc = new WebPLossyEncoderGpu(acc);
            var o = new WebPLossyEncodeOptions { Quality = 90, Method = 4 };
            await enc.EncodeAsync(rgba, w, h, o);
            sw.Restart();
            byte[] file = await enc.EncodeAsync(rgba, w, h, o);
            double frameMs = sw.Elapsed.TotalMilliseconds;
            int mbs = ((w + 15) >> 4) * ((h + 15) >> 4);
            double rdDispatches = 5.0 * mbs;
            Console.WriteLine($"[report] {acc.AcceleratorType}: empty dispatch {perLaunchUs:F1} us ({launches} launches, sum {got - 1}); " +
                $"lossy 512x352 q90 frame {frameMs:F0} ms ({file.Length} B); RD-loop dispatches {rdDispatches:F0} x {perLaunchUs:F1} us = " +
                $"{rdDispatches * perLaunchUs / 1000.0:F0} ms ({100.0 * rdDispatches * perLaunchUs / 1000.0 / frameMs:F0}% of the frame)");
            Equal(launches + 1, got, "every probe launch ran");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
