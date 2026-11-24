// Probe (2026-11-23): the fixed cost of a small dispatch on this accelerator, with no work in it. The lossy WebP RD loop
// launches ~1,400 tiny grouped dispatches per 512x352 frame (5 stages per macroblock diagonal, 1-4 groups each). On the
// RTX 4070 in Chrome WebGPU a frame takes ~188 ms (CUDA ~175 ms), on an RX 590 in Chrome 1.28 s (Lore). WebGPU gives
// every dispatch its own usage scope, so D3D12 places a barrier between them; this measures what that costs per dispatch
// on the GPU at hand: N launches of a near-empty kernel shaped like the RD stages (4 groups of 32 lanes, a barrier, one
// store), paced like the RD loop (LaunchPacer 64 / 256), timed after a warm-up. If N x this is most of a frame on a GPU,
// fewer, fatter dispatches are the fix there.

using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static void DispatchCostKernel(ArrayView<int> sink, int step)
    {
        int lane = Group.IdxX;
        Group.Barrier();
        if (lane == 0) sink[Grid.IdxX] = step;
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task Probe_DispatchCost_TinyGroupedLaunches()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var kernel = await acc.LoadStreamKernelAsync<ArrayView<int>, int>(DispatchCostKernel);
            using var sink = acc.Allocate1D<int>(4);
            const int launches = 1400;
            async Task<double> RunAsync()
            {
                var pacer = new LaunchPacer(acc, 64, 256);
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < launches; i++)
                {
                    kernel(new KernelConfig(4, 32), sink.View, i);
                    await pacer.LaunchedAsync();
                }
                await acc.SynchronizeAsync();
                return sw.Elapsed.TotalMilliseconds;
            }
            await RunAsync();   // warm (compile, first submits)
            double a = await RunAsync(), b = await RunAsync();
            var last = await sink.View.CopyToHostAsync();
            Equal(launches - 1, last[3], "the last launch's store");
            double ms = Math.Min(a, b);
            Console.WriteLine($"[report] {acc.AcceleratorType} ({acc.Name}): {launches} tiny grouped launches (4 x 32 lanes, paced 64/256) " +
                $"in {ms:F1} ms = {1000 * ms / launches:F1} us per launch (runs {a:F1} / {b:F1} ms)");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
