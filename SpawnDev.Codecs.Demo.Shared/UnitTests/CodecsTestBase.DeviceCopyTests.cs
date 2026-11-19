// Device-to-device copies between sub-views of int buffers, the pattern WebPEncodeKernels.SaveMatchesAsync /
// RestoreMatches use (a 3n-int block at an offset of one buffer to the start of another, and back). Guards the
// SpawnDev.ILGPU copy path on every accelerator (2026-11-19: on Wasm the target offset arrived as garbage, e.g.
// 7 * 2^32, and the copy threw "Target copy range ... exceeds target buffer size").

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static void DeviceCopyFillKernel(Index1D i, ArrayView<int> a, int salt) => a[i] = i * 7 + 1 + salt;

    /// <summary>
    /// The same copies issued while kernels are still queued (no SynchronizeAsync between the launch that writes the source
    /// and the copy), the way the encoder issues them: the browser backends take their ordered copy path then.
    /// </summary>
    [TestMethod(Timeout = 120000)]
    public async Task DeviceCopy_IntSubViews_AfterQueuedKernel()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var fill = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int>(DeviceCopyFillKernel);
            foreach (int n in new[] { 37, 4800, 20000 })
            {
                long off = 2L * n, len = 3L * n;
                using var a = acc.Allocate1D<int>(5L * n);
                fill(5 * n, a.View, 0);                                // queued, not awaited
                using var b = acc.Allocate1D<int>(len);
                a.View.SubView(off, len).CopyTo(b.View.SubView(0, len));
                fill(5 * n, a.View, 1000);                             // rewrites the source after the copy, still queued
                b.View.SubView(0, len).CopyTo(a.View.SubView(off, len)); // and back over it
                await acc.SynchronizeAsync();
                var got = await a.View.CopyToHostAsync();
                for (int i = 0; i < 5 * n; i++)
                {
                    int want = (i >= off && i < off + len) ? i * 7 + 1 : i * 7 + 1 + 1000;
                    if (got[i] != want) throw new Exception($"n {n}: a[{i}] = {got[i]}, want {want}");
                }
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    // the encoder's shape: the copy inside an async method, after an await (WebPEncodeKernels.SaveMatchesAsync)
    private static async ValueTask DeviceCopyAfterAwaitAsync(Accelerator acc, MemoryBuffer1D<int, Stride1D.Dense> src,
        MemoryBuffer1D<int, Stride1D.Dense> dst, long off, long len)
    {
        await acc.SynchronizeAsync();
        src.View.SubView(off, len).CopyTo(dst.View.SubView(0, len));
    }

    /// <summary>The copy issued from an async method after an await, many times (hot), as SaveMatchesAsync does.</summary>
    [TestMethod(Timeout = 300000)]
    public async Task DeviceCopy_IntSubViews_InAsyncMethod_Hot()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var fill = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int>(DeviceCopyFillKernel);
            int n = 6912;
            long off = 2L * n, len = 3L * n;
            using var a = acc.Allocate1D<int>(5L * n);
            using var b = acc.Allocate1D<int>(len);
            for (int iter = 0; iter < 300; iter++)
            {
                fill(5 * n, a.View, iter);
                await DeviceCopyAfterAwaitAsync(acc, a, b, off, len);
                if (iter % 50 == 0 || iter == 299)
                {
                    await acc.SynchronizeAsync();
                    var got = await b.View.CopyToHostAsync();
                    for (long i = 0; i < len; i++)
                        if (got[i] != (int)((off + i) * 7 + 1 + iter)) throw new Exception($"iter {iter}: b[{i}] = {got[i]}, want {(off + i) * 7 + 1 + iter}");
                }
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 120000)]
    public async Task DeviceCopy_IntSubViews_ToAndFromAnotherBuffer()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            foreach (int n in new[] { 37, 4800, 20000 })
            {
                long off = 2L * n, len = 3L * n;
                var src = new int[5 * n];
                for (int i = 0; i < src.Length; i++) src[i] = i * 7 + 1;
                using var a = acc.Allocate1D(src);
                using var b = acc.Allocate1D<int>(len);
                // a[off .. off + len) -> b[0 .. len)
                a.View.SubView(off, len).CopyTo(b.View.SubView(0, len));
                await acc.SynchronizeAsync();
                var got = await b.View.CopyToHostAsync();
                for (long i = 0; i < len; i++)
                    if (got[i] != src[off + i]) throw new Exception($"n {n}: b[{i}] = {got[i]}, want {src[off + i]}");
                // and back: b[0 .. len) -> a[off .. off + len) after clearing that range in a
                a.View.SubView(off, len).CopyFromCPU(new int[len]);
                b.View.SubView(0, len).CopyTo(a.View.SubView(off, len));
                await acc.SynchronizeAsync();
                var back = await a.View.CopyToHostAsync();
                for (int i = 0; i < src.Length; i++)
                    if (back[i] != src[i]) throw new Exception($"n {n}: a[{i}] = {back[i]} after the copy back, want {src[i]}");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
