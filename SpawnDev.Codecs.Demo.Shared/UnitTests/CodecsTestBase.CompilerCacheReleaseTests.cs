// What KernelLoadBatch's ReleaseCompilerCaches saves and costs, measured on the codec that made it necessary: the lossy
// WebP encoder (about 20 kernels; SpawnGen's animated WebP export). Data measured the ILGPU Context's compiler cache
// growing +20.1 MiB (Wasm) / +28.4 MiB (WebGPU) for this encoder + 3 encodes, 16.1 / 19.6 MiB of it released by
// ReleaseCompilerCaches (2026-11-20). In a browser the Context lives as long as the page, so that cache is what ran the
// .NET heap out in long sessions. This test runs the same encoder on fresh Contexts with the release off and on
// (KernelLoadBatch.ReleaseEnabled, tests only), with SpawnDev.ILGPU's shader-artifact cache off so every run builds the
// kernels' IR (that cache lets a page's later loads skip IR entirely), alternating the order anyway, and checks:
// - the release leaves the encoded bytes unchanged (same file in every run),
// - the heap that stays alive with a warmed-up encoder is smaller with the release by at least MinFreedMiB,
// and reports the warm-up times of both modes (the release's cost: the re-import of shared code by later loads).

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
    /// <summary>The heap the release must save for a lossy WebP encoder with 3 encodes behind it.</summary>
    private const double MinFreedMiB = 8;

    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL")]
    public async Task Codecs_CompilerCacheRelease_LossyWebP_SmallerHeap_SameBytes()
    {
        const int w = 512, h = 352;
        var rgba = WebPTokenizeStressPicture(4, w, h);
        for (int i = 3; i < rgba.Length; i += 4) rgba[i] = 255;
        var options = new WebPLossyEncodeOptions { Quality = 90, Method = 4 };

        async Task<(double HeapMiB, double WarmUpMs, int Releases, byte[] File)> Run(bool release)
        {
            (Context ctx, Accelerator acc) fresh;
            try { fresh = await CreateKernelAcceleratorAsync(); }
            catch (NotSupportedException ex) { throw new UnsupportedTestException(ex.Message); }
            var (ctx, acc) = fresh;
            KernelLoadBatch.ReleaseEnabled = release;
            // Every run compiles from IL. WebGPU's page-wide shader-artifact cache otherwise serves a kernel loaded
            // before (by an earlier run, or an earlier test in the page) without building its IR at all, so nothing
            // reaches the Context's cache whatever the mode (measured 11-22: WebGPU "off" 41.3 MiB cold, 1.4 MiB second).
            bool artifactCache = ShaderArtifactCache.Enabled;
            ShaderArtifactCache.Enabled = false;
            try
            {
                if (!acc.Device.Satisfies(AtomicKernels)) throw new UnsupportedTestException("the lossy WebP encoder needs atomics");
                long before = GC.GetTotalMemory(true);
                using var enc = new WebPLossyEncoderGpu(acc);
                var sw = Stopwatch.StartNew();
                await enc.WarmUpAsync();
                double warmUpMs = sw.Elapsed.TotalMilliseconds;
                byte[] file = null!;
                for (int i = 0; i < 3; i++) file = await enc.EncodeAsync(rgba, w, h, options);
                // the encoder (and so its kernels) stays alive: what is measured is what a page holds while it encodes
                long after = GC.GetTotalMemory(true);
                GC.KeepAlive(enc);
                return ((after - before) / 1048576.0, warmUpMs, KernelLoadBatch.ReleaseCount(ctx), file);
            }
            finally
            {
                ShaderArtifactCache.Enabled = artifactCache;
                KernelLoadBatch.ReleaseEnabled = true;
                acc.Dispose();
                ctx.Dispose();
            }
        }

        var off1 = await Run(false);
        var on1 = await Run(true);
        var on2 = await Run(true);
        var off2 = await Run(false);

        Equal(0, off1.Releases, "releases with the release switched off");
        True(on1.Releases > 0 && on2.Releases > 0, $"no release with the release switched on ({on1.Releases}, {on2.Releases})");
        foreach (var (name, r) in new[] { ("off 2", off2), ("on 1", on1), ("on 2", on2) })
            True(r.File.AsSpan().SequenceEqual(off1.File), $"run '{name}' wrote different bytes than run 'off 1' ({r.File.Length} vs {off1.File.Length} B)");

        double offHeap = (off1.HeapMiB + off2.HeapMiB) / 2, onHeap = (on1.HeapMiB + on2.HeapMiB) / 2;
        string report = $"{AcceleratorTypeName()}: lossy WebP encoder + 3 encodes 512x352 q90 m4, live heap growth MiB: " +
            $"release off {off1.HeapMiB:F1} / {off2.HeapMiB:F1}, on {on1.HeapMiB:F1} / {on2.HeapMiB:F1} (saved {offHeap - onHeap:F1}); " +
            $"warm-up ms (run order off, on, on, off): {off1.WarmUpMs:F0}, {on1.WarmUpMs:F0}, {on2.WarmUpMs:F0}, {off2.WarmUpMs:F0}; " +
            $"releases per run {on1.Releases}; file {off1.File.Length} B";
        Console.WriteLine("[report] " + report);
        True(offHeap - onHeap >= MinFreedMiB, $"the release saved {offHeap - onHeap:F1} MiB, expected at least {MinFreedMiB}. " + report);

        string AcceleratorTypeName() => GetType().Name.Replace("CodecsTests", "");
    }
}
