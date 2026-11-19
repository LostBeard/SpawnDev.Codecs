// The parallel VP8L colour analysis (AnalyzeInsertKernel + count / rank / delta) must give what the serial scan (the
// encoder before 2026-11-18, kept as WebPEncodeKernels.SerialAnalysis) gives, and what a host oracle computes from the
// pixels: HasAlpha, PaletteAvailable (the number of distinct ARGB values when at most 256, else 0), the palette sorted
// ascending as unsigned values, and its delta-coded form. Cases at the edges: exactly 256 and 257 colours (with and
// without the colour 0, which the parallel set keeps apart from its table), colours with the top bit set (unsigned
// order), one colour, one pixel, every pixel 0, noise (thousands of colours), a picture past one launch with few colours
// (hash contention), and the palette switched off.
// Then the whole encoder, serial vs parallel analysis, on every WebP source fixture: the same file, byte for byte.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.Codecs.Images.WebP;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static (string Name, uint[] Argb, int W, int H, bool AllowPalette)[] WebPAnalysisCases()
    {
        var cases = new List<(string, uint[], int, int, bool)>();
        uint[] Fill(int w, int h, Func<int, int, uint> f)
        {
            var a = new uint[w * h];
            for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) a[y * w + x] = f(x, y);
            return a;
        }
        // k distinct colours spread over the picture, with or without 0, some with the top bit set
        uint[] Distinct(int w, int h, int k, bool withZero, uint alphaBits)
        {
            var colours = new uint[k];
            for (int i = 0; i < k; i++) colours[i] = alphaBits | (uint)(i * 0x010307 + 0x20);
            if (withZero) colours[k / 2] = 0u;
            return Fill(w, h, (x, y) => colours[(x * 7 + y * 13) % k]);
        }
        cases.Add(("256 colours, opaque", Distinct(64, 48, 256, false, 0xFF000000u), 64, 48, true));
        cases.Add(("257 colours, opaque", Distinct(64, 48, 257, false, 0xFF000000u), 64, 48, true));
        cases.Add(("256 colours with 0", Distinct(64, 48, 256, true, 0xFF000000u), 64, 48, true));
        cases.Add(("257 colours with 0", Distinct(64, 48, 257, true, 0xFF000000u), 64, 48, true));
        cases.Add(("200 colours, alpha 0x80 (top bit, unsigned order)", Distinct(50, 40, 200, false, 0x80000000u), 50, 40, true));
        cases.Add(("mixed top bit", Fill(33, 17, (x, y) => (x + y) % 3 == 0 ? 0x7FFFFFFFu : (x + y) % 3 == 1 ? 0x80000000u : 0xFFFFFFFFu), 33, 17, true));
        cases.Add(("one colour", Fill(40, 30, (x, y) => 0xFF336699u), 40, 30, true));
        cases.Add(("one pixel", new[] { 0x12345678u }, 1, 1, true));
        cases.Add(("every pixel 0", new uint[97 * 13], 97, 13, true));
        var rng = new Random(1811);
        cases.Add(("noise 257x129", Fill(257, 129, (x, y) => (uint)rng.Next() | ((uint)rng.Next(2) << 31)), 257, 129, true));
        // past one launch (AnalyzePixelsPerLaunch) with few colours: every thread hits the same few slots
        cases.Add(("12 colours 1920x1080", Fill(1920, 1080, (x, y) => 0xFF000000u | (uint)(((x / 160) * 0x151515) & 0xFFFFFF)), 1920, 1080, true));
        // colours by row: most of them only appear in later launches; and one non-opaque pixel, the very last one
        cases.Add(("12 colours by row 1920x1080", Fill(1920, 1080, (x, y) => 0xFF000000u | (uint)(((y / 90) * 0x151515) & 0xFFFFFF)), 1920, 1080, true));
        cases.Add(("last pixel translucent 640x480", Fill(640, 480, (x, y) => x == 639 & y == 479 ? 0x7F102030u : 0xFF102030u), 640, 480, true));
        cases.Add(("alpha plane (green only) 512x352", Fill(512, 352, (x, y) => 0xFF000000u | ((uint)((x * 3 + y) & 0xFF) << 8)), 512, 352, true));
        cases.Add(("palette off", Distinct(64, 48, 20, true, 0xFF000000u), 64, 48, false));
        return cases.ToArray();
    }

    private static (int HasAlpha, int Palette, uint[] Entries, uint[] Delta) WebPAnalysisOracle(uint[] argb, bool allowPalette)
    {
        int hasAlpha = argb.Any(p => (p >> 24) != 255u) ? 1 : 0;
        var set = new SortedSet<uint>(argb);
        if (!allowPalette || set.Count > 256) return (hasAlpha, 0, Array.Empty<uint>(), Array.Empty<uint>());
        var entries = set.ToArray();
        var delta = new uint[entries.Length];
        for (int i = 0; i < entries.Length; i++)
        {
            if (i == 0) { delta[0] = entries[0]; continue; }
            uint a = entries[i], b = entries[i - 1];
            uint ag = 0x00ff00ffu + (a & 0xff00ff00u) - (b & 0xff00ff00u);
            uint rb = 0xff00ff00u + (a & 0x00ff00ffu) - (b & 0x00ff00ffu);
            delta[i] = (ag & 0xff00ff00u) | (rb & 0x00ff00ffu);
        }
        return (hasAlpha, entries.Length, entries, delta);
    }

    [TestMethod(Timeout = 900000, Category = "NotOnWebGL")]
    public async Task WebPLosslessEncoderGpu_ParallelAnalysis_MatchesSerialScanAndOracle()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, AtomicKernels);
        try
        {
            var failures = new List<string>();
            using (var kernels = new WebPEncodeKernels(acc))
            {
                await kernels.EnsureKernelsAsync();
                using var meta = acc.Allocate1D<int>(WebPEncMeta.Size);
                using var work = acc.Allocate1D<int>(Vp8lEncLayout.FirstDynamic);
                foreach (var (name, argb, w, h, allowPalette) in WebPAnalysisCases())
                {
                    var want = WebPAnalysisOracle(argb, allowPalette);
                    using var dArgb = acc.Allocate1D(argb);
                    foreach (bool serial in new[] { true, false })
                    {
                        string tag = $"{name} [{(serial ? "serial" : "parallel")}]";
                        var hostMeta = new int[WebPEncMeta.Size];
                        hostMeta[WebPEncMeta.Width] = w;
                        hostMeta[WebPEncMeta.Height] = h;
                        hostMeta[WebPEncMeta.AllowPalette] = allowPalette ? 1 : 0;
                        meta.View.CopyFromCPU(hostMeta);
                        // poison the palette region: an entry the analysis should have written and did not shows up
                        work.View.SubView(0, Vp8lEncLayout.Analysis).CopyFromCPU(Enumerable.Repeat(unchecked((int)0xA5A5A5A5u), Vp8lEncLayout.Analysis).ToArray());
                        WebPEncodeKernels.SerialAnalysis = serial;
                        await kernels.AnalyzeAsync(dArgb.View, work.View, meta.View, argb.Length);
                        WebPEncodeKernels.SerialAnalysis = false;
                        await acc.SynchronizeAsync();
                        var m = await meta.View.CopyToHostAsync();
                        var pal = await work.View.SubView(0, Vp8lEncLayout.Analysis).CopyToHostAsync();
                        if (m[WebPEncMeta.HasAlpha] != want.HasAlpha) failures.Add($"{tag}: HasAlpha {m[WebPEncMeta.HasAlpha]}, want {want.HasAlpha}");
                        if (m[WebPEncMeta.PaletteAvailable] != want.Palette) { failures.Add($"{tag}: PaletteAvailable {m[WebPEncMeta.PaletteAvailable]}, want {want.Palette}"); continue; }
                        for (int i = 0; i < want.Palette; i++)
                        {
                            if ((uint)pal[Vp8lEncLayout.Palette + i] != want.Entries[i]) { failures.Add($"{tag}: palette[{i}] {(uint)pal[Vp8lEncLayout.Palette + i]:X8}, want {want.Entries[i]:X8}"); break; }
                            if ((uint)pal[Vp8lEncLayout.PaletteDelta + i] != want.Delta[i]) { failures.Add($"{tag}: delta[{i}] {(uint)pal[Vp8lEncLayout.PaletteDelta + i]:X8}, want {want.Delta[i]:X8}"); break; }
                        }
                    }
                }
            }

            // the whole encoder: the same file with either analysis
            using var enc = new WebPLosslessEncoderGpu(acc);
            foreach (var row in WebPSources())
            {
                var rgba = WebPSource(row.Name);
                foreach (int q in new[] { 30, 80 })
                {
                    var o = new WebPEncodeOptions { Quality = q, Exact = false };
                    WebPEncodeKernels.SerialAnalysis = true;
                    byte[] serial = await enc.EncodeAsync(rgba, row.Width, row.Height, o);
                    WebPEncodeKernels.SerialAnalysis = false;
                    byte[] parallel = await enc.EncodeAsync(rgba, row.Width, row.Height, o);
                    if (!serial.AsSpan().SequenceEqual(parallel)) failures.Add($"{row.Name} q{q}: file differs (parallel {parallel.Length} B, serial {serial.Length} B)");
                }
            }
            True(failures.Count == 0, $"{failures.Count} failures:\n" + string.Join("\n", failures.Take(30)));
        }
        finally
        {
            WebPEncodeKernels.SerialAnalysis = false;
            ReleaseAccelerator(ctx, acc);
        }
    }
}
