// The predictor-mode choice in two kernels (Vp8lPredictorChoose.HistKernel: every tile of a window at once, then
// WalkKernel: the tiles in order) must make ChooseKernel's choices (the one-kernel walk, kept as the oracle): the same mode
// image and the same accumulated histograms, bit for bit, because the walk computes each cost term exactly as ChooseKernel
// does. Pictures: dense transparency over a few tiles, the cost-oracle picture (opaque tile column = the plain residual
// path, exact ties decided by the neighbour bias), and pictures past one window (128 tiles) up to 1920x1080; both fill
// walks (wavefront, raster order); the predictor off. Then the whole encoder, one kernel vs two, the same files.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Images.WebP;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static uint ChooseTestHash(int k)
    {
        uint v = (uint)k * 0x9E3779B1u;
        v ^= v >> 15; v *= 0x85EBCA77u; v ^= v >> 13; v *= 0xC2B2AE3Du; v ^= v >> 16;
        return v;
    }

    private static uint[] ChooseTestPicture(int kind, int w, int h)
    {
        var argb = new uint[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                uint rgb;
                bool transparent;
                if (kind == 0)
                {
                    rgb = ((uint)Vp9TestTexture(x, y) << 16) | ((uint)Vp9TestTexture(x + 300, y) << 8) | Vp9TestTexture(x, y + 300);
                    transparent = ((x * 3 + y * 5) % 7) < 3 || (x % 9 == 4) || (y % 11 == 6);
                }
                else
                {
                    // tiles built for different predictors (as the cost-oracle test), flat tiles (exact ties), noise
                    int tx = x >> 4, ty = y >> 4;
                    int pattern = (tx * 3 + ty * 2) % 7;
                    int key = pattern switch { 0 => x, 1 => y + 1000, 2 => x - y + 2000, 3 => x + y + 3000, _ => y * w + x + 5000 };
                    rgb = ChooseTestHash(key) & 0x00ffffffu;
                    if (pattern == 5) rgb = ChooseTestHash(tx * 31 + ty + 9000) & 0x00ffffffu;
                    else if (pattern == 6) rgb = 0x00406080u + (ChooseTestHash(key) & 0x00030303u);
                    // some tile columns opaque (the plain residual path); elsewhere runs of transparency, sparser for kind 2
                    transparent = tx % 5 != 1 && (kind == 1 ? (((x * 3 + y * 5) % 7) < 2 || (y % 11 == 6)) : ((x * 7 + y * 3) % 23) == 0);
                }
                argb[y * w + x] = (transparent ? 0u : 0xff000000u) | rgb;
            }
        return argb;
    }

    [TestMethod(Timeout = 1200000, Category = "NotOnWebGL")]
    public async Task Vp8lPredictorChoose_TwoKernels_SameAsOneKernel()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, AtomicKernels);
        try
        {
            var choose = acc.LoadStreamKernel<ArrayView<uint>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(Vp8lPredictorChoose.ChooseKernel);
            var hist = acc.LoadStreamKernel<ArrayView<uint>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(Vp8lPredictorChoose.HistKernel);
            var walk = acc.LoadStreamKernel<ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(Vp8lPredictorChoose.WalkKernel);
            var clear = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(Vp8lPredictorChoose.ClearKernel);
            int groupSize = Vp8lPredictorChoose.GroupSizeFor(acc);
            using var dPc = acc.Allocate1D<int>(Vp8lPredictorChoose.Total);
            using var dHist = acc.Allocate1D<int>(Vp8lPredictorChoose.WTotal);
            using var dFill = acc.Allocate1D<int>(Vp8lPredictorChoose.FillTotal);
            var failures = new List<string>();
            int checkedTiles = 0, windowsSeen = 0;

            var cases = new List<(string Name, int Kind, int W, int H, int Bits, bool UsePredictor)>
            {
                ("dense transparency 40x36", 0, 40, 36, 4, true),
                ("cost-oracle pattern 120x76", 1, 120, 76, 4, true),
                ("cost-oracle pattern 400x352 (550 tiles, 5 windows)", 1, 400, 352, 4, true),
                ("sparse transparency 333x250 bits 3 (1344 tiles)", 2, 333, 250, 3, true),
                ("sparse transparency 512x352 bits 5", 2, 512, 352, 5, true),
                ("predictor off 120x76", 1, 120, 76, 4, false),
            };
            // the production size: CUDA, OpenCL and Velocity. Not on the CPU accelerator (the debug backend, correctness only):
            // there the one-kernel oracle walks 8,160 tiles in one work-group of real threads, ~12 of the test's 12.5 minutes
            // (2026-11-18); the cases above already take it past one window. Not in the browser (time).
            if (!OperatingSystem.IsBrowser() && acc.AcceleratorType != AcceleratorType.CPU)
                cases.Add(("dense transparency 1920x1080 (8160 tiles)", 0, 1920, 1080, 4, true));

            foreach (var (name, kind, w, h, bits, usePredictor) in cases)
            {
                int side = 1 << bits;
                int tiles = ((w + side - 1) >> bits) * ((h + side - 1) >> bits);
                var meta = new int[WebPEncMeta.Size];
                meta[WebPEncMeta.UsePredictor] = usePredictor ? 1 : 0;
                meta[WebPEncMeta.PackedWidth] = w;
                meta[WebPEncMeta.Height] = h;
                meta[WebPEncMeta.PredictorBits] = bits;
                meta[WebPEncMeta.RegionPredictor] = 0;
                meta[WebPEncMeta.Transparent] = 1;
                meta[WebPEncMeta.PaletteCount] = 0;
                using var dArgb = acc.Allocate1D(ChooseTestPicture(kind, w, h));
                using var dMeta = acc.Allocate1D(meta);
                using var dW = acc.Allocate1D<int>(tiles);
                foreach (int options in new[] { 1, 3 })
                {
                    if (options == 3 && tiles > 2000) continue;   // the raster-order walk is one thread per mode: small pictures
                    string tag = $"{name} options {options}";
                    // the oracle
                    dW.View.MemSetToZero();
                    clear(Vp8lPredictorChoose.Res, dPc.View);
                    for (int first = 0; first < tiles; first += Vp8lPredictorChoose.TilesPerLaunch)
                        choose(new KernelConfig(1, groupSize), dArgb.View, dW.View, dMeta.View, dPc.View, options, first);
                    await acc.SynchronizeAsync();
                    int[] wantModes = await dW.View.CopyToHostAsync();
                    int[] wantAcc = await dPc.View.SubView(0, 1024).CopyToHostAsync();
                    // two kernels; the slots start dirty (a previous case's leftovers must not matter)
                    dW.View.MemSetToZero();
                    clear(Vp8lPredictorChoose.Res, dHist.View);
                    for (int first = 0; first < tiles; first += Vp8lPredictorChoose.WindowTiles)
                    {
                        hist(new KernelConfig(Math.Min(Vp8lPredictorChoose.WindowTiles, tiles - first), groupSize), dArgb.View, dMeta.View, dHist.View, dFill.View, options, first);
                        walk(new KernelConfig(1, groupSize), dW.View, dMeta.View, dHist.View, options, first);
                        windowsSeen++;
                    }
                    await acc.SynchronizeAsync();
                    int[] gotModes = await dW.View.CopyToHostAsync();
                    int[] gotAcc = await dHist.View.SubView(0, 1024).CopyToHostAsync();
                    for (int t = 0; t < tiles; t++)
                        if (gotModes[t] != wantModes[t]) { failures.Add($"{tag}: tile {t} mode {(gotModes[t] >> 8) & 0xFF}, one kernel {(wantModes[t] >> 8) & 0xFF}"); break; }
                    for (int b = 0; b < 1024; b++)
                        if (gotAcc[b] != wantAcc[b]) { failures.Add($"{tag}: accumulated bin {b} = {gotAcc[b]}, one kernel {wantAcc[b]}"); break; }
                    if (usePredictor && wantAcc.Sum() == 0) failures.Add($"{tag}: the oracle accumulated nothing (the case does not exercise the choice)");
                    if (!usePredictor && gotModes.Any(v => v != 0)) failures.Add($"{tag}: predictor off, but modes were written");
                    checkedTiles += tiles;
                }
            }

            // the whole encoder: the same file with either form
            using var enc = new WebPLosslessEncoderGpu(acc);
            var pics = new List<(string, byte[], int, int)> { ("gradient + transparency 512x352", WebPTokenizeStressPicture(3, 512, 352), 512, 352) };
            var rgba = new byte[400 * 352 * 4];
            var px = ChooseTestPicture(1, 400, 352);
            for (int i = 0; i < px.Length; i++)
            {
                rgba[i * 4] = (byte)(px[i] >> 16); rgba[i * 4 + 1] = (byte)(px[i] >> 8); rgba[i * 4 + 2] = (byte)px[i]; rgba[i * 4 + 3] = (byte)(px[i] >> 24);
            }
            pics.Add(("cost-oracle pattern 400x352", rgba, 400, 352));
            foreach (var (name, pic, w, h) in pics)
                foreach (int q in new[] { 75, 95 })
                {
                    var o = new WebPEncodeOptions { Quality = q, Exact = false };
                    WebPEncodeKernels.SinglePassChooser = true;
                    byte[] one = await enc.EncodeAsync(pic, w, h, o);
                    WebPEncodeKernels.SinglePassChooser = false;
                    byte[] two = await enc.EncodeAsync(pic, w, h, o);
                    if (!one.AsSpan().SequenceEqual(two)) failures.Add($"{name} q{q}: file differs (two kernels {two.Length} B, one kernel {one.Length} B)");
                }
            True(failures.Count == 0, $"{failures.Count} failures ({checkedTiles} tiles, {windowsSeen} windows):\n" + string.Join("\n", failures.Take(30)));
        }
        finally
        {
            WebPEncodeKernels.SinglePassChooser = false;
            ReleaseAccelerator(ctx, acc);
        }
    }
}
