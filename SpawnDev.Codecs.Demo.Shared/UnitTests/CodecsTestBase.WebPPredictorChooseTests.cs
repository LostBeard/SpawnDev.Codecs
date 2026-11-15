// The entropy-based predictor-mode choice (Vp8lPredictorChoose) walks each tile's transparent pixels as a wavefront
// (the pixels with the same x + 2y at once): it must compute exactly what the raster-order walk computes. The kernel
// runs on a picture with dense transparency over several tiles both ways; the mode image, the accumulated residual
// histograms (every tile's chosen residuals) and the last tile's residuals and filled pixels of all 14 modes (chosen or
// not) must be identical.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Images.WebP;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Category = "NotOnWebGL")]
    public async Task Vp8lPredictorChoose_Wavefront_SameAsRasterWalk()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            const int w = 40, h = 36, bits = 4;   // 3 x 3 tiles, partial ones on the right and bottom
            var argb = new uint[w * h];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    uint r = Vp9TestTexture(x, y), g = Vp9TestTexture(x + 300, y), b = Vp9TestTexture(x, y + 300);
                    // dense transparency: runs along rows and columns, so filled pixels feed left, top and top-right
                    bool transparent = ((x * 3 + y * 5) % 7) < 3 || (x % 9 == 4) || (y % 11 == 6);
                    argb[y * w + x] = (transparent ? 0u : 0xff000000u) | (r << 16) | (g << 8) | b;
                }
            int tiles = ((w + 15) >> bits) * ((h + 15) >> bits);
            var meta = new int[WebPEncMeta.Size];
            meta[WebPEncMeta.UsePredictor] = 1;
            meta[WebPEncMeta.PackedWidth] = w;
            meta[WebPEncMeta.Height] = h;
            meta[WebPEncMeta.PredictorBits] = bits;
            meta[WebPEncMeta.RegionPredictor] = 0;
            meta[WebPEncMeta.Transparent] = 1;
            meta[WebPEncMeta.PaletteCount] = 0;

            var choose = acc.LoadStreamKernel<ArrayView<uint>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(Vp8lPredictorChoose.ChooseKernel);
            var clear = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(Vp8lPredictorChoose.ClearKernel);
            using var dArgb = acc.Allocate1D(argb);
            using var dMeta = acc.Allocate1D(meta);
            using var dW = acc.Allocate1D<int>(tiles);
            using var dPc = acc.Allocate1D<int>(Vp8lPredictorChoose.Total);

            async Task<(int[] Modes, int[] Pc)> Run(int options)
            {
                dPc.View.MemSetToZero();
                dW.View.MemSetToZero();
                clear(Vp8lPredictorChoose.Res, dPc.View);
                choose(new KernelConfig(1, Vp8lPredictorChoose.GroupSizeFor(acc)), dArgb.View, dW.View, dMeta.View, dPc.View, options, 0);
                await acc.SynchronizeAsync();
                return (await dW.View.CopyToHostAsync(), await dPc.View.CopyToHostAsync());
            }

            var walked = await Run(1 | 2);   // fill, raster walk
            var wave = await Run(1);         // fill, wavefront
            var failures = new List<string>();
            for (int t = 0; t < tiles; t++)
                if (walked.Modes[t] != wave.Modes[t]) failures.Add($"tile {t}: mode {(wave.Modes[t] >> 8) & 0xFF}, raster walk {(walked.Modes[t] >> 8) & 0xFF}");
            int lastTp = (w - 32) * (h - 32);   // the last tile is 8 x 4
            for (int i = 0; i < Vp8lPredictorChoose.Res && failures.Count < 20; i++)
                if (walked.Pc[i] != wave.Pc[i]) failures.Add($"accumulated histogram bin {i}: {wave.Pc[i]}, raster walk {walked.Pc[i]}");
            for (int m = 0; m < Vp8lPredictorChoose.Modes && failures.Count < 20; m++)
                for (int k = 0; k < lastTp; k++)
                {
                    int res = Vp8lPredictorChoose.Res + m * (1 << (2 * Vp8lPredictorChoose.MaxTileBits)) + k;
                    int fil = Vp8lPredictorChoose.Fill + m * (1 << (2 * Vp8lPredictorChoose.MaxTileBits)) + k;
                    if (walked.Pc[res] != wave.Pc[res]) failures.Add($"last tile, mode {m}, pixel {k}: residual {wave.Pc[res]:x8}, raster walk {walked.Pc[res]:x8}");
                    if (walked.Pc[fil] != wave.Pc[fil]) failures.Add($"last tile, mode {m}, pixel {k}: filled {wave.Pc[fil]:x8}, raster walk {walked.Pc[fil]:x8}");
                }
            int transparentCount = argb.Count(p => (p >> 24) == 0);
            int[] waveModes = wave.Modes.Select(m => (m >> 8) & 0xFF).ToArray();   // no LINQ inside the WASM log call
            Console.WriteLine($"[report] {w}x{h}, {tiles} tiles, {transparentCount} transparent pixels: modes {string.Join(",", waveModes)}");
            True(failures.Count == 0, $"{failures.Count} differences:\n" + string.Join("\n", failures.Take(20)));
        }
        finally
        {
            ReleaseAccelerator(ctx, acc);
        }
    }

    // The choice itself against a managed oracle written from libwebp 1.6.0's cost (predictor_enc.c, as documented in
    // Vp8lPredictorChoose's header), NOT from the kernel's reduced form: per channel the full
    // VP8LCombinedShannonEntropy(tile, accumulated) + PredictionCostBias(tile, 1, 0.94), in doubles, minus 15 bits per
    // neighbour (left, above) with the same mode; the residuals from a raster walk of each tile per mode with Exact = false
    // fill. The oracle replays the device's choices (its accumulated histograms take the device's chosen mode of every
    // tile), so one near-tie cannot cascade: each device mode must cost no more than the oracle's minimum plus the float
    // tolerance, and the accumulated histograms must match exactly. Tiles with and without transparent pixels, partial
    // tiles on the right and bottom.
    [TestMethod(Category = "NotOnWebGL")]
    public async Task Vp8lPredictorChoose_MatchesLibwebpCostOracle()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            const int w = 120, h = 76, bits = 4, side = 1 << bits;   // 8 x 5 tiles, partial on the right and bottom
            int tilesPerRow = (w + side - 1) >> bits, tileRows = (h + side - 1) >> bits, tiles = tilesPerRow * tileRows;
            // Every tile is built for one predictor, so the modes differ: columns constant (Top wins), rows constant (Left),
            // down-right diagonals constant (TopLeft), down-left diagonals constant (TopRight), or noise, where every mode
            // costs about the same and the neighbour bias (15 bits per neighbour with the same mode) decides; flat tiles,
            // where many modes tie EXACTLY (zero residuals) and the neighbour bias + first minimum decide; low-amplitude noise
            // on a flat colour, where the near-zero bias term weighs against the entropy.
            static uint Hash(int k)
            {
                uint v = (uint)k * 0x9E3779B1u;
                v ^= v >> 15; v *= 0x85EBCA77u; v ^= v >> 13; v *= 0xC2B2AE3Du; v ^= v >> 16;
                return v;
            }
            var argb = new uint[w * h];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    int tx = x >> bits, ty = y >> bits;
                    int pattern = (tx * 3 + ty * 2) % 7;
                    int key = pattern switch { 0 => x, 1 => y + 1000, 2 => x - y + 2000, 3 => x + y + 3000, _ => y * w + x + 5000 };
                    uint rgb = Hash(key) & 0x00ffffffu;
                    if (pattern == 5) rgb = Hash(tx * 31 + ty + 9000) & 0x00ffffffu;   // flat: one colour per tile
                    else if (pattern == 6) rgb = (0x00406080u + (Hash(key) & 0x00030303u));   // flat + noise 0..3 per channel
                    // tile column 1 is opaque (the kernel's plain residual path); elsewhere runs of transparency
                    bool transparent = tx != 1 && (((x * 3 + y * 5) % 7) < 2 || (y % 11 == 6));
                    argb[y * w + x] = (transparent ? 0u : 0xff000000u) | rgb;
                }
            // An EXACT tie decided by both neighbour biases and the first minimum: tile (1, 1) is one opaque colour C, and so
            // is every pixel its predictions read outside it (the bottom row of tile (1, 0), the right column of tile (0, 1),
            // the corner (15, 15), the top-right column x = 32), so modes 1..13 give identical (all-zero) residuals. Tile
            // (1, 0) has constant rows (Left, mode 1, wins it), tile (0, 1) constant columns (Top, mode 2): both neighbour
            // modes get -15 bits, they tie, and libwebp keeps mode 1. Without the above bias, or with the last minimum
            // winning, mode 2 would be chosen.
            // The neighbours' patterns are diagonals, which one predictor alone reproduces: down-left diagonals constant ->
            // TopRight (mode 3), down-right diagonals constant -> TopLeft (mode 4). aboveLow: the tile above gets mode 3 and the
            // left tile mode 4 (the above bias decides the tie), else the reverse (the left bias decides). The test checks
            // that the picture has an exact tie of each kind.
            void TieTile(int ttx, int tty, uint c, bool aboveLow)
            {
                int x0 = ttx * side, y0 = tty * side;
                for (int y = y0 - side; y < y0 + side; y++)
                    for (int x = x0 - side; x <= x0 + side; x++)
                    {
                        uint v;
                        bool inT = (x >= x0) & (x < x0 + side) & (y >= y0);
                        bool above = (x >= x0) & (x < x0 + side) & (y < y0);
                        bool left = (x < x0) & (y >= y0);
                        uint downLeft = 0xff000000u | (Hash(x + y + 7000 + 97 * ttx) & 0x00ffffffu);    // TopRight wins
                        uint downRight = 0xff000000u | (Hash(x - y + 8000 + 89 * tty) & 0x00ffffffu);   // TopLeft wins
                        if (inT) v = c;
                        else if (above) v = y == y0 - 1 ? c : aboveLow ? downLeft : downRight;
                        else if (left) v = x == x0 - 1 ? c : aboveLow ? downRight : downLeft;
                        else if ((x == x0 - 1) & (y == y0 - 1)) v = c;
                        else if ((x == x0 + side) & (y >= y0 - 1)) v = c;
                        else v = argb[y * w + x] | 0xff000000u;   // the rest around it opaque
                        argb[y * w + x] = v;
                    }
            }
            TieTile(1, 1, 0xff336699u, aboveLow: true);    // above mode 3 < left mode 4: the above bias decides
            TieTile(5, 2, 0xff996633u, aboveLow: false);   // left mode 3 < above mode 4: the left bias decides
            // NOT pinned by this picture (red-checked 2026-11-15): the PredictionCostBias term (removing it, or doubling its
            // 0.1 weight, changes no tile's choice). A tile it decides needs two modes of near-equal entropy with different
            // near-zero residual counts; planes with constant residuals per mode are predicted exactly by mode 12 (gradient).
            // The term is libwebp's as documented in Vp8lPredictorChoose's header.
            var meta = new int[WebPEncMeta.Size];
            meta[WebPEncMeta.UsePredictor] = 1;
            meta[WebPEncMeta.PackedWidth] = w;
            meta[WebPEncMeta.Height] = h;
            meta[WebPEncMeta.PredictorBits] = bits;
            meta[WebPEncMeta.RegionPredictor] = 0;
            meta[WebPEncMeta.Transparent] = 1;
            meta[WebPEncMeta.PaletteCount] = 0;

            var choose = acc.LoadStreamKernel<ArrayView<uint>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int, int>(Vp8lPredictorChoose.ChooseKernel);
            var clear = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(Vp8lPredictorChoose.ClearKernel);
            using var dArgb = acc.Allocate1D(argb);
            using var dMeta = acc.Allocate1D(meta);
            using var dW = acc.Allocate1D<int>(tiles);
            using var dPc = acc.Allocate1D<int>(Vp8lPredictorChoose.Total);
            dPc.View.MemSetToZero();
            dW.View.MemSetToZero();
            clear(Vp8lPredictorChoose.Res, dPc.View);
            choose(new KernelConfig(1, Vp8lPredictorChoose.GroupSizeFor(acc)), dArgb.View, dW.View, dMeta.View, dPc.View, 1, 0);
            await acc.SynchronizeAsync();
            int[] devW = await dW.View.CopyToHostAsync();
            int[] devPc = await dPc.View.CopyToHostAsync();

            static uint Sub(uint a, uint b)
            {
                uint r = 0;
                for (int s = 0; s < 32; s += 8) r |= (uint)((((a >> s) & 0xFF) - ((b >> s) & 0xFF)) & 0xFF) << s;
                return r;
            }
            static double SLog(double n) => n > 0 ? n * Math.Log2(n) : 0;

            var accum = new long[4, 256];   // the oracle's accumulated histograms (channel 0 alpha, 1 red, 2 green, 3 blue)
            int[] devModes = devW.Select(v => (v >> 8) & 0xFF).ToArray();
            var failures = new List<string>();
            int nearTies = 0, transparentTiles = 0, exactTies = 0, tiedTiles = 0, aboveDecides = 0, leftDecides = 0;
            double worstExcess = 0;
            const double Tolerance = 0.25;   // bits: float sums of ~1024 terms on the device vs doubles here
            for (int t = 0; t < tiles; t++)
            {
                int ty = t / tilesPerRow, tx = t - ty * tilesPerRow;
                int x0 = tx * side, y0 = ty * side, x1 = Math.Min(x0 + side, w), y1 = Math.Min(y0 + side, h);
                bool anyTransparent = false;
                for (int y = y0; y < y1; y++)
                    for (int x = x0; x < x1; x++) anyTransparent |= (argb[y * w + x] >> 24) == 0;
                if (anyTransparent) transparentTiles++;
                var cost = new double[Vp8lPredictorChoose.Modes];
                var hists = new long[Vp8lPredictorChoose.Modes][,];
                for (int m = 0; m < Vp8lPredictorChoose.Modes; m++)
                {
                    // raster walk: the tile's walked pixels take their filled values, everything else is the original picture
                    var cur = (uint[])argb.Clone();
                    var hist = new long[4, 256];
                    for (int y = y0; y < y1; y++)
                        for (int x = x0; x < x1; x++)
                        {
                            int i = y * w + x;
                            uint pred;
                            if (i == 0) pred = 0xff000000u;
                            else if (y == 0) pred = cur[i - 1];
                            else if (x == 0) pred = cur[i - w];
                            else pred = WebPKernels.Predict(m, cur[i - 1], cur[i - w], x + 1 == w ? cur[y * w] : cur[i - w + 1], cur[i - w - 1]);
                            uint p = argb[i];
                            uint res = Sub(p, pred);
                            if ((p >> 24) == 0)
                            {
                                res &= 0xff000000u;
                                cur[i] = pred & 0x00ffffffu;
                            }
                            for (int c = 0; c < 4; c++) hist[c, (res >> (24 - 8 * c)) & 0xFF]++;
                        }
                    hists[m] = hist;
                    double total = 0;
                    for (int c = 0; c < 4; c++)
                    {
                        // VP8LCombinedShannonEntropy(X = tile, Y = accumulated)
                        double sumX = 0, sumXY = 0, terms = 0;
                        for (int v = 0; v < 256; v++)
                        {
                            double xv = hist[c, v], yv = accum[c, v];
                            if (xv != 0) { sumX += xv; sumXY += xv + yv; terms += SLog(xv) + SLog(xv + yv); }
                            else if (yv != 0) { sumXY += yv; terms += SLog(yv); }
                        }
                        double entropy = SLog(sumX) + SLog(sumXY) - terms;
                        // PredictionCostBias(counts, weight_0 = 1, exp = 0.94): -(counts[0] + sum_{i=1..15} exp * 0.6^(i-1) * (counts[i] + counts[256 - i])) / 10
                        double bits0 = hist[c, 0], e = 0.94;
                        for (int v = 1; v < 16; v++) { bits0 += e * (hist[c, v] + hist[c, 256 - v]); e *= 0.6; }
                        total += entropy - bits0 / 10.0;
                    }
                    int left = tx > 0 ? devModes[t - 1] : -1, above = ty > 0 ? devModes[t - tilesPerRow] : -1;
                    if (m == left) total -= 15;
                    if (m == above) total -= 15;
                    cost[m] = total;
                }
                double min = cost.Min();
                int oracleBest = Array.IndexOf(cost, min);
                int d = devModes[t];
                if (d < 0 || d >= Vp8lPredictorChoose.Modes) { failures.Add($"tile {t}: device mode {d} out of range"); continue; }
                double excess = cost[d] - min;
                worstExcess = Math.Max(worstExcess, excess);
                if (excess > Tolerance)
                    failures.Add($"tile {t}: device mode {d} costs {cost[d]:F3}, oracle best mode {oracleBest} {min:F3} (+{excess:F3} bits)");
                else if (d != oracleBest) nearTies++;
                // an EXACT tie (identical histograms and neighbour bias give bit-identical costs, here and on the device):
                // libwebp keeps the first mode
                for (int m = 0; m < d; m++)
                    if (Math.Abs(cost[m] - cost[d]) < 1e-9)
                    {
                        failures.Add($"tile {t}: device mode {d} ties exactly with earlier mode {m} ({cost[d]:F3} bits): the first minimum wins");
                        exactTies++;
                        break;
                    }
                for (int m = 0; m < Vp8lPredictorChoose.Modes; m++)
                    if (m != d && Math.Abs(cost[m] - cost[d]) < 1e-9) { tiedTiles++; break; }
                // coverage: the left and above modes differ and tie exactly at the minimum, so only the lower one's bias and
                // the first-minimum rule decide (dropping either neighbour bias must change such a tile)
                int lm = tx > 0 ? devModes[t - 1] : -1, am = ty > 0 ? devModes[t - tilesPerRow] : -1;
                if ((lm >= 0) & (am >= 0) && lm != am && Math.Abs(cost[lm] - min) < 1e-9 && Math.Abs(cost[am] - min) < 1e-9)
                {
                    if (am < lm) aboveDecides++;
                    else leftDecides++;
                }
                // the device's choice joins the oracle's accumulated histograms (replay)
                for (int c = 0; c < 4; c++)
                    for (int v = 0; v < 256; v++) accum[c, v] += hists[d][c, v];
            }
            for (int c = 0; c < 4 && failures.Count < 30; c++)
                for (int v = 0; v < 256 && failures.Count < 30; v++)
                    if (devPc[Vp8lPredictorChoose.Acc + c * 256 + v] != accum[c, v])
                        failures.Add($"accumulated channel {c} bin {v}: device {devPc[Vp8lPredictorChoose.Acc + c * 256 + v]}, oracle {accum[c, v]}");
            int distinct = devModes.Distinct().Count();
            Console.WriteLine($"[report] {w}x{h}, {tiles} tiles ({transparentTiles} with transparency), group {Vp8lPredictorChoose.GroupSizeFor(acc)}: " +
                $"modes {string.Join(",", devModes)} ({distinct} distinct), near ties {nearTies}, tiles with an exact tie {tiedTiles} (broken late {exactTies}), bias ties above/left {aboveDecides}/{leftDecides}, worst excess {worstExcess:F4} bits");
            // the picture must exercise the choice: several modes, and both residual paths
            True(distinct >= 3, $"only {distinct} distinct modes: the picture does not exercise the choice");
            True(tiedTiles > 0, "no tile with an exact tie: the picture does not exercise the first-minimum rule");
            True(aboveDecides > 0 && leftDecides > 0,
                $"neighbour-bias ties: above decides {aboveDecides}, left decides {leftDecides}; the picture must exercise both biases");
            True(transparentTiles > 0 && transparentTiles < tiles, $"{transparentTiles} of {tiles} tiles with transparency: both paths must run");
            True(failures.Count == 0, $"{failures.Count} differences:\n" + string.Join("\n", failures.Take(30)));
        }
        finally
        {
            ReleaseAccelerator(ctx, acc);
        }
    }
}
