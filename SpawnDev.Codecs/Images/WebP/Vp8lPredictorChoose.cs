// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The predictor mode of every tile chosen as libwebp 1.6.0 chooses it at method <= 4 (src/enc/predictor_enc.c,
// GetBestPredictorForTile + PredictionCostSpatialHistogram, BSD): tiles in raster order; for each, all 14 modes are
// scored on the tile's residual histograms by
//     sum over the 4 channels of  PredictionCostBias(tile, 1, 0.94)            (favours residuals near 0)
//                               + VP8LCombinedShannonEntropy(tile, accumulated) (entropy of the tile, and of the tile
//                                                                              added to every earlier tile's choice)
//     - 15 bits when the mode is the left tile's mode, and again when it is the above tile's,
// the first minimum wins and its histograms join `accumulated`. Exact = false: inside the tile, transparent pixels
// take their prediction (their residual is the alpha residual only); the rows above, the left column and the
// top-right past the tile are the original picture, as libwebp's tile rows.
//
// The tiles depend on each other (the accumulated histograms, the neighbours' modes); the work inside a tile does not.
// One work-group (256 threads, fewer where the accelerator allows fewer) walks the tiles, a bounded number per launch:
//   A. the tile's residuals for all 14 modes: every thread a share of the (mode, pixel) pairs, or, when the fill
//      applies and the tile has a transparent pixel, one thread per mode walking the tile in raster order;
//   B. per mode: every thread adds its pixels into one 1024-bin histogram in shared memory (atomics), then each thread
//      reads-and-clears its 1024 / group-size bins (atomic exchange) and sums their cost terms against `accumulated`;
//   C. one thread: the choice and the mode image; then the chosen mode's histogram joins `accumulated` (each thread
//      owns 1024 / group-size bins of it: plain stores only).
// Shared memory: the histogram (4 KB, only ever touched by atomics: WebGPU cannot mix atomic and plain access on one
// array) and `accumulated` (4 KB, plain); `accumulated` is loaded at the start of a launch and stored at the end.
// Only the terms that differ between modes are computed: slog(sum X), slog(sum X + sum Y) and sum_Y slog(Y) are the
// same for every mode of a tile, so a mode's relative entropy is -sum over its NONZERO bins of
// slog(X) + slog(X + Y) - slog(Y), with slog(n) = n log2 n and the difference in the stable form
// X log2(X + Y) + Y (log2(X + Y) - log2(Y)). Integers (bits in Q32, 64-bit sums, log2 from a table computed with integer
// arithmetic on the host): the same choice on every device and in both kernel forms (BinCost). The values are ours,
// not libwebp's fixed point (our VP8L bytes are not libwebp's anyway); the pixels stay exact whatever the modes.

using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Images.WebP;

/// <summary>The predictor-mode choice of libwebp 1.6.0 (see the file header).</summary>
public static class Vp8lPredictorChoose
{
    /// <summary>Most threads of the work-group (the shared-memory partials are sized for it).</summary>
    public const int MaxGroupSize = 256;

    /// <summary>
    /// The work-group size on <paramref name="accelerator"/>: <see cref="MaxGroupSize"/>, or the accelerator's limit when
    /// lower (the CPU accelerator allows 64), rounded down to a power of two so it divides the 1024 histogram bins. The
    /// kernel reads it from Group.DimX.
    /// </summary>
    public static int GroupSizeFor(Accelerator accelerator)
    {
        int g = MaxGroupSize;
        while (g > accelerator.MaxNumThreadsPerGroup) g >>= 1;
        if (g < Modes) throw new NotSupportedException($"Vp8lPredictorChoose needs a work-group of at least {Modes} threads; this accelerator allows {accelerator.MaxNumThreadsPerGroup}.");
        return g;
    }

    /// <summary>Prediction modes (VP8L).</summary>
    public const int Modes = 14;

    /// <summary>Largest tile side supported (1 &lt;&lt; 5 = 32 pixels).</summary>
    public const int MaxTileBits = 5;

    /// <summary>Tiles per launch (bounded launches: a long launch stalls the desktop).</summary>
    public const int TilesPerLaunch = 256;

    private const int TileMax = 1 << (2 * MaxTileBits);
    private const int Bins = 4 * 256;
    // modes whose histograms are built in one round (3 x 4 KB of shared memory: WebGPU's default limit is 16 KB)
    private const int Batch = 2;

    /// <summary>Offsets in the device scratch: accumulated histograms (1024), residuals and filled pixels per mode
    /// (14 x TileMax each).</summary>
    public const int Acc = 0, Res = Acc + Bins, Fill = Res + Modes * TileMax, Total = Fill + Modes * TileMax;

    // ------------------------------------------------------------------
    // Costs in integers (2026-11-18): bits in Q32 fixed point, 64-bit sums. Floats made the choice device-dependent: a
    // near tie between two modes rounded differently per compiler (the same 25 frames gave other files on an RTX 4070 and
    // an RX 590 in 10 of 25 frames, and AMD's OpenCL compiler even differed between two kernels with the same float
    // expressions). Integer sums do not depend on the order either, so every backend and both kernel forms choose alike.
    // ------------------------------------------------------------------

    /// <summary>Entries of the exact log2 table (log2 v for v = 0..<see cref="LogTableMax"/>); larger v interpolate.</summary>
    public const int LogTableMax = 65536;

    /// <summary>Offset of the 16 bias weights in the cost table.</summary>
    public const int BiasOff = LogTableMax + 1;

    /// <summary>Longs of the cost table (<see cref="CostTable"/>).</summary>
    public const int CostTableSize = BiasOff + 16;

    private static long[]? _costTable;

    /// <summary>
    /// The cost table, computed on the host with integer arithmetic only (so it is the same on every machine):
    /// [v] = floor(log2(v) * 2^32) for v = 1..65536 ([0] = 0); [BiasOff + i] = floor(w_i * 2^32) with libwebp's
    /// PredictionCostBias(counts, 1, 94) weights, w_0 = 1 and w_i = 0.94 * 0.6^(i - 1) for |residual| = i in 1..15.
    /// </summary>
    public static long[] CostTable => _costTable ??= BuildCostTable();

    private static long[] BuildCostTable()
    {
        var t = new long[CostTableSize];
        for (int v = 1; v <= LogTableMax; v++) t[v] = Log2Q32Exact((uint)v);
        t[BiasOff] = 1L << 32;
        for (int i = 1; i < 16; i++)
        {
            // 94/100 * (6/10)^(i-1) * 2^32, exactly, then floor
            UInt128 num = 94, den = 100;
            for (int k = 1; k < i; k++) { num *= 6; den *= 10; }
            t[BiasOff + i] = (long)((num << 32) / den);
        }
        return t;
    }

    // floor(log2(v) * 2^32) by repeated squaring of the mantissa in Q62: exact, integer only
    private static long Log2Q32Exact(uint v)
    {
        int e = System.Numerics.BitOperations.Log2(v);
        UInt128 m = (UInt128)v << (62 - e);          // v / 2^e in [1, 2), Q62
        long bits = 0;
        for (int i = 0; i < 32; i++)
        {
            m = (m * m) >> 62;                        // squared, still Q62, in [1, 4)
            bits <<= 1;
            if (m >= ((UInt128)1 << 63))
            {
                bits |= 1;
                m >>= 1;
            }
        }
        return ((long)e << 32) | bits;
    }

    // log2(v) in Q32: the table up to 65536, above it interpolated between the table entries of v's top 16 bits (error
    // below 2e-10 bits per bit of v: well under a thousandth of a bit for every count a picture can have)
    internal static long Log2Q32(ArrayView<long> t, int v)
    {
        long r;
        if (v <= LogTableMax)
        {
            r = t[v];
        }
        else
        {
            int e = 31 - IntrinsicMath.LeadingZeroCount(v);
            int shift = e - 15;
            int idx = v >> shift;                     // 32768 .. 65535
            long frac = v - (idx << shift);
            long lo = t[idx], hi = t[idx + 1];
            r = lo + ((long)shift << 32) + (((hi - lo) * frac) >> shift);
        }
        return r;
    }

    /// <summary>
    /// The cost (bits, Q32) one nonzero bin of a mode's tile histogram adds: -(slog(x) + slog(x + y) - slog(y)) - x w_v / 10,
    /// with x the tile's count, y the accumulated count, v the bin's residual value and slog(n) = n log2 n; the difference in
    /// the stable form x log2(x + y) + y (log2(x + y) - log2(y)) (no slog of a large y: it would not fit in 64 bits).
    /// </summary>
    internal static long BinCost(ArrayView<long> t, int x, int y, int v)
    {
        long lx = Log2Q32(t, x);
        long step;
        if (y > 0)
        {
            long lxy = Log2Q32(t, x + y);
            step = x * lxy + y * (lxy - Log2Q32(t, y));
        }
        else
        {
            step = x * lx;
        }
        int wi = v == 0 ? 0 : v < 16 ? v : v > 240 ? 256 - v : -1;
        long bias = 0;
        if (wi >= 0) bias = (x * t[BiasOff + wi]) / 10;
        return -(x * lx + step) - bias;
    }

    /// <summary>The neighbour bias of libwebp's choice, 15 bits, in Q32.</summary>
    private const long NeighbourBonus = 15L << 32;

    // bin of channel c (0 alpha, 1 red, 2 green, 3 blue) of a residual
    private static int Bin(uint r, int c) => c * 256 + (int)((r >> (24 - 8 * c)) & 0xFFu);

    /// <summary>
    /// Tiles [<paramref name="first"/>, <paramref name="first"/> + <see cref="TilesPerLaunch"/>) in raster order (one
    /// work-group of <see cref="GroupSizeFor"/> threads). Writes each tile's mode into the predictor image
    /// (w[RegionPredictor + t] = 0xff000000 | mode &lt;&lt; 8). The accumulated histograms must be zero before the first
    /// tile of an image (<see cref="ClearKernel"/>).
    /// </summary>
    public static void ChooseKernel(ArrayView<uint> argb, ArrayView<int> w, ArrayView<int> meta, ArrayView<int> pc, ArrayView<long> lt, int options, int first)
    {
        var hist = SharedMemory.Allocate<int>(Batch * Bins);   // atomics only
        var part = SharedMemory.Allocate<long>(Batch * MaxGroupSize);
        var cost = SharedMemory.Allocate<long>(Modes);
        var flags = SharedMemory.Allocate<int>(2);         // [1] the chosen mode ([0] unused since the group-wide OR)
        int lane = Group.IdxX;
        int groupSize = Group.DimX;               // a power of two <= MaxGroupSize (GroupSizeFor)
        int binsPerThread = Bins / groupSize;
        bool active = meta[WebPEncMeta.UsePredictor] == 1;
        int width = meta[WebPEncMeta.PackedWidth];
        int height = meta[WebPEncMeta.Height];
        int bits = meta[WebPEncMeta.PredictorBits];
        int side = 1 << bits;
        int tilesPerRow = (width + side - 1) >> bits;
        int tiles = tilesPerRow * ((height + side - 1) >> bits);
        int modes = meta[WebPEncMeta.RegionPredictor];
        // options: bit 0 = transparent pixels take their predictions (Exact = false), bit 1 = walk them in raster order,
        // one thread per mode (tests: the oracle of the wavefront)
        bool fill = ((options & 1) != 0) & (meta[WebPEncMeta.Transparent] == 1) & (meta[WebPEncMeta.PaletteCount] == 0);
        bool serialWalk = (options & 2) != 0;
        int last = first + TilesPerLaunch;
        if (last > tiles) last = tiles;
        // Only pictures whose transparent pixels take their predictions (Exact = false): measured 2026-11-14 on SpawnGen's
        // LTX frames, this choice gave the same size as the summed-magnitude one (PredictorModeKernel, which runs first
        // and whose modes stay) on opaque frames, and 3.6% smaller animations (blended frames full of transparency).
        if (!active | !fill) last = first;   // uniform across the group: every thread reads the same meta
        int b0 = lane * binsPerThread;

        // the accumulated histogram stays in device memory (bin b is read and written only by thread b / binsPerThread)
        for (int q = 0; q < Batch; q++)
            for (int j = 0; j < binsPerThread; j++) Atomic.Exchange(ref hist[q * Bins + b0 + j], 0);
        Group.Barrier();

        for (int t = first; t < last; t++)
        {
            int ty = t / tilesPerRow;
            int tx = t - ty * tilesPerRow;
            int x0 = tx << bits, y0 = ty << bits;
            int x1 = x0 + side;
            if (x1 > width) x1 = width;
            int y1 = y0 + side;
            if (y1 > height) y1 = height;
            int tw = x1 - x0;
            int tp = tw * (y1 - y0);

            // A. the residuals of every mode
            bool mine = false;
            if (fill)
            {
                for (int k = lane; k < tp; k += groupSize)
                {
                    int ky = k / tw;
                    if ((argb[(y0 + ky) * width + x0 + (k - ky * tw)] >> 24) == 0u) mine = true;
                }
            }
            // the tile has a transparent pixel (and the fill applies): a group-wide OR, so the branch below - barriers
            // inside - is uniform (WGSL's validator rejects a barrier under a condition read from shared memory:
            // "storageBarrier must only be called from uniform control flow", 2026-11-15)
            bool walkFill = Group.BarrierOr(mine);
            if (walkFill)
            {
                // transparent pixels take their prediction, which feeds the pixels after them. A pixel needs its left,
                // top, top-left and top-right neighbours, so the pixels with the same x + 2y (tile coordinates) are
                // independent: a wavefront of (tw - 1) + 2 (th - 1) + 1 steps, every (mode, row) one item per step.
                int th = y1 - y0;
                int steps = (tw - 1) + 2 * (th - 1) + 1;
                if (serialWalk)
                {
                    if (lane < Modes)
                        for (int y = y0; y < y1; y++)
                            for (int x = x0; x < x1; x++) FillCell(argb, pc, lane, x, y, x0, x1, y0, tw, width);
                    steps = 0;
                }
                for (int d = 0; d < steps; d++)
                {
                    for (int item = lane; item < Modes * th; item += groupSize)
                    {
                        int mode = item / th;
                        int ly = item - mode * th;
                        int lx = d - 2 * ly;
                        if ((lx >= 0) & (lx < tw)) FillCell(argb, pc, mode, x0 + lx, y0 + ly, x0, x1, y0, tw, width);
                    }
                    // step d + 1 reads cells step d wrote (other lanes). Required by the memory model; NOT observable by the
                    // wavefront test on CUDA, Velocity or the CPU accelerator (removing it passes there, 2026-11-15).
                    Group.Barrier();
                }
            }
            else
            {
                for (int idx = lane; idx < Modes * tp; idx += groupSize)
                {
                    int mode = idx / tp;
                    int k = idx - mode * tp;
                    int ky = k / tw;
                    int y = y0 + ky, x = x0 + (k - ky * tw);
                    int i = y * width + x;
                    uint pred;
                    if (i == 0) pred = 0xff000000u;
                    else if (y == 0) pred = argb[i - 1];
                    else if (x == 0) pred = argb[i - width];
                    else pred = WebPKernels.Predict(mode, argb[i - 1], argb[i - width], argb[i - width + 1], argb[i - width - 1]);
                    pc[Res + mode * TileMax + k] = (int)SubPixels(argb[i], pred);
                }
            }
            Group.Barrier();

            // B. per round of Batch modes: their histograms (atomics in shared memory), each thread's 16 bins of each read
            //    and cleared, a partial cost per mode; three threads sum the partials
            for (int mb = 0; mb < Modes; mb += Batch)
            {
                int nb = Modes - mb;
                if (nb > Batch) nb = Batch;
                for (int k = lane; k < tp; k += groupSize)
                {
                    for (int q = 0; q < nb; q++)
                    {
                        uint r = (uint)pc[Res + (mb + q) * TileMax + k];
                        int hb = q * Bins;
                        Atomic.Add(ref hist[hb + Bin(r, 0)], 1);
                        Atomic.Add(ref hist[hb + Bin(r, 1)], 1);
                        Atomic.Add(ref hist[hb + Bin(r, 2)], 1);
                        Atomic.Add(ref hist[hb + Bin(r, 3)], 1);
                    }
                }
                Group.Barrier();
                for (int q = 0; q < nb; q++)
                {
                    long c = 0;
                    for (int j = 0; j < binsPerThread; j++)
                    {
                        int b = b0 + j;
                        int count = Atomic.Exchange(ref hist[q * Bins + b], 0);
                        // PredictionCostBias (-bits / 10) is inside BinCost. The oracle test does not pin that term (no tile
                        // of its picture is decided by it): keep it equal to libwebp's text when touching it.
                        if (count > 0) c += BinCost(lt, count, pc[Acc + b], b & 255);
                    }
                    part[q * groupSize + lane] = c;
                }
                Group.Barrier();
                if (lane < nb)
                {
                    long sum = 0;
                    for (int u = 0; u < groupSize; u++) sum += part[lane * groupSize + u];
                    cost[mb + lane] = sum;
                }
                // the next round writes `part` again
                Group.Barrier();
            }
            Group.Barrier();

            // C. the choice (first minimum, libwebp's order) and the mode image
            if (lane == 0)
            {
                int left = tx > 0 ? (w[modes + t - 1] >> 8) & 0xFF : 0xFF;
                int above = ty > 0 ? (w[modes + t - tilesPerRow] >> 8) & 0xFF : 0xFF;
                int best = 0;
                long bestCost = 0;
                for (int m = 0; m < Modes; m++)
                {
                    long c = cost[m];
                    if (m == left) c -= NeighbourBonus;
                    if (m == above) c -= NeighbourBonus;
                    if ((m == 0) | (c < bestCost))
                    {
                        bestCost = c;
                        best = m;
                    }
                }
                w[modes + t] = (int)(0xff000000u | ((uint)best << 8));
                flags[1] = best;
            }
            Group.Barrier();

            // the chosen mode's histogram joins the accumulated one
            int chosen = Res + flags[1] * TileMax;
            for (int k = lane; k < tp; k += groupSize)
            {
                uint r = (uint)pc[chosen + k];
                Atomic.Add(ref hist[Bin(r, 0)], 1);
                Atomic.Add(ref hist[Bin(r, 1)], 1);
                Atomic.Add(ref hist[Bin(r, 2)], 1);
                Atomic.Add(ref hist[Bin(r, 3)], 1);
            }
            Group.Barrier();
            for (int j = 0; j < binsPerThread; j++) pc[Acc + b0 + j] = pc[Acc + b0 + j] + Atomic.Exchange(ref hist[b0 + j], 0);
            Group.Barrier();
        }
    }

    // ------------------------------------------------------------------
    // Two-kernel form (the default since 2026-11-18): everything that does not depend on the tiles before - each tile's
    // residuals for all 14 modes (with the transparent fill) and their histograms - is computed for a window of tiles at
    // once, one work-group per tile (HistKernel, the whole GPU); the walk (WalkKernel, one work-group, tiles in order) only
    // scores the stored histograms against the accumulated one, chooses, and accumulates: 4 barriers per tile instead of
    // the wavefront's and the 7 histogram rounds'. The walk computes every cost term exactly as ChooseKernel does (same
    // expressions, same bins per thread, same summation order), so the choices are ChooseKernel's (kept as the oracle).
    // ------------------------------------------------------------------

    private const int HalfModes = Modes / 2;

    /// <summary>Tiles whose histograms are built per HistKernel launch (and scored per WalkKernel launch).</summary>
    public const int WindowTiles = 128;

    /// <summary>Ints of one window slot of the histogram scratch: 14 histograms of 1024 bins.</summary>
    public const int HistSlot = Modes * Bins;

    /// <summary>Ints of one window slot of the fill scratch: 14 fill buffers of up to <see cref="TileMax"/> pixels.</summary>
    public const int FillSlot = Modes * TileMax;

    /// <summary>
    /// The histogram scratch: accumulated histograms (1024, the walk's only), then <see cref="WindowTiles"/> slots (atomics
    /// only in HistKernel). The fill pixels live in a second buffer (plain access only): WebGPU cannot mix atomic and plain
    /// access to one array within a kernel.
    /// </summary>
    public const int WAcc = 0, WSlots = Bins, WTotal = WSlots + WindowTiles * HistSlot, FillTotal = WindowTiles * FillSlot;

    /// <summary>
    /// One work-group (<see cref="GroupSizeFor"/> threads) per tile <c>first + Grid.IdxX</c>: the 14 modes' residual
    /// histograms of the tile into its slot (device-memory atomics; the slot is cleared first by atomic exchange: WebGPU
    /// cannot mix atomic and plain access to one array in a kernel). Residuals as ChooseKernel's phase A.
    /// </summary>
    public static void HistKernel(ArrayView<uint> argb, ArrayView<int> meta, ArrayView<int> hist, ArrayView<int> fillPx, int options, int first)
    {
        int lane = Group.IdxX;
        int groupSize = Group.DimX;
        bool active = meta[WebPEncMeta.UsePredictor] == 1;
        int width = meta[WebPEncMeta.PackedWidth];
        int height = meta[WebPEncMeta.Height];
        int bits = meta[WebPEncMeta.PredictorBits];
        int side = 1 << bits;
        int tilesPerRow = (width + side - 1) >> bits;
        int tiles = tilesPerRow * ((height + side - 1) >> bits);
        bool fill = ((options & 1) != 0) & (meta[WebPEncMeta.Transparent] == 1) & (meta[WebPEncMeta.PaletteCount] == 0);
        bool serialWalk = (options & 2) != 0;
        int t = first + Grid.IdxX;
        // The same value in every thread (the same meta, the same t), but read from a storage buffer, which WGSL's
        // uniformity analysis calls non-uniform: the barriers below would be rejected ("storageBarrier must only be called
        // from uniform control flow", WebGPU 2026-11-18). A group-wide OR makes it uniform.
        bool run = Group.BarrierOr(active & fill & (t < tiles));
        if (run)
        {
            int hb = WSlots + Grid.IdxX * HistSlot;
            int fb = Grid.IdxX * FillSlot;
            int ty = t / tilesPerRow;
            int tx = t - ty * tilesPerRow;
            int x0 = tx << bits, y0 = ty << bits;
            int x1 = x0 + side;
            if (x1 > width) x1 = width;
            int y1 = y0 + side;
            if (y1 > height) y1 = height;
            int tw = x1 - x0;
            int tp = tw * (y1 - y0);

            for (int b = lane; b < Modes * Bins; b += groupSize) Atomic.Exchange(ref hist[hb + b], 0);
            bool mine = false;
            for (int k = lane; k < tp; k += groupSize)
            {
                int ky = k / tw;
                if ((argb[(y0 + ky) * width + x0 + (k - ky * tw)] >> 24) == 0u) mine = true;
            }
            // also the barrier after the clear
            bool walkFill = Group.BarrierOr(mine);
            if (walkFill)
            {
                int th = y1 - y0;
                int steps = (tw - 1) + 2 * (th - 1) + 1;
                if (serialWalk)
                {
                    if (lane < Modes)
                        for (int y = y0; y < y1; y++)
                            for (int x = x0; x < x1; x++) FillCellHist(argb, hist, fillPx, hb, fb, lane, x, y, x0, x1, y0, tw, width);
                    steps = 0;
                }
                for (int d = 0; d < steps; d++)
                {
                    for (int item = lane; item < Modes * th; item += groupSize)
                    {
                        int mode = item / th;
                        int ly = item - mode * th;
                        int lx = d - 2 * ly;
                        if ((lx >= 0) & (lx < tw)) FillCellHist(argb, hist, fillPx, hb, fb, mode, x0 + lx, y0 + ly, x0, x1, y0, tw, width);
                    }
                    // step d + 1 reads fill cells step d wrote (other lanes)
                    Group.Barrier();
                }
            }
            else
            {
                for (int idx = lane; idx < Modes * tp; idx += groupSize)
                {
                    int mode = idx / tp;
                    int k = idx - mode * tp;
                    int ky = k / tw;
                    int y = y0 + ky, x = x0 + (k - ky * tw);
                    int i = y * width + x;
                    uint pred;
                    if (i == 0) pred = 0xff000000u;
                    else if (y == 0) pred = argb[i - 1];
                    else if (x == 0) pred = argb[i - width];
                    else pred = WebPKernels.Predict(mode, argb[i - 1], argb[i - width], argb[i - width + 1], argb[i - width - 1]);
                    AddResidual(hist, hb + mode * Bins, SubPixels(argb[i], pred));
                }
            }
        }
    }

    private static void AddResidual(ArrayView<int> pc, int h, uint r)
    {
        Atomic.Add(ref pc[h + Bin(r, 0)], 1);
        Atomic.Add(ref pc[h + Bin(r, 1)], 1);
        Atomic.Add(ref pc[h + Bin(r, 2)], 1);
        Atomic.Add(ref pc[h + Bin(r, 3)], 1);
    }

    // FillCell of the two-kernel form: the residual goes into the mode's histogram of the slot instead of a residual buffer.
    private static void FillCellHist(ArrayView<uint> argb, ArrayView<int> hist, ArrayView<int> fillPx, int hb, int fbBase, int mode, int x, int y, int x0, int x1, int y0, int tw, int width)
    {
        int fb = fbBase + mode * TileMax;
        int i = y * width + x;
        uint pred;
        if (i == 0) pred = 0xff000000u;
        else if (y == 0) pred = Neighbour(argb, fillPx, fb, x - 1, y, x0, x1, y0, tw, width, x, y);
        else if (x == 0) pred = Neighbour(argb, fillPx, fb, x, y - 1, x0, x1, y0, tw, width, x, y);
        else
        {
            bool lastCol = x + 1 == width;
            pred = WebPKernels.Predict(mode,
                Neighbour(argb, fillPx, fb, x - 1, y, x0, x1, y0, tw, width, x, y),
                Neighbour(argb, fillPx, fb, x, y - 1, x0, x1, y0, tw, width, x, y),
                Neighbour(argb, fillPx, fb, lastCol ? 0 : x + 1, lastCol ? y : y - 1, x0, x1, y0, tw, width, x, y),
                Neighbour(argb, fillPx, fb, x - 1, y - 1, x0, x1, y0, tw, width, x, y));
        }
        uint p = argb[i];
        uint r = SubPixels(p, pred);
        if ((p >> 24) == 0u)
        {
            r &= 0xff000000u;
            p = pred & 0x00ffffffu;
        }
        AddResidual(hist, hb + mode * Bins, r);
        fillPx[fb + (y - y0) * tw + (x - x0)] = (int)p;
    }

    /// <summary>
    /// One work-group: tiles [<paramref name="first"/>, <paramref name="first"/> + <see cref="WindowTiles"/>) in raster
    /// order, their histograms from HistKernel's slots: each mode scored against the accumulated histograms (ChooseKernel's
    /// phase B arithmetic, term for term), the choice (phase C), the chosen histogram added to the accumulated one.
    /// </summary>
    public static void WalkKernel(ArrayView<int> w, ArrayView<int> meta, ArrayView<int> pc, ArrayView<long> lt, int options, int first)
    {
        // half the modes per round: 7 x 256 longs = 14 KB of shared memory (WebGPU's default limit is 16 KB)
        var part = SharedMemory.Allocate<long>(HalfModes * MaxGroupSize);
        var cost = SharedMemory.Allocate<long>(Modes);
        var flags = SharedMemory.Allocate<int>(2);
        int lane = Group.IdxX;
        int groupSize = Group.DimX;
        int binsPerThread = Bins / groupSize;
        bool active = meta[WebPEncMeta.UsePredictor] == 1;
        int width = meta[WebPEncMeta.PackedWidth];
        int height = meta[WebPEncMeta.Height];
        int bits = meta[WebPEncMeta.PredictorBits];
        int side = 1 << bits;
        int tilesPerRow = (width + side - 1) >> bits;
        int tiles = tilesPerRow * ((height + side - 1) >> bits);
        int modes = meta[WebPEncMeta.RegionPredictor];
        bool fill = ((options & 1) != 0) & (meta[WebPEncMeta.Transparent] == 1) & (meta[WebPEncMeta.PaletteCount] == 0);
        int last = first + WindowTiles;
        if (last > tiles) last = tiles;
        if (!active | !fill) last = first;
        int b0 = lane * binsPerThread;

        for (int t = first; t < last; t++)
        {
            int hb = WSlots + (t - first) * HistSlot;
            int ty = t / tilesPerRow;
            int tx = t - ty * tilesPerRow;
            for (int half = 0; half < 2; half++)
            {
                for (int q = 0; q < HalfModes; q++)
                {
                    int m = half * HalfModes + q;
                    long c = 0;
                    for (int j = 0; j < binsPerThread; j++)
                    {
                        int b = b0 + j;
                        int count = pc[hb + m * Bins + b];
                        if (count > 0) c += BinCost(lt, count, pc[WAcc + b], b & 255);
                    }
                    part[q * groupSize + lane] = c;
                }
                Group.Barrier();
                if (lane < HalfModes)
                {
                    long sum = 0;
                    for (int u = 0; u < groupSize; u++) sum += part[lane * groupSize + u];
                    cost[half * HalfModes + lane] = sum;
                }
                // the next half writes `part` again; after the second, lane 0 reads `cost`
                Group.Barrier();
            }
            if (lane == 0)
            {
                int left = tx > 0 ? (w[modes + t - 1] >> 8) & 0xFF : 0xFF;
                int above = ty > 0 ? (w[modes + t - tilesPerRow] >> 8) & 0xFF : 0xFF;
                int best = 0;
                long bestCost = 0;
                for (int m = 0; m < Modes; m++)
                {
                    long c = cost[m];
                    if (m == left) c -= NeighbourBonus;
                    if (m == above) c -= NeighbourBonus;
                    if ((m == 0) | (c < bestCost))
                    {
                        bestCost = c;
                        best = m;
                    }
                }
                w[modes + t] = (int)(0xff000000u | ((uint)best << 8));
                flags[1] = best;
            }
            Group.Barrier();
            int chosen = hb + flags[1] * Bins;
            for (int j = 0; j < binsPerThread; j++) pc[WAcc + b0 + j] = pc[WAcc + b0 + j] + pc[chosen + b0 + j];
            // the next tile reads the accumulated bins of every thread, and lane 0 the mode image
            Group.Barrier();
        }
    }

    /// <summary>Thread i: clears the accumulated histograms (before the first tile of an image).</summary>
    public static void ClearKernel(Index1D i, ArrayView<int> pc)
    {
        if (i < Res) pc[Acc + i] = 0;
    }

    // One pixel (x, y) of the fill walk of mode `mode`: its residual (a transparent pixel takes its prediction, kept in
    // the mode's fill buffer for the pixels after it, and keeps only its alpha residual). Its in-tile neighbours (all
    // before it in raster order) must be done.
    private static void FillCell(ArrayView<uint> argb, ArrayView<int> pc, int mode, int x, int y, int x0, int x1, int y0, int tw, int width)
    {
        int res = Res + mode * TileMax;
        int fb = Fill + mode * TileMax;
        int i = y * width + x;
        uint pred;
        if (i == 0) pred = 0xff000000u;
        else if (y == 0) pred = Neighbour(argb, pc, fb, x - 1, y, x0, x1, y0, tw, width, x, y);
        else if (x == 0) pred = Neighbour(argb, pc, fb, x, y - 1, x0, x1, y0, tw, width, x, y);
        else
        {
            // top-right of a row's last pixel is the row's first pixel
            bool lastCol = x + 1 == width;
            pred = WebPKernels.Predict(mode,
                Neighbour(argb, pc, fb, x - 1, y, x0, x1, y0, tw, width, x, y),
                Neighbour(argb, pc, fb, x, y - 1, x0, x1, y0, tw, width, x, y),
                Neighbour(argb, pc, fb, lastCol ? 0 : x + 1, lastCol ? y : y - 1, x0, x1, y0, tw, width, x, y),
                Neighbour(argb, pc, fb, x - 1, y - 1, x0, x1, y0, tw, width, x, y));
        }
        uint p = argb[i];
        uint r = SubPixels(p, pred);
        if ((p >> 24) == 0u)
        {
            r &= 0xff000000u;
            p = pred & 0x00ffffffu;
        }
        int k = (y - y0) * tw + (x - x0);
        pc[res + k] = (int)r;
        pc[fb + k] = (int)p;
    }

    // A neighbour (column jx, row jy) of pixel (x, y) while the tile [x0, x1) x [y0, ...) is walked: the tile's pixels
    // already walked come from the mode's fill buffer, everything else is the original picture.
    private static uint Neighbour(ArrayView<uint> argb, ArrayView<int> pc, int fb, int jx, int jy, int x0, int x1, int y0,
        int tw, int width, int x, int y)
    {
        bool walked = (jx >= x0) & (jx < x1) & (jy >= y0) & ((jy < y) | ((jy == y) & (jx < x)));
        uint v;
        if (walked) v = (uint)pc[fb + (jy - y0) * tw + (jx - x0)];
        else v = argb[jy * width + jx];
        return v;
    }

    private static uint SubPixels(uint a, uint b)
    {
        uint ag = 0x00ff00ffu + (a & 0xff00ff00u) - (b & 0xff00ff00u);
        uint rb = 0xff00ff00u + (a & 0x00ff00ffu) - (b & 0x00ff00ffu);
        return (ag & 0xff00ff00u) | (rb & 0x00ff00ffu);
    }
}
