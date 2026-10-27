// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// BC7 block encoder on the accelerator: a port of bc7enc (Richard Geldreich, Jr., MIT or public domain; the copy etcpak
// 0.9.15 builds, its portable paths) at its default settings: perceptual weights, all modes, 64 partitions, least squares,
// the mode 1 / 7 partition filter bank, uber level 0. Opaque blocks try mode 6 then mode 1; blocks with alpha mode 6
// (RGBA), mode 5 (separate alpha) and mode 7. Each mode is one launch, one thread per block, in bc7enc's order; it keeps
// its result as the block's best (40 packed bytes) when it is the first or strictly better, comparing as bc7enc does
// (the error rounded through a float times the mode weight), and is skipped once the best error is 0. A last launch
// packs the 16 bytes, one thread per output word. bc7enc stops a mode 1 / 7 search early once it cannot win; finishing
// it changes no choice. The arithmetic is bc7enc's float arithmetic step by step: the IEEE backends (CPU, Velocity,
// Wasm) write bc7enc's blocks exactly; GPUs whose division, square root or fused multiply-add round differently can
// land on another endpoint now and then (the tests measure how often).

using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Images.Texture;

/// <summary>The BC7 encode launches (used by <see cref="BcnEncoderGpu"/>).</summary>
internal sealed class Bc7EncoderGpu : IDisposable
{
    // ---- int tables ----
    private const int TP2 = 0;                          // 2-subset partitions [64][16]
    private const int TA2 = TP2 + 1024;                 // anchor of subset 1 [64]
    private const int TOrder = TA2 + 64;                // estimator partition order [64]
    private const int TPred = TOrder + 64;              // partition predictors [35] (as int bits)
    private const int TM1 = TPred + 35;                 // mode 1 optimal single colour [256][2]: err | lo << 16 | hi << 24
    private const int TM7 = TM1 + 512;                  // mode 7 optimal single colour [256][hp 2][lp 2]
    private const int TW2 = TM7 + 1024, TW3 = TW2 + 4, TW4 = TW3 + 8;
    private const int TPW = TW4 + 16;                   // the 4 perceptual weights
    private const int TableInts = TPW + 4;
    // ---- float tables ----
    private const int FW2 = 0, FW3 = FW2 + 16, FW4 = FW3 + 32;      // least-squares weights x4
    private const int FM1 = FW4 + 64;                   // mode 1 midpoints [64][2]
    private const int FM5 = FM1 + 128;                  // mode 5 midpoints [128]
    private const int FM7 = FM5 + 128;                  // mode 7 midpoints [32][2]
    private const int TableFloats = FM7 + 64;

    // ---- a mode's candidate (thread-local, unpacked) ----
    private const int CLow0 = 0, CHigh0 = 4, CLow1 = 8, CHigh1 = 12, CPbits = 16, CSel = 20, CASel = 36, CandInts = 52;
    // ---- the block's best so far (scratch, packed) ----
    // 0, 1: its weighted error (low, high word); 2: mode | partition << 4 | p-bits << 10 (subset 0 low, high, subset 1
    // low, high); 3..6: low 0, high 0, low 1, high 1 (r | g << 8 | b << 16 | a << 24); 7, 8: colour selectors (4 bits
    // each); 9: alpha selectors (2 bits each).
    private const int BErrLo = 0, BErrHi = 1, BInfo = 2, BEnd = 3, BSel = 7, BASel = 9;
    internal const int BlockInts = 10;

    private readonly MemoryBuffer1D<int, Stride1D.Dense> _ti;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _tf;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<float>, int, int, int, int> _mode;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<byte>, ArrayView<int>> _pack;
    // Threads per block in a mode launch: 1 (it stores the whole best), or BlockInts where a thread may store only its
    // own element (WebGL): thread j of a block computes the same candidate and stores int j.
    private readonly int _lanes;

    public Bc7EncoderGpu(Accelerator accelerator)
    {
        _lanes = accelerator.Device.Satisfies(new AcceleratorRequirements { RequiresScatterStores = true }) ? 1 : BlockInts;
        _ti = accelerator.Allocate1D<int>(TableInts);
        _tf = accelerator.Allocate1D<float>(TableFloats);
        BuildTables(out int[] ti, out float[] tf);
        _ti.View.CopyFromCPU(ti);
        _tf.View.CopyFromCPU(tf);
        _mode = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<float>, int, int, int, int>(ModeKernel);
        _pack = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<byte>, ArrayView<int>>(PackKernel);
    }

    /// <summary>The tables bc7enc_compress_block_init builds, the static ones, and the default perceptual weights.</summary>
    private static void BuildTables(out int[] ti, out float[] tf)
    {
        ti = new int[TableInts];
        tf = new float[TableFloats];
        for (int i = 0; i < 1024; i++) ti[TP2 + i] = Bc7Tables.Partition2[i];
        for (int i = 0; i < 64; i++) ti[TA2 + i] = Bc7Tables.Anchor2[i];
        for (int i = 0; i < 64; i++) ti[TOrder + i] = Bc7EncTables.SortedPartitionOrder[i];
        for (int i = 0; i < 35; i++) ti[TPred + i] = unchecked((int)Bc7EncTables.PartitionPredictors[i]);
        int[] w2 = { 0, 21, 43, 64 }, w3 = { 0, 9, 18, 27, 37, 46, 55, 64 }, w4 = { 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 };
        w2.CopyTo(ti, TW2);
        w3.CopyTo(ti, TW3);
        w4.CopyTo(ti, TW4);
        Bc7EncTables.Weights2x.CopyTo(tf, FW2);
        Bc7EncTables.Weights3x.CopyTo(tf, FW3);
        Bc7EncTables.Weights4x.CopyTo(tf, FW4);

        // Mode 7 midpoints.
        for (int p = 0; p < 2; p++)
            for (int i = 0; i < 32; i++)
            {
                int vl = ((i << 1) | p) << 2;
                vl |= vl >> 6;
                float lo = vl / 255.0f;
                int vh = ((Math.Min(31, i + 1) << 1) | p) << 2;
                vh |= vh >> 6;
                float hi = vh / 255.0f;
                tf[FM7 + i * 2 + p] = i == 31 ? 1.0f : (lo + hi) / 2.0f;
            }
        // Mode 1 midpoints.
        for (int p = 0; p < 2; p++)
            for (int i = 0; i < 64; i++)
            {
                int vl = ((i << 1) | p) << 1;
                vl |= vl >> 7;
                float lo = vl / 255.0f;
                int vh = ((Math.Min(63, i + 1) << 1) | p) << 1;
                vh |= vh >> 7;
                float hi = vh / 255.0f;
                tf[FM1 + i * 2 + p] = i == 63 ? 1.0f : (lo + hi) / 2.0f;
            }
        // Mode 5 midpoints.
        for (int i = 0; i < 128; i++)
        {
            int vl = i << 1;
            vl |= vl >> 7;
            float lo = vl / 255.0f;
            int vh = Math.Min(127, i + 1) << 1;
            vh |= vh >> 7;
            float hi = vh / 255.0f;
            tf[FM5 + i] = i == 127 ? 1.0f : (lo + hi) / 2.0f;
        }
        // Mode 1 optimal single colour (index 2 of the 3-bit weights).
        for (int c = 0; c < 256; c++)
            for (int lp = 0; lp < 2; lp++)
            {
                int bestErr = ushort.MaxValue, bestLo = 0, bestHi = 0;
                for (int l = 0; l < 64; l++)
                {
                    int low = ((l << 1) | lp) << 1;
                    low |= low >> 7;
                    for (int h = 0; h < 64; h++)
                    {
                        int high = ((h << 1) | lp) << 1;
                        high |= high >> 7;
                        int k = (low * (64 - w3[2]) + high * w3[2] + 32) >> 6;
                        int err = (k - c) * (k - c);
                        if (err < bestErr) { bestErr = err; bestLo = l; bestHi = h; }
                    }
                }
                ti[TM1 + c * 2 + lp] = bestErr | (bestLo << 16) | (bestHi << 24);
            }
        // Mode 7 optimal single colour (index 1 of the 2-bit weights).
        for (int c = 0; c < 256; c++)
            for (int hp = 0; hp < 2; hp++)
                for (int lp = 0; lp < 2; lp++)
                {
                    int bestErr = ushort.MaxValue, bestLo = 0, bestHi = 0;
                    for (int l = 0; l < 32; l++)
                    {
                        int low = ((l << 1) | lp) << 2;
                        low |= low >> 6;
                        for (int h = 0; h < 32; h++)
                        {
                            int high = ((h << 1) | hp) << 2;
                            high |= high >> 6;
                            int k = (low * (64 - w2[1]) + high * w2[1] + 32) >> 6;
                            int err = (k - c) * (k - c);
                            if (err < bestErr) { bestErr = err; bestLo = l; bestHi = h; }
                        }
                    }
                    ti[TM7 + c * 4 + hp * 2 + lp] = bestErr | (bestLo << 16) | (bestHi << 24);
                }
        // bc7enc_compress_block: perceptual weights (128, 64, 16, 32) scaled to YCbCr.
        float prWeight = (.5f / (1.0f - .2126f)) * (.5f / (1.0f - .2126f));
        float pbWeight = (.5f / (1.0f - .0722f)) * (.5f / (1.0f - .0722f));
        ti[TPW] = (int)(128 * 4.0f);
        ti[TPW + 1] = (int)(64 * 4.0f * prWeight);
        ti[TPW + 2] = (int)(16 * 4.0f * pbWeight);
        ti[TPW + 3] = 32 * 4;
    }

    /// <summary>Ints each of the two scratch buffers needs for a width x height picture (the per-block best).</summary>
    public static long ScratchInts(int width, int height) => (long)BlockInts * ((width + 3) / 4) * ((height + 3) / 4);

    /// <summary>
    /// Queue the BC7 encode of rgba (width x height, edge pixels repeated) into blocks; scratchA and scratchB hold
    /// ScratchInts ints each. The mode launches run in bc7enc's order (6, 1, 5, 7), each reading the block's best from
    /// one scratch buffer and writing the new best (its candidate when strictly better, else the old best) to the other,
    /// so no launch reads what it writes.
    /// </summary>
    public void Encode(ArrayView<byte> rgba, int width, int height, ArrayView<int> scratchA, ArrayView<int> scratchB, ArrayView<byte> blocks)
    {
        int count = ((width + 3) / 4) * ((height + 3) / 4);
        for (int which = 0; which < 4; which++)
        {
            bool even = (which & 1) == 0;
            _mode(count * _lanes, rgba, even ? scratchB : scratchA, even ? scratchA : scratchB, _ti.View, _tf.View, which, width, height, _lanes);
        }
        _pack(count * 4, scratchB, blocks, _ti.View);
    }

    public void Dispose()
    {
        _ti.Dispose();
        _tf.Dispose();
    }

    // ==================================================================
    // Kernels
    // ==================================================================

    /// <summary>The block's pixels (edge pixels repeated) into px (r, g, b, a per pixel); true when any alpha is below 255.</summary>
    private static bool LoadBlock(ArrayView<byte> rgba, int block, int width, int height, ArrayView<int> px)
    {
        int bw = (width + 3) >> 2;
        int x0 = (block % bw) * 4, y0 = (block / bw) * 4;
        bool alpha = false;
        for (int p = 0; p < 16; p++)
        {
            int x = x0 + (p & 3), y = y0 + (p >> 2);
            if (x >= width) x = width - 1;
            if (y >= height) y = height - 1;
            int o = (y * width + x) * 4;
            for (int c = 0; c < 4; c++) px[p * 4 + c] = rgba[o + c];
            alpha |= px[p * 4 + 3] < 255;
        }
        return alpha;
    }

    /// <summary>
    /// <paramref name="lanes"/> threads per block: the candidate of one mode (which 0: mode 6, 1: mode 1, 2: mode 5, 3:
    /// mode 7), the block's new best when it is the first or its weighted error is strictly lower than the best in
    /// <paramref name="prev"/> (bc7enc's choice); the new best goes to <paramref name="next"/>. Opaque blocks try modes 6
    /// and 1, blocks with alpha 6, 5 and 7; once the best error is 0 the later modes are skipped. With one lane a thread
    /// stores the whole best, with BlockInts lanes thread j stores its int j (whose index is the thread's).
    /// </summary>
    private static void ModeKernel(Index1D index, ArrayView<byte> rgba, ArrayView<int> prev, ArrayView<int> next, ArrayView<int> ti,
        ArrayView<float> tf, int which, int width, int height, int lanes)
    {
        int block = index / lanes;
        int lane = index - block * lanes;
        var px = LocalMemory.Allocate<int>(64);
        var idx = LocalMemory.Allocate<int>(16);
        var sel = LocalMemory.Allocate<int>(16);
        var selTemp = LocalMemory.Allocate<int>(16);
        var res = LocalMemory.Allocate<int>(32);   // low[4], high[4], pbits[2] of the best solution; 16..25 a one-colour candidate
        var est = LocalMemory.Allocate<int>(40);   // the partition estimator's interpolated colours and thresholds
        var cand = LocalMemory.Allocate<int>(CandInts);
        for (int i = 0; i < CandInts; i++) cand[i] = 0;
        bool alpha = LoadBlock(rgba, block, width, height, px);
        int w0 = ti[TPW], w1 = ti[TPW + 1], w2 = ti[TPW + 2], w3 = ti[TPW + 3];

        // One call site per heavy helper (each is inlined where it is called).
        int mode = which == 0 ? 6 : (which == 1 ? 1 : (which == 2 ? 5 : 7));
        bool active = (which == 0) | ((which == 1) & !alpha) | ((which >= 2) & alpha);
        int b = block * BlockInts;
        ulong best = which == 0 ? 0 : BestErr(prev, b);
        bool run = active & ((which == 0) | (best != 0));
        bool take = false;
        ulong weighted = 0;
        int part = 0;
        if (run)
        {
            bool partitioned = (mode == 1) | (mode == 7);
            bool cellAlpha = (mode == 7) | ((mode == 6) & alpha);
            if (partitioned) part = EstimatePartition(mode, px, idx, est, ti, w0, w1, w2, w3);
            ulong total = 0;
            int subsets = partitioned ? 2 : 1;
            for (int subset = 0; subset < subsets; subset++)
            {
                int n = 0;
                for (int i = 0; i < 16; i++)
                    if ((!partitioned) | (ti[TP2 + part * 16 + i] == subset)) { idx[n] = i; n++; }
                total += CellCompression(mode, n, cellAlpha, px, idx, sel, selTemp, res, ti, tf, w0, w1, w2, w3);
                StoreCand(cand, res, subset, sel, idx, n);
            }
            if (mode == 5) total += Mode5Alpha(cand, px, sel, tf, w3);
            weighted = Weighted(total);
            take = (which == 0) | (weighted < best);
        }
        for (int j = 0; j < BlockInts; j++)
        {
            if ((lanes == 1) | (j == lane))
                next[b + j] = take ? BestInt(j, weighted, mode, part, cand) : prev[b + j];
        }
    }

    /// <summary>Subset <paramref name="subset"/>'s endpoints, p-bits and selectors (at the pixel indices) into the candidate.</summary>
    private static void StoreCand(ArrayView<int> cand, ArrayView<int> res, int subset, ArrayView<int> sel, ArrayView<int> idx, int n)
    {
        int lowAt = subset == 0 ? CLow0 : CLow1, highAt = subset == 0 ? CHigh0 : CHigh1;
        for (int c = 0; c < 4; c++)
        {
            cand[lowAt + c] = res[c];
            cand[highAt + c] = res[4 + c];
        }
        cand[CPbits + subset * 2] = res[8];
        cand[CPbits + subset * 2 + 1] = res[9];
        for (int i = 0; i < n; i++) cand[CSel + idx[i]] = sel[i];
    }

    private static ulong BestErr(ArrayView<int> scratch, int b) => (uint)scratch[b + BErrLo] | ((ulong)(uint)scratch[b + BErrHi] << 32);

    /// <summary>bc7enc's (uint64_t)(err * weight + .5f) with the default weight 1: the error rounded through a float.</summary>
    private static ulong Weighted(ulong err) => (ulong)((float)err * 1.0f + .5f);

    /// <summary>Int j of the candidate packed as the block's best.</summary>
    private static int BestInt(int j, ulong err, int mode, int part, ArrayView<int> cand)
    {
        int v = 0;
        if (j == BErrLo) v = (int)(uint)err;
        else if (j == BErrHi) v = (int)(uint)(err >> 32);
        else if (j == BInfo)
            v = mode | (part << 4) | ((cand[CPbits] | (cand[CPbits + 1] << 1) | (cand[CPbits + 2] << 2) | (cand[CPbits + 3] << 3)) << 10);
        else if (j < BSel)
        {
            int at = (j - BEnd) * 4;
            v = cand[at] | (cand[at + 1] << 8) | (cand[at + 2] << 16) | (cand[at + 3] << 24);
        }
        else if (j < BASel)
        {
            int at = CSel + (j - BSel) * 8;
            for (int i = 0; i < 8; i++) v |= cand[at + i] << (i * 4);
        }
        else
            for (int i = 0; i < 16; i++) v |= cand[CASel + i] << (i * 2);
        return v;
    }

    /// <summary>handle_alpha_block_mode5's alpha part: the alpha endpoints and 2-bit selectors into the candidate; returns its error.</summary>
    private static ulong Mode5Alpha(ArrayView<int> cand, ArrayView<int> px, ArrayView<int> trialSel, ArrayView<float> tf, int w3)
    {
        int loA = 255, hiA = 0;
        for (int i = 0; i < 16; i++)
        {
            int a = px[i * 4 + 3];
            loA = a < loA ? a : loA;
            hiA = a > hiA ? a : hiA;
        }
        ulong alphaErr;
        if (loA == hiA)
        {
            alphaErr = 0;
            cand[CLow0 + 3] = loA;
            cand[CHigh0 + 3] = hiA;
            for (int i = 0; i < 16; i++) cand[CASel + i] = 0;
        }
        else
        {
            alphaErr = ulong.MaxValue;
            for (int pass = 0; pass < 2; pass++)
            {
                int v0 = loA, v3 = hiA;
                int v1 = (v0 * (64 - 21) + v3 * 21 + 32) >> 6;
                int v2 = (v0 * (64 - 43) + v3 * 43 + 32) >> 6;
                ulong trialErr = 0;
                for (int i = 0; i < 16; i++)
                {
                    int a = px[i * 4 + 3];
                    int sv = 0;
                    int be = Abs(a - v0);
                    int e = Abs(a - v1); if (e < be) { be = e; sv = 1; }
                    e = Abs(a - v2); if (e < be) { be = e; sv = 2; }
                    e = Abs(a - v3); if (e < be) { be = e; sv = 3; }
                    trialSel[i] = sv;
                    trialErr += (uint)(be * be) * (uint)w3;
                }
                if (trialErr < alphaErr)
                {
                    alphaErr = trialErr;
                    cand[CLow0 + 3] = loA;
                    cand[CHigh0 + 3] = hiA;
                    for (int i = 0; i < 16; i++) cand[CASel + i] = trialSel[i];
                }
                if (pass == 0)
                {
                    // compute_least_squares_endpoints_a over the 2-bit weights
                    float z00 = 0, z10 = 0, z11 = 0, q00 = 0, t = 0;
                    for (int i = 0; i < 16; i++)
                    {
                        int sl = trialSel[i];
                        z00 += tf[FW2 + sl * 4];
                        z10 += tf[FW2 + sl * 4 + 1];
                        z11 += tf[FW2 + sl * 4 + 2];
                        float w = tf[FW2 + sl * 4 + 3];
                        q00 += w * px[i * 4 + 3];
                        t += px[i * 4 + 3];
                    }
                    float q10 = t - q00;
                    float z01 = z10;
                    float det = z00 * z11 - z01 * z10;
                    if (det != 0.0f) det = 1.0f / det;
                    float iz00 = z11 * det, iz01 = -z01 * det, iz10 = -z10 * det, iz11 = z00 * det;
                    // (bc7enc's out-of-range fallback needs every alpha equal, which this branch excludes.)
                    float xl = iz00 * q00 + iz01 * q10, xh = iz10 * q00 + iz11 * q10;
                    int newLo = Clamp((int)XMath.Floor(xl + .5f), 0, 255);
                    int newHi = Clamp((int)XMath.Floor(xh + .5f), 0, 255);
                    if (newLo > newHi) { int tmp = newLo; newLo = newHi; newHi = tmp; }
                    if ((newLo == loA) & (newHi == hiA)) pass = 2;
                    else { loA = newLo; hiA = newHi; }
                }
            }
        }
        return alphaErr;
    }

    private static int Abs(int v) => v < 0 ? -v : v;
    private static int Clamp(int v, int lo, int hi) => v < lo ? lo : (v > hi ? hi : v);
    private static float Saturate(float v) => v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);

    // ------------------------------------------------------------------
    // color_cell_compression
    // ------------------------------------------------------------------

    /// <summary>
    /// bc7enc's color_cell_compression for mode 1, 5, 6 or 7 over the n pixels px[idx[0 .. n - 1]]: returns the best error,
    /// the selectors in sel[0 .. n - 1] and low / high / p-bits in res[0..9].
    /// </summary>
    private static ulong CellCompression(int mode, int n, bool hasAlpha, ArrayView<int> px, ArrayView<int> idx, ArrayView<int> sel,
        ArrayView<int> selTemp, ArrayView<int> res, ArrayView<int> ti, ArrayView<float> tf, int w0, int w1, int w2, int w3)
    {
        int numW = mode == 6 ? 16 : (mode == 1 ? 8 : 4);
        int compBits = mode == 1 ? 6 : (mode == 7 ? 5 : 7);
        bool hasPbits = mode != 5;
        bool sharePbit = mode == 1;
        ulong best = ulong.MaxValue;
        for (int i = 0; i < 10; i++) res[i] = 0;

        // Mode 1 / 7 subsets of one colour take the single-colour tables only.
        bool partitioned = (mode == 1) | (mode == 7);
        bool oneOnly = partitioned;
        int p0 = idx[0] * 4;
        for (int i = 1; i < n; i++)
        {
            int pi = idx[i] * 4;
            bool same = (px[pi] == px[p0]) & (px[pi + 1] == px[p0 + 1]) & (px[pi + 2] == px[p0 + 2]);
            if (mode == 7) same &= px[pi + 3] == px[p0 + 3];
            oneOnly &= same;
        }

        // Mean colour and principal axis.
        float m0 = 0, m1 = 0, m2 = 0, m3 = 0;
        for (int i = 0; i < n; i++)
        {
            int pi = idx[i] * 4;
            m0 += px[pi]; m1 += px[pi + 1]; m2 += px[pi + 2]; m3 += px[pi + 3];
        }
        float invN = 1.0f / (float)n;
        float s0 = m0 * invN, s1 = m1 * invN, s2 = m2 * invN, s3 = m3 * invN;
        float invN255 = 1.0f / (float)(n * 255.0f);
        float mc0 = Saturate(m0 * invN255), mc1 = Saturate(m1 * invN255), mc2 = Saturate(m2 * invN255), mc3 = Saturate(m3 * invN255);

        if (!oneOnly)
        {
            float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            if (hasAlpha)
            {
                for (int i = 0; i < n; i++)
                {
                    int pi = idx[i] * 4;
                    float c0 = px[pi] - s0, c1 = px[pi + 1] - s1, c2 = px[pi + 2] - s2, c3 = px[pi + 3] - s3;
                    float n0 = i != 0 ? a0 : c0, n1 = i != 0 ? a1 : c1, n2 = i != 0 ? a2 : c2, n3 = i != 0 ? a3 : c3;
                    float ss = n0 * n0 + n1 * n1 + n2 * n2 + n3 * n3;
                    if (ss != 0.0f)
                    {
                        ss = 1.0f / XMath.Sqrt(ss);
                        n0 *= ss; n1 *= ss; n2 *= ss; n3 *= ss;
                    }
                    float ax = c0 * c0 * n0 + c1 * c0 * n1 + c2 * c0 * n2 + c3 * c0 * n3;
                    float ay = c0 * c1 * n0 + c1 * c1 * n1 + c2 * c1 * n2 + c3 * c1 * n3;
                    float az = c0 * c2 * n0 + c1 * c2 * n1 + c2 * c2 * n2 + c3 * c2 * n3;
                    float aw = c0 * c3 * n0 + c1 * c3 * n1 + c2 * c3 * n2 + c3 * c3 * n3;
                    a0 += ax; a1 += ay; a2 += az; a3 += aw;
                }
                float ss2 = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
                if (ss2 != 0.0f)
                {
                    ss2 = 1.0f / XMath.Sqrt(ss2);
                    a0 *= ss2; a1 *= ss2; a2 *= ss2; a3 *= ss2;
                }
            }
            else
            {
                float c00 = 0, c01 = 0, c02 = 0, c11 = 0, c12 = 0, c22 = 0;
                for (int i = 0; i < n; i++)
                {
                    int pi = idx[i] * 4;
                    float r = px[pi] - s0, g = px[pi + 1] - s1, b = px[pi + 2] - s2;
                    c00 += r * r; c01 += r * g; c02 += r * b; c11 += g * g; c12 += g * b; c22 += b * b;
                }
                float vfr = .9f, vfg = 1.0f, vfb = .7f;
                for (int iter = 0; iter < 3; iter++)
                {
                    float r = vfr * c00 + vfg * c01 + vfb * c02;
                    float g = vfr * c01 + vfg * c11 + vfb * c12;
                    float b = vfr * c02 + vfg * c12 + vfb * c22;
                    float m = XMath.Max(XMath.Max(XMath.Abs(r), XMath.Abs(g)), XMath.Abs(b));
                    if (m > 1e-10f)
                    {
                        m = 1.0f / m;
                        r *= m; g *= m; b *= m;
                    }
                    vfr = r; vfg = g; vfb = b;
                }
                float len = vfr * vfr + vfg * vfg + vfb * vfb;
                if (len < 1e-10f) { a0 = 0; a1 = 0; a2 = 0; a3 = 0; }
                else
                {
                    len = 1.0f / XMath.Sqrt(len);
                    a0 = vfr * len; a1 = vfg * len; a2 = vfb * len; a3 = 0;
                }
            }
            if (a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3 < .5f)
            {
                a0 = .213f; a1 = .715f; a2 = .072f; a3 = hasAlpha ? .715f : 0.0f;
                float ss = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
                if (ss != 0.0f)
                {
                    ss = 1.0f / XMath.Sqrt(ss);
                    a0 *= ss; a1 *= ss; a2 *= ss; a3 *= ss;
                }
            }

            float l = 1e+9f, h = -1e+9f;
            for (int i = 0; i < n; i++)
            {
                int pi = idx[i] * 4;
                float d = (px[pi] - s0) * a0 + (px[pi + 1] - s1) * a1 + (px[pi + 2] - s2) * a2 + (px[pi + 3] - s3) * a3;
                l = d < l ? d : l;
                h = d > h ? d : h;
            }
            l *= 1.0f / 255.0f;
            h *= 1.0f / 255.0f;
            float lo0 = Saturate(mc0 + a0 * l), lo1 = Saturate(mc1 + a1 * l), lo2 = Saturate(mc2 + a2 * l), lo3 = Saturate(mc3 + a3 * l);
            float hi0 = Saturate(mc0 + a0 * h), hi1 = Saturate(mc1 + a1 * h), hi2 = Saturate(mc2 + a2 * h), hi3 = Saturate(mc3 + a3 * h);
            if (lo0 + lo1 + lo2 + lo3 > hi0 + hi1 + hi2 + hi3)
            {
                float t0 = lo0, t1 = lo1, t2 = lo2, t3 = lo3;
                lo0 = hi0; lo1 = hi1; lo2 = hi2; lo3 = hi3;
                hi0 = t0; hi1 = t1; hi2 = t2; hi3 = t3;
            }

            // PCA endpoints, then (pass 1) least squares from the selectors found.
            for (int pass = 0; pass < 2; pass++)
            {
                if (pass == 1)
                {
                    LeastSquares(mode, n, hasAlpha, px, idx, sel, tf, out lo0, out lo1, out lo2, out lo3, out hi0, out hi1, out hi2, out hi3);
                    lo0 *= 1.0f / 255.0f; lo1 *= 1.0f / 255.0f; lo2 *= 1.0f / 255.0f; lo3 *= 1.0f / 255.0f;
                    hi0 *= 1.0f / 255.0f; hi1 *= 1.0f / 255.0f; hi2 *= 1.0f / 255.0f; hi3 *= 1.0f / 255.0f;
                }
                best = FindOptimal(mode, n, hasAlpha, numW, compBits, hasPbits, sharePbit, lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3,
                    px, idx, sel, selTemp, res, best, ti, tf, w0, w1, w2, w3);
                if (best == 0) return 0;
            }
        }

        // Mode 1 / 7: the single colour (the subset's colour, or its mean, kept only when better).
        if (partitioned)
        {
            int r = (int)(.5f + mc0 * 255.0f), g = (int)(.5f + mc1 * 255.0f), b = (int)(.5f + mc2 * 255.0f), a = (int)(.5f + mc3 * 255.0f);
            if (oneOnly) { r = px[p0]; g = px[p0 + 1]; b = px[p0 + 2]; a = px[p0 + 3]; }
            ulong e = OneColour(mode, n, px, idx, ti, r, g, b, a, w0, w1, w2, w3, res);
            if (oneOnly | (e < best))
            {
                best = e;
                for (int i = 0; i < 10; i++) res[i] = res[16 + i];
                for (int i = 0; i < n; i++) sel[i] = mode == 1 ? 2 : 1;
            }
        }
        return best;
    }

    /// <summary>
    /// pack_mode1_to_one_color / pack_mode7_to_one_color: endpoints from the optimal single-colour tables into
    /// res[16 .. 25] (low[4], high[4], p-bits[2]); returns the error.
    /// </summary>
    private static ulong OneColour(int mode, int n, ArrayView<int> px, ArrayView<int> idx, ArrayView<int> ti,
        int r, int g, int b, int a, int w0, int w1, int w2, int w3, ArrayView<int> res)
    {
        bool m7 = mode == 7;
        int bestErr = int.MaxValue, bestP = 0;
        int pCount = m7 ? 4 : 2;
        for (int p = 0; p < pCount; p++)
        {
            int err;
            if (m7)
            {
                int o = (p >> 1) * 2 + (p & 1);
                err = (ti[TM7 + r * 4 + o] & 0xFFFF) + (ti[TM7 + g * 4 + o] & 0xFFFF) + (ti[TM7 + b * 4 + o] & 0xFFFF) + (ti[TM7 + a * 4 + o] & 0xFFFF);
            }
            else err = (ti[TM1 + r * 2 + p] & 0xFFFF) + (ti[TM1 + g * 2 + p] & 0xFFFF) + (ti[TM1 + b * 2 + p] & 0xFFFF);
            if (err < bestErr)
            {
                bestErr = err;
                bestP = p;
                if (bestErr == 0) p = pCount;
            }
        }
        int er, eg, eb, ea;
        if (m7)
        {
            int bo = (bestP >> 1) * 2 + (bestP & 1);
            er = ti[TM7 + r * 4 + bo]; eg = ti[TM7 + g * 4 + bo]; eb = ti[TM7 + b * 4 + bo]; ea = ti[TM7 + a * 4 + bo];
            res[24] = bestP & 1;
            res[25] = bestP >> 1;
        }
        else
        {
            er = ti[TM1 + r * 2 + bestP]; eg = ti[TM1 + g * 2 + bestP]; eb = ti[TM1 + b * 2 + bestP]; ea = 0;
            res[24] = bestP;
            res[25] = 0;
        }
        res[16] = (er >> 16) & 0xFF; res[17] = (eg >> 16) & 0xFF; res[18] = (eb >> 16) & 0xFF; res[19] = (ea >> 16) & 0xFF;
        res[20] = (er >> 24) & 0xFF; res[21] = (eg >> 24) & 0xFF; res[22] = (eb >> 24) & 0xFF; res[23] = (ea >> 24) & 0xFF;
        int pr, pg, pb, pa = 255;
        if (m7)
        {
            pr = OneColourValue7(er, bestP); pg = OneColourValue7(eg, bestP); pb = OneColourValue7(eb, bestP); pa = OneColourValue7(ea, bestP);
        }
        else
        {
            pr = OneColourValue1(er, bestP); pg = OneColourValue1(eg, bestP); pb = OneColourValue1(eb, bestP);
        }
        ulong total = 0;
        for (int i = 0; i < n; i++)
        {
            int pi = idx[i] * 4;
            uint e = DistRgb(pr, pg, pb, px[pi], px[pi + 1], px[pi + 2], w0, w1, w2);
            if (m7)
            {
                int da = pa - px[pi + 3];
                e += (uint)w3 * (uint)(da * da);
            }
            total += e;
        }
        return total;
    }

    private static int OneColourValue1(int entry, int p)
    {
        int lo = (entry >> 16) & 0xFF, hi = (entry >> 24) & 0xFF;
        int low = ((lo << 1) | p) << 1;
        low |= low >> 7;
        int high = ((hi << 1) | p) << 1;
        high |= high >> 7;
        return (low * (64 - 18) + high * 18 + 32) >> 6;
    }

    private static int OneColourValue7(int entry, int p)
    {
        int lo = (entry >> 16) & 0xFF, hi = (entry >> 24) & 0xFF;
        int low = (lo << 1) | (p & 1);
        int high = (hi << 1) | (p >> 1);
        low = (low << 2) | (low >> 6);
        high = (high << 2) | (high >> 6);
        return (low * (64 - 21) + high * 21 + 32) >> 6;
    }

    /// <summary>compute_color_distance_rgb, perceptual (YCbCr) form; e1 is the endpoint colour, e2 the pixel.</summary>
    private static uint DistRgb(int r1, int g1, int b1, int r2, int g2, int b2, int w0, int w1, int w2)
    {
        int l1 = r1 * 109 + g1 * 366 + b1 * 37;
        int cr1 = (r1 << 9) - l1;
        int cb1 = (b1 << 9) - l1;
        int l2 = r2 * 109 + g2 * 366 + b2 * 37;
        int cr2 = (r2 << 9) - l2;
        int cb2 = (b2 << 9) - l2;
        int dr = (l1 - l2) >> 8, dg = (cr1 - cr2) >> 8, db = (cb1 - cb2) >> 8;
        return (uint)w0 * (uint)(dr * dr) + (uint)w1 * (uint)(dg * dg) + (uint)w2 * (uint)(db * db);
    }

    /// <summary>compute_least_squares_endpoints_rgb / _rgba from the current selectors (sel[0 .. n - 1]).</summary>
    private static void LeastSquares(int mode, int n, bool hasAlpha, ArrayView<int> px, ArrayView<int> idx, ArrayView<int> sel, ArrayView<float> tf,
        out float l0, out float l1, out float l2, out float l3, out float h0, out float h1, out float h2, out float h3)
    {
        int fw = mode == 6 ? FW4 : (mode == 1 ? FW3 : FW2);
        float z00 = 0, z10 = 0, z11 = 0;
        float q0 = 0, q1 = 0, q2 = 0, q3 = 0, t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        for (int i = 0; i < n; i++)
        {
            int s = sel[i];
            z00 += tf[fw + s * 4];
            z10 += tf[fw + s * 4 + 1];
            z11 += tf[fw + s * 4 + 2];
            float w = tf[fw + s * 4 + 3];
            int pi = idx[i] * 4;
            q0 += w * px[pi]; t0 += px[pi];
            q1 += w * px[pi + 1]; t1 += px[pi + 1];
            q2 += w * px[pi + 2]; t2 += px[pi + 2];
            if (hasAlpha) { q3 += w * px[pi + 3]; t3 += px[pi + 3]; }
        }
        float r0 = t0 - q0, r1 = t1 - q1, r2 = t2 - q2, r3 = t3 - q3;
        float z01 = z10;
        float det = z00 * z11 - z01 * z10;
        if (det != 0.0f) det = 1.0f / det;
        float iz00 = z11 * det, iz01 = -z01 * det, iz10 = -z10 * det, iz11 = z00 * det;
        l0 = iz00 * q0 + iz01 * r0; h0 = iz10 * q0 + iz11 * r0;
        l1 = iz00 * q1 + iz01 * r1; h1 = iz10 * q1 + iz11 * r1;
        l2 = iz00 * q2 + iz01 * r2; h2 = iz10 * q2 + iz11 * r2;
        if (hasAlpha) { l3 = iz00 * q3 + iz01 * r3; h3 = iz10 * q3 + iz11 * r3; }
        else { l3 = 255.0f; h3 = 255.0f; }
        int channels = hasAlpha ? 4 : 3;
        for (int c = 0; c < channels; c++)
        {
            float lc = c == 0 ? l0 : (c == 1 ? l1 : (c == 2 ? l2 : l3));
            float hc = c == 0 ? h0 : (c == 1 ? h1 : (c == 2 ? h2 : h3));
            if ((lc < 0.0f) | (hc > 255.0f))
            {
                int loV = int.MaxValue, hiV = 0;
                for (int i = 0; i < n; i++)
                {
                    int v = px[idx[i] * 4 + c];
                    loV = v < loV ? v : loV;
                    hiV = v > hiV ? v : hiV;
                }
                if (loV == hiV)
                {
                    if (c == 0) { l0 = loV; h0 = hiV; }
                    else if (c == 1) { l1 = loV; h1 = hiV; }
                    else if (c == 2) { l2 = loV; h2 = hiV; }
                    else { l3 = loV; h3 = hiV; }
                }
            }
        }
    }

    /// <summary>
    /// find_optimal_solution: quantize the float endpoints (with the best p-bits), then evaluate them when they differ from
    /// the best so far. Returns the best error.
    /// </summary>
    private static ulong FindOptimal(int mode, int n, bool hasAlpha, int numW, int compBits, bool hasPbits, bool sharePbit,
        float xl0, float xl1, float xl2, float xl3, float xh0, float xh1, float xh2, float xh3,
        ArrayView<int> px, ArrayView<int> idx, ArrayView<int> sel, ArrayView<int> selTemp, ArrayView<int> res, ulong best,
        ArrayView<int> ti, ArrayView<float> tf, int w0, int w1, int w2, int w3)
    {
        xl0 = Saturate(xl0); xl1 = Saturate(xl1); xl2 = Saturate(xl2); xl3 = Saturate(xl3);
        xh0 = Saturate(xh0); xh1 = Saturate(xh1); xh2 = Saturate(xh2); xh3 = Saturate(xh3);
        int bl0, bl1, bl2, bl3, bh0, bh1, bh2, bh3, bp0, bp1;
        if (hasPbits)
        {
            int iscalep = (1 << (compBits + 1)) - 1;
            float scalep = iscalep;
            int totalComps = hasAlpha ? 4 : 3;
            bl0 = bl1 = bl2 = bl3 = bh0 = bh1 = bh2 = bh3 = 0;
            bp0 = bp1 = 0;
            if (!sharePbit)
            {
                float bestErr0 = 1e+9f, bestErr1 = 1e+9f;
                for (int p = 0; p < 2; p++)
                {
                    int ql0 = QuantP(compBits, xl0, p, scalep, iscalep, tf), ql1 = QuantP(compBits, xl1, p, scalep, iscalep, tf);
                    int ql2 = QuantP(compBits, xl2, p, scalep, iscalep, tf), ql3 = QuantP(compBits, xl3, p, scalep, iscalep, tf);
                    int qh0 = QuantP(compBits, xh0, p, scalep, iscalep, tf), qh1 = QuantP(compBits, xh1, p, scalep, iscalep, tf);
                    int qh2 = QuantP(compBits, xh2, p, scalep, iscalep, tf), qh3 = QuantP(compBits, xh3, p, scalep, iscalep, tf);
                    int sn = compBits + 1;
                    float err0 = Sq(Scale(ql0, sn) - xl0 * 255.0f) + Sq(Scale(ql1, sn) - xl1 * 255.0f) + Sq(Scale(ql2, sn) - xl2 * 255.0f);
                    float err1 = Sq(Scale(qh0, sn) - xh0 * 255.0f) + Sq(Scale(qh1, sn) - xh1 * 255.0f) + Sq(Scale(qh2, sn) - xh2 * 255.0f);
                    if (totalComps == 4)
                    {
                        err0 += Sq(Scale(ql3, sn) - xl3 * 255.0f);
                        err1 += Sq(Scale(qh3, sn) - xh3 * 255.0f);
                    }
                    if (err0 < bestErr0)
                    {
                        bestErr0 = err0;
                        bp0 = p;
                        bl0 = ql0 >> 1; bl1 = ql1 >> 1; bl2 = ql2 >> 1; bl3 = ql3 >> 1;
                    }
                    if (err1 < bestErr1)
                    {
                        bestErr1 = err1;
                        bp1 = p;
                        bh0 = qh0 >> 1; bh1 = qh1 >> 1; bh2 = qh2 >> 1; bh3 = qh3 >> 1;
                    }
                }
            }
            else
            {
                float bestErr = 1e+9f;
                for (int p = 0; p < 2; p++)
                {
                    int ql0 = QuantP(compBits, xl0, p, scalep, iscalep, tf), ql1 = QuantP(compBits, xl1, p, scalep, iscalep, tf);
                    int ql2 = QuantP(compBits, xl2, p, scalep, iscalep, tf), ql3 = QuantP(compBits, xl3, p, scalep, iscalep, tf);
                    int qh0 = QuantP(compBits, xh0, p, scalep, iscalep, tf), qh1 = QuantP(compBits, xh1, p, scalep, iscalep, tf);
                    int qh2 = QuantP(compBits, xh2, p, scalep, iscalep, tf), qh3 = QuantP(compBits, xh3, p, scalep, iscalep, tf);
                    int sn = compBits + 1;
                    float err = Sq(Scale(ql0, sn) / 255.0f - xl0) + Sq(Scale(qh0, sn) / 255.0f - xh0);
                    err += Sq(Scale(ql1, sn) / 255.0f - xl1) + Sq(Scale(qh1, sn) / 255.0f - xh1);
                    err += Sq(Scale(ql2, sn) / 255.0f - xl2) + Sq(Scale(qh2, sn) / 255.0f - xh2);
                    if (totalComps == 4) err += Sq(Scale(ql3, sn) / 255.0f - xl3) + Sq(Scale(qh3, sn) / 255.0f - xh3);
                    if (err < bestErr)
                    {
                        bestErr = err;
                        bp0 = p; bp1 = p;
                        bl0 = ql0 >> 1; bl1 = ql1 >> 1; bl2 = ql2 >> 1; bl3 = ql3 >> 1;
                        bh0 = qh0 >> 1; bh1 = qh1 >> 1; bh2 = qh2 >> 1; bh3 = qh3 >> 1;
                    }
                }
            }
            if (mode == 1)
            {
                int iscale = iscalep >> 1;
                FixDegenerate(ref bl0, ref bh0, xl0, xh0, iscale);
                FixDegenerate(ref bl1, ref bh1, xl1, xh1, iscale);
                FixDegenerate(ref bl2, ref bh2, xl2, xh2, iscale);
            }
        }
        else
        {
            // Mode 5 (7-bit, no p-bits): the mode 5 midpoints.
            bl0 = Quant5(xl0, tf); bl1 = Quant5(xl1, tf); bl2 = Quant5(xl2, tf); bl3 = Quant5(xl3, tf);
            bh0 = Quant5(xh0, tf); bh1 = Quant5(xh1, tf); bh2 = Quant5(xh2, tf); bh3 = Quant5(xh3, tf);
            bp0 = res[8]; bp1 = res[9];
        }

        bool differs = (best == ulong.MaxValue) | (bl0 != res[0]) | (bl1 != res[1]) | (bl2 != res[2]) | (bl3 != res[3])
            | (bh0 != res[4]) | (bh1 != res[5]) | (bh2 != res[6]) | (bh3 != res[7]);
        if (hasPbits) differs |= (bp0 != res[8]) | (bp1 != res[9]);
        if (differs)
        {
            ulong err = Evaluate(mode, n, hasAlpha, numW, compBits, hasPbits, sharePbit, bl0, bl1, bl2, bl3, bh0, bh1, bh2, bh3, bp0, bp1,
                px, idx, selTemp, ti, w0, w1, w2, w3);
            if (err < best)
            {
                best = err;
                res[0] = bl0; res[1] = bl1; res[2] = bl2; res[3] = bl3;
                res[4] = bh0; res[5] = bh1; res[6] = bh2; res[7] = bh3;
                res[8] = bp0; res[9] = bp1;
                for (int i = 0; i < n; i++) sel[i] = selTemp[i];
            }
        }
        return best;
    }

    private static float Sq(float v) => v * v;

    /// <summary>scale_color for one channel: an n-bit value (p-bit included) to 8 bits.</summary>
    private static int Scale(int v, int n)
    {
        int s = v << (8 - n);
        return s | (s >> n);
    }

    /// <summary>One channel with p-bit p, as find_optimal_solution quantizes it (mode 7: midpoints; mode 1: midpoints; mode 6: rounding).</summary>
    private static int QuantP(int compBits, float x, int p, float scalep, int iscalep, ArrayView<float> tf)
    {
        if (compBits == 5)
        {
            int v = (int)(x * 31.0f);
            if (x > tf[FM7 + v * 2 + p]) v++;
            return Clamp(v * 2 + p, p, 63 - 1 + p);
        }
        if (compBits == 6)
        {
            int v = (int)(x * 63.0f);
            if (x > tf[FM1 + v * 2 + p]) v++;
            return Clamp(v * 2 + p, p, 127 - 1 + p);
        }
        return Clamp(((int)((x * scalep - p) / 2.0f + .5f)) * 2 + p, p, iscalep - 1 + p);
    }

    private static int Quant5(float x, ArrayView<float> tf)
    {
        int v = (int)(x * 127.0f);
        if (x > tf[FM5 + v]) v++;
        return Clamp(v, 0, 127);
    }

    /// <summary>fixDegenerateEndpoints (mode 1): a channel collapsed to one value while the float endpoints differ.</summary>
    private static void FixDegenerate(ref int lo, ref int hi, float xl, float xh, int iscale)
    {
        if ((lo == hi) & (XMath.Abs(xl - xh) > 0.0f))
        {
            if (lo > (iscale >> 1))
            {
                if (lo > 0) lo--;
                else if (hi < iscale) hi++;
            }
            else
            {
                if (hi < iscale) hi++;
                else if (lo > 0) lo--;
            }
        }
    }

    /// <summary>
    /// evaluate_solution (perceptual): the interpolated colours of the quantized endpoints, every pixel's nearest by the
    /// perceptual distance; selectors into selTemp, returns the error.
    /// </summary>
    private static ulong Evaluate(int mode, int n, bool hasAlpha, int numW, int compBits, bool hasPbits, bool sharePbit,
        int l0, int l1, int l2, int l3, int h0, int h1, int h2, int h3, int p0, int p1,
        ArrayView<int> px, ArrayView<int> idx, ArrayView<int> selTemp, ArrayView<int> ti, int w0, int w1, int w2, int w3)
    {
        int sn = compBits + (hasPbits ? 1 : 0);
        int minP = p0, maxP = sharePbit ? p0 : p1;
        int q0 = l0, q1 = l1, q2 = l2, q3 = l3, r0 = h0, r1 = h1, r2 = h2, r3 = h3;
        if (hasPbits)
        {
            q0 = ((l0 << 1) | minP) & 0xFF; q1 = ((l1 << 1) | minP) & 0xFF; q2 = ((l2 << 1) | minP) & 0xFF; q3 = ((l3 << 1) | minP) & 0xFF;
            r0 = ((h0 << 1) | maxP) & 0xFF; r1 = ((h1 << 1) | maxP) & 0xFF; r2 = ((h2 << 1) | maxP) & 0xFF; r3 = ((h3 << 1) | maxP) & 0xFF;
        }
        q0 = Scale(q0, sn) & 0xFF; q1 = Scale(q1, sn) & 0xFF; q2 = Scale(q2, sn) & 0xFF; q3 = Scale(q3, sn) & 0xFF;
        r0 = Scale(r0, sn) & 0xFF; r1 = Scale(r1, sn) & 0xFF; r2 = Scale(r2, sn) & 0xFF; r3 = Scale(r3, sn) & 0xFF;
        int wBase = numW == 16 ? TW4 : (numW == 8 ? TW3 : TW2);
        uint total = 0;
        for (int i = 0; i < n; i++)
        {
            int pi = idx[i] * 4;
            uint bestE = uint.MaxValue;
            int bestS = 0;
            for (int j = 0; j < numW; j++)
            {
                int c0, c1, c2, c3;
                if (j == 0) { c0 = q0; c1 = q1; c2 = q2; c3 = q3; }
                else if (j == numW - 1) { c0 = r0; c1 = r1; c2 = r2; c3 = r3; }
                else
                {
                    int w = ti[wBase + j];
                    c0 = (q0 * (64 - w) + r0 * w + 32) >> 6;
                    c1 = (q1 * (64 - w) + r1 * w + 32) >> 6;
                    c2 = (q2 * (64 - w) + r2 * w + 32) >> 6;
                    c3 = (q3 * (64 - w) + r3 * w + 32) >> 6;
                }
                uint e = DistRgb(c0, c1, c2, px[pi], px[pi + 1], px[pi + 2], w0, w1, w2);
                if (hasAlpha)
                {
                    int da = c3 - px[pi + 3];
                    e += (uint)w3 * (uint)(da * da);
                }
                if (e < bestE) { bestE = e; bestS = j; }
            }
            total += bestE;
            selTemp[i] = bestS;
        }
        return total;
    }

    // ------------------------------------------------------------------
    // estimate_partition (modes 1 and 7)
    // ------------------------------------------------------------------

    private static int EstimatePartition(int mode, ArrayView<int> px, ArrayView<int> idx, ArrayView<int> est, ArrayView<int> ti, int w0, int w1, int w2, int w3)
    {
        ulong bestErr = ulong.MaxValue;
        int bestPartition = 0, bestKey = 0;
        bool going = true;
        for (int iter = 0; (iter < 64) & going; iter++)
        {
            if (bestErr == 0) { going = false; }
            else
            {
                int partition = ti[TOrder + iter];
                bool skip = false;
                if ((iter >= 14) & (iter <= 34))
                {
                    uint mask = 1u << (bestKey + 1);
                    if (((uint)ti[TPred + partition] & mask) == 0)
                    {
                        if (iter == 34) going = false;
                        skip = true;
                    }
                }
                if (going & !skip)
                {
                    ulong total = 0;
                    for (int subset = 0; (subset < 2) & (total < bestErr); subset++)
                    {
                        int n = 0;
                        for (int i = 0; i < 16; i++)
                            if (ti[TP2 + partition * 16 + i] == subset) { idx[n] = i; n++; }
                        total += EstimateSubset(mode, n, px, idx, est, ti, bestErr, w0, w1, w2, w3);
                    }
                    if (total < bestErr)
                    {
                        bestErr = total;
                        bestPartition = partition;
                    }
                    if ((partition == 34) & (bestPartition != 34)) going = false;
                    if (iter == 13) bestKey = bestPartition;
                }
            }
        }
        return bestPartition;
    }

    /// <summary>color_cell_compression_est_mode1 / _mode7 (perceptual): bounding-box endpoints, thresholded selectors.</summary>
    private static ulong EstimateSubset(int mode, int n, ArrayView<int> px, ArrayView<int> idx, ArrayView<int> est, ArrayView<int> ti, ulong bestSoFar,
        int w0, int w1, int w2, int w3)
    {
        int lr = 255, lg = 255, lb = 255, la = 255, hr = 0, hg = 0, hb = 0, ha = 0;
        for (int i = 0; i < n; i++)
        {
            int pi = idx[i] * 4;
            lr = px[pi] < lr ? px[pi] : lr; hr = px[pi] > hr ? px[pi] : hr;
            lg = px[pi + 1] < lg ? px[pi + 1] : lg; hg = px[pi + 1] > hg ? px[pi + 1] : hg;
            lb = px[pi + 2] < lb ? px[pi + 2] : lb; hb = px[pi + 2] > hb ? px[pi + 2] : hb;
            la = px[pi + 3] < la ? px[pi + 3] : la; ha = px[pi + 3] > ha ? px[pi + 3] : ha;
        }
        bool m7 = mode == 7;
        int numW = m7 ? 4 : 8;
        int wBase = m7 ? TW2 : TW3;
        if (!m7) { la = 0; ha = 0; }
        int ar = hr - lr, ag = hg - lg, ab = hb - lb, aa = ha - la;
        // Per interpolated colour k: est[k] = Y, est[8 + k] = Cr, est[16 + k] = Cb, est[24 + k] = alpha; est[32 + k] the
        // threshold between k and k + 1 (midway between neighbouring dots with the box diagonal).
        int prevDot = 0;
        for (int k = 0; k < numW; k++)
        {
            int w = ti[wBase + k];
            int cr = (lr * (64 - w) + hr * w + 32) >> 6, cg = (lg * (64 - w) + hg * w + 32) >> 6;
            int cb = (lb * (64 - w) + hb * w + 32) >> 6, ca = (la * (64 - w) + ha * w + 32) >> 6;
            if (k == 0) { cr = lr; cg = lg; cb = lb; ca = la; }
            if (k == numW - 1) { cr = hr; cg = hg; cb = hb; ca = ha; }
            int y = cr * 109 + cg * 366 + cb * 37;
            est[k] = y;
            est[8 + k] = (cr << 9) - y;
            est[16 + k] = (cb << 9) - y;
            est[24 + k] = ca;
            int dot = cr * ar + cg * ag + cb * ab + ca * aa;
            if (k > 0) est[32 + k - 1] = (prevDot + dot + 1) >> 1;
            prevDot = dot;
        }
        ulong total = 0;
        bool going = true;
        for (int i = 0; (i < n) & going; i++)
        {
            int pi = idx[i] * 4;
            int d = ar * px[pi] + ag * px[pi + 1] + ab * px[pi + 2] + (m7 ? aa * px[pi + 3] : 0);
            // The approximate selector: the highest whose threshold d reaches (the thresholds rise with k).
            int s = 0;
            for (int k = 0; k < numW - 1; k++)
                if (d >= est[32 + k]) s = k + 1;
            int l2 = px[pi] * 109 + px[pi + 1] * 366 + px[pi + 2] * 37;
            int cr2 = (px[pi] << 9) - l2, cb2 = (px[pi + 2] << 9) - l2;
            int dl = (est[s] - l2) >> 8, dcr = (est[8 + s] - cr2) >> 8, dcb = (est[16 + s] - cb2) >> 8;
            int ie = w0 * dl * dl + w1 * dcr * dcr + w2 * dcb * dcb;
            if (m7)
            {
                int dca = px[pi + 3] - est[24 + s];
                ie += w3 * dca * dca;
            }
            total += (ulong)(long)ie;
            if (total > bestSoFar) going = false;
        }
        return total;
    }

    // ------------------------------------------------------------------
    // Choice and packing
    // ------------------------------------------------------------------

    /// <summary>One thread per output word: that 32-bit word of the block's best, as encode_bc7_block writes it.</summary>
    private static void PackKernel(Index1D word, ArrayView<int> scratch, ArrayView<byte> blocks, ArrayView<int> ti)
    {
        uint v = PackWord(scratch, (word >> 2) * BlockInts, ti, word & 3);
        int o = word * 4;
        blocks[o] = (byte)v;
        blocks[o + 1] = (byte)(v >> 8);
        blocks[o + 2] = (byte)(v >> 16);
        blocks[o + 3] = (byte)(v >> 24);
    }

    private static int BSelector(ArrayView<int> scratch, int b, int i) => (scratch[b + BSel + (i >> 3)] >> ((i & 7) * 4)) & 15;
    private static int BAlphaSelector(ArrayView<int> scratch, int b, int i) => (scratch[b + BASel] >> (i * 2)) & 3;
    /// <summary>Endpoint channel c of subset k (high: the high endpoint).</summary>
    private static int BEndpoint(ArrayView<int> scratch, int b, int k, bool high, int c) => (scratch[b + BEnd + k * 2 + (high ? 1 : 0)] >> (c * 8)) & 0xFF;
    private static int BPbit(ArrayView<int> scratch, int b, int k, int j) => (scratch[b + BInfo] >> (10 + k * 2 + j)) & 1;

    /// <summary>encode_bc7_block for modes 1, 5, 6, 7 (no rotation, index selector 0): word wi of the 128-bit block.</summary>
    private static uint PackWord(ArrayView<int> scratch, int s, ArrayView<int> ti, int wi)
    {
        int mode = scratch[s + BInfo] & 15;
        int subsets = (mode == 1) | (mode == 7) ? 2 : 1;
        int part = (scratch[s + BInfo] >> 4) & 63;
        int anchor1 = subsets == 2 ? ti[TA2 + part] : -1;
        int colourBits = mode == 6 ? 4 : (mode == 1 ? 3 : 2);
        int alphaBits = mode == 5 ? 2 : 0;
        int cPrec = mode == 1 ? 6 : (mode == 7 ? 5 : 7);
        int aPrec = mode == 5 ? 8 : (mode == 6 ? 7 : (mode == 7 ? 5 : 0));
        bool separateAlpha = mode == 5;
        bool sharedP = mode == 1;
        bool hasP = mode != 5;

        // Per subset: whether its anchor's selector MSB is set (then flip selectors and swap endpoints).
        bool flip0 = (BSelector(scratch, s, 0) & (1 << (colourBits - 1))) != 0;
        bool flip1 = (subsets == 2) && (BSelector(scratch, s, anchor1) & (1 << (colourBits - 1))) != 0;
        bool aflip = separateAlpha && (BAlphaSelector(scratch, s, 0) & 2) != 0;

        ulong lo = 0, hi = 0;
        int pos = 0;
        Put(ref lo, ref hi, ref pos, 1u << mode, mode + 1);
        if (mode == 5) Put(ref lo, ref hi, ref pos, 0, 2);     // rotation
        if (subsets == 2) Put(ref lo, ref hi, ref pos, (uint)part, 6);
        int comps = mode >= 4 ? 4 : 3;
        for (int c = 0; c < comps; c++)
        {
            int prec = c == 3 ? aPrec : cPrec;
            for (int k = 0; k < subsets; k++)
            {
                bool f = k == 0 ? flip0 : flip1;
                // Mode 5 swaps colour (c < 3) on the colour flip and alpha on the alpha flip; the others swap whole endpoints.
                bool swap = separateAlpha ? (c < 3 ? f : aflip) : f;
                int vl = BEndpoint(scratch, s, k, swap, c);
                int vh = BEndpoint(scratch, s, k, !swap, c);
                Put(ref lo, ref hi, ref pos, (uint)vl, prec);
                Put(ref lo, ref hi, ref pos, (uint)vh, prec);
            }
        }
        if (hasP)
        {
            for (int k = 0; k < subsets; k++)
            {
                bool f = k == 0 ? flip0 : flip1;
                int pa = BPbit(scratch, s, k, 0), pb = BPbit(scratch, s, k, 1);
                if (sharedP) Put(ref lo, ref hi, ref pos, (uint)pa, 1);
                else
                {
                    Put(ref lo, ref hi, ref pos, (uint)(f ? pb : pa), 1);
                    Put(ref lo, ref hi, ref pos, (uint)(f ? pa : pb), 1);
                }
            }
        }
        int maxC = (1 << colourBits) - 1;
        for (int i = 0; i < 16; i++)
        {
            int k = subsets == 2 ? ti[TP2 + part * 16 + i] : 0;
            bool f = k == 0 ? flip0 : flip1;
            int sv = BSelector(scratch, s, i);
            if (f) sv = maxC - sv;
            int nbits = colourBits - (((i == 0) | (i == anchor1)) ? 1 : 0);
            Put(ref lo, ref hi, ref pos, (uint)sv, nbits);
        }
        if (separateAlpha)
        {
            int maxA = (1 << alphaBits) - 1;
            for (int i = 0; i < 16; i++)
            {
                int sv = BAlphaSelector(scratch, s, i);
                if (aflip) sv = maxA - sv;
                Put(ref lo, ref hi, ref pos, (uint)sv, alphaBits - (i == 0 ? 1 : 0));
            }
        }
        return wi == 0 ? (uint)lo : (wi == 1 ? (uint)(lo >> 32) : (wi == 2 ? (uint)hi : (uint)(hi >> 32)));
    }

    /// <summary>Appends n bits of v (LSB first) to the 128-bit block (lo, hi).</summary>
    private static void Put(ref ulong lo, ref ulong hi, ref int pos, uint v, int n)
    {
        ulong val = v & ((n >= 32) ? 0xFFFFFFFFul : ((1ul << n) - 1));
        if (pos < 64)
        {
            lo |= val << pos;
            if (pos + n > 64) hi |= val >> (64 - pos);
        }
        else hi |= val << (pos - 64);
        pos += n;
    }
}
