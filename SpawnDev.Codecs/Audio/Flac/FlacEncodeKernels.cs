// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Kernels of the FLAC stream encoder (FlacEncoderGpu.EncodeAsync), written from RFC 9639 (the FLAC format) and the
// textbook linear prediction method (windowed autocorrelation, Levinson-Durbin recursion, coefficient quantization with
// error feedback) and Rice / Golomb coding. Every stage that touches samples or bits runs here; the host only uploads,
// dispatches and reads back. A chunk of frames goes through:
//   * Prep: one thread per (frame, candidate channel): constant detection, wasted low bits, sample range check;
//   * Window + Autocorr: Tukey(0.5) window (one thread per sample), autocorrelation lags (one thread per lag);
//   * Lpc: one thread per subframe: Levinson-Durbin up to the level's order, every order quantized (15-bit precision);
//   * PartSum: one thread per (subframe, predictor, finest Rice partition): sum of zigzag-mapped residuals;
//   * PredCost: one thread per (subframe, predictor): the cheapest partition order by estimated Rice bits;
//   * Subframe: one thread per subframe: the cheapest predictor, exact Rice parameters and bits per partition (or an
//     escape-coded partition of raw residuals when that is smaller, e.g. 0 bits for an all-zero run), verbatim fallback;
//   * FrameSize: one thread per frame: the stereo decorrelation (independent / left-side / right-side / mid-side) with
//     the fewest bits, the frame's byte count; Offsets: one thread, where each frame of the chunk starts;
//   * WriteFrame: one thread per frame: header (CRC-8), subframes, padding, CRC-16, written at the frame's offset.
// A "candidate" is a signal a subframe can code: the channels, plus mid and side for stereo (candidates 0 L, 1 R,
// 2 mid = (L + R) >> 1, 3 side = L - R, which carries one extra bit).

using ILGPU;
using ILGPU.Algorithms;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>Kernels of the FLAC stream encoder (see the file header).</summary>
public static class FlacEncodeKernels
{
    /// <summary>Parameter ints (the prm view).</summary>
    public const int PChannels = 0, PBps = 1, PCands = 2, PBlock = 3, PLastN = 4, PFrames = 5, PMaxFixed = 6, PMaxLpc = 7,
        PMaxPo = 8, PPrecision = 9, PRateCode = 10, PRate = 11, PBpsCode = 12, PStereo = 13, PHeaderLen = 14, PPreds = 15,
        PrmInts = 16;

    /// <summary>Limits: candidates per frame, LPC order, predictors (fixed 0..4 then LPC 1..12), Rice partition order.</summary>
    public const int MaxCands = 8, MaxLpcOrder = 12, MaxPreds = 5 + MaxLpcOrder, MaxPo = 8, MaxParts = 1 << MaxPo, Lags = MaxLpcOrder + 1;

    /// <summary>Subframe record ints: constant flag and value, wasted bits, the coding chosen, LPC coefficients, Rice parameters.</summary>
    public const int SConst = 0, SConstValue = 1, SWasted = 2, SType = 3, SOrder = 4, SPo = 5, SMethod = 6, SShift = 7,
        SCoef = 8, SK = SCoef + MaxLpcOrder, SubInts = SK + MaxParts;

    /// <summary>Subframe types (SType).</summary>
    public const int TConstant = 0, TVerbatim = 1, TFixed = 2, TLpc = 3;

    /// <summary>Per-frame ints: channel assignment code, frame bytes, the candidate coded as each channel.</summary>
    public const int FCode = 0, FBytes = 1, FCand = 2, FrameInts = FCand + MaxCands;

    /// <summary>Meta longs: running byte offset, smallest and largest frame, out-of-range sample flag.</summary>
    public const int MRun = 0, MMinFrame = 1, MMaxFrame = 2, MRangeError = 3, MetaLongs = 4;

    /// <summary>Quantized LPC ints per subframe: for each order, MaxLpcOrder coefficients then the shift (-1: unusable).</summary>
    public const int LpcStride = MaxLpcOrder + 1, LpcInts = MaxLpcOrder * LpcStride;

    /// <summary>The CRC tables view: CRC-16 (polynomial 0x8005) at [0, 256), CRC-8 (polynomial 0x07) at [256, 512).</summary>
    public static int[] BuildCrcTables()
    {
        var t = new int[512];
        for (int i = 0; i < 256; i++)
        {
            int c = i << 8;
            for (int k = 0; k < 8; k++) c = (c & 0x8000) != 0 ? ((c << 1) ^ 0x8005) & 0xFFFF : (c << 1) & 0xFFFF;
            t[i] = c;
            int d = i;
            for (int k = 0; k < 8; k++) d = (d & 0x80) != 0 ? ((d << 1) ^ 0x07) & 0xFF : (d << 1) & 0xFF;
            t[256 + i] = d;
        }
        return t;
    }

    // ------------------------------------------------------------------ shared helpers

    /// <summary>Samples per channel of frame <paramref name="f"/>.</summary>
    private static int FrameN(ArrayView<int> prm, int f)
    {
        int n = prm[PBlock];
        if (f == prm[PFrames] - 1) n = prm[PLastN];
        return n;
    }

    /// <summary>Sample <paramref name="i"/> of candidate <paramref name="cand"/> in the frame starting at <paramref name="frameStart"/>.</summary>
    private static long Cand(ArrayView<int> input, int channels, int stereo, long frameStart, int i, int cand)
    {
        long at = (frameStart + i) * channels;
        if (stereo < 0) return input[at + cand];
        long l = input[at];
        long r = input[at + 1];
        long v = l - r;
        if (cand == 0) v = l;
        else if (cand == 1) v = r;
        else if (cand == 2) v = (l + r) >> 1;
        return v;
    }

    /// <summary>Bits per sample of a candidate (the side signal needs one more).</summary>
    private static int CandBps(ArrayView<int> prm, int cand)
    {
        int b = prm[PBps];
        if (prm[PStereo] >= 0 && cand == 3) b++;
        return b;
    }

    /// <summary>The finest Rice partition order for a block of <paramref name="n"/> samples (n divisible by 2^order).</summary>
    private static int FinestPo(int n, int maxPo)
    {
        int po = 0;
        bool go = po < maxPo && (n % (2 << po)) == 0;
        while (go)
        {
            po++;
            go = po < maxPo && (n % (2 << po)) == 0;
        }
        return po;
    }

    /// <summary>
    /// Residual of sample <paramref name="i"/> (>= order) under predictor <paramref name="pred"/>: fixed polynomial
    /// orders 0..4 (pred 0..4) or the quantized LPC of order pred - 4 whose coefficients start at coefs[coefBase].
    /// </summary>
    private static long Residual(ArrayView<int> input, int channels, int stereo, long frameStart, int cand, int wasted, int i,
        int pred, ArrayView<int> coefs, long coefBase, int shift)
    {
        long x0 = Cand(input, channels, stereo, frameStart, i, cand) >> wasted;
        if (pred == 0) return x0;
        if (pred < 5)
        {
            long x1 = Cand(input, channels, stereo, frameStart, i - 1, cand) >> wasted;
            if (pred == 1) return x0 - x1;
            long x2 = Cand(input, channels, stereo, frameStart, i - 2, cand) >> wasted;
            if (pred == 2) return x0 - 2 * x1 + x2;
            long x3 = Cand(input, channels, stereo, frameStart, i - 3, cand) >> wasted;
            if (pred == 3) return x0 - 3 * x1 + 3 * x2 - x3;
            long x4 = Cand(input, channels, stereo, frameStart, i - 4, cand) >> wasted;
            return x0 - 4 * x1 + 6 * x2 - 4 * x3 + x4;
        }
        int order = pred - 4;
        long sum = 0;
        for (int j = 0; j < order; j++)
            sum += coefs[coefBase + j] * (Cand(input, channels, stereo, frameStart, i - 1 - j, cand) >> wasted);
        return x0 - (sum >> shift);
    }

    private static long ZigZag(long e)
    {
        long u = 2 * e;
        if (e < 0) u = -2 * e - 1;
        return u;
    }

    /// <summary>Estimated Rice bits of m residuals whose zigzag values sum to s, parameter k (low k bits taken as uniform).</summary>
    private static long EstBits(long s, long m, int k)
    {
        long q = (2 * s - m * ((1L << k) - 1)) >> (k + 1);
        if (q < 0) q = 0;
        return m * (k + 1) + q;
    }

    /// <summary>The Rice parameter (0..30) with the fewest estimated bits for m residuals summing to s.</summary>
    private static int EstK(long s, long m)
    {
        if (m <= 0) return 0;
        int k = 0;
        bool go = (m << (k + 1)) <= s;
        while (go)
        {
            k++;
            go = k < 30 && (m << (k + 1)) <= s;
        }
        int best = k;
        long bestBits = EstBits(s, m, k);
        if (k > 0)
        {
            long b = EstBits(s, m, k - 1);
            if (b < bestBits) { bestBits = b; best = k - 1; }
        }
        if (k < 30)
        {
            long b = EstBits(s, m, k + 1);
            if (b < bestBits) { best = k + 1; }
        }
        return best;
    }

    /// <summary>Order of predictor <paramref name="pred"/> (fixed 0..4, then LPC 1..).</summary>
    private static int PredOrder(int pred)
    {
        int order = pred;
        if (pred >= 5) order = pred - 4;
        return order;
    }

    /// <summary>Whether predictor <paramref name="pred"/> may code a subframe of <paramref name="n"/> samples at this level.</summary>
    private static bool PredUsable(ArrayView<int> prm, ArrayView<int> lpcQ, long qBase, int pred, int n)
    {
        int order = PredOrder(pred);
        bool ok = order < n;
        if (pred < 5)
        {
            if (pred > prm[PMaxFixed]) ok = false;
        }
        else
        {
            if (order > prm[PMaxLpc]) ok = false;
            else if (lpcQ[qBase + (order - 1) * LpcStride + MaxLpcOrder] < 0) ok = false;
        }
        return ok;
    }

    /// <summary>Header bits of a subframe beyond its residual: type byte, wasted bits, warm-up, LPC precision/shift/coefficients, residual method + order.</summary>
    private static long Overhead(int pred, int wasted, int ebps, int precision)
    {
        int order = PredOrder(pred);
        long bits = 8 + wasted + (long)order * ebps + 6;
        if (pred >= 5) bits += 4 + 5 + (long)order * precision;
        return bits;
    }

    // ------------------------------------------------------------------ analysis

    /// <summary>One thread per window sample: Tukey(0.5) over the block size at [0, block), over the last block at [block, ...).</summary>
    public static void WindowKernel(Index1D index, ArrayView<float> win, ArrayView<int> prm)
    {
        int block = prm[PBlock];
        int i = index;
        int n = block;
        if (i >= block) { n = prm[PLastN]; i -= block; }
        double w = 1.0;
        double half = 0.25 * (n - 1);     // alpha (N - 1) / 2 with alpha = 0.5
        if (half > 0)
        {
            if (i < half) w = 0.5 * (1.0 - XMath.Cos((float)(Math.PI * i / half)));
            else if (i > (n - 1) - half) w = 0.5 * (1.0 - XMath.Cos((float)(Math.PI * ((n - 1) - i) / half)));
        }
        win[index] = (float)w;
    }

    /// <summary>One thread per (frame, candidate): constant flag, wasted bits, range check of the input channels.</summary>
    public static void PrepKernel(Index1D index, ArrayView<int> input, ArrayView<int> prm, ArrayView<int> sub, ArrayView<long> meta,
        int firstFrame)
    {
        int cands = prm[PCands];
        int lf = index / cands;
        int cand = index - lf * cands;
        int f = firstFrame + lf;
        int n = FrameN(prm, f);
        int channels = prm[PChannels], stereo = prm[PStereo], bps = prm[PBps];
        long frameStart = (long)f * prm[PBlock];
        long first = Cand(input, channels, stereo, frameStart, 0, cand);
        long or = 0;
        bool same = true;
        bool check = stereo < 0 || cand < 2;
        long lo = -(1L << (bps - 1)), hi = (1L << (bps - 1)) - 1;
        bool bad = false;
        for (int i = 0; i < n; i++)
        {
            long v = Cand(input, channels, stereo, frameStart, i, cand);
            or |= v;
            if (v != first) same = false;
            if (v < lo) bad = true;
            if (v > hi) bad = true;
        }
        if (bad && check) meta[MRangeError] = 1;
        int wasted = 0;
        if (!same && or != 0)
        {
            bool go = ((or >> wasted) & 1) == 0;
            while (go)
            {
                wasted++;
                go = wasted < 32 && ((or >> wasted) & 1) == 0;
            }
        }
        long sb = (long)index.X * SubInts;
        sub[sb + SConst] = same ? 1 : 0;
        sub[sb + SConstValue] = (int)first;
        sub[sb + SWasted] = wasted;
    }

    /// <summary>One thread per (frame, candidate, lag): autocorrelation of the windowed signal (lags above the level's order: 0).</summary>
    public static void AutocorrKernel(Index1D index, ArrayView<int> input, ArrayView<int> prm, ArrayView<int> sub, ArrayView<float> win,
        ArrayView<double> acf, int firstFrame)
    {
        int cands = prm[PCands];
        int lag = index % Lags;
        int sc = index / Lags;
        int lf = sc / cands;
        int cand = sc - lf * cands;
        int f = firstFrame + lf;
        int n = FrameN(prm, f);
        long sb = (long)sc * SubInts;
        double sum = 0;
        if (sub[sb + SConst] == 0 && lag <= prm[PMaxLpc])
        {
            int channels = prm[PChannels], stereo = prm[PStereo];
            int wasted = sub[sb + SWasted];
            long frameStart = (long)f * prm[PBlock];
            int wb = 0;
            if (f == prm[PFrames] - 1) wb = prm[PBlock];
            for (int i = lag; i < n; i++)
            {
                double a = win[wb + i] * (double)(Cand(input, channels, stereo, frameStart, i, cand) >> wasted);
                double b = win[wb + i - lag] * (double)(Cand(input, channels, stereo, frameStart, i - lag, cand) >> wasted);
                sum += a * b;
            }
        }
        acf[index] = sum;
    }

    /// <summary>
    /// One thread per (frame, candidate): Levinson-Durbin recursion on the autocorrelation up to the level's LPC order;
    /// the predictor of every order quantized to the precision with the largest shift (at most 15) that fits, rounding with
    /// error feedback. scratch holds 2 * Lags doubles per subframe.
    /// </summary>
    public static void LpcKernel(Index1D index, ArrayView<int> prm, ArrayView<int> sub, ArrayView<double> acf, ArrayView<double> scratch,
        ArrayView<int> lpcQ)
    {
        long qb = (long)index.X * LpcInts;
        for (int o = 0; o < MaxLpcOrder; o++) lpcQ[qb + o * LpcStride + MaxLpcOrder] = -1;
        long sb = (long)index.X * SubInts;
        if (sub[sb + SConst] != 0) return;
        long ab = (long)index.X * Lags;
        double r0 = acf[ab];
        if (!(r0 > 0)) return;
        int maxLpc = prm[PMaxLpc];
        int precision = prm[PPrecision];
        long a = (long)index.X * 2 * Lags;     // a[1..order]
        long tmp = a + Lags;
        double err = r0;
        int m = 1;
        bool go = m <= maxLpc;
        while (go)
        {
            double acc = acf[ab + m];
            for (int j = 1; j < m; j++) acc -= scratch[a + j] * acf[ab + m - j];
            double k = acc / err;
            for (int j = 1; j < m; j++) scratch[tmp + j] = scratch[a + j] - k * scratch[a + m - j];
            for (int j = 1; j < m; j++) scratch[a + j] = scratch[tmp + j];
            scratch[a + m] = k;
            err *= 1.0 - k * k;

            // quantize this order: |q| < 2^(precision - 1)
            double cmax = 0;
            for (int j = 1; j <= m; j++) cmax = XMath.Max(cmax, XMath.Abs(scratch[a + j]));
            int shift = 15;
            if (cmax > 0)
            {
                // e: 2^(e - 1) <= cmax < 2^e
                int e = 0;
                double p2 = 1.0;
                bool up = p2 <= cmax;
                while (up) { p2 *= 2.0; e++; up = p2 <= cmax; }
                bool down = p2 * 0.5 > cmax;
                while (down) { p2 *= 0.5; e--; down = p2 * 0.5 > cmax; }
                shift = precision - 1 - e;
                if (shift > 15) shift = 15;
            }
            long ob = qb + (m - 1) * LpcStride;
            if (shift >= 0)
            {
                double scale = 1 << shift;
                long qmax = (1L << (precision - 1)) - 1, qmin = -(1L << (precision - 1));
                double fb = 0;
                for (int j = 1; j <= m; j++)
                {
                    double v = scratch[a + j] * scale + fb;
                    long q = (long)XMath.Floor(v + 0.5);
                    if (q > qmax) q = qmax;
                    if (q < qmin) q = qmin;
                    fb = v - q;
                    lpcQ[ob + j - 1] = (int)q;
                }
                lpcQ[ob + MaxLpcOrder] = shift;
            }
            m++;
            // stop once the prediction error vanishes or the recursion leaves the stable region
            go = m <= maxLpc && err > 0 && XMath.Abs(k) < 1.0;
        }
    }

    /// <summary>
    /// One thread per (frame, candidate, predictor, finest partition): the sum of the zigzag-mapped residuals of the
    /// partition (samples below the predictor order excluded); -1 when the predictor cannot code the subframe (not allowed
    /// at this level, order too high, or a residual outside the 32-bit range RFC 9639 allows).
    /// </summary>
    public static void PartSumKernel(Index1D index, ArrayView<int> input, ArrayView<int> prm, ArrayView<int> sub, ArrayView<int> lpcQ,
        ArrayView<long> sums, int firstFrame)
    {
        int cands = prm[PCands], preds = prm[PPreds];
        int part = index % MaxParts;
        int t = index / MaxParts;
        int pred = t % preds;
        int sc = t / preds;
        int lf = sc / cands;
        int cand = sc - lf * cands;
        int f = firstFrame + lf;
        int n = FrameN(prm, f);
        int poF = FinestPo(n, prm[PMaxPo]);
        if (part >= (1 << poF)) return;
        long sb = (long)sc * SubInts;
        if (sub[sb + SConst] != 0) { sums[index] = 0; return; }
        long qb = (long)sc * LpcInts;
        if (!PredUsable(prm, lpcQ, qb, pred, n)) { sums[index] = -1; return; }
        int order = PredOrder(pred);
        long coefBase = qb;
        int shift = 0;
        if (pred >= 5)
        {
            coefBase = qb + (order - 1) * LpcStride;
            shift = lpcQ[coefBase + MaxLpcOrder];
        }
        int channels = prm[PChannels], stereo = prm[PStereo];
        int wasted = sub[sb + SWasted];
        long frameStart = (long)f * prm[PBlock];
        int len = n >> poF;
        int start = part * len;
        int end = start + len;
        if (start < order) start = order;
        long sum = 0;
        bool overflow = false;
        for (int i = start; i < end; i++)
        {
            long e = Residual(input, channels, stereo, frameStart, cand, wasted, i, pred, lpcQ, coefBase, shift);
            if (e >= 2147483647L) overflow = true;
            if (e <= -2147483647L) overflow = true;
            sum += ZigZag(e);
        }
        if (overflow) sum = -1;
        sums[index] = sum;
    }

    /// <summary>
    /// One thread per (frame, candidate, predictor): the partition order with the fewest estimated bits and that bit
    /// count (overhead included); long.MaxValue when the predictor is unusable.
    /// </summary>
    public static void PredCostKernel(Index1D index, ArrayView<int> prm, ArrayView<int> sub, ArrayView<long> sums, ArrayView<long> cost,
        ArrayView<int> costPo, int firstFrame)
    {
        int cands = prm[PCands], preds = prm[PPreds];
        int pred = index % preds;
        int sc = index / preds;
        int lf = sc / cands;
        int cand = sc - lf * cands;
        int f = firstFrame + lf;
        int n = FrameN(prm, f);
        long sb = (long)sc * SubInts;
        cost[index] = long.MaxValue;
        costPo[index] = 0;
        if (sub[sb + SConst] != 0) return;
        int poF = FinestPo(n, prm[PMaxPo]);
        int fine = 1 << poF;
        long pb = (long)index.X * MaxParts;
        bool usable = true;
        for (int p = 0; p < fine; p++)
            if (sums[pb + p] < 0) usable = false;
        if (!usable) return;
        int order = PredOrder(pred);
        int wasted = sub[sb + SWasted];
        int ebps = CandBps(prm, cand) - wasted;
        long best = long.MaxValue;
        int bestPo = 0;
        for (int po = poF; po >= 0; po--)
        {
            if ((n >> po) <= order) continue;
            int parts = 1 << po;
            int per = 1 << (poF - po);
            long bits = 0;
            bool wide = false;
            for (int p = 0; p < parts; p++)
            {
                long s = 0;
                for (int q = 0; q < per; q++) s += sums[pb + p * per + q];
                long m = n >> po;
                if (p == 0) m -= order;
                int k = EstK(s, m);
                long eb = EstBits(s, m, k);
                // an all-zero partition: escape code + a 5-bit width of 0, no residual bits
                if (s == 0 && eb > 5) eb = 5;
                else if (k > 14) wide = true;
                bits += eb;
            }
            int paramBits = 4;
            if (wide) paramBits = 5;
            bits += (long)parts * paramBits;
            if (bits < best) { best = bits; bestPo = po; }
        }
        if (best == long.MaxValue) return;
        cost[index] = best + Overhead(pred, wasted, ebps, prm[PPrecision]);
        costPo[index] = bestPo;
    }

    /// <summary>
    /// One thread per (frame, candidate): the predictor with the fewest estimated bits, its Rice parameters chosen
    /// exactly (around the estimate) per partition, the exact subframe bits; verbatim when that is smaller, constant when
    /// every sample is equal. Fills the subframe record and subBits.
    /// </summary>
    public static void SubframeKernel(Index1D index, ArrayView<int> input, ArrayView<int> prm, ArrayView<int> sub, ArrayView<int> lpcQ,
        ArrayView<long> cost, ArrayView<int> costPo, ArrayView<long> subBits, int firstFrame)
    {
        int cands = prm[PCands], preds = prm[PPreds];
        int lf = index / cands;
        int cand = index - lf * cands;
        int f = firstFrame + lf;
        int n = FrameN(prm, f);
        long sb = (long)index.X * SubInts;
        int sbps = CandBps(prm, cand);
        if (sub[sb + SConst] != 0)
        {
            sub[sb + SType] = TConstant;
            sub[sb + SWasted] = 0;
            subBits[index] = 8 + sbps;
            return;
        }
        int wasted = sub[sb + SWasted];
        int ebps = sbps - wasted;
        long verbatim = 8 + wasted + (long)n * ebps;
        long cb = (long)index.X * preds;
        int bestPred = -1;
        long bestCost = long.MaxValue;
        for (int p = 0; p < preds; p++)
        {
            long c = cost[cb + p];
            if (c < bestCost) { bestCost = c; bestPred = p; }
        }
        sub[sb + SType] = TVerbatim;
        subBits[index] = verbatim;
        if (bestPred < 0) return;

        int order = PredOrder(bestPred);
        int po = costPo[cb + bestPred];
        long qb = (long)index.X * LpcInts;
        long coefBase = qb;
        int shift = 0;
        if (bestPred >= 5)
        {
            coefBase = qb + (order - 1) * LpcStride;
            shift = lpcQ[coefBase + MaxLpcOrder];
        }
        int channels = prm[PChannels], stereo = prm[PStereo];
        long frameStart = (long)f * prm[PBlock];
        int parts = 1 << po;
        int len = n >> po;
        long bits = 0;
        bool wide = false;
        for (int p = 0; p < parts; p++)
        {
            int start = p * len;
            if (p == 0) start = order;
            int end = (p + 1) * len;
            long m = end - start;
            long s = 0, lo = 0, hi = 0;
            for (int i = start; i < end; i++)
            {
                long e = Residual(input, channels, stereo, frameStart, cand, wasted, i, bestPred, lpcQ, coefBase, shift);
                s += ZigZag(e);
                if (e < lo) lo = e;
                if (e > hi) hi = e;
            }
            int k0 = EstK(s, m);
            // exact bits for k0 - 1, k0, k0 + 1
            int kLo = k0 - 1;
            if (kLo < 0) kLo = 0;
            int kHi = k0 + 1;
            if (kHi > 30) kHi = 30;
            long q0 = 0, q1 = 0, q2 = 0;
            for (int i = start; i < end; i++)
            {
                long u = ZigZag(Residual(input, channels, stereo, frameStart, cand, wasted, i, bestPred, lpcQ, coefBase, shift));
                q0 += u >> kLo;
                q1 += u >> (kLo + 1);
                q2 += u >> (kLo + 2);
            }
            int k = kLo;
            long kb = q0 + m * (kLo + 1);
            if (kLo + 1 <= kHi)
            {
                long b1 = q1 + m * (kLo + 2);
                if (b1 < kb) { kb = b1; k = kLo + 1; }
            }
            if (kLo + 2 <= kHi)
            {
                long b2 = q2 + m * (kLo + 3);
                if (b2 < kb) { kb = b2; k = kLo + 2; }
            }
            // escape-coded partition (RFC 9639 section 9.2.7.1): a 5-bit width w, then each residual as a w-bit signed
            // value (w = 0: all zero); stored as SK = -1 - w
            int w = 0;
            bool fits = hi > 0 || lo < 0;
            while (fits)
            {
                w++;
                fits = w < 32 && (lo < -(1L << (w - 1)) || hi > (1L << (w - 1)) - 1);
            }
            long escBits = 5 + m * w;
            if (w <= 31 && escBits < kb)
            {
                sub[sb + SK + p] = -1 - w;
                bits += escBits;
            }
            else
            {
                if (k > 14) wide = true;
                sub[sb + SK + p] = k;
                bits += kb;
            }
        }
        int paramBits = 4;
        if (wide) paramBits = 5;
        bits += (long)parts * paramBits + Overhead(bestPred, wasted, ebps, prm[PPrecision]);
        if (bits >= verbatim) return;
        sub[sb + SOrder] = order;
        sub[sb + SPo] = po;
        sub[sb + SMethod] = wide ? 1 : 0;
        if (bestPred >= 5)
        {
            sub[sb + SType] = TLpc;
            sub[sb + SShift] = shift;
            for (int j = 0; j < order; j++) sub[sb + SCoef + j] = lpcQ[coefBase + j];
        }
        else sub[sb + SType] = TFixed;
        subBits[index] = bits;
    }

    // ------------------------------------------------------------------ frame layout and writing

    /// <summary>RFC 9639 block size code of a block of <paramref name="n"/> samples (6 / 7: explicit 8 / 16-bit value).</summary>
    private static int BlockCode(int n)
    {
        int code = 7;
        if (n == 192) code = 1;
        else if (n == 576) code = 2;
        else if (n == 1152) code = 3;
        else if (n == 2304) code = 4;
        else if (n == 4608) code = 5;
        else if (n == 256) code = 8;
        else if (n == 512) code = 9;
        else if (n == 1024) code = 10;
        else if (n == 2048) code = 11;
        else if (n == 4096) code = 12;
        else if (n == 8192) code = 13;
        else if (n == 16384) code = 14;
        else if (n == 32768) code = 15;
        else if (n <= 256) code = 6;
        return code;
    }

    /// <summary>Bytes of the UTF-8-style coded frame number.</summary>
    private static int Utf8Bytes(long v)
    {
        int b = 7;
        if (v < 0x80) b = 1;
        else if (v < 0x800) b = 2;
        else if (v < 0x10000) b = 3;
        else if (v < 0x200000) b = 4;
        else if (v < 0x4000000) b = 5;
        else if (v < 0x80000000L) b = 6;
        return b;
    }

    /// <summary>Bits of a frame header (CRC-8 included).</summary>
    private static long HeaderBits(ArrayView<int> prm, int f, int n)
    {
        long bits = 32 + 8L * Utf8Bytes(f) + 8;
        int bc = BlockCode(n);
        if (bc == 6) bits += 8;
        if (bc == 7) bits += 16;
        int rc = prm[PRateCode];
        if (rc == 12) bits += 8;
        if (rc == 13) bits += 16;
        if (rc == 14) bits += 16;
        return bits;
    }

    /// <summary>
    /// One thread per frame: the channel assignment (for stereo the decorrelation with the fewest bits, or the one the
    /// options force) and the frame's byte count.
    /// </summary>
    public static void FrameSizeKernel(Index1D index, ArrayView<int> prm, ArrayView<long> subBits, ArrayView<int> frameInfo, int firstFrame)
    {
        int lf = index;
        int f = firstFrame + lf;
        int n = FrameN(prm, f);
        int cands = prm[PCands], channels = prm[PChannels], stereo = prm[PStereo];
        long sb = (long)lf * cands;
        long fb = (long)lf * FrameInts;
        long bits = 0;
        int code = channels - 1;
        int c0 = 0, c1 = 1;
        if (stereo < 0)
        {
            for (int c = 0; c < channels; c++)
            {
                bits += subBits[sb + c];
                frameInfo[fb + FCand + c] = c;
            }
        }
        else
        {
            long b0 = subBits[sb], b1 = subBits[sb + 1], b2 = subBits[sb + 2], b3 = subBits[sb + 3];
            long indep = b0 + b1, leftSide = b0 + b3, rightSide = b3 + b1, midSide = b2 + b3;
            int mode = stereo;
            if (mode == 0)
            {
                mode = 4;
                long best = indep;
                if (leftSide < best) { best = leftSide; mode = 1; }
                if (rightSide < best) { best = rightSide; mode = 2; }
                if (midSide < best) { mode = 3; }
            }
            code = 1;
            bits = indep;
            if (mode == 1) { code = 8; c0 = 0; c1 = 3; bits = leftSide; }
            if (mode == 2) { code = 9; c0 = 3; c1 = 1; bits = rightSide; }
            if (mode == 3) { code = 10; c0 = 2; c1 = 3; bits = midSide; }
            frameInfo[fb + FCand] = c0;
            frameInfo[fb + FCand + 1] = c1;
        }
        bits += HeaderBits(prm, f, n);
        frameInfo[fb + FCode] = code;
        frameInfo[fb + FBytes] = (int)((bits + 7) / 8 + 2);
    }

    /// <summary>One thread: each frame's byte offset (frames after the stream header), the running total, the smallest and largest frame.</summary>
    public static void OffsetsKernel(Index1D _, ArrayView<int> frameInfo, ArrayView<long> offsets, ArrayView<long> meta, int count)
    {
        long run = meta[MRun], mn = meta[MMinFrame], mx = meta[MMaxFrame];
        for (int lf = 0; lf < count; lf++)
        {
            long bytes = frameInfo[(long)lf * FrameInts + FBytes];
            offsets[lf] = run;
            run += bytes;
            if (bytes < mn) mn = bytes;
            if (bytes > mx) mx = bytes;
        }
        meta[MRun] = run;
        meta[MMinFrame] = mn;
        meta[MMaxFrame] = mx;
    }

    /// <summary>Bit writer state of one frame: output position, pending bits, running CRC-8 and CRC-16.</summary>
    private struct BitSink
    {
        public long Pos;
        public ulong Acc;
        public int Bits;
        public int Crc8;
        public int Crc16;
    }

    /// <summary>Append the low <paramref name="n"/> (0..32) bits of <paramref name="value"/>, MSB first.</summary>
    private static void Put(ArrayView<byte> output, ArrayView<int> crc, ref BitSink s, long value, int n)
    {
        ulong mask = (1UL << n) - 1;
        s.Acc = (s.Acc << n) | ((ulong)value & mask);
        s.Bits += n;
        bool go = s.Bits >= 8;
        while (go)
        {
            s.Bits -= 8;
            int b = (int)((s.Acc >> s.Bits) & 0xFF);
            output[s.Pos] = (byte)b;
            s.Pos++;
            s.Crc8 = crc[256 + (s.Crc8 ^ b)];
            s.Crc16 = ((s.Crc16 << 8) & 0xFFFF) ^ crc[((s.Crc16 >> 8) ^ b) & 0xFF];
            go = s.Bits >= 8;
        }
    }

    private static void PutZeros(ArrayView<byte> output, ArrayView<int> crc, ref BitSink s, long count)
    {
        long left = count;
        bool go = left >= 32;
        while (go)
        {
            Put(output, crc, ref s, 0, 32);
            left -= 32;
            go = left >= 32;
        }
        Put(output, crc, ref s, 0, (int)left);
    }

    /// <summary>One thread per frame: the frame's bytes at its offset (see the file header).</summary>
    public static void WriteFrameKernel(Index1D index, ArrayView<int> input, ArrayView<int> prm, ArrayView<int> sub, ArrayView<int> frameInfo,
        ArrayView<long> offsets, ArrayView<int> crc, ArrayView<byte> output, int firstFrame)
    {
        int lf = index;
        int f = firstFrame + lf;
        int n = FrameN(prm, f);
        int cands = prm[PCands], channels = prm[PChannels], stereo = prm[PStereo];
        long fb = (long)lf * FrameInts;
        long frameStart = (long)f * prm[PBlock];
        var s = new BitSink();
        s.Pos = prm[PHeaderLen] + offsets[lf];

        // frame header (RFC 9639 section 9.1): sync + fixed block size strategy, codes, coded frame number, extras, CRC-8
        int bc = BlockCode(n);
        int rc = prm[PRateCode];
        Put(output, crc, ref s, 0xFFF8, 16);
        Put(output, crc, ref s, bc, 4);
        Put(output, crc, ref s, rc, 4);
        Put(output, crc, ref s, frameInfo[fb + FCode], 4);
        Put(output, crc, ref s, prm[PBpsCode], 3);
        Put(output, crc, ref s, 0, 1);
        int nb = Utf8Bytes(f);
        if (nb == 1) Put(output, crc, ref s, f, 8);
        else
        {
            long lead = (0xFF00 >> nb) & 0xFF;
            Put(output, crc, ref s, lead | ((long)f >> (6 * (nb - 1))), 8);
            for (int k = nb - 2; k >= 0; k--) Put(output, crc, ref s, 0x80 | (((long)f >> (6 * k)) & 0x3F), 8);
        }
        if (bc == 6) Put(output, crc, ref s, n - 1, 8);
        if (bc == 7) Put(output, crc, ref s, n - 1, 16);
        int rate = prm[PRate];
        if (rc == 12) Put(output, crc, ref s, rate / 1000, 8);
        if (rc == 13) Put(output, crc, ref s, rate, 16);
        if (rc == 14) Put(output, crc, ref s, rate / 10, 16);
        Put(output, crc, ref s, s.Crc8, 8);

        int precision = prm[PPrecision];
        for (int c = 0; c < channels; c++)
        {
            int cand = frameInfo[fb + FCand + c];
            long sb = ((long)lf * cands + cand) * SubInts;
            int type = sub[sb + SType];
            int sbps = CandBps(prm, cand);
            int wasted = sub[sb + SWasted];
            int ebps = sbps - wasted;
            int order = sub[sb + SOrder];
            int typeBits = 1;
            if (type == TConstant) typeBits = 0;
            if (type == TFixed) typeBits = 8 + order;
            if (type == TLpc) typeBits = 31 + order;
            Put(output, crc, ref s, typeBits, 7);       // zero bit + 6 type bits
            if (wasted > 0)
            {
                Put(output, crc, ref s, 1, 1);
                PutZeros(output, crc, ref s, wasted - 1);
                Put(output, crc, ref s, 1, 1);
            }
            else Put(output, crc, ref s, 0, 1);
            if (type == TConstant)
            {
                Put(output, crc, ref s, sub[sb + SConstValue], sbps);
                continue;
            }
            int warm = n;
            if (type != TVerbatim) warm = order;
            for (int i = 0; i < warm; i++)
                Put(output, crc, ref s, Cand(input, channels, stereo, frameStart, i, cand) >> wasted, ebps);
            if (type == TVerbatim) continue;
            int pred = order;
            int shift = 0;
            if (type == TLpc)
            {
                pred = order + 4;
                shift = sub[sb + SShift];
                Put(output, crc, ref s, precision - 1, 4);
                Put(output, crc, ref s, shift, 5);
                for (int j = 0; j < order; j++) Put(output, crc, ref s, sub[sb + SCoef + j], precision);
            }
            int method = sub[sb + SMethod];
            int po = sub[sb + SPo];
            int paramBits = 4 + method;
            Put(output, crc, ref s, method, 2);
            Put(output, crc, ref s, po, 4);
            int len = n >> po;
            for (int p = 0; p < (1 << po); p++)
            {
                int k = sub[sb + SK + p];
                int start = p * len;
                if (p == 0) start = order;
                int end = (p + 1) * len;
                if (k < 0)
                {
                    // escape code (all parameter bits set), the width, the residuals as signed width-bit values
                    int width = -1 - k;
                    Put(output, crc, ref s, (1 << paramBits) - 1, paramBits);
                    Put(output, crc, ref s, width, 5);
                    if (width > 0)
                        for (int i = start; i < end; i++)
                            Put(output, crc, ref s, Residual(input, channels, stereo, frameStart, cand, wasted, i, pred, sub, sb + SCoef, shift), width);
                    continue;
                }
                Put(output, crc, ref s, k, paramBits);
                for (int i = start; i < end; i++)
                {
                    long u = ZigZag(Residual(input, channels, stereo, frameStart, cand, wasted, i, pred, sub, sb + SCoef, shift));
                    PutZeros(output, crc, ref s, u >> k);
                    Put(output, crc, ref s, 1, 1);
                    Put(output, crc, ref s, u, k);
                }
            }
        }
        // byte alignment, then CRC-16 of everything before it
        if (s.Bits > 0) Put(output, crc, ref s, 0, 8 - s.Bits);
        Put(output, crc, ref s, s.Crc16, 16);
    }

    /// <summary>One thread: the smallest / largest frame size and the MD5 of the samples into STREAMINFO (bytes 12..17, 26..41).</summary>
    public static void PatchStreamInfoKernel(Index1D _, ArrayView<long> meta, ArrayView<uint> md5, ArrayView<byte> output, int frames)
    {
        long mn = 0, mx = 0;
        if (frames > 0) { mn = meta[MMinFrame]; mx = meta[MMaxFrame]; }
        for (int i = 0; i < 3; i++)
        {
            output[12 + i] = (byte)((mn >> (16 - 8 * i)) & 0xFF);
            output[15 + i] = (byte)((mx >> (16 - 8 * i)) & 0xFF);
        }
        for (int i = 0; i < 16; i++) output[26 + i] = (byte)md5[FlacMd5Kernels.Md5Digest + i];
    }

    /// <summary>One thread per byte: dst[first + i] = src[first + i].</summary>
    public static void CopyBytesKernel(Index1D index, ArrayView<byte> src, ArrayView<byte> dst, long first)
    {
        long i = first + index;
        dst[i] = src[i];
    }
}
