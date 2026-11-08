// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Accelerator side of FlacEncoderGpu.EncodeAsync: a port of FFmpeg 8.1's libavcodec/flacenc.c (the encoder PyAV and
// so ComfyUI's Save Audio use) with its compression levels 0..8, written so every backend produces the same bytes:
//
//   * levels 0..2 (fixed predictors) are integer in FFmpeg too: the same decisions, the same bytes;
//   * levels 3..8 (LPC) run FFmpeg's double-precision analysis as its C code does (window, autocorrelation,
//     Levinson-Durbin, order estimate, coefficient quantization). FFmpeg on x86 sums the autocorrelation with SIMD, so
//     its coefficients can differ in the last bits from ours; WebGPU / WebGL emulate doubles (not IEEE), so there the
//     coefficients - and the bytes - can differ from the desktop backends. The decoded samples are exact either way.
//     (An integer Levinson-Durbin was tried first: tonal signals need near-double precision - the error energy falls by
//     a factor of 10^6 within a few orders - and its coefficients came out wrong.)
//
// Everything after the coefficients is FFmpeg's integer logic: stereo mode estimate, decorrelation, wasted bits,
// constant / verbatim / fixed / LPC subframes, the rice partition search, the verbatim fallback, the bit stream.
//
// Buffers: samples X and residuals R per (frame, channel) slot of BlockSize ints; a record per subframe (Sub*) and per
// frame (F*); rice parameters (256 per subframe); a scratch per subframe (partition sums, LPC coefficient table).

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Audio.Flac;

internal static class FlacStreamKernels
{
    // ---- global meta (ints)
    public const int MChannels = 0, MBlockSize = 1, MFrames = 2, MTotalLo = 3, MTotalHi = 4, MBps = 5, MBpsCode = 6,
        MSrCode0 = 7, MSrCode1 = 8, MMinOrder = 9, MMaxOrder = 10, MLpcType = 11, MOrderMethod = 12, MMinPorder = 13,
        MMaxPorder = 14, MPrecision = 15, MChMode = 16, MFramesStart = 17, MTotalBytes = 18, MMinFrame = 19, MMaxFrame = 20,
        MMaxFrameInit = 21, MLastN = 22, MRate = 23, MSize = 32;

    // lpc types (FFLPCType) and order methods (flacenc ORDER_METHOD_*)
    public const int LpcNone = 0, LpcFixed = 1, LpcLevinson = 2;
    public const int OrderEst = 0, Order2Level = 1, Order4Level = 2, Order8Level = 3, OrderSearch = 4, OrderLog = 5;

    // ---- subframe record (ints, per frame * channels + channel)
    public const int SType = 0, SOrder = 1, SShift = 2, SPorder = 3, SCodingMode = 4, SWasted = 5, SObits = 6, SBits = 7,
        SCoef = 8, SubStride = 48;
    public const int TypeConstant = 0, TypeVerbatim = 1, TypeFixed = 2, TypeLpc = 3;

    // ---- frame record
    public const int FN = 0, FChMode = 1, FVerbatim = 2, FBytes = 3, FOffset = 4, FrameStride = 8;

    public const int MaxPartitions = 256, MaxLpcOrder = 12;

    // ---- scratch per subframe: partition sums (longs), and as ints: LPC coefficient table + shifts
    public const int ScratchLongs = MaxPartitions + MaxLpcOrder;   // partition sums, then the LOG method's per-order bits
    public const int LpcTable = 0, LpcShifts = MaxLpcOrder * MaxLpcOrder, LpcInts = LpcShifts + MaxLpcOrder;   // quantized rows, their shifts

    private const int ChIndependent = 0, ChLeftSide = 1, ChRightSide = 2, ChMidSide = 3;

    // ------------------------------------------------------------------ helpers

    /// <summary>av_log2: index of the highest set bit (0 for 0 and 1).</summary>
    private static int Log2(ulong v)
    {
        int n = 0;
        for (int b = 32; b > 0; b >>= 1)
        {
            if ((v >> b) != 0)
            {
                v >>= b;
                n += b;
            }
        }
        return n;
    }

    private static long Abs(long v) => v < 0 ? -v : v;

    /// <summary>((unsigned)x << 1) ^ (x >> 31): the rice-folded value.</summary>
    private static uint Fold(int x) => (uint)(x << 1) ^ (uint)(x >> 31);

    /// <summary>find_optimal_param (FFmpeg): the rice parameter estimate from a partition's folded sum.</summary>
    private static int FindOptimalParam(ulong sum, int n, int maxParam)
    {
        ulong half = (ulong)(n >> 1);
        if (sum <= half) return 0;
        ulong sum2 = sum - half;
        ulong q = sum2 / (ulong)n;
        if (q > int.MaxValue) q = int.MaxValue;
        int k = Log2(q);
        return k < maxParam ? k : maxParam;
    }

    /// <summary>rice_encode_count (FFmpeg), unsigned arithmetic included.</summary>
    private static ulong RiceEncodeCount(ulong sum, int n, int k) =>
        unchecked((ulong)n * (ulong)(k + 1) + ((sum - (ulong)(n >> 1)) >> k));

    /// <summary>get_max_p_order (FFmpeg).</summary>
    private static int MaxPorder(int maxPorder, int n, int order)
    {
        int p = Log2((ulong)(uint)(n ^ (n - 1)));
        if (maxPorder < p) p = maxPorder;
        if (order > 0)
        {
            int q = Log2((ulong)(n / order));
            if (q < p) p = q;
        }
        return p;
    }

    private static int Slot(int f, int c, int channels, int blockSize) => (f * channels + c) * blockSize;

    // ------------------------------------------------------------------ 1. deinterleave, stereo, wasted bits

    /// <summary>One thread per input sample (from <paramref name="first"/>): interleaved input into the per-(frame, channel) slots.</summary>
    public static void DeinterleaveKernel(Index1D index, ArrayView<int> input, ArrayView<int> x, ArrayView<int> meta, long first)
    {
        int channels = meta[MChannels], b = meta[MBlockSize];
        long g = first + index.X;
        long s = g / channels;
        int c = (int)(g - s * channels);
        int f = (int)(s / b), t = (int)(s - (long)f * b);
        x[(long)Slot(f, c, channels, b) + t] = input[g];
    }

    /// <summary>One thread per frame: its sample count and (stereo) estimate_stereo_mode / the forced mode.</summary>
    public static void StereoModeKernel(Index1D index, ArrayView<int> x, ArrayView<int> frames, ArrayView<int> meta)
    {
        int f = index.X, channels = meta[MChannels], b = meta[MBlockSize];
        int n = f == meta[MFrames] - 1 ? meta[MLastN] : b;
        int fr = f * FrameStride;
        frames[fr + FN] = n;
        frames[fr + FVerbatim] = 0;
        int mode = ChIndependent;
        if (channels == 2)
        {
            if (meta[MChMode] >= 0)
            {
                mode = meta[MChMode];
            }
            else
            {
                int l = Slot(f, 0, 2, b), r = Slot(f, 1, 2, b);
                ulong s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                for (int i = 2; i < n; i++)
                {
                    int lt = x[l + i] - 2 * x[l + i - 1] + x[l + i - 2];
                    int rt = x[r + i] - 2 * x[r + i - 1] + x[r + i - 2];
                    s2 += (ulong)Abs((lt + rt) >> 1);
                    s3 += (ulong)Abs(lt - rt);
                    s0 += (ulong)Abs(lt);
                    s1 += (ulong)Abs(rt);
                }
                int maxParam = meta[MBps] > 16 ? 30 : 14;   // (1 << coding_mode) - 2 of subframe 0
                s0 = RiceEncodeCount(2 * s0, n, FindOptimalParam(2 * s0, n, maxParam));
                s1 = RiceEncodeCount(2 * s1, n, FindOptimalParam(2 * s1, n, maxParam));
                s2 = RiceEncodeCount(2 * s2, n, FindOptimalParam(2 * s2, n, maxParam));
                s3 = RiceEncodeCount(2 * s3, n, FindOptimalParam(2 * s3, n, maxParam));
                ulong score0 = s0 + s1, score1 = s0 + s3, score2 = s1 + s3, score3 = s2 + s3;
                ulong best = score0;
                if (score1 < best) { best = score1; mode = ChLeftSide; }
                if (score2 < best) { best = score2; mode = ChRightSide; }
                if (score3 < best) { mode = ChMidSide; }
            }
        }
        frames[fr + FChMode] = mode;
    }

    /// <summary>One thread per (frame, sample) of a stereo stream: channel_decorrelation.</summary>
    public static void DecorrelateKernel(Index1D index, ArrayView<int> x, ArrayView<int> frames, ArrayView<int> meta)
    {
        int b = meta[MBlockSize];
        int f = index.X / b, t = index.X % b;
        int mode = frames[f * FrameStride + FChMode];
        if (t >= frames[f * FrameStride + FN]) return;
        int l = Slot(f, 0, 2, b) + t, r = Slot(f, 1, 2, b) + t;
        int lv = x[l], rv = x[r];
        if (mode == ChMidSide)
        {
            x[l] = (lv + rv) >> 1;
            x[r] = lv - rv;
        }
        else if (mode == ChLeftSide)
        {
            x[r] = lv - rv;
        }
        else if (mode == ChRightSide)
        {
            x[l] = lv - rv;
        }
    }

    /// <summary>
    /// One thread per frame: init_frame's per-subframe bits and coding mode, the side channel's extra bit, then
    /// remove_wasted_bits (channels in order; FFmpeg stops at the first channel without wasted bits).
    /// </summary>
    public static void WastedBitsKernel(Index1D index, ArrayView<int> x, ArrayView<int> frames, ArrayView<int> subs, ArrayView<int> meta)
    {
        int f = index.X, channels = meta[MChannels], b = meta[MBlockSize], bps = meta[MBps];
        int n = frames[f * FrameStride + FN], mode = frames[f * FrameStride + FChMode];
        for (int c = 0; c < channels; c++)
        {
            int sr = (f * channels + c) * SubStride;
            int obits = bps;
            if ((mode == ChMidSide | mode == ChLeftSide) & c == 1) obits++;
            if (mode == ChRightSide & c == 0) obits++;
            subs[sr + SObits] = obits;
            subs[sr + SWasted] = 0;
            subs[sr + SCodingMode] = bps > 16 ? 5 : 4;
        }
        bool stop = false;
        for (int c = 0; c < channels; c++)
        {
            if (stop) break;
            int slot = Slot(f, c, channels, b);
            int v = 0;
            for (int i = 0; i < n; i++)
            {
                v |= x[slot + i];
                if ((v & 1) != 0) break;
            }
            if (v == 0 | (v & 1) != 0)
            {
                stop = true;
            }
            else
            {
                int w = 0;
                while (((v >> w) & 1) == 0) w++;
                for (int i = 0; i < n; i++) x[slot + i] >>= w;
                int sr = (f * channels + c) * SubStride;
                subs[sr + SWasted] = w;
                subs[sr + SObits] -= w;
                if (subs[sr + SObits] <= 17) subs[sr + SCodingMode] = 4;
            }
        }
    }

    // ------------------------------------------------------------------ 2. LPC analysis (FFmpeg lpc.c, in double)

    /// <summary>Doubles of scratch per subframe: the Levinson rows (MaxLpcOrder x MaxLpcOrder), the autocorrelation.</summary>
    public const int LpcRowsD = 0, LpcAutocD = MaxLpcOrder * MaxLpcOrder, LpcDoubles = LpcAutocD + MaxLpcOrder + 1;

    /// <summary>lpc_apply_welch_window_c: the weight of sample <paramref name="t"/> of an <paramref name="n"/>-sample block.</summary>
    private static double Window(int t, int n)
    {
        if (n == 1) return 0.0;
        int n2 = n >> 1;
        double c = 2.0 / (n - 1.0);
        double w;
        if ((n & 1) != 0)
        {
            if (t == n2) return 0.0;
            int i = t < n2 ? t : n - 1 - t;
            w = c - i - 1.0;
        }
        else
        {
            int i = t < n2 ? n2 - 1 - t : t - n2;
            w = c - n2 + i;
        }
        return 1.0 - (w * w);
    }

    /// <summary>The windowed sample (zero before the block, like the padded windowed_samples buffer).</summary>
    private static double Windowed(ArrayView<int> x, int slot, int i, int n) => i < 0 ? 0.0 : x[slot + i] * Window(i, n);

    /// <summary>
    /// One thread per subframe that will use LPC: ff_lpc_calc_coefs for FF_LPC_TYPE_LEVINSON - the window,
    /// lpc_compute_autocorr_c (lags in pairs, sums from 1.0), compute_lpc_coefs (normalized Levinson-Durbin) for every
    /// order up to the maximum, estimate_best_order on the reflection coefficients (|lpc[i][i]| above 0.10) and
    /// quantize_lpc_coefs of orders min..max into the integer table (rows + shifts). Writes the estimate into SOrder.
    /// </summary>
    public static void LpcAnalysisKernel(Index1D index, ArrayView<int> x, ArrayView<int> frames, ArrayView<int> subs,
        ArrayView<int> lpcScratch, ArrayView<double> lpcD, ArrayView<int> meta)
    {
        int fc = index.X, channels = meta[MChannels], b = meta[MBlockSize];
        int f = fc / channels, c = fc % channels;
        int n = frames[f * FrameStride + FN];
        int maxOrder = meta[MMaxOrder], minOrder = meta[MMinOrder];
        int sr = fc * SubStride;
        bool lpc = meta[MLpcType] == LpcLevinson & n > maxOrder & frames[f * FrameStride + FVerbatim] == 0 & n >= 5;
        if (!lpc) return;
        int slot = Slot(f, c, channels, b);
        var d = lpcD.SubView((long)fc * LpcDoubles, LpcDoubles);
        var t = lpcScratch.SubView((long)fc * LpcInts, LpcInts);

        // lpc_compute_autocorr_c(windowed, n, maxOrder, autoc)
        int j = 0;
        for (; j < maxOrder; j += 2)
        {
            double sum0 = 1.0, sum1 = 1.0;
            for (int i = j; i < n; i++)
            {
                double wi = Windowed(x, slot, i, n);
                sum0 += wi * Windowed(x, slot, i - j, n);
                sum1 += wi * Windowed(x, slot, i - j - 1, n);
            }
            d[LpcAutocD + j] = sum0;
            d[LpcAutocD + j + 1] = sum1;
        }
        if (j == maxOrder)
        {
            double sum = 1.0;
            for (int i = j - 1; i < n; i++) sum += Windowed(x, slot, i, n) * Windowed(x, slot, i - j, n);
            d[LpcAutocD + j] = sum;
        }

        // compute_lpc_coefs(autoc, 0, maxOrder, lpc, MAX_LPC_ORDER, fail 0, normalize 1): row i holds order i + 1
        double err = d[LpcAutocD];
        int est = minOrder;
        for (int i = 0; i < maxOrder; i++)
        {
            int row = LpcRowsD + i * MaxLpcOrder, last = LpcRowsD + (i - 1) * MaxLpcOrder;
            double r = -d[LpcAutocD + 1 + i];
            for (int k = 0; k < i; k++) r -= d[last + k] * d[LpcAutocD + i - k];
            if (err != 0.0) r /= err;
            err *= 1.0 - (r * r);
            d[row + i] = r;
            for (int k = 0; k < (i + 1) >> 1; k++)
            {
                double fk = d[last + k];
                double bk = d[last + i - 1 - k];
                d[row + k] = fk + r * bk;
                d[row + i - 1 - k] = bk + r * fk;
            }
        }
        // estimate_best_order(ref = |lpc[i][i]|)
        for (int i = maxOrder - 1; i >= minOrder - 1; i--)
        {
            double rf = d[LpcRowsD + i * MaxLpcOrder + i];
            if (rf < 0) rf = -rf;
            if (rf > 0.10)
            {
                est = i + 1;
                break;
            }
        }
        int precision = meta[MPrecision];
        for (int i = minOrder - 1; i < maxOrder; i++) Quantize(d, t, i, i + 1, precision);
        subs[sr + SOrder] = est;
        subs[sr + SType] = TypeLpc;
    }

    /// <summary>
    /// quantize_lpc_coefs of row <paramref name="row"/> (precision bits, shifts 0..15, zero shift 0) into the integer
    /// table: the largest shift that keeps the coefficients within qmax, error feedback, lrintf (the double error
    /// converted to float, rounded half to even).
    /// </summary>
    private static void Quantize(ArrayView<double> d, ArrayView<int> t, int row, int order, int precision)
    {
        int qmax = (1 << (precision - 1)) - 1;
        int src = LpcRowsD + row * MaxLpcOrder, dst = LpcTable + row * MaxLpcOrder;
        double cmax = 0.0;
        for (int k = 0; k < order; k++)
        {
            double v = d[src + k];
            if (v < 0) v = -v;
            if (v > cmax) cmax = v;
        }
        if (cmax * (1 << 15) < 1.0)
        {
            for (int k = 0; k < order; k++) t[dst + k] = 0;
            t[LpcShifts + row] = 0;
            return;
        }
        int sh = 15;
        while ((cmax * (1 << sh) > qmax) & (sh > 0)) sh--;
        bool scaled = (sh == 0) & (cmax > qmax);
        double scale = scaled ? qmax / cmax : 1.0;
        double error = 0;
        for (int k = 0; k < order; k++)
        {
            double v = d[src + k];
            if (scaled) v *= scale;
            error -= v * (1 << sh);
            float fe = (float)error;
            float fl = MathF.Floor(fe);
            float frac = fe - fl;
            int q = (int)fl;
            if ((frac > 0.5f) | ((frac == 0.5f) & ((q & 1) != 0))) q++;
            if (q > qmax) q = qmax;
            if (q < -qmax) q = -qmax;
            t[dst + k] = q;
            error -= q;
        }
        t[LpcShifts + row] = sh;
    }

    // ------------------------------------------------------------------ 3. subframe choice (encode_residual_ch)

    /// <summary>encode_residual_fixed into R; false when a residual leaves int32 (never for <= 29-bit input).</summary>
    private static void FixedResidual(ArrayView<int> x, ArrayView<int> r, int slot, int n, int order)
    {
        for (int i = 0; i < order; i++) r[slot + i] = x[slot + i];
        for (int i = order; i < n; i++)
        {
            int s = x[slot + i];
            int p = 0;
            if (order == 1) p = x[slot + i - 1];
            else if (order == 2) p = 2 * x[slot + i - 1] - x[slot + i - 2];
            else if (order == 3) p = 3 * x[slot + i - 1] - 3 * x[slot + i - 2] + x[slot + i - 3];
            else if (order == 4) p = 4 * x[slot + i - 1] - 6 * x[slot + i - 2] + 4 * x[slot + i - 3] - x[slot + i - 4];
            r[slot + i] = s - p;
        }
    }

    /// <summary>LPC residual in int64; false when one leaves int32 (FFmpeg then writes a verbatim subframe).</summary>
    private static bool LpcResidual(ArrayView<int> x, ArrayView<int> r, int slot, int n, ArrayView<int> table, int row, int order, int shift)
    {
        int rowAt = LpcTable + row * MaxLpcOrder;
        for (int i = 0; i < order; i++) r[slot + i] = x[slot + i];
        bool ok = true;
        for (int i = order; i < n; i++)
        {
            long p = 0;
            for (int j = 0; j < order; j++) p += (long)table[rowAt + j] * x[slot + i - 1 - j];
            p >>= shift;
            long v = x[slot + i] - p;
            if (v <= int.MinValue | v > int.MaxValue) ok = false;
            r[slot + i] = (int)v;
        }
        return ok;
    }

    /// <summary>
    /// calc_rice_params + find_subframe_rice_params (non-exact rice parameters): the partition order and parameters of
    /// the residual in R (written to <paramref name="parms"/>); returns FFmpeg's bit estimate of the subframe.
    /// </summary>
    private static ulong RiceParams(ArrayView<int> r, int slot, int n, int order, int obits, int codingMode, bool lpc, int precision,
        int minPorderOpt, int maxPorderOpt, ArrayView<long> sums, ArrayView<int> parms, ArrayView<int> subs, int sr, bool write)
    {
        int pmin = MaxPorder(minPorderOpt, n, order), pmax = MaxPorder(maxPorderOpt, n, order);
        int kmax = (1 << codingMode) - 2;
        // calc_sum_top at pmax (kmax 0: plain folded sums)
        int parts = 1 << pmax;
        int psize = n >> pmax;
        for (int p = 0; p < parts; p++)
        {
            int start = p == 0 ? order : p * psize;
            int end = (p + 1) * psize;
            ulong s = 0;
            for (int i = start; i < end; i++) s += Fold(r[slot + i]);
            sums[p] = (long)s;
        }
        int optP = pmin;
        ulong optBits = 0xFFFFFFFFUL;   // bits[pmin] = UINT32_MAX
        for (int p = pmax; p >= pmin; p--)
        {
            int part = 1 << p;
            ulong all = 4UL * (ulong)part;
            int cnt = (n >> p) - order;
            for (int i = 0; i < part; i++)
            {
                int k = FindOptimalParam((ulong)sums[i], cnt, kmax);
                all += RiceEncodeCount((ulong)sums[i], cnt, k);
                cnt = n >> p;
            }
            if (all < optBits | pmax == pmin)
            {
                optBits = all;
                optP = p;
            }
            if (p > pmin)
            {
                // calc_sum_next(p - 1)
                int lower = 1 << (p - 1);
                for (int i = 0; i < lower; i++) sums[i] = sums[2 * i] + sums[2 * i + 1];
            }
        }
        if (write)
        {
            // the chosen partition order's parameters (sums of that level, recomputed)
            int part = 1 << optP;
            int size = n >> optP;
            int cnt = size - order;
            for (int i = 0; i < part; i++)
            {
                int start = i == 0 ? order : i * size;
                ulong s = 0;
                for (int j = start; j < (i + 1) * size; j++) s += Fold(r[slot + j]);
                parms[i] = FindOptimalParam(s, cnt, kmax);
                cnt = size;
            }
            subs[sr + SPorder] = optP;
        }
        ulong bits = 8UL + (ulong)(order * obits) + 2UL + (ulong)codingMode;
        if (lpc) bits += (ulong)(4 + 5 + order * precision);
        return bits + optBits;
    }

    /// <summary>subframe_count_exact: the subframe's exact size in bits from its type, order and rice parameters.</summary>
    private static int SubframeBits(ArrayView<int> r, int slot, int n, ArrayView<int> subs, int sr, ArrayView<int> parms, int precision)
    {
        int type = subs[sr + SType], obits = subs[sr + SObits];
        long count = 8 + subs[sr + SWasted];
        if (type == TypeConstant) count += obits;
        else if (type == TypeVerbatim) count += (long)n * obits;
        else
        {
            int order = subs[sr + SOrder];
            count += (long)order * obits;
            if (type == TypeLpc) count += 4 + 5 + order * precision;
            count += 2 + 4;
            int porder = subs[sr + SPorder];
            int psize = n >> porder;
            int cm = subs[sr + SCodingMode];
            int i = order, end = psize;
            for (int p = 0; p < (1 << porder); p++)
            {
                int k = parms[p];
                count += cm;
                for (int j = i; j < end; j++) count += (Fold(r[slot + j]) >> k) + 1 + k;
                i = end;
                end = end + psize < n ? end + psize : n;
            }
        }
        return (int)count;
    }

    /// <summary>
    /// One thread per subframe: encode_residual_ch. CONSTANT, VERBATIM (frame in verbatim mode, or fewer than 5
    /// samples), FIXED (lpc type fixed, or a block no longer than the maximum order: orders searched by bit estimate),
    /// LPC (the order method over the quantized rows of LpcAnalysisKernel). Leaves the final residual in R, the rice
    /// parameters, and the exact bit count in SBits.
    /// </summary>
    public static void ChooseKernel(Index1D index, ArrayView<int> x, ArrayView<int> r, ArrayView<int> frames, ArrayView<int> subs,
        ArrayView<int> lpcScratch, ArrayView<long> sumScratch, ArrayView<int> parmsAll, ArrayView<int> meta, int verbatimPass)
    {
        int fc = index.X, channels = meta[MChannels], b = meta[MBlockSize];
        int f = fc / channels, c = fc % channels;
        // the verbatim pass re-encodes only the frames FrameBitsKernel marked (FFmpeg: verbatim_only = 1, encode again)
        if (verbatimPass != 0 & frames[f * FrameStride + FVerbatim] != 1) return;
        int n = frames[f * FrameStride + FN];
        int sr = fc * SubStride, slot = Slot(f, c, channels, b);
        var parms = parmsAll.SubView((long)fc * MaxPartitions, MaxPartitions);
        var sums = sumScratch.SubView((long)fc * ScratchLongs, ScratchLongs);
        var table = lpcScratch.SubView((long)fc * LpcInts, LpcInts);
        int obits = subs[sr + SObits], cm = subs[sr + SCodingMode];
        int precision = meta[MPrecision], minP = meta[MMinPorder], maxP = meta[MMaxPorder];
        int minOrder = meta[MMinOrder], maxOrder = meta[MMaxOrder], lpcType = meta[MLpcType], method = meta[MOrderMethod];
        bool verbatimOnly = frames[f * FrameStride + FVerbatim] != 0;

        // CONSTANT
        bool constant = true;
        int s0 = x[slot];
        for (int i = 1; i < n; i++)
        {
            if (x[slot + i] != s0)
            {
                constant = false;
                break;
            }
        }
        int type;
        if (constant)
        {
            type = TypeConstant;
            r[slot] = s0;
        }
        else if (verbatimOnly | n < 5)
        {
            type = TypeVerbatim;
        }
        else if (lpcType == LpcNone | lpcType == LpcFixed | n <= maxOrder)
        {
            type = ChooseFixed(x, r, slot, n, obits, cm, precision, minOrder, maxOrder, minP, maxP, sums, parms, subs, sr);
        }
        else
        {
            type = ChooseLpc(x, r, slot, n, obits, cm, precision, minOrder, maxOrder, method, minP, maxP, sums, parms, subs, sr, table);
        }
        if (type == TypeVerbatim)
        {
            for (int i = 0; i < n; i++) r[slot + i] = x[slot + i];
            subs[sr + SOrder] = 0;
        }
        subs[sr + SType] = type;
        subs[sr + SBits] = SubframeBits(r, slot, n, subs, sr, parms, precision);
    }

    private static int ChooseFixed(ArrayView<int> x, ArrayView<int> r, int slot, int n, int obits, int cm, int precision,
        int minOrder, int maxOrder, int minP, int maxP, ArrayView<long> sums, ArrayView<int> parms, ArrayView<int> subs, int sr)
    {
        if (maxOrder > 4) maxOrder = 4;
        // bits[0] = UINT32_MAX; bits[i] of each tried order; opt_order the first order with the fewest bits
        int opt = 0;
        ulong bits0 = 0xFFFFFFFFUL, optBits = 0xFFFFFFFFUL;
        for (int i = minOrder; i <= maxOrder; i++)
        {
            FixedResidual(x, r, slot, n, i);
            ulong bits = RiceParams(r, slot, n, i, obits, cm, false, precision, minP, maxP, sums, parms, subs, sr, true);
            if (i == 0) bits0 = bits;
            ulong current = opt == 0 ? bits0 : optBits;
            if (bits < current)
            {
                opt = i;
                optBits = bits;
            }
        }
        if (opt == 0 & bits0 == 0xFFFFFFFFUL) return TypeVerbatim;   // no order tried (cannot happen for <= 29-bit input)
        subs[sr + SOrder] = opt;
        if (opt != maxOrder)
        {
            FixedResidual(x, r, slot, n, opt);
            RiceParams(r, slot, n, opt, obits, cm, false, precision, minP, maxP, sums, parms, subs, sr, true);
        }
        return TypeFixed;
    }

    private static int ChooseLpc(ArrayView<int> x, ArrayView<int> r, int slot, int n, int obits, int cm, int precision,
        int minOrder, int maxOrder, int method, int minP, int maxP, ArrayView<long> sums, ArrayView<int> parms, ArrayView<int> subs, int sr,
        ArrayView<int> table)
    {
        int opt = subs[sr + SOrder];   // the estimate (ORDER_METHOD_EST)
        if (method == Order2Level | method == Order4Level | method == Order8Level)
        {
            int levels = 1 << method;
            int order = -1, optIndex = levels - 1;
            opt = maxOrder - 1;
            ulong optBits = 0xFFFFFFFFUL;
            for (int i = levels - 1; i >= 0; i--)
            {
                int last = order;
                order = minOrder + (((maxOrder - minOrder + 1) * (i + 1)) / levels) - 1;
                if (order < minOrder - 1) order = minOrder - 1;
                if (order > maxOrder - 1) order = maxOrder - 1;
                if (order == last) continue;
                if (!LpcResidual(x, r, slot, n, table, order, order + 1, table[LpcShifts + order])) continue;
                ulong bits = RiceParams(r, slot, n, order + 1, obits, cm, true, precision, minP, maxP, sums, parms, subs, sr, false);
                if (bits < optBits)
                {
                    optBits = bits;
                    optIndex = i;
                    opt = order;
                }
            }
            _ = optIndex;
            opt++;
        }
        else if (method == OrderSearch)
        {
            ulong optBits = 0xFFFFFFFFUL;
            int o = 0;
            for (int i = minOrder - 1; i < maxOrder; i++)
            {
                if (!LpcResidual(x, r, slot, n, table, i, i + 1, table[LpcShifts + i])) continue;
                ulong bits = RiceParams(r, slot, n, i + 1, obits, cm, true, precision, minP, maxP, sums, parms, subs, sr, false);
                if (bits < optBits)
                {
                    optBits = bits;
                    o = i;
                }
            }
            opt = o + 1;
        }
        else if (method == OrderLog)
        {
            int o = minOrder - 1 + (maxOrder - minOrder) / 3;
            // bits[] memset to -1 (UINT64_MAX): one word per order after the partition sums
            for (int i = 0; i < MaxLpcOrder; i++) sums[MaxPartitions + i] = -1;
            for (int step = 16; step > 0; step >>= 1)
            {
                int last = o;
                for (int i = last - step; i <= last + step; i += step)
                {
                    if (i < minOrder - 1 | i >= maxOrder) continue;
                    if ((ulong)sums[MaxPartitions + i] < 0xFFFFFFFFUL) continue;
                    if (!LpcResidual(x, r, slot, n, table, i, i + 1, table[LpcShifts + i])) continue;
                    ulong bits = RiceParams(r, slot, n, i + 1, obits, cm, true, precision, minP, maxP, sums, parms, subs, sr, false);
                    sums[MaxPartitions + i] = (long)bits;
                    if (bits < (ulong)sums[MaxPartitions + o]) o = i;
                }
            }
            opt = o + 1;
        }
        subs[sr + SOrder] = opt;
        subs[sr + SShift] = table[LpcShifts + opt - 1];
        for (int j = 0; j < opt; j++) subs[sr + SCoef + j] = table[LpcTable + (opt - 1) * MaxLpcOrder + j];
        if (!LpcResidual(x, r, slot, n, table, opt - 1, opt, subs[sr + SShift])) return TypeVerbatim;
        RiceParams(r, slot, n, opt, obits, cm, true, precision, minP, maxP, sums, parms, subs, sr, true);
        return TypeLpc;
    }
}
