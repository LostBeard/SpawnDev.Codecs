// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// The MP3 (MPEG-1/2/2.5 Layer III) encoder's kernels, written from ISO/IEC 11172-3 / 13818-3 (no LAME code):
//   * Analysis: the polyphase filterbank (2.4.3.2 / C.1.3: 512-tap window C[i], 64 partial sums, 32-band matrixing),
//     one thread per (channel, time slot, band);
//   * Mdct: frequency inversion of the odd bands, the long-block window and 36 -> 18 MDCT, one thread per
//     (channel, granule, band); AliasReduce: the encoder's 8 butterflies between neighbouring bands, one thread each;
//   * Quantize: one thread per (granule, channel): the global gain that fits the granule's bit budget (binary search),
//     big_values / count1 regions, region split and Huffman tables by least bits;
//   * WriteGranule: one thread per (granule, channel): the granule's Huffman bits into its slot;
//   * WriteFrame: one thread per frame: header, side info, the granules' bits, stuffing.

using ILGPU;
using ILGPU.Algorithms;

namespace SpawnDev.Codecs.Audio.Mp3;

/// <summary>Kernels of the MP3 encoder (see the file header).</summary>
public static class Mp3EncodeKernels
{
    /// <summary>Per-(granule, channel) side info ints.</summary>
    public const int SPart23 = 0, SBigValues = 1, SGlobalGain = 2, SScfCompress = 3, STable0 = 4, STable1 = 5, STable2 = 6,
        SRegion0 = 7, SRegion1 = 8, SCount1Table = 9, SBudget = 10, SScfScale = 11, SScf = 12, SideInts = 33;

    /// <summary>The psychoacoustic constants view (floats per long band): Bark centre [0..22), the threshold of hearing as
    /// an MDCT energy [22..44).</summary>
    public const int PBark = 0, PAth = 22, PsyConsts = 44;

    /// <summary>Ints of the quantizer's per-unit work view: band-over flags (22), best scalefactors (21).</summary>
    public const int WorkInts = 43;

    /// <summary>Largest scalefactor of a long band: 15 in bands 0..10, 7 in 11..20 (4 / 3 bits), none in band 21.</summary>
    private static int ScfMax(int b) => b < 11 ? 15 : b < 21 ? 7 : 0;

    /// <summary>Bytes of one granule's main-data slot (part2_3_length is at most 4095 bits).</summary>
    public const int SlotBytes = 512;

    /// <summary>The constants view (floats): C window [0..512), M matrix 32 x 64 [512..2560), MDCT cos 18 x 36
    /// [2560..3208), long window sin 36 [3208..3244), cs 8 [3244..3252), ca 8 [3252..3260).</summary>
    public const int FWin = 0, FMat = 512, FCos = 2560, FSin = 3208, FCs = 3244, FCa = 3252, FloatConsts = 3260;

    /// <summary>The int tables view: Huffman codes [0..8192) (32 x 256), lengths [8192..16384), linbits [16384..16416),
    /// max value [16416..16448), count1 codes [16448..16480), count1 lengths [16480..16512), long band bounds of the
    /// stream's rate [16512..16535).</summary>
    public const int IHCode = 0, IHLen = 8192, ILinbits = 16384, IMax = 16416, IC1Code = 16448, IC1Len = 16480, IBounds = 16512,
        IntConsts = 16535;

    // ------------------------------------------------------------------ analysis filterbank

    /// <summary>
    /// One thread per (channel, slot, i in 0..63): the partial sum Y[i] = sum_j C[i + 64 j] * X[i + 64 j] of time slot t,
    /// where X[m] = input[32 t + 31 - m] of the channel (zero outside the input). y[(c * slots + t) * 64 + i].
    /// </summary>
    public static void PolyphaseKernel(Index1D index, ArrayView<float> pcm, ArrayView<float> consts, ArrayView<float> y,
        int channels, long samples, int slots, long first)
    {
        long id = first + index;
        int i = (int)(id % 64);
        long rest = id / 64;
        long t = rest % slots;
        int c = (int)(rest / slots);
        long newest = 32L * t + 31;
        float acc = 0;
        for (int j = 0; j < 8; j++)
        {
            int m = i + 64 * j;
            long n = newest - m;
            if (n >= 0 && n < samples) acc += consts[FWin + m] * pcm[n * channels + c];
        }
        y[id] = acc;
    }

    /// <summary>One thread per (channel, slot, band k): S[k] = sum_i M[k][i] * Y[i]; sub[(c * slots + t) * 32 + k].</summary>
    public static void MatrixKernel(Index1D index, ArrayView<float> y, ArrayView<float> consts, ArrayView<float> sub, long first)
    {
        long id = first + index;
        int k = (int)(id % 32);
        long ct = id / 32;
        float s = 0;
        for (int i = 0; i < 64; i++) s += consts[FMat + k * 64 + i] * y[ct * 64 + i];
        sub[id] = s;
    }

    // ------------------------------------------------------------------ MDCT

    /// <summary>
    /// One thread per (channel, granule, band): the 36 subband samples of granules g - 1 and g (zeros before the first),
    /// odd bands' odd time samples inverted, the long window, the MDCT; xr[(g * channels + c) * 576 + band * 18 + k].
    /// </summary>
    public static void MdctKernel(Index1D index, ArrayView<float> sub, ArrayView<float> consts, ArrayView<float> xr,
        int granules, int slots, int channels, long first)
    {
        long id = first + index;
        int sb = (int)(id % 32);
        long rest = id / 32;
        int g = (int)(rest % granules);
        int c = (int)(rest / granules);
        long outAt = ((long)g * channels + c) * 576 + sb * 18;
        for (int k = 0; k < 18; k++)
        {
            float acc = 0;
            for (int i = 0; i < 36; i++)
            {
                int t = (g - 1) * 18 + i;   // slot of this sample
                if (t < 0) continue;
                float v = sub[((long)c * slots + t) * 32 + sb];
                if ((sb & 1) != 0 && (i & 1) != 0) v = -v;   // frequency inversion (time index within the granule: i % 18)
                acc += v * consts[FSin + i] * consts[FCos + k * 36 + i];
            }
            xr[outAt + k] = acc;
        }
    }

    /// <summary>One thread per (channel-granule, boundary sb 0..30, butterfly i 0..7): the encoder's alias reduction.</summary>
    public static void AliasReduceKernel(Index1D index, ArrayView<float> xr, ArrayView<float> consts, long first)
    {
        long id = first + index;
        int i = (int)(id % 8);
        long rest = id / 8;
        int sb = (int)(rest % 31);
        long cg = rest / 31;
        long a = cg * 576 + sb * 18 + 17 - i, b = cg * 576 + (sb + 1) * 18 + i;
        float bu = xr[a], bd = xr[b];
        float cs = consts[FCs + i], ca = consts[FCa + i];
        xr[a] = bu * cs + bd * ca;
        xr[b] = bd * cs - bu * ca;
    }

    // ------------------------------------------------------------------ frame layout (CBR)

    /// <summary>
    /// One thread per frame (and one more for the end): the frame's byte offset f * coeff * kbps / rate (a padding byte
    /// where the running total needs it), its bit rate index and padding bit, and the bit budget of each of its granules
    /// (the frame's main data split evenly, at most 4095 bits).
    /// </summary>
    public static void FrameLayoutKernel(Index1D index, ArrayView<long> frameOffsets, ArrayView<int> frameInfo, ArrayView<int> side,
        long frames, long num, int sampleRate, int bitrateIndex, int sideBytes, int granulesPerFrame, int channels)
    {
        long f = index;
        long t0 = f * num / sampleRate;
        frameOffsets[f] = t0;
        if (f >= frames) return;
        long size = (f + 1) * num / sampleRate - t0;
        frameInfo[f * 2] = bitrateIndex;
        frameInfo[f * 2 + 1] = (int)(size - num / sampleRate);
        int perGranule = (int)XMath.Min(4095L, (size - 4 - sideBytes) * 8 / (granulesPerFrame * channels));
        for (int g = 0; g < granulesPerFrame; g++)
            for (int c = 0; c < channels; c++)
                side[((f * granulesPerFrame + g) * channels + c) * SideInts + SBudget] = perGranule;
    }

    // ------------------------------------------------------------------ psychoacoustic model (MDCT domain)

    /// <summary>
    /// One thread per (granule, channel): the noise each long scalefactor band may carry (xmin, in MDCT energy), from the
    /// granule's own spectrum. Band energies; tonality from the spectral flatness of a band's lines (white noise sits
    /// near -5.5 dB, a tone far below); energy spread across bands with Schroeder's spreading function in Bark; the
    /// threshold below the spread energy by ISO model 2's offsets (18 dB under tonal maskers, 6 dB under noise); never
    /// below the threshold of hearing. <paramref name="quality"/> scales the result (1 = the model's threshold).
    /// <paramref name="scratch"/> holds 44 floats per unit (energies, tonalities).
    /// </summary>
    public static void PsyKernel(Index1D i0, ArrayView<float> xr, ArrayView<int> tables, ArrayView<float> psy, ArrayView<float> scratch,
        ArrayView<float> xmin, float quality, int first)
    {
        int index = first + i0;
        long at = (long)index * 576;
        long so = (long)index * 44;
        for (int b = 0; b < 22; b++)
        {
            int lo = tables[IBounds + b], hi = tables[IBounds + b + 1];
            float e = 0, logSum = 0;
            for (int i = lo; i < hi; i++)
            {
                float v = xr[at + i] * xr[at + i];
                e += v;
                logSum += XMath.Log(v + 1e-30f);
            }
            int n = hi - lo;
            float sfmDb = 10f * XMath.Log10(XMath.Exp(logSum / n) / (e / n + 1e-30f) + 1e-30f);
            scratch[so + b] = e;
            scratch[so + 22 + b] = XMath.Clamp((sfmDb + 5.5f) / -25f, 0f, 1f);
        }
        for (int b = 0; b < 22; b++)
        {
            float zb = psy[PBark + b];
            float spread = 0, toneWeighted = 0;
            for (int j = 0; j < 22; j++)
            {
                float ej = scratch[so + j];
                if (ej <= 0) continue;
                float dz = zb - psy[PBark + j] + 0.474f;
                float sfDb = 15.81f + 7.5f * dz - 17.5f * XMath.Sqrt(1f + dz * dz);
                if (sfDb < -60f) continue;
                float w = ej * XMath.Pow(10f, sfDb / 10f);
                spread += w;
                toneWeighted += w * scratch[so + 22 + j];
            }
            float t = spread > 0 ? toneWeighted / spread : 0;
            float offsetDb = 18f * t + 6f * (1 - t);
            float thr = spread * XMath.Pow(10f, -offsetDb / 10f);
            xmin[(long)index * 22 + b] = XMath.Max(thr, psy[PAth + b]) * quality;
        }
    }

    // ------------------------------------------------------------------ quantization + bit counting

    private static int Quant(float x, float stepInv)
    {
        float a = XMath.Abs(x) * stepInv;
        float v = XMath.Pow(a, 0.75f) - 0.0946f;
        return v <= 0 ? 0 : (int)(v + 0.5f);
    }

    /// <summary>Bits of the pairs [from, to) (sample indices) with table t (0 = all zero).</summary>
    private static int PairBits(ArrayView<int> ix, long at, int from, int to, int t, ArrayView<int> tables)
    {
        if (t == 0) return 0;
        int lin = tables[ILinbits + t];
        int bits = 0;
        for (int i = from; i < to; i += 2)
        {
            int x = XMath.Abs(ix[at + i]), y = XMath.Abs(ix[at + i + 1]);
            int cx = x > 15 ? 15 : x, cy = y > 15 ? 15 : y;
            bits += tables[IHLen + t * 256 + cx * 16 + cy];
            if (x != 0) bits++;
            if (y != 0) bits++;
            if (lin > 0 && x >= 15) bits += lin;
            if (lin > 0 && y >= 15) bits += lin;
        }
        return bits;
    }

    /// <summary>The table with the fewest bits for pairs [from, to) whose largest value is <paramref name="max"/>; bits out.</summary>
    private static int BestTable(ArrayView<int> ix, long at, int from, int to, int max, ArrayView<int> tables, out int bits)
    {
        bits = 0;
        if (from >= to || max == 0) return 0;
        int best = -1, bestBits = int.MaxValue;
        if (max <= 15)
        {
            for (int t = 1; t < 16; t++)
            {
                if (t == 4 || t == 14 || tables[IMax + t] < max) continue;
                int b = PairBits(ix, at, from, to, t, tables);
                if (b < bestBits) { bestBits = b; best = t; }
            }
        }
        // escape tables: in each group (16..23, 24..31) the codes are shared, the fewest linbits that reach max is best
        for (int grp = 16; grp < 32; grp += 8)
        {
            int t = -1;
            for (int u = grp; u < grp + 8; u++)
                if (max - 15 < (1 << tables[ILinbits + u]) || (max <= 15 && u == grp)) { t = u; break; }
            if (t < 0) continue;
            int b = PairBits(ix, at, from, to, t, tables);
            if (b < bestBits) { bestBits = b; best = t; }
        }
        bits = bestBits;
        return best;
    }

    private static int MaxAbs(ArrayView<int> ix, long at, int from, int to)
    {
        int m = 0;
        for (int i = from; i < to; i++) m = XMath.Max(m, XMath.Abs(ix[at + i]));
        return m;
    }

    /// <summary>
    /// Regions of a quantized granule: big_values (pairs) and the count1 end (samples); count1 bits with the better
    /// table out.
    /// </summary>
    private static void Regions(ArrayView<int> ix, long at, ArrayView<int> tables, out int bigValues, out int count1End,
        out int count1Table, out int count1Bits)
    {
        int last = 575;
        while (last >= 0 && ix[at + last] == 0) last--;
        // count1 region: quadruples of |v| <= 1 going down from the last non-zero value
        int end = last < 0 ? 0 : ((last >> 2) + 1) << 2;
        if (end > 576) end = 576;
        int start = end;
        while (start >= 4)
        {
            bool small = true;
            for (int i = start - 4; i < start; i++) small &= XMath.Abs(ix[at + i]) <= 1;
            if (!small) break;
            start -= 4;
        }
        // big_values must cover pairs; start is a multiple of 4 so it is even
        bigValues = start / 2;
        if (bigValues > 288) { bigValues = 288; }
        count1End = end;
        int ba = 0, bb = 0;
        for (int q = start; q < end; q += 4)
        {
            int v = XMath.Abs(ix[at + q]), w = XMath.Abs(ix[at + q + 1]), x = XMath.Abs(ix[at + q + 2]), y = XMath.Abs(ix[at + q + 3]);
            int code = v * 8 + w * 4 + x * 2 + y;
            int signs = v + w + x + y;
            ba += tables[IC1Len + code] + signs;
            bb += tables[IC1Len + 16 + code] + signs;
        }
        count1Table = bb < ba ? 1 : 0;
        count1Bits = bb < ba ? bb : ba;
    }

    /// <summary>
    /// Huffman bits of a granule with the region split (region0_count, region1_count) and best tables; returns the bits
    /// and the chosen tables. <paramref name="exhaustive"/> tries every split, otherwise a fixed split.
    /// </summary>
    private static int BigValueBits(ArrayView<int> ix, long at, ArrayView<int> tables, int bigValues, bool exhaustive,
        out int region0, out int region1, out int t0, out int t1, out int t2)
    {
        int end = bigValues * 2;
        region0 = 0; region1 = 0; t0 = 0; t1 = 0; t2 = 0;
        if (end == 0) return 0;
        int best = int.MaxValue;
        int r0Max = exhaustive ? 15 : 0, r1Max = exhaustive ? 7 : 0;
        for (int r0 = exhaustive ? 0 : 7; r0 <= (exhaustive ? r0Max : 7); r0++)
        {
            int b0 = tables[IBounds + XMath.Min(r0 + 1, 22)];
            for (int r1 = exhaustive ? 0 : 7; r1 <= (exhaustive ? r1Max : 7); r1++)
            {
                if (r0 + r1 + 2 > 22) break;
                int b1 = tables[IBounds + XMath.Min(r0 + r1 + 2, 22)];
                int e0 = XMath.Min(b0, end), e1 = XMath.Min(b1, end);
                int m0 = MaxAbs(ix, at, 0, e0), m1 = MaxAbs(ix, at, e0, e1), m2 = MaxAbs(ix, at, e1, end);
                int a0 = BestTable(ix, at, 0, e0, m0, tables, out int x0);
                int a1 = BestTable(ix, at, e0, e1, m1, tables, out int x1);
                int a2 = BestTable(ix, at, e1, end, m2, tables, out int x2);
                int bits = x0 + x1 + x2;
                if (bits < best)
                {
                    best = bits;
                    region0 = r0; region1 = r1; t0 = a0; t1 = a1; t2 = a2;
                }
            }
        }
        return best;
    }

    /// <summary>Bits of the scalefactors (part2) and scalefac_compress for the current maxima; -1 when they do not fit.</summary>
    private static int ScfBits(ArrayView<int> side, long s, int lsf, out int compress)
    {
        int m1 = 0, m2 = 0, m3 = 0, m4 = 0;
        for (int b = 0; b < 21; b++)
        {
            int v = side[s + SScf + b];
            if (b < 6) m1 = XMath.Max(m1, v);
            else if (b < 11) m2 = XMath.Max(m2, v);
            else if (b < 16) m3 = XMath.Max(m3, v);
            else m4 = XMath.Max(m4, v);
        }
        compress = 0;
        if (lsf == 0)
        {
            // MPEG-1: slen1 for bands 0..10, slen2 for 11..20; the scalefac_compress with the fewest bits that holds them
            int best = -1;
            for (int c = 0; c < 16; c++)
            {
                int s1 = Slen1(c), s2 = Slen2(c);
                if ((1 << s1) <= XMath.Max(m1, m2) || (1 << s2) <= XMath.Max(m3, m4)) continue;
                int bits = 11 * s1 + 10 * s2;
                if (best < 0 || bits < best) { best = bits; compress = c; }
            }
            return best;
        }
        // MPEG-2 / 2.5 long blocks, no intensity stereo: partitions of 6, 5, 5, 5 bands, slen1 / slen2 0..4, slen3 / slen4 0..3
        int l1 = BitsFor(m1), l2 = BitsFor(m2), l3 = BitsFor(m3), l4 = BitsFor(m4);
        if (l1 > 4 || l2 > 4 || l3 > 3 || l4 > 3) return -1;
        compress = ((l1 * 5 + l2) << 4) + (l3 << 2) + l4;
        return 6 * l1 + 5 * l2 + 5 * l3 + 5 * l4;
    }

    private static int BitsFor(int v)
    {
        int n = 0;
        while ((1 << n) <= v) n++;
        return n;
    }

    private static int Slen1(int c) => c < 4 ? 0 : c == 4 ? 3 : c < 8 ? 1 : c < 11 ? 2 : c < 14 ? 3 : 4;
    private static int Slen2(int c) => c < 4 ? c : c == 4 ? 0 : c < 8 ? c - 4 : c < 11 ? c - 7 : c < 14 ? c - 10 : c - 12;

    /// <summary>The factor the quantizer applies to band b: 2^(multiplier * scalefactor), multiplier 0.5 or 1.</summary>
    private static float Amp(ArrayView<int> side, long s, int b) =>
        b >= 21 ? 1f : XMath.Pow(2f, (side[s + SScfScale] != 0 ? 1f : 0.5f) * side[s + SScf + b]);

    /// <summary>Quantize the granule at gain g with the current scalefactors into ix.</summary>
    private static void QuantizeAll(ArrayView<float> xr, ArrayView<int> ix, long at, ArrayView<int> side, long s, ArrayView<int> tables, int g)
    {
        float stepInv = XMath.Pow(2f, -(g - 210) / 4f);
        for (int b = 0; b < 22; b++)
        {
            float k = stepInv * Amp(side, s, b);
            int lo = tables[IBounds + b], hi = tables[IBounds + b + 1];
            for (int i = lo; i < hi; i++) ix[at + i] = Quant(xr[at + i], k) * (xr[at + i] < 0 ? -1 : 1);
        }
    }

    /// <summary>Inner loop: the smallest gain in [lo, 255] whose Huffman bits fit; -1 when none does.</summary>
    private static int InnerLoop(ArrayView<float> xr, ArrayView<int> ix, long at, ArrayView<int> side, long s, ArrayView<int> tables,
        int lo, int huffBudget)
    {
        int hi = 255, found = -1;
        while (lo <= hi)
        {
            int g = (lo + hi) >> 1;
            QuantizeAll(xr, ix, at, side, s, tables, g);
            Regions(ix, at, tables, out int bv, out _, out _, out int c1Bits);
            int bits = BigValueBits(ix, at, tables, bv, false, out _, out _, out _, out _, out _) + c1Bits;
            if (bits <= huffBudget && MaxAbs(ix, at, 0, 576) <= 8206) { found = g; hi = g - 1; }
            else lo = g + 1;
        }
        return found;
    }

    /// <summary>
    /// One thread per (granule, channel): ISO 11172-3 Annex C's two loops. The inner loop finds the smallest global gain
    /// whose bits (scalefactors + Huffman) fit side[SBudget]; the outer loop measures the noise in each long band against
    /// its allowed noise xmin and amplifies (scalefactor + 1) every band over it, switching to scalefac_scale 1 when a
    /// scalefactor would overflow, until no band is over or nothing can be amplified. The iteration with the fewest bands
    /// over (then the least noise above the masks) is kept; ix and the side info written.
    /// </summary>
    public static void QuantizeKernel(Index1D i0, ArrayView<float> xr, ArrayView<int> ix, ArrayView<int> side, ArrayView<int> tables,
        ArrayView<float> xmin, ArrayView<int> work, int lsf, int first)
    {
        int index = first + i0;
        long at = (long)index * 576;
        long s = (long)index * SideInts;
        long xo = (long)index * 22;
        long wo = (long)index * WorkInts;   // [0..22) band over its mask, [22..43) the best iteration's scalefactors
        int budget = XMath.Min(side[s + SBudget], 4095);
        side[s + SScfScale] = 0;
        for (int b = 0; b < 21; b++) side[s + SScf + b] = 0;
        float maxAbs = 0;
        for (int i = 0; i < 576; i++) maxAbs = XMath.Max(maxAbs, XMath.Abs(xr[at + i]));
        if (maxAbs < 1e-20f || budget <= 0)
        {
            for (int i = 0; i < 576; i++) ix[at + i] = 0;
            side[s + SPart23] = 0; side[s + SBigValues] = 0; side[s + SGlobalGain] = 0; side[s + SScfCompress] = 0;
            side[s + STable0] = 0; side[s + STable1] = 0; side[s + STable2] = 0; side[s + SRegion0] = 0; side[s + SRegion1] = 0;
            side[s + SCount1Table] = 0;
            return;
        }
        // the smallest gain that keeps every value within the escape range (8206), before amplification
        float need = 4f * XMath.Log2(maxAbs / XMath.Pow(8206.5f + 0.0946f, 4f / 3f));
        int gMin = XMath.Clamp((int)XMath.Ceiling(210 + need), 0, 255);

        // best iteration so far: (bands over, noise above the masks in dB), its gain, scalefac_scale and scalefactors
        int bestOver = int.MaxValue, bestGain = 255, bestScale = 0;
        float bestOverDb = float.MaxValue;
        for (int b = 0; b < 21; b++) work[wo + 22 + b] = 0;
        for (int iter = 0; iter < 64; iter++)
        {
            int part2 = ScfBits(side, s, lsf, out _);
            if (part2 < 0) break;
            int g = InnerLoop(xr, ix, at, side, s, tables, gMin, budget - part2);
            if (g < 0) break;
            QuantizeAll(xr, ix, at, side, s, tables, g);
            // noise per band against its mask
            float step = XMath.Pow(2f, (g - 210) / 4f);
            int over = 0;
            float overDb = 0;
            for (int b = 0; b < 22; b++)
            {
                float inv = step / Amp(side, s, b);
                int lo = tables[IBounds + b], hi = tables[IBounds + b + 1];
                float noise = 0;
                for (int i = lo; i < hi; i++)
                {
                    float q = XMath.Pow(XMath.Abs(ix[at + i]), 4f / 3f) * inv;
                    float d = XMath.Abs(xr[at + i]) - q;
                    noise += d * d;
                }
                float allowed = xmin[xo + b];
                bool isOver = noise > allowed;
                if (isOver)
                {
                    over++;
                    overDb += 10f * XMath.Log10(noise / allowed);
                }
                work[wo + b] = isOver ? 1 : 0;
            }
            if (over < bestOver || (over == bestOver && overDb < bestOverDb))
            {
                bestOver = over; bestOverDb = overDb; bestGain = g; bestScale = side[s + SScfScale];
                for (int b = 0; b < 21; b++) work[wo + 22 + b] = side[s + SScf + b];
            }
            if (over == 0) break;
            // amplify every band over its mask
            bool overflow = false, amplified = false;
            for (int b = 0; b < 21; b++)
            {
                if (work[wo + b] == 0) continue;
                side[s + SScf + b]++;
                amplified = true;
                if (side[s + SScf + b] > ScfMax(b)) overflow = true;
            }
            if (!amplified) break;
            if (overflow)
            {
                if (side[s + SScfScale] != 0) break;
                // scalefac_scale 1 doubles the step of every scalefactor: halve them (rounding up keeps the amplification)
                side[s + SScfScale] = 1;
                for (int b = 0; b < 21; b++) side[s + SScf + b] = (side[s + SScf + b] + 1) >> 1;
            }
        }
        // the best iteration, then the best region split and tables
        side[s + SScfScale] = bestScale;
        for (int b = 0; b < 21; b++) side[s + SScf + b] = work[wo + 22 + b];
        int p2 = ScfBits(side, s, lsf, out int compress);
        if (p2 < 0)
        {
            // cannot happen for a kept iteration; fall back to no scalefactors
            side[s + SScfScale] = 0;
            for (int b = 0; b < 21; b++) side[s + SScf + b] = 0;
            p2 = 0; compress = 0;
            bestGain = XMath.Max(InnerLoop(xr, ix, at, side, s, tables, gMin, budget), 0);
        }
        int gain = bestGain;
        while (true)
        {
            QuantizeAll(xr, ix, at, side, s, tables, gain);
            Regions(ix, at, tables, out int bv, out _, out int c1t, out int c1Bits);
            int bits = BigValueBits(ix, at, tables, bv, true, out int r0, out int r1, out int t0, out int t1, out int t2) + c1Bits;
            if (bits + p2 <= budget || gain == 255)
            {
                side[s + SPart23] = bits + p2;
                side[s + SBigValues] = bv;
                side[s + SGlobalGain] = gain;
                side[s + SScfCompress] = compress;
                side[s + STable0] = t0; side[s + STable1] = t1; side[s + STable2] = t2;
                side[s + SRegion0] = r0; side[s + SRegion1] = r1;
                side[s + SCount1Table] = c1t;
                return;
            }
            gain++;
        }
    }

    // ------------------------------------------------------------------ bit writing

    private static void PutBits(ArrayView<byte> buf, long baseByte, ref int bitPos, uint value, int n)
    {
        for (int k = n - 1; k >= 0; k--)
        {
            if (((value >> k) & 1) != 0)
            {
                long at = baseByte + (bitPos >> 3);
                buf[at] = (byte)(buf[at] | (0x80 >> (bitPos & 7)));
            }
            bitPos++;
        }
    }

    /// <summary>One thread per (granule, channel): the Huffman-coded values into slot[index * SlotBytes] (zeroed first).</summary>
    public static void WriteGranuleKernel(Index1D i0, ArrayView<int> ix, ArrayView<int> side, ArrayView<int> tables, ArrayView<byte> slots,
        int lsf, int first)
    {
        int index = first + i0;
        long at = (long)index * 576;
        long s = (long)index * SideInts;
        long slot = (long)index * SlotBytes;
        for (int i = 0; i < SlotBytes; i++) slots[slot + i] = 0;
        int bitPos = 0;
        // part2: the scalefactors of long bands 0..20 (scfsi 0)
        int compress = side[s + SScfCompress];
        for (int b = 0; b < 21; b++)
        {
            int n;
            if (lsf == 0) n = b < 11 ? Slen1(compress) : Slen2(compress);
            else
            {
                int sl1 = (compress >> 4) / 5, sl2 = (compress >> 4) % 5, sl3 = (compress >> 2) & 3, sl4 = compress & 3;
                n = b < 6 ? sl1 : b < 11 ? sl2 : b < 16 ? sl3 : sl4;
            }
            PutBits(slots, slot, ref bitPos, (uint)side[s + SScf + b], n);
        }
        int bv = side[s + SBigValues];
        int end = bv * 2;
        int b0 = XMath.Min(tables[IBounds + XMath.Min(side[s + SRegion0] + 1, 22)], end);
        int b1 = XMath.Min(tables[IBounds + XMath.Min(side[s + SRegion0] + side[s + SRegion1] + 2, 22)], end);
        for (int i = 0; i < end; i += 2)
        {
            int t = i < b0 ? side[s + STable0] : i < b1 ? side[s + STable1] : side[s + STable2];
            if (t == 0) continue;
            int lin = tables[ILinbits + t];
            int x = ix[at + i], y = ix[at + i + 1];
            int ax = XMath.Abs(x), ay = XMath.Abs(y);
            int cx = ax > 15 ? 15 : ax, cy = ay > 15 ? 15 : ay;
            PutBits(slots, slot, ref bitPos, (uint)tables[IHCode + t * 256 + cx * 16 + cy], tables[IHLen + t * 256 + cx * 16 + cy]);
            if (lin > 0 && ax >= 15) PutBits(slots, slot, ref bitPos, (uint)(ax - 15), lin);
            if (ax != 0) PutBits(slots, slot, ref bitPos, x < 0 ? 1u : 0u, 1);
            if (lin > 0 && ay >= 15) PutBits(slots, slot, ref bitPos, (uint)(ay - 15), lin);
            if (ay != 0) PutBits(slots, slot, ref bitPos, y < 0 ? 1u : 0u, 1);
        }
        // count1
        int last = 575;
        while (last >= 0 && ix[at + last] == 0) last--;
        int c1End = last < 0 ? 0 : XMath.Min(576, ((last >> 2) + 1) << 2);
        int tbl = side[s + SCount1Table];
        for (int q = end; q < c1End; q += 4)
        {
            int v = ix[at + q], w = ix[at + q + 1], x = ix[at + q + 2], y = ix[at + q + 3];
            int code = (v != 0 ? 8 : 0) + (w != 0 ? 4 : 0) + (x != 0 ? 2 : 0) + (y != 0 ? 1 : 0);
            PutBits(slots, slot, ref bitPos, (uint)tables[IC1Code + tbl * 16 + code], tables[IC1Len + tbl * 16 + code]);
            if (v != 0) PutBits(slots, slot, ref bitPos, v < 0 ? 1u : 0u, 1);
            if (w != 0) PutBits(slots, slot, ref bitPos, w < 0 ? 1u : 0u, 1);
            if (x != 0) PutBits(slots, slot, ref bitPos, x < 0 ? 1u : 0u, 1);
            if (y != 0) PutBits(slots, slot, ref bitPos, y < 0 ? 1u : 0u, 1);
        }
    }

    /// <summary>
    /// One thread per frame: header, side info and the granules' bits (main_data_begin 0: each frame holds its own main
    /// data), at out[frameOffsets[f]]; frameInfo[f] = (bitrate index, padding). <paramref name="lsf"/> = MPEG-2/2.5.
    /// </summary>
    public static void WriteFrameKernel(Index1D i0, ArrayView<int> side, ArrayView<byte> slots, ArrayView<long> frameOffsets,
        ArrayView<int> frameInfo, ArrayView<byte> output, int channels, int lsf, int version, int rateIndex, int first)
    {
        int f = first + i0;
        long baseByte = frameOffsets[f];
        long frameBytes = frameOffsets[f + 1] - baseByte;
        for (long i = 0; i < frameBytes; i++) output[baseByte + i] = 0;
        int bitPos = 0;
        int granules = lsf != 0 ? 1 : 2;
        // header
        PutBits(output, baseByte, ref bitPos, 0x7FF, 11);
        PutBits(output, baseByte, ref bitPos, (uint)version, 2);    // 3 = MPEG-1, 2 = MPEG-2, 0 = MPEG-2.5
        PutBits(output, baseByte, ref bitPos, 1, 2);                // layer III
        PutBits(output, baseByte, ref bitPos, 1, 1);                // no CRC
        PutBits(output, baseByte, ref bitPos, (uint)frameInfo[f * 2], 4);
        PutBits(output, baseByte, ref bitPos, (uint)rateIndex, 2);
        PutBits(output, baseByte, ref bitPos, (uint)frameInfo[f * 2 + 1], 1);
        PutBits(output, baseByte, ref bitPos, 0, 1);                // private
        PutBits(output, baseByte, ref bitPos, channels == 1 ? 3u : 0u, 2);   // mono / stereo
        PutBits(output, baseByte, ref bitPos, 0, 2);                // mode extension
        PutBits(output, baseByte, ref bitPos, 0, 1);                // copyright
        PutBits(output, baseByte, ref bitPos, 1, 1);                // original
        PutBits(output, baseByte, ref bitPos, 0, 2);                // emphasis
        // side info
        if (lsf == 0)
        {
            PutBits(output, baseByte, ref bitPos, 0, 9);            // main_data_begin
            PutBits(output, baseByte, ref bitPos, 0, channels == 1 ? 5 : 3);
            PutBits(output, baseByte, ref bitPos, 0, 4 * channels); // scfsi
        }
        else
        {
            PutBits(output, baseByte, ref bitPos, 0, 8);
            PutBits(output, baseByte, ref bitPos, 0, channels == 1 ? 1 : 2);
        }
        for (int g = 0; g < granules; g++)
        {
            for (int c = 0; c < channels; c++)
            {
                long s = (((long)f * granules + g) * channels + c) * SideInts;
                PutBits(output, baseByte, ref bitPos, (uint)side[s + SPart23], 12);
                PutBits(output, baseByte, ref bitPos, (uint)side[s + SBigValues], 9);
                PutBits(output, baseByte, ref bitPos, (uint)side[s + SGlobalGain], 8);
                PutBits(output, baseByte, ref bitPos, (uint)side[s + SScfCompress], lsf != 0 ? 9 : 4);
                PutBits(output, baseByte, ref bitPos, 0, 1);        // window_switching_flag
                PutBits(output, baseByte, ref bitPos, (uint)side[s + STable0], 5);
                PutBits(output, baseByte, ref bitPos, (uint)side[s + STable1], 5);
                PutBits(output, baseByte, ref bitPos, (uint)side[s + STable2], 5);
                PutBits(output, baseByte, ref bitPos, (uint)side[s + SRegion0], 4);
                PutBits(output, baseByte, ref bitPos, (uint)side[s + SRegion1], 3);
                if (lsf == 0) PutBits(output, baseByte, ref bitPos, 0, 1);   // preflag
                PutBits(output, baseByte, ref bitPos, (uint)side[s + SScfScale], 1);   // scalefac_scale
                PutBits(output, baseByte, ref bitPos, (uint)side[s + SCount1Table], 1);
            }
        }
        // main data: the granules in order
        for (int g = 0; g < granules; g++)
        {
            for (int c = 0; c < channels; c++)
            {
                long gi = ((long)f * granules + g) * channels + c;
                int bits = side[gi * SideInts + SPart23];
                long slot = gi * SlotBytes;
                for (int b = 0; b < bits; b++)
                {
                    int bit = (slots[slot + (b >> 3)] >> (7 - (b & 7))) & 1;
                    if (bit != 0)
                    {
                        long o = baseByte + (bitPos >> 3);
                        output[o] = (byte)(output[o] | (0x80 >> (bitPos & 7)));
                    }
                    bitPos++;
                }
            }
        }
    }
}
