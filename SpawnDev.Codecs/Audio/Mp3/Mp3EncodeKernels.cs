// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// The MP3 (MPEG-1/2/2.5 Layer III) encoder's kernels, written from ISO/IEC 11172-3 / 13818-3 (no LAME code):
//   * Analysis: the polyphase filterbank (2.4.3.2 / C.1.3: 512-tap window C[i], 64 partial sums, 32-band matrixing),
//     one thread per (channel, time slot, band);
//   * Attack / BlockType: transient detection per granule (both channels jointly) and the block type sequence
//     (long 0, start 1, short 2, stop 3), one thread per granule;
//   * Mdct: frequency inversion of the odd bands, the block type's window and 36 -> 18 MDCT (long, start, stop) or
//     three 12 -> 6 MDCTs (short, stored in the decoder's order: short band, window, frequency), one thread per
//     (channel, granule, band); AliasReduce: the encoder's 8 butterflies between neighbouring bands (not for short
//     blocks), one thread each;
//   * Quantize: one thread per (granule, channel): the global gain that fits the granule's bit budget (binary search),
//     big_values / count1 regions, region split and Huffman tables by least bits;
//   * WriteGranule: one thread per (granule, channel): the granule's Huffman bits into its slot;
//   * WriteFrame: one thread per frame: header, side info, the granules' bits, stuffing.
//
// A granule's "groups" are its scalefactor bands: 22 long bands, or for a short granule the 39 (short band, window)
// pairs in the decoder's order (group q = short band * 3 + window), 36 of them with scalefactors (short band 12 has
// none).

using ILGPU;
using ILGPU.Algorithms;

namespace SpawnDev.Codecs.Audio.Mp3;

/// <summary>Kernels of the MP3 encoder (see the file header).</summary>
public static class Mp3EncodeKernels
{
    /// <summary>Per-(granule, channel) side info ints.</summary>
    public const int SPart23 = 0, SBigValues = 1, SGlobalGain = 2, SScfCompress = 3, STable0 = 4, STable1 = 5, STable2 = 6,
        SRegion0 = 7, SRegion1 = 8, SCount1Table = 9, SBudget = 10, SScfScale = 11, SScf = 12, SRequant = 51, SBlockType = 52,
        SSubGain = 53, SideInts = 56;

    /// <summary>Allowed-noise floats per (granule, channel) (one per group: 22 long, 39 short).</summary>
    public const int XStride = 39;

    /// <summary>The VBR layout's meta view (longs): total bytes of the audio frames.</summary>
    public const int VAudioBytes = 0, VMeta = 1;

    /// <summary>The psychoacoustic constants view (floats): per long band the Bark centre [0..22) and the threshold of
    /// hearing as an MDCT energy [22..44); the same per short band [44..57), [57..70).</summary>
    public const int PBark = 0, PAth = 22, PSBark = 44, PSAth = 57, PsyConsts = 70;

    /// <summary>The psychoacoustic model's scratch floats per unit: group energies (39), tonalities (39).</summary>
    public const int ScratchFloats = 78;

    /// <summary>Ints of the quantizer's per-unit work view: group-over flags (39), best scalefactors (36).</summary>
    public const int WorkInts = 75;

    /// <summary>
    /// Largest scalefactor of a group: long bands 15 in 0..10, 7 in 11..20 (4 / 3 bits), none in band 21; short groups 15
    /// in short bands 0..5, 7 in 6..11 (MPEG-1 slen1 / slen2; MPEG-2 partitions of 9 values, slen1 / slen2 up to 4 bits,
    /// slen3 / slen4 up to 3), none in short band 12.
    /// </summary>
    private static int ScfMax(int bt, int q) => bt == 2 ? (q < 18 ? 15 : q < 36 ? 7 : 0) : (q < 11 ? 15 : q < 21 ? 7 : 0);

    /// <summary>Groups of a granule: 22 long bands or 39 (short band, window) pairs.</summary>
    private static int Groups(int bt) => bt == 2 ? 39 : 22;

    /// <summary>Groups with a scalefactor: 21 long, 36 short.</summary>
    private static int ScfCount(int bt) => bt == 2 ? 36 : 21;

    /// <summary>First line of group q (q = Groups(bt) gives 576).</summary>
    private static int GroupLo(ArrayView<int> tables, int bt, int q) => tables[(bt == 2 ? IGroup : IBounds) + q];

    /// <summary>Bytes of one granule's main-data slot (part2_3_length is at most 4095 bits).</summary>
    public const int SlotBytes = 512;

    /// <summary>The constants view (floats): C window [0..512), M matrix 32 x 64 [512..2560), MDCT cos 18 x 36
    /// [2560..3208), long window sin 36 [3208..3244), cs 8 [3244..3252), ca 8 [3252..3260), start window 36
    /// [3260..3296), stop window 36 [3296..3332), short window 12 [3332..3344), short MDCT cos 6 x 12 [3344..3416).</summary>
    public const int FWin = 0, FMat = 512, FCos = 2560, FSin = 3208, FCs = 3244, FCa = 3252, FStart = 3260, FStop = 3296,
        FShortWin = 3332, FShortCos = 3344, FloatConsts = 3416;

    /// <summary>The int tables view: Huffman codes [0..8192) (32 x 256), lengths [8192..16384), linbits [16384..16416),
    /// max value [16416..16448), count1 codes [16448..16480), count1 lengths [16480..16512), long band bounds of the
    /// stream's rate [16512..16535), short band bounds within one window [16535..16549), the 39 short groups' first
    /// lines in a granule (+ 576) [16549..16589).</summary>
    public const int IHCode = 0, IHLen = 8192, ILinbits = 16384, IMax = 16416, IC1Code = 16448, IC1Len = 16480, IBounds = 16512,
        IShort = 16535, IGroup = 16549, IntConsts = 16589;

    // ------------------------------------------------------------------ math through Exp2 / Log2 only
    // Every backend maps Exp2 / Log2 to a native instruction; Log, Log10, Exp and Pow need ILGPU.Algorithms on CUDA.

    private static float PowP(float a, float b) => a <= 0 ? 0 : XMath.Exp2(b * XMath.Log2(a));
    private static float Ln(float x) => XMath.Log2(x) * 0.6931472f;
    private static float Lg10(float x) => XMath.Log2(x) * 0.30103f;
    private static float Exp(float x) => XMath.Exp2(x * 1.442695f);
    private static float Exp10(float x) => XMath.Exp2(x * 3.321928f);

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

    // ------------------------------------------------------------------ window switching

    /// <summary>
    /// One thread per granule: whether its signal holds an attack (both channels together, so mid / side keeps working).
    /// Granule g's MDCT window spans input samples about 576 (g - 1) - 225 .. 576 (g + 1) - 225 (subband slot t's newest
    /// sample is 32 t + 31, its 512-tap window centred 256 earlier); its three short windows cover the middle
    /// 576 g - 608 .. 576 g + 160, a start window before them reaches 576 g - 416 and a stop window after them starts at
    /// 576 g - 32. So an attack at [576 g - 416, 576 g + 160) belongs to granule g. Energies of 128-sample blocks at a hop
    /// of 64 (block m = samples [64 m, 64 m + 128), centre 64 m + 64): block m is an attack when its energy is over 8 times
    /// the mean of the four blocks of the 512 samples before it (m - 2, -4, -6, -8) and above a floor of about -70 dBFS,
    /// either in the signal or in its first difference (a high-pass that catches a click over a loud bass note). Blocks
    /// whose history starts before the input (m &lt; 8) are not judged. attack[g] = 1 / 0.
    /// </summary>
    public static void AttackKernel(Index1D index, ArrayView<float> pcm, ArrayView<int> attack, int channels, long samples, int first)
    {
        int g = first + index;
        float floor = 1.3e-5f * channels;
        int found = 0;
        // a block whose history reaches before the input is not judged: the zeros there are padding, not a quiet passage
        // (and pre-echo before sample 0 falls in the decoder delay that gapless decoding trims)
        for (long m = XMath.Max(9L * g - 7, 8L); m <= 9L * g + 1 && found == 0; m++)
        {
            BlockEnergy(pcm, channels, samples, 64 * m, out float e, out float d);
            float pe = 0, pd = 0;
            for (int j = 2; j <= 8; j += 2)
            {
                BlockEnergy(pcm, channels, samples, 64 * (m - j), out float ej, out float dj);
                pe += ej;
                pd += dj;
            }
            pe *= 0.25f;
            pd *= 0.25f;
            if ((e > floor && e > 8f * pe) || (d > floor && d > 8f * pd)) found = 1;
        }
        attack[g] = found;
    }

    /// <summary>Energy of samples [start, start + 128) over the channels (zero outside the input), and of the first difference.</summary>
    private static void BlockEnergy(ArrayView<float> pcm, int channels, long samples, long start, out float e, out float d)
    {
        e = 0;
        d = 0;
        for (long n = start; n < start + 128; n++)
        {
            if (n < 0 || n >= samples) continue;
            for (int c = 0; c < channels; c++)
            {
                float x = pcm[n * channels + c];
                float prev = n > 0 ? pcm[(n - 1) * channels + c] : 0f;
                e += x * x;
                d += (x - prev) * (x - prev);
            }
        }
    }

    /// <summary>
    /// One thread per granule: the block type of every channel (side[SBlockType]). Short (2) where an attack is, and
    /// between two attacks (a granule there could be neither a start nor a stop); start (1) before a short granule, stop
    /// (3) after one, long (0) elsewhere. The window halves of neighbours then always match (the right half of a start or
    /// a short granule is short, of a stop or a long granule long).
    /// </summary>
    public static void BlockTypeKernel(Index1D index, ArrayView<int> attack, ArrayView<int> side, int granules, int channels, int first)
    {
        int g = first + index;
        int bt = IsShort(attack, g, granules) ? 2
            : IsShort(attack, g + 1, granules) ? 1
            : IsShort(attack, g - 1, granules) ? 3 : 0;
        for (int c = 0; c < channels; c++) side[((long)g * channels + c) * SideInts + SBlockType] = bt;
    }

    private static bool IsShort(ArrayView<int> attack, int g, int granules)
    {
        if (g < 0 || g >= granules) return false;
        if (attack[g] != 0) return true;
        return g > 0 && g + 1 < granules && attack[g - 1] != 0 && attack[g + 1] != 0;
    }

    // ------------------------------------------------------------------ MDCT

    /// <summary>Subband sample i (0..35) of granule g's window: slot (g - 1) * 18 + i, odd bands' odd samples inverted.</summary>
    private static float WindowSample(ArrayView<float> sub, int c, int slots, int sb, int g, int i)
    {
        int t = (g - 1) * 18 + i;
        if (t < 0) return 0;
        float v = sub[((long)c * slots + t) * 32 + sb];
        return (sb & 1) != 0 && (i & 1) != 0 ? -v : v;   // frequency inversion (time index within the granule: i % 18)
    }

    /// <summary>
    /// One thread per (channel, granule, band): the 36 subband samples of granules g - 1 and g (zeros before the first),
    /// odd bands' odd time samples inverted, then by the granule's block type: the long (0), start (1) or stop (3) window
    /// and the 36 -> 18 MDCT into xr[unit * 576 + band * 18 + k]; or (short, 2) three 12-point MDCTs over samples
    /// 6 + 6 w .. 17 + 6 w with the short sine window, X_w[k] = sum z_w[i] cos(pi / 24 (2 i + 7)(2 k + 1)) / 3, whose
    /// frequency f = band * 6 + k goes to the decoder's order: short band, window, frequency.
    /// </summary>
    public static void MdctKernel(Index1D index, ArrayView<float> sub, ArrayView<float> consts, ArrayView<float> xr, ArrayView<int> side,
        ArrayView<int> tables, int granules, int slots, int channels, long first)
    {
        long id = first + index;
        int sb = (int)(id % 32);
        long rest = id / 32;
        int g = (int)(rest % granules);
        int c = (int)(rest / granules);
        long unit = (long)g * channels + c;
        long outAt = unit * 576;
        int bt = side[unit * SideInts + SBlockType];
        if (bt != 2)
        {
            int win = bt == 1 ? FStart : bt == 3 ? FStop : FSin;
            for (int k = 0; k < 18; k++)
            {
                float acc = 0;
                for (int i = 0; i < 36; i++) acc += WindowSample(sub, c, slots, sb, g, i) * consts[win + i] * consts[FCos + k * 36 + i];
                xr[outAt + sb * 18 + k] = acc;
            }
            return;
        }
        for (int w = 0; w < 3; w++)
        {
            for (int k = 0; k < 6; k++)
            {
                float acc = 0;
                for (int i = 0; i < 12; i++)
                    acc += WindowSample(sub, c, slots, sb, g, 6 + 6 * w + i) * consts[FShortWin + i] * consts[FShortCos + k * 12 + i];
                int f = sb * 6 + k;
                int sfb = 0;
                while (sfb < 12 && tables[IShort + sfb + 1] <= f) sfb++;
                xr[outAt + tables[IGroup + sfb * 3 + w] + f - tables[IShort + sfb]] = acc;
            }
        }
    }

    /// <summary>
    /// One thread per (channel-granule, boundary sb 0..30, butterfly i 0..7): the encoder's alias reduction (long, start
    /// and stop granules; a short granule has none).
    /// </summary>
    public static void AliasReduceKernel(Index1D index, ArrayView<float> xr, ArrayView<float> consts, ArrayView<int> side, long first)
    {
        long id = first + index;
        int i = (int)(id % 8);
        long rest = id / 8;
        int sb = (int)(rest % 31);
        long cg = rest / 31;
        if (side[cg * SideInts + SBlockType] == 2) return;
        long a = cg * 576 + sb * 18 + 17 - i, b = cg * 576 + (sb + 1) * 18 + i;
        float bu = xr[a], bd = xr[b];
        float cs = consts[FCs + i], ca = consts[FCa + i];
        xr[a] = bu * cs + bd * ca;
        xr[b] = bd * cs - bu * ca;
    }

    // ------------------------------------------------------------------ frame layout (CBR)

    /// <summary>
    /// One thread per frame (and one more for the end): the frame's byte offset f * coeff * kbps / rate (a padding byte
    /// where the running total needs it), its bit rate index and padding bit, and the bit budget of each of its
    /// granule-channels: the frame's main data shared by perceptual entropy (SplitBudget).
    /// </summary>
    public static void FrameLayoutKernel(Index1D index, ArrayView<long> frameOffsets, ArrayView<int> frameInfo, ArrayView<int> side,
        ArrayView<float> pe, long frames, long num, int sampleRate, int bitrateIndex, int sideBytes, int granulesPerFrame, int channels,
        long baseOffset)
    {
        long f = index;
        long t0 = f * num / sampleRate;
        frameOffsets[f] = baseOffset + t0;
        if (f >= frames) return;
        long size = (f + 1) * num / sampleRate - t0;
        frameInfo[f * 2] = bitrateIndex;
        frameInfo[f * 2 + 1] = (int)(size - num / sampleRate);
        SplitBudget(side, pe, f * granulesPerFrame * channels, granulesPerFrame * channels, (int)((size - 4 - sideBytes) * 8));
    }

    /// <summary>
    /// Share <paramref name="bits"/> among the <paramref name="count"/> granule-channels from unit <paramref name="u0"/> by
    /// perceptual entropy (weight PE + a quarter of their mean PE + 1, so a near-silent one keeps a little), at most 4095
    /// each: what a capped one cannot take goes to the others by their weights.
    /// </summary>
    private static void SplitBudget(ArrayView<int> side, ArrayView<float> pe, long u0, int count, int bits)
    {
        float mean = 0;
        for (int u = 0; u < count; u++) mean += pe[u0 + u];
        mean /= count;
        int left = bits;
        float weights = 0;
        for (int u = 0; u < count; u++)
        {
            weights += pe[u0 + u] + 0.25f * mean + 1f;
            side[(u0 + u) * SideInts + SBudget] = -1;   // not yet assigned
        }
        // assign in rounds: a unit whose share reaches the cap takes 4095 and leaves the round's rest to the others
        for (int round = 0; round < count && left > 0 && weights > 0; round++)
        {
            bool capped = false;
            for (int u = 0; u < count; u++)
            {
                long s = (u0 + u) * SideInts;
                if (side[s + SBudget] >= 0) continue;
                float wgt = pe[u0 + u] + 0.25f * mean + 1f;
                if (left * (wgt / weights) >= 4095f)
                {
                    side[s + SBudget] = 4095;
                    left -= 4095;
                    weights -= wgt;
                    capped = true;
                }
            }
            if (!capped) break;
        }
        int given = 0, lastOpen = -1;
        for (int u = 0; u < count; u++)
        {
            long s = (u0 + u) * SideInts;
            if (side[s + SBudget] >= 0) continue;
            int b = weights > 0 ? (int)(XMath.Max(left, 0) * ((pe[u0 + u] + 0.25f * mean + 1f) / weights)) : 0;
            side[s + SBudget] = b;
            given += b;
            lastOpen = u;
        }
        // rounding remainder to the last open unit
        if (lastOpen >= 0)
        {
            long s = (u0 + lastOpen) * SideInts;
            side[s + SBudget] = XMath.Min(4095, side[s + SBudget] + XMath.Max(left, 0) - given);
        }
    }

    // ------------------------------------------------------------------ psychoacoustic model (MDCT domain)

    /// <summary>
    /// One thread per (granule, channel): the noise each long scalefactor band may carry (xmin, in MDCT energy), from the
    /// granule's own spectrum. Band energies; tonality from the spectral flatness of a band's lines (white noise sits
    /// near -5.5 dB, a tone far below); energy spread across bands with Schroeder's spreading function in Bark; the
    /// threshold below the spread energy by ISO model 2's offsets (18 dB under tonal maskers, 6 dB under noise); never
    /// below the threshold of hearing. <paramref name="quality"/> scales the result (1 = the model's threshold).
    /// A short granule gets the same per (short band, window): each window's threshold comes from its own spectrum only,
    /// so the quiet window before an attack keeps a low threshold (the pre-echo it may carry). xmin[unit * XStride + q].
    /// <paramref name="scratch"/> holds ScratchFloats per unit (group energies, tonalities).
    /// </summary>
    public static void PsyKernel(Index1D i0, ArrayView<float> xr, ArrayView<int> tables, ArrayView<int> side, ArrayView<float> psy,
        ArrayView<float> scratch, ArrayView<float> xmin, float quality, int first)
    {
        int index = first + i0;
        long at = (long)index * 576;
        long so = (long)index * ScratchFloats;
        long xo = (long)index * XStride;
        int bt = side[(long)index * SideInts + SBlockType];
        int groups = Groups(bt);
        for (int q = 0; q < groups; q++)
        {
            int lo = GroupLo(tables, bt, q), hi = GroupLo(tables, bt, q + 1);
            float e = 0, logSum = 0;
            for (int i = lo; i < hi; i++)
            {
                float v = xr[at + i] * xr[at + i];
                e += v;
                logSum += Ln(v + 1e-30f);
            }
            int n = hi - lo;
            float sfmDb = 10f * Lg10(Exp(logSum / n) / (e / n + 1e-30f) + 1e-30f);
            scratch[so + q] = e;
            scratch[so + 39 + q] = XMath.Clamp((sfmDb + 5.5f) / -25f, 0f, 1f);
        }
        // long: 22 bands, one "window"; short: 13 bands in each of 3 windows (group = band * 3 + window)
        int bands = bt == 2 ? 13 : 22, windows = bt == 2 ? 3 : 1;
        int barkAt = bt == 2 ? PSBark : PBark, athAt = bt == 2 ? PSAth : PAth;
        for (int w = 0; w < windows; w++)
        {
            for (int b = 0; b < bands; b++)
            {
                float zb = psy[barkAt + b];
                float spread = 0, toneWeighted = 0;
                for (int j = 0; j < bands; j++)
                {
                    float ej = scratch[so + j * windows + w];
                    if (ej <= 0) continue;
                    float dz = zb - psy[barkAt + j] + 0.474f;
                    float sfDb = 15.81f + 7.5f * dz - 17.5f * XMath.Sqrt(1f + dz * dz);
                    if (sfDb < -60f) continue;
                    float wt = ej * Exp10(sfDb / 10f);
                    spread += wt;
                    toneWeighted += wt * scratch[so + 39 + j * windows + w];
                }
                float t = spread > 0 ? toneWeighted / spread : 0;
                float offsetDb = 18f * t + 6f * (1 - t);
                float thr = spread * Exp10(-offsetDb / 10f);
                xmin[xo + b * windows + w] = XMath.Max(thr, psy[athAt + b]) * quality;
            }
        }
        // pre-echo control between short windows: a window's noise spreads over all of its 384 samples, the part before
        // an attack included, so it may not exceed twice the previous window's allowed noise; backward masking covers a
        // little pre-echo, so the window's own threshold drops by at most 15 dB (an onset from digital silence does not
        // demand lossless coding)
        for (int w = 1; w < windows; w++)
            for (int b = 0; b < bands; b++)
            {
                float own = xmin[xo + b * windows + w];
                xmin[xo + b * windows + w] = XMath.Max(XMath.Min(own, 2f * xmin[xo + b * windows + w - 1]), own * (1f / 32f));
            }
        for (int q = groups; q < XStride; q++) xmin[xo + q] = 0;
    }

    // ------------------------------------------------------------------ Info / Xing frame (gapless + seek data)

    /// <summary>InfoFrameKernel's parameter view (longs).</summary>
    public const int InfoVersion = 0, InfoRate = 1, InfoBitrate = 2, InfoChannels = 3, InfoSide = 4, InfoBytes = 5, InfoFrames = 6,
        InfoTotal = 7, InfoVbr = 8, InfoDelay = 9, InfoPadding = 10, InfoParams = 11;

    /// <summary>The CRC-16 the Info frame's extension carries (polynomial 0x8005 bit-reversed, initial 0).</summary>
    private static int Crc16Step(int crc, int b, ArrayView<int> table) => ((crc >> 8) ^ table[(crc ^ b) & 0xFF]) & 0xFFFF;

    /// <summary>
    /// One thread: continue the CRC-16 of the audio frames over output[from, from + count) (state in state[0]); bounded
    /// work per launch.
    /// </summary>
    public static void MusicCrcKernel(Index1D _, ArrayView<byte> output, ArrayView<int> table, ArrayView<int> state, long from, int count)
    {
        int crc = state[0];
        for (int i = 0; i < count; i++) crc = Crc16Step(crc, output[from + i], table);
        state[0] = crc;
    }

    /// <summary>
    /// One thread: the Info frame at output[0..infoBytes): an MPEG frame header (the stream's version, rate and channel
    /// layout; stereo or mono mode, no CRC, original 0) and zeroed side info, then at +sideBytes + 4: "Info" (constant
    /// bit rate) or "Xing" (variable), flags 0x0F, the audio frame count, the byte count (Info frame included), the
    /// 100-entry seek table (entry i: 256 * end of audio frame floor(i * frames / 100) / bytes, entry 0 = 0), quality 0;
    /// then the LAME-format extension: a 9-byte encoder string, encoder delay / padding (12 + 12 bits), the music length
    /// (= the byte count), the CRC-16 of the audio frames (state[0]) and the CRC-16 of the frame's first 190 bytes.
    /// </summary>
    public static void InfoFrameKernel(Index1D _, ArrayView<byte> output, ArrayView<long> frameOffsets, ArrayView<byte> encoderName,
        ArrayView<int> table, ArrayView<int> state, ArrayView<long> p)
    {
        int version = (int)p[InfoVersion], rateIndex = (int)p[InfoRate], bitrateIndex = (int)p[InfoBitrate];
        int channels = (int)p[InfoChannels], sideBytes = (int)p[InfoSide], infoBytes = (int)p[InfoBytes];
        long frames = p[InfoFrames], totalBytes = p[InfoTotal];
        int vbr = (int)p[InfoVbr], delay = (int)p[InfoDelay], padding = (int)p[InfoPadding];
        for (int i = 0; i < infoBytes; i++) output[i] = 0;
        output[0] = 0xFF;
        output[1] = (byte)(0xE0 | (version << 3) | (1 << 1) | 1);       // layer III, no CRC
        output[2] = (byte)((bitrateIndex << 4) | (rateIndex << 2));
        output[3] = (byte)(channels == 1 ? 0xC0 : 0x00);
        int x = 4 + sideBytes;
        if (vbr != 0) { output[x] = (byte)'X'; output[x + 1] = (byte)'i'; output[x + 2] = (byte)'n'; output[x + 3] = (byte)'g'; }
        else { output[x] = (byte)'I'; output[x + 1] = (byte)'n'; output[x + 2] = (byte)'f'; output[x + 3] = (byte)'o'; }
        PutBe32(output, x + 4, 0x0F);
        PutBe32(output, x + 8, (uint)frames);
        PutBe32(output, x + 12, (uint)totalBytes);
        for (int i = 0; i < 100; i++)
        {
            long v = 0;
            if (i > 0)
            {
                long j = (long)i * frames / 100;
                v = frameOffsets[j + 1] * 256 / totalBytes;
                if (v > 255) v = 255;
            }
            output[x + 16 + i] = (byte)v;
        }
        PutBe32(output, x + 116, 0);                                       // quality
        int e = x + 120;
        for (int i = 0; i < 9; i++) output[e + i] = encoderName[i];
        output[e + 21] = (byte)(delay >> 4);
        output[e + 22] = (byte)(((delay & 15) << 4) | ((padding >> 8) & 15));
        output[e + 23] = (byte)padding;
        PutBe32(output, e + 28, (uint)totalBytes);
        int music = state[0];
        output[e + 32] = (byte)(music >> 8);
        output[e + 33] = (byte)music;
        int crc = 0;
        for (int i = 0; i < 190; i++) crc = Crc16Step(crc, output[i], table);
        output[e + 34] = (byte)(crc >> 8);
        output[e + 35] = (byte)crc;
    }

    private static void PutBe32(ArrayView<byte> o, long at, uint v)
    {
        o[at] = (byte)(v >> 24); o[at + 1] = (byte)(v >> 16); o[at + 2] = (byte)(v >> 8); o[at + 3] = (byte)v;
    }

    // ------------------------------------------------------------------ perceptual entropy

    /// <summary>
    /// One thread per (granule, channel): the perceptual entropy of the granule, sum over its groups (long bands or short
    /// band-windows) of width * log2(1 + energy / allowed noise): an estimate of the bits it needs to keep its noise
    /// under the masks.
    /// </summary>
    public static void PerceptualEntropyKernel(Index1D i0, ArrayView<float> xr, ArrayView<float> xmin, ArrayView<int> tables,
        ArrayView<int> side, ArrayView<float> pe, int first)
    {
        int index = first + i0;
        long at = (long)index * 576;
        int bt = side[(long)index * SideInts + SBlockType];
        float total = 0;
        for (int b = 0; b < Groups(bt); b++)
        {
            int lo = GroupLo(tables, bt, b), hi = GroupLo(tables, bt, b + 1);
            float e = 0;
            for (int i = lo; i < hi; i++) e += xr[at + i] * xr[at + i];
            float allowed = xmin[(long)index * XStride + b];
            if (allowed > 0 && e > allowed) total += (hi - lo) * XMath.Log2(1f + e / allowed);
        }
        pe[index] = total;
    }

    // ------------------------------------------------------------------ mid / side stereo

    /// <summary>
    /// One thread per frame (stereo): mid / side when it concentrates the energy, i.e. when the smaller of the mid and side
    /// energies is below 40 % of the smaller of left and right (identical or opposite channels give a near-empty mid or side;
    /// independent channels of equal power do not). Then, for every granule of the frame, M = (L + R) / sqrt 2 and
    /// S = (L - R) / sqrt 2 in place, and each band's allowed noise becomes the smaller of the left and right masks.
    /// ms[f] = 1 when the frame is coded mid / side. Both channels of a granule share its block type, so the coefficients
    /// and groups line up.
    /// </summary>
    public static void MidSideKernel(Index1D fIndex, ArrayView<float> xr, ArrayView<float> xmin, ArrayView<int> side, ArrayView<int> ms,
        int granulesPerFrame, int enabled, int first)
    {
        int f = first + fIndex;
        float el = 0, er = 0, em = 0, es = 0;
        for (int g = 0; g < granulesPerFrame; g++)
        {
            long l = ((long)(f * granulesPerFrame + g) * 2) * 576, r = l + 576;
            for (int i = 0; i < 576; i++)
            {
                float a = xr[l + i], b = xr[r + i];
                el += a * a; er += b * b;
                em += (a + b) * (a + b) * 0.5f;
                es += (a - b) * (a - b) * 0.5f;
            }
        }
        bool useMs = enabled != 0 && XMath.Min(em, es) < 0.4f * XMath.Min(el, er);
        ms[f] = useMs ? 1 : 0;
        if (!useMs) return;
        const float k = 0.70710678f;
        for (int g = 0; g < granulesPerFrame; g++)
        {
            long unit = (long)(f * granulesPerFrame + g) * 2;
            long l = unit * 576, r = l + 576;
            for (int i = 0; i < 576; i++)
            {
                float a = xr[l + i], b = xr[r + i];
                xr[l + i] = (a + b) * k;
                xr[r + i] = (a - b) * k;
            }
            int groups = Groups(side[unit * SideInts + SBlockType]);
            for (int bnd = 0; bnd < groups; bnd++)
            {
                float m = XMath.Min(xmin[unit * XStride + bnd], xmin[(unit + 1) * XStride + bnd]);
                xmin[unit * XStride + bnd] = m;
                xmin[(unit + 1) * XStride + bnd] = m;
            }
        }
    }

    // ------------------------------------------------------------------ quantization + bit counting

    private static int Quant(float x, float stepInv)
    {
        float a = XMath.Abs(x) * stepInv;
        float v = PowP(a, 0.75f) - 0.0946f;
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
    /// The end of region 0 of a window-switched granule (implicit in the bitstream): the first 8 long bands (start /
    /// stop) or the first 9 (short band, window) groups (short); region 1 is the rest of big_values.
    /// </summary>
    private static int SwitchedRegion0End(ArrayView<int> tables, int bt) => bt == 2 ? tables[IGroup + 9] : tables[IBounds + 8];

    /// <summary>
    /// Huffman bits of a granule with the region split and best tables; returns the bits and the chosen tables. Long
    /// granules (block type 0): <paramref name="exhaustive"/> tries every (region0_count, region1_count), otherwise a fixed
    /// split. Window-switched granules (1, 2, 3): the implicit two regions.
    /// </summary>
    private static int BigValueBits(ArrayView<int> ix, long at, ArrayView<int> tables, int bigValues, int bt, bool exhaustive,
        out int region0, out int region1, out int t0, out int t1, out int t2)
    {
        int end = bigValues * 2;
        region0 = 0; region1 = 0; t0 = 0; t1 = 0; t2 = 0;
        if (end == 0) return 0;
        if (bt != 0)
        {
            int e = XMath.Min(SwitchedRegion0End(tables, bt), end);
            t0 = BestTable(ix, at, 0, e, MaxAbs(ix, at, 0, e), tables, out int y0);
            t1 = BestTable(ix, at, e, end, MaxAbs(ix, at, e, end), tables, out int y1);
            return y0 + y1;
        }
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

    /// <summary>
    /// Bits of the scalefactors (part2) and scalefac_compress for the current maxima; -1 when they do not fit. Long:
    /// MPEG-1 slen1 for bands 0..10, slen2 for 11..20; MPEG-2 / 2.5 (no intensity stereo) partitions of 6, 5, 5, 5 bands.
    /// Short (non-mixed): MPEG-1 slen1 for short bands 0..5, slen2 for 6..11 (18 values each); MPEG-2 / 2.5 partitions of
    /// 9, 9, 9, 9 values (short bands 0-2, 3-5, 6-8, 9-11), the same scalefac_compress formula (ISO/IEC 13818-3
    /// nr_of_sfb_block, scalefac_compress below 400).
    /// </summary>
    private static int ScfBits(ArrayView<int> side, long s, int lsf, out int compress)
    {
        int bt = side[s + SBlockType];
        bool sh = bt == 2;
        int m1 = 0, m2 = 0, m3 = 0, m4 = 0;
        for (int b = 0; b < ScfCount(bt); b++)
        {
            int v = side[s + SScf + b];
            int part = sh ? b / 9 : b < 6 ? 0 : b < 11 ? 1 : b < 16 ? 2 : 3;
            if (part == 0) m1 = XMath.Max(m1, v);
            else if (part == 1) m2 = XMath.Max(m2, v);
            else if (part == 2) m3 = XMath.Max(m3, v);
            else m4 = XMath.Max(m4, v);
        }
        compress = 0;
        if (lsf == 0)
        {
            // MPEG-1: the scalefac_compress with the fewest bits whose slen1 / slen2 hold the maxima
            int n1 = sh ? 18 : 11, n2 = sh ? 18 : 10;
            int best = -1;
            for (int c = 0; c < 16; c++)
            {
                int s1 = Slen1(c), s2 = Slen2(c);
                if ((1 << s1) <= XMath.Max(m1, m2) || (1 << s2) <= XMath.Max(m3, m4)) continue;
                int bits = n1 * s1 + n2 * s2;
                if (best < 0 || bits < best) { best = bits; compress = c; }
            }
            return best;
        }
        // MPEG-2 / 2.5, no intensity stereo: slen1 / slen2 0..4, slen3 / slen4 0..3
        int l1 = BitsFor(m1), l2 = BitsFor(m2), l3 = BitsFor(m3), l4 = BitsFor(m4);
        if (l1 > 4 || l2 > 4 || l3 > 3 || l4 > 3) return -1;
        compress = ((l1 * 5 + l2) << 4) + (l3 << 2) + l4;
        return sh ? 9 * (l1 + l2 + l3 + l4) : 6 * l1 + 5 * l2 + 5 * l3 + 5 * l4;
    }

    private static int BitsFor(int v)
    {
        int n = 0;
        while ((1 << n) <= v) n++;
        return n;
    }

    private static int Slen1(int c) => c < 4 ? 0 : c == 4 ? 3 : c < 8 ? 1 : c < 11 ? 2 : c < 14 ? 3 : 4;
    private static int Slen2(int c) => c < 4 ? c : c == 4 ? 0 : c < 8 ? c - 4 : c < 11 ? c - 7 : c < 14 ? c - 10 : c - 12;

    /// <summary>
    /// The factor the quantizer applies to group q: 2^(multiplier * scalefactor) with multiplier 0.5 or 1 (scalefac_scale),
    /// times 2^(2 * subblock_gain) of the group's window in a short granule (the decoder's step is 2^((global_gain - 210
    /// - 8 subblock_gain) / 4) 2^(-multiplier * scalefactor)).
    /// </summary>
    private static float Amp(ArrayView<int> side, long s, int bt, int q)
    {
        float e = q < ScfCount(bt) ? (side[s + SScfScale] != 0 ? 1f : 0.5f) * side[s + SScf + q] : 0f;
        if (bt == 2) e += 2f * side[s + SSubGain + q % 3];
        return XMath.Exp2(e);
    }

    /// <summary>Quantize the granule at gain g with the current scalefactors into ix.</summary>
    private static void QuantizeAll(ArrayView<float> xr, ArrayView<int> ix, long at, ArrayView<int> side, long s, ArrayView<int> tables, int g)
    {
        int bt = side[s + SBlockType];
        float stepInv = XMath.Exp2(-(g - 210) / 4f);
        for (int b = 0; b < Groups(bt); b++)
        {
            float k = stepInv * Amp(side, s, bt, b);
            int lo = GroupLo(tables, bt, b), hi = GroupLo(tables, bt, b + 1);
            for (int i = lo; i < hi; i++) ix[at + i] = Quant(xr[at + i], k) * (xr[at + i] < 0 ? -1 : 1);
        }
    }

    /// <summary>Inner loop: the smallest gain in [lo, 255] whose Huffman bits fit; -1 when none does.</summary>
    private static int InnerLoop(ArrayView<float> xr, ArrayView<int> ix, long at, ArrayView<int> side, long s, ArrayView<int> tables,
        int lo, int huffBudget)
    {
        int bt = side[s + SBlockType];
        int hi = 255, found = -1;
        while (lo <= hi)
        {
            int g = (lo + hi) >> 1;
            QuantizeAll(xr, ix, at, side, s, tables, g);
            Regions(ix, at, tables, out int bv, out _, out _, out int c1Bits);
            int bits = BigValueBits(ix, at, tables, bv, bt, false, out _, out _, out _, out _, out _) + c1Bits;
            if (bits <= huffBudget && MaxAbs(ix, at, 0, 576) <= 8206) { found = g; hi = g - 1; }
            else lo = g + 1;
        }
        return found;
    }

    /// <summary>The smallest global gain that keeps every value within the escape range (8206) before amplification; -1 for
    /// a silent granule.</summary>
    private static int MinGain(ArrayView<float> xr, long at)
    {
        float maxAbs = 0;
        for (int i = 0; i < 576; i++) maxAbs = XMath.Max(maxAbs, XMath.Abs(xr[at + i]));
        if (maxAbs < 1e-20f) return -1;
        // the smallest gain that keeps every value within the escape range (8206), before amplification
        float need = 4f * XMath.Log2(maxAbs / PowP(8206.5f + 0.0946f, 4f / 3f));
        return XMath.Clamp((int)XMath.Ceiling(210 + need), 0, 255);
    }

    /// <summary>
    /// The subblock gains of a short granule (zero otherwise): the three windows of an attack differ by tens of dB, more
    /// than scalefactors reach (22.5 dB in short bands 0..5, 10.5 dB in 6..11 at scalefac_scale 0). Each window needs a
    /// step whose square is about its finest group's allowed noise per line (xmin / width); a window needing a finer step
    /// than the coarsest window by a power ratio R gets subblock_gain floor(log2(R) / 4) (amplitude 2^(2 sbg) = R^(1/2)),
    /// 0..7, and the scalefactors handle the rest.
    /// </summary>
    private static void SetSubblockGains(ArrayView<int> side, long s, ArrayView<int> tables, ArrayView<float> xmin, long xo)
    {
        int bt = side[s + SBlockType];
        for (int w = 0; w < 3; w++) side[s + SSubGain + w] = 0;
        if (bt != 2) return;
        float n0 = float.MaxValue, n1 = float.MaxValue, n2 = float.MaxValue;
        for (int q = 0; q < 39; q++)
        {
            float perLine = xmin[xo + q] / (GroupLo(tables, bt, q + 1) - GroupLo(tables, bt, q));
            int w = q % 3;
            if (w == 0) n0 = XMath.Min(n0, perLine);
            else if (w == 1) n1 = XMath.Min(n1, perLine);
            else n2 = XMath.Min(n2, perLine);
        }
        float top = XMath.Max(n0, XMath.Max(n1, n2));
        if (!(top > 0)) return;
        side[s + SSubGain + 0] = n0 > 0 ? XMath.Clamp((int)XMath.Floor(XMath.Log2(top / n0) / 4f), 0, 7) : 7;
        side[s + SSubGain + 1] = n1 > 0 ? XMath.Clamp((int)XMath.Floor(XMath.Log2(top / n1) / 4f), 0, 7) : 7;
        side[s + SSubGain + 2] = n2 > 0 ? XMath.Clamp((int)XMath.Floor(XMath.Log2(top / n2) / 4f), 0, 7) : 7;
    }

    /// <summary>A silent (or budget-less) granule: no values, no scalefactors.</summary>
    private static void ZeroGranule(ArrayView<int> ix, long at, ArrayView<int> side, long s)
    {
        for (int i = 0; i < 576; i++) ix[at + i] = 0;
        side[s + SPart23] = 0; side[s + SBigValues] = 0; side[s + SGlobalGain] = 0; side[s + SScfCompress] = 0;
        side[s + STable0] = 0; side[s + STable1] = 0; side[s + STable2] = 0; side[s + SRegion0] = 0; side[s + SRegion1] = 0;
        side[s + SCount1Table] = 0;
        for (int w = 0; w < 3; w++) side[s + SSubGain + w] = 0;
    }

    /// <summary>
    /// One thread per (granule, channel): ISO 11172-3 Annex C's two loops. The inner loop finds the smallest global gain
    /// whose bits (scalefactors + Huffman) fit side[SBudget]; the outer loop measures the noise in each group (long band,
    /// or short band of one window) against its allowed noise xmin and amplifies (scalefactor + 1) every group over it,
    /// switching to scalefac_scale 1 when a scalefactor would overflow, until no group is over or nothing can be
    /// amplified. The iteration with the fewest groups over (then the least noise above the masks) is kept; ix and the
    /// side info written.
    /// </summary>
    public static void QuantizeKernel(Index1D i0, ArrayView<float> xr, ArrayView<int> ix, ArrayView<int> side, ArrayView<int> tables,
        ArrayView<float> xmin, ArrayView<int> work, int lsf, int requantOnly, int first)
    {
        int index = first + i0;
        long at = (long)index * 576;
        long s = (long)index * SideInts;
        if (requantOnly != 0 && side[s + SRequant] == 0) return;
        long xo = (long)index * XStride;
        long wo = (long)index * WorkInts;   // [0..39) group over its mask, [39..75) the best iteration's scalefactors
        int bt = side[s + SBlockType];
        int groups = Groups(bt), nScf = ScfCount(bt);
        int budget = XMath.Min(side[s + SBudget], 4095);
        side[s + SScfScale] = 0;
        for (int b = 0; b < 39; b++) side[s + SScf + b] = 0;
        SetSubblockGains(side, s, tables, xmin, xo);
        int gMin = MinGain(xr, at);
        if (gMin < 0 || budget <= 0)
        {
            ZeroGranule(ix, at, side, s);
            return;
        }

        // best iteration so far: (groups over, noise above the masks in dB), its gain, scalefac_scale and scalefactors
        int bestOver = int.MaxValue, bestGain = 255, bestScale = 0;
        float bestOverDb = float.MaxValue;
        for (int b = 0; b < nScf; b++) work[wo + 39 + b] = 0;
        for (int iter = 0; iter < 64; iter++)
        {
            int part2 = ScfBits(side, s, lsf, out _);
            if (part2 < 0) break;
            int g = InnerLoop(xr, ix, at, side, s, tables, gMin, budget - part2);
            if (g < 0) break;
            QuantizeAll(xr, ix, at, side, s, tables, g);
            // noise per group against its mask
            float step = XMath.Exp2((g - 210) / 4f);
            int over = 0;
            float overDb = 0;
            for (int b = 0; b < groups; b++)
            {
                float inv = step / Amp(side, s, bt, b);
                int lo = GroupLo(tables, bt, b), hi = GroupLo(tables, bt, b + 1);
                float noise = 0;
                for (int i = lo; i < hi; i++)
                {
                    float q = PowP(XMath.Abs(ix[at + i]), 4f / 3f) * inv;
                    float d = XMath.Abs(xr[at + i]) - q;
                    noise += d * d;
                }
                float allowed = xmin[xo + b];
                bool isOver = noise > allowed;
                if (isOver)
                {
                    over++;
                    overDb += 10f * Lg10(noise / allowed);
                }
                work[wo + b] = isOver ? 1 : 0;
            }
            if (over < bestOver || (over == bestOver && overDb < bestOverDb))
            {
                bestOver = over; bestOverDb = overDb; bestGain = g; bestScale = side[s + SScfScale];
                for (int b = 0; b < nScf; b++) work[wo + 39 + b] = side[s + SScf + b];
            }
            if (over == 0) break;
            // amplify every group over its mask
            bool overflow = false, amplified = false;
            for (int b = 0; b < nScf; b++)
            {
                if (work[wo + b] == 0) continue;
                side[s + SScf + b]++;
                amplified = true;
                if (side[s + SScf + b] > ScfMax(bt, b)) overflow = true;
            }
            if (!amplified) break;
            if (overflow)
            {
                if (side[s + SScfScale] != 0) break;
                // scalefac_scale 1 doubles the step of every scalefactor: halve them (rounding up keeps the amplification)
                side[s + SScfScale] = 1;
                for (int b = 0; b < nScf; b++) side[s + SScf + b] = (side[s + SScf + b] + 1) >> 1;
            }
        }
        // the best iteration, then the best region split and tables
        side[s + SScfScale] = bestScale;
        for (int b = 0; b < nScf; b++) side[s + SScf + b] = work[wo + 39 + b];
        int p2 = ScfBits(side, s, lsf, out int compress);
        if (p2 < 0)
        {
            // cannot happen for a kept iteration; fall back to no scalefactors
            side[s + SScfScale] = 0;
            for (int b = 0; b < nScf; b++) side[s + SScf + b] = 0;
            p2 = 0; compress = 0;
            bestGain = XMath.Max(InnerLoop(xr, ix, at, side, s, tables, gMin, budget), 0);
        }
        int gain = bestGain;
        while (true)
        {
            QuantizeAll(xr, ix, at, side, s, tables, gain);
            Regions(ix, at, tables, out int bv, out _, out int c1t, out int c1Bits);
            int bits = BigValueBits(ix, at, tables, bv, bt, true, out int r0, out int r1, out int t0, out int t1, out int t2) + c1Bits;
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

    // ------------------------------------------------------------------ variable bit rate

    /// <summary>Noise of a group at gain g with amplification amp; float.MaxValue when a value leaves the escape range.</summary>
    private static float BandNoise(ArrayView<float> xr, long at, int lo, int hi, float stepInv, float amp)
    {
        float k = stepInv * amp;
        float back = 1f / k;   // dequantization: |ix|^(4/3) / (stepInv * amp)
        float noise = 0;
        for (int i = lo; i < hi; i++)
        {
            int q = Quant(xr[at + i], k);
            if (q > 8206) return float.MaxValue;
            float d = XMath.Abs(xr[at + i]) - PowP(q, 4f / 3f) * back;
            noise += d * d;
        }
        return noise;
    }

    /// <summary>
    /// At gain g: the smallest scalefactor of each group whose noise is within its mask (written to side), and whether
    /// every group found one.
    /// </summary>
    private static bool FitScalefactors(ArrayView<float> xr, long at, ArrayView<int> side, long s, ArrayView<int> tables,
        ArrayView<float> xmin, long xo, int g, bool commitBest)
    {
        int bt = side[s + SBlockType];
        float stepInv = XMath.Exp2(-(g - 210) / 4f);
        bool all = true;
        for (int b = 0; b < Groups(bt); b++)
        {
            int lo = GroupLo(tables, bt, b), hi = GroupLo(tables, bt, b + 1);
            int max = ScfMax(bt, b);
            float sub = bt == 2 ? 2f * side[s + SSubGain + b % 3] : 0f;   // the window's subblock gain (log2 amplitude)
            int chosen = -1;
            for (int sf = 0; sf <= max; sf++)
            {
                if (BandNoise(xr, at, lo, hi, stepInv, XMath.Exp2(0.5f * sf + sub)) <= xmin[xo + b]) { chosen = sf; break; }
            }
            if (chosen < 0) { all = false; chosen = max; }
            if (b < ScfCount(bt) && commitBest) side[s + SScf + b] = chosen;
        }
        return all;
    }

    /// <summary>
    /// One thread per (granule, channel), variable bit rate: the coarsest global gain at which every group's noise can be
    /// kept within its mask by a scalefactor (binary search; scalefac_scale 0), those scalefactors, then the best region
    /// split and tables. side[SRequant] = 1 when the granule needs more than 4095 bits (the layout re-quantizes it to a
    /// budget).
    /// </summary>
    public static void VbrQuantizeKernel(Index1D i0, ArrayView<float> xr, ArrayView<int> ix, ArrayView<int> side, ArrayView<int> tables,
        ArrayView<float> xmin, int lsf, int first)
    {
        int index = first + i0;
        long at = (long)index * 576;
        long s = (long)index * SideInts;
        long xo = (long)index * XStride;
        int bt = side[s + SBlockType];
        side[s + SScfScale] = 0;
        side[s + SRequant] = 0;
        for (int b = 0; b < 39; b++) side[s + SScf + b] = 0;
        SetSubblockGains(side, s, tables, xmin, xo);
        int gMin = MinGain(xr, at);
        if (gMin < 0)
        {
            ZeroGranule(ix, at, side, s);
            return;
        }
        // the largest gain that still satisfies every mask (coarser gains need fewer bits)
        int lo = gMin, hi = 255, best = gMin;
        while (lo <= hi)
        {
            int g = (lo + hi) >> 1;
            if (FitScalefactors(xr, at, side, s, tables, xmin, xo, g, false)) { best = g; lo = g + 1; }
            else hi = g - 1;
        }
        FitScalefactors(xr, at, side, s, tables, xmin, xo, best, true);
        QuantizeAll(xr, ix, at, side, s, tables, best);
        int p2 = ScfBits(side, s, lsf, out int compress);
        Regions(ix, at, tables, out int bv, out _, out int c1t, out int c1Bits);
        int bits = BigValueBits(ix, at, tables, bv, bt, true, out int r0, out int r1, out int t0, out int t1, out int t2) + c1Bits;
        side[s + SPart23] = bits + p2;
        side[s + SBigValues] = bv;
        side[s + SGlobalGain] = best;
        side[s + SScfCompress] = compress;
        side[s + STable0] = t0; side[s + STable1] = t1; side[s + STable2] = t2;
        side[s + SRegion0] = r0; side[s + SRegion1] = r1;
        side[s + SCount1Table] = c1t;
        if (bits + p2 > 4095 || p2 < 0) side[s + SRequant] = 1;
    }

    /// <summary>
    /// One thread: each frame's bit rate (the smallest whose frame holds its side info and granules), offsets after the
    /// Info frame, and the frames that do not fit even at the highest rate: their granules get a budget (the highest
    /// rate's main data split evenly) and SRequant = 1 for the re-quantization pass. meta[VAudioBytes] = the audio bytes.
    /// </summary>
    public static void VbrLayoutKernel(Index1D _, ArrayView<int> side, ArrayView<long> frameOffsets, ArrayView<int> frameInfo,
        ArrayView<long> meta, ArrayView<int> bitrates, int frames, int sampleRate, int coeff, int sideBytes, int granulesPerFrame,
        int channels, long baseOffset)
    {
        long offset = baseOffset;
        int units = granulesPerFrame * channels;
        for (int f = 0; f < frames; f++)
        {
            frameOffsets[f] = offset;
            long bits = 0;
            bool requant = false;
            for (int u = 0; u < units; u++)
            {
                long su = ((long)f * units + u) * SideInts;
                bits += side[su + SPart23];
                requant |= side[su + SRequant] != 0;
            }
            long needBytes = 4 + sideBytes + (bits + 7) / 8;
            int idx = 14;
            if (!requant)
            {
                idx = -1;
                for (int k = 1; k <= 14; k++)
                {
                    if ((long)coeff * bitrates[k] / sampleRate >= needBytes) { idx = k; break; }
                }
                if (idx < 0) { idx = 14; requant = true; }
            }
            long size = (long)coeff * bitrates[idx] / sampleRate;
            frameInfo[f * 2] = idx;
            frameInfo[f * 2 + 1] = 0;
            if (requant)
            {
                int per = (int)XMath.Min(4095L, (size - 4 - sideBytes) * 8 / units);
                for (int u = 0; u < units; u++)
                {
                    long su = ((long)f * units + u) * SideInts;
                    side[su + SRequant] = 1;
                    side[su + SBudget] = per;
                }
            }
            offset += size;
        }
        frameOffsets[frames] = offset;
        meta[VAudioBytes] = offset - baseOffset;
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
        // part2 (scfsi 0): the scalefactors of long bands 0..20, or of short groups 0..35 (short band * 3 + window)
        int bt = side[s + SBlockType];
        bool sh = bt == 2;
        int compress = side[s + SScfCompress];
        for (int b = 0; b < ScfCount(bt); b++)
        {
            int n;
            if (lsf == 0) n = b < (sh ? 18 : 11) ? Slen1(compress) : Slen2(compress);
            else
            {
                int sl1 = (compress >> 4) / 5, sl2 = (compress >> 4) % 5, sl3 = (compress >> 2) & 3, sl4 = compress & 3;
                int part = sh ? b / 9 : b < 6 ? 0 : b < 11 ? 1 : b < 16 ? 2 : 3;
                n = part == 0 ? sl1 : part == 1 ? sl2 : part == 2 ? sl3 : sl4;
            }
            PutBits(slots, slot, ref bitPos, (uint)side[s + SScf + b], n);
        }
        int bv = side[s + SBigValues];
        int end = bv * 2;
        int b0, b1;
        if (bt != 0)
        {
            b0 = XMath.Min(SwitchedRegion0End(tables, bt), end);
            b1 = end;
        }
        else
        {
            b0 = XMath.Min(tables[IBounds + XMath.Min(side[s + SRegion0] + 1, 22)], end);
            b1 = XMath.Min(tables[IBounds + XMath.Min(side[s + SRegion0] + side[s + SRegion1] + 2, 22)], end);
        }
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
        ArrayView<int> frameInfo, ArrayView<byte> output, ArrayView<int> ms, int channels, int lsf, int version, int rateIndex, int first)
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
        PutBits(output, baseByte, ref bitPos, channels == 1 ? 3u : 1u, 2);   // mono / joint stereo
        PutBits(output, baseByte, ref bitPos, channels == 2 && ms[f] != 0 ? 2u : 0u, 2);   // mode extension: mid / side
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
                int bt = side[s + SBlockType];
                if (bt != 0)
                {
                    // window_switching_flag, block_type, mixed_block_flag 0, two table_selects, three subblock_gains
                    PutBits(output, baseByte, ref bitPos, 1, 1);
                    PutBits(output, baseByte, ref bitPos, (uint)bt, 2);
                    PutBits(output, baseByte, ref bitPos, 0, 1);
                    PutBits(output, baseByte, ref bitPos, (uint)side[s + STable0], 5);
                    PutBits(output, baseByte, ref bitPos, (uint)side[s + STable1], 5);
                    for (int w = 0; w < 3; w++) PutBits(output, baseByte, ref bitPos, (uint)side[s + SSubGain + w], 3);
                }
                else
                {
                    PutBits(output, baseByte, ref bitPos, 0, 1);    // window_switching_flag
                    PutBits(output, baseByte, ref bitPos, (uint)side[s + STable0], 5);
                    PutBits(output, baseByte, ref bitPos, (uint)side[s + STable1], 5);
                    PutBits(output, baseByte, ref bitPos, (uint)side[s + STable2], 5);
                    PutBits(output, baseByte, ref bitPos, (uint)side[s + SRegion0], 4);
                    PutBits(output, baseByte, ref bitPos, (uint)side[s + SRegion1], 3);
                }
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
