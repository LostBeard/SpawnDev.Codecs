// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// CELT packet-loss concealment on the accelerator, ported from libopus 1.5.2 (fixed-point build, no deep PLC): celt_decode_lost
// and prefilter_and_fold (celt/celt_decoder.c), the pitch search it runs on the decoded history (celt/pitch.c pitch_downsample,
// pitch_search, find_best_pitch, celt_pitch_xcorr) and the LPC helpers (celt/celt_lpc.c _celt_autocorr, _celt_lpc, celt_fir,
// celt_iir). libopus' opus_val16 arrays are ints here holding 16-bit values: every store into one truncates to 16 bits, and
// MULT16_16 / MAC16_16 / MULT16_32_Q15 truncate their 16-bit operand, as in C. libopus' unrolled xcorr / FIR / IIR forms compute
// the same integer sums as the plain loops used here. 1.5.2's MULT32_32_Q31 / Q16 are its exact 64-bit forms (OPUS_FAST_INT64).

using ILGPU;
using static SpawnDev.Codecs.Audio.Opus.Celt.CeltFixed;
using static SpawnDev.Codecs.Audio.Opus.Celt.CeltDecoderGpu;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>CELT packet-loss concealment (libopus 1.5.2 celt_decode_lost) and its helpers (GPU-callable).</summary>
public static class CeltPlcGpu
{
    private const int LpcOrder = 24, MaxPeriod = 1024, PlcPitchLagMax = 720, PlcPitchLagMin = 100, SigShift = 12, SigSat = 300000000;
    private const int Nb = CeltModeTables.NbEBands;

    /// <summary>Concealment scratch in the decoder's work view (ints), used only while concealing.</summary>
    public const int LpAt = 0, Xlp4At = 1024, Ylp4At = 1360, XcorrAt = 1848, ExcAt = 2160, FirTmpAt = 3208, XxAt = 4232, AcAt = 5256,
        LpcMemAt = 5288, Lpc4At = 5312, ScratchEnd = 5328;

    // libopus fixed-point macros with their C truncations (fixed_generic.h, the OPUS_FAST_INT64 forms).
    private static int S16(int x) => (short)x;
    private static int Mult16_16(int a, int b) => (short)a * (short)b;
    private static int Mac16_16(int c, int a, int b) => c + (short)a * (short)b;
    private static int Mult16_16Q15(int a, int b) => ((short)a * (short)b) >> 15;
    private static int Mult16_32Q15(int a, int b) => (int)(((long)(short)a * b) >> 15);
    private static int Mult32_32Q31(int a, int b) => (int)(((long)a * b) >> 31);
    private static int Mult32_32Q16(int a, int b) => (int)(((long)a * b) >> 16);
    private static int Pshr32(int a, int shift) => (a + ((1 << shift) >> 1)) >> shift;
    private static int Vshr32(int a, int shift) => shift > 0 ? a >> shift : a << -shift;
    private static int Round16(int x, int a) => (short)Pshr32(x, a);
    private static int Sround16(int x, int a)
    {
        int v = Pshr32(x, a);
        return (short)(v > 32767 ? 32767 : v < -32767 ? -32767 : v);   // SATURATE(PSHR32(x, a), 32767)
    }
    private static int Saturate(int v, int a) => v > a ? a : v < -a ? -a : v;
    private static int Abs32(int x) => x < 0 ? -x : x;
    private static int Ilog2(int x) => EC_ILOG((uint)x) - 1;

    /// <summary>libopus frac_div32 (mathops.c): a / b in Q31, for |a| &lt;= |b|.</summary>
    public static int FracDiv32(int a, int b)
    {
        int shift = Ilog2(b) - 29;
        a = Vshr32(a, shift);
        b = Vshr32(b, shift);
        // 16-bit reciprocal
        int rcp = Round16(celt_rcp(Round16(b, 16)), 3);
        int result = Mult16_32Q15(rcp, a);
        int rem = Pshr32(a, 2) - Mult32_32Q31(result, b);
        result = result + (Mult16_32Q15(rcp, rem) << 2);
        if (result >= 536870912) return 2147483647;       // 2^29 -> 2^31 - 1
        else if (result <= -536870912) return -2147483647;
        else return result << 2;
    }

    /// <summary>libopus _celt_autocorr: <paramref name="lag"/> + 1 autocorrelations of the n 16-bit samples at <paramref name="xAt"/>
    /// (windowed over <paramref name="overlap"/> samples at both ends when overlap &gt; 0) to <paramref name="acAt"/>; returns the
    /// scale shift. <paramref name="xxAt"/> holds n ints of scratch.</summary>
    public static int Autocorr(CeltModeGpu m, ArrayView<int> v, int xAt, int acAt, int overlap, int lag, int n, int xxAt)
    {
        int fastN = n - lag;
        int xp = xAt;
        if (overlap != 0)
        {
            for (int i = 0; i < n; i++) v[xxAt + i] = v[xAt + i];
            for (int i = 0; i < overlap; i++)
            {
                v[xxAt + i] = S16(Mult16_16Q15(v[xAt + i], m.WindowOf(i)));
                v[xxAt + n - i - 1] = S16(Mult16_16Q15(v[xAt + n - i - 1], m.WindowOf(i)));
            }
            xp = xxAt;
        }
        int shift;
        {
            int ac0 = 1 + (n << 7);
            if ((n & 1) != 0) ac0 += Mult16_16(v[xp], v[xp]) >> 9;
            for (int i = n & 1; i < n; i += 2)
            {
                ac0 += Mult16_16(v[xp + i], v[xp + i]) >> 9;
                ac0 += Mult16_16(v[xp + i + 1], v[xp + i + 1]) >> 9;
            }
            shift = Ilog2(ac0) - 30 + 10;
            shift = shift / 2;
            if (shift > 0)
            {
                for (int i = 0; i < n; i++) v[xxAt + i] = S16(Pshr32(v[xp + i], shift));
                xp = xxAt;
            }
            else shift = 0;
        }
        for (int k = 0; k <= lag; k++)
        {
            int d = 0;
            for (int i = k; i < n; i++) d = Mac16_16(d, v[xp + i], v[xp + i - k]);
            v[acAt + k] = d;
        }
        shift = 2 * shift;
        if (shift <= 0) v[acAt] += 1 << -shift;
        if (v[acAt] < 268435456)
        {
            int shift2 = 29 - EC_ILOG((uint)v[acAt]);
            for (int i = 0; i <= lag; i++) v[acAt + i] = v[acAt + i] << shift2;
            shift -= shift2;
        }
        else if (v[acAt] >= 536870912)
        {
            int shift2 = 1;
            if (v[acAt] >= 1073741824) shift2++;
            for (int i = 0; i <= lag; i++) v[acAt + i] = v[acAt + i] >> shift2;
            shift += shift2;
        }
        return shift;
    }

    /// <summary>libopus _celt_lpc: <paramref name="p"/> LPC coefficients (Q12, 16-bit) from the autocorrelations at
    /// <paramref name="acAt"/>, to <paramref name="lpcAt"/> in <paramref name="lpcView"/>; <paramref name="tmpAt"/> holds p ints.</summary>
    public static void Lpc(ArrayView<int> lpcView, int lpcAt, ArrayView<int> v, int acAt, int p, int tmpAt)
    {
        int error = v[acAt];
        for (int i = 0; i < p; i++) v[tmpAt + i] = 0;
        if (v[acAt] != 0)
        {
            for (int i = 0; i < p; i++)
            {
                // Sum up this iteration's reflection coefficient
                int rr = 0;
                for (int j = 0; j < i; j++) rr += Mult32_32Q31(v[tmpAt + j], v[acAt + i - j]);
                rr += v[acAt + i + 1] >> 6;
                int r = -FracDiv32(rr << 6, error);
                // Update LPC coefficients and total error
                v[tmpAt + i] = r >> 6;
                for (int j = 0; j < (i + 1) >> 1; j++)
                {
                    int tmp1 = v[tmpAt + j];
                    int tmp2 = v[tmpAt + i - 1 - j];
                    v[tmpAt + j] = tmp1 + Mult32_32Q31(r, tmp2);
                    v[tmpAt + i - 1 - j] = tmp2 + Mult32_32Q31(r, tmp1);
                }
                error = error - Mult32_32Q31(Mult32_32Q31(r, r), error);
                // Bail out once we get 30 dB gain
                if (error <= v[acAt] >> 10) break;
            }
        }
        // Convert the int32 lpcs to int16 with no wrap-around (silk_LPC_fit / silk_bwexpander_32).
        int iter, idx = 0;
        for (iter = 0; iter < 10; iter++)
        {
            int maxabs = 0;
            for (int i = 0; i < p; i++)
            {
                int absval = Abs32(v[tmpAt + i]);
                if (absval > maxabs)
                {
                    maxabs = absval;
                    idx = i;
                }
            }
            maxabs = Pshr32(maxabs, 13);   // Q25 -> Q12
            if (maxabs > 32767)
            {
                maxabs = IMIN(maxabs, 163838);
                int chirp_Q16 = 65470 - ((maxabs - 32767) << 14) / ((maxabs * (idx + 1)) >> 2);   // QCONST32(0.999, 16)
                int chirp_minus_one_Q16 = chirp_Q16 - 65536;
                // Apply bandwidth expansion.
                for (int i = 0; i < p - 1; i++)
                {
                    v[tmpAt + i] = Mult32_32Q16(chirp_Q16, v[tmpAt + i]);
                    chirp_Q16 += Pshr32(chirp_Q16 * chirp_minus_one_Q16, 16);
                }
                v[tmpAt + p - 1] = Mult32_32Q16(chirp_Q16, v[tmpAt + p - 1]);
            }
            else break;
        }
        if (iter == 10)
        {
            // Still not 16-bit after 10 iterations: the A(z) = 1 filter.
            for (int i = 0; i < p; i++) lpcView[lpcAt + i] = 0;
            lpcView[lpcAt] = 4096;   // Q12
        }
        else
        {
            for (int i = 0; i < p; i++) lpcView[lpcAt + i] = S16(Pshr32(v[tmpAt + i], 13));   // Q25 -> Q12
        }
    }

    /// <summary>libopus celt_fir: y = the FIR of the 16-bit x (reads <paramref name="ord"/> samples before
    /// <paramref name="xAt"/>), not in place.</summary>
    public static void Fir(ArrayView<int> v, int xAt, ArrayView<int> num, int numAt, int yAt, int N, int ord)
    {
        for (int i = 0; i < N; i++)
        {
            int sum = v[xAt + i] << SigShift;
            for (int j = 0; j < ord; j++) sum = Mac16_16(sum, num[numAt + j], v[xAt + i - 1 - j]);
            v[yAt + i] = Sround16(sum, SigShift);
        }
    }

    /// <summary>libopus celt_iir in place on the 32-bit signal at <paramref name="xAt"/>, with the 16-bit filter memory at
    /// <paramref name="memAt"/> (the outputs before the signal, newest first).</summary>
    public static void Iir(ArrayView<int> x, int xAt, ArrayView<int> den, int denAt, int N, int ord, ArrayView<int> v, int memAt)
    {
        for (int i = 0; i < N; i++)
        {
            int sum = x[xAt + i];
            for (int j = 0; j < ord; j++) sum -= Mult16_16(den[denAt + j], v[memAt + j]);
            for (int j = ord - 1; j >= 1; j--) v[memAt + j] = v[memAt + j - 1];
            v[memAt] = Sround16(sum, SigShift);
            x[xAt + i] = sum;
        }
    }

    private static int MaxAbs16(ArrayView<int> v, int at, int len)
    {
        int maxval = 0, minval = 0;
        for (int i = 0; i < len; i++)
        {
            maxval = IMAX(maxval, v[at + i]);
            minval = IMIN(minval, v[at + i]);
        }
        return IMAX(maxval, -minval);
    }

    // libopus find_best_pitch (fixed point): the two best lags of xcorr (len samples of y per lag).
    private static void FindBestPitch(ArrayView<int> v, int xcorrAt, int yAt, int len, int max_pitch, ref int best0, ref int best1,
        int yshift, int maxcorr)
    {
        int Syy = 1;
        int best_num0 = -1, best_num1 = -1, best_den0 = 0, best_den1 = 0;
        int xshift = Ilog2(maxcorr) - 14;
        best0 = 0;
        best1 = 1;
        for (int j = 0; j < len; j++) Syy = Syy + (Mult16_16(v[yAt + j], v[yAt + j]) >> yshift);
        for (int i = 0; i < max_pitch; i++)
        {
            if (v[xcorrAt + i] > 0)
            {
                int xcorr16 = S16(Vshr32(v[xcorrAt + i], xshift));
                int num = S16(Mult16_16Q15(xcorr16, xcorr16));
                if (Mult16_32Q15(num, best_den1) > Mult16_32Q15(best_num1, Syy))
                {
                    if (Mult16_32Q15(num, best_den0) > Mult16_32Q15(best_num0, Syy))
                    {
                        best_num1 = best_num0;
                        best_den1 = best_den0;
                        best1 = best0;
                        best_num0 = num;
                        best_den0 = Syy;
                        best0 = i;
                    }
                    else
                    {
                        best_num1 = num;
                        best_den1 = Syy;
                        best1 = i;
                    }
                }
            }
            Syy += (Mult16_16(v[yAt + i + len], v[yAt + i + len]) >> yshift) - (Mult16_16(v[yAt + i], v[yAt + i]) >> yshift);
            Syy = IMAX(1, Syy);
        }
    }

    /// <summary>
    /// libopus celt_plc_pitch_search: pitch_downsample of both channels' decode_mem (DECODE_BUFFER_SIZE samples) into the lp
    /// buffer, then pitch_search over it; returns the pitch lag, PLC_PITCH_LAG_MIN..PLC_PITCH_LAG_MAX.
    /// </summary>
    public static int PitchSearchPlc(CeltModeGpu m, ArrayView<int> st, int C, ArrayView<int> w)
    {
        int len = DecodeBufferSize;
        // pitch_downsample
        int d0 = DecodeMemAt, d1 = DecodeMemAt + ChannelStride;
        int maxabs = 0;
        for (int c = 0; c < C; c++)
        {
            int maxval = 0, minval = 0;
            for (int i = 0; i < len; i++)
            {
                int x = st[DecodeMemAt + c * ChannelStride + i];
                maxval = IMAX(maxval, x);
                minval = IMIN(minval, x);
            }
            maxabs = IMAX(maxabs, IMAX(maxval, -minval));
        }
        if (maxabs < 1) maxabs = 1;
        int shift = Ilog2(maxabs) - 10;
        if (shift < 0) shift = 0;
        if (C == 2) shift++;
        for (int i = 1; i < len >> 1; i++)
            w[LpAt + i] = S16((st[d0 + 2 * i - 1] >> (shift + 2)) + (st[d0 + 2 * i + 1] >> (shift + 2)) + (st[d0 + 2 * i] >> (shift + 1)));
        w[LpAt] = S16((st[d0 + 1] >> (shift + 2)) + (st[d0] >> (shift + 1)));
        if (C == 2)
        {
            for (int i = 1; i < len >> 1; i++)
                w[LpAt + i] = S16(w[LpAt + i] + (st[d1 + 2 * i - 1] >> (shift + 2)) + (st[d1 + 2 * i + 1] >> (shift + 2)) + (st[d1 + 2 * i] >> (shift + 1)));
            w[LpAt] = S16(w[LpAt] + (st[d1 + 1] >> (shift + 2)) + (st[d1] >> (shift + 1)));
        }
        Autocorr(m, w, LpAt, AcAt, 0, 4, len >> 1, XxAt);
        // Noise floor -40 dB
        w[AcAt] += w[AcAt] >> 13;
        // Lag windowing
        for (int i = 1; i <= 4; i++) w[AcAt + i] -= Mult16_32Q15(2 * i * i, w[AcAt + i]);
        Lpc(w, Lpc4At, w, AcAt, 4, LpcMemAt);
        int tmp = 32767;
        for (int i = 0; i < 4; i++)
        {
            tmp = S16(Mult16_16Q15(29491, tmp));                     // QCONST16(.9f, 15)
            w[Lpc4At + i] = S16(Mult16_16Q15(w[Lpc4At + i], tmp));
        }
        // Add a zero: lpc2 (5 taps) after the 4 lpc values
        const int c1 = 26214;                                         // QCONST16(.8f, 15)
        int l0 = w[Lpc4At], l1 = w[Lpc4At + 1], l2 = w[Lpc4At + 2], l3 = w[Lpc4At + 3];
        int num0 = S16(l0 + 3277);                                    // QCONST16(.8f, SIG_SHIFT)
        int num1 = S16(l1 + Mult16_16Q15(c1, l0));
        int num2 = S16(l2 + Mult16_16Q15(c1, l1));
        int num3 = S16(l3 + Mult16_16Q15(c1, l2));
        int num4 = S16(Mult16_16Q15(c1, l3));
        // celt_fir5, in place
        int mem0 = 0, mem1 = 0, mem2 = 0, mem3 = 0, mem4 = 0;
        for (int i = 0; i < len >> 1; i++)
        {
            int x = w[LpAt + i];
            int sum = x << SigShift;
            sum = Mac16_16(sum, num0, mem0);
            sum = Mac16_16(sum, num1, mem1);
            sum = Mac16_16(sum, num2, mem2);
            sum = Mac16_16(sum, num3, mem3);
            sum = Mac16_16(sum, num4, mem4);
            mem4 = mem3;
            mem3 = mem2;
            mem2 = mem1;
            mem1 = mem0;
            mem0 = x;
            w[LpAt + i] = Round16(sum, SigShift);
        }
        // pitch_search(lp + (PLC_PITCH_LAG_MAX >> 1), lp, DECODE_BUFFER_SIZE - PLC_PITCH_LAG_MAX, PLC_PITCH_LAG_MAX - PLC_PITCH_LAG_MIN)
        int xlp = LpAt + (PlcPitchLagMax >> 1), y = LpAt;
        int slen = DecodeBufferSize - PlcPitchLagMax, max_pitch = PlcPitchLagMax - PlcPitchLagMin;
        int lag = slen + max_pitch;
        for (int j = 0; j < slen >> 2; j++) w[Xlp4At + j] = w[xlp + 2 * j];
        for (int j = 0; j < lag >> 2; j++) w[Ylp4At + j] = w[y + 2 * j];
        int xmax = MaxAbs16(w, Xlp4At, slen >> 2), ymax = MaxAbs16(w, Ylp4At, lag >> 2);
        int sh = Ilog2(IMAX(1, IMAX(xmax, ymax))) - 11;
        if (sh > 0)
        {
            for (int j = 0; j < slen >> 2; j++) w[Xlp4At + j] = S16(w[Xlp4At + j] >> sh);
            for (int j = 0; j < lag >> 2; j++) w[Ylp4At + j] = S16(w[Ylp4At + j] >> sh);
            // Use double the shift for a MAC
            sh *= 2;
        }
        else sh = 0;
        // Coarse search with 4x decimation
        int maxcorr = 1;
        for (int i = 0; i < max_pitch >> 2; i++)
        {
            int sum = 0;
            for (int j = 0; j < slen >> 2; j++) sum = Mac16_16(sum, w[Xlp4At + j], w[Ylp4At + i + j]);
            w[XcorrAt + i] = sum;
            maxcorr = IMAX(maxcorr, sum);
        }
        int best0 = 0, best1 = 0;
        FindBestPitch(w, XcorrAt, Ylp4At, slen >> 2, max_pitch >> 2, ref best0, ref best1, 0, maxcorr);
        // Finer search with 2x decimation
        maxcorr = 1;
        for (int i = 0; i < max_pitch >> 1; i++)
        {
            w[XcorrAt + i] = 0;
            if (Abs32(i - 2 * best0) > 2 && Abs32(i - 2 * best1) > 2) continue;
            int sum = 0;
            for (int j = 0; j < slen >> 1; j++) sum += Mult16_16(w[xlp + j], w[y + i + j]) >> sh;
            w[XcorrAt + i] = IMAX(-1, sum);
            maxcorr = IMAX(maxcorr, sum);
        }
        FindBestPitch(w, XcorrAt, y, slen >> 1, max_pitch >> 1, ref best0, ref best1, sh + 1, maxcorr);
        // Refine by pseudo-interpolation
        int offset;
        if (best0 > 0 && best0 < (max_pitch >> 1) - 1)
        {
            int a = w[XcorrAt + best0 - 1], b = w[XcorrAt + best0], c = w[XcorrAt + best0 + 1];
            if ((c - a) > Mult16_32Q15(22938, b - a)) offset = 1;          // QCONST16(.7f, 15)
            else if ((a - c) > Mult16_32Q15(22938, b - c)) offset = -1;
            else offset = 0;
        }
        else offset = 0;
        int pitch_index = 2 * best0 - offset;
        return PlcPitchLagMax - pitch_index;
    }

    /// <summary>
    /// libopus prefilter_and_fold: the pre-filter on the concealed frame's MDCT overlap (the post-filter is re-applied after
    /// the next frame's MDCT), then TDAC folding so it blends with that MDCT. <paramref name="etmpAt"/> holds overlap ints in w.
    /// </summary>
    public static void PrefilterAndFold(CeltModeGpu m, ArrayView<int> st, int CC, int N, ArrayView<int> w, int etmpAt)
    {
        int s0 = ScalarsAt;
        for (int c = 0; c < CC; c++)
        {
            int x = DecodeMemAt + c * ChannelStride + DecodeBufferSize - N;
            CeltCombFilterGpu.CombFilter(m, w, etmpAt, st, x, st[s0 + SPostfilterPeriodOld], st[s0 + SPostfilterPeriod], Overlap,
                -st[s0 + SPostfilterGainOld], -st[s0 + SPostfilterGain], st[s0 + SPostfilterTapsetOld], st[s0 + SPostfilterTapset], 0);
            // Simulate TDAC on the concealed audio so that it blends with the MDCT of the next frame.
            for (int i = 0; i < Overlap / 2; i++)
                st[x + i] = Mult16_32Q15(m.WindowOf(i), w[etmpAt + Overlap - 1 - i]) + Mult16_32Q15(m.WindowOf(Overlap - i - 1), w[etmpAt + i]);
        }
    }

    /// <summary>
    /// libopus 1.5.2 celt_decode_lost (no deep PLC): conceal one frame of N 48 kHz samples in each of the CC channels' decode_mem.
    /// Noise-based (after 40+ ms of loss, in a hybrid frame, or before two good packets) or pitch-based (the history's pitch
    /// period extrapolated through its LPC excitation, decaying). The caller de-emphasises out_syn afterwards.
    /// </summary>
    public static void DecodeLost(CeltModeGpu m, ArrayView<int> st, ArrayView<int> w, int xAt, int CC, int LM, int start, int end,
        int downsample, int etmpAt)
    {
        int N = ShortMdctSize << LM;
        int s0 = ScalarsAt;
        int loss_duration = st[s0 + SLossDuration];
        int noise_based = loss_duration >= 40 || start != 0 || st[s0 + SSkipPlc] != 0 ? 1 : 0;
        if (noise_based != 0)
        {
            // Noise-based PLC / CNG
            int effEnd = IMAX(start, IMIN(end, Nb));
            for (int c = 0; c < CC; c++)
            {
                int d = DecodeMemAt + c * ChannelStride;
                for (int i = 0; i < DecodeBufferSize - N + Overlap; i++) st[d + i] = st[d + N + i];
            }
            if (st[s0 + SPrefilterAndFold] != 0) PrefilterAndFold(m, st, CC, N, w, etmpAt);
            // Energy decay
            int decay = loss_duration == 0 ? 1536 : 512;   // QCONST16(1.5f / .5f, DB_SHIFT)
            for (int c = 0; c < CC; c++)
                for (int i = start; i < end; i++)
                    st[OldBandEAt + c * Nb + i] = S16(IMAX(st[BackgroundLogEAt + c * Nb + i], st[OldBandEAt + c * Nb + i] - decay));
            uint seed = (uint)st[s0 + SRng];
            for (int c = 0; c < CC; c++)
            {
                for (int i = start; i < effEnd; i++)
                {
                    int boffs = N * c + (m.EBandsOf(i) << LM);
                    int blen = (m.EBandsOf(i + 1) - m.EBandsOf(i)) << LM;
                    for (int j = 0; j < blen; j++)
                    {
                        seed = CeltBandsGpu.CeltLcgRand(seed);
                        w[xAt + boffs + j] = S16((int)seed >> 20);
                    }
                    CeltPvqGpu.RenormaliseVector(w, xAt + boffs, blen, 32767);
                }
            }
            st[s0 + SRng] = (int)seed;
            Synthesis(m, st, w, xAt, CC, CC, 0, LM, start, effEnd, downsample, 0);
            st[s0 + SPrefilterAndFold] = 0;
            // Skip regular PLC until we get two consecutive packets.
            st[s0 + SSkipPlc] = 1;
        }
        else
        {
            // Pitch-based PLC
            int fade = 32767;
            int pitch_index;
            if (loss_duration == 0)
            {
                pitch_index = PitchSearchPlc(m, st, CC, w);
                st[s0 + SLastPitchIndex] = pitch_index;
            }
            else
            {
                pitch_index = st[s0 + SLastPitchIndex];
                fade = 26214;   // QCONST16(.8f, 15)
            }
            // The excitation for 2 pitch periods, to look for a decaying signal; no more than MAX_PERIOD.
            int exc_length = IMIN(2 * pitch_index, MaxPeriod);
            int exc = ExcAt + LpcOrder;
            for (int c = 0; c < CC; c++)
            {
                int buf = DecodeMemAt + c * ChannelStride;
                int lpcAt = LpcAt + c * LpcOrder;
                for (int i = 0; i < MaxPeriod + LpcOrder; i++)
                    w[exc + i - LpcOrder] = Sround16(st[buf + DecodeBufferSize - MaxPeriod - LpcOrder + i], SigShift);
                if (loss_duration == 0)
                {
                    // LPC coefficients of the last MAX_PERIOD samples before the first loss (the excitation-filter domain).
                    Autocorr(m, w, exc, AcAt, Overlap, LpcOrder, MaxPeriod, XxAt);
                    // Add a noise floor of -40 dB.
                    w[AcAt] += w[AcAt] >> 13;
                    // Lag windowing to stabilize the Levinson-Durbin recursion.
                    for (int i = 1; i <= LpcOrder; i++) w[AcAt + i] -= Mult16_32Q15(2 * i * i, w[AcAt + i]);
                    Lpc(st, lpcAt, w, AcAt, LpcOrder, LpcMemAt);
                    // Bandwidth expansion until no overflow can happen in the IIR filter: 32768 * sum(abs(filter)) < 2^31.
                    while (true)
                    {
                        int tmp = 32767;
                        int sum = 4096;   // QCONST16(1., SIG_SHIFT)
                        for (int i = 0; i < LpcOrder; i++) sum += IMAX(st[lpcAt + i], -st[lpcAt + i]);
                        if (sum < 65535) break;
                        for (int i = 0; i < LpcOrder; i++)
                        {
                            tmp = S16(Mult16_16Q15(32440, tmp));   // QCONST16(.99f, 15)
                            st[lpcAt + i] = S16(Mult16_16Q15(st[lpcAt + i], tmp));
                        }
                    }
                }
                // The excitation for exc_length samples before the loss (celt_fir cannot filter in place: a copy).
                Fir(w, exc + MaxPeriod - exc_length, st, lpcAt, FirTmpAt, exc_length, LpcOrder);
                for (int i = 0; i < exc_length; i++) w[exc + MaxPeriod - exc_length + i] = w[FirTmpAt + i];
                // Is the waveform decaying, and how fast? (no energy added when concealing a decaying segment)
                int decay;
                {
                    int E1 = 1, E2 = 1;
                    int shift = IMAX(0, 2 * celt_zlog2(MaxAbs16(w, exc + MaxPeriod - exc_length, exc_length)) - 20);
                    int decay_length = exc_length >> 1;
                    for (int i = 0; i < decay_length; i++)
                    {
                        int e = w[exc + MaxPeriod - decay_length + i];
                        E1 += Mult16_16(e, e) >> shift;
                        e = w[exc + MaxPeriod - 2 * decay_length + i];
                        E2 += Mult16_16(e, e) >> shift;
                    }
                    E1 = IMIN(E1, E2);
                    decay = S16(celt_sqrt(FracDiv32(E1 >> 1, E2)));
                }
                // Move the decoder memory one frame to the left; the overlap past the end of the buffer is not used.
                for (int i = 0; i < DecodeBufferSize - N; i++) st[buf + i] = st[buf + N + i];
                // Extrapolate from the end of the excitation with a period of pitch_index, scaling each period by decay.
                int extrapolation_offset = MaxPeriod - pitch_index;
                // Enough samples to cover a complete MDCT window (overlap / 2 samples on both sides).
                int extrapolation_len = N + Overlap;
                // Fading too if this is not the first loss.
                int attenuation = S16(Mult16_16Q15(fade, decay));
                int S1 = 0;
                for (int i = 0, j = 0; i < extrapolation_len; i++, j++)
                {
                    if (j >= pitch_index)
                    {
                        j -= pitch_index;
                        attenuation = S16(Mult16_16Q15(attenuation, decay));
                    }
                    st[buf + DecodeBufferSize - N + i] = Mult16_16Q15(attenuation, w[exc + extrapolation_offset + j]) << SigShift;
                    // The energy of the previously decoded signal whose excitation we are copying.
                    int tmp = Sround16(st[buf + DecodeBufferSize - MaxPeriod - N + extrapolation_offset + j], SigShift);
                    S1 += Mult16_16(tmp, tmp) >> 10;
                }
                {
                    // The last decoded samples (before the overlap region) as synthesis filter memory, for a continuous signal.
                    for (int i = 0; i < LpcOrder; i++)
                        w[LpcMemAt + i] = Sround16(st[buf + DecodeBufferSize - N - 1 - i], SigShift);
                    // The synthesis filter turns the excitation back into the signal.
                    Iir(st, buf + DecodeBufferSize - N, st, lpcAt, extrapolation_len, LpcOrder, w, LpcMemAt);
                    for (int i = 0; i < extrapolation_len; i++)
                        st[buf + DecodeBufferSize - N + i] = Saturate(st[buf + DecodeBufferSize - N + i], SigSat);
                }
                // Attenuate if the synthesis energy is higher than expected (the signal changing within the window).
                {
                    int S2 = 0;
                    for (int i = 0; i < extrapolation_len; i++)
                    {
                        int tmp = Sround16(st[buf + DecodeBufferSize - N + i], SigShift);
                        S2 += Mult16_16(tmp, tmp) >> 10;
                    }
                    // An "explosion" in the synthesis.
                    if (!(S1 > (S2 >> 2)))
                    {
                        for (int i = 0; i < extrapolation_len; i++) st[buf + DecodeBufferSize - N + i] = 0;
                    }
                    else if (S1 < S2)
                    {
                        int ratio = S16(celt_sqrt(FracDiv32((S1 >> 1) + 1, S2 + 1)));
                        for (int i = 0; i < Overlap; i++)
                        {
                            int tmp_g = S16(32767 - Mult16_16Q15(m.WindowOf(i), 32767 - ratio));
                            st[buf + DecodeBufferSize - N + i] = Mult16_32Q15(tmp_g, st[buf + DecodeBufferSize - N + i]);
                        }
                        for (int i = Overlap; i < extrapolation_len; i++)
                            st[buf + DecodeBufferSize - N + i] = Mult16_32Q15(ratio, st[buf + DecodeBufferSize - N + i]);
                    }
                }
            }
            st[s0 + SPrefilterAndFold] = 1;
        }
        // Saturate to something large to avoid wrap-around.
        st[s0 + SLossDuration] = IMIN(10000, loss_duration + (1 << LM));
    }
}
