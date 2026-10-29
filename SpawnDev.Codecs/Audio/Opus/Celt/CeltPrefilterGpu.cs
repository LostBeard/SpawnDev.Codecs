// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// The CELT encoder's pitch pre-filter on the accelerator, ported from libopus 1.5.2 (fixed point; celt/celt_encoder.c
// run_prefilter, celt/pitch.c remove_doubling and compute_pitch_gain) with 1.5.2's C arithmetic (CeltOps). The pitch search it
// runs (pitch_downsample, pitch_search) is the one the decoder's concealment uses (CeltPlcGpu). No tonality analysis: libopus'
// fixed-point build runs it only at complexity 10, so analysis->valid is 0 and max_pitch_ratio never applies.

using ILGPU;
using static SpawnDev.Codecs.Audio.Opus.Celt.CeltOps;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>CELT encoder pitch pre-filter (libopus 1.5.2 run_prefilter, GPU-callable).</summary>
public static class CeltPrefilterGpu
{
    /// <summary>COMBFILTER_MAXPERIOD / COMBFILTER_MINPERIOD.</summary>
    public const int MaxPeriod = 1024, MinPeriod = 15;

    private const int Overlap = 120, ShortMdctSize = 120;   // the static 48 kHz mode

    /// <summary>The scratch ints <see cref="RunPrefilter"/> needs at its scratchAt, for CC channels of N samples.</summary>
    public static int ScratchInts(int CC, int N) => CC * (N + MaxPeriod) + 2 * ((MaxPeriod + N) >> 1) + 40
        + (N >> 2) + ((N + MaxPeriod - 3 * MinPeriod) >> 2) + ((MaxPeriod - 3 * MinPeriod) >> 1) + (MaxPeriod >> 1) + 1;

    // libopus second_check[16] (pitch.c).
    private static int SecondCheck(int k) => k == 2 || k == 4 || k == 8 || k == 10 || k == 14 ? 3 : k == 6 || k == 12 ? 5 : k < 2 ? 0 : 2;

    private static int InnerProd(ArrayView<int> v, int xAt, int yAt, int N)
    {
        int xy = 0;
        for (int i = 0; i < N; i++) xy = MAC16_16(xy, v[xAt + i], v[yAt + i]);
        return xy;
    }

    /// <summary>libopus compute_pitch_gain (fixed point): xy / sqrt(xx * yy) in Q15.</summary>
    internal static int ComputePitchGain(int xy, int xx, int yy)
    {
        if (xy == 0 || xx == 0 || yy == 0) return 0;
        int sx = celt_ilog2(xx) - 14;
        int sy = celt_ilog2(yy) - 14;
        int shift = sx + sy;
        int x2y2 = MULT16_16(VSHR32(xx, sx), VSHR32(yy, sy)) >> 14;
        if ((shift & 1) != 0)
        {
            if (x2y2 < 32768)
            {
                x2y2 <<= 1;
                shift--;
            }
            else
            {
                x2y2 >>= 1;
                shift++;
            }
        }
        int den = celt_rsqrt_norm(x2y2);
        int g = MULT16_32_Q15(den, xy);
        g = VSHR32(g, (shift >> 1) - 1);
        return S16(IMIN(g, Q15ONE));
    }

    /// <summary>
    /// libopus remove_doubling: checks the period <paramref name="T0_"/> (in / out, the full-rate lag) against its
    /// sub-multiples T0 / k on the decimated-by-2 signal at <paramref name="xAt"/> (maxperiod / 2 samples of history, then
    /// N / 2), favouring the previous frame's period; returns the pitch gain (Q15). <paramref name="yyAt"/> holds
    /// maxperiod / 2 + 1 ints of scratch.
    /// </summary>
    public static int RemoveDoubling(ArrayView<int> v, int xAt, int maxperiod, int minperiod, int N, ref int T0_, int prev_period,
        int prev_gain, int yyAt)
    {
        int minperiod0 = minperiod;
        maxperiod /= 2;
        minperiod /= 2;
        T0_ /= 2;
        prev_period /= 2;
        N /= 2;
        int x = xAt + maxperiod;
        if (T0_ >= maxperiod) T0_ = maxperiod - 1;
        int T0 = T0_, T = T0_;
        // dual_inner_prod(x, x, x - T0)
        int xx = 0, xy = 0;
        for (int i = 0; i < N; i++)
        {
            xx = MAC16_16(xx, v[x + i], v[x + i]);
            xy = MAC16_16(xy, v[x + i], v[x + i - T0]);
        }
        v[yyAt] = xx;
        int yy = xx;
        for (int i = 1; i <= maxperiod; i++)
        {
            yy = yy + MULT16_16(v[x - i], v[x - i]) - MULT16_16(v[x + N - i], v[x + N - i]);
            v[yyAt + i] = IMAX(0, yy);
        }
        yy = v[yyAt + T0];
        int best_xy = xy, best_yy = yy;
        int g0 = ComputePitchGain(xy, xx, yy), g = g0;
        // Look for any pitch at T / k
        for (int k = 2; k <= 15; k++)
        {
            int T1 = (2 * T0 + k) / (2 * k);
            if (T1 < minperiod) break;
            // Look for another strong correlation at T1b
            int T1b;
            if (k == 2)
            {
                if (T1 + T0 > maxperiod) T1b = T0;
                else T1b = T0 + T1;
            }
            else T1b = (2 * SecondCheck(k) * T0 + k) / (2 * k);
            int xy1 = 0, xy2 = 0;
            for (int i = 0; i < N; i++)
            {
                xy1 = MAC16_16(xy1, v[x + i], v[x + i - T1]);
                xy2 = MAC16_16(xy2, v[x + i], v[x + i - T1b]);
            }
            xy = HALF32(xy1 + xy2);
            yy = HALF32(v[yyAt + T1] + v[yyAt + T1b]);
            int g1 = ComputePitchGain(xy, xx, yy);
            int cont;
            if (ABS32(T1 - prev_period) <= 1) cont = prev_gain;
            else if (ABS32(T1 - prev_period) <= 2 && 5 * k * k < T0) cont = HALF16(prev_gain);
            else cont = 0;
            int thresh = S16(IMAX(9830, MULT16_16_Q15(22938, g0) - cont));                // .3, .7
            // Bias against very high pitch (very short period) to avoid false-positives due to short-term correlation
            if (T1 < 3 * minperiod) thresh = S16(IMAX(13107, MULT16_16_Q15(27853, g0) - cont));   // .4, .85
            else if (T1 < 2 * minperiod) thresh = S16(IMAX(16384, MULT16_16_Q15(29491, g0) - cont));   // .5, .9
            if (g1 > thresh)
            {
                best_xy = xy;
                best_yy = yy;
                T = T1;
                g = g1;
            }
        }
        best_xy = IMAX(0, best_xy);
        int pg;
        if (best_yy <= best_xy) pg = Q15ONE;
        else pg = S16(frac_div32(best_xy, best_yy + 1) >> 16);
        int xc0 = InnerProd(v, x, x - (T - 1), N);
        int xc1 = InnerProd(v, x, x - T, N);
        int xc2 = InnerProd(v, x, x - (T + 1), N);
        int offset;
        if ((xc2 - xc0) > MULT16_32_Q15(22938, xc1 - xc0)) offset = 1;
        else if ((xc0 - xc2) > MULT16_32_Q15(22938, xc1 - xc2)) offset = -1;
        else offset = 0;
        if (pg > g) pg = g;
        T0_ = 2 * T + offset;
        if (T0_ < minperiod0) T0_ = minperiod0;
        return pg;
    }

    /// <summary>
    /// libopus run_prefilter: finds this frame's pitch period and gain and runs the comb pre-filter (negated gains) in place on
    /// the CC channels' pre-emphasised input at <paramref name="inAt"/> (channel c at inAt + c * (N + overlap): overlap samples
    /// of the previous frame, then N), cross-fading from the previous frame's filter. <paramref name="st"/> holds the encoder's
    /// prefilter_mem (CC * COMBFILTER_MAXPERIOD ints at <paramref name="prefilterMemAt"/>) and in_mem (CC * overlap at
    /// <paramref name="inMemAt"/>), both updated. <paramref name="prefilterPeriod"/> is st->prefilter_period (raised to at least
    /// COMBFILTER_MINPERIOD), <paramref name="prefilterGain"/> / <paramref name="prefilterTapsetOld"/> the previous frame's.
    /// Returns pf_on and sets the pitch, the gain (Q15) and its 3-bit index. <paramref name="w"/> at <paramref name="scratchAt"/>
    /// holds <see cref="ScratchInts"/> ints.
    /// </summary>
    public static int RunPrefilter(CeltModeGpu m, ArrayView<int> v, int inAt, ArrayView<int> st, int prefilterMemAt, int inMemAt,
        int CC, int N, int prefilter_tapset, ref int pitch, ref int gain, ref int qgain, int enabled, int nbAvailableBytes,
        ref int prefilterPeriod, int prefilterGain, int prefilterTapsetOld, int lossRate, ArrayView<int> w, int scratchAt)
    {
        int P = N + MaxPeriod;
        int pre = scratchAt;
        int pitchBuf = pre + CC * P, pbLen = (MaxPeriod + N) >> 1;
        int acAt = pitchBuf + pbLen, lpc4At = acAt + 8, lpcMemAt = lpc4At + 8, xxAt = lpcMemAt + 24;
        int maxPitch = MaxPeriod - 3 * MinPeriod;
        int xlp4At = xxAt + pbLen, ylp4At = xlp4At + (N >> 2), xcorrAt = ylp4At + ((N + maxPitch) >> 2);
        int yyAt = xcorrAt + (maxPitch >> 1);
        for (int c = 0; c < CC; c++)
        {
            for (int i = 0; i < MaxPeriod; i++) w[pre + c * P + i] = st[prefilterMemAt + c * MaxPeriod + i];
            for (int i = 0; i < N; i++) w[pre + c * P + MaxPeriod + i] = v[inAt + c * (N + Overlap) + Overlap + i];
        }
        int pitch_index, gain1;
        if (enabled != 0)
        {
            CeltPlcGpu.PitchDownsample(m, w, pre, pre + P, MaxPeriod + N, CC, w, pitchBuf, acAt, lpc4At, lpcMemAt, xxAt);
            // Don't search for the fir last 1.5 octave of the range because there's too many false-positives due to
            // short-term correlation
            pitch_index = CeltPlcGpu.PitchSearch(w, pitchBuf + (MaxPeriod >> 1), pitchBuf, N, maxPitch, xlp4At, ylp4At, xcorrAt);
            pitch_index = MaxPeriod - pitch_index;
            gain1 = RemoveDoubling(w, pitchBuf, MaxPeriod, MinPeriod, N, ref pitch_index, prefilterPeriod, prefilterGain, yyAt);
            if (pitch_index > MaxPeriod - 2) pitch_index = MaxPeriod - 2;
            gain1 = S16(MULT16_16_Q15(22938, gain1));   // QCONST16(.7f, 15)
            if (lossRate > 2) gain1 = S16(HALF32(gain1));
            if (lossRate > 4) gain1 = S16(HALF32(gain1));
            if (lossRate > 8) gain1 = 0;
        }
        else
        {
            gain1 = 0;
            pitch_index = MinPeriod;
        }
        // Gain threshold for enabling the prefilter / postfilter
        int pf_threshold = 6554;   // QCONST16(.2f, 15)
        // Adjusting the threshold based on rate and continuity
        if (ABS32(pitch_index - prefilterPeriod) * 10 > pitch_index) pf_threshold += 6554;
        if (nbAvailableBytes < 25) pf_threshold += 3277;
        if (nbAvailableBytes < 35) pf_threshold += 3277;
        if (prefilterGain > 13107) pf_threshold -= 3277;   // .4
        if (prefilterGain > 18022) pf_threshold -= 3277;   // .55
        // Hard threshold at 0.2
        pf_threshold = S16(IMAX(pf_threshold, 6554));
        int pf_on, qg;
        if (gain1 < pf_threshold)
        {
            gain1 = 0;
            pf_on = 0;
            qg = 0;
        }
        else
        {
            // This block is not gated by a total bits check only because of the nbAvailableBytes check above.
            if (ABS32(gain1 - prefilterGain) < 3277) gain1 = prefilterGain;
            qg = ((gain1 + 1536) >> 10) / 3 - 1;
            qg = IMAX(0, IMIN(7, qg));
            gain1 = S16(3072 * (qg + 1));   // QCONST16(0.09375f, 15)
            pf_on = 1;
        }
        for (int c = 0; c < CC; c++)
        {
            int offset = ShortMdctSize - Overlap;
            prefilterPeriod = IMAX(prefilterPeriod, MinPeriod);
            int inc = inAt + c * (N + Overlap);
            for (int i = 0; i < Overlap; i++) v[inc + i] = st[inMemAt + c * Overlap + i];
            if (offset != 0)
                CeltCombFilterGpu.CombFilter(m, v, inc + Overlap, w, pre + c * P + MaxPeriod, prefilterPeriod, prefilterPeriod, offset,
                    -prefilterGain, -prefilterGain, prefilterTapsetOld, prefilterTapsetOld, 0);
            CeltCombFilterGpu.CombFilter(m, v, inc + Overlap + offset, w, pre + c * P + MaxPeriod + offset, prefilterPeriod, pitch_index,
                N - offset, -prefilterGain, -gain1, prefilterTapsetOld, prefilter_tapset, Overlap);
            for (int i = 0; i < Overlap; i++) st[inMemAt + c * Overlap + i] = v[inc + N + i];
            if (N > MaxPeriod)
            {
                for (int i = 0; i < MaxPeriod; i++) st[prefilterMemAt + c * MaxPeriod + i] = w[pre + c * P + N + i];
            }
            else
            {
                for (int i = 0; i < MaxPeriod - N; i++) st[prefilterMemAt + c * MaxPeriod + i] = st[prefilterMemAt + c * MaxPeriod + N + i];
                for (int i = 0; i < N; i++) st[prefilterMemAt + c * MaxPeriod + MaxPeriod - N + i] = w[pre + c * P + MaxPeriod + i];
            }
        }
        gain = gain1;
        pitch = pitch_index;
        qgain = qg;
        return pf_on;
    }
}
