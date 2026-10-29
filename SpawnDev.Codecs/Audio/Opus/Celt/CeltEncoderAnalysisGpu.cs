// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// The CELT encoder's front end on the accelerator, ported from libopus 1.5.2 (fixed point; celt/celt_encoder.c
// celt_preemphasis, transient_analysis, patch_transient_decision, compute_mdcts; celt/bands.c compute_band_energies,
// normalise_bands; celt/quant_bands.c amp2Log2) with 1.5.2's C arithmetic (CeltOps): pre-emphasis of the input, the
// time-domain transient detector, the forward MDCTs, the band energies (linear and log2) and the normalised spectrum.

using ILGPU;
using static SpawnDev.Codecs.Audio.Opus.Celt.CeltOps;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>CELT encoder analysis stages (libopus 1.5.2, GPU-callable).</summary>
public static class CeltEncoderAnalysisGpu
{
    private const int Preemph0 = 27853;   // mode->preemph[0]; preemph[1] is 0 in the static mode

    /// <summary>
    /// libopus celt_preemphasis (no clipping in the fixed-point build): N samples of one channel, every
    /// <paramref name="CC"/>-th int16 from <paramref name="pcmAt"/>, to the Q12 signal at <paramref name="inAt"/>; zero-stuffed by
    /// <paramref name="upsample"/> (16 / 24 kHz input). The filter memory is at <paramref name="memAt"/>.
    /// </summary>
    public static void Preemphasis(ArrayView<short> pcm, int pcmAt, int CC, ArrayView<int> inp, int inAt, int N, int upsample,
        ArrayView<int> mem, int memAt)
    {
        int m = mem[memAt];
        if (upsample == 1)
        {
            for (int i = 0; i < N; i++)
            {
                int x = pcm[pcmAt + CC * i];
                inp[inAt + i] = SHL32(x, SIG_SHIFT) - m;
                m = MULT16_16(Preemph0, x) >> (15 - SIG_SHIFT);
            }
            mem[memAt] = m;
            return;
        }
        int Nu = N / upsample;
        for (int i = 0; i < N; i++) inp[inAt + i] = 0;
        for (int i = 0; i < Nu; i++) inp[inAt + i * upsample] = pcm[pcmAt + CC * i];
        for (int i = 0; i < N; i++)
        {
            int x = S16(inp[inAt + i]);
            inp[inAt + i] = SHL32(x, SIG_SHIFT) - m;
            m = MULT16_16(Preemph0, x) >> (15 - SIG_SHIFT);
        }
        mem[memAt] = m;
    }

    /// <summary>
    /// libopus transient_analysis: is the frame transient? The C channels' <paramref name="len"/> pre-emphasised samples are at
    /// <paramref name="inAt"/> + c * len. <paramref name="tmpAt"/> holds len ints of scratch. Returns is_transient and sets the
    /// tf estimate (Q14), the channel with the strongest transient and the weak-transient flag.
    /// </summary>
    public static int TransientAnalysis(CeltModeGpu mode, ArrayView<int> v, int inAt, int len, int C, ref int tf_estimate, ref int tf_chan,
        int allow_weak_transients, ref int weak_transient, int tmpAt)
    {
        int mask_metric = 0;
        // Forward masking: 6.7 dB/ms (3.3 dB/ms for weak transients at low bitrates).
        int forward_shift = allow_weak_transients != 0 ? 5 : 4;
        weak_transient = 0;
        int len2 = len / 2;
        for (int c = 0; c < C; c++)
        {
            int mem0 = 0, mem1 = 0;
            // High-pass filter: (1 - 2*z^-1 + z^-2) / (1 - z^-1 + .5*z^-2)
            for (int i = 0; i < len; i++)
            {
                int x = v[inAt + i + c * len] >> SIG_SHIFT;
                int y = mem0 + x;
                mem0 = mem1 + y - SHL32(x, 1);
                mem1 = x - (y >> 1);
                v[tmpAt + i] = SROUND16(y, 2);
            }
            // First few samples are bad because we don't propagate the memory
            for (int i = 0; i < 12; i++) v[tmpAt + i] = 0;
            // Normalize tmp to max range
            int shift = 14 - celt_ilog2(IMAX(1, MaxAbs(v, tmpAt, len)));
            if (shift != 0)
                for (int i = 0; i < len; i++) v[tmpAt + i] = SHL16(v[tmpAt + i], shift);
            int mean = 0;
            mem0 = 0;
            // Grouping by two; forward pass to compute the post-echo threshold
            for (int i = 0; i < len2; i++)
            {
                int x2 = S16(PSHR32(MULT16_16(v[tmpAt + 2 * i], v[tmpAt + 2 * i]) + MULT16_16(v[tmpAt + 2 * i + 1], v[tmpAt + 2 * i + 1]), 16));
                mean += x2;
                v[tmpAt + i] = S16(mem0 + PSHR32(x2 - mem0, forward_shift));
                mem0 = v[tmpAt + i];
            }
            mem0 = 0;
            int maxE = 0;
            // Backward pass to compute the pre-echo threshold (backward masking: 13.9 dB/ms)
            for (int i = len2 - 1; i >= 0; i--)
            {
                v[tmpAt + i] = S16(mem0 + PSHR32(v[tmpAt + i] - mem0, 3));
                mem0 = v[tmpAt + i];
                maxE = IMAX(maxE, mem0);
            }
            // Frame energy = geometric mean of the energy and half the max (two sqrt() to avoid overflows)
            mean = MULT16_16(CeltFixed.celt_sqrt(mean), CeltFixed.celt_sqrt(MULT16_16(maxE, len2 >> 1)));
            // Inverse of the mean energy in Q15+6
            int norm = SHL32(len2, 6 + 14) / (EPSILON + (mean >> 1));
            // Harmonic mean over 1/4 of the samples, without the unreliable boundaries
            int unmask = 0;
            for (int i = 12; i < len2 - 5; i += 4)
            {
                int id = IMAX(0, IMIN(127, MULT16_32_Q15(v[tmpAt + i] + EPSILON, norm)));   // do not round to nearest
                unmask += mode.TransientInvTableOf(id);
            }
            // Normalize, compensate for the 1/4th of the samples and the factor of 6 in the inverse table
            unmask = 64 * unmask * 4 / (6 * (len2 - 17));
            if (unmask > mask_metric)
            {
                tf_chan = c;
                mask_metric = unmask;
            }
        }
        int is_transient = mask_metric > 200 ? 1 : 0;
        // "Weak transients" at low bitrates are handled differently to avoid partial collapse.
        if (allow_weak_transients != 0 && is_transient != 0 && mask_metric < 600)
        {
            is_transient = 0;
            weak_transient = 1;
        }
        // Arbitrary metric for VBR boost
        int tf_max = S16(IMAX(0, CeltFixed.celt_sqrt(27 * mask_metric) - 42));
        // tf_estimate = 1 + MIN16(1, sqrt(MAX16(0, tf_max-30))/20)
        tf_estimate = S16(CeltFixed.celt_sqrt(IMAX(0, SHL32(MULT16_16(113, IMIN(163, tf_max)), 14) - 37312528)));   // QCONST16(0.0069,14), QCONST32(0.139,28)
        return is_transient;
    }

    /// <summary>
    /// libopus patch_transient_decision: a sudden energy increase over the previous frame (log2 energies, channel c at
    /// + c * NbEBands) makes the frame transient after all. <paramref name="tmpAt"/> holds NbEBands ints of scratch.
    /// </summary>
    public static int PatchTransientDecision(ArrayView<int> newE, int newAt, ArrayView<int> oldE, int oldAt, int start, int end, int C,
        ArrayView<int> v, int tmpAt)
    {
        int nb = CeltModeTables.NbEBands;
        // An aggressive (-6 dB/Bark) spreading of the old frame avoids false detections from irrelevant bands
        if (C == 1)
        {
            v[tmpAt + start] = oldE[oldAt + start];
            for (int i = start + 1; i < end; i++)
                v[tmpAt + i] = IMAX(v[tmpAt + i - 1] - 1024, oldE[oldAt + i]);
        }
        else
        {
            v[tmpAt + start] = IMAX(oldE[oldAt + start], oldE[oldAt + start + nb]);
            for (int i = start + 1; i < end; i++)
                v[tmpAt + i] = IMAX(v[tmpAt + i - 1] - 1024, IMAX(oldE[oldAt + i], oldE[oldAt + i + nb]));
        }
        for (int i = end - 2; i >= start; i--)
            v[tmpAt + i] = IMAX(v[tmpAt + i], v[tmpAt + i + 1] - 1024);
        // Mean increase
        int mean_diff = 0;
        for (int c = 0; c < C; c++)
            for (int i = IMAX(2, start); i < end - 1; i++)
            {
                int x1 = IMAX(0, newE[newAt + i + c * nb]);
                int x2 = IMAX(0, v[tmpAt + i]);
                mean_diff += IMAX(0, SUB16(x1, x2));
            }
        mean_diff = mean_diff / (C * (end - 1 - IMAX(2, start)));
        return mean_diff > 1024 ? 1 : 0;   // QCONST16(1.f, DB_SHIFT)
    }

    /// <summary>
    /// libopus compute_mdcts: the windowed forward MDCT of every block of the CC channels' pre-emphasised input (channel c at
    /// <paramref name="inAt"/> + c * (N + overlap)), interleaved by block, to <paramref name="outAt"/> + c * N; a mono stream of
    /// a stereo input is the average; with <paramref name="upsample"/> the bins above the input band are cleared.
    /// <paramref name="scratchAt"/> holds 2 * 960 ints (the MDCT's).
    /// </summary>
    public static void ComputeMdcts(CeltModeGpu m, int shortBlocks, ArrayView<int> v, int inAt, int outAt, int C, int CC, int LM,
        int upsample, int scratchAt)
    {
        const int overlap = 120;
        int B, N, shift;
        if (shortBlocks != 0)
        {
            B = shortBlocks;
            N = 120;
            shift = 3;
        }
        else
        {
            B = 1;
            N = 120 << LM;
            shift = 3 - LM;
        }
        for (int c = 0; c < CC; c++)
            for (int b = 0; b < B; b++)
                // Interleaving the sub-frames while doing the MDCTs
                CeltMdctGpu.Forward(m, v, inAt + c * (B * N + overlap) + b * N, v, outAt + b + c * N * B, overlap, shift, B, v, scratchAt);
        if (CC == 2 && C == 1)
            for (int i = 0; i < B * N; i++)
                v[outAt + i] = HALF32(v[outAt + i]) + HALF32(v[outAt + B * N + i]);
        if (upsample != 1)
        {
            for (int c = 0; c < C; c++)
            {
                int bound = B * N / upsample;
                for (int i = 0; i < bound; i++) v[outAt + c * B * N + i] *= upsample;
                for (int i = bound; i < B * N; i++) v[outAt + c * B * N + i] = 0;
            }
        }
    }

    /// <summary>libopus compute_band_energies (fixed point): each band's amplitude (sqrt energy) of the MDCT at
    /// <paramref name="xAt"/> (channel c at + c * N) to <paramref name="eAt"/> + c * NbEBands.</summary>
    public static void ComputeBandEnergies(CeltModeGpu m, ArrayView<int> X, int xAt, ArrayView<int> bandE, int eAt, int end, int C, int LM)
    {
        int nb = CeltModeTables.NbEBands;
        int N = 120 << LM;
        for (int c = 0; c < C; c++)
            for (int i = 0; i < end; i++)
            {
                int lo = m.EBandsOf(i) << LM, hi = m.EBandsOf(i + 1) << LM;
                int maxval = MaxAbs(X, xAt + c * N + lo, hi - lo);
                if (maxval > 0)
                {
                    int shift = celt_ilog2(maxval) - 14 + (((m.LogNOf(i) >> BITRES) + LM + 1) >> 1);
                    int sum = 0;
                    if (shift > 0)
                        for (int j = lo; j < hi; j++)
                        {
                            int s = S16(X[xAt + j + c * N] >> shift);
                            sum = MAC16_16(sum, s, s);
                        }
                    else
                        for (int j = lo; j < hi; j++)
                        {
                            int s = S16(SHL32(X[xAt + j + c * N], -shift));
                            sum = MAC16_16(sum, s, s);
                        }
                    // Adding one ensures the normalised band isn't larger than unity norm
                    bandE[eAt + i + c * nb] = EPSILON + VSHR32(CeltFixed.celt_sqrt(sum), -shift);
                }
                else
                    bandE[eAt + i + c * nb] = EPSILON;
            }
    }

    /// <summary>libopus normalise_bands: the MDCT at <paramref name="freqAt"/> scaled to unit-energy bands (celt_norm, 16-bit)
    /// at <paramref name="xAt"/>.</summary>
    public static void NormaliseBands(CeltModeGpu m, ArrayView<int> freq, int freqAt, ArrayView<int> X, int xAt, ArrayView<int> bandE, int eAt,
        int end, int C, int M)
    {
        int nb = CeltModeTables.NbEBands;
        int N = M * 120;
        for (int c = 0; c < C; c++)
            for (int i = 0; i < end; i++)
            {
                int be = bandE[eAt + i + c * nb];
                int shift = celt_zlog2(be) - 13;
                int E = S16(VSHR32(be, shift));
                int g = S16(CeltFixed.celt_rcp(SHL32(E, 3)));
                for (int j = M * m.EBandsOf(i); j < M * m.EBandsOf(i + 1); j++)
                    X[xAt + j + c * N] = S16(MULT16_16_Q15(VSHR32(freq[freqAt + j + c * N], shift - 1), g));
            }
    }

    // libopus l1_metric: L1 norm of the band, biased towards good frequency resolution
    private static int L1Metric(ArrayView<int> v, int at, int N, int LM, int bias)
    {
        int L1 = 0;
        for (int i = 0; i < N; i++) L1 += ABS32(v[at + i]);
        return MAC16_32_Q15(L1, LM * bias, L1);
    }

    /// <summary>
    /// libopus tf_analysis: per band, the TF resolution change (Haar levels) that minimises the L1 norm of the channel
    /// <paramref name="tf_chan"/>'s normalised spectrum (X at <paramref name="xAt"/>, channels <paramref name="N0"/> apart), then a
    /// Viterbi pass over the bands with switching cost <paramref name="lambda"/> and per-band <paramref name="importance"/>.
    /// Writes tf_res (0 / 1) and returns tf_select. <paramref name="scratchAt"/> holds 3 * NbEBands + 2 * 176 ints.
    /// </summary>
    public static int TfAnalysis(CeltModeGpu m, int len, int isTransient, ArrayView<int> tf_res, int tfAt, int lambda, ArrayView<int> X, int xAt,
        int N0, int LM, int tf_estimate, int tf_chan, ArrayView<int> importance, int impAt, ArrayView<int> v, int scratchAt)
    {
        int metricAt = scratchAt, path0At = scratchAt + 21, path1At = scratchAt + 42, tmpAt = scratchAt + 63, tmp1At = tmpAt + 176;
        int bias = MULT16_16_Q14(1311, IMAX(-4096, 8192 - tf_estimate));   // QCONST16(.04f,15), QCONST16(.25f,14), QCONST16(.5f,14)
        for (int i = 0; i < len; i++)
        {
            int N = (m.EBandsOf(i + 1) - m.EBandsOf(i)) << LM;
            // band is too narrow to be split down to LM=-1
            int narrow = m.EBandsOf(i + 1) - m.EBandsOf(i) == 1 ? 1 : 0;
            for (int j = 0; j < N; j++) v[tmpAt + j] = X[xAt + tf_chan * N0 + (m.EBandsOf(i) << LM) + j];
            int L1 = L1Metric(v, tmpAt, N, isTransient != 0 ? LM : 0, bias);
            int best_L1 = L1, best_level = 0;
            // Check the -1 case for transients
            if (isTransient != 0 && narrow == 0)
            {
                for (int j = 0; j < N; j++) v[tmp1At + j] = v[tmpAt + j];
                CeltBandsGpu.Haar1(v, tmp1At, N >> LM, 1 << LM);
                L1 = L1Metric(v, tmp1At, N, LM + 1, bias);
                if (L1 < best_L1)
                {
                    best_L1 = L1;
                    best_level = -1;
                }
            }
            for (int k = 0; k < LM + (isTransient != 0 || narrow != 0 ? 0 : 1); k++)
            {
                int B = isTransient != 0 ? LM - k - 1 : k + 1;
                CeltBandsGpu.Haar1(v, tmpAt, N >> k, 1 << k);
                L1 = L1Metric(v, tmpAt, N, B, bias);
                if (L1 < best_L1)
                {
                    best_L1 = L1;
                    best_level = k + 1;
                }
            }
            // metric is in Q1 to be able to select the mid-point (-0.5) for narrower bands
            int metric = isTransient != 0 ? 2 * best_level : -2 * best_level;
            // Bands that can't be split to -1 get the half-way point, to avoid biasing the decision
            if (narrow != 0 && (metric == 0 || metric == -2 * LM)) metric -= 1;
            v[metricAt + i] = metric;
        }
        // Search for the optimal tf resolution, including tf_select
        int tsBase = LM * 8 + 4 * isTransient;
        int selcost0 = 0, selcost1 = 0;
        for (int sel = 0; sel < 2; sel++)
        {
            int t0 = 2 * m.TfSelectTableOf(tsBase + 2 * sel), t1 = 2 * m.TfSelectTableOf(tsBase + 2 * sel + 1);
            int cost0 = importance[impAt] * ABS32(v[metricAt] - t0);
            int cost1 = importance[impAt] * ABS32(v[metricAt] - t1) + (isTransient != 0 ? 0 : lambda);
            for (int i = 1; i < len; i++)
            {
                int curr0 = IMIN(cost0, cost1 + lambda);
                int curr1 = IMIN(cost0 + lambda, cost1);
                cost0 = curr0 + importance[impAt + i] * ABS32(v[metricAt + i] - t0);
                cost1 = curr1 + importance[impAt + i] * ABS32(v[metricAt + i] - t1);
            }
            cost0 = IMIN(cost0, cost1);
            if (sel == 0) selcost0 = cost0; else selcost1 = cost0;
        }
        // Conservative: tf_select = 1 only for transients
        int tf_select = selcost1 < selcost0 && isTransient != 0 ? 1 : 0;
        {
            int t0 = 2 * m.TfSelectTableOf(tsBase + 2 * tf_select), t1 = 2 * m.TfSelectTableOf(tsBase + 2 * tf_select + 1);
            int cost0 = importance[impAt] * ABS32(v[metricAt] - t0);
            int cost1 = importance[impAt] * ABS32(v[metricAt] - t1) + (isTransient != 0 ? 0 : lambda);
            // Viterbi forward pass
            for (int i = 1; i < len; i++)
            {
                int curr0, curr1;
                int from0 = cost0, from1 = cost1 + lambda;
                if (from0 < from1) { curr0 = from0; v[path0At + i] = 0; }
                else { curr0 = from1; v[path0At + i] = 1; }
                from0 = cost0 + lambda;
                from1 = cost1;
                if (from0 < from1) { curr1 = from0; v[path1At + i] = 0; }
                else { curr1 = from1; v[path1At + i] = 1; }
                cost0 = curr0 + importance[impAt + i] * ABS32(v[metricAt + i] - t0);
                cost1 = curr1 + importance[impAt + i] * ABS32(v[metricAt + i] - t1);
            }
            tf_res[tfAt + len - 1] = cost0 < cost1 ? 0 : 1;
        }
        // Viterbi backward pass to check the decisions
        for (int i = len - 2; i >= 0; i--)
            tf_res[tfAt + i] = tf_res[tfAt + i + 1] == 1 ? v[path1At + i + 1] : v[path0At + i + 1];
        return tf_select;
    }

    /// <summary>
    /// libopus spreading_decision: how peaky the normalised bands are (a rough CDF of |x|^2 N per band, weighted by
    /// spread_weight, recursively averaged with hysteresis) picks the spread rotation (0 none .. 3 aggressive); with
    /// <paramref name="update_hf"/> the high bands' average also updates the pitch tapset decision.
    /// </summary>
    public static int SpreadingDecision(CeltModeGpu m, ArrayView<int> X, int xAt, ref int average, int last_decision, ref int hf_average,
        ref int tapset_decision, int update_hf, int end, int C, int M, ArrayView<int> spread_weight, int swAt)
    {
        int sum = 0, nbBands = 0, hf_sum = 0;
        int N0 = M * 120;
        if (M * (m.EBandsOf(end) - m.EBandsOf(end - 1)) <= 8)
            return 0;   // SPREAD_NONE
        for (int c = 0; c < C; c++)
            for (int i = 0; i < end; i++)
            {
                int x = xAt + M * m.EBandsOf(i) + c * N0;
                int N = M * (m.EBandsOf(i + 1) - m.EBandsOf(i));
                if (N <= 8) continue;
                int t0 = 0, t1 = 0, t2 = 0;
                // Compute rough CDF of |x[j]|
                for (int j = 0; j < N; j++)
                {
                    int x2N = MULT16_16(MULT16_16_Q15(X[x + j], X[x + j]), N);   // Q13
                    if (x2N < 2048) t0++;    // QCONST16(0.25f, 13)
                    if (x2N < 512) t1++;     // QCONST16(0.0625f, 13)
                    if (x2N < 128) t2++;     // QCONST16(0.015625f, 13)
                }
                // Only include four last bands (8 kHz and up)
                if (i > CeltModeTables.NbEBands - 4)
                    hf_sum += (int)((uint)(32 * (t1 + t0)) / (uint)N);
                int tmp = (2 * t2 >= N ? 1 : 0) + (2 * t1 >= N ? 1 : 0) + (2 * t0 >= N ? 1 : 0);
                sum += tmp * spread_weight[swAt + i];
                nbBands += spread_weight[swAt + i];
            }
        if (update_hf != 0)
        {
            if (hf_sum != 0)
                hf_sum = (int)((uint)hf_sum / (uint)(C * (4 - CeltModeTables.NbEBands + end)));
            hf_average = (hf_average + hf_sum) >> 1;
            hf_sum = hf_average;
            if (tapset_decision == 2) hf_sum += 4;
            else if (tapset_decision == 0) hf_sum -= 4;
            if (hf_sum > 22) tapset_decision = 2;
            else if (hf_sum > 18) tapset_decision = 1;
            else tapset_decision = 0;
        }
        sum = (int)((uint)(sum << 8) / (uint)nbBands);
        // Recursive averaging
        sum = (sum + average) >> 1;
        average = sum;
        // Hysteresis
        sum = (3 * sum + (((3 - last_decision) << 7) + 64) + 2) >> 2;
        if (sum < 80) return 3;          // SPREAD_AGGRESSIVE
        else if (sum < 256) return 2;    // SPREAD_NORMAL
        else if (sum < 384) return 1;    // SPREAD_LIGHT
        return 0;                        // SPREAD_NONE
    }

    private static int InnerProd(ArrayView<int> v, int xAt, int yAt, int N)
    {
        int xy = 0;
        for (int i = 0; i < N; i++) xy = MAC16_16(xy, v[xAt + i], v[yAt + i]);
        return xy;
    }

    /// <summary>
    /// libopus alloc_trim_analysis without tonality analysis (analysis->valid is 0 below complexity 10) and without surround
    /// masking: the allocation trim (0..10) from the bitrate, the inter-channel correlation (stereo; updates
    /// <paramref name="stereo_saving"/>, Q8), the spectral tilt and the tf estimate.
    /// </summary>
    public static int AllocTrimAnalysis(CeltModeGpu m, ArrayView<int> X, int xAt, ArrayView<int> bandLogE, int logAt, int end, int LM, int C,
        int N0, ref int stereo_saving, int tf_estimate, int intensity, int equiv_rate)
    {
        int nb = CeltModeTables.NbEBands;
        int trim = 1280;   // QCONST16(5.f, 8)
        // At low bitrate, reducing the trim seems to help.
        if (equiv_rate < 64000)
            trim = 1024;
        else if (equiv_rate < 80000)
            trim = S16(1024 + 16 * ((equiv_rate - 64000) >> 10));
        if (C == 2)
        {
            int sum = 0;   // Q10
            // Inter-channel correlation for low frequencies
            for (int i = 0; i < 8; i++)
            {
                int partial = InnerProd(X, xAt + (m.EBandsOf(i) << LM), xAt + N0 + (m.EBandsOf(i) << LM), (m.EBandsOf(i + 1) - m.EBandsOf(i)) << LM);
                sum = ADD16(sum, S16(partial >> 18));
            }
            sum = S16(MULT16_16_Q15(4096, sum));                  // QCONST16(1.f/8, 15)
            sum = IMIN(1024, ABS32(sum));
            int minXC = sum;
            for (int i = 8; i < intensity; i++)
            {
                int partial = InnerProd(X, xAt + (m.EBandsOf(i) << LM), xAt + N0 + (m.EBandsOf(i) << LM), (m.EBandsOf(i + 1) - m.EBandsOf(i)) << LM);
                minXC = IMIN(minXC, ABS32(S16(partial >> 18)));
            }
            minXC = IMIN(1024, ABS32(minXC));
            // mid-side savings estimations based on the LF average, and on the min correlation
            int logXC = celt_log2(1049625 - MULT16_16(sum, sum));   // QCONST32(1.001f, 20)
            int logXC2 = IMAX(HALF16(logXC), celt_log2(1049625 - MULT16_16(minXC, minXC)));
            // Compensate for Q20 vs Q14 input and convert output to Q8
            logXC = S16(PSHR32(logXC - 6144, DB_SHIFT - 8));
            logXC2 = S16(PSHR32(logXC2 - 6144, DB_SHIFT - 8));
            trim = S16(trim + IMAX(-1024, MULT16_16_Q15(24576, logXC)));   // QCONST16(.75f, 15)
            stereo_saving = S16(IMIN(stereo_saving + 64, -HALF16(logXC2)));
        }
        // Estimate spectral tilt
        int diff = 0;
        for (int c = 0; c < C; c++)
            for (int i = 0; i < end - 1; i++)
                diff += bandLogE[logAt + i + c * nb] * (2 + 2 * i - end);
        diff /= C * (end - 1);
        trim = S16(trim - IMAX(-512, IMIN(512, ((diff + 1024) >> (DB_SHIFT - 8)) / 6)));
        trim = S16(trim - 2 * SHR16(tf_estimate, 14 - 8));
        int trim_index = PSHR32(trim, 8);
        return IMAX(0, IMIN(10, trim_index));
    }

    /// <summary>libopus stereo_analysis: 1 to code the frame dual stereo (L/R) rather than mid / side, by an L1 entropy model
    /// of the 13 lowest bands.</summary>
    public static int StereoAnalysis(CeltModeGpu m, ArrayView<int> X, int xAt, int LM, int N0)
    {
        int sumLR = EPSILON, sumMS = EPSILON;
        // Use the L1 norm to model the entropy of the L/R signal vs the M/S signal
        for (int i = 0; i < 13; i++)
            for (int j = m.EBandsOf(i) << LM; j < m.EBandsOf(i + 1) << LM; j++)
            {
                int L = X[xAt + j], R = X[xAt + N0 + j];
                int Mi = L + R, S = L - R;
                sumLR += ABS32(L) + ABS32(R);
                sumMS += ABS32(Mi) + ABS32(S);
            }
        sumMS = MULT16_32_Q15(23170, sumMS);   // QCONST16(0.707107f, 15)
        int thetas = 13;
        // We don't need thetas for lower bands with LM<=1
        if (LM <= 1) thetas -= 8;
        return MULT16_32_Q15((m.EBandsOf(13) << (LM + 1)) + thetas, sumMS) > MULT16_32_Q15(m.EBandsOf(13) << (LM + 1), sumLR) ? 1 : 0;
    }

    private static int MedianOf5(ArrayView<int> v, int at)
    {
        int t0, t1, t2 = v[at + 2], t3, t4;
        if (v[at] > v[at + 1]) { t0 = v[at + 1]; t1 = v[at]; } else { t0 = v[at]; t1 = v[at + 1]; }
        if (v[at + 3] > v[at + 4]) { t3 = v[at + 4]; t4 = v[at + 3]; } else { t3 = v[at + 3]; t4 = v[at + 4]; }
        if (t0 > t3)
        {
            int tmp = t0; t0 = t3; t3 = tmp;
            tmp = t1; t1 = t4; t4 = tmp;
        }
        if (t2 > t1)
            return t1 < t3 ? IMIN(t2, t3) : IMIN(t4, t1);
        return t2 < t3 ? IMIN(t1, t3) : IMIN(t2, t4);
    }

    private static int MedianOf3(ArrayView<int> v, int at)
    {
        int t0, t1, t2 = v[at + 2];
        if (v[at] > v[at + 1]) { t0 = v[at + 1]; t1 = v[at]; } else { t0 = v[at]; t1 = v[at + 1]; }
        if (t1 < t2) return t1;
        else if (t0 < t2) return t2;
        return t0;
    }

    /// <summary>
    /// libopus dynalloc_analysis without surround masking or tonality analysis: per-band dynamic allocation boosts
    /// (<paramref name="offsets"/>, in quanta), their total (<paramref name="tot_boost"/>, 1/8 bits), the importance of each band
    /// (for tf_analysis), the spread weights (for spreading_decision); returns maxDepth. <paramref name="scratchAt"/> holds 6 *
    /// NbEBands ints.
    /// </summary>
    public static int DynallocAnalysis(CeltModeGpu m, ArrayView<int> bandLogE, int logAt, ArrayView<int> bandLogE2, int log2At,
        ArrayView<int> oldBandE, int oldAt, int start, int end, int C, ArrayView<int> offsets, int offAt, int lsb_depth, int isTransient,
        int vbr, int constrained_vbr, int LM, int effectiveBytes, ref int tot_boost_, int lfe, ArrayView<int> importance, int impAt,
        ArrayView<int> spread_weight, int swAt, ArrayView<int> v, int scratchAt)
    {
        int nb = CeltModeTables.NbEBands;
        int followerAt = scratchAt, floorAt = scratchAt + 2 * nb, e3At = scratchAt + 3 * nb, maskAt = scratchAt + 4 * nb, sigAt = scratchAt + 5 * nb;
        int tot_boost = 0;
        for (int i = 0; i < nb; i++) offsets[offAt + i] = 0;
        // Dynamic allocation code
        int maxDepth = -32666;   // -QCONST16(31.9f, DB_SHIFT)
        for (int i = 0; i < end; i++)
        {
            // Noise floor: eMeans, the depth, the width of the bands and the preemphasis filter (approx. square of bark band ID)
            v[floorAt + i] = S16(MULT16_16(64, m.LogNOf(i)) + 512 + SHL16(9 - lsb_depth, DB_SHIFT) - SHL16(m.EMeansOf(i), 6)
                + MULT16_16(6, (i + 5) * (i + 5)));
        }
        for (int c = 0; c < C; c++)
            for (int i = 0; i < end; i++)
                maxDepth = S16(IMAX(maxDepth, bandLogE[logAt + c * nb + i] - v[floorAt + i]));
        {
            // A really simple masking model, to avoid taking completely masked bands into account in the spreading decision.
            for (int i = 0; i < end; i++)
                v[maskAt + i] = S16(bandLogE[logAt + i] - v[floorAt + i]);
            if (C == 2)
                for (int i = 0; i < end; i++)
                    v[maskAt + i] = S16(IMAX(v[maskAt + i], bandLogE[logAt + nb + i] - v[floorAt + i]));
            for (int i = 0; i < end; i++) v[sigAt + i] = v[maskAt + i];
            for (int i = 1; i < end; i++)
                v[maskAt + i] = S16(IMAX(v[maskAt + i], v[maskAt + i - 1] - 2048));
            for (int i = end - 2; i >= 0; i--)
                v[maskAt + i] = S16(IMAX(v[maskAt + i], v[maskAt + i + 1] - 3072));
            for (int i = 0; i < end; i++)
            {
                // SMR: the mask is never more than 72 dB below the peak and never below the noise floor.
                int smr = S16(v[sigAt + i] - IMAX(IMAX(0, maxDepth - 12288), v[maskAt + i]));
                // Clamp SMR to make sure we're not shifting by something negative or too large.
                int shift = -PSHR32(IMAX(-5120, IMIN(0, smr)), DB_SHIFT);
                spread_weight[swAt + i] = 32 >> shift;
            }
        }
        // Dynamic allocation can't bust the budget: from 24 kb/s for 20-ms frames and 96 kb/s for 2.5 ms frames.
        if (effectiveBytes >= (30 + 5 * LM) && lfe == 0)
        {
            int last = 0;
            for (int c = 0; c < C; c++)
            {
                for (int i = 0; i < end; i++) v[e3At + i] = bandLogE2[log2At + c * nb + i];
                if (LM == 0)
                {
                    // 2.5 ms: the first 8 bands have one bin and an unreliable energy; use at least 2 bins' worth.
                    for (int i = 0; i < IMIN(8, end); i++)
                        v[e3At + i] = S16(IMAX(bandLogE2[log2At + c * nb + i], oldBandE[oldAt + c * nb + i]));
                }
                int f = followerAt + c * nb;
                v[f] = v[e3At];
                for (int i = 1; i < end; i++)
                {
                    // The last band at least 3 dB above the previous one is the last we consider (band-limited signals).
                    if (v[e3At + i] > v[e3At + i - 1] + 512)
                        last = i;
                    v[f + i] = S16(IMIN(v[f + i - 1] + 1536, v[e3At + i]));
                }
                for (int i = last - 1; i >= 0; i--)
                    v[f + i] = S16(IMIN(v[f + i], IMIN(v[f + i + 1] + 2048, v[e3At + i])));
                // Combine with a median filter to avoid dynalloc triggering unnecessarily.
                const int offset = 1024;
                for (int i = 2; i < end - 2; i++)
                    v[f + i] = S16(IMAX(v[f + i], MedianOf5(v, e3At + i - 2) - offset));
                int tmp = S16(MedianOf3(v, e3At) - offset);
                v[f] = IMAX(v[f], tmp);
                v[f + 1] = IMAX(v[f + 1], tmp);
                tmp = S16(MedianOf3(v, e3At + end - 3) - offset);
                v[f + end - 2] = IMAX(v[f + end - 2], tmp);
                v[f + end - 1] = IMAX(v[f + end - 1], tmp);
                for (int i = 0; i < end; i++)
                    v[f + i] = IMAX(v[f + i], v[floorAt + i]);
            }
            if (C == 2)
            {
                for (int i = start; i < end; i++)
                {
                    // Consider 24 dB "cross-talk"
                    v[followerAt + nb + i] = S16(IMAX(v[followerAt + nb + i], v[followerAt + i] - 4096));
                    v[followerAt + i] = S16(IMAX(v[followerAt + i], v[followerAt + nb + i] - 4096));
                    v[followerAt + i] = S16(HALF16(IMAX(0, bandLogE[logAt + i] - v[followerAt + i])
                        + IMAX(0, bandLogE[logAt + nb + i] - v[followerAt + nb + i])));
                }
            }
            else
            {
                for (int i = start; i < end; i++)
                    v[followerAt + i] = S16(IMAX(0, bandLogE[logAt + i] - v[followerAt + i]));
            }
            // (surround_dynalloc is 0 without an energy mask)
            for (int i = start; i < end; i++)
                importance[impAt + i] = PSHR32(13 * CeltFixed.celt_exp2(IMIN(v[followerAt + i], 4096)), 16);
            // For non-transient CBR/CVBR frames, halve the dynalloc contribution
            if ((vbr == 0 || constrained_vbr != 0) && isTransient == 0)
                for (int i = start; i < end; i++)
                    v[followerAt + i] = HALF16(v[followerAt + i]);
            for (int i = start; i < end; i++)
            {
                if (i < 8) v[followerAt + i] = S16(v[followerAt + i] * 2);
                if (i >= 12) v[followerAt + i] = HALF16(v[followerAt + i]);
            }
            for (int i = start; i < end; i++)
            {
                v[followerAt + i] = IMIN(v[followerAt + i], 4096);
                int width = C * (m.EBandsOf(i + 1) - m.EBandsOf(i)) << LM;
                int boost, boost_bits;
                if (width < 6)
                {
                    boost = v[followerAt + i] >> DB_SHIFT;
                    boost_bits = boost * width << BITRES;
                }
                else if (width > 48)
                {
                    boost = (v[followerAt + i] * 8) >> DB_SHIFT;
                    boost_bits = (boost * width << BITRES) / 8;
                }
                else
                {
                    boost = (v[followerAt + i] * width / 6) >> DB_SHIFT;
                    boost_bits = boost * 6 << BITRES;
                }
                // For CBR and non-transient CVBR frames, limit dynalloc to 2/3 of the bits
                if ((vbr == 0 || (constrained_vbr != 0 && isTransient == 0)) && (tot_boost + boost_bits) >> BITRES >> 3 > 2 * effectiveBytes / 3)
                {
                    int cap = (2 * effectiveBytes / 3) << BITRES << 3;
                    offsets[offAt + i] = cap - tot_boost;
                    tot_boost = cap;
                    break;
                }
                offsets[offAt + i] = boost;
                tot_boost += boost_bits;
            }
        }
        else
        {
            for (int i = start; i < end; i++)
                importance[impAt + i] = 13;
        }
        tot_boost_ = tot_boost;
        return maxDepth;
    }

    /// <summary>libopus tf_encode: the per-band TF changes (differential) and tf_select, then tf_res mapped through
    /// tf_select_table.</summary>
    public static void TfEncode(CeltModeGpu m, int start, int end, int isTransient, ArrayView<int> tf_res, int tfAt, int LM, int tf_select,
        ref OpusRangeEncoderGpuState enc, ArrayView<byte> buf, int bufStart, uint storage)
    {
        uint budget = storage * 8;
        uint tell = (uint)OpusRangeEncoderGpu.Tell(ref enc);
        int logp = isTransient != 0 ? 2 : 4;
        // Reserve space to code the tf_select decision.
        int tf_select_rsv = LM > 0 && tell + (uint)logp + 1 <= budget ? 1 : 0;
        budget -= (uint)tf_select_rsv;
        int curr = 0, tf_changed = 0;
        for (int i = start; i < end; i++)
        {
            if (tell + (uint)logp <= budget)
            {
                OpusRangeEncoderGpu.EncodeBitLogP(ref enc, buf, bufStart, storage, tf_res[tfAt + i] ^ curr, logp);
                tell = (uint)OpusRangeEncoderGpu.Tell(ref enc);
                curr = tf_res[tfAt + i];
                tf_changed |= curr;
            }
            else
                tf_res[tfAt + i] = curr;
            logp = isTransient != 0 ? 4 : 5;
        }
        // Only code tf_select if it would actually make a difference.
        int tsBase = LM * 8 + 4 * isTransient;
        if (tf_select_rsv != 0 && m.TfSelectTableOf(tsBase + tf_changed) != m.TfSelectTableOf(tsBase + 2 + tf_changed))
            OpusRangeEncoderGpu.EncodeBitLogP(ref enc, buf, bufStart, storage, tf_select, 1);
        else
            tf_select = 0;
        for (int i = start; i < end; i++)
            tf_res[tfAt + i] = m.TfSelectTableOf(tsBase + 2 * tf_select + tf_res[tfAt + i]);
    }

    /// <summary>libopus amp2Log2: the band amplitudes as log2 energies relative to eMeans (DB_SHIFT); bands from effEnd to end
    /// get -14.</summary>
    public static void Amp2Log2(CeltModeGpu m, int effEnd, int end, ArrayView<int> bandE, int eAt, ArrayView<int> bandLogE, int logAt, int C)
    {
        int nb = CeltModeTables.NbEBands;
        for (int c = 0; c < C; c++)
        {
            for (int i = 0; i < effEnd; i++)
            {
                int l = S16(celt_log2(bandE[eAt + i + c * nb]) - SHL16(m.EMeansOf(i), 6));
                // Compensate for bandE[] being Q12 but celt_log2() taking a Q14 input.
                bandLogE[logAt + i + c * nb] = S16(l + 2048);
            }
            for (int i = effEnd; i < end; i++)
                bandLogE[logAt + c * nb + i] = -14336;   // -QCONST16(14.f, DB_SHIFT)
        }
    }
}
