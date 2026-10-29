// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// The CELT encoder on the accelerator: libopus 1.5.2 celt_encode_with_ec (fixed point, celt/celt_encoder.c) as one GPU
// thread per call, with the encoder state (CELTEncoder from ENCODER_RESET_START, plus in_mem / prefilter_mem / the band
// energy histories) in a device view that persists across calls. The stages are CeltEncoderAnalysisGpu, CeltPrefilterGpu,
// CeltEnergyEncGpu, CeltRateEncGpu and CeltQuantBandsEncGpu; the host only allocates, uploads the CTLs of each call and
// dispatches (SpawnDev.Codecs' host-is-coordinator rule).
//
// Scope: the Opus layer's CELT calls (enc == NULL here: the Opus layer's CELT-only frames start the range coder at tell 1),
// no custom modes (signalling 0), no tonality analysis (libopus' fixed-point build runs it only at complexity 10), no
// surround energy mask.

using ILGPU;
using ILGPU.Runtime;
using static SpawnDev.Codecs.Audio.Opus.Celt.CeltOps;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>The CTLs and buffers of one celt_encode_with_ec call (ints, in <see cref="CeltEncoderGpu.Ctl"/> order).</summary>
public static class CeltEncoderGpu
{
    private const int Nb = CeltModeTables.NbEBands, Overlap = 120, ShortMdctSize = 120, MaxPeriod = CeltPrefilterGpu.MaxPeriod;
    private const int SpreadNone = 0, SpreadNormal = 2, BitrateMax = -1;

    /// <summary>The per-call CTL view layout (ints).</summary>
    public enum Ctl
    {
        /// <summary>First int16 of the call's interleaved input in the pcm view.</summary>
        PcmAt, FrameSize, CC, NbCompressedBytes, Bitrate, Vbr, ConstrainedVbr, Complexity, Start, End, C, LsbDepth, DisablePf,
        ForceIntra, Lfe, DisableInv, LossRate, Upsample,
        /// <summary>Where the CELT bytes go in the out view, and where the byte count goes in the length view.</summary>
        OutAt, LengthAt,
        /// <summary>Ints in the CTL view.</summary>
        Count
    }

    // state layout (ints)
    /// <summary>State scalars.</summary>
    public const int SRng = 0, SSpread = 1, SDelayedIntra = 2, STonalAverage = 3, SLastCodedBands = 4, SHfAverage = 5, STapset = 6,
        SPfPeriod = 7, SPfGain = 8, SPfTapset = 9, SConsecTransient = 10, SVbrReservoir = 11, SVbrDrift = 12, SVbrOffset = 13,
        SVbrCount = 14, SOverlapMax = 15, SStereoSaving = 16, SIntensity = 17, SSpecAvg = 18, SPreemph = 19;
    /// <summary>in_mem (2 * overlap), prefilter_mem (2 * COMBFILTER_MAXPERIOD), oldBandE, oldLogE, oldLogE2, energyError (2 * NbEBands each).</summary>
    public const int InMemAt = 24, PrefilterMemAt = InMemAt + 2 * Overlap, OldBandEAt = PrefilterMemAt + 2 * MaxPeriod,
        OldLogEAt = OldBandEAt + 2 * Nb, OldLogE2At = OldLogEAt + 2 * Nb, EnergyErrorAt = OldLogE2At + 2 * Nb,
        StateInts = EnergyErrorAt + 2 * Nb;

    // work layout (ints)
    /// <summary>Per-call work: the pre-emphasised input, the MDCT, the normalised bands, the per-band arrays, the decisions
    /// (<see cref="WRes"/>, for probes) and the stages' scratch.</summary>
    public const int WIn = 0, WFreq = WIn + 2 * (960 + Overlap), WX = WFreq + 2 * 960, WBandE = WX + 2 * 960, WBandLogE = WBandE + 2 * Nb,
        WBandLogE2 = WBandLogE + 2 * Nb, WError = WBandLogE2 + 2 * Nb, WOffsets = WError + 2 * Nb, WImportance = WOffsets + Nb,
        WSpreadWeight = WImportance + Nb, WTfRes = WSpreadWeight + Nb, WCap = WTfRes + Nb, WPulses = WCap + Nb, WFineQuant = WPulses + Nb,
        WFinePriority = WFineQuant + Nb, WRes = WFinePriority + Nb, WVars = WRes + 32, WScr = WVars + 48, WorkInts = WScr + 8192;
    /// <summary>Decision slots at <see cref="WRes"/>.</summary>
    public const int RPfOn = 0, RPitch = 1, RGain = 2, RQg = 3, RTransient = 4, RShortBlocks = 5, RTfEstimate = 6, RTfSelect = 7,
        RSpread = 8, RTotBoost = 9, RTotalBoost = 10, RDual = 11, RIntensity = 12, RTrim = 13, RCodedBands = 14, RBalance = 15,
        RAntiCollapse = 16, RMaxDepth = 17, RBytes = 18, RError = 19, RSilence = 20, RTfChan = 21, REquivRate = 22, RBits = 23,
        RTellCoarse = 24, RTellAlloc = 25;
    // Per-frame values carried from one stage kernel to the next, at WVars: the range coder (10 ints) and the frame's decisions
    private const int VEnc = 0, VStorage = 10, VNb = 11, VNbAvail = 12, VTotalBits = 13, VEffBytes = 14, VVbrRate = 15, VEquivRate = 16,
        VLM = 17, VSilence = 18, VPitch = 19, VGain1 = 20, VTapset = 21, VPfOn = 22, VTransient = 23, VShortBlocks = 24, VTfEstimate = 25,
        VTransDisabled = 26, VTemporalVbr = 27, VMaxDepth = 28, VTotBoost = 29, VTfSelect = 30, VDual = 31, VIntensity = 32, VCoded = 33,
        VBalance = 34, VAntiCollapseRsv = 35, VValid = 36;

    private static void SaveEnc(ArrayView<int> w, int at, ref OpusRangeEncoderGpuState e)
    {
        w[at] = (int)e.Offs; w[at + 1] = (int)e.EndOffs; w[at + 2] = (int)e.EndWindow; w[at + 3] = e.NEndBits; w[at + 4] = e.NBitsTotal;
        w[at + 5] = (int)e.Rng; w[at + 6] = (int)e.Val; w[at + 7] = (int)e.Ext; w[at + 8] = e.Rem; w[at + 9] = e.Error;
    }

    private static OpusRangeEncoderGpuState LoadEnc(ArrayView<int> w, int at)
    {
        var e = OpusRangeEncoderGpu.Init();
        e.Offs = (uint)w[at]; e.EndOffs = (uint)w[at + 1]; e.EndWindow = (uint)w[at + 2]; e.NEndBits = w[at + 3]; e.NBitsTotal = w[at + 4];
        e.Rng = (uint)w[at + 5]; e.Val = (uint)w[at + 6]; e.Ext = (uint)w[at + 7]; e.Rem = w[at + 8]; e.Error = w[at + 9];
        return e;
    }

    internal static int MaxAbs16(ArrayView<short> v, int at, int len)
    {
        int maxval = 0, minval = 0;
        for (int i = 0; i < len; i++)
        {
            int x = v[at + i];
            maxval = IMAX(maxval, x);
            minval = IMIN(minval, x);
        }
        return IMAX(maxval, -minval);
    }

    private static int IntensityThreshold(int i) => i switch
    {
        0 => 1, 1 => 2, 2 => 3, 3 => 4, 4 => 5, 5 => 6, 6 => 7, 7 => 8, 8 => 16, 9 => 24, 10 => 36, 11 => 44, 12 => 50, 13 => 56,
        14 => 62, 15 => 67, 16 => 72, 17 => 79, 18 => 88, 19 => 106, _ => 134,
    };

    private static int IntensityHysteresis(int i) => i < 7 ? 1 : i < 14 ? 2 : i < 16 ? 3 : i == 16 ? 4 : i == 17 ? 5 : i == 18 ? 6 : 8;

    /// <summary>libopus hysteresis_decision over the intensity thresholds (21 of them).</summary>
    internal static int IntensityDecision(int val, int prev)
    {
        int i;
        for (i = 0; i < Nb; i++)
            if (val < IntensityThreshold(i)) break;
        if (i > prev && val < IntensityThreshold(prev) + IntensityHysteresis(prev)) i = prev;
        if (i < prev && val > IntensityThreshold(prev - 1) - IntensityHysteresis(prev - 1)) i = prev;
        return i;
    }

    /// <summary>libopus OPUS_RESET_STATE of a CELT encoder, on the device: the state view zeroed, then the non-zero resets.</summary>
    public static void ResetState(ArrayView<int> st, int stAt)
    {
        for (int i = 0; i < StateInts; i++) st[stAt + i] = 0;
        for (int i = 0; i < 2 * Nb; i++)
        {
            st[stAt + OldLogEAt + i] = -28672;   // -QCONST16(28.f, DB_SHIFT)
            st[stAt + OldLogE2At + i] = -28672;
        }
        st[stAt + SDelayedIntra] = 1;
        st[stAt + SSpread] = SpreadNormal;
        st[stAt + STonalAverage] = 256;
    }

    /// <summary>libopus compute_vbr without tonality analysis or a surround mask: the VBR target (1/8 bits).</summary>
    internal static int ComputeVbr(CeltModeGpu m, int base_target, int LM, int bitrate, int lastCodedBands, int C, int intensity,
        int constrained_vbr, int stereo_saving, int tot_boost, int tf_estimate, int maxDepth, int lfe, int temporal_vbr)
    {
        int coded_bands = lastCodedBands != 0 ? lastCodedBands : Nb;
        int coded_bins = m.EBandsOf(coded_bands) << LM;
        if (C == 2) coded_bins += m.EBandsOf(IMIN(intensity, coded_bands)) << LM;
        int target = base_target;
        // Stereo savings
        if (C == 2)
        {
            int coded_stereo_bands = IMIN(intensity, coded_bands);
            int coded_stereo_dof = (m.EBandsOf(coded_stereo_bands) << LM) - coded_stereo_bands;
            // Maximum fraction of the bits we can save if the signal is mono.
            int max_frac = DIV32_16(MULT16_16(26214, coded_stereo_dof), coded_bins);
            stereo_saving = S16(IMIN(stereo_saving, 256));
            target -= IMIN(MULT16_32_Q15(max_frac, target), MULT16_16(stereo_saving - 26, coded_stereo_dof << BITRES) >> 8);
        }
        // Boost the rate according to dynalloc (minus the dynalloc average for calibration).
        target += tot_boost - (19 << LM);
        // Apply transient boost, compensating for average boost.
        target += SHL32(MULT16_32_Q15(tf_estimate - 721, target), 1);   // QCONST16(0.044f, 14)
        {
            int bins = m.EBandsOf(Nb - 2) << LM;
            int floor_depth = MULT16_16(C * bins << BITRES, maxDepth) >> DB_SHIFT;
            floor_depth = IMAX(floor_depth, target >> 2);
            target = IMIN(target, floor_depth);
        }
        // Make VBR less aggressive for constrained VBR because we can't keep a higher bitrate for long.
        if (constrained_vbr != 0) target = base_target + MULT16_32_Q15(21954, target - base_target);   // QCONST16(0.67f, 15)
        if (tf_estimate < 3277)   // QCONST16(.2f, 14)
        {
            int amount = S16(MULT16_16_Q15(3329, IMAX(0, IMIN(32000, 96000 - bitrate))));   // QCONST16(.0000031f, 30)
            int tvbr_factor = S16(MULT16_16(temporal_vbr, amount) >> DB_SHIFT);
            target += MULT16_32_Q15(tvbr_factor, target);
        }
        // Don't allow more than doubling the rate
        return IMIN(2 * base_target, target);
    }

    /// <summary>Stage 1 of celt_encode_with_ec: the budget, silence, pre-emphasis, pitch pre-filter, transient analysis, MDCTs, band energies, normalisation, dynalloc and tf analysis.</summary>
    public static void EncodeAnalysis(CeltModeGpu mode, ArrayView<int> ctl, int ctlAt, ArrayView<short> pcm, ArrayView<int> st, int stAt,
        ArrayView<int> w, int wAt, ArrayView<byte> outv, ArrayView<int> len, ArrayView<uint> pvqU)
    {
        int pcmAt = ctl[ctlAt + (int)Ctl.PcmAt], frame_size = ctl[ctlAt + (int)Ctl.FrameSize], CC = ctl[ctlAt + (int)Ctl.CC];
        int nbCompressedBytes = ctl[ctlAt + (int)Ctl.NbCompressedBytes], bitrate = ctl[ctlAt + (int)Ctl.Bitrate];
        int vbr = ctl[ctlAt + (int)Ctl.Vbr], constrained_vbr = ctl[ctlAt + (int)Ctl.ConstrainedVbr], complexity = ctl[ctlAt + (int)Ctl.Complexity];
        int start = ctl[ctlAt + (int)Ctl.Start], end = ctl[ctlAt + (int)Ctl.End], C = ctl[ctlAt + (int)Ctl.C];
        int lsb_depth = ctl[ctlAt + (int)Ctl.LsbDepth], disable_pf = ctl[ctlAt + (int)Ctl.DisablePf], force_intra = ctl[ctlAt + (int)Ctl.ForceIntra];
        int lfe = ctl[ctlAt + (int)Ctl.Lfe], disable_inv = ctl[ctlAt + (int)Ctl.DisableInv], loss_rate = ctl[ctlAt + (int)Ctl.LossRate];
        int upsample = ctl[ctlAt + (int)Ctl.Upsample], outAt = ctl[ctlAt + (int)Ctl.OutAt], lengthAt = ctl[ctlAt + (int)Ctl.LengthAt];
        int s = stAt, R = wAt + WRes, scr = wAt + WScr;
        int V = wAt + WVars;
        for (int i = 0; i < 32; i++) w[R + i] = 0;
        int hybrid = start != 0 ? 1 : 0;
        int tf_estimate = 0;
        w[V + VValid] = 0;
        if (nbCompressedBytes < 2) { len[lengthAt] = -1; return; }
        frame_size *= upsample;
        int LM;
        for (LM = 0; LM <= 3; LM++)
            if (ShortMdctSize << LM == frame_size) break;
        if (LM > 3) { len[lengthAt] = -1; return; }
        int M = 1 << LM;
        int N = M * ShortMdctSize;
        int tell0_frac = 1, tell = 1, nbFilledBytes = 0;
        // Can't produce more than 1275 output bytes
        nbCompressedBytes = IMIN(nbCompressedBytes, 1275);
        int nbAvailableBytes = nbCompressedBytes - nbFilledBytes;
        int vbr_rate, effectiveBytes;
        if (vbr != 0 && bitrate != BitrateMax)
        {
            int den = 48000 >> BITRES;
            vbr_rate = (bitrate * frame_size + (den >> 1)) / den;
            effectiveBytes = vbr_rate >> (3 + BITRES);
        }
        else
        {
            vbr_rate = 0;
            int tmp = bitrate * frame_size;
            if (bitrate != BitrateMax) nbCompressedBytes = IMAX(2, IMIN(nbCompressedBytes, (tmp + 4 * 48000) / (8 * 48000)));
            effectiveBytes = nbCompressedBytes - nbFilledBytes;
        }
        int equiv_rate = (nbCompressedBytes * 8 * 50 << (3 - LM)) - (40 * C + 20) * ((400 >> LM) - 50);
        if (bitrate != BitrateMax) equiv_rate = IMIN(equiv_rate, bitrate - (40 * C + 20) * ((400 >> LM) - 50));
        var enc = OpusRangeEncoderGpu.Init();
        uint storage = (uint)nbCompressedBytes;
        if (vbr_rate > 0 && constrained_vbr != 0)
        {
            // The max bit-rate allowed in VBR mode, up front so that the bust-prevention logic triggers correctly.
            int vbr_bound = vbr_rate;
            int max_allowed = IMIN(IMAX(tell == 1 ? 2 : 0, (vbr_rate + vbr_bound - st[s + SVbrReservoir]) >> (BITRES + 3)), nbAvailableBytes);
            if (max_allowed < nbAvailableBytes)
            {
                nbCompressedBytes = nbFilledBytes + max_allowed;
                nbAvailableBytes = max_allowed;
                storage = OpusRangeEncoderGpu.Shrink(ref enc, outv, outAt, storage, (uint)nbCompressedBytes);
            }
        }
        int total_bits = nbCompressedBytes * 8;
        int effEnd = IMIN(end, Nb);
        int inAt = wAt + WIn;
        int sample_max = IMAX(st[s + SOverlapMax], MaxAbs16(pcm, pcmAt, C * (N - Overlap) / upsample));
        st[s + SOverlapMax] = MaxAbs16(pcm, pcmAt + C * (N - Overlap) / upsample, C * Overlap / upsample);
        sample_max = IMAX(sample_max, st[s + SOverlapMax]);
        int silence = sample_max == 0 ? 1 : 0;
        if (tell == 1) OpusRangeEncoderGpu.EncodeBitLogP(ref enc, outv, outAt, storage, silence, 15);
        else silence = 0;
        if (silence != 0)
        {
            // In VBR mode there is no need to send more than the minimum.
            if (vbr_rate > 0)
            {
                effectiveBytes = nbCompressedBytes = IMIN(nbCompressedBytes, nbFilledBytes + 2);
                total_bits = nbCompressedBytes * 8;
                nbAvailableBytes = 2;
                storage = OpusRangeEncoderGpu.Shrink(ref enc, outv, outAt, storage, (uint)nbCompressedBytes);
            }
            // Pretend we've filled all the remaining bits with zeros (that's what the initialiser did anyway)
            tell = nbCompressedBytes * 8;
            enc.NBitsTotal += tell - OpusRangeEncoderGpu.Tell(ref enc);
        }
        w[R + RSilence] = silence;
        for (int c = 0; c < CC; c++)
            CeltEncoderAnalysisGpu.Preemphasis(pcm, pcmAt + c, CC, w, inAt + c * (N + Overlap) + Overlap, N, upsample, st, s + SPreemph + c);

        // Find pitch period and gain
        int pitch_index = CeltPrefilterGpu.MinPeriod, gain1 = 0, qg = 0, pitch_change = 0;
        int prefilter_tapset = st[s + STapset];
        int pf_on;
        {
            int enabled = ((lfe != 0 && nbAvailableBytes > 3) || nbAvailableBytes > 12 * C) && hybrid == 0 && silence == 0 && disable_pf == 0
                && complexity >= 5 ? 1 : 0;
            int prefilterPeriod = st[s + SPfPeriod];
            int prevGain = st[s + SPfGain];
            pf_on = CeltPrefilterGpu.RunPrefilter(mode, w, inAt, st, s + PrefilterMemAt, s + InMemAt, CC, N, prefilter_tapset, ref pitch_index,
                ref gain1, ref qg, enabled, nbAvailableBytes, ref prefilterPeriod, prevGain, st[s + SPfTapset], loss_rate, w, scr);
            st[s + SPfPeriod] = prefilterPeriod;
            if ((gain1 > 13107 || prevGain > 13107)
                && ((double)pitch_index > 1.26 * prefilterPeriod || (double)pitch_index < .79 * prefilterPeriod))
                pitch_change = 1;
            if (pf_on == 0)
            {
                if (hybrid == 0 && tell + 16 <= total_bits) OpusRangeEncoderGpu.EncodeBitLogP(ref enc, outv, outAt, storage, 0, 1);
            }
            else
            {
                // This block is not gated by a total bits check only because of the nbAvailableBytes check above.
                OpusRangeEncoderGpu.EncodeBitLogP(ref enc, outv, outAt, storage, 1, 1);
                pitch_index += 1;
                int octave = CeltFixed.EC_ILOG((uint)pitch_index) - 5;
                OpusRangeEncoderGpu.EncodeUint(ref enc, outv, outAt, storage, (uint)octave, 6);
                OpusRangeEncoderGpu.EncodeBits(ref enc, outv, outAt, storage, (uint)(pitch_index - (16 << octave)), 4 + octave);
                pitch_index -= 1;
                OpusRangeEncoderGpu.EncodeBits(ref enc, outv, outAt, storage, (uint)qg, 3);
                CeltEnergyEncGpu.EncodeIcdfMode(ref enc, outv, outAt, storage, prefilter_tapset, mode, mode.TapsetIcdfAt, 2);
            }
        }
        w[R + RPfOn] = pf_on; w[R + RPitch] = pitch_index; w[R + RGain] = gain1; w[R + RQg] = qg;

        int isTransient = 0, shortBlocks = 0, tf_chan = 0, weak_transient = 0, transient_got_disabled = 0;
        if (complexity >= 1 && lfe == 0)
        {
            // allow_weak_transients needs hybrid mode
            isTransient = CeltEncoderAnalysisGpu.TransientAnalysis(mode, w, inAt, N + Overlap, CC, ref tf_estimate, ref tf_chan, 0,
                ref weak_transient, scr);
        }
        if (LM > 0 && OpusRangeEncoderGpu.Tell(ref enc) + 3 <= total_bits)
        {
            if (isTransient != 0) shortBlocks = M;
        }
        else
        {
            isTransient = 0;
            transient_got_disabled = 1;
        }
        int freqAt = wAt + WFreq, bandE = wAt + WBandE, bandLogE = wAt + WBandLogE, bandLogE2 = wAt + WBandLogE2;
        int secondMdct = shortBlocks != 0 && complexity >= 8 ? 1 : 0;
        int temporal_vbr = 0;
        // The MDCT passes through one call site (WGSL inlines each): the long-block MDCT of the second-MDCT analysis (into
        // bandLogE2), the frame's own, and the short-block one when the energies show a transient after all
        int patched = 0;
        for (int pass = 0; pass < 3; pass++)
        {
            if (pass == 0 && secondMdct == 0) continue;
            if (pass == 2 && patched == 0) break;
            int target = pass == 0 ? bandLogE2 : bandLogE;
            CeltEncoderAnalysisGpu.ComputeMdcts(mode, pass == 0 ? 0 : shortBlocks, w, inAt, freqAt, C, CC, LM, upsample, scr);
            if (pass == 1 && CC == 2 && C == 1) tf_chan = 0;
            CeltEncoderAnalysisGpu.ComputeBandEnergies(mode, w, freqAt, w, bandE, effEnd, C, LM);
            if (pass == 1 && lfe != 0)
            {
                for (int i = 2; i < end; i++)
                {
                    w[bandE + i] = IMIN(w[bandE + i], MULT16_32_Q15(3, w[bandE]));   // QCONST16(1e-4f, 15)
                    w[bandE + i] = IMAX(w[bandE + i], EPSILON);
                }
            }
            CeltEncoderAnalysisGpu.Amp2Log2(mode, effEnd, end, w, bandE, w, target, C);
            // Compensate for the scaling of short vs long mdcts (the second MDCT's long energies; the patched frame's bandLogE2)
            if (pass != 1)
                for (int c = 0; c < C; c++)
                    for (int i = 0; i < end; i++) w[bandLogE2 + Nb * c + i] = S16(w[bandLogE2 + Nb * c + i] + HALF16(SHL16(LM, DB_SHIFT)));
            if (pass == 2) tf_estimate = 3277;   // QCONST16(.2f, 14)
            if (pass == 1)
            {
                // Temporal VBR (but not for LFE)
                if (lfe == 0)
                {
                    int follow = -10240;   // -QCONST16(10.0f, DB_SHIFT)
                    int frame_avg = 0;
                    int offset = shortBlocks != 0 ? HALF16(SHL16(LM, DB_SHIFT)) : 0;
                    for (int i = start; i < end; i++)
                    {
                        follow = S16(IMAX(follow - 1024, w[bandLogE + i] - offset));
                        if (C == 2) follow = S16(IMAX(follow, w[bandLogE + i + Nb] - offset));
                        frame_avg += follow;
                    }
                    frame_avg /= end - start;
                    temporal_vbr = S16(SUB16(frame_avg, st[s + SSpecAvg]));
                    temporal_vbr = S16(IMIN(3072, IMAX(-1536, temporal_vbr)));
                    st[s + SSpecAvg] = S16(st[s + SSpecAvg] + MULT16_16_Q15(655, temporal_vbr));   // QCONST16(.02f, 15)
                }
                if (secondMdct == 0)
                    for (int i = 0; i < C * Nb; i++) w[bandLogE2 + i] = w[bandLogE + i];
                // Last chance to catch any transient we might have missed in the time-domain analysis
                if (LM > 0 && OpusRangeEncoderGpu.Tell(ref enc) + 3 <= total_bits && isTransient == 0 && complexity >= 5 && lfe == 0 && hybrid == 0
                    && CeltEncoderAnalysisGpu.PatchTransientDecision(w, bandLogE, st, s + OldBandEAt, start, end, C, w, scr) != 0)
                {
                    isTransient = 1;
                    shortBlocks = M;
                    patched = 1;
                }
            }
        }
        if (LM > 0 && OpusRangeEncoderGpu.Tell(ref enc) + 3 <= total_bits)
            OpusRangeEncoderGpu.EncodeBitLogP(ref enc, outv, outAt, storage, isTransient, 3);
        w[R + RTransient] = isTransient; w[R + RShortBlocks] = shortBlocks; w[R + RTfEstimate] = tf_estimate; w[R + RTfChan] = tf_chan;

        // Band normalisation
        int X = wAt + WX;
        CeltEncoderAnalysisGpu.NormaliseBands(mode, w, freqAt, w, X, w, bandE, effEnd, C, M);
        int enable_tf_analysis = effectiveBytes >= 15 * C && hybrid == 0 && complexity >= 2 && lfe == 0 ? 1 : 0;
        int offsets = wAt + WOffsets, importance = wAt + WImportance, spread_weight = wAt + WSpreadWeight, tf_res = wAt + WTfRes;
        int tot_boost = 0;
        int maxDepth = CeltEncoderAnalysisGpu.DynallocAnalysis(mode, w, bandLogE, w, bandLogE2, st, s + OldBandEAt, start, end, C, w, offsets,
            lsb_depth, isTransient, vbr, constrained_vbr, LM, effectiveBytes, ref tot_boost, lfe, w, importance, w, spread_weight, w, scr);
        w[R + RMaxDepth] = maxDepth; w[R + RTotBoost] = tot_boost;
        int tf_select;
        // Disable variable tf resolution for hybrid and at very low bitrate
        if (enable_tf_analysis != 0)
        {
            int lambda = IMAX(80, 20480 / effectiveBytes + 2);
            tf_select = CeltEncoderAnalysisGpu.TfAnalysis(mode, effEnd, isTransient, w, tf_res, lambda, w, X, N, LM, tf_estimate, tf_chan, w,
                importance, w, scr);
            for (int i = effEnd; i < end; i++) w[tf_res + i] = w[tf_res + effEnd - 1];
        }
        else
        {
            for (int i = 0; i < end; i++) w[tf_res + i] = isTransient;
            tf_select = 0;
        }
        w[R + RTfSelect] = tf_select;
        SaveEnc(w, V + VEnc, ref enc);
        w[V + VStorage] = (int)storage; w[V + VNb] = nbCompressedBytes; w[V + VNbAvail] = nbAvailableBytes; w[V + VTotalBits] = total_bits;
        w[V + VEffBytes] = effectiveBytes; w[V + VVbrRate] = vbr_rate; w[V + VEquivRate] = equiv_rate; w[V + VLM] = LM;
        w[V + VSilence] = silence; w[V + VPitch] = pitch_index; w[V + VGain1] = gain1; w[V + VTapset] = prefilter_tapset; w[V + VPfOn] = pf_on;
        w[V + VTransient] = isTransient; w[V + VShortBlocks] = shortBlocks; w[V + VTfEstimate] = tf_estimate;
        w[V + VTransDisabled] = transient_got_disabled; w[V + VTemporalVbr] = temporal_vbr; w[V + VMaxDepth] = maxDepth;
        w[V + VTotBoost] = tot_boost; w[V + VTfSelect] = tf_select; w[V + VValid] = 1;
    }

    /// <summary>Stage 2: coarse energy, tf, spreading, dynalloc boosts, stereo decisions, trim, VBR, the allocation and fine energy.</summary>
    public static void EncodeCoding(CeltModeGpu mode, ArrayView<int> ctl, int ctlAt, ArrayView<short> pcm, ArrayView<int> st, int stAt,
        ArrayView<int> w, int wAt, ArrayView<byte> outv, ArrayView<int> len, ArrayView<uint> pvqU)
    {
        int pcmAt = ctl[ctlAt + (int)Ctl.PcmAt], frame_size = ctl[ctlAt + (int)Ctl.FrameSize], CC = ctl[ctlAt + (int)Ctl.CC];
        int nbCompressedBytes = ctl[ctlAt + (int)Ctl.NbCompressedBytes], bitrate = ctl[ctlAt + (int)Ctl.Bitrate];
        int vbr = ctl[ctlAt + (int)Ctl.Vbr], constrained_vbr = ctl[ctlAt + (int)Ctl.ConstrainedVbr], complexity = ctl[ctlAt + (int)Ctl.Complexity];
        int start = ctl[ctlAt + (int)Ctl.Start], end = ctl[ctlAt + (int)Ctl.End], C = ctl[ctlAt + (int)Ctl.C];
        int lsb_depth = ctl[ctlAt + (int)Ctl.LsbDepth], disable_pf = ctl[ctlAt + (int)Ctl.DisablePf], force_intra = ctl[ctlAt + (int)Ctl.ForceIntra];
        int lfe = ctl[ctlAt + (int)Ctl.Lfe], disable_inv = ctl[ctlAt + (int)Ctl.DisableInv], loss_rate = ctl[ctlAt + (int)Ctl.LossRate];
        int upsample = ctl[ctlAt + (int)Ctl.Upsample], outAt = ctl[ctlAt + (int)Ctl.OutAt], lengthAt = ctl[ctlAt + (int)Ctl.LengthAt];
        int s = stAt, R = wAt + WRes, scr = wAt + WScr;
        int V = wAt + WVars;
        if (w[V + VValid] == 0) return;
        var enc = LoadEnc(w, V + VEnc);
        uint storage = (uint)w[V + VStorage];
        nbCompressedBytes = w[V + VNb];
        int LM = w[V + VLM];
        int M = 1 << LM, N = M * ShortMdctSize, effEnd = IMIN(end, Nb), hybrid = start != 0 ? 1 : 0;
        int X = wAt + WX, bandE = wAt + WBandE, bandLogE = wAt + WBandLogE, error = wAt + WError, offsets = wAt + WOffsets;
        int spread_weight = wAt + WSpreadWeight, tf_res = wAt + WTfRes, pulses = wAt + WPulses, fine_quant = wAt + WFineQuant;
        int fine_priority = wAt + WFinePriority;
        int nbAvailableBytes = w[V + VNbAvail], total_bits = w[V + VTotalBits], effectiveBytes = w[V + VEffBytes], vbr_rate = w[V + VVbrRate];
        int equiv_rate = w[V + VEquivRate], silence = w[V + VSilence], pf_on = w[V + VPfOn], isTransient = w[V + VTransient];
        int shortBlocks = w[V + VShortBlocks], tf_estimate = w[V + VTfEstimate], temporal_vbr = w[V + VTemporalVbr], maxDepth = w[V + VMaxDepth];
        int tot_boost = w[V + VTotBoost], tf_select = w[V + VTfSelect];
        int tell;
        for (int c = 0; c < C; c++)
        {
            for (int i = start; i < end; i++)
            {
                // When the energy is stable, slightly bias energy quantization towards the previous error to make the gain more stable
                if (ABS32(w[bandLogE + i + c * Nb] - st[s + OldBandEAt + i + c * Nb]) < 2048)   // QCONST16(2.f, DB_SHIFT)
                    w[bandLogE + i + c * Nb] = S16(w[bandLogE + i + c * Nb] - MULT16_16_Q15(st[s + EnergyErrorAt + i + c * Nb], 8192));
            }
        }
        int delayedIntra = st[s + SDelayedIntra];
        CeltEnergyEncGpu.QuantCoarseEnergy(mode, start, end, effEnd, w, bandLogE, st, s + OldBandEAt, total_bits, w, error, ref enc, outv, outAt,
            storage, C, LM, nbAvailableBytes, force_intra, ref delayedIntra, complexity >= 4 ? 1 : 0, loss_rate, lfe, w, scr);
        st[s + SDelayedIntra] = delayedIntra;
        w[R + RTellCoarse] = OpusRangeEncoderGpu.Tell(ref enc);
        CeltEncoderAnalysisGpu.TfEncode(mode, start, end, isTransient, w, tf_res, LM, tf_select, ref enc, outv, outAt, storage);

        if (OpusRangeEncoderGpu.Tell(ref enc) + 4 <= total_bits)
        {
            if (lfe != 0)
            {
                st[s + STapset] = 0;
                st[s + SSpread] = SpreadNormal;
            }
            else if (shortBlocks != 0 || complexity < 3 || nbAvailableBytes < 10 * C)
            {
                st[s + SSpread] = complexity == 0 ? SpreadNone : SpreadNormal;
            }
            else
            {
                int tonal_average = st[s + STonalAverage], hf_average = st[s + SHfAverage], tapset = st[s + STapset];
                st[s + SSpread] = CeltEncoderAnalysisGpu.SpreadingDecision(mode, w, X, ref tonal_average, st[s + SSpread], ref hf_average,
                    ref tapset, pf_on != 0 && shortBlocks == 0 ? 1 : 0, effEnd, C, M, w, spread_weight);
                st[s + STonalAverage] = tonal_average; st[s + SHfAverage] = hf_average; st[s + STapset] = tapset;
            }
            CeltEnergyEncGpu.EncodeIcdfMode(ref enc, outv, outAt, storage, st[s + SSpread], mode, mode.SpreadIcdfAt, 5);
        }
        w[R + RSpread] = st[s + SSpread];

        // For LFE, everything interesting is in the first band
        if (lfe != 0) w[offsets] = IMIN(8, effectiveBytes / 3);
        int cap = wAt + WCap;
        CeltRateGpu.InitCaps(mode, w, cap, LM, C);
        int dynalloc_logp = 6;
        total_bits <<= BITRES;
        int total_boost = 0;
        tell = (int)OpusRangeEncoderGpu.TellFrac(ref enc);
        for (int i = start; i < end; i++)
        {
            int width = C * (mode.EBandsOf(i + 1) - mode.EBandsOf(i)) << LM;
            // quanta is 6 bits, but no more than 1 bit/sample and no less than 1/8 bit/sample
            int quanta = IMIN(width << BITRES, IMAX(6 << BITRES, width));
            int dynalloc_loop_logp = dynalloc_logp;
            int boost = 0;
            int j;
            for (j = 0; tell + (dynalloc_loop_logp << BITRES) < total_bits - total_boost && boost < w[cap + i]; j++)
            {
                int flag = j < w[offsets + i] ? 1 : 0;
                OpusRangeEncoderGpu.EncodeBitLogP(ref enc, outv, outAt, storage, flag, dynalloc_loop_logp);
                tell = (int)OpusRangeEncoderGpu.TellFrac(ref enc);
                if (flag == 0) break;
                boost += quanta;
                total_boost += quanta;
                dynalloc_loop_logp = 1;
            }
            // Making dynalloc more likely
            if (j != 0) dynalloc_logp = IMAX(2, dynalloc_logp - 1);
            w[offsets + i] = boost;
        }
        w[R + RTotalBoost] = total_boost;
        int dual_stereo = 0;
        if (C == 2)
        {
            // Always use MS for 2.5 ms frames until we can do a better analysis
            if (LM != 0) dual_stereo = CeltEncoderAnalysisGpu.StereoAnalysis(mode, w, X, LM, N);
            st[s + SIntensity] = IntensityDecision(S16(equiv_rate / 1000), st[s + SIntensity]);
            st[s + SIntensity] = IMIN(end, IMAX(start, st[s + SIntensity]));
        }
        int alloc_trim = 5;
        if (tell + (6 << BITRES) <= total_bits - total_boost)
        {
            if (start > 0 || lfe != 0)
            {
                st[s + SStereoSaving] = 0;
                alloc_trim = 5;
            }
            else
            {
                int stereo_saving = st[s + SStereoSaving];
                alloc_trim = CeltEncoderAnalysisGpu.AllocTrimAnalysis(mode, w, X, w, bandLogE, end, LM, C, N, ref stereo_saving, tf_estimate,
                    st[s + SIntensity], equiv_rate);
                st[s + SStereoSaving] = stereo_saving;
            }
            CeltEnergyEncGpu.EncodeIcdfMode(ref enc, outv, outAt, storage, alloc_trim, mode, mode.TrimIcdfAt, 7);
            tell = (int)OpusRangeEncoderGpu.TellFrac(ref enc);
        }
        w[R + RTrim] = alloc_trim;

        // Variable bitrate
        if (vbr_rate > 0)
        {
            int lm_diff = 3 - LM;
            // Don't attempt to use more than 510 kb/s, even for frames smaller than 20 ms.
            nbCompressedBytes = IMIN(nbCompressedBytes, 1275 >> (3 - LM));
            int base_target = vbr_rate - ((40 * C + 20) << BITRES);
            if (constrained_vbr != 0) base_target += st[s + SVbrOffset] >> lm_diff;
            int target = ComputeVbr(mode, base_target, LM, equiv_rate, st[s + SLastCodedBands], C, st[s + SIntensity], constrained_vbr,
                st[s + SStereoSaving], tot_boost, tf_estimate, maxDepth, lfe, temporal_vbr);
            // The current offset is removed from the target and the space used so far is added
            target = target + tell;
            // The frame must not shrink so much that the encoder runs out of bits; 2 bytes of margin keep the decoder's
            // bust-prevention from triggering so far.
            int min_allowed = ((tell + total_boost + (1 << (BITRES + 3)) - 1) >> (BITRES + 3)) + 2;
            nbAvailableBytes = (target + (1 << (BITRES + 2))) >> (BITRES + 3);
            nbAvailableBytes = IMAX(min_allowed, nbAvailableBytes);
            nbAvailableBytes = IMIN(nbCompressedBytes, nbAvailableBytes);
            // By how much did we "miss" the target on that frame
            int delta = target - vbr_rate;
            target = nbAvailableBytes << (BITRES + 3);
            // A silent frame does not adjust the drift (the bit reservoir may refill)
            if (silence != 0)
            {
                nbAvailableBytes = 2;
                target = 2 * 8 << BITRES;
                delta = 0;
            }
            int alpha;
            if (st[s + SVbrCount] < 970)
            {
                st[s + SVbrCount]++;
                alpha = S16(celt_rcp(SHL32(st[s + SVbrCount] + 20, 16)));
            }
            else alpha = 33;   // QCONST16(.001f, 15)
            // How many bits have we used in excess of what we're allowed
            if (constrained_vbr != 0) st[s + SVbrReservoir] += target - vbr_rate;
            // The offset needed to reach the target
            if (constrained_vbr != 0)
            {
                st[s + SVbrDrift] += MULT16_32_Q15(alpha, (delta * (1 << lm_diff)) - st[s + SVbrOffset] - st[s + SVbrDrift]);
                st[s + SVbrOffset] = -st[s + SVbrDrift];
            }
            if (constrained_vbr != 0 && st[s + SVbrReservoir] < 0)
            {
                // We're under the min value -- increase rate, unless we're just coding silence
                int adjust = -st[s + SVbrReservoir] / (8 << BITRES);
                nbAvailableBytes += silence != 0 ? 0 : adjust;
                st[s + SVbrReservoir] = 0;
            }
            nbCompressedBytes = IMIN(nbCompressedBytes, nbAvailableBytes);
            // This moves the raw bits to take into account the new compressed size
            storage = OpusRangeEncoderGpu.Shrink(ref enc, outv, outAt, storage, (uint)nbCompressedBytes);
        }

        // Bit allocation: packet size - where we are - safety
        int bits = (nbCompressedBytes * 8 << BITRES) - (int)OpusRangeEncoderGpu.TellFrac(ref enc) - 1;
        int anti_collapse_rsv = isTransient != 0 && LM >= 2 && bits >= (LM + 2) << BITRES ? 1 << BITRES : 0;
        bits -= anti_collapse_rsv;
        int signalBandwidth = end - 1;
        if (lfe != 0) signalBandwidth = 1;
        w[R + RBits] = bits;
        int intensity = st[s + SIntensity];
        int codedBands = CeltRateEncGpu.ComputeAllocationEncode(mode, start, end, w.SubView(offsets, Nb), w.SubView(cap, Nb), alloc_trim,
            ref intensity, ref dual_stereo, bits, out int balance, w.SubView(pulses, Nb), w.SubView(fine_quant, Nb), w.SubView(fine_priority, Nb),
            0, w, scr, C, LM, ref enc, outv, outAt, storage, st[s + SLastCodedBands], signalBandwidth);
        st[s + SIntensity] = intensity;
        w[R + RTellAlloc] = (int)OpusRangeEncoderGpu.TellFrac(ref enc);
        w[R + RCodedBands] = codedBands; w[R + RBalance] = balance; w[R + RDual] = dual_stereo; w[R + RIntensity] = intensity;
        if (st[s + SLastCodedBands] != 0)
            st[s + SLastCodedBands] = IMIN(st[s + SLastCodedBands] + 1, IMAX(st[s + SLastCodedBands] - 1, codedBands));
        else st[s + SLastCodedBands] = codedBands;
        CeltEnergyEncGpu.QuantFineEnergy(start, end, st, s + OldBandEAt, w, error, w, fine_quant, ref enc, outv, outAt, storage, C);
        SaveEnc(w, V + VEnc, ref enc);
        w[V + VStorage] = (int)storage; w[V + VNb] = nbCompressedBytes; w[V + VDual] = dual_stereo; w[V + VIntensity] = intensity;
        w[V + VCoded] = codedBands; w[V + VBalance] = balance; w[V + VAntiCollapseRsv] = anti_collapse_rsv;
    }

    /// <summary>Stage 3: the band shapes (quant_all_bands).</summary>
    public static void EncodeBands(CeltModeGpu mode, ArrayView<int> ctl, int ctlAt, ArrayView<short> pcm, ArrayView<int> st, int stAt,
        ArrayView<int> w, int wAt, ArrayView<byte> outv, ArrayView<int> len, ArrayView<uint> pvqU)
    {
        int pcmAt = ctl[ctlAt + (int)Ctl.PcmAt], frame_size = ctl[ctlAt + (int)Ctl.FrameSize], CC = ctl[ctlAt + (int)Ctl.CC];
        int nbCompressedBytes = ctl[ctlAt + (int)Ctl.NbCompressedBytes], bitrate = ctl[ctlAt + (int)Ctl.Bitrate];
        int vbr = ctl[ctlAt + (int)Ctl.Vbr], constrained_vbr = ctl[ctlAt + (int)Ctl.ConstrainedVbr], complexity = ctl[ctlAt + (int)Ctl.Complexity];
        int start = ctl[ctlAt + (int)Ctl.Start], end = ctl[ctlAt + (int)Ctl.End], C = ctl[ctlAt + (int)Ctl.C];
        int lsb_depth = ctl[ctlAt + (int)Ctl.LsbDepth], disable_pf = ctl[ctlAt + (int)Ctl.DisablePf], force_intra = ctl[ctlAt + (int)Ctl.ForceIntra];
        int lfe = ctl[ctlAt + (int)Ctl.Lfe], disable_inv = ctl[ctlAt + (int)Ctl.DisableInv], loss_rate = ctl[ctlAt + (int)Ctl.LossRate];
        int upsample = ctl[ctlAt + (int)Ctl.Upsample], outAt = ctl[ctlAt + (int)Ctl.OutAt], lengthAt = ctl[ctlAt + (int)Ctl.LengthAt];
        int s = stAt, R = wAt + WRes, scr = wAt + WScr;
        int V = wAt + WVars;
        if (w[V + VValid] == 0) return;
        var enc = LoadEnc(w, V + VEnc);
        uint storage = (uint)w[V + VStorage];
        nbCompressedBytes = w[V + VNb];
        int LM = w[V + VLM];
        int M = 1 << LM, N = M * ShortMdctSize, effEnd = IMIN(end, Nb), hybrid = start != 0 ? 1 : 0;
        int X = wAt + WX, bandE = wAt + WBandE, bandLogE = wAt + WBandLogE, error = wAt + WError, offsets = wAt + WOffsets;
        int spread_weight = wAt + WSpreadWeight, tf_res = wAt + WTfRes, pulses = wAt + WPulses, fine_quant = wAt + WFineQuant;
        int fine_priority = wAt + WFinePriority;
        int shortBlocks = w[V + VShortBlocks], dual_stereo = w[V + VDual], intensity = w[V + VIntensity], codedBands = w[V + VCoded];
        int balance = w[V + VBalance], anti_collapse_rsv = w[V + VAntiCollapseRsv];
        // Residual quantisation (libopus resynthesises the bands only for theta RDO: stereo, no dual stereo, complexity >= 8)
        {
            uint seed = (uint)st[s + SRng];
            CeltBandsEncGpu.QuantAllBands(mode, start, end, w, X, C == 2 ? X + N : -1, w, bandE, w, pulses, shortBlocks, st[s + SSpread],
                dual_stereo, intensity, w, tf_res, nbCompressedBytes * (8 << BITRES) - anti_collapse_rsv, balance, ref enc, outv, outAt,
                storage, LM, codedBands, ref seed, complexity, disable_inv, scr, pvqU);
        }
        SaveEnc(w, V + VEnc, ref enc);
    }

    /// <summary>Stage 4: anti-collapse, the final energy bits, the state update and ec_enc_done; writes the byte count.</summary>
    public static void EncodeFinish(CeltModeGpu mode, ArrayView<int> ctl, int ctlAt, ArrayView<short> pcm, ArrayView<int> st, int stAt,
        ArrayView<int> w, int wAt, ArrayView<byte> outv, ArrayView<int> len, ArrayView<uint> pvqU)
    {
        int pcmAt = ctl[ctlAt + (int)Ctl.PcmAt], frame_size = ctl[ctlAt + (int)Ctl.FrameSize], CC = ctl[ctlAt + (int)Ctl.CC];
        int nbCompressedBytes = ctl[ctlAt + (int)Ctl.NbCompressedBytes], bitrate = ctl[ctlAt + (int)Ctl.Bitrate];
        int vbr = ctl[ctlAt + (int)Ctl.Vbr], constrained_vbr = ctl[ctlAt + (int)Ctl.ConstrainedVbr], complexity = ctl[ctlAt + (int)Ctl.Complexity];
        int start = ctl[ctlAt + (int)Ctl.Start], end = ctl[ctlAt + (int)Ctl.End], C = ctl[ctlAt + (int)Ctl.C];
        int lsb_depth = ctl[ctlAt + (int)Ctl.LsbDepth], disable_pf = ctl[ctlAt + (int)Ctl.DisablePf], force_intra = ctl[ctlAt + (int)Ctl.ForceIntra];
        int lfe = ctl[ctlAt + (int)Ctl.Lfe], disable_inv = ctl[ctlAt + (int)Ctl.DisableInv], loss_rate = ctl[ctlAt + (int)Ctl.LossRate];
        int upsample = ctl[ctlAt + (int)Ctl.Upsample], outAt = ctl[ctlAt + (int)Ctl.OutAt], lengthAt = ctl[ctlAt + (int)Ctl.LengthAt];
        int s = stAt, R = wAt + WRes, scr = wAt + WScr;
        int V = wAt + WVars;
        if (w[V + VValid] == 0) return;
        var enc = LoadEnc(w, V + VEnc);
        uint storage = (uint)w[V + VStorage];
        nbCompressedBytes = w[V + VNb];
        int LM = w[V + VLM];
        int M = 1 << LM, N = M * ShortMdctSize, effEnd = IMIN(end, Nb), hybrid = start != 0 ? 1 : 0;
        int X = wAt + WX, bandE = wAt + WBandE, bandLogE = wAt + WBandLogE, error = wAt + WError, offsets = wAt + WOffsets;
        int spread_weight = wAt + WSpreadWeight, tf_res = wAt + WTfRes, pulses = wAt + WPulses, fine_quant = wAt + WFineQuant;
        int fine_priority = wAt + WFinePriority;
        int anti_collapse_rsv = w[V + VAntiCollapseRsv], silence = w[V + VSilence], pitch_index = w[V + VPitch], gain1 = w[V + VGain1];
        int prefilter_tapset = w[V + VTapset], isTransient = w[V + VTransient], transient_got_disabled = w[V + VTransDisabled];
        int anti_collapse_on = 0;
        if (anti_collapse_rsv > 0)
        {
            anti_collapse_on = st[s + SConsecTransient] < 2 ? 1 : 0;
            OpusRangeEncoderGpu.EncodeBits(ref enc, outv, outAt, storage, (uint)anti_collapse_on, 1);
        }
        w[R + RAntiCollapse] = anti_collapse_on;
        CeltEnergyEncGpu.QuantEnergyFinalise(start, end, st, s + OldBandEAt, w, error, w, fine_quant, w, fine_priority,
            nbCompressedBytes * 8 - OpusRangeEncoderGpu.Tell(ref enc), ref enc, outv, outAt, storage, C);
        for (int i = 0; i < CC * Nb; i++) st[s + EnergyErrorAt + i] = 0;
        for (int c = 0; c < C; c++)
            for (int i = start; i < end; i++)
                st[s + EnergyErrorAt + i + c * Nb] = S16(IMAX(-16384, IMIN(16384, w[error + i + c * Nb])));
        if (silence != 0)
            for (int i = 0; i < C * Nb; i++) st[s + OldBandEAt + i] = -28672;

        st[s + SPfPeriod] = pitch_index;
        st[s + SPfGain] = gain1;
        st[s + SPfTapset] = prefilter_tapset;
        if (CC == 2 && C == 1)
            for (int i = 0; i < Nb; i++) st[s + OldBandEAt + Nb + i] = st[s + OldBandEAt + i];
        if (isTransient == 0)
        {
            for (int i = 0; i < CC * Nb; i++)
            {
                st[s + OldLogE2At + i] = st[s + OldLogEAt + i];
                st[s + OldLogEAt + i] = st[s + OldBandEAt + i];
            }
        }
        else
        {
            for (int i = 0; i < CC * Nb; i++) st[s + OldLogEAt + i] = IMIN(st[s + OldLogEAt + i], st[s + OldBandEAt + i]);
        }
        // In case start or end were to change
        for (int c = 0; c < CC; c++)
        {
            for (int i = 0; i < start; i++)
            {
                st[s + OldBandEAt + c * Nb + i] = 0;
                st[s + OldLogEAt + c * Nb + i] = st[s + OldLogE2At + c * Nb + i] = -28672;
            }
            for (int i = end; i < Nb; i++)
            {
                st[s + OldBandEAt + c * Nb + i] = 0;
                st[s + OldLogEAt + c * Nb + i] = st[s + OldLogE2At + c * Nb + i] = -28672;
            }
        }
        if (isTransient != 0 || transient_got_disabled != 0) st[s + SConsecTransient]++;
        else st[s + SConsecTransient] = 0;
        st[s + SRng] = (int)enc.Rng;
        // If there's any room left (can only happen for very high rates), it's already filled with zeros
        OpusRangeEncoderGpu.Done(ref enc, outv, outAt, storage);
        w[R + RBytes] = nbCompressedBytes;
        w[R + RError] = enc.Error;
        len[lengthAt] = enc.Error != 0 ? -1 : nbCompressedBytes;
    }

    /// <summary>
    /// libopus 1.5.2 celt_encode_with_ec (enc == NULL): codes one frame of the CC channels' interleaved int16 input at
    /// ctl[PcmAt] into the out view at ctl[OutAt] and writes the byte count (or -1 on an internal error) to len[ctl[LengthAt]].
    /// <paramref name="st"/> holds the encoder state at <paramref name="stAt"/> (<see cref="StateInts"/>, set up by
    /// <see cref="ResetState"/>), <paramref name="w"/> the work at <paramref name="wAt"/> (<see cref="WorkInts"/>). The four
    /// stages in a row; <see cref="AnalysisKernel"/> .. <see cref="FinishKernel"/> run them as separate kernels (each a
    /// WGSL shader of its own size).
    /// </summary>
    public static void EncodeFrame(CeltModeGpu mode, ArrayView<int> ctl, int ctlAt, ArrayView<short> pcm, ArrayView<int> st, int stAt,
        ArrayView<int> w, int wAt, ArrayView<byte> outv, ArrayView<int> len, ArrayView<uint> pvqU)
    {
        EncodeAnalysis(mode, ctl, ctlAt, pcm, st, stAt, w, wAt, outv, len, pvqU);
        EncodeCoding(mode, ctl, ctlAt, pcm, st, stAt, w, wAt, outv, len, pvqU);
        EncodeBands(mode, ctl, ctlAt, pcm, st, stAt, w, wAt, outv, len, pvqU);
        EncodeFinish(mode, ctl, ctlAt, pcm, st, stAt, w, wAt, outv, len, pvqU);
    }

    /// <summary>One celt_encode_with_ec call per thread: thread i codes the call whose CTLs are at ctl[i * Ctl.Count], with the
    /// encoder state at st[i * StateInts] and work at w[i * WorkInts] (independent encoders, one call each).</summary>
    public static void EncodeKernel(Index1D i, ArrayView<int> ctl, ArrayView<short> pcm, ArrayView<int> st, ArrayView<int> w,
        ArrayView<byte> outv, ArrayView<int> len, CeltModeGpu mode, ArrayView<uint> pvqU)
    {
        EncodeFrame(mode, ctl, i * (int)Ctl.Count, pcm, st, i * StateInts, w, i * WorkInts, outv, len, pvqU);
    }

    /// <summary>Stage kernels: launch AnalysisKernel, CodingKernel, BandsKernel, FinishKernel in order (same arguments as
    /// <see cref="EncodeKernel"/>) to code the calls the way EncodeKernel does, each stage its own shader.</summary>
    public static void AnalysisKernel(Index1D i, ArrayView<int> ctl, ArrayView<short> pcm, ArrayView<int> st, ArrayView<int> w,
        ArrayView<byte> outv, ArrayView<int> len, CeltModeGpu mode, ArrayView<uint> pvqU) =>
        EncodeAnalysis(mode, ctl, i * (int)Ctl.Count, pcm, st, i * StateInts, w, i * WorkInts, outv, len, pvqU);

    /// <summary>Stage 2 kernel (see <see cref="AnalysisKernel"/>).</summary>
    public static void CodingKernel(Index1D i, ArrayView<int> ctl, ArrayView<short> pcm, ArrayView<int> st, ArrayView<int> w,
        ArrayView<byte> outv, ArrayView<int> len, CeltModeGpu mode, ArrayView<uint> pvqU) =>
        EncodeCoding(mode, ctl, i * (int)Ctl.Count, pcm, st, i * StateInts, w, i * WorkInts, outv, len, pvqU);

    /// <summary>Stage 3 kernel (see <see cref="AnalysisKernel"/>).</summary>
    public static void BandsKernel(Index1D i, ArrayView<int> ctl, ArrayView<short> pcm, ArrayView<int> st, ArrayView<int> w,
        ArrayView<byte> outv, ArrayView<int> len, CeltModeGpu mode, ArrayView<uint> pvqU) =>
        EncodeBands(mode, ctl, i * (int)Ctl.Count, pcm, st, i * StateInts, w, i * WorkInts, outv, len, pvqU);

    /// <summary>Stage 4 kernel (see <see cref="AnalysisKernel"/>).</summary>
    public static void FinishKernel(Index1D i, ArrayView<int> ctl, ArrayView<short> pcm, ArrayView<int> st, ArrayView<int> w,
        ArrayView<byte> outv, ArrayView<int> len, CeltModeGpu mode, ArrayView<uint> pvqU) =>
        EncodeFinish(mode, ctl, i * (int)Ctl.Count, pcm, st, i * StateInts, w, i * WorkInts, outv, len, pvqU);

    /// <summary>Resets the encoder states (one per thread) on the device.</summary>
    public static void ResetKernel(Index1D i, ArrayView<int> st) => ResetState(st, i * StateInts);
}
