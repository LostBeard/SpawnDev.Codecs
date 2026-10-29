// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// SILK packet-loss concealment and comfort noise on the accelerator: libopus silk/PLC.c (silk_PLC: the state update on
// good frames, silk_PLC_conceal on lost ones, silk_PLC_glue_frames) and silk/CNG.c (silk_CNG), statement for statement.
// This is libopus 1.5.2 (the CNG signal in Q14 with saturating updates, the gain applied at the output, the faster
// smoothed-gain adaptation, the saturating LPC update in the PLC synthesis). libopus silk_CNG also copies the frame's strongest subframe excitation into CNG_exc_buf_Q14 after the memmove;
// Concentus 2.2.2 omits that memcpy, so its comfort noise differs from libopus. This follows libopus.
//
// One channel's state lives in the decoder's packed buffers at the offsets in SilkPlcChannel: the PLC / CNG ints and
// shorts, plus views of the channel's decode state (outBuf, sLPC_Q14_buf, exc_Q14, prevNLSF_Q15) and shared scratch.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

using ILGPU;
using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>Where one channel's PLC / CNG state, decode state and the shared scratch live.</summary>
public struct SilkPlcChannel
{
    /// <summary>PLC / CNG ints (<see cref="SilkPlcGpu.IntsPerChannel"/>) in the core int buffer.</summary>
    public int IntsAt;
    /// <summary>PLC / CNG shorts (<see cref="SilkPlcGpu.ShortsPerChannel"/>) in the core short buffer.</summary>
    public int ShortsAt;
    /// <summary>The channel's outBuf (core shorts).</summary>
    public int OutBufAt;
    /// <summary>The channel's sLPC_Q14_buf (core ints).</summary>
    public int SLpcAt;
    /// <summary>The channel's exc_Q14 (core ints).</summary>
    public int ExcAt;
    /// <summary>The channel's prevNLSF_Q15 (parameter shorts).</summary>
    public int PrevNlsfAt;
    /// <summary>silk_LSFCosTab_FIX_Q12 (parameter shorts).</summary>
    public int LsfCosAt;
    /// <summary>Scratch shorts: sLTP (MAX_LTP_MEM_LENGTH) then exc_buf (2 * MAX_SUB_FRAME_LENGTH).</summary>
    public int ScratchShortsAt;
    /// <summary>Scratch ints: sLTP_Q14 (MAX_LTP_MEM_LENGTH + MAX_FRAME_LENGTH), CNG_sig_Q10, NLSF2A and inverse-gain scratch.</summary>
    public int ScratchIntsAt;
}

/// <summary>GPU-callable SILK PLC / CNG (libopus silk/PLC.c, silk/CNG.c).</summary>
public static class SilkPlcGpu
{
    // libopus constants (silk/define.h, silk/PLC.h, silk/tuning_parameters.h).
    private const int LTP_ORDER = 5;
    private const int RAND_BUF_SIZE = 128;
    private const int RAND_BUF_MASK = RAND_BUF_SIZE - 1;
    private const int V_PITCH_GAIN_START_MIN_Q14 = 11469;
    private const int V_PITCH_GAIN_START_MAX_Q14 = 15565;
    private const int BWE_COEF_Q16 = 64881;               // SILK_FIX_CONST(0.99, 16)
    private const int LOG2_INV_LPC_GAIN_HIGH_THRES = 3;
    private const int LOG2_INV_LPC_GAIN_LOW_THRES = 8;
    private const int PITCH_DRIFT_FAC_Q16 = 655;
    private const int MAX_PITCH_LAG_MS = 18;
    private const int CNG_BUF_MASK_MAX = 255;
    private const int CNG_GAIN_SMTH_Q16 = 4634;
    private const int CNG_GAIN_SMTH_THRESHOLD_Q16 = 46396;
    private const int CNG_NLSF_SMTH_Q16 = 16348;
    private const int MAX_LPC_ORDER = 16;
    private const int MAX_SUB_FRAME_LENGTH = 80;

    // ---- per-channel ints ----
    private const int IPitchLQ8 = 0, ILastFrameLost = 1, IRandSeed = 2, IRandScaleQ14 = 3, IConcEnergy = 4, IConcEnergyShift = 5,
        IPrevLtpScaleQ14 = 6, IPrevGain0 = 7, IPrevGain1 = 8, IFsKHz = 9, INbSubfr = 10, ISubfrLength = 11,
        ICngFsKHz = 12, ICngSmthGainQ16 = 13, ICngRandSeed = 14, ICngSynthState = 16, ICngExcBuf = ICngSynthState + MAX_LPC_ORDER;
    /// <summary>Ints of PLC / CNG state per channel.</summary>
    public const int IntsPerChannel = ICngExcBuf + SilkConstants.MAX_FRAME_LENGTH; // CNG_exc_buf_Q14[MAX_FRAME_LENGTH]

    // ---- per-channel shorts ----
    private const int SLtpCoefQ14 = 0, SPrevLpcQ12 = 8, SCngSmthNlsfQ15 = SPrevLpcQ12 + MAX_LPC_ORDER, SCngAQ12 = SCngSmthNlsfQ15 + MAX_LPC_ORDER;
    /// <summary>Shorts of PLC / CNG state per channel.</summary>
    public const int ShortsPerChannel = SCngAQ12 + MAX_LPC_ORDER;

    /// <summary>Scratch shorts shared by the channels.</summary>
    public const int ScratchShorts = SilkConstants.MAX_LTP_MEM_LENGTH + 2 * MAX_SUB_FRAME_LENGTH;
    private const int ScratchSLtpQ14 = 0;
    private const int ScratchCngSig = ScratchSLtpQ14 + SilkConstants.MAX_LTP_MEM_LENGTH + SilkConstants.MAX_FRAME_LENGTH;
    private const int ScratchNlsf2a = ScratchCngSig + SilkConstants.MAX_FRAME_LENGTH + MAX_LPC_ORDER;
    private const int ScratchInvGain = ScratchNlsf2a + 64;
    /// <summary>Scratch ints shared by the channels.</summary>
    public const int ScratchInts = ScratchInvGain + MAX_LPC_ORDER;

    /// <summary>silk_init_decoder's PLC / CNG part: everything zero, prevGain_Q16 = 1.0, CNG seed 3176576; the fs_kHz
    /// mismatch then runs silk_PLC_Reset / silk_CNG_Reset at the first use, as libopus does.</summary>
    public static void Reset(SilkPlcChannel c, ArrayView<short> cs, ArrayView<int> ci)
    {
        for (int k = 0; k < IntsPerChannel; k++) ci[c.IntsAt + k] = 0;
        for (int k = 0; k < ShortsPerChannel; k++) cs[c.ShortsAt + k] = 0;
        ci[c.IntsAt + IPrevGain0] = 65536;
        ci[c.IntsAt + IPrevGain1] = 65536;
        ci[c.IntsAt + ISubfrLength] = 20;
        ci[c.IntsAt + INbSubfr] = 2;
        ci[c.IntsAt + ICngRandSeed] = 3176576;
    }

    // silk_PLC_Reset.
    private static void PlcReset(SilkPlcChannel c, ArrayView<int> ci, int frameLength)
    {
        ci[c.IntsAt + IPitchLQ8] = frameLength << (8 - 1);
        ci[c.IntsAt + IPrevGain0] = 65536;
        ci[c.IntsAt + IPrevGain1] = 65536;
        ci[c.IntsAt + ISubfrLength] = 20;
        ci[c.IntsAt + INbSubfr] = 2;
    }

    /// <summary>
    /// silk_PLC with lost = 0 (silk_PLC_update) for a decoded frame: the parameters come from the parameter decoder's
    /// outputs (<paramref name="ps"/> / <paramref name="pi"/> per SilkDecodedParametersLayout).
    /// </summary>
    public static void Update(SilkPlcChannel c, ArrayView<short> cs, ArrayView<int> ci, ArrayView<short> ps, ArrayView<int> pi,
        int fsKHz, int frameLength, int nbSubfr, int subfrLength, int lpcOrder, int signalType)
    {
        if (fsKHz != ci[c.IntsAt + IFsKHz])
        {
            PlcReset(c, ci, frameLength);
            ci[c.IntsAt + IFsKHz] = fsKHz;
        }
        int ltpCoef = c.ShortsAt + SLtpCoefQ14;
        int ltpGainQ14 = 0;
        if (signalType == SilkConstants.TYPE_VOICED)
        {
            int lastPitchL = pi[SilkDecodedParametersLayout.IntPitchLOffset + nbSubfr - 1];
            for (int j = 0; j * subfrLength < lastPitchL; j++)
            {
                if (j == nbSubfr) break;
                int temp = 0;
                for (int i = 0; i < LTP_ORDER; i++) temp += ps[SilkDecodedParametersLayout.ShortLtpCoefQ14Offset + (nbSubfr - 1 - j) * LTP_ORDER + i];
                if (temp > ltpGainQ14)
                {
                    ltpGainQ14 = temp;
                    for (int i = 0; i < LTP_ORDER; i++)
                        cs[ltpCoef + i] = ps[SilkDecodedParametersLayout.ShortLtpCoefQ14Offset + (nbSubfr - 1 - j) * LTP_ORDER + i];
                    ci[c.IntsAt + IPitchLQ8] = pi[SilkDecodedParametersLayout.IntPitchLOffset + nbSubfr - 1 - j] << 8;
                }
            }
            for (int i = 0; i < LTP_ORDER; i++) cs[ltpCoef + i] = 0;
            cs[ltpCoef + LTP_ORDER / 2] = (short)ltpGainQ14;
            if (ltpGainQ14 < V_PITCH_GAIN_START_MIN_Q14)
            {
                int scaleQ10 = silk_DIV32(V_PITCH_GAIN_START_MIN_Q14 << 10, silk_max_32(ltpGainQ14, 1));
                for (int i = 0; i < LTP_ORDER; i++) cs[ltpCoef + i] = (short)silk_RSHIFT(silk_SMULBB(cs[ltpCoef + i], scaleQ10), 10);
            }
            else if (ltpGainQ14 > V_PITCH_GAIN_START_MAX_Q14)
            {
                int scaleQ14 = silk_DIV32(V_PITCH_GAIN_START_MAX_Q14 << 14, silk_max_32(ltpGainQ14, 1));
                for (int i = 0; i < LTP_ORDER; i++) cs[ltpCoef + i] = (short)silk_RSHIFT(silk_SMULBB(cs[ltpCoef + i], scaleQ14), 14);
            }
        }
        else
        {
            ci[c.IntsAt + IPitchLQ8] = silk_SMULBB(fsKHz, 18) << 8;
            for (int i = 0; i < LTP_ORDER; i++) cs[ltpCoef + i] = 0;
        }
        for (int i = 0; i < lpcOrder; i++)
            cs[c.ShortsAt + SPrevLpcQ12 + i] = ps[SilkDecodedParametersLayout.ShortPredCoefQ12Half2Offset + i];
        ci[c.IntsAt + IPrevLtpScaleQ14] = ps[SilkDecodedParametersLayout.ShortLtpScaleQ14Offset];
        ci[c.IntsAt + IPrevGain0] = pi[SilkDecodedParametersLayout.IntGainsQ16Offset + nbSubfr - 2];
        ci[c.IntsAt + IPrevGain1] = pi[SilkDecodedParametersLayout.IntGainsQ16Offset + nbSubfr - 1];
        ci[c.IntsAt + ISubfrLength] = subfrLength;
        ci[c.IntsAt + INbSubfr] = nbSubfr;
    }

    /// <summary>
    /// silk_PLC with lost = 1 (silk_PLC_conceal): extrapolates <paramref name="frameLength"/> samples into
    /// <paramref name="frame"/> at <paramref name="frameAt"/>, updates sLPC_Q14_buf and the PLC state. Returns the
    /// final pitch lag (psDecCtrl->pitchL[*]). The caller increments lossCnt afterwards.
    /// </summary>
    public static int Conceal(SilkPlcChannel c, ArrayView<short> cs, ArrayView<int> ci, ArrayView<short> frame, int frameAt,
        int fsKHz, int frameLength, int nbSubfr, int subfrLength, int ltpMemLength, int lpcOrder,
        int lossCnt, int prevSignalType, int firstFrameAfterReset)
    {
        if (fsKHz != ci[c.IntsAt + IFsKHz])
        {
            PlcReset(c, ci, frameLength);
            ci[c.IntsAt + IFsKHz] = fsKHz;
        }
        int sLtp = c.ScratchShortsAt;                                       // shorts, ltp_mem_length
        int excBuf = c.ScratchShortsAt + SilkConstants.MAX_LTP_MEM_LENGTH;  // shorts, 2 * subfr_length
        int sLtpQ14 = c.ScratchIntsAt + ScratchSLtpQ14;                     // ints, ltp_mem_length + frame_length
        int prevLpc = c.ShortsAt + SPrevLpcQ12;
        int bQ14 = c.ShortsAt + SLtpCoefQ14;

        int prevGainQ10_0 = ci[c.IntsAt + IPrevGain0] >> 6;
        int prevGainQ10_1 = ci[c.IntsAt + IPrevGain1] >> 6;
        if (firstFrameAfterReset != 0)
            for (int i = 0; i < MAX_LPC_ORDER; i++) cs[prevLpc + i] = 0;

        // silk_PLC_energy: the last two subframes' excitation scaled by their gains, energy of each.
        for (int k = 0; k < 2; k++)
        {
            int g = k == 0 ? prevGainQ10_0 : prevGainQ10_1;
            for (int i = 0; i < subfrLength; i++)
                cs[excBuf + k * subfrLength + i] = (short)silk_SAT16(silk_RSHIFT(silk_SMULWW(ci[c.ExcAt + i + (k + nbSubfr - 2) * subfrLength], g), 8));
        }
        SilkSumSqrShiftGpu.Compute(cs, excBuf, subfrLength, out int energy1, out int shift1);
        SilkSumSqrShiftGpu.Compute(cs, excBuf + subfrLength, subfrLength, out int energy2, out int shift2);

        int plcNbSubfr = ci[c.IntsAt + INbSubfr], plcSubfrLength = ci[c.IntsAt + ISubfrLength];
        int randPtr;
        if (silk_RSHIFT(energy1, shift2) < silk_RSHIFT(energy2, shift1))
            randPtr = silk_max_int(0, (plcNbSubfr - 1) * plcSubfrLength - RAND_BUF_SIZE);
        else
            randPtr = silk_max_int(0, plcNbSubfr * plcSubfrLength - RAND_BUF_SIZE);

        int randScaleQ14 = (short)ci[c.IntsAt + IRandScaleQ14];
        int attIdx = silk_min_int(1, lossCnt);
        int harmGainQ15 = attIdx == 0 ? 32440 : 31130;
        int randGainQ15 = prevSignalType == SilkConstants.TYPE_VOICED
            ? (attIdx == 0 ? 31130 : 26214)
            : (attIdx == 0 ? 32440 : 29491);

        SilkBwexpanderGpu.Expand16(cs, prevLpc, lpcOrder, BWE_COEF_Q16);

        if (lossCnt == 0)
        {
            randScaleQ14 = 1 << 14;
            if (prevSignalType == SilkConstants.TYPE_VOICED)
            {
                for (int i = 0; i < LTP_ORDER; i++) randScaleQ14 = (short)(randScaleQ14 - cs[bQ14 + i]);
                randScaleQ14 = randScaleQ14 > 3277 ? randScaleQ14 : 3277;
                randScaleQ14 = (short)silk_RSHIFT(silk_SMULBB(randScaleQ14, (short)ci[c.IntsAt + IPrevLtpScaleQ14]), 14);
            }
            else
            {
                int invGainQ30 = SilkLpcInvPredGainGpu.Compute(cs, prevLpc, lpcOrder, ci, c.ScratchIntsAt + ScratchInvGain);
                int downScaleQ30 = silk_min_32((1 << 30) >> LOG2_INV_LPC_GAIN_HIGH_THRES, invGainQ30);
                downScaleQ30 = silk_max_32((1 << 30) >> LOG2_INV_LPC_GAIN_LOW_THRES, downScaleQ30);
                downScaleQ30 = downScaleQ30 << LOG2_INV_LPC_GAIN_HIGH_THRES;
                randGainQ15 = silk_RSHIFT(silk_SMULWB(downScaleQ30, randGainQ15), 14);
            }
        }

        int randSeed = ci[c.IntsAt + IRandSeed];
        int lag = silk_RSHIFT_ROUND(ci[c.IntsAt + IPitchLQ8], 8);
        int sLtpBufIdx = ltpMemLength;

        // Rewhiten the LTP state.
        int idx = ltpMemLength - lag - lpcOrder - LTP_ORDER / 2;
        for (int i = 0; i < lpcOrder; i++) cs[sLtp + idx + i] = 0;
        for (int ix = lpcOrder; ix < ltpMemLength - idx; ix++)
            SilkLpcAnalysisFilterGpu.ApplyAt(cs, c.OutBufAt + idx, cs, prevLpc, cs, sLtp + idx, lpcOrder, ix);
        int invGainQ30b = silk_INVERSE32_varQ(ci[c.IntsAt + IPrevGain1], 46);
        invGainQ30b = silk_min(invGainQ30b, int.MaxValue >> 1);
        for (int i = idx + lpcOrder; i < ltpMemLength; i++) ci[sLtpQ14 + i] = silk_SMULWB(invGainQ30b, cs[sLtp + i]);

        // LTP synthesis.
        for (int k = 0; k < nbSubfr; k++)
        {
            int predLagPtr = sLtpBufIdx - lag + LTP_ORDER / 2;
            for (int i = 0; i < subfrLength; i++)
            {
                int ltpPredQ12 = 2;
                ltpPredQ12 = silk_SMLAWB(ltpPredQ12, ci[sLtpQ14 + predLagPtr], cs[bQ14 + 0]);
                ltpPredQ12 = silk_SMLAWB(ltpPredQ12, ci[sLtpQ14 + predLagPtr - 1], cs[bQ14 + 1]);
                ltpPredQ12 = silk_SMLAWB(ltpPredQ12, ci[sLtpQ14 + predLagPtr - 2], cs[bQ14 + 2]);
                ltpPredQ12 = silk_SMLAWB(ltpPredQ12, ci[sLtpQ14 + predLagPtr - 3], cs[bQ14 + 3]);
                ltpPredQ12 = silk_SMLAWB(ltpPredQ12, ci[sLtpQ14 + predLagPtr - 4], cs[bQ14 + 4]);
                predLagPtr++;
                randSeed = silk_RAND(randSeed);
                int r = silk_RSHIFT(randSeed, 25) & RAND_BUF_MASK;
                ci[sLtpQ14 + sLtpBufIdx] = silk_LSHIFT32(silk_SMLAWB(ltpPredQ12, ci[c.ExcAt + randPtr + r], randScaleQ14), 2);
                sLtpBufIdx++;
            }
            for (int j = 0; j < LTP_ORDER; j++) cs[bQ14 + j] = (short)silk_RSHIFT(silk_SMULBB(harmGainQ15, cs[bQ14 + j]), 15);
            randScaleQ14 = (short)silk_RSHIFT(silk_SMULBB(randScaleQ14, randGainQ15), 15);
            int pitchLQ8 = ci[c.IntsAt + IPitchLQ8];
            pitchLQ8 = silk_SMLAWB(pitchLQ8, pitchLQ8, PITCH_DRIFT_FAC_Q16);
            pitchLQ8 = silk_min_32(pitchLQ8, silk_SMULBB(MAX_PITCH_LAG_MS, fsKHz) << 8);
            ci[c.IntsAt + IPitchLQ8] = pitchLQ8;
            lag = silk_RSHIFT_ROUND(pitchLQ8, 8);
        }

        // LPC synthesis.
        int sLpcPtr = ltpMemLength - MAX_LPC_ORDER;
        for (int i = 0; i < MAX_LPC_ORDER; i++) ci[sLtpQ14 + sLpcPtr + i] = ci[c.SLpcAt + i];
        for (int i = 0; i < frameLength; i++)
        {
            int maxi = sLtpQ14 + sLpcPtr + MAX_LPC_ORDER + i;
            int lpcPredQ10 = silk_RSHIFT(lpcOrder, 1);
            for (int j = 0; j < lpcOrder; j++) lpcPredQ10 = silk_SMLAWB(lpcPredQ10, ci[maxi - j - 1], cs[prevLpc + j]);
            ci[maxi] = silk_ADD_SAT32(ci[maxi], silk_LSHIFT_SAT32(lpcPredQ10, 4));
            frame[frameAt + i] = (short)silk_SAT16(silk_SAT16(silk_RSHIFT_ROUND(silk_SMULWW(ci[maxi], prevGainQ10_1), 8)));
        }
        for (int i = 0; i < MAX_LPC_ORDER; i++) ci[c.SLpcAt + i] = ci[sLtpQ14 + sLpcPtr + frameLength + i];

        ci[c.IntsAt + IRandSeed] = randSeed;
        ci[c.IntsAt + IRandScaleQ14] = (short)randScaleQ14;
        return lag;
    }

    /// <summary>
    /// silk_CNG on a frame of <paramref name="length"/> samples at <paramref name="frameAt"/>: on a decoded frame of
    /// no voice activity it updates the smoothed NLSFs, gains and excitation buffer (<paramref name="ps"/> /
    /// <paramref name="pi"/> = the parameter decoder's outputs); on a lost frame it adds comfort noise.
    /// </summary>
    public static void Cng(SilkPlcChannel c, ArrayView<short> cs, ArrayView<int> ci, ArrayView<short> paramsShorts,
        ArrayView<int> pi, ArrayView<short> frame, int frameAt,
        int fsKHz, int length, int nbSubfr, int subfrLength, int lpcOrder, int lossCnt, int prevSignalType)
    {
        int smthNlsf = c.ShortsAt + SCngSmthNlsfQ15;
        if (fsKHz != ci[c.IntsAt + ICngFsKHz])
        {
            // silk_CNG_Reset.
            int stepQ15 = silk_DIV32_16(short.MaxValue, lpcOrder + 1);
            int accQ15 = 0;
            for (int i = 0; i < lpcOrder; i++)
            {
                accQ15 += stepQ15;
                cs[smthNlsf + i] = (short)accQ15;
            }
            ci[c.IntsAt + ICngSmthGainQ16] = 0;
            ci[c.IntsAt + ICngRandSeed] = 3176576;
            ci[c.IntsAt + ICngFsKHz] = fsKHz;
        }

        if (lossCnt == 0 && prevSignalType == SilkConstants.TYPE_NO_VOICE_ACTIVITY)
        {
            for (int i = 0; i < lpcOrder; i++)
                cs[smthNlsf + i] = (short)(cs[smthNlsf + i] + silk_SMULWB(paramsShorts[c.PrevNlsfAt + i] - cs[smthNlsf + i], CNG_NLSF_SMTH_Q16));
            int maxGainQ16 = 0, subfr = 0;
            for (int i = 0; i < nbSubfr; i++)
            {
                int g = pi[SilkDecodedParametersLayout.IntGainsQ16Offset + i];
                if (g > maxGainQ16) { maxGainQ16 = g; subfr = i; }
            }
            int exc = c.IntsAt + ICngExcBuf;
            // memmove(&buf[subfr_length], buf, (nb_subfr - 1) * subfr_length): overlapping, copy backwards.
            for (int i = (nbSubfr - 1) * subfrLength - 1; i >= 0; i--) ci[exc + subfrLength + i] = ci[exc + i];
            for (int i = 0; i < subfrLength; i++) ci[exc + i] = ci[c.ExcAt + subfr * subfrLength + i];
            for (int i = 0; i < nbSubfr; i++)
            {
                int g = pi[SilkDecodedParametersLayout.IntGainsQ16Offset + i];
                int s = ci[c.IntsAt + ICngSmthGainQ16];
                s += silk_SMULWB(g - s, CNG_GAIN_SMTH_Q16);
                // libopus 1.3+: a smoothed gain 3 dB above this subframe's gain snaps to it (faster adaptation).
                if (silk_SMULWW(s, CNG_GAIN_SMTH_THRESHOLD_Q16) > g) s = g;
                ci[c.IntsAt + ICngSmthGainQ16] = s;
            }
        }

        int synth = c.IntsAt + ICngSynthState;
        if (lossCnt != 0)
        {
            int sig = c.ScratchIntsAt + ScratchCngSig;
            int smthGain = ci[c.IntsAt + ICngSmthGainQ16];
            int gainQ16 = silk_SMULWW((short)ci[c.IntsAt + IRandScaleQ14], ci[c.IntsAt + IPrevGain1]);
            if (gainQ16 >= (1 << 21) || smthGain > (1 << 23))
            {
                gainQ16 = SmulTT(gainQ16, gainQ16);
                gainQ16 = silk_SUB_LSHIFT32(SmulTT(smthGain, smthGain), gainQ16, 5);
                gainQ16 = silk_LSHIFT32(SqrtApprox(gainQ16), 16);
            }
            else
            {
                gainQ16 = silk_SMULWW(gainQ16, gainQ16);
                gainQ16 = silk_SUB_LSHIFT32(silk_SMULWW(smthGain, smthGain), gainQ16, 5);
                gainQ16 = silk_LSHIFT32(SqrtApprox(gainQ16), 8);
            }
            int gainQ10 = silk_RSHIFT(gainQ16, 6);
            // silk_CNG_exc (libopus 1.3+: the raw Q14 buffer samples; the gain is applied at the output).
            int excMask = CNG_BUF_MASK_MAX;
            while (excMask > length) excMask >>= 1;
            int seed = ci[c.IntsAt + ICngRandSeed];
            for (int i = 0; i < length; i++)
            {
                seed = silk_RAND(seed);
                int r = silk_RSHIFT(seed, 24) & excMask;
                ci[sig + MAX_LPC_ORDER + i] = ci[c.IntsAt + ICngExcBuf + r];
            }
            ci[c.IntsAt + ICngRandSeed] = seed;

            int aQ12 = c.ShortsAt + SCngAQ12;
            SilkNlsf2AGpu.ComputeAt(cs, aQ12, cs, smthNlsf, paramsShorts, c.LsfCosAt, ci, c.ScratchIntsAt + ScratchNlsf2a, lpcOrder);
            for (int i = 0; i < MAX_LPC_ORDER; i++) ci[sig + i] = ci[synth + i];
            for (int i = 0; i < length; i++)
            {
                int lpci = sig + MAX_LPC_ORDER + i;
                int sumQ6 = silk_RSHIFT(lpcOrder, 1);
                for (int j = 0; j < lpcOrder; j++) sumQ6 = silk_SMLAWB(sumQ6, ci[lpci - j - 1], cs[aQ12 + j]);
                ci[lpci] = silk_ADD_SAT32(ci[lpci], silk_LSHIFT_SAT32(sumQ6, 4));
                frame[frameAt + i] = silk_ADD_SAT16(frame[frameAt + i], silk_SAT16(silk_RSHIFT_ROUND(silk_SMULWW(ci[lpci], gainQ10), 8)));
            }
            for (int i = 0; i < MAX_LPC_ORDER; i++) ci[synth + i] = ci[sig + length + i];
        }
        else
        {
            for (int i = 0; i < lpcOrder; i++) ci[synth + i] = 0;
        }
    }

    /// <summary>silk_PLC_glue_frames on the frame of <paramref name="length"/> samples at <paramref name="frameAt"/>.</summary>
    public static void Glue(SilkPlcChannel c, ArrayView<int> ci, ArrayView<short> frame, int frameAt, int length, int lossCnt)
    {
        if (lossCnt != 0)
        {
            SilkSumSqrShiftGpu.Compute(frame, frameAt, length, out int e, out int s);
            ci[c.IntsAt + IConcEnergy] = e;
            ci[c.IntsAt + IConcEnergyShift] = s;
            ci[c.IntsAt + ILastFrameLost] = 1;
        }
        else
        {
            if (ci[c.IntsAt + ILastFrameLost] != 0)
            {
                SilkSumSqrShiftGpu.Compute(frame, frameAt, length, out int energy, out int energyShift);
                int concEnergy = ci[c.IntsAt + IConcEnergy];
                int concShift = ci[c.IntsAt + IConcEnergyShift];
                if (energyShift > concShift) concEnergy = silk_RSHIFT(concEnergy, energyShift - concShift);
                else if (energyShift < concShift) energy = silk_RSHIFT(energy, concShift - energyShift);
                if (energy > concEnergy)
                {
                    int lz = silk_CLZ32(concEnergy) - 1;
                    concEnergy = concEnergy << lz;
                    energy = silk_RSHIFT(energy, silk_max_32(24 - lz, 0));
                    int fracQ24 = silk_DIV32(concEnergy, silk_max_32(energy, 1));
                    int gainQ16 = SqrtApprox(fracQ24) << 4;
                    int slopeQ16 = silk_DIV32_16((1 << 16) - gainQ16, length);
                    slopeQ16 = slopeQ16 << 2;
                    for (int i = frameAt; i < frameAt + length; i++)
                    {
                        frame[i] = (short)silk_SMULWB(gainQ16, frame[i]);
                        gainQ16 += slopeQ16;
                        if (gainQ16 > 1 << 16) break;
                    }
                }
                ci[c.IntsAt + IConcEnergy] = concEnergy;
            }
            ci[c.IntsAt + ILastFrameLost] = 0;
        }
    }

    // silk_SMULTT(a, b) = (a >> 16) * (b >> 16).
    private static int SmulTT(int a, int b) => (a >> 16) * (b >> 16);

    // silk_SQRT_APPROX.
    private static int SqrtApprox(int x)
    {
        if (x <= 0) return 0;
        silk_CLZ_FRAC(x, out int lz, out int fracQ7);
        int y = (lz & 1) != 0 ? 32768 : 46214;
        y >>= silk_RSHIFT(lz, 1);
        y = silk_SMLAWB(y, y, silk_SMULBB(213, fracQ7));
        return y;
    }
}
