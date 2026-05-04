// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable port of libopus silk/decode_core.c (CPU mirror:
// SilkDecodeCore.Decode). Drives the SILK synthesis chain for one frame:
// excitation dequantization -> per-subframe LTP rewhitening / state update
// -> 5-tap LTP prediction -> LPC synthesis -> int16 PCM output. Updates
// the persistent channel-decoder state in place.
//
// Composes existing GPU primitives:
//   - SilkExcitationDequantizerGpu.DequantizeAt (per-frame)
//   - SilkGainAdjustGpu.ApplyAt (per-subframe gain ratio)
//   - SilkLpcAnalysisFilterGpu.ApplyAt (per-sample, called inside
//     rewhitening loop)
//   - SilkLpcSynthesisFilterGpu.ApplyAt (per-subframe synthesis)
//
// Sequential per stream: the per-sample LTP loop has a recurrence on
// sLtpQ15 and the LPC synthesis is recursive. One thread per stream;
// multi-channel decode parallelizes across threads.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Body-struct kernel parameter for SilkDecodeCoreGpu callers. Bundles
/// every per-frame ArrayView the synthesis chain needs (parameters,
/// pulses, persistent state, scratch buffers, output PCM). Pairs with
/// <see cref="SilkDecodeCoreScalars"/>.
/// </summary>
public struct SilkDecodeCoreInputs
{
    /// <summary>Predictor coefficients Q12 (2 halves x MAX_LPC_ORDER, length 32).</summary>
    public ArrayView<short> PredCoefQ12;
    /// <summary>Per-subframe gains Q16 (length nbSubfr).</summary>
    public ArrayView<int> GainsQ16;
    /// <summary>Per-subframe pitch lag (length nbSubfr; ignored if unvoiced).</summary>
    public ArrayView<int> PitchL;
    /// <summary>Per-subframe LTP filter taps Q14 (length nbSubfr*5; ignored if unvoiced).</summary>
    public ArrayView<short> LtpCoefQ14;
    /// <summary>Decoded pulse magnitudes (length frameLength).</summary>
    public ArrayView<short> Pulses;

    // ---- Persistent channel-decoder state (in/out) ----
    /// <summary>LTP output buffer, length MAX_LTP_MEM_LENGTH + MAX_FRAME_LENGTH = 640.
    /// History at [0..ltpMemLength); kernel writes the staged xqOut into the tail
    /// when k == 2 rewhitens.</summary>
    public ArrayView<short> OutBufInOut;
    /// <summary>LPC filter history Q14 (length MAX_LPC_ORDER = 16). Kernel reads at
    /// the start of frame and writes back at end.</summary>
    public ArrayView<int> SLpcQ14BufInOut;
    /// <summary>Excitation buffer Q14 (length MAX_FRAME_LENGTH = 320). Written by
    /// step 1 (dequant) and read by step 3 (per-subframe loop).</summary>
    public ArrayView<int> ExcQ14Out;
    /// <summary>Single-int previous-frame gain Q16. Kernel reads at start, then
    /// updates after each subframe via SilkGainAdjustGpu, and the final value is
    /// the last subframe's gain.</summary>
    public ArrayView<int> PrevGainQ16InOut;

    // ---- Scratch buffers (caller pre-allocates; contents undefined on entry) ----
    /// <summary>LPC synthesis state scratch, length MAX_LPC_ORDER + MAX_SUB_FRAME_LENGTH = 96.
    /// Layout [history (16) | output samples (subfrLength up to 80)].</summary>
    public ArrayView<int> SLpcScratch;
    /// <summary>LTP Q15 scratch, length MAX_LTP_MEM_LENGTH + MAX_FRAME_LENGTH = 640.</summary>
    public ArrayView<int> SLtpQ15Scratch;
    /// <summary>LTP Q-untouched scratch (rewhitened LPC residual), length MAX_LTP_MEM_LENGTH = 320.</summary>
    public ArrayView<short> SLtpScratch;
    /// <summary>Per-subframe residual scratch presQ14, length MAX_SUB_FRAME_LENGTH = 80.</summary>
    public ArrayView<int> PresQ14Scratch;
    /// <summary>Single-int output for SilkGainAdjustGpu.</summary>
    public ArrayView<int> GainAdjScratch;

    // ---- Output ----
    /// <summary>Output PCM (int16), length frameLength.</summary>
    public ArrayView<short> XqOut;
}

/// <summary>
/// Scalar kernel parameter for SilkDecodeCoreGpu. Holds frame geometry +
/// per-frame configuration that the kernel branches on. Pairs with
/// <see cref="SilkDecodeCoreInputs"/>.
/// </summary>
public struct SilkDecodeCoreScalars
{
    /// <summary>SILK signal type (0 inactive, 1 unvoiced, 2 voiced).</summary>
    public int SignalType;
    /// <summary>SILK quantizer offset type (0 or 1).</summary>
    public int QuantOffsetType;
    /// <summary>Excitation PRNG seed (from decoded indices).</summary>
    public int Seed;
    /// <summary>LPC filter order (10 for NB/MB, 16 for WB).</summary>
    public int LpcOrder;
    /// <summary>Subframe count (2 or 4).</summary>
    public int NbSubfr;
    /// <summary>Subframe length in samples.</summary>
    public int SubfrLength;
    /// <summary>Frame length in samples (= NbSubfr * SubfrLength).</summary>
    public int FrameLength;
    /// <summary>LTP buffer length in samples.</summary>
    public int LtpMemLength;
    /// <summary>LTP scale factor Q14 (one of 15565 / 12288 / 8192). 0 if unvoiced.</summary>
    public int LtpScaleQ14;
    /// <summary>1 if NLSF interpolation was active for this frame
    /// (gates the k==2 LTP rewhitening), 0 otherwise.</summary>
    public int NlsfInterpEnabled;
}

/// <summary>
/// GPU-callable orchestrator for the SILK synthesis chain. Mirror of
/// <c>SilkDecodeCore.Decode</c>.
/// </summary>
public static class SilkDecodeCoreGpu
{
    /// <summary>Libopus LTP_ORDER = 5.</summary>
    public const int LtpOrder = 5;
    /// <summary>Libopus MAX_LPC_ORDER = 16.</summary>
    public const int MaxLpcOrder = 16;
    /// <summary>Libopus TYPE_VOICED = 2.</summary>
    public const int TypeVoiced = 2;

    /// <summary>
    /// Run the SILK synthesis chain for one frame on the GPU. Writes
    /// <c>inputs.XqOut[0..frameLength)</c>, updates <c>inputs.SLpcQ14BufInOut</c>
    /// + <c>inputs.PrevGainQ16InOut</c> + <c>inputs.ExcQ14Out</c>.
    /// </summary>
    public static void Decode(SilkDecodeCoreInputs inputs, SilkDecodeCoreScalars scalars)
    {
        int frameLength = scalars.FrameLength;
        int subfrLength = scalars.SubfrLength;
        int ltpMemLength = scalars.LtpMemLength;
        int lpcOrder = scalars.LpcOrder;
        int nbSubfr = scalars.NbSubfr;
        int signalType = scalars.SignalType;
        int ltpScaleQ14 = scalars.LtpScaleQ14;
        bool nlsfInterpEnabled = scalars.NlsfInterpEnabled != 0;

        // ---- Step 1: excitation dequantization (writes to inputs.ExcQ14Out). ----
        SilkExcitationDequantizerGpu.DequantizeAt(
            inputs.ExcQ14Out, 0,
            inputs.Pulses, 0,
            signalType, scalars.QuantOffsetType, scalars.Seed, frameLength);

        // ---- Step 2: scratch buffer setup. Copy 16-entry LPC history into
        //              SLpcScratch[0..16). The remaining slots [16..96) are
        //              the synthesis output area, written by ApplyAt below. ----
        for (int i = 0; i < MaxLpcOrder; i++)
            inputs.SLpcScratch[i] = inputs.SLpcQ14BufInOut[i];

        long sLtpBufIdx = ltpMemLength;
        int prevGain = inputs.PrevGainQ16InOut[0];

        // ---- Step 3: per-subframe loop. ----
        long pexcOff = 0;
        long pxqOff = 0;
        for (int k = 0; k < nbSubfr; k++)
        {
            long aBase = (long)(k >> 1) * MaxLpcOrder;       // PredCoefQ12 half offset
            long bBase = (long)k * LtpOrder;                  // LtpCoefQ14 offset

            int gainQ16 = inputs.GainsQ16[k];
            int gainQ10 = gainQ16 >> 6;
            int invGainQ31 = SilkInverse32VarQ47(gainQ16);

            // ---- Gain-adjust the LPC state buffer; returns Q16 gain ratio in scratch slot 0. ----
            SilkGainAdjustGpu.ApplyAt(
                inputs.SLpcScratch, 0,
                prevGain, gainQ16,
                inputs.GainAdjScratch, 0);
            int gainAdjQ16 = inputs.GainAdjScratch[0];
            prevGain = gainQ16;

            int lag = 0;

            if (signalType == TypeVoiced)
            {
                lag = inputs.PitchL[k];
                bool rewhitenSubframe = (k == 0) || (k == 2 && nlsfInterpEnabled);

                if (rewhitenSubframe)
                {
                    // Rewhitening: apply LPC analysis filter to previous output buffer
                    // contents, writing to SLtpScratch[startIdx..]. Result is the
                    // unscaled LTP state for this lag.
                    int startIdx = ltpMemLength - lag - lpcOrder - (LtpOrder >> 1);

                    if (k == 2)
                    {
                        // Stage first 2 subframes of the current xq output into
                        // OutBuf's tail so the analysis filter can read them as
                        // prediction history.
                        for (int i = 0; i < 2 * subfrLength; i++)
                            inputs.OutBufInOut[ltpMemLength + i] = inputs.XqOut[i];
                    }

                    // Pre-zero SLtpScratch[startIdx..startIdx+lpcOrder) (analysis filter
                    // requires this prelude to be zero before it writes from ix=lpcOrder).
                    for (int i = 0; i < lpcOrder; i++)
                        inputs.SLtpScratch[startIdx + i] = 0;

                    // Per-sample analysis filter loop, ix in [lpcOrder, ltpMemLength - startIdx).
                    long inBase = (long)startIdx + (long)k * subfrLength;
                    int filterLen = ltpMemLength - startIdx;
                    for (int ix = lpcOrder; ix < filterLen; ix++)
                    {
                        SilkLpcAnalysisFilterGpu.ApplyAt(
                            inputs.OutBufInOut, inBase,
                            inputs.PredCoefQ12, aBase,
                            inputs.SLtpScratch, startIdx,
                            lpcOrder, ix);
                    }

                    // Scale sLTP into sLtpQ15 with inv_gain (with LTP_scale adj on k==0).
                    int invGainForScale = invGainQ31;
                    if (k == 0)
                    {
                        // silk_LSHIFT(silk_SMULWB(invGainQ31, ltpScaleQ14), 2)
                        int smulwb = (int)((long)invGainQ31 * (short)ltpScaleQ14 >> 16);
                        invGainForScale = smulwb << 2;
                    }

                    int numTaps = lag + (LtpOrder >> 1);
                    for (int i = 0; i < numTaps; i++)
                    {
                        int sLtpVal = inputs.SLtpScratch[ltpMemLength - i - 1];
                        // silk_SMULWB(invGainForScale, sLtpVal): pulses are (short) by spec.
                        int scaled = (int)((long)invGainForScale * (short)sLtpVal >> 16);
                        inputs.SLtpQ15Scratch[sLtpBufIdx - i - 1] = scaled;
                    }
                }
                else if (gainAdjQ16 != (1 << 16))
                {
                    // Non-rewhitening subframe with gain change: scale existing LTP state.
                    int numTaps = lag + (LtpOrder >> 1);
                    for (int i = 0; i < numTaps; i++)
                    {
                        int sLtpQ15Val = inputs.SLtpQ15Scratch[sLtpBufIdx - i - 1];
                        inputs.SLtpQ15Scratch[sLtpBufIdx - i - 1] = SmulWW(gainAdjQ16, sLtpQ15Val);
                    }
                }
            }

            // ---- Compute residual presQ14. ----
            if (signalType == TypeVoiced)
            {
                // 5-tap LTP prediction + residual + sLtpQ15 state update.
                long predLagPtrOff = sLtpBufIdx - lag + (LtpOrder >> 1);

                short b0 = inputs.LtpCoefQ14[bBase + 0];
                short b1 = inputs.LtpCoefQ14[bBase + 1];
                short b2 = inputs.LtpCoefQ14[bBase + 2];
                short b3 = inputs.LtpCoefQ14[bBase + 3];
                short b4 = inputs.LtpCoefQ14[bBase + 4];

                for (int i = 0; i < subfrLength; i++)
                {
                    int ltpPredQ13 = 2;
                    ltpPredQ13 = SmlaWB(ltpPredQ13, inputs.SLtpQ15Scratch[predLagPtrOff + 0], b0);
                    ltpPredQ13 = SmlaWB(ltpPredQ13, inputs.SLtpQ15Scratch[predLagPtrOff - 1], b1);
                    ltpPredQ13 = SmlaWB(ltpPredQ13, inputs.SLtpQ15Scratch[predLagPtrOff - 2], b2);
                    ltpPredQ13 = SmlaWB(ltpPredQ13, inputs.SLtpQ15Scratch[predLagPtrOff - 3], b3);
                    ltpPredQ13 = SmlaWB(ltpPredQ13, inputs.SLtpQ15Scratch[predLagPtrOff - 4], b4);
                    predLagPtrOff++;

                    // silk_ADD_LSHIFT32(exc, ltpPred, 1) = exc + (ltpPred << 1).
                    int pres = inputs.ExcQ14Out[pexcOff + i] + (ltpPredQ13 << 1);
                    inputs.PresQ14Scratch[i] = pres;

                    // sLtpQ15[sLtpBufIdx] = silk_LSHIFT(pres, 1). Then sLtpBufIdx++.
                    inputs.SLtpQ15Scratch[sLtpBufIdx] = pres << 1;
                    sLtpBufIdx++;
                }
            }
            else
            {
                // Unvoiced / inactive: residual IS the excitation, no LTP prediction.
                for (int i = 0; i < subfrLength; i++)
                    inputs.PresQ14Scratch[i] = inputs.ExcQ14Out[pexcOff + i];
            }

            // ---- LPC synthesis: presQ14 -> PCM via LPC synthesis filter. ----
            SilkLpcSynthesisFilterGpu.ApplyAt(
                inputs.SLpcScratch, 0,
                inputs.PresQ14Scratch, 0,
                inputs.PredCoefQ12, aBase,
                gainQ10, lpcOrder, subfrLength,
                inputs.XqOut, pxqOff);

            // Slide the LPC state: trailing MAX_LPC_ORDER samples become the
            // history for the next subframe.
            for (int i = 0; i < MaxLpcOrder; i++)
                inputs.SLpcScratch[i] = inputs.SLpcScratch[subfrLength + i];

            pexcOff += subfrLength;
            pxqOff += subfrLength;
        }

        // ---- Step 4: save the LPC filter state for the next frame. ----
        for (int i = 0; i < MaxLpcOrder; i++)
            inputs.SLpcQ14BufInOut[i] = inputs.SLpcScratch[i];

        // Save updated PrevGainQ16 (= last subframe's gain).
        inputs.PrevGainQ16InOut[0] = prevGain;
    }

    // ---- Inline silk math (kept private + duplicated to avoid taking external deps;
    //      mirrors SilkMacros / SilkLpcSynthesisFilterGpu.SmulWW). ----

    /// <summary>silk_SMLAWB(c, a, b) = c + (int)((long)a * (short)b >> 16).</summary>
    private static int SmlaWB(int c32, int a32, short b16) =>
        c32 + (int)((long)a32 * b16 >> 16);

    /// <summary>silk_SMULWW(a, b) = SMULWB(a, b) + a * RSHIFT_ROUND(b, 16).</summary>
    private static int SmulWW(int a32, int b32)
    {
        int smulwb = (int)((long)a32 * (short)b32 >> 16);
        int rshiftRound = (b32 + (1 << 15)) >> 16;
        return smulwb + a32 * rshiftRound;
    }

    /// <summary>
    /// silk_INVERSE32_varQ(b32, 47). Mirrors SilkMacros.silk_INVERSE32_varQ
    /// for Qres == 47 (the only Qres used by silk_decode_core). Uses a
    /// portable bit-by-bit CLZ to stay GPU-callable across every backend.
    /// </summary>
    private static int SilkInverse32VarQ47(int b32)
    {
        // SILK gains are positive Q16 (well below INT_MAX). Compute
        // bHeadrm = silk_CLZ32(|b32|) - 1 = number of leading zeros above the top set bit.
        int absB = b32 < 0 ? -b32 : b32;
        int bHeadrm = -1;
        for (uint v = (uint)absB; v != 0u && (v & 0x80000000u) == 0u; v <<= 1)
            bHeadrm++;
        if (bHeadrm < 0) bHeadrm = 0; // defensive (should not occur for SILK gains)

        int b32Nrm = b32 << bHeadrm;

        // silk_DIV32_16(silk_int32_MAX >> 2, b32Nrm >> 16): int32 / int16 division.
        int b32NrmHi = (b32Nrm >> 16);
        int b32Inv = (int.MaxValue >> 2) / b32NrmHi;

        int result = b32Inv << 16;

        // err_Q32 = LSHIFT((1 << 29) - SMULWB(b32Nrm, b32Inv), 3).
        int smulwb = (int)((long)b32Nrm * (short)b32Inv >> 16);
        int errQ32 = ((1 << 29) - smulwb) << 3;

        // result = SMLAWW(result, err_Q32, b32Inv).
        // SMLAWW(a, b, c) = a + SMULWW(b, c).
        int smulww = SmulWW(errQ32, b32Inv);
        result += smulww;

        // Final shift to land at Qres == 47.
        // libopus: lshift = 61 - bHeadrm - Qres (= 14 - bHeadrm).
        int lshift = 61 - bHeadrm - 47;
        if (lshift <= 0)
        {
            // Saturating left shift.
            int shamt = -lshift;
            if (shamt >= 32) return result == 0 ? 0 : (result > 0 ? int.MaxValue : int.MinValue);
            int max = int.MaxValue >> shamt;
            int min = int.MinValue >> shamt;
            if (result > max) return int.MaxValue;
            if (result < min) return int.MinValue;
            return result << shamt;
        }
        if (lshift < 32) return result >> lshift;
        return 0;
    }
}
