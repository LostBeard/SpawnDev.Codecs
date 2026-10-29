// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// The SILK resampler (libopus silk/resampler.c + resampler_private_*.c) as GPU code: silk_resampler on one thread,
// statement for statement with the CPU SilkResampler.Apply in SpawnDev.Codecs.References, built on the per-stage
// GPU helpers (SilkResamplerUp2HqGpu, SilkResamplerIirFirInterpolGpu, SilkResamplerAr2Gpu,
// SilkResamplerDownFirInterpolGpu). The configuration (silk_resampler_init's rate-pair decisions) is host metadata
// in SilkResamplerGpuConfig; the filter state (sIIR, sFIR, delayBuf) lives in device buffers at the config's offsets.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// One SILK resampler's configuration (libopus <c>silk_resampler_init</c>) plus where its state and scratch live:
/// ints at <see cref="SIirAt"/> (6), <see cref="SFirI32At"/> (36) and <see cref="IntScratchAt"/>; shorts at
/// <see cref="SFirI16At"/> (36), <see cref="DelayBufAt"/> (96) and <see cref="ShortScratchAt"/>; tables (shorts) at
/// <see cref="CoefsAt"/> and <see cref="FracFir12At"/>.
/// </summary>
public struct SilkResamplerGpuConfig
{
    /// <summary><c>USE_COPY</c> / <c>USE_UP2_HQ_WRAPPER</c> / <c>USE_IIR_FIR</c> / <c>USE_DOWN_FIR</c>.</summary>
    public int Function;
    /// <summary>Input rate in kHz.</summary>
    public int FsInKHz;
    /// <summary>Output rate in kHz.</summary>
    public int FsOutKHz;
    /// <summary>Input samples per batch (<c>FsInKHz * 10</c>).</summary>
    public int BatchSize;
    /// <summary>Input delay in samples (libopus <c>delay_matrix_dec</c> / <c>_enc</c>).</summary>
    public int InputDelay;
    /// <summary>Output-step increment in Q16.</summary>
    public int InvRatioQ16;
    /// <summary>Down-FIR order (18, 24 or 36).</summary>
    public int FirOrder;
    /// <summary>Down-FIR polyphase fractions.</summary>
    public int FirFracs;
    /// <summary>Down-FIR table (2 AR2 coefficients, then the FIR taps) in the table view.</summary>
    public int CoefsAt;
    /// <summary><c>silk_resampler_frac_FIR_12</c> in the table view.</summary>
    public int FracFir12At;
    /// <summary>sIIR (6 ints).</summary>
    public int SIirAt;
    /// <summary>sFIR as int32 (36 ints; down-FIR).</summary>
    public int SFirI32At;
    /// <summary>sFIR as int16 (36 shorts; IIR-FIR).</summary>
    public int SFirI16At;
    /// <summary>delayBuf (96 shorts).</summary>
    public int DelayBufAt;
    /// <summary>Int scratch, <c>BatchSize + 36</c> (down-FIR buffer).</summary>
    public int IntScratchAt;
    /// <summary>Short scratch, <c>2 * BatchSize + 8</c> (IIR-FIR buffer).</summary>
    public int ShortScratchAt;
}

/// <summary>GPU-callable SILK resampler (libopus <c>silk_resampler</c>).</summary>
public static class SilkResamplerGpu
{
    /// <summary>Ints of state + scratch one resampler needs (decoder side: input at most 16 kHz).</summary>
    public const int IntsPerResampler = SilkResamplerConstants.MAX_IIR_ORDER + SilkResamplerConstants.MAX_FIR_ORDER
        + SilkResamplerConstants.MAX_BATCH_SIZE_IN + SilkResamplerConstants.MAX_FIR_ORDER;
    /// <summary>Shorts of state + scratch one resampler needs.</summary>
    public const int ShortsPerResampler = SilkResamplerConstants.MAX_FIR_ORDER + SilkResamplerConstants.DELAY_BUF_SIZE
        + 2 * SilkResamplerConstants.MAX_BATCH_SIZE_IN + SilkResamplerConstants.ORDER_FIR_12;

    private static readonly sbyte[,] DelayMatrixDec =
    {
        //  out:  8  12  16  24  48  96
        /*  8 */ {  4,  0,  2,  0,  0,  0 },
        /* 12 */ {  0,  9,  4,  7,  4,  4 },
        /* 16 */ {  0,  3, 12,  7,  7,  7 },
    };

    private static int RateId(int rateHz) => Math.Min(5, (((rateHz >> 12) - (rateHz > 16000 ? 1 : 0)) >> (rateHz > 24000 ? 1 : 0)) - 1);

    /// <summary>
    /// Host side of libopus <c>silk_resampler_init</c> (decoder direction): the rate-pair decisions as a config. The
    /// caller zeroes the state (<c>SIirAt</c>, <c>SFirI32At</c>, <c>SFirI16At</c>, <c>DelayBufAt</c>) on the
    /// accelerator. <paramref name="tableOffsets"/> gives where <see cref="SilkResamplerTables"/>' tables were packed:
    /// FracFir12, Coefs3To4, Coefs2To3, Coefs1To2, Coefs1To3, Coefs1To4, Coefs1To6.
    /// </summary>
    public static SilkResamplerGpuConfig InitDecoder(int fsHzIn, int fsHzOut, ReadOnlySpan<int> tableOffsets,
        int intBase, int shortBase)
    {
        if ((fsHzIn != 8000 && fsHzIn != 12000 && fsHzIn != 16000) ||
            (fsHzOut != 8000 && fsHzOut != 12000 && fsHzOut != 16000 && fsHzOut != 24000 && fsHzOut != 48000))
            throw new ArgumentException($"Unsupported decoder rate pair: {fsHzIn} -> {fsHzOut}.");
        var c = new SilkResamplerGpuConfig
        {
            InputDelay = DelayMatrixDec[RateId(fsHzIn), RateId(fsHzOut)],
            FsInKHz = fsHzIn / 1000,
            FsOutKHz = fsHzOut / 1000,
            FracFir12At = tableOffsets[0],
            SIirAt = intBase,
            SFirI32At = intBase + SilkResamplerConstants.MAX_IIR_ORDER,
            IntScratchAt = intBase + SilkResamplerConstants.MAX_IIR_ORDER + SilkResamplerConstants.MAX_FIR_ORDER,
            SFirI16At = shortBase,
            DelayBufAt = shortBase + SilkResamplerConstants.MAX_FIR_ORDER,
            ShortScratchAt = shortBase + SilkResamplerConstants.MAX_FIR_ORDER + SilkResamplerConstants.DELAY_BUF_SIZE,
        };
        c.BatchSize = c.FsInKHz * SilkResamplerConstants.MAX_BATCH_SIZE_MS;
        int up2x = 0;
        if (fsHzOut > fsHzIn)
        {
            if (fsHzOut == fsHzIn * 2) c.Function = SilkResamplerConstants.USE_UP2_HQ_WRAPPER;
            else { c.Function = SilkResamplerConstants.USE_IIR_FIR; up2x = 1; }
        }
        else if (fsHzOut < fsHzIn)
        {
            c.Function = SilkResamplerConstants.USE_DOWN_FIR;
            if (fsHzOut * 4 == fsHzIn * 3) { c.FirFracs = 3; c.FirOrder = SilkResamplerConstants.DOWN_ORDER_FIR0; c.CoefsAt = tableOffsets[1]; }
            else if (fsHzOut * 3 == fsHzIn * 2) { c.FirFracs = 2; c.FirOrder = SilkResamplerConstants.DOWN_ORDER_FIR0; c.CoefsAt = tableOffsets[2]; }
            else if (fsHzOut * 2 == fsHzIn) { c.FirFracs = 1; c.FirOrder = SilkResamplerConstants.DOWN_ORDER_FIR1; c.CoefsAt = tableOffsets[3]; }
            else if (fsHzOut * 3 == fsHzIn) { c.FirFracs = 1; c.FirOrder = SilkResamplerConstants.DOWN_ORDER_FIR2; c.CoefsAt = tableOffsets[4]; }
            else if (fsHzOut * 4 == fsHzIn) { c.FirFracs = 1; c.FirOrder = SilkResamplerConstants.DOWN_ORDER_FIR2; c.CoefsAt = tableOffsets[5]; }
            else if (fsHzOut * 6 == fsHzIn) { c.FirFracs = 1; c.FirOrder = SilkResamplerConstants.DOWN_ORDER_FIR2; c.CoefsAt = tableOffsets[6]; }
            else throw new ArgumentException($"Unsupported downsample rate pair: {fsHzIn} -> {fsHzOut}.");
        }
        else
        {
            c.Function = SilkResamplerConstants.USE_COPY;
        }
        // InvRatio_Q16 = (fsIn << (14 + up2x)) / fsOut << 2, then rounded up until it covers the ratio.
        c.InvRatioQ16 = ((fsHzIn << (14 + up2x)) / fsHzOut) << 2;
        while ((int)(((long)c.InvRatioQ16 * fsHzOut) >> 16) < (fsHzIn << up2x)) c.InvRatioQ16++;
        return c;
    }

    /// <summary>The resampler tables packed in the order <see cref="InitDecoder"/> expects their offsets.</summary>
    public static short[][] Tables => new[]
    {
        SilkResamplerTables.FracFir12, SilkResamplerTables.Coefs3To4, SilkResamplerTables.Coefs2To3, SilkResamplerTables.Coefs1To2,
        SilkResamplerTables.Coefs1To3, SilkResamplerTables.Coefs1To4, SilkResamplerTables.Coefs1To6,
    };

    /// <summary>
    /// libopus <c>silk_resampler</c>: resample <paramref name="inLen"/> samples (at least <c>FsInKHz</c>) of
    /// <paramref name="input"/> into <paramref name="output"/>. One thread.
    /// </summary>
    public static void Apply(SilkResamplerGpuConfig c, ArrayView<int> ints, ArrayView<short> shorts, ArrayView<short> tables,
        ArrayView<short> input, long inBase, ArrayView<short> output, long outBase, int inLen)
    {
        int nSamples = c.FsInKHz - c.InputDelay;
        for (int k = 0; k < nSamples; k++) shorts[c.DelayBufAt + c.InputDelay + k] = input[inBase + k];

        if (c.Function == SilkResamplerConstants.USE_COPY)
        {
            for (int k = 0; k < c.FsInKHz; k++) output[outBase + k] = shorts[c.DelayBufAt + k];
            for (int k = 0; k < inLen - c.FsInKHz; k++) output[outBase + c.FsOutKHz + k] = input[inBase + nSamples + k];
        }
        else if (c.Function == SilkResamplerConstants.USE_UP2_HQ_WRAPPER)
        {
            SilkResamplerUp2HqGpu.ApplyAt(ints, c.SIirAt, output, outBase, shorts, c.DelayBufAt, c.FsInKHz);
            SilkResamplerUp2HqGpu.ApplyAt(ints, c.SIirAt, output, outBase + c.FsOutKHz, input, inBase + nSamples, inLen - c.FsInKHz);
        }
        else if (c.Function == SilkResamplerConstants.USE_IIR_FIR)
        {
            IirFir(c, ints, shorts, tables, shorts, c.DelayBufAt, c.FsInKHz, output, outBase);
            IirFir(c, ints, shorts, tables, input, inBase + nSamples, inLen - c.FsInKHz, output, outBase + c.FsOutKHz);
        }
        else
        {
            DownFir(c, ints, shorts, tables, shorts, c.DelayBufAt, c.FsInKHz, output, outBase);
            DownFir(c, ints, shorts, tables, input, inBase + nSamples, inLen - c.FsInKHz, output, outBase + c.FsOutKHz);
        }

        for (int k = 0; k < c.InputDelay; k++) shorts[c.DelayBufAt + k] = input[inBase + inLen - c.InputDelay + k];
    }

    // Outputs one interpolation pass writes: the index runs 0, inc, 2 inc, ... while below maxIndexQ16.
    private static int OutputCount(long maxIndexQ16, int indexIncrementQ16)
        => maxIndexQ16 <= 0 ? 0 : (int)((maxIndexQ16 + indexIncrementQ16 - 1) / indexIncrementQ16);

    // libopus silk_resampler_private_IIR_FIR.
    private static void IirFir(SilkResamplerGpuConfig c, ArrayView<int> ints, ArrayView<short> shorts, ArrayView<short> tables,
        ArrayView<short> input, long inBase, int inLen, ArrayView<short> output, long outBase)
    {
        const int firOrder = SilkResamplerConstants.ORDER_FIR_12;
        int buf = c.ShortScratchAt;
        for (int k = 0; k < firOrder; k++) shorts[buf + k] = shorts[c.SFirI16At + k];
        long outOffset = 0;
        int remaining = inLen;
        int inOffset = 0;
        int nSamplesInBatch;
        while (true)
        {
            nSamplesInBatch = remaining < c.BatchSize ? remaining : c.BatchSize;
            SilkResamplerUp2HqGpu.ApplyAt(ints, c.SIirAt, shorts, buf + firOrder, input, inBase + inOffset, nSamplesInBatch);
            long maxIndexQ16 = (long)nSamplesInBatch << (16 + 1);
            int count = OutputCount(maxIndexQ16, c.InvRatioQ16);
            for (int k = 0; k < count; k++)
                SilkResamplerIirFirInterpolGpu.ApplyAt(shorts, buf, tables, c.FracFir12At, output, outBase + outOffset, c.InvRatioQ16, k);
            outOffset += count;
            inOffset += nSamplesInBatch;
            remaining -= nSamplesInBatch;
            if (remaining > 0)
            {
                for (int k = 0; k < firOrder; k++) shorts[buf + k] = shorts[buf + (nSamplesInBatch << 1) + k];
            }
            else
            {
                break;
            }
        }
        for (int k = 0; k < firOrder; k++) shorts[c.SFirI16At + k] = shorts[buf + (nSamplesInBatch << 1) + k];
    }

    // libopus silk_resampler_private_down_FIR.
    private static void DownFir(SilkResamplerGpuConfig c, ArrayView<int> ints, ArrayView<short> shorts, ArrayView<short> tables,
        ArrayView<short> input, long inBase, int inLen, ArrayView<short> output, long outBase)
    {
        int firOrder = c.FirOrder;
        int buf = c.IntScratchAt;
        for (int k = 0; k < firOrder; k++) ints[buf + k] = ints[c.SFirI32At + k];
        long outOffset = 0;
        int remaining = inLen;
        int inOffset = 0;
        int nSamplesInBatch;
        while (true)
        {
            nSamplesInBatch = remaining < c.BatchSize ? remaining : c.BatchSize;
            SilkResamplerAr2Gpu.ApplyAt(ints, c.SIirAt, ints, buf + firOrder, input, inBase + inOffset, tables, c.CoefsAt, nSamplesInBatch);
            long maxIndexQ16 = (long)nSamplesInBatch << 16;
            int count = OutputCount(maxIndexQ16, c.InvRatioQ16);
            for (int k = 0; k < count; k++)
            {
                if (firOrder == SilkResamplerConstants.DOWN_ORDER_FIR0)
                    SilkResamplerDownFirInterpolGpu.ApplyFir0At(ints, buf, tables, c.CoefsAt + 2, output, outBase + outOffset, c.InvRatioQ16, c.FirFracs, k);
                else if (firOrder == SilkResamplerConstants.DOWN_ORDER_FIR1)
                    SilkResamplerDownFirInterpolGpu.ApplyFir1At(ints, buf, tables, c.CoefsAt + 2, output, outBase + outOffset, c.InvRatioQ16, k);
                else
                    SilkResamplerDownFirInterpolGpu.ApplyFir2At(ints, buf, tables, c.CoefsAt + 2, output, outBase + outOffset, c.InvRatioQ16, k);
            }
            outOffset += count;
            inOffset += nSamplesInBatch;
            remaining -= nSamplesInBatch;
            if (remaining > 1)
            {
                for (int k = 0; k < firOrder; k++) ints[buf + k] = ints[buf + nSamplesInBatch + k];
            }
            else
            {
                break;
            }
        }
        for (int k = 0; k < firOrder; k++) ints[c.SFirI32At + k] = ints[buf + nSamplesInBatch + k];
    }
}
