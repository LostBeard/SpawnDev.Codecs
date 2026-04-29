// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/resampler.c to clean C#. Implements the
// dispatcher in silk_resampler_init + silk_resampler. Per-variant filter
// implementations (up2 / down_FIR / IIR_FIR) land in companion files;
// this slice wires the identity-rate pass-through path and defers the
// other variants to NotImplementedException until their slices ship.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// SILK resampler. Converts between the SILK internal rates (8/12/16 kHz) and
/// the broader set of Opus API rates (8/12/16/24/48 kHz). Maintains IIR + FIR
/// filter state across successive calls via <see cref="SilkResamplerState"/>.
/// </summary>
internal static class SilkResampler
{
    /// <summary>
    /// Delay-compensation matrix for encoder direction (rows = input rate index,
    /// columns = output rate index). Matches libopus <c>delay_matrix_enc</c>.
    /// </summary>
    private static readonly sbyte[,] DelayMatrixEnc =
    {
        //  out:  8  12  16
        /*  8 */ {  6,  0,  3 },
        /* 12 */ {  0,  7,  3 },
        /* 16 */ {  0,  1, 10 },
        /* 24 */ {  0,  2,  6 },
        /* 48 */ { 18, 10, 12 },
        /* 96 */ {  0,  0, 44 },
    };

    /// <summary>
    /// Delay-compensation matrix for decoder direction (rows = input rate index,
    /// columns = output rate index). Matches libopus <c>delay_matrix_dec</c>.
    /// </summary>
    private static readonly sbyte[,] DelayMatrixDec =
    {
        //  out:  8  12  16  24  48  96
        /*  8 */ {  4,  0,  2,  0,  0,  0 },
        /* 12 */ {  0,  9,  4,  7,  4,  4 },
        /* 16 */ {  0,  3, 12,  7,  7,  7 },
    };

    /// <summary>
    /// Convert a supported sample rate in Hz to its 0-based rate index.
    /// Matches libopus <c>rateID(R) = IMIN(5, (((R&gt;&gt;12) - (R&gt;16000)) &gt;&gt; (R&gt;24000)) - 1)</c>.
    /// </summary>
    private static int RateId(int rateHz)
    {
        int id = (((rateHz >> 12) - (rateHz > 16000 ? 1 : 0)) >> (rateHz > 24000 ? 1 : 0)) - 1;
        return Math.Min(5, id);
    }

    /// <summary>
    /// Initialise the resampler state for a given input/output rate pair.
    /// </summary>
    /// <param name="state">State to populate. Cleared first via <see cref="SilkResamplerState.Clear"/>.</param>
    /// <param name="fsHzIn">Input sample rate in Hz (8000, 12000, 16000, 24000, or 48000).</param>
    /// <param name="fsHzOut">Output sample rate in Hz. Valid set depends on <paramref name="forEncode"/>.</param>
    /// <param name="forEncode">True for encoder-side resampling (Fs_Hz_out restricted to 8/12/16 kHz),
    /// false for decoder-side (Fs_Hz_in restricted to 8/12/16 kHz, Fs_Hz_out broader).</param>
    /// <returns>0 on success; throws <see cref="ArgumentException"/> on unsupported rate pairs.</returns>
    internal static int Init(SilkResamplerState state, int fsHzIn, int fsHzOut, bool forEncode)
    {
        if (state is null) throw new ArgumentNullException(nameof(state));
        state.Clear();

        if (forEncode)
        {
            if ((fsHzIn != 8000 && fsHzIn != 12000 && fsHzIn != 16000 && fsHzIn != 24000 && fsHzIn != 48000) ||
                (fsHzOut != 8000 && fsHzOut != 12000 && fsHzOut != 16000))
            {
                throw new ArgumentException(
                    $"Unsupported encoder rate pair: {fsHzIn} -> {fsHzOut}.");
            }
            state.InputDelay = DelayMatrixEnc[RateId(fsHzIn), RateId(fsHzOut)];
        }
        else
        {
            if ((fsHzIn != 8000 && fsHzIn != 12000 && fsHzIn != 16000) ||
                (fsHzOut != 8000 && fsHzOut != 12000 && fsHzOut != 16000 && fsHzOut != 24000 && fsHzOut != 48000))
            {
                throw new ArgumentException(
                    $"Unsupported decoder rate pair: {fsHzIn} -> {fsHzOut}.");
            }
            state.InputDelay = DelayMatrixDec[RateId(fsHzIn), RateId(fsHzOut)];
        }

        state.FsInKHz = fsHzIn / 1000;
        state.FsOutKHz = fsHzOut / 1000;
        state.BatchSize = state.FsInKHz * SilkResamplerConstants.MAX_BATCH_SIZE_MS;

        int up2x = 0;
        if (fsHzOut > fsHzIn)
        {
            if (fsHzOut == fsHzIn * 2)
            {
                state.ResamplerFunction = SilkResamplerConstants.USE_UP2_HQ_WRAPPER;
            }
            else
            {
                state.ResamplerFunction = SilkResamplerConstants.USE_IIR_FIR;
                up2x = 1;
            }
        }
        else if (fsHzOut < fsHzIn)
        {
            state.ResamplerFunction = SilkResamplerConstants.USE_DOWN_FIR;
            // Rate-specific coefficient selection is deferred until the down_FIR
            // slice lands the tables. For now record the ratio so init succeeds on
            // recognized pairs; Apply() will throw if the coefficients aren't ready.
            if (fsHzOut * 4 == fsHzIn * 3)
            {
                state.FirFracs = 3;
                state.FirOrder = SilkResamplerConstants.DOWN_ORDER_FIR0;
                state.Coefs = SilkResamplerTables.Coefs3To4;
            }
            else if (fsHzOut * 3 == fsHzIn * 2)
            {
                state.FirFracs = 2;
                state.FirOrder = SilkResamplerConstants.DOWN_ORDER_FIR0;
                state.Coefs = SilkResamplerTables.Coefs2To3;
            }
            else if (fsHzOut * 2 == fsHzIn)
            {
                state.FirFracs = 1;
                state.FirOrder = SilkResamplerConstants.DOWN_ORDER_FIR1;
                state.Coefs = SilkResamplerTables.Coefs1To2;
            }
            else if (fsHzOut * 3 == fsHzIn)
            {
                state.FirFracs = 1;
                state.FirOrder = SilkResamplerConstants.DOWN_ORDER_FIR2;
                state.Coefs = SilkResamplerTables.Coefs1To3;
            }
            else if (fsHzOut * 4 == fsHzIn)
            {
                state.FirFracs = 1;
                state.FirOrder = SilkResamplerConstants.DOWN_ORDER_FIR2;
                state.Coefs = SilkResamplerTables.Coefs1To4;
            }
            else if (fsHzOut * 6 == fsHzIn)
            {
                state.FirFracs = 1;
                state.FirOrder = SilkResamplerConstants.DOWN_ORDER_FIR2;
                state.Coefs = SilkResamplerTables.Coefs1To6;
            }
            else
            {
                throw new ArgumentException(
                    $"Unsupported downsample rate pair: {fsHzIn} -> {fsHzOut}.");
            }
        }
        else
        {
            state.ResamplerFunction = SilkResamplerConstants.USE_COPY;
        }

        // InvRatio_Q16: scales by 2^14 (plus an extra bit when up-sampling through IIR-FIR).
        state.InvRatioQ16 = silk_LSHIFT32(silk_DIV32(silk_LSHIFT32(fsHzIn, 14 + up2x), fsHzOut), 2);
        while (silk_SMULWW(state.InvRatioQ16, fsHzOut) < silk_LSHIFT32(fsHzIn, up2x))
        {
            state.InvRatioQ16++;
        }

        return 0;
    }

    /// <summary>
    /// Apply the resampler to a batch of input samples, producing output at the configured rate.
    /// <para>
    /// This slice supports the identity-rate pass-through only; the up2 / IIR-FIR / down-FIR
    /// variants throw <see cref="NotImplementedException"/> and will be filled in by
    /// subsequent slices.
    /// </para>
    /// </summary>
    /// <param name="state">Initialised resampler state.</param>
    /// <param name="output">Output buffer. Length &gt;= ceil(inLen * FsOutKHz / FsInKHz).</param>
    /// <param name="input">Input buffer. Length &gt;= <paramref name="inLen"/>.</param>
    /// <param name="inLen">Number of input samples to consume.</param>
    internal static void Apply(SilkResamplerState state, Span<short> output, ReadOnlySpan<short> input, int inLen)
    {
        if (state is null) throw new ArgumentNullException(nameof(state));
        if (inLen < state.FsInKHz)
            throw new ArgumentException(
                $"inLen ({inLen}) must be >= FsInKHz ({state.FsInKHz}) per libopus contract.", nameof(inLen));
        if (state.InputDelay > state.FsInKHz)
            throw new InvalidOperationException(
                $"inputDelay ({state.InputDelay}) > FsInKHz ({state.FsInKHz}); state corruption.");

        int nSamples = state.FsInKHz - state.InputDelay;

        // Stage the first chunk into the delay buffer.
        input.Slice(0, nSamples).CopyTo(state.DelayBuf.AsSpan(state.InputDelay, nSamples));

        switch (state.ResamplerFunction)
        {
            case SilkResamplerConstants.USE_COPY:
                // Identity: copy the delay buffer's first FsInKHz samples, then the
                // remaining input past the delay buffer.
                state.DelayBuf.AsSpan(0, state.FsInKHz).CopyTo(output);
                input.Slice(nSamples, inLen - state.FsInKHz).CopyTo(output.Slice(state.FsOutKHz));
                break;

            case SilkResamplerConstants.USE_UP2_HQ_WRAPPER:
                // Two halves: first from delayBuf's FsInKHz samples, then from the remaining input.
                Up2HqWrapper(state, output, state.DelayBuf.AsSpan(0, state.FsInKHz), state.FsInKHz);
                Up2HqWrapper(state, output.Slice(state.FsOutKHz),
                    input.Slice(nSamples, inLen - state.FsInKHz), inLen - state.FsInKHz);
                break;

            case SilkResamplerConstants.USE_IIR_FIR:
                IirFir(state, output, state.DelayBuf.AsSpan(0, state.FsInKHz), state.FsInKHz);
                IirFir(state, output.Slice(state.FsOutKHz),
                    input.Slice(nSamples, inLen - state.FsInKHz), inLen - state.FsInKHz);
                break;

            case SilkResamplerConstants.USE_DOWN_FIR:
                DownFir(state, output, state.DelayBuf.AsSpan(0, state.FsInKHz), state.FsInKHz);
                DownFir(state, output.Slice(state.FsOutKHz),
                    input.Slice(nSamples, inLen - state.FsInKHz), inLen - state.FsInKHz);
                break;

            default:
                throw new InvalidOperationException(
                    $"Unknown ResamplerFunction {state.ResamplerFunction}.");
        }

        // Slide the trailing InputDelay samples of the input into the delay buffer for the next call.
        input.Slice(inLen - state.InputDelay, state.InputDelay).CopyTo(state.DelayBuf);
    }

    // ---- IIR + FIR arbitrary upsample ----

    /// <summary>
    /// Arbitrary-ratio upsampler for cases where <c>fsOut / fsIn</c> is not exactly 2.
    /// Runs the input through the 2x HQ upsampler (doubles the sample count), then
    /// applies a 12-phase fractional FIR to produce the final output. Matches libopus
    /// <c>silk_resampler_private_IIR_FIR</c>.
    /// </summary>
    private static void IirFir(SilkResamplerState state, Span<short> output,
        ReadOnlySpan<short> input, int inLen)
    {
        int firOrder = SilkResamplerConstants.ORDER_FIR_12;
        // buf holds the up2 output (2 * batchSize samples) plus the FIR history prefix (firOrder samples).
        Span<short> buf = stackalloc short[2 * state.BatchSize + SilkResamplerConstants.ORDER_FIR_12];

        // Prime buf with the persisted FIR history (first 8 entries).
        state.SFirI16.AsSpan(0, firOrder).CopyTo(buf);

        int indexIncrementQ16 = state.InvRatioQ16;
        int outOffset = 0;
        int remaining = inLen;
        int inOffset = 0;
        int nSamplesInBatch;

        while (true)
        {
            nSamplesInBatch = Math.Min(remaining, state.BatchSize);

            // Upsample this batch by 2 into buf[firOrder..firOrder + 2*nSamplesInBatch].
            Up2Hq(state.SIir, buf.Slice(firOrder), input.Slice(inOffset, nSamplesInBatch), nSamplesInBatch);

            long maxIndexQ16 = (long)nSamplesInBatch << (16 + 1);
            outOffset = IirFirInterpol(output, outOffset, buf, maxIndexQ16, indexIncrementQ16);

            inOffset += nSamplesInBatch;
            remaining -= nSamplesInBatch;

            if (remaining > 0)
            {
                // Slide the trailing firOrder samples of the up2-output back to the head.
                buf.Slice(nSamplesInBatch << 1, firOrder).CopyTo(buf);
            }
            else
            {
                break;
            }
        }

        // Persist the FIR history for the next call.
        buf.Slice(nSamplesInBatch << 1, firOrder).CopyTo(state.SFirI16);
    }

    /// <summary>
    /// 12-phase fractional FIR interpolator. Each output sample reads 4 coefficients
    /// from the low half of the table (rows <c>tableIdx</c>) and 4 from the mirrored
    /// high half (rows <c>11 - tableIdx</c>), applied to 8 consecutive buffer samples.
    /// Matches libopus <c>silk_resampler_private_IIR_FIR_INTERPOL</c>.
    /// </summary>
    private static int IirFirInterpol(Span<short> output, int outOffset, ReadOnlySpan<short> buf,
        long maxIndexQ16, int indexIncrementQ16)
    {
        for (long indexQ16 = 0; indexQ16 < maxIndexQ16; indexQ16 += indexIncrementQ16)
        {
            int tableIndex = silk_SMULWB((int)indexQ16 & 0xFFFF, 12);
            int bufStart = (int)(indexQ16 >> 16);
            int rowLow = tableIndex * 4;
            int rowHigh = (11 - tableIndex) * 4;

            int resQ15 = silk_SMULBB(buf[bufStart + 0], SilkResamplerTables.FracFir12[rowLow + 0]);
            resQ15 = silk_SMLABB(resQ15, buf[bufStart + 1], SilkResamplerTables.FracFir12[rowLow + 1]);
            resQ15 = silk_SMLABB(resQ15, buf[bufStart + 2], SilkResamplerTables.FracFir12[rowLow + 2]);
            resQ15 = silk_SMLABB(resQ15, buf[bufStart + 3], SilkResamplerTables.FracFir12[rowLow + 3]);
            resQ15 = silk_SMLABB(resQ15, buf[bufStart + 4], SilkResamplerTables.FracFir12[rowHigh + 3]);
            resQ15 = silk_SMLABB(resQ15, buf[bufStart + 5], SilkResamplerTables.FracFir12[rowHigh + 2]);
            resQ15 = silk_SMLABB(resQ15, buf[bufStart + 6], SilkResamplerTables.FracFir12[rowHigh + 1]);
            resQ15 = silk_SMLABB(resQ15, buf[bufStart + 7], SilkResamplerTables.FracFir12[rowHigh + 0]);

            output[outOffset++] = silk_SAT16(silk_RSHIFT_ROUND(resQ15, 15));
        }
        return outOffset;
    }

    // ---- 2x HQ upsample ----

    /// <summary>Coefficients for the 2x high-quality upsample filter's even-sample all-pass cascade.</summary>
    private static readonly short[] Up2Hq0 = { 1746, 14986, (short)(39083 - 65536) };

    /// <summary>Coefficients for the 2x high-quality upsample filter's odd-sample all-pass cascade.</summary>
    private static readonly short[] Up2Hq1 = { 6854, 25769, (short)(55542 - 65536) };

    /// <summary>
    /// Thin wrapper matching libopus <c>silk_resampler_private_up2_HQ_wrapper</c>: forwards
    /// to <see cref="Up2Hq"/> using the first 6 entries of the state's IIR buffer.
    /// </summary>
    private static void Up2HqWrapper(SilkResamplerState state, Span<short> output, ReadOnlySpan<short> input, int len)
    {
        Up2Hq(state.SIir, output, input, len);
    }

    // ---- Downsample FIR ----

    /// <summary>
    /// Downsample via AR2 pre-filter + polyphase interpolated FIR. Matches
    /// libopus <c>silk_resampler_private_down_FIR</c>. The FIR order + number
    /// of polyphase fractions + coefficient table are all pre-selected in
    /// <see cref="Init"/> based on the specific input/output rate ratio.
    /// </summary>
    private static void DownFir(SilkResamplerState state, Span<short> output,
        ReadOnlySpan<short> input, int inLen)
    {
        if (state.Coefs is null)
            throw new InvalidOperationException("DownFir: state.Coefs not initialized.");

        int firOrder = state.FirOrder;
        // buf length is batchSize + FIR_Order int32 values.
        Span<int> buf = stackalloc int[state.BatchSize + firOrder];

        // Prime buf with the persisted FIR history (first FIR_Order entries).
        state.SFirI32.AsSpan(0, firOrder).CopyTo(buf);

        ReadOnlySpan<short> firCoefs = state.Coefs.AsSpan(2);

        int outOffset = 0;
        int indexIncrementQ16 = state.InvRatioQ16;
        int remaining = inLen;
        int inOffset = 0;
        int nSamplesInBatch;

        while (true)
        {
            nSamplesInBatch = Math.Min(remaining, state.BatchSize);

            // AR2 pre-filter writes to buf[FIR_Order..FIR_Order+nSamplesInBatch].
            Ar2(state.SIir, buf.Slice(firOrder, nSamplesInBatch),
                input.Slice(inOffset, nSamplesInBatch), state.Coefs.AsSpan(0, 2));

            long maxIndexQ16 = (long)nSamplesInBatch << 16;
            outOffset = DownFirInterpol(output, outOffset, buf, firCoefs,
                firOrder, state.FirFracs, maxIndexQ16, indexIncrementQ16);

            inOffset += nSamplesInBatch;
            remaining -= nSamplesInBatch;

            if (remaining > 1)
            {
                // Slide the trailing FIR_Order samples of buf back to the head for the next batch.
                buf.Slice(nSamplesInBatch, firOrder).CopyTo(buf);
            }
            else
            {
                break;
            }
        }

        // Persist the FIR history for the next Apply() call.
        buf.Slice(nSamplesInBatch, firOrder).CopyTo(state.SFirI32);
    }

    /// <summary>
    /// AR2 IIR pre-filter (2 coefficients in Q14). Matches libopus
    /// <c>silk_resampler_private_AR2</c>.
    /// </summary>
    internal static void Ar2(Span<int> S, Span<int> outQ8, ReadOnlySpan<short> input, ReadOnlySpan<short> aQ14)
    {
        for (int k = 0; k < outQ8.Length; k++)
        {
            int out32 = silk_ADD_LSHIFT32(S[0], input[k], 8);
            outQ8[k] = out32;
            out32 = silk_LSHIFT(out32, 2);
            S[0] = silk_SMLAWB(S[1], out32, aQ14[0]);
            S[1] = silk_SMULWB(out32, aQ14[1]);
        }
    }

    /// <summary>
    /// Polyphase interpolated FIR downsampler. Selects one of three implementations
    /// based on FIR_Order (18 / 24 / 36). Matches libopus <c>silk_resampler_private_down_FIR_INTERPOL</c>.
    /// </summary>
    private static int DownFirInterpol(Span<short> output, int outOffset, ReadOnlySpan<int> buf,
        ReadOnlySpan<short> firCoefs, int firOrder, int firFracs, long maxIndexQ16, int indexIncrementQ16)
    {
        switch (firOrder)
        {
            case SilkResamplerConstants.DOWN_ORDER_FIR0:
                for (long indexQ16 = 0; indexQ16 < maxIndexQ16; indexQ16 += indexIncrementQ16)
                {
                    int bufStart = (int)(indexQ16 >> 16);
                    int interpolInd = silk_SMULWB((int)indexQ16 & 0xFFFF, firFracs);
                    int interpolStart = (SilkResamplerConstants.DOWN_ORDER_FIR0 / 2) * interpolInd;

                    int resQ6 = silk_SMULWB(buf[bufStart + 0], firCoefs[interpolStart + 0]);
                    resQ6 = silk_SMLAWB(resQ6, buf[bufStart + 1], firCoefs[interpolStart + 1]);
                    resQ6 = silk_SMLAWB(resQ6, buf[bufStart + 2], firCoefs[interpolStart + 2]);
                    resQ6 = silk_SMLAWB(resQ6, buf[bufStart + 3], firCoefs[interpolStart + 3]);
                    resQ6 = silk_SMLAWB(resQ6, buf[bufStart + 4], firCoefs[interpolStart + 4]);
                    resQ6 = silk_SMLAWB(resQ6, buf[bufStart + 5], firCoefs[interpolStart + 5]);
                    resQ6 = silk_SMLAWB(resQ6, buf[bufStart + 6], firCoefs[interpolStart + 6]);
                    resQ6 = silk_SMLAWB(resQ6, buf[bufStart + 7], firCoefs[interpolStart + 7]);
                    resQ6 = silk_SMLAWB(resQ6, buf[bufStart + 8], firCoefs[interpolStart + 8]);

                    int interpolStart2 = (SilkResamplerConstants.DOWN_ORDER_FIR0 / 2) * (firFracs - 1 - interpolInd);
                    resQ6 = silk_SMLAWB(resQ6, buf[bufStart + 17], firCoefs[interpolStart2 + 0]);
                    resQ6 = silk_SMLAWB(resQ6, buf[bufStart + 16], firCoefs[interpolStart2 + 1]);
                    resQ6 = silk_SMLAWB(resQ6, buf[bufStart + 15], firCoefs[interpolStart2 + 2]);
                    resQ6 = silk_SMLAWB(resQ6, buf[bufStart + 14], firCoefs[interpolStart2 + 3]);
                    resQ6 = silk_SMLAWB(resQ6, buf[bufStart + 13], firCoefs[interpolStart2 + 4]);
                    resQ6 = silk_SMLAWB(resQ6, buf[bufStart + 12], firCoefs[interpolStart2 + 5]);
                    resQ6 = silk_SMLAWB(resQ6, buf[bufStart + 11], firCoefs[interpolStart2 + 6]);
                    resQ6 = silk_SMLAWB(resQ6, buf[bufStart + 10], firCoefs[interpolStart2 + 7]);
                    resQ6 = silk_SMLAWB(resQ6, buf[bufStart + 9], firCoefs[interpolStart2 + 8]);

                    output[outOffset++] = silk_SAT16(silk_RSHIFT_ROUND(resQ6, 6));
                }
                break;

            case SilkResamplerConstants.DOWN_ORDER_FIR1:
                for (long indexQ16 = 0; indexQ16 < maxIndexQ16; indexQ16 += indexIncrementQ16)
                {
                    int bufStart = (int)(indexQ16 >> 16);
                    int resQ6 = silk_SMULWB(silk_ADD32(buf[bufStart + 0], buf[bufStart + 23]), firCoefs[0]);
                    for (int k = 1; k < 12; k++)
                    {
                        resQ6 = silk_SMLAWB(resQ6,
                            silk_ADD32(buf[bufStart + k], buf[bufStart + 23 - k]),
                            firCoefs[k]);
                    }
                    output[outOffset++] = silk_SAT16(silk_RSHIFT_ROUND(resQ6, 6));
                }
                break;

            case SilkResamplerConstants.DOWN_ORDER_FIR2:
                for (long indexQ16 = 0; indexQ16 < maxIndexQ16; indexQ16 += indexIncrementQ16)
                {
                    int bufStart = (int)(indexQ16 >> 16);
                    int resQ6 = silk_SMULWB(silk_ADD32(buf[bufStart + 0], buf[bufStart + 35]), firCoefs[0]);
                    for (int k = 1; k < 18; k++)
                    {
                        resQ6 = silk_SMLAWB(resQ6,
                            silk_ADD32(buf[bufStart + k], buf[bufStart + 35 - k]),
                            firCoefs[k]);
                    }
                    output[outOffset++] = silk_SAT16(silk_RSHIFT_ROUND(resQ6, 6));
                }
                break;

            default:
                throw new InvalidOperationException($"Unsupported FIR order {firOrder}.");
        }
        return outOffset;
    }

    /// <summary>
    /// 2x high-quality upsampler. Produces 2*<paramref name="len"/> output samples from
    /// <paramref name="len"/> input samples via 3 cascaded all-pass filters per output phase.
    /// Matches libopus <c>silk_resampler_private_up2_HQ</c> bit-exactly.
    /// </summary>
    /// <param name="S">State buffer, at least 6 entries ([0..2] = even branch, [3..5] = odd branch).</param>
    /// <param name="output">Output samples (length &gt;= 2 * len).</param>
    /// <param name="input">Input samples (length &gt;= len).</param>
    /// <param name="len">Number of input samples.</param>
    internal static void Up2Hq(Span<int> S, Span<short> output, ReadOnlySpan<short> input, int len)
    {
        // All state + internal variables are in Q10.
        for (int k = 0; k < len; k++)
        {
            int in32 = silk_LSHIFT((int)input[k], 10);

            // Even-sample branch: three all-pass sections using Up2Hq0.
            int Y = silk_SUB32(in32, S[0]);
            int X = silk_SMULWB(Y, Up2Hq0[0]);
            int out32_1 = silk_ADD32(S[0], X);
            S[0] = silk_ADD32(in32, X);

            Y = silk_SUB32(out32_1, S[1]);
            X = silk_SMULWB(Y, Up2Hq0[1]);
            int out32_2 = silk_ADD32(S[1], X);
            S[1] = silk_ADD32(out32_1, X);

            Y = silk_SUB32(out32_2, S[2]);
            X = silk_SMLAWB(Y, Y, Up2Hq0[2]);
            out32_1 = silk_ADD32(S[2], X);
            S[2] = silk_ADD32(out32_2, X);

            output[2 * k] = silk_SAT16(silk_RSHIFT_ROUND(out32_1, 10));

            // Odd-sample branch: three all-pass sections using Up2Hq1.
            Y = silk_SUB32(in32, S[3]);
            X = silk_SMULWB(Y, Up2Hq1[0]);
            out32_1 = silk_ADD32(S[3], X);
            S[3] = silk_ADD32(in32, X);

            Y = silk_SUB32(out32_1, S[4]);
            X = silk_SMULWB(Y, Up2Hq1[1]);
            out32_2 = silk_ADD32(S[4], X);
            S[4] = silk_ADD32(out32_1, X);

            Y = silk_SUB32(out32_2, S[5]);
            X = silk_SMLAWB(Y, Y, Up2Hq1[2]);
            out32_1 = silk_ADD32(S[5], X);
            S[5] = silk_ADD32(out32_2, X);

            output[2 * k + 1] = silk_SAT16(silk_RSHIFT_ROUND(out32_1, 10));
        }
    }
}
