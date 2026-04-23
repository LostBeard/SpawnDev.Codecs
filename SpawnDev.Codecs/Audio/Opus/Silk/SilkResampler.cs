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
            }
            else if (fsHzOut * 3 == fsHzIn * 2)
            {
                state.FirFracs = 2;
                state.FirOrder = SilkResamplerConstants.DOWN_ORDER_FIR0;
            }
            else if (fsHzOut * 2 == fsHzIn)
            {
                state.FirFracs = 1;
                state.FirOrder = SilkResamplerConstants.DOWN_ORDER_FIR1;
            }
            else if (fsHzOut * 3 == fsHzIn)
            {
                state.FirFracs = 1;
                state.FirOrder = SilkResamplerConstants.DOWN_ORDER_FIR2;
            }
            else if (fsHzOut * 4 == fsHzIn)
            {
                state.FirFracs = 1;
                state.FirOrder = SilkResamplerConstants.DOWN_ORDER_FIR2;
            }
            else if (fsHzOut * 6 == fsHzIn)
            {
                state.FirFracs = 1;
                state.FirOrder = SilkResamplerConstants.DOWN_ORDER_FIR2;
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
                throw new NotImplementedException(
                    "silk_resampler_private_up2_HQ_wrapper not yet ported (slice 43+).");

            case SilkResamplerConstants.USE_IIR_FIR:
                throw new NotImplementedException(
                    "silk_resampler_private_IIR_FIR not yet ported (slice 44+).");

            case SilkResamplerConstants.USE_DOWN_FIR:
                throw new NotImplementedException(
                    "silk_resampler_private_down_FIR not yet ported (slice 45+).");

            default:
                throw new InvalidOperationException(
                    $"Unknown ResamplerFunction {state.ResamplerFunction}.");
        }

        // Slide the trailing InputDelay samples of the input into the delay buffer for the next call.
        input.Slice(inLen - state.InputDelay, state.InputDelay).CopyTo(state.DelayBuf);
    }
}
