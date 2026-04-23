// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus silk/decode_pulses.c + silk/code_signs.c to clean C#.
// Decodes the full excitation (pulses) for a SILK frame: rate level -> per-block
// pulse counts (with LSB-extension escape for large counts) -> shell coder per
// block -> LSB bits on top of shell output -> sign bits.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

using SpawnDev.Codecs.EntropyCoders;
using static SpawnDev.Codecs.Audio.Opus.Silk.SilkMacros;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Full SILK pulse (excitation) decoder, orchestrating the rate-level selection,
/// per-block pulse-count decode (with optional LSB-extension escape), shell coder,
/// LSB-bit extension, and sign decoding. Matches libopus <c>silk_decode_pulses</c>.
/// </summary>
internal static class SilkPulsesDecoder
{
    /// <summary>
    /// Decode the signed excitation for a SILK frame.
    /// </summary>
    /// <param name="pulses">Output: signed pulse magnitudes, length = <paramref name="frameLength"/>.</param>
    /// <param name="rangeDec">Range decoder positioned at the start of the pulses block.</param>
    /// <param name="signalType">SILK signal type (0 inactive, 1 unvoiced, 2 voiced).</param>
    /// <param name="quantOffsetType">SILK quantizer offset type (0 or 1).</param>
    /// <param name="frameLength">Frame length in samples (must be a multiple of
    /// <see cref="SilkConstants.SHELL_CODEC_FRAME_LENGTH"/>, or 120 for 15 ms MB frames).</param>
    internal static void Decode(
        Span<short> pulses,
        OpusRangeDecoder rangeDec,
        int signalType,
        int quantOffsetType,
        int frameLength)
    {
        if (rangeDec is null) throw new ArgumentNullException(nameof(rangeDec));
        if ((uint)signalType > 2) throw new ArgumentOutOfRangeException(nameof(signalType));
        if ((uint)quantOffsetType > 1) throw new ArgumentOutOfRangeException(nameof(quantOffsetType));
        if (frameLength <= 0) throw new ArgumentOutOfRangeException(nameof(frameLength));
        if (pulses.Length < frameLength)
            throw new ArgumentException($"pulses too small (need {frameLength}).", nameof(pulses));

        int shellLen = SilkConstants.SHELL_CODEC_FRAME_LENGTH;

        // Rate-level selection (9-symbol iCDF per signal-type-rough-class).
        int rateLevelIndex = rangeDec.DecodeIcdf(
            SilkIcdfTables.RateLevels.AsSpan(
                SilkIcdfTables.RateLevelsOffset(signalType),
                SilkIcdfTables.RateLevelsEntriesPerType),
            8);

        int iter = frameLength >> SilkConstants.LOG2_SHELL_CODEC_FRAME_LENGTH;
        if (iter * shellLen < frameLength)
        {
            // 120-sample MB frame: decoded in an extra shell block that's only partially used.
            iter++;
        }
        if (iter > SilkConstants.MAX_NB_SHELL_BLOCKS)
            throw new ArgumentException($"frameLength {frameLength} yields {iter} shell blocks, exceeds max {SilkConstants.MAX_NB_SHELL_BLOCKS}.", nameof(frameLength));

        Span<int> sumPulses = stackalloc int[SilkConstants.MAX_NB_SHELL_BLOCKS];
        Span<int> nLshifts = stackalloc int[SilkConstants.MAX_NB_SHELL_BLOCKS];

        int cdfOffset = SilkIcdfTables.PulsesPerBlockOffset(rateLevelIndex);
        int escapeRowStart = SilkIcdfTables.PulsesPerBlockOffset(SilkConstants.N_RATE_LEVELS - 1);

        for (int i = 0; i < iter; i++)
        {
            nLshifts[i] = 0;
            sumPulses[i] = rangeDec.DecodeIcdf(
                SilkIcdfTables.PulsesPerBlock.AsSpan(cdfOffset, SilkIcdfTables.PulsesPerBlockEntriesPerRow),
                8);

            while (sumPulses[i] == SilkConstants.SILK_MAX_PULSES + 1)
            {
                nLshifts[i]++;
                // Escape path: after 10 consecutive escapes, read from the row shifted by one.
                // Libopus encodes that by indexing: silk_pulses_per_block_iCDF[N_RATE_LEVELS-1] + (nLshifts == 10)
                // i.e. starting one byte later in the last rate-level row.
                int adjust = nLshifts[i] == 10 ? 1 : 0;
                int escOffset = escapeRowStart + adjust;
                int escLen = SilkIcdfTables.PulsesPerBlockEntriesPerRow - adjust;
                sumPulses[i] = rangeDec.DecodeIcdf(
                    SilkIcdfTables.PulsesPerBlock.AsSpan(escOffset, escLen),
                    8);
            }
        }

        // Shell-decode magnitudes for each block (or zero out empty blocks).
        for (int i = 0; i < iter; i++)
        {
            Span<short> blockPulses = pulses.Slice(i * shellLen, shellLen);
            if (sumPulses[i] > 0)
            {
                SilkShellCoder.Decode(blockPulses, rangeDec, sumPulses[i]);
            }
            else
            {
                blockPulses.Clear();
            }
        }

        // LSB-bit extension: for each escape-stretched block, append `nLshifts` bits per sample
        // starting from the MSB side (equivalent to left-shift then OR with new LSB bit).
        for (int i = 0; i < iter; i++)
        {
            if (nLshifts[i] > 0)
            {
                int nLs = nLshifts[i];
                Span<short> blockPulses = pulses.Slice(i * shellLen, shellLen);
                for (int k = 0; k < shellLen; k++)
                {
                    int absQ = blockPulses[k];
                    for (int j = 0; j < nLs; j++)
                    {
                        absQ = silk_LSHIFT(absQ, 1);
                        absQ += rangeDec.DecodeIcdf(SilkIcdfTables.Lsb, 8);
                    }
                    blockPulses[k] = (short)absQ;
                }
                // Pack nLshifts into bits 5..9 of sumPulses for the sign decoder.
                sumPulses[i] |= nLs << 5;
            }
        }

        DecodeSigns(pulses, rangeDec, frameLength, signalType, quantOffsetType, sumPulses);
    }

    /// <summary>
    /// Sign-decode step. Uses <see cref="SilkIcdfTables.Sign"/>; the row is keyed on
    /// <c>(signalType, quantOffsetType)</c> and the column on <c>min(sumPulses[i] &amp; 0x1F, 6)</c>.
    /// Flips the sign of each non-zero pulse per bit read from the row's 2-symbol iCDF.
    /// </summary>
    private static void DecodeSigns(
        Span<short> pulses,
        OpusRangeDecoder rangeDec,
        int frameLength,
        int signalType,
        int quantOffsetType,
        ReadOnlySpan<int> sumPulses)
    {
        int shellLen = SilkConstants.SHELL_CODEC_FRAME_LENGTH;

        Span<byte> icdf = stackalloc byte[2];
        icdf[1] = 0;

        int rowStart = SilkIcdfTables.SignOffset(signalType, quantOffsetType);
        int length = (frameLength + shellLen / 2) >> SilkConstants.LOG2_SHELL_CODEC_FRAME_LENGTH;

        for (int i = 0; i < length; i++)
        {
            int p = sumPulses[i];
            if (p > 0)
            {
                int col = silk_min(p & 0x1F, 6);
                icdf[0] = SilkIcdfTables.Sign[rowStart + col];
                Span<short> block = pulses.Slice(i * shellLen, shellLen);
                for (int j = 0; j < shellLen; j++)
                {
                    if (block[j] > 0)
                    {
                        int decoded = rangeDec.DecodeIcdf(icdf, 8);
                        // silk_dec_map(x) = (x << 1) - 1 -> x=0 gives -1, x=1 gives +1.
                        int sign = silk_LSHIFT(decoded, 1) - 1;
                        block[j] = (short)(block[j] * sign);
                    }
                }
            }
        }
    }

    // ------------ Test-friendly encoder counterpart ------------

    /// <summary>
    /// Encode the signed pulses for a frame. Inverse of <see cref="Decode"/>; used by
    /// tests and, later, by a SILK encoder.
    /// </summary>
    /// <param name="rangeEnc">Range encoder to write into.</param>
    /// <param name="pulses">Signed pulse magnitudes, length <paramref name="frameLength"/>.</param>
    /// <param name="signalType">SILK signal type.</param>
    /// <param name="quantOffsetType">SILK quantizer offset type.</param>
    /// <param name="frameLength">Frame length in samples.</param>
    /// <param name="rateLevelIndex">Rate-level index to encode (0..8).</param>
    internal static void Encode(
        OpusRangeEncoder rangeEnc,
        ReadOnlySpan<short> pulses,
        int signalType,
        int quantOffsetType,
        int frameLength,
        int rateLevelIndex)
    {
        if (rangeEnc is null) throw new ArgumentNullException(nameof(rangeEnc));
        if ((uint)signalType > 2) throw new ArgumentOutOfRangeException(nameof(signalType));
        if ((uint)quantOffsetType > 1) throw new ArgumentOutOfRangeException(nameof(quantOffsetType));
        if (pulses.Length < frameLength)
            throw new ArgumentException($"pulses too small (need {frameLength}).", nameof(pulses));

        int shellLen = SilkConstants.SHELL_CODEC_FRAME_LENGTH;
        int iter = frameLength >> SilkConstants.LOG2_SHELL_CODEC_FRAME_LENGTH;
        if (iter * shellLen < frameLength) iter++;

        rangeEnc.EncodeIcdf(rateLevelIndex,
            SilkIcdfTables.RateLevels.AsSpan(
                SilkIcdfTables.RateLevelsOffset(signalType),
                SilkIcdfTables.RateLevelsEntriesPerType),
            8);

        Span<int> sumPulses = stackalloc int[SilkConstants.MAX_NB_SHELL_BLOCKS];
        Span<int> nLshifts = stackalloc int[SilkConstants.MAX_NB_SHELL_BLOCKS];

        // Compute absolute sums per block and determine lshift counts.
        for (int i = 0; i < iter; i++)
        {
            int sum = 0;
            for (int k = 0; k < shellLen; k++)
            {
                int abs = Math.Abs((int)pulses[i * shellLen + k]);
                sum += abs;
            }
            int nLs = 0;
            int encSum = sum;
            while (encSum > SilkConstants.SILK_MAX_PULSES)
            {
                encSum = (encSum + 1) >> 1;
                nLs++;
            }
            sumPulses[i] = encSum;
            nLshifts[i] = nLs;
        }

        // Emit the per-block pulse counts with the LSB-extension escape chain.
        int cdfOffset = SilkIcdfTables.PulsesPerBlockOffset(rateLevelIndex);
        int escapeRowStart = SilkIcdfTables.PulsesPerBlockOffset(SilkConstants.N_RATE_LEVELS - 1);

        for (int i = 0; i < iter; i++)
        {
            int nLs = nLshifts[i];
            for (int e = 0; e < nLs; e++)
            {
                // Emit the escape token (SILK_MAX_PULSES + 1) from appropriate iCDF.
                int adjust = e == 10 ? 1 : 0;
                int escLen = SilkIcdfTables.PulsesPerBlockEntriesPerRow - adjust;
                if (e == 0)
                {
                    rangeEnc.EncodeIcdf(SilkConstants.SILK_MAX_PULSES + 1,
                        SilkIcdfTables.PulsesPerBlock.AsSpan(cdfOffset, SilkIcdfTables.PulsesPerBlockEntriesPerRow),
                        8);
                }
                else
                {
                    int prevAdjust = (e - 1) == 10 ? 1 : 0;
                    int prevLen = SilkIcdfTables.PulsesPerBlockEntriesPerRow - prevAdjust;
                    rangeEnc.EncodeIcdf(SilkConstants.SILK_MAX_PULSES + 1 - prevAdjust,
                        SilkIcdfTables.PulsesPerBlock.AsSpan(escapeRowStart + prevAdjust, prevLen),
                        8);
                }
            }
            // Final non-escape count from the appropriate iCDF.
            if (nLs == 0)
            {
                rangeEnc.EncodeIcdf(sumPulses[i],
                    SilkIcdfTables.PulsesPerBlock.AsSpan(cdfOffset, SilkIcdfTables.PulsesPerBlockEntriesPerRow),
                    8);
            }
            else
            {
                int adjust = nLs == 10 ? 1 : 0;
                int escLen = SilkIcdfTables.PulsesPerBlockEntriesPerRow - adjust;
                rangeEnc.EncodeIcdf(sumPulses[i],
                    SilkIcdfTables.PulsesPerBlock.AsSpan(escapeRowStart + adjust, escLen),
                    8);
            }
        }

        // Shell-code the top-bit magnitudes per block (or skip empty blocks).
        Span<short> blockMagnitudes = stackalloc short[shellLen];
        for (int i = 0; i < iter; i++)
        {
            int nLs = nLshifts[i];
            int sum = sumPulses[i];
            if (sum > 0)
            {
                for (int k = 0; k < shellLen; k++)
                {
                    int abs = Math.Abs((int)pulses[i * shellLen + k]);
                    blockMagnitudes[k] = (short)(abs >> nLs);
                }
                // Enforce the shell-coder invariant: pulses sum to sumPulses[i].
                int shellSum = 0;
                for (int k = 0; k < shellLen; k++) shellSum += blockMagnitudes[k];
                if (shellSum != sum)
                {
                    // When LSB-extension is active, the post-shift sum of abs values can be
                    // slightly less than sum due to rounding. For this test-side encoder we
                    // require the input pulses to produce a clean shell sum; else throw.
                    throw new ArgumentException(
                        $"Block {i}: shell-coded sum {shellSum} != expected {sum}. " +
                        "Input pulses do not round-trip cleanly at this lshift level.");
                }
                SilkShellCoder.Encode(rangeEnc, blockMagnitudes, sum);
            }
        }

        // Emit the low-order bits for any block that used LSB extension.
        for (int i = 0; i < iter; i++)
        {
            int nLs = nLshifts[i];
            if (nLs > 0)
            {
                for (int k = 0; k < shellLen; k++)
                {
                    int abs = Math.Abs((int)pulses[i * shellLen + k]);
                    // Emit the nLs LSB bits from MSB side downward.
                    for (int j = nLs - 1; j >= 0; j--)
                    {
                        int bit = (abs >> j) & 1;
                        rangeEnc.EncodeIcdf(bit, SilkIcdfTables.Lsb, 8);
                    }
                }
                sumPulses[i] |= nLs << 5;
            }
        }

        // Signs.
        int rowStart = SilkIcdfTables.SignOffset(signalType, quantOffsetType);
        int length = (frameLength + shellLen / 2) >> SilkConstants.LOG2_SHELL_CODEC_FRAME_LENGTH;
        Span<byte> icdf = stackalloc byte[2];
        icdf[1] = 0;

        for (int i = 0; i < length; i++)
        {
            int p = sumPulses[i];
            if (p > 0)
            {
                int col = silk_min(p & 0x1F, 6);
                icdf[0] = SilkIcdfTables.Sign[rowStart + col];
                for (int j = 0; j < shellLen; j++)
                {
                    int v = pulses[i * shellLen + j];
                    if (Math.Abs(v) > 0)
                    {
                        // silk_enc_map(a) = (a >> 15) + 1 -> positive -> 1, negative -> 0.
                        int sym = v > 0 ? 1 : 0;
                        rangeEnc.EncodeIcdf(sym, icdf, 8);
                    }
                }
            }
        }
    }
}
