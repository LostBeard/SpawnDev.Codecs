// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable port of SilkPulsesDecoder.Decode (libopus
// silk/decode_pulses.c + silk/code_signs.c). Decodes the full excitation
// (pulses) for a SILK frame: rate level -> per-block pulse counts (with
// LSB-extension escape for large counts) -> shell coder per block ->
// LSB bits on top of shell output -> sign bits.
//
// Composes existing GPU primitives:
//   - OpusRangeDecoderGpu.DecodeIcdf (for rate level, per-block counts,
//     LSB extensions, sign bits)
//   - SilkShellCoderGpu.Decode (for per-block 16-pulse magnitudes)
//
// Sequential per-stream because every stage shares range decoder state.
// Single thread per stream; multi-channel decode parallelizes across
// threads.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>
/// Body-struct kernel parameter for SilkPulsesDecoderGpu callers.
/// Bundles all per-stream iCDF tables + scratches + the nested
/// <see cref="SilkShellCoderTables"/> for the shell-decode step.
/// </summary>
public struct SilkPulsesInputs
{
    /// <summary>The four iCDF tables, packed (offsets below; <see cref="PackedDeviceBuffer{T}"/>).</summary>
    public ArrayView<byte> Bytes;
    /// <summary>silk_rate_levels_iCDF offset in <see cref="Bytes"/>.</summary>
    public int RateLevelsIcdfAt;
    /// <summary>silk_pulses_per_block_iCDF offset.</summary>
    public int PulsesPerBlockIcdfAt;
    /// <summary>silk_lsb_iCDF offset.</summary>
    public int LsbIcdfAt;
    /// <summary>silk_sign_iCDF offset.</summary>
    public int SignIcdfAt;
    /// <summary>Where the shell code tables sit in <see cref="Bytes"/> (offsets only: a second view of the same buffer would be a second read_write binding, which WebGPU rejects).</summary>
    public SilkShellCoderTableOffsets ShellTables;
    /// <summary>Int scratch: sumPulses[MaxNbShellBlocks] and nLshifts[MaxNbShellBlocks] regions.</summary>
    public ArrayView<int> Scratch;
    /// <summary>sumPulses region offset in <see cref="Scratch"/>.</summary>
    public int SumPulsesScratchAt;
    /// <summary>nLshifts region offset in <see cref="Scratch"/>.</summary>
    public int NLshiftsScratchAt;

    /// <summary>
    /// Appends the standard libopus pulse iCDFs + shell code tables to <paramref name="bytes"/> and reserves the two
    /// scratch regions in <paramref name="ints"/>; then <see cref="Bind"/> the uploaded buffers.
    /// </summary>
    public static SilkPulsesInputs AddStandard(PackedDeviceBuffer<byte> bytes, PackedDeviceBuffer<int> ints) => new SilkPulsesInputs
    {
        RateLevelsIcdfAt = bytes.Add(SilkIcdfTables.RateLevels),
        PulsesPerBlockIcdfAt = bytes.Add(SilkIcdfTables.PulsesPerBlock),
        LsbIcdfAt = bytes.Add(SilkIcdfTables.Lsb),
        SignIcdfAt = bytes.Add(SilkIcdfTables.Sign),
        ShellTables = SilkShellCoderTableOffsets.AddStandard(bytes),
        SumPulsesScratchAt = ints.Reserve(SilkPulsesDecoderGpu.MaxNbShellBlocks),
        NLshiftsScratchAt = ints.Reserve(SilkPulsesDecoderGpu.MaxNbShellBlocks),
    };

    /// <summary>Points the struct at the uploaded byte buffer (which holds the shell tables too) and int scratch.</summary>
    public void Bind(ArrayView<byte> bytes, ArrayView<int> scratch)
    {
        Bytes = bytes;
        Scratch = scratch;
    }
}

/// <summary>
/// GPU-callable orchestrator for the SILK pulse (excitation) decode.
/// Mirror of `SilkPulsesDecoder.Decode`.
/// </summary>
public static class SilkPulsesDecoderGpu
{
    /// <summary>SilkConstants.SHELL_CODEC_FRAME_LENGTH = 16.</summary>
    public const int ShellCodecFrameLength = 16;
    /// <summary>SilkConstants.LOG2_SHELL_CODEC_FRAME_LENGTH = 4.</summary>
    public const int Log2ShellCodecFrameLength = 4;
    /// <summary>SilkConstants.MAX_NB_SHELL_BLOCKS = 20.</summary>
    public const int MaxNbShellBlocks = 20;
    /// <summary>SilkConstants.SILK_MAX_PULSES = 16.</summary>
    public const int SilkMaxPulses = 16;
    /// <summary>SilkConstants.N_RATE_LEVELS = 10.</summary>
    public const int NRateLevels = 10;
    /// <summary>silk_rate_levels_iCDF entries per row.</summary>
    public const int RateLevelsEntriesPerType = 9;
    /// <summary>silk_pulses_per_block_iCDF entries per row.</summary>
    public const int PulsesPerBlockEntriesPerRow = 18;
    /// <summary>silk_sign_iCDF entries per row.</summary>
    public const int SignEntriesPerRow = 7;

    /// <summary>
    /// Decode the signed excitation for a SILK frame.
    /// </summary>
    /// <param name="state">Range decoder state (advanced in place).</param>
    /// <param name="buf">Encoded packet buffer.</param>
    /// <param name="bufStart">Offset of the packet in <paramref name="buf"/>.</param>
    /// <param name="storage">Length of the packet in bytes.</param>
    /// <param name="inputs">Body-struct holding all iCDF tables + shell tables + scratches.</param>
    /// <param name="signalType">SILK signal type (0/1/2).</param>
    /// <param name="quantOffsetType">SILK quantizer offset type (0/1).</param>
    /// <param name="frameLength">Frame length in samples (multiple of
    /// <see cref="ShellCodecFrameLength"/>, or 120 for 15ms MB frames).</param>
    /// <param name="pulsesOut">Output ArrayView&lt;short&gt; (length must be aligned-up
    /// to ShellCodecFrameLength boundary &gt;= frameLength).</param>
    /// <param name="pulsesOutBase">Offset.</param>
    public static void Decode(
        ref OpusRangeDecoderGpuState state,
        ArrayView<byte> buf, int bufStart, uint storage,
        SilkPulsesInputs inputs,
        int signalType, int quantOffsetType, int frameLength,
        ArrayView<short> pulsesOut, long pulsesOutBase)
    {
        int shellLen = ShellCodecFrameLength;

        // 1. Rate-level selection.
        int rateLevelOffset = (signalType >> 1) * RateLevelsEntriesPerType;
        int rateLevelIndex = OpusRangeDecoderGpu.DecodeIcdf(
            ref state, buf, bufStart, storage,
            inputs.Bytes, inputs.RateLevelsIcdfAt + rateLevelOffset, 8);

        // 2. Per-block pulse counts with LSB-extension escape.
        int iter = frameLength >> Log2ShellCodecFrameLength;
        // 120-sample 15ms MB frame: extra partial shell block.
        int alignedLen = (frameLength + shellLen - 1) & ~(shellLen - 1);
        if (iter * shellLen < frameLength) iter++;

        long cdfOffset = (long)rateLevelIndex * PulsesPerBlockEntriesPerRow;
        long escapeRowStart = (long)(NRateLevels - 1) * PulsesPerBlockEntriesPerRow;

        for (int i = 0; i < iter; i++)
        {
            inputs.Scratch[inputs.NLshiftsScratchAt + i] = 0;
            int sum = OpusRangeDecoderGpu.DecodeIcdf(
                ref state, buf, bufStart, storage,
                inputs.Bytes, inputs.PulsesPerBlockIcdfAt + cdfOffset, 8);
            while (sum == SilkMaxPulses + 1)
            {
                inputs.Scratch[inputs.NLshiftsScratchAt + i]++;
                int adjust = inputs.Scratch[inputs.NLshiftsScratchAt + i] == 10 ? 1 : 0;
                long escOffset = escapeRowStart + adjust;
                sum = OpusRangeDecoderGpu.DecodeIcdf(
                    ref state, buf, bufStart, storage,
                    inputs.Bytes, inputs.PulsesPerBlockIcdfAt + escOffset, 8);
            }
            inputs.Scratch[inputs.SumPulsesScratchAt + i] = sum;
        }

        // 3. Shell-decode magnitudes per block (or zero for empty blocks).
        for (int i = 0; i < iter; i++)
        {
            long blockBase = pulsesOutBase + (long)i * shellLen;
            int sum = inputs.Scratch[inputs.SumPulsesScratchAt + i];
            if (sum > 0)
            {
                SilkShellCoderGpu.Decode(
                    ref state, buf, bufStart, storage,
                    inputs.Bytes, inputs.ShellTables, sum,
                    pulsesOut, blockBase);
            }
            else
            {
                for (int k = 0; k < shellLen; k++)
                    pulsesOut[blockBase + k] = 0;
            }
        }

        // 4. LSB-bit extension.
        for (int i = 0; i < iter; i++)
        {
            int nLs = inputs.Scratch[inputs.NLshiftsScratchAt + i];
            if (nLs > 0)
            {
                long blockBase = pulsesOutBase + (long)i * shellLen;
                for (int k = 0; k < shellLen; k++)
                {
                    int absQ = pulsesOut[blockBase + k];
                    for (int j = 0; j < nLs; j++)
                    {
                        absQ <<= 1;
                        absQ += OpusRangeDecoderGpu.DecodeIcdf(
                            ref state, buf, bufStart, storage,
                            inputs.Bytes, inputs.LsbIcdfAt, 8);
                    }
                    pulsesOut[blockBase + k] = (short)absQ;
                }
                // Pack nLshifts into bits 5..9 of sumPulses for the sign decoder.
                inputs.Scratch[inputs.SumPulsesScratchAt + i] |= nLs << 5;
            }
        }

        // 5. Sign decode.
        long signRowStart = (long)SignEntriesPerRow * (quantOffsetType + 2 * signalType);
        int signLength = (frameLength + shellLen / 2) >> Log2ShellCodecFrameLength;
        for (int i = 0; i < signLength; i++)
        {
            int p = inputs.Scratch[inputs.SumPulsesScratchAt + i];
            if (p > 0)
            {
                int col = (p & 0x1F) < 6 ? (p & 0x1F) : 6;
                long blockBase = pulsesOutBase + (long)i * shellLen;
                for (int j = 0; j < shellLen; j++)
                {
                    if (pulsesOut[blockBase + j] > 0)
                    {
                        // Inline 2-symbol DecodeIcdf:
                        //   icdf[0] = SignIcdf[signRowStart + col], icdf[1] = 0
                        // Read decoded bit -> 0 means sign = -1, 1 means sign = +1.
                        int decoded = DecodeSignBit(
                            ref state, buf, bufStart, storage,
                            inputs.Bytes, inputs.SignIcdfAt + signRowStart + col);
                        int sign = (decoded << 1) - 1;
                        pulsesOut[blockBase + j] = (short)((int)pulsesOut[blockBase + j] * sign);
                    }
                }
            }
        }
    }

    /// <summary>
    /// Inline 2-symbol DecodeIcdf for sign bits. Avoids needing a
    /// stack-allocated 2-byte iCDF buffer; reads <paramref name="signIcdf"/>
    /// at <paramref name="rowOffset"/> and treats the second entry as 0
    /// (sentinel terminator). Returns 0 or 1.
    /// </summary>
    private static int DecodeSignBit(
        ref OpusRangeDecoderGpuState state,
        ArrayView<byte> buf, int bufStart, uint storage,
        ArrayView<byte> signIcdf, long rowOffset)
    {
        // Mirrors OpusRangeDecoderGpu.DecodeIcdf inline for a 2-symbol
        // iCDF with hardcoded second-entry-zero terminator.
        uint s = state.Rng;
        uint d = state.Val;
        uint r = s >> 8;
        uint t;
        // First symbol (ret=0).
        t = s;
        s = r * signIcdf[rowOffset];
        if (d < s)
        {
            // Second symbol (ret=1, icdf[1]==0).
            t = s;
            s = 0;
            // (Loop condition d < 0 is false for uint d; exit naturally.)
            state.Val = d - s;
            state.Rng = t - s;
            // Normalize.
            NormalizeInline(ref state, buf, bufStart, storage);
            return 1;
        }
        state.Val = d - s;
        state.Rng = t - s;
        NormalizeInline(ref state, buf, bufStart, storage);
        return 0;
    }

    /// <summary>Inline copy of OpusRangeDecoderGpu's Normalize (which is private).</summary>
    private static void NormalizeInline(
        ref OpusRangeDecoderGpuState state,
        ArrayView<byte> buf, int bufStart, uint storage)
    {
        // EC_CODE_BOT = 0x800000; EC_SYM_BITS = 8; EC_CODE_TOP = 0x80000000.
        const uint EC_CODE_BOT = 0x00800000u;
        const uint EC_CODE_TOP = 0x80000000u;
        const int EC_SYM_BITS = 8;
        const int EC_CODE_EXTRA = 7;
        const uint EC_SYM_MAX = 0xFFu;

        while (state.Rng <= EC_CODE_BOT)
        {
            state.NBitsTotal += EC_SYM_BITS;
            state.Rng <<= EC_SYM_BITS;
            int sym = state.Rem;
            int newRem = 0;
            if (state.Offs < storage)
            {
                newRem = buf[bufStart + (int)state.Offs];
                state.Offs++;
            }
            state.Rem = newRem;
            sym = (sym << EC_SYM_BITS | newRem) >> (EC_SYM_BITS - EC_CODE_EXTRA);
            state.Val = ((state.Val << EC_SYM_BITS) + (EC_SYM_MAX & (uint)~sym))
                & (EC_CODE_TOP - 1u);
        }
    }
}
