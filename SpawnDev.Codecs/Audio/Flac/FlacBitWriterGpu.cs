// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// FLAC MSB-first bit writer, GPU-callable form. Bit-exact mirror of
// FlacBitWriter for in-kernel use by the upcoming FlacEncoderGpu
// pipeline.
//
// State holds the (currentByte, bitPos, outLen) triple. Output bytes
// land in a caller-provided ArrayView&lt;byte&gt;. Caller pre-allocates
// the buffer worst-case-bounded (e.g. blockSize * channels * bps / 8
// + frame header overhead).

using ILGPU;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>In-kernel state for the FLAC bit writer.</summary>
public struct FlacBitWriterGpuState
{
    /// <summary>Bits accumulated in the current byte (high bits valid).</summary>
    public int CurrentByte;
    /// <summary>Bits written into <see cref="CurrentByte"/> so far (0..7).</summary>
    public int BitPos;
    /// <summary>Number of complete bytes written to the output view.</summary>
    public long OutLen;
}

/// <summary>
/// GPU-callable FLAC bit writer. Mirrors <see cref="FlacBitWriter"/>
/// bit-for-bit. State + static helper pattern so ILGPU can pass state
/// by ref into kernel-local invocations.
/// </summary>
public static class FlacBitWriterGpu
{
    /// <summary>Initialize a fresh writer state.</summary>
    public static FlacBitWriterGpuState Init() => new FlacBitWriterGpuState
    {
        CurrentByte = 0,
        BitPos = 0,
        OutLen = 0,
    };

    /// <summary>
    /// Write <paramref name="bits"/> bits of <paramref name="value"/>
    /// MSB-first to <paramref name="outBuf"/>. <paramref name="bits"/>
    /// must be in [0, 32].
    /// </summary>
    public static void Write(
        ref FlacBitWriterGpuState state, ArrayView<byte> outBuf,
        uint value, int bits)
    {
        while (bits > 0)
        {
            int free = 8 - state.BitPos;
            int take = free < bits ? free : bits;
            int shift = bits - take;
            uint mask = take == 32 ? 0xFFFFFFFFu : ((1u << take) - 1);
            uint chunk = (value >> shift) & mask;
            state.CurrentByte = (state.CurrentByte << take) | (int)chunk;
            state.BitPos += take;
            bits -= take;
            if (state.BitPos == 8)
            {
                outBuf[state.OutLen] = (byte)state.CurrentByte;
                state.OutLen++;
                state.CurrentByte = 0;
                state.BitPos = 0;
            }
        }
    }

    /// <summary>Write a signed integer in two's-complement form at <paramref name="bits"/> bits.</summary>
    public static void WriteSigned(
        ref FlacBitWriterGpuState state, ArrayView<byte> outBuf,
        int value, int bits)
    {
        uint mask = bits == 32 ? 0xFFFFFFFFu : ((1u << bits) - 1);
        uint raw = (uint)value & mask;
        Write(ref state, outBuf, raw, bits);
    }

    /// <summary>Write a unary code: <paramref name="zeroCount"/> zero bits followed by a 1.</summary>
    public static void WriteUnary(
        ref FlacBitWriterGpuState state, ArrayView<byte> outBuf,
        int zeroCount)
    {
        for (int i = 0; i < zeroCount; i++) Write(ref state, outBuf, 0u, 1);
        Write(ref state, outBuf, 1u, 1);
    }

    /// <summary>Flush partial-byte state to output. Trailing bits are zero-padded.</summary>
    public static void AlignToByte(
        ref FlacBitWriterGpuState state, ArrayView<byte> outBuf)
    {
        if (state.BitPos != 0)
        {
            outBuf[state.OutLen] = (byte)(state.CurrentByte << (8 - state.BitPos));
            state.OutLen++;
            state.CurrentByte = 0;
            state.BitPos = 0;
        }
    }
}
