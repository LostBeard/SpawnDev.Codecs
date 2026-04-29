// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// FLAC MSB-first bit reader, GPU-callable form. Bit-exact mirror of
// FlacBitReader for in-kernel use by the upcoming FlacDecoderGpu
// pipeline.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>In-kernel state for the FLAC bit reader.</summary>
public struct FlacBitReaderGpuState
{
    /// <summary>Current byte offset in the input view.</summary>
    public int BytePos;
    /// <summary>Bits consumed in the current byte (0..7).</summary>
    public int BitPos;
    /// <summary>Length of the input view in bytes (set at Init time).</summary>
    public int Length;
}

/// <summary>
/// GPU-callable FLAC bit reader. Mirrors <see cref="FlacBitReader"/>
/// bit-for-bit. Caller passes the input ArrayView&lt;byte&gt; on every call.
/// </summary>
public static class FlacBitReaderGpu
{
    /// <summary>Initialize a fresh reader state for the given input length.</summary>
    public static FlacBitReaderGpuState Init(int length) => new FlacBitReaderGpuState
    {
        BytePos = 0,
        BitPos = 0,
        Length = length,
    };

    /// <summary>Bits remaining in the input buffer.</summary>
    public static int BitsRemaining(in FlacBitReaderGpuState state)
        => (state.Length - state.BytePos) * 8 - state.BitPos;

    /// <summary>True once all bytes consumed.</summary>
    public static bool IsEnd(in FlacBitReaderGpuState state)
        => state.BytePos >= state.Length;

    /// <summary>
    /// Read the next <paramref name="nBits"/> bits as an unsigned integer.
    /// </summary>
    public static uint ReadBits(
        ref FlacBitReaderGpuState state, ArrayView<byte> data, int nBits)
    {
        uint result = 0;
        while (nBits > 0)
        {
            int available = 8 - state.BitPos;
            int take = available < nBits ? available : nBits;
            int shift = available - take;
            int b = data[state.BytePos];
            uint bits = (uint)((b >> shift) & ((1 << take) - 1));
            result = (result << take) | bits;
            state.BitPos += take;
            nBits -= take;
            if (state.BitPos == 8)
            {
                state.BytePos++;
                state.BitPos = 0;
            }
        }
        return result;
    }

    /// <summary>Read the next <paramref name="nBits"/> bits as a signed integer (two's complement).</summary>
    public static int ReadBitsSigned(
        ref FlacBitReaderGpuState state, ArrayView<byte> data, int nBits)
    {
        uint u = ReadBits(ref state, data, nBits);
        if (nBits < 32)
        {
            uint signMask = 1u << (nBits - 1);
            if ((u & signMask) != 0)
            {
                uint extend = ~((1u << nBits) - 1);
                u |= extend;
            }
        }
        return unchecked((int)u);
    }

    /// <summary>Read a unary-coded integer: count leading zero bits then consume the terminating 1.</summary>
    public static int ReadUnary(ref FlacBitReaderGpuState state, ArrayView<byte> data)
    {
        int count = 0;
        while (!IsEnd(in state))
        {
            uint bit = ReadBits(ref state, data, 1);
            if (bit == 1) return count;
            count++;
        }
        return count;
    }

    /// <summary>Skip to the next byte boundary.</summary>
    public static void AlignToByte(ref FlacBitReaderGpuState state)
    {
        if (state.BitPos != 0)
        {
            state.BytePos++;
            state.BitPos = 0;
        }
    }
}
