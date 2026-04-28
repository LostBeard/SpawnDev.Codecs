// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 boolean range decoder, GPU-callable form. Bit-exact mirror of
// Vp8BoolDecoder. Symmetric companion to Vp8BoolEncoderGpu - same
// host-as-pure-coordinator design rule applies to the decoder side.
//
// State struct holds the running (value, range, count, bufPos)
// quartet plus the buffer end. Static helper methods take state by
// ref and the input buffer as ArrayView<byte>.
//
// VP8 spec note: end-of-stream is handled by stuffing virtual zero
// bits when the buffer runs out (count gets bumped by LotsOfBits),
// matching libvpx vp8dx_bool_decoder_fill semantics.

using ILGPU;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// In-kernel state for the VP8 boolean range decoder. Mirrors the
/// internal fields of <see cref="Vp8BoolDecoder"/>.
/// </summary>
public struct Vp8BoolDecoderGpuState
{
    /// <summary>Current input buffer offset (absolute).</summary>
    public int BufPos;

    /// <summary>Buffer end offset (exclusive).</summary>
    public int BufEnd;

    /// <summary>Range coder value register (64-bit working register).</summary>
    public ulong Value;

    /// <summary>Bit-position counter; starts at -8, increments on each shift.</summary>
    public int Count;

    /// <summary>Range coder range (8-bit aligned).</summary>
    public uint Range;
}

/// <summary>
/// Static GPU-callable helpers for the VP8 boolean range decoder.
/// Mirrors <see cref="Vp8BoolDecoder"/> bit-for-bit.
/// </summary>
public static class Vp8BoolDecoderGpu
{
    /// <summary>VP8_BD_VALUE_SIZE = 64.</summary>
    public const int BdValueSize = 64;

    /// <summary>VP8_LOTS_OF_BITS sentinel for end-of-stream.</summary>
    public const int LotsOfBits = 0x40000000;

    /// <summary>
    /// Initialize the decoder state to read <paramref name="length"/>
    /// bytes from <paramref name="buf"/> starting at <paramref name="offset"/>.
    /// Mirrors libvpx <c>vp8dx_start_decode</c>.
    /// </summary>
    public static Vp8BoolDecoderGpuState Init(
        ArrayView<byte> buf, int offset, int length)
    {
        var state = new Vp8BoolDecoderGpuState
        {
            BufPos = offset,
            BufEnd = offset + length,
            Value = 0,
            Count = -8,
            Range = 255,
        };
        Fill(ref state, buf);
        return state;
    }

    /// <summary>
    /// Decode one bit given <paramref name="probability"/> in [1, 255].
    /// Mirrors libvpx <c>vp8dx_decode_bool</c>.
    /// </summary>
    public static int DecodeBool(
        ref Vp8BoolDecoderGpuState state,
        ArrayView<byte> buf,
        int probability)
    {
        uint split = 1u + (((state.Range - 1u) * (uint)probability) >> 8);

        if (state.Count < 0) Fill(ref state, buf);

        ulong bigsplit = (ulong)split << (BdValueSize - 8);

        uint bit;
        uint range;
        if (state.Value >= bigsplit)
        {
            range = state.Range - split;
            state.Value -= bigsplit;
            bit = 1;
        }
        else
        {
            range = split;
            bit = 0;
        }

        int shift = LeadingZeros8((byte)range);
        state.Range = range << shift;
        state.Value <<= shift;
        state.Count -= shift;

        return (int)bit;
    }

    /// <summary>
    /// Decode <paramref name="bits"/> raw bits at flat probability 128.
    /// MSB-first. Mirrors libvpx <c>vp8_decode_value</c>.
    /// </summary>
    public static int DecodeValue(
        ref Vp8BoolDecoderGpuState state,
        ArrayView<byte> buf,
        int bits)
    {
        int z = 0;
        for (int b = bits - 1; b >= 0; b--)
            z |= DecodeBool(ref state, buf, 0x80) << b;
        return z;
    }

    /// <summary>
    /// Refill the value register from the input buffer. Mirrors libvpx
    /// <c>vp8dx_bool_decoder_fill</c>.
    /// </summary>
    private static void Fill(ref Vp8BoolDecoderGpuState state, ArrayView<byte> buf)
    {
        int shift = BdValueSize - 8 - (state.Count + 8);
        int bytesLeft = state.BufEnd - state.BufPos;
        int bitsLeft = bytesLeft * 8;
        int x = shift + 8 - bitsLeft;
        int loopEnd = 0;

        if (x >= 0)
        {
            state.Count += LotsOfBits;
            loopEnd = x;
        }

        if (x < 0 || bitsLeft != 0)
        {
            while (shift >= loopEnd)
            {
                state.Count += 8;
                state.Value |= (ulong)buf[state.BufPos] << shift;
                state.BufPos++;
                shift -= 8;
            }
        }
    }

    /// <summary>
    /// Count leading zeros of a byte. Returns 0 for zero input
    /// (matches the CPU helper).
    /// </summary>
    private static int LeadingZeros8(byte b)
    {
        if (b == 0) return 0;
        if ((b & 0x80) != 0) return 0;
        if ((b & 0x40) != 0) return 1;
        if ((b & 0x20) != 0) return 2;
        if ((b & 0x10) != 0) return 3;
        if ((b & 0x08) != 0) return 4;
        if ((b & 0x04) != 0) return 5;
        if ((b & 0x02) != 0) return 6;
        return 7; // b == 0x01
    }
}
