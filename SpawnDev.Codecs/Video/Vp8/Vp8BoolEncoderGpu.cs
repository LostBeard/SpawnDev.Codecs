// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 boolean range encoder, GPU-callable form. Bit-exact mirror of
// Vp8BoolEncoder. Used by GPU entropy kernels to keep encoded
// bitstream output GPU-resident.
//
// The CPU version uses List<byte> for the output buffer (grows). GPU
// kernels can't allocate dynamically, so the caller pre-sizes a
// worst-case-bounded ArrayView<byte> and the encoder maintains an
// offset into it. The state struct + static helper methods pattern
// is so ILGPU can pass state by ref into the helper methods.
//
// Carry propagation scans backward through the already-written
// portion of the output buffer and bumps 0xFF -> 0x00 carries until
// it hits a non-0xFF byte that gets incremented. This is sequential
// within a single bool coder; ports cleanly to a per-thread loop.

using ILGPU;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// In-kernel state for the VP8 boolean range encoder. Mirrors the
/// internal fields of <see cref="Vp8BoolEncoder"/>. Caller maintains
/// a per-thread <see cref="ArrayView{Byte}"/> output buffer and an
/// <see cref="OutLen"/> offset into it. See the static helpers in
/// <see cref="Vp8BoolEncoderGpu"/> for the encode operations.
/// </summary>
public struct Vp8BoolEncoderGpuState
{
    /// <summary>Range coder low value (24-bit working register).</summary>
    public uint LowValue;

    /// <summary>Range coder range (8-bit-aligned).</summary>
    public uint Range;

    /// <summary>Bit-position counter; starts at -24, increments on each
    /// renormalize shift, emits a byte when it reaches 0.</summary>
    public int Count;

    /// <summary>Number of bytes written to the output view so far.</summary>
    public long OutLen;
}

/// <summary>
/// Static GPU-callable helpers for the VP8 boolean range encoder.
/// Mirrors <see cref="Vp8BoolEncoder"/> bit-for-bit. Caller passes a
/// <see cref="Vp8BoolEncoderGpuState"/> by ref and an
/// <see cref="ArrayView{Byte}"/> output buffer.
/// </summary>
public static class Vp8BoolEncoderGpu
{
    /// <summary>Initial state: range=255, count=-24, lowvalue=0, outLen=0.</summary>
    public static Vp8BoolEncoderGpuState Init() => new Vp8BoolEncoderGpuState
    {
        LowValue = 0,
        Range = 255,
        Count = -24,
        OutLen = 0,
    };

    /// <summary>
    /// Encode one bit. Probability is the encoder's estimate of the
    /// bit being 0, scaled by 256 (i.e. valid range [1, 255]).
    /// Mirrors libvpx <c>vp8_encode_bool</c> bit-for-bit.
    /// </summary>
    public static void EncodeBool(
        ref Vp8BoolEncoderGpuState state,
        ArrayView<byte> outBuf,
        int bit,
        int probability)
    {
        uint split = 1u + (((state.Range - 1u) * (uint)probability) >> 8);
        uint range = split;
        uint lowvalue = state.LowValue;
        int count = state.Count;

        if (bit != 0)
        {
            lowvalue += split;
            range = state.Range - split;
        }

        int shift = LeadingZeros8((byte)range);
        range <<= shift;
        count += shift;

        if (count >= 0)
        {
            int offset = shift - count;

            // Carry propagation. The bit just shifted out (top bit of
            // (lowvalue << (offset-1))) is 1 -> propagate carry backward
            // through 0xFF bytes in the already-written output buffer.
            if ((((ulong)lowvalue) << (offset - 1) & 0x80000000UL) != 0)
            {
                long x = state.OutLen - 1;
                while (x >= 0 && outBuf[x] == 0xFF)
                {
                    outBuf[x] = 0;
                    x--;
                }
                if (x >= 0) outBuf[x] = (byte)(outBuf[x] + 1);
            }

            outBuf[state.OutLen] = (byte)((lowvalue >> (24 - offset)) & 0xFF);
            state.OutLen += 1;

            shift = count;
            lowvalue = (lowvalue << offset) & 0xFFFFFFu;
            count -= 8;
        }

        lowvalue <<= shift;

        state.LowValue = lowvalue;
        state.Range = range;
        state.Count = count;
    }

    /// <summary>
    /// Encode <paramref name="bits"/> raw bits MSB-first at flat
    /// probability 128. Mirrors libvpx <c>vp8_encode_value</c>.
    /// </summary>
    public static void EncodeValue(
        ref Vp8BoolEncoderGpuState state,
        ArrayView<byte> outBuf,
        int data,
        int bits)
    {
        for (int b = bits - 1; b >= 0; b--)
            EncodeBool(ref state, outBuf, (data >> b) & 1, 0x80);
    }

    /// <summary>
    /// Finalize the bitstream. Emits 32 trailing zero-prob-128 bits to
    /// flush remaining state. Caller reads <c>state.OutLen</c> as the
    /// final byte count. Mirrors libvpx <c>vp8_stop_encode</c>.
    /// </summary>
    public static void Stop(
        ref Vp8BoolEncoderGpuState state,
        ArrayView<byte> outBuf)
    {
        for (int i = 0; i < 32; i++) EncodeBool(ref state, outBuf, 0, 128);
    }

    /// <summary>
    /// Count leading zeros of a byte's value. Returns 0 for a zero
    /// input (matches the CPU helper). For non-zero, returns the
    /// number of leading zero bits in the byte (so b=1 -> 7, b=128 -> 0).
    /// </summary>
    private static int LeadingZeros8(byte b)
    {
        if (b == 0) return 0;
        // ILGPU emits CLZ on backends that support it; otherwise the
        // backend uses a fallback. Returns leading zeros of the
        // 32-bit unsigned, less 24 to map back to the 8-bit space.
        // libvpx's vp8_norm[range] table is equivalent to this for
        // range != 0. The CPU helper uses BitOperations.LeadingZeroCount.
        // For ILGPU compatibility we use a manual classifier.
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
