// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 boolean arithmetic encoder. Structural port of libvpx
// vp8/encoder/boolhuff.{h,c} to clean C#. RFC 6386 sec 7 (decode side
// inverted - encoder state matches the decoder's renormalize/fill pattern
// in reverse).
//
// Upstream Copyright (c) 2010 The WebM project authors.
// Upstream license: BSD 3-Clause. See NOTICE.md.
//
// Pairs with <see cref="Vp8BoolDecoder"/>. Round-trip-tested in
// vp8_bool_coder_roundtrip.cs.
//
// Difference from the C reference: output buffer is a C# List<byte> so we
// don't have to size-pre-allocate. The carry-propagation logic and the
// 32-zero-bit flush in Stop() are byte-for-byte identical.

using System.Numerics;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// VP8 boolean arithmetic encoder. Pairs with <see cref="Vp8BoolDecoder"/>.
/// Stateful, single partition.
/// </summary>
public sealed class Vp8BoolEncoder
{
    private readonly List<byte> _buf = new();
    private uint _lowvalue;
    private uint _range;
    private int _count;

    /// <summary>Construct an empty encoder.</summary>
    public Vp8BoolEncoder()
    {
        Reset();
    }

    /// <summary>Reset encoder state. Mirrors <c>vp8_start_encode</c>.</summary>
    public void Reset()
    {
        _buf.Clear();
        _lowvalue = 0;
        _range = 255;
        _count = -24;
    }

    /// <summary>
    /// Encode one bit with probability <paramref name="probability"/> in [1, 255]
    /// (encoder's estimate of the bit being 0, scaled by 256). Mirrors libvpx
    /// <c>vp8_encode_bool</c>.
    /// </summary>
    public void EncodeBool(int bit, int probability)
    {
        if (probability < 1 || probability > 255)
            throw new ArgumentOutOfRangeException(nameof(probability), "must be in [1, 255]");

        uint split = 1u + (((_range - 1u) * (uint)probability) >> 8);
        uint range = split;
        uint lowvalue = _lowvalue;
        int count = _count;

        if (bit != 0)
        {
            lowvalue += split;
            range = _range - split;
        }

        int shift = LeadingZeros8((byte)range);
        range <<= shift;
        count += shift;

        if (count >= 0)
        {
            int offset = shift - count;

            // Carry propagation: if the bit just shifted out is 1, propagate
            // a carry backward through the existing 0xFF bytes.
            if ((((ulong)lowvalue) << (offset - 1) & 0x80000000UL) != 0)
            {
                int x = _buf.Count - 1;
                while (x >= 0 && _buf[x] == 0xFF)
                {
                    _buf[x] = 0;
                    x--;
                }
                if (x >= 0) _buf[x] = (byte)(_buf[x] + 1);
            }

            _buf.Add((byte)((lowvalue >> (24 - offset)) & 0xFF));

            shift = count;
            // libvpx: lowvalue = (lowvalue << offset) & 0xffffff
            lowvalue = (lowvalue << offset) & 0xFFFFFFu;
            count -= 8;
        }

        // libvpx: lowvalue <<= shift (final shift is the renorm amount when no
        // emit; when emit happened, shift was reassigned to `count` so the
        // total lowvalue shift across the iteration sums to the original
        // vp8_norm[range] amount).
        lowvalue <<= shift;

        _lowvalue = lowvalue;
        _range = range;
        _count = count;
    }

    /// <summary>Encode <paramref name="bits"/> raw bits MSB-first at flat
    /// probability 128. Mirrors libvpx <c>vp8_encode_value</c>.</summary>
    public void EncodeValue(int data, int bits)
    {
        if (bits < 0 || bits > 31) throw new ArgumentOutOfRangeException(nameof(bits));
        for (int b = bits - 1; b >= 0; b--)
            EncodeBool((data >> b) & 1, 0x80);
    }

    /// <summary>
    /// Finalize the bitstream and return the encoded bytes. Mirrors libvpx
    /// <c>vp8_stop_encode</c> - emits 32 trailing zeros at prob 128 to flush
    /// the remaining state into the output buffer.
    /// </summary>
    public byte[] Stop()
    {
        for (int i = 0; i < 32; i++) EncodeBool(0, 128);
        return _buf.ToArray();
    }

    private static int LeadingZeros8(byte b)
    {
        if (b == 0) return 0;
        return BitOperations.LeadingZeroCount((uint)b) - 24;
    }
}
