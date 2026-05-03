// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 boolean arithmetic decoder. Structural port of libvpx
// vp8/decoder/dboolhuff.{h,c} to clean C#. RFC 6386 sec 7.
//
// Upstream Copyright (c) 2010 The WebM project authors.
// Upstream license: BSD 3-Clause. See NOTICE.md.
// Upstream source: https://github.com/webmproject/libvpx
//
// VP8's bool decoder is the simpler ancestor of VP9's `Vp9BoolDecoder`:
//   - 8-bit probability (vs Q15 in AV1)
//   - 8-bit symbol size (range, split)
//   - Renormalization via vp8_norm[range] (leading-zero count clamped to 7)
//   - End-of-stream handled by stuffing virtual bits when the buffer runs out
//     (count gets incremented by VP8_LOTS_OF_BITS so subsequent bits are zero)
//
// Algorithm: Boolean arithmetic coding (range coding with binary alphabet).
// Each call decodes one bit given a probability p in [1, 255]; the actual
// probability of "1" is (256 - p) / 256.
//
// Sequential by nature (state-dependent); pure C# CPU.

using System.Numerics;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// VP8 boolean decoder. RFC 6386 sec 7. Stateful and not thread-safe.
/// One instance per partition (VP8 streams have an uncompressed-header
/// partition followed by 1, 2, 4, or 8 token partitions, each requiring
/// its own bool decoder instance).
/// </summary>
public sealed class Vp8BoolDecoder
{
    /// <summary>VP8_BD_VALUE_SIZE from libvpx (sizeof(size_t) * 8 = 32 or 64).</summary>
    public const int BdValueSize = 64;
    /// <summary>VP8_LOTS_OF_BITS from libvpx - sentinel for end-of-stream.</summary>
    public const int LotsOfBits = 0x40000000;

    private readonly byte[] _buf;
    private readonly int _bufEnd;     // absolute offset (exclusive)
    private int _bufPos;              // absolute read offset

    private ulong _value;
    private int _count;
    private uint _range;

    /// <summary>Construct decoder over the entire buffer.</summary>
    public Vp8BoolDecoder(byte[] buf) : this(buf, 0, buf?.Length ?? 0) { }

    /// <summary>
    /// Construct decoder reading <paramref name="length"/> bytes starting at
    /// <paramref name="offset"/>. Mirrors libvpx <c>vp8dx_start_decode</c>.
    /// </summary>
    public Vp8BoolDecoder(byte[] buf, int offset, int length)
    {
        ArgumentNullException.ThrowIfNull(buf);
        if ((uint)offset > (uint)buf.Length) throw new ArgumentOutOfRangeException(nameof(offset));
        if (length < 0 || offset + length > buf.Length) throw new ArgumentOutOfRangeException(nameof(length));

        _buf = buf;
        _bufPos = offset;
        _bufEnd = offset + length;
        _value = 0;
        _count = -8;
        _range = 255;
        Fill();
    }

    /// <summary>
    /// Decode one bit given <paramref name="probability"/> in [1, 255]. The
    /// probability is the encoder's estimate of the bit being 0, scaled by
    /// 256. Mirrors libvpx <c>vp8dx_decode_bool</c>.
    /// </summary>
    public int DecodeBool(int probability)
    {
        if (probability < 1 || probability > 255)
            throw new ArgumentOutOfRangeException(nameof(probability), "must be in [1, 255]");

        uint split = 1u + (((_range - 1u) * (uint)probability) >> 8);

        if (_count < 0) Fill();

        ulong bigsplit = (ulong)split << (BdValueSize - 8);

        uint bit;
        uint range;
        if (_value >= bigsplit)
        {
            range = _range - split;
            _value -= bigsplit;
            bit = 1;
        }
        else
        {
            range = split;
            bit = 0;
        }

        // Renormalize: shift range and value left until range >= 128.
        int shift = LeadingZeros8((byte)range);
        _range = range << shift;
        _value <<= shift;
        _count -= shift;

        return (int)bit;
    }

    /// <summary>
    /// Decode <paramref name="bits"/> raw bits at flat probability 128.
    /// Equivalent to libvpx <c>vp8_decode_value</c>. Bit reads are MSB-first.
    /// </summary>
    public int DecodeValue(int bits)
    {
        if (bits < 0 || bits > 31) throw new ArgumentOutOfRangeException(nameof(bits));
        int z = 0;
        for (int b = bits - 1; b >= 0; b--)
            z |= DecodeBool(0x80) << b;
        return z;
    }

    /// <summary>True once the user buffer is exhausted AND a subsequent bit
    /// has been requested past the end. Mirrors <c>vp8dx_bool_error</c>.</summary>
    public bool Error => _count > BdValueSize && _count < LotsOfBits;

    /// <summary>Read pointer offset within the original buffer.</summary>
    public int Position => _bufPos;

    private void Fill()
    {
        int shift = BdValueSize - 8 - (_count + 8);
        int bytesLeft = _bufEnd - _bufPos;
        int bitsLeft = bytesLeft * 8;
        int x = shift + 8 - bitsLeft;
        int loopEnd = 0;

        if (x >= 0)
        {
            _count += LotsOfBits;
            loopEnd = x;
        }

        if (x < 0 || bitsLeft != 0)
        {
            while (shift >= loopEnd)
            {
                _count += 8;
                _value |= (ulong)_buf[_bufPos] << shift;
                _bufPos++;
                shift -= 8;
            }
        }
    }

    /// <summary>
    /// libvpx vp8_norm[byte] - leading-zero count of the byte, clamped so
    /// that vp8_norm[0] = 0. Equivalent to BitOperations.LeadingZeroCount(b)
    /// applied to the byte's 8-bit representation, with vp8_norm[0] = 0
    /// (libvpx: when range collapses to 0, decoder is in error state and
    /// the result doesn't matter - shifted value will just be 0).
    /// </summary>
    private static int LeadingZeros8(byte b)
    {
        if (b == 0) return 0;
        // BitOperations.LeadingZeroCount on uint b returns leading zeros in the
        // 32-bit representation (0..31). We want leading zeros in the 8-bit
        // representation (0..7).
        return BitOperations.LeadingZeroCount((uint)b) - 24;
    }
}
