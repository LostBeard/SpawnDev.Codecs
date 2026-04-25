// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// MSB-first bit reader for VP9 uncompressed headers. VP9 packs bits
// high-bit-first within each byte and reads multi-bit fields as
// big-endian unsigned integers (e.g. frame_width_minus_1 is f(16) with
// the high byte first). This reader matches that convention exactly.
//
// Scoped to Vp9 for now. If AV1 or the future VP8 parser need the same
// shape, promote to Codecs.Common without rewriting callers.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>MSB-first bit reader over a byte span. Read-only; no seeking backwards.</summary>
internal ref struct Vp9BitReader
{
    private readonly ReadOnlySpan<byte> _data;
    private int _bytePos;
    private int _bitPos; // bits already consumed in the current byte, 0..7.

    internal Vp9BitReader(ReadOnlySpan<byte> data)
    {
        _data = data;
        _bytePos = 0;
        _bitPos = 0;
    }

    internal int Position => _bytePos * 8 + _bitPos;
    internal int BitsRemaining => (_data.Length - _bytePos) * 8 - _bitPos;

    /// <summary>Read the next <paramref name="nBits"/> bits as an unsigned integer (0..32 bits).</summary>
    internal uint ReadBits(int nBits)
    {
        if (nBits < 0 || nBits > 32)
            throw new ArgumentOutOfRangeException(nameof(nBits), "nBits must be 0..32");
        if (nBits == 0) return 0;
        if (BitsRemaining < nBits)
            throw new InvalidDataException($"VP9 bit reader: only {BitsRemaining} bits left, need {nBits}");

        uint value = 0;
        int bitsLeft = nBits;
        while (bitsLeft > 0)
        {
            int availInByte = 8 - _bitPos;
            int take = Math.Min(availInByte, bitsLeft);
            int shift = availInByte - take;
            // Extract `take` bits from the current byte starting at _bitPos.
            uint chunk = ((uint)_data[_bytePos] >> shift) & ((1u << take) - 1);
            value = (value << take) | chunk;
            _bitPos += take;
            if (_bitPos == 8)
            {
                _bitPos = 0;
                _bytePos++;
            }
            bitsLeft -= take;
        }
        return value;
    }

    /// <summary>Read a single bit as a bool (1 -> true).</summary>
    internal bool ReadFlag() => ReadBits(1) == 1;

    /// <summary>
    /// Read a signed-magnitude literal: <paramref name="nBits"/> bits of
    /// magnitude followed by 1 sign bit (1 = negative). Mirror of
    /// libvpx <c>vpx_rb_read_signed_literal</c>.
    /// </summary>
    internal int ReadSignedLiteral(int nBits)
    {
        int value = (int)ReadBits(nBits);
        return ReadFlag() ? -value : value;
    }
}
