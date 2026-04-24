// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// MSB-first bit reader matching FLAC's bitstream convention. FLAC packs bits
// within each byte high-bit-first, and the byte ordering is big-endian for
// multi-byte integer fields.

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// MSB-first bit reader over a byte span. Reads fields of 1-32 bits as unsigned
/// or signed integers. Also supports the unary prefix read used by Rice coding
/// and the UTF-8-coded variable-length integer used in FLAC frame headers.
/// </summary>
internal ref struct FlacBitReader
{
    private readonly ReadOnlySpan<byte> _data;
    private int _bytePos;
    private int _bitPos; // bits already consumed in the current byte, 0..7.

    /// <summary>Construct a reader positioned at the start of <paramref name="data"/>.</summary>
    internal FlacBitReader(ReadOnlySpan<byte> data)
    {
        _data = data;
        _bytePos = 0;
        _bitPos = 0;
    }

    /// <summary>True once the reader has consumed all bits (including the current byte).</summary>
    internal bool IsEnd => _bytePos >= _data.Length;

    /// <summary>Current read position in bits since start of the span.</summary>
    internal int Position => _bytePos * 8 + _bitPos;

    /// <summary>Number of bits remaining in the span.</summary>
    internal int BitsRemaining => (_data.Length - _bytePos) * 8 - _bitPos;

    /// <summary>
    /// Read the next <paramref name="nBits"/> bits as an unsigned integer.
    /// Supports <paramref name="nBits"/> in <c>[0, 32]</c>.
    /// </summary>
    internal uint ReadBits(int nBits)
    {
        if (nBits < 0 || nBits > 32)
            throw new ArgumentOutOfRangeException(nameof(nBits), "must be in [0, 32].");
        if (nBits == 0) return 0;
        if (BitsRemaining < nBits)
            throw new InvalidOperationException(
                $"Not enough bits remaining: requested {nBits}, have {BitsRemaining}.");

        uint result = 0;
        while (nBits > 0)
        {
            int available = 8 - _bitPos;
            int take = Math.Min(available, nBits);
            int shift = available - take;
            byte b = _data[_bytePos];
            uint bits = (uint)((b >> shift) & ((1 << take) - 1));
            result = (result << take) | bits;
            _bitPos += take;
            nBits -= take;
            if (_bitPos == 8)
            {
                _bytePos++;
                _bitPos = 0;
            }
        }
        return result;
    }

    /// <summary>
    /// Read the next <paramref name="nBits"/> bits as a signed integer (two's complement).
    /// </summary>
    internal int ReadBitsSigned(int nBits)
    {
        if (nBits <= 0 || nBits > 32)
            throw new ArgumentOutOfRangeException(nameof(nBits), "must be in [1, 32].");
        uint u = ReadBits(nBits);
        if (nBits < 32)
        {
            // Sign-extend from nBits to 32.
            uint signMask = 1u << (nBits - 1);
            if ((u & signMask) != 0)
            {
                uint extend = ~((1u << nBits) - 1);
                u |= extend;
            }
        }
        return unchecked((int)u);
    }

    /// <summary>
    /// Read a single bit.
    /// </summary>
    internal uint ReadBit() => ReadBits(1);

    /// <summary>
    /// Read a unary-coded integer: count the number of consecutive zero bits then consume the
    /// terminating 1 bit. Returns the zero count.
    /// </summary>
    internal int ReadUnary()
    {
        int count = 0;
        while (!IsEnd)
        {
            if (ReadBit() == 1) return count;
            count++;
        }
        throw new InvalidOperationException("Unary code exceeded stream.");
    }

    /// <summary>
    /// Skip to the next byte boundary (discard any partial bits in the current byte).
    /// </summary>
    internal void AlignToByte()
    {
        if (_bitPos != 0)
        {
            _bytePos++;
            _bitPos = 0;
        }
    }

    /// <summary>
    /// Read a UTF-8-coded variable-length unsigned integer as used in FLAC frame headers.
    /// Matches libFLAC's read_utf8_uint32_ (31-bit frame number) and read_utf8_uint64_
    /// (36-bit sample number).
    /// </summary>
    /// <param name="maxBytes">Maximum total encoded byte length. Pass <c>6</c> for a 31-bit
    /// frame number (disallows the 7-byte 0xFE lead) or <c>7</c> for a 36-bit sample number.</param>
    internal ulong ReadUtf8VariableLength(int maxBytes)
    {
        if (maxBytes is not (6 or 7))
            throw new ArgumentException("maxBytes must be 6 or 7.", nameof(maxBytes));

        uint firstByte = ReadBits(8);
        int count;
        ulong value;
        if ((firstByte & 0x80) == 0)
        {
            return firstByte;
        }
        if ((firstByte & 0xE0) == 0xC0) { count = 1; value = firstByte & 0x1Fu; }
        else if ((firstByte & 0xF0) == 0xE0) { count = 2; value = firstByte & 0x0Fu; }
        else if ((firstByte & 0xF8) == 0xF0) { count = 3; value = firstByte & 0x07u; }
        else if ((firstByte & 0xFC) == 0xF8) { count = 4; value = firstByte & 0x03u; }
        else if ((firstByte & 0xFE) == 0xFC) { count = 5; value = firstByte & 0x01u; }
        else if (firstByte == 0xFE && maxBytes >= 7) { count = 6; value = 0; }
        else throw new InvalidOperationException($"Invalid UTF-8 continuation byte: 0x{firstByte:X2}.");

        for (int i = 0; i < count; i++)
        {
            uint b = ReadBits(8);
            if ((b & 0xC0) != 0x80)
                throw new InvalidOperationException($"Invalid UTF-8 continuation byte at index {i}: 0x{b:X2}.");
            value = (value << 6) | (b & 0x3Fu);
        }
        return value;
    }
}
