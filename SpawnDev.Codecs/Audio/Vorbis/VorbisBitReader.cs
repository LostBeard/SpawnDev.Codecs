// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Vorbis packs integers least-significant-bit-first, opposite to FLAC (and
// opposite to Opus's range coder). This reader handles the LSB-first
// convention used throughout the Vorbis I setup header and audio packets.
//
// Vorbis I spec, section 2.1.2:
//   "Integers are written and read LSB first. [...] When reading an n-bit
//    integer, the first bit read is the LSB; the nth bit read is the MSB."

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// LSB-first bit reader over a byte span. Reads fields of 0..32 bits as
/// unsigned integers. Matches Vorbis I's packing convention.
/// </summary>
internal ref struct VorbisBitReader
{
    private readonly ReadOnlySpan<byte> _data;
    private int _bytePos;
    private int _bitPos; // bits already consumed in current byte, 0..7 (from LSB up).

    internal VorbisBitReader(ReadOnlySpan<byte> data)
    {
        _data = data;
        _bytePos = 0;
        _bitPos = 0;
    }

    /// <summary>Current read position in bits since start of the span.</summary>
    internal int Position => _bytePos * 8 + _bitPos;

    /// <summary>Number of bits remaining.</summary>
    internal int BitsRemaining => (_data.Length - _bytePos) * 8 - _bitPos;

    /// <summary>True after all bits have been consumed.</summary>
    internal bool IsEnd => BitsRemaining <= 0;

    /// <summary>
    /// Read <paramref name="nBits"/> bits as an unsigned integer. The first bit
    /// read appears in bit 0 of the result (LSB-first), the last bit in bit
    /// <c>nBits - 1</c>. <paramref name="nBits"/> must be in [0, 32].
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
        int outBit = 0;
        while (nBits > 0)
        {
            int available = 8 - _bitPos;
            int take = Math.Min(available, nBits);
            byte b = _data[_bytePos];
            uint chunk = (uint)((b >> _bitPos) & ((1 << take) - 1));
            result |= chunk << outBit;
            _bitPos += take;
            nBits -= take;
            outBit += take;
            if (_bitPos == 8)
            {
                _bytePos++;
                _bitPos = 0;
            }
        }
        return result;
    }

    /// <summary>Read a single bit (0 or 1).</summary>
    internal uint ReadBit() => ReadBits(1);

    /// <summary>
    /// Try to read <paramref name="nBits"/> bits. Returns false (without
    /// advancing the read cursor) when the packet has fewer remaining bits.
    /// Vorbis I sec 8.6.5 specifies that residue decode treats end-of-packet
    /// as a graceful termination signal rather than an error, so the
    /// EOP-aware decode paths use this overload to short-circuit cleanly.
    /// </summary>
    internal bool TryReadBits(int nBits, out uint value)
    {
        if (nBits < 0 || nBits > 32)
            throw new ArgumentOutOfRangeException(nameof(nBits), "must be in [0, 32].");
        if (nBits == 0) { value = 0; return true; }
        if (BitsRemaining < nBits) { value = 0; return false; }
        value = ReadBits(nBits);
        return true;
    }

    /// <summary>EOP-aware single-bit read. Returns false on end-of-packet.</summary>
    internal bool TryReadBit(out uint bit) => TryReadBits(1, out bit);

    /// <summary>
    /// Read a signed integer of <paramref name="nBits"/> bits, where the top
    /// bit of the read value is interpreted as the sign per two's complement.
    /// </summary>
    internal int ReadBitsSigned(int nBits)
    {
        if (nBits <= 0 || nBits > 32)
            throw new ArgumentOutOfRangeException(nameof(nBits), "must be in [1, 32].");
        uint u = ReadBits(nBits);
        if (nBits < 32)
        {
            uint signMask = 1u << (nBits - 1);
            if ((u & signMask) != 0)
                u |= ~((1u << nBits) - 1);
        }
        return unchecked((int)u);
    }

    /// <summary>Skip to the next byte boundary.</summary>
    internal void AlignToByte()
    {
        if (_bitPos != 0)
        {
            _bytePos++;
            _bitPos = 0;
        }
    }
}
