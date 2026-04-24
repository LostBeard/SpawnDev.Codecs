// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// MSB-first bit writer, inverse of FlacBitReader. Used by the FLAC encoder
// (and by unit tests that hand-build FLAC bitstreams). Bytes are accumulated
// in a growable buffer; the final byte is padded with trailing zero bits
// if the bit count is not a multiple of 8.

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// MSB-first bit writer. Accepts up to 32 bits per <see cref="Write"/> call and
/// produces a byte buffer via <see cref="ToArray"/>.
/// </summary>
internal sealed class FlacBitWriter
{
    private readonly List<byte> _bytes = new();
    private int _currentByte;
    private int _bitPos; // bits already written into _currentByte, 0..7.

    /// <summary>Current output length in bits (includes unflushed partial byte).</summary>
    internal int BitCount => _bytes.Count * 8 + _bitPos;

    /// <summary>
    /// Write <paramref name="bits"/> bits of <paramref name="value"/> MSB-first.
    /// <paramref name="bits"/> must be in [0, 32].
    /// </summary>
    internal void Write(uint value, int bits)
    {
        if (bits < 0 || bits > 32)
            throw new ArgumentOutOfRangeException(nameof(bits), "must be in [0, 32].");
        while (bits > 0)
        {
            int free = 8 - _bitPos;
            int take = Math.Min(free, bits);
            int shift = bits - take;
            uint chunk = (value >> shift) & ((1u << take) - 1);
            _currentByte = (_currentByte << take) | (int)chunk;
            _bitPos += take;
            bits -= take;
            if (_bitPos == 8)
            {
                _bytes.Add((byte)_currentByte);
                _currentByte = 0;
                _bitPos = 0;
            }
        }
    }

    /// <summary>
    /// Write a signed integer in two's-complement form at <paramref name="bits"/> bits.
    /// </summary>
    internal void WriteSigned(int value, int bits)
    {
        uint mask = bits == 32 ? 0xFFFFFFFFu : ((1u << bits) - 1);
        uint raw = (uint)value & mask;
        Write(raw, bits);
    }

    /// <summary>
    /// Write a unary code: <paramref name="zeroCount"/> zero bits followed by a 1.
    /// </summary>
    internal void WriteUnary(int zeroCount)
    {
        for (int i = 0; i < zeroCount; i++) Write(0, 1);
        Write(1, 1);
    }

    /// <summary>Flush partial-byte state to output. Trailing bits are zero-padded.</summary>
    internal void AlignToByte()
    {
        if (_bitPos != 0)
        {
            _bytes.Add((byte)(_currentByte << (8 - _bitPos)));
            _currentByte = 0;
            _bitPos = 0;
        }
    }

    /// <summary>Produce the final byte buffer (flushes partial byte with zero padding).</summary>
    internal byte[] ToArray()
    {
        if (_bitPos > 0)
        {
            _bytes.Add((byte)(_currentByte << (8 - _bitPos)));
            // Note: _bytes contains this byte but partial-byte state is NOT reset here, so
            // ToArray() is expected to be called once at the very end.
        }
        return _bytes.ToArray();
    }
}
