// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// LSB-first bit writer for Vorbis bit packing. Pairs with VorbisBitReader.
// Vorbis I spec, section 2.1.2: integers are written LSB-first. The first bit
// written ends up in bit 0 of the first byte, the next in bit 1, etc.
//
// Used by the Vorbis encoder to emit the setup header packet and per-packet
// audio bit streams.

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// LSB-first bit writer. Builds an output byte buffer one bit-field at a time
/// and produces a <c>byte[]</c> on demand. The companion to
/// <see cref="VorbisBitReader"/>: bits written here read back in the same order.
/// </summary>
internal sealed class VorbisBitWriter
{
    private readonly List<byte> _bytes = new();
    private int _currentByte; // accumulator for the in-progress byte
    private int _bitPos;      // next bit position within _currentByte (0..7)

    /// <summary>Total bits written so far.</summary>
    internal int BitsWritten => _bytes.Count * 8 + _bitPos;

    /// <summary>
    /// Append <paramref name="nBits"/> low-order bits of <paramref name="value"/>
    /// to the stream. <paramref name="nBits"/> must be in [0, 32].
    /// </summary>
    internal void WriteBits(uint value, int nBits)
    {
        if (nBits < 0 || nBits > 32)
            throw new ArgumentOutOfRangeException(nameof(nBits), "must be in [0, 32].");
        if (nBits == 0) return;
        // Mask off any garbage in the high bits of value.
        if (nBits < 32) value &= (1u << nBits) - 1u;

        int remaining = nBits;
        int srcBit = 0;
        while (remaining > 0)
        {
            int spaceInByte = 8 - _bitPos;
            int take = Math.Min(spaceInByte, remaining);
            uint chunk = (value >> srcBit) & ((1u << take) - 1u);
            _currentByte |= (int)(chunk << _bitPos);
            _bitPos += take;
            srcBit += take;
            remaining -= take;
            if (_bitPos == 8)
            {
                _bytes.Add((byte)_currentByte);
                _currentByte = 0;
                _bitPos = 0;
            }
        }
    }

    /// <summary>Convenience: write a single bit.</summary>
    internal void WriteBit(uint bit) => WriteBits(bit & 1u, 1);

    /// <summary>
    /// Flush the in-progress byte (zero-pads the high bits) and return the
    /// final byte sequence. Calling this finalises the stream.
    /// </summary>
    internal byte[] ToArray()
    {
        if (_bitPos != 0)
        {
            _bytes.Add((byte)_currentByte);
            _currentByte = 0;
            _bitPos = 0;
        }
        return _bytes.ToArray();
    }
}
