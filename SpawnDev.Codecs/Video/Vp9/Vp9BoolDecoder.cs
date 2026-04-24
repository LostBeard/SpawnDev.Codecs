// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 boolean arithmetic decoder - bit-exact port of libvpx vpx_reader.
// Every entropy-coded element in a VP9 frame flows through this
// decoder: the compressed frame header, segmentation + loop-filter
// delta updates, mode info trees, motion vectors, and the per-block
// coefficient tree. Bit-exactness is non-negotiable because VP9's
// probability tables are normative.
//
// libvpx reference: vpx_dsp/bitreader.h + bitreader.c.
// VP9 spec: sec 9 "Parsing Process".
//
// Design notes
//   - BD_VALUE is `ulong` (64 bits). libvpx uses `size_t` which is 32
//     or 64 depending on platform; we always use 64 for deterministic
//     C# behaviour across .NET runtimes.
//   - The value register holds the incoming arithmetic-coded bits
//     left-aligned; the top 8 bits form the "current decision window"
//     that probability splits compare against.
//   - range starts at 255, shrinks after each read, and re-normalises
//     back up to [128, 255] by left-shifting both range and value.
//   - count tracks how many buffer bits are currently loaded in value
//     minus 8 (matches libvpx convention). When count < 0 we refill
//     from the byte buffer.
//   - Normalisation shift count is derived from the leading-zero
//     count of the post-split range (treated as the high byte of a
//     32-bit value) - this replaces libvpx's vpx_norm[256] lookup
//     table with a built-in intrinsic.

using System.Numerics;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Stateful VP9 arithmetic decoder. Reads boolean values coded against
/// probability tables. Matches libvpx vpx_reader bit-for-bit.
/// </summary>
public sealed class Vp9BoolDecoder
{
    // Bit-width of the value register. 64 bits keeps each Fill() load
    // wide enough to absorb 7 bytes at once before refilling.
    private const int BdValueSize = 64;

    private ulong _value;
    private uint _range;
    private int _count;
    private int _bufPos;
    private readonly byte[] _buffer;
    private readonly int _bufferEnd;

    /// <summary>
    /// Initialise the decoder over <paramref name="buffer"/> starting at
    /// <paramref name="offset"/> for <paramref name="length"/> bytes.
    /// </summary>
    /// <exception cref="ArgumentException">Offset/length out of range.</exception>
    /// <exception cref="InvalidDataException">
    /// The initial marker bit read during libvpx's vpx_reader_init is non-zero,
    /// which signals a corrupt stream per the reference implementation.
    /// </exception>
    public Vp9BoolDecoder(byte[] buffer, int offset, int length)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        if (offset < 0 || length < 0 || offset + length > buffer.Length)
            throw new ArgumentException("offset+length out of range");

        _buffer = buffer;
        _bufPos = offset;
        _bufferEnd = offset + length;
        _value = 0;
        _range = 255;
        _count = -8;
        Fill();

        // libvpx reads a sentinel bit during init; a non-zero bit
        // indicates stream corruption.
        if (ReadBit() != 0)
            throw new InvalidDataException("VP9 bool decoder init: marker bit non-zero.");
    }

    /// <summary>
    /// Read one bit coded against probability <paramref name="prob"/>
    /// (a value in [1, 255]; higher = bit=0 more likely).
    /// </summary>
    public int Read(int prob)
    {
        // libvpx: split = 1 + (((range - 1) * prob) >> 8)
        uint split = 1u + (((_range - 1) * (uint)prob) >> 8);
        // bigsplit places split at the top of the value register's
        // 64-bit word, aligned with how value was loaded.
        ulong bigsplit = (ulong)split << (BdValueSize - 8);

        int bit;
        if (_value < bigsplit)
        {
            _range = split;
            bit = 0;
        }
        else
        {
            _range -= split;
            _value -= bigsplit;
            bit = 1;
        }

        // Normalise: shift both range and value left until range >= 128.
        // range << 24 treats range as the high byte of a 32-bit word;
        // LeadingZeroCount of that gives us the shift.
        int shift = BitOperations.LeadingZeroCount(_range) - 24;
        if (shift > 0)
        {
            _range <<= shift;
            _value <<= shift;
            _count -= shift;
            if (_count < 0) Fill();
        }

        return bit;
    }

    /// <summary>
    /// Read one equiprobable bit (prob = 128). Common enough that
    /// libvpx inlines it as a convenience wrapper.
    /// </summary>
    public int ReadBit() => Read(128);

    /// <summary>
    /// Read <paramref name="nBits"/> equiprobable bits as an unsigned
    /// integer, MSB first (matches libvpx vpx_read_literal).
    /// </summary>
    public uint ReadLiteral(int nBits)
    {
        uint v = 0;
        for (int i = 0; i < nBits; i++)
            v = (v << 1) | (uint)ReadBit();
        return v;
    }

    /// <summary>
    /// Has the decoder consumed all its input AND has the arithmetic
    /// coder landed in an error state? libvpx uses this as the stream
    /// validity check after a frame is fully decoded.
    /// </summary>
    public bool HasError =>
        _count > BdValueSize && _count < LotsOfBits;

    private const int LotsOfBits = 0x40000000;

    /// <summary>
    /// Load bytes into the value register until count is non-negative.
    /// Byte-by-byte port of libvpx's slow-path fill loop (the fast path
    /// loads 8 bytes via bswap; slow path is sufficient for correctness).
    /// </summary>
    private void Fill()
    {
        int shift = BdValueSize - 8 - (_count + 8);
        int bytesLeft = _bufferEnd - _bufPos;
        int bitsLeft = bytesLeft * 8;
        int bitsOver = shift + 8 - bitsLeft;
        int loopEnd = 0;
        if (bitsOver >= 0)
        {
            // Buffer exhausted relative to the requested shift.
            _count += LotsOfBits;
            loopEnd = bitsOver;
        }
        if (bitsOver < shift)
        {
            while (shift >= loopEnd)
            {
                _count += 8;
                _value |= (ulong)_buffer[_bufPos++] << shift;
                shift -= 8;
            }
        }
    }
}
