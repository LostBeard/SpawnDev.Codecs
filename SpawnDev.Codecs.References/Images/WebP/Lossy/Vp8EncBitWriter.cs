// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). Port of libwebp 1.6.0 src/utils/bit_writer_utils.c (VP8BitWriter),
// BSD licence, see NOTICE.md. Statement for statement: range is kept as "range - 1", 0xff bytes are delayed in `run` for carry.

using SpawnDev.Codecs.Images.WebP.Lossy;

namespace SpawnDev.Codecs.References.Images.WebP.Lossy;

/// <summary>libwebp's VP8 boolean encoder.</summary>
internal sealed class Vp8EncBitWriter
{
    private int _range;
    private int _value;
    private int _run;
    private int _nbBits;
    private byte[] _buf;
    private int _pos;

    public Vp8EncBitWriter(int expectedSize = 0)
    {
        _range = 255 - 1;
        _value = 0;
        _run = 0;
        _nbBits = -8;
        _pos = 0;
        _buf = new byte[Math.Max(1024, expectedSize)];
    }

    /// <summary>VP8BitWriterPos: bits written so far.</summary>
    public long Pos => (long)(_pos + _run) * 8 + 8 + _nbBits;

    /// <summary>VP8BitWriterSize: bytes in the buffer.</summary>
    public int Size => _pos;

    public ReadOnlySpan<byte> Buffer => _buf.AsSpan(0, _pos);

    private void Ensure(int extra)
    {
        int needed = _pos + extra;
        if (needed <= _buf.Length) return;
        int size = Math.Max(Math.Max(2 * _buf.Length, needed), 1024);
        System.Array.Resize(ref _buf, size);
    }

    private void Flush()
    {
        int s = 8 + _nbBits;
        int bits = _value >> s;
        _value -= bits << s;
        _nbBits -= 8;
        if ((bits & 0xff) != 0xff)
        {
            int pos = _pos;
            Ensure(_run + 1);
            if ((bits & 0x100) != 0)
            {
                if (pos > 0) _buf[pos - 1]++;
            }
            if (_run > 0)
            {
                byte value = (bits & 0x100) != 0 ? (byte)0x00 : (byte)0xff;
                for (; _run > 0; --_run) _buf[pos++] = value;
            }
            _buf[pos++] = (byte)(bits & 0xff);
            _pos = pos;
        }
        else
        {
            _run++;
        }
    }

    public int PutBit(int bit, int prob)
    {
        int split = (_range * prob) >> 8;
        if (bit != 0)
        {
            _value += split + 1;
            _range -= split + 1;
        }
        else
        {
            _range = split;
        }
        if (_range < 127)
        {
            int shift = Vp8EncTables.kNorm[_range];
            _range = Vp8EncTables.kNewRange[_range];
            _value <<= shift;
            _nbBits += shift;
            if (_nbBits > 0) Flush();
        }
        return bit;
    }

    public int PutBit(bool bit, int prob) => PutBit(bit ? 1 : 0, prob);

    public int PutBitUniform(int bit)
    {
        int split = _range >> 1;
        if (bit != 0)
        {
            _value += split + 1;
            _range -= split + 1;
        }
        else
        {
            _range = split;
        }
        if (_range < 127)
        {
            _range = Vp8EncTables.kNewRange[_range];
            _value <<= 1;
            _nbBits += 1;
            if (_nbBits > 0) Flush();
        }
        return bit;
    }

    public int PutBitUniform(bool bit) => PutBitUniform(bit ? 1 : 0);

    public void PutBits(uint value, int nbBits)
    {
        for (uint mask = 1u << (nbBits - 1); mask != 0; mask >>= 1)
            PutBitUniform((value & mask) != 0 ? 1 : 0);
    }

    public void PutSignedBits(int value, int nbBits)
    {
        if (PutBitUniform(value != 0 ? 1 : 0) == 0) return;
        if (value < 0) PutBits((uint)(((-value) << 1) | 1), nbBits + 1);
        else PutBits((uint)(value << 1), nbBits + 1);
    }

    /// <summary>VP8BitWriterFinish.</summary>
    public void Finish()
    {
        PutBits(0, 9 - _nbBits);
        _nbBits = 0;
        Flush();
    }
}
