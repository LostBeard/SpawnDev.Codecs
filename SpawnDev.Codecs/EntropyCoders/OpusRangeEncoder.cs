// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Structural port of libopus celt/entenc.h + celt/entenc.c to clean C#.
//
// Upstream Copyright (c) 2001-2011 Timothy B. Terriberry
// Upstream Copyright (c) 2008-2009 Xiph.Org Foundation
// Upstream license: BSD 3-Clause (Xiph.Org license). See NOTICE.md.
// Upstream source: https://github.com/xiph/opus

using System.Numerics;

namespace SpawnDev.Codecs.EntropyCoders;

/// <summary>
/// Opus range encoder. Encodes symbols into a compressed Opus bitstream per RFC 6716 section 4.1.
/// Stateful, not thread-safe. Call <see cref="Done"/> to finalize the stream before reading output.
/// </summary>
public sealed class OpusRangeEncoder
{
    // Constants from mfrngcod.h (mirror OpusRangeDecoder)
    private const int EC_SYM_BITS = 8;
    private const int EC_CODE_BITS = 32;
    private const uint EC_SYM_MAX = (1u << EC_SYM_BITS) - 1u;            // 0xFF
    private const int EC_CODE_SHIFT = EC_CODE_BITS - EC_SYM_BITS - 1;    // 23
    private const uint EC_CODE_TOP = 1u << (EC_CODE_BITS - 1);           // 0x80000000
    private const uint EC_CODE_BOT = EC_CODE_TOP >> EC_SYM_BITS;         // 0x00800000

    private const int EC_UINT_BITS = 8;
    private const int BITRES = 3;
    private const int EC_WINDOW_SIZE = 32;

    private readonly byte[] _buf;
    private readonly int _bufOffset;

    private uint _storage;
    private uint _endOffs;
    private uint _endWindow;
    private int _nendBits;
    private int _nbitsTotal;
    private uint _offs;
    private uint _rng;
    private uint _val;
    private uint _ext;
    private int _rem;
    private int _error;
    private bool _isDone;

    /// <summary>
    /// Creates an encoder with an internally-allocated buffer of the given capacity.
    /// </summary>
    public OpusRangeEncoder(int capacity)
        : this(new byte[capacity < 0 ? throw new ArgumentOutOfRangeException(nameof(capacity)) : capacity], 0, capacity)
    {
    }

    /// <summary>
    /// Creates an encoder writing into an existing buffer at the given offset / length.
    /// </summary>
    public OpusRangeEncoder(byte[] buf, int offset, int length)
    {
        ArgumentNullException.ThrowIfNull(buf);
        if ((uint)offset > (uint)buf.Length) throw new ArgumentOutOfRangeException(nameof(offset));
        if (length < 0 || offset + length > buf.Length) throw new ArgumentOutOfRangeException(nameof(length));

        _buf = buf;
        _bufOffset = offset;
        _storage = (uint)length;
        Reset();
    }

    /// <summary>Number of bytes used by the range-coded portion of the stream so far.</summary>
    public uint RangeBytes => _offs;

    /// <summary>Number of bytes used by raw bits written at the end of the stream.</summary>
    public uint EndBytes => _endOffs;

    /// <summary>Non-zero if the encoder has encountered an error (e.g. buffer overflow).</summary>
    public int Error => _error;

    /// <summary>True once <see cref="Done"/> has been called.</summary>
    public bool IsDone => _isDone;

    /// <summary>Number of whole bits used by encoded symbols so far.</summary>
    public int Tell => _nbitsTotal - EcIlog(_rng);

    /// <summary>Number of bits used scaled by 2^BITRES (1/8 bits).</summary>
    public uint TellFrac
    {
        get
        {
            ReadOnlySpan<uint> correction = new uint[]
            {
                35733, 38967, 42495, 46340,
                50535, 55109, 60097, 65535
            };
            uint nbits = (uint)_nbitsTotal << BITRES;
            int l = EcIlog(_rng);
            uint r = _rng >> (l - 16);
            uint b = (r >> 12) - 8;
            if (r > correction[(int)b]) b++;
            l = (l << 3) + (int)b;
            return nbits - (uint)l;
        }
    }

    /// <summary>
    /// Returns a view over the encoded bytes after <see cref="Done"/> has been called.
    /// The view covers <see cref="RangeBytes"/> range-coder bytes at the front plus
    /// <see cref="EndBytes"/> raw-bit bytes at the back, with the middle cleared by Done().
    /// </summary>
    public ReadOnlySpan<byte> EncodedSpan
    {
        get
        {
            if (!_isDone) throw new InvalidOperationException("Call Done() before reading encoded output.");
            return new ReadOnlySpan<byte>(_buf, _bufOffset, (int)_storage);
        }
    }

    /// <summary>Copies the encoded bytes into a new array. Requires <see cref="Done"/> to have been called.</summary>
    public byte[] ToArray() => EncodedSpan.ToArray();

    /// <summary>
    /// Encodes a symbol given its cumulative-frequency range <c>[fl, fh)</c> out of total <paramref name="ft"/>.
    /// </summary>
    public void Encode(uint fl, uint fh, uint ft)
    {
        CheckNotDone();
        uint r = _rng / ft;
        if (fl > 0)
        {
            _val += _rng - r * (ft - fl);
            _rng = r * (fh - fl);
        }
        else
        {
            _rng -= r * (ft - fh);
        }
        Normalize();
    }

    /// <summary>
    /// Equivalent to <see cref="Encode"/> with <c>ft = 1u &lt;&lt; bits</c> but avoids the division.
    /// </summary>
    public void EncodeBin(uint fl, uint fh, int bits)
    {
        CheckNotDone();
        uint r = _rng >> bits;
        uint ft = 1u << bits;
        if (fl > 0)
        {
            _val += _rng - r * (ft - fl);
            _rng = r * (fh - fl);
        }
        else
        {
            _rng -= r * (ft - fh);
        }
        Normalize();
    }

    /// <summary>
    /// Encodes a bit with probability <c>1 / (1 &lt;&lt; logp)</c> of being 1.
    /// </summary>
    public void EncodeBitLogP(int val, int logp)
    {
        CheckNotDone();
        uint r = _rng;
        uint l = _val;
        uint s = r >> logp;
        r -= s;
        if (val != 0) _val = l + r;
        _rng = val != 0 ? s : r;
        Normalize();
    }

    /// <summary>
    /// Encodes symbol index <paramref name="s"/> from the given inverse-CDF table.
    /// </summary>
    public void EncodeIcdf(int s, ReadOnlySpan<byte> icdf, int ftb)
    {
        CheckNotDone();
        uint r = _rng >> ftb;
        if (s > 0)
        {
            _val += _rng - r * icdf[s - 1];
            _rng = r * (uint)(icdf[s - 1] - icdf[s]);
        }
        else
        {
            _rng -= r * icdf[s];
        }
        Normalize();
    }

    /// <summary>
    /// Encodes symbol index <paramref name="s"/> from the given 16-bit inverse-CDF table.
    /// </summary>
    public void EncodeIcdf16(int s, ReadOnlySpan<ushort> icdf, int ftb)
    {
        CheckNotDone();
        uint r = _rng >> ftb;
        if (s > 0)
        {
            _val += _rng - r * icdf[s - 1];
            _rng = r * (uint)(icdf[s - 1] - icdf[s]);
        }
        else
        {
            _rng -= r * icdf[s];
        }
        Normalize();
    }

    /// <summary>
    /// Encodes a raw unsigned integer <paramref name="fl"/> in <c>[0, ft)</c>. <paramref name="ft"/> must be at least 2.
    /// </summary>
    public void EncodeUint(uint fl, uint ft)
    {
        CheckNotDone();
        if (ft <= 1) throw new ArgumentOutOfRangeException(nameof(ft), "ft must be at least 2.");

        uint decoded = ft - 1u;
        int ftb = EcIlog(decoded);
        if (ftb > EC_UINT_BITS)
        {
            ftb -= EC_UINT_BITS;
            uint scaledFt = (decoded >> ftb) + 1u;
            uint scaledFl = fl >> ftb;
            Encode(scaledFl, scaledFl + 1u, scaledFt);
            EncodeBits(fl & ((1u << ftb) - 1u), ftb);
        }
        else
        {
            Encode(fl, fl + 1u, decoded + 1u);
        }
    }

    /// <summary>
    /// Encodes a sequence of raw bits. <paramref name="bits"/> must be in <c>[1, 25]</c>.
    /// </summary>
    public void EncodeBits(uint fl, int bits)
    {
        CheckNotDone();
        if (bits <= 0) throw new ArgumentOutOfRangeException(nameof(bits), "bits must be positive.");

        uint window = _endWindow;
        int used = _nendBits;
        if (used + bits > EC_WINDOW_SIZE)
        {
            do
            {
                _error |= WriteByteAtEnd(window & EC_SYM_MAX);
                window >>= EC_SYM_BITS;
                used -= EC_SYM_BITS;
            }
            while (used >= EC_SYM_BITS);
        }
        window |= fl << used;
        used += bits;
        _endWindow = window;
        _nendBits = used;
        _nbitsTotal += bits;
    }

    /// <summary>
    /// Overwrites the first <paramref name="nbits"/> bits of the stream. Used to backfill flags
    /// whose values are known only late in encoding. <paramref name="nbits"/> must be at most 8.
    /// </summary>
    public void PatchInitialBits(uint val, int nbits)
    {
        CheckNotDone();
        if (nbits < 0 || nbits > EC_SYM_BITS) throw new ArgumentOutOfRangeException(nameof(nbits));

        int shift = EC_SYM_BITS - nbits;
        uint mask = (uint)(((1 << nbits) - 1) << shift);
        if (_offs > 0)
        {
            _buf[_bufOffset] = (byte)((_buf[_bufOffset] & ~mask) | (val << shift));
        }
        else if (_rem >= 0)
        {
            _rem = (int)(((uint)_rem & ~mask) | (val << shift));
        }
        else if (_rng <= (EC_CODE_TOP >> nbits))
        {
            _val = (_val & ~(mask << EC_CODE_SHIFT)) | (val << (EC_CODE_SHIFT + shift));
        }
        else
        {
            _error = -1;
        }
    }

    /// <summary>
    /// Compacts the stream to fit in <paramref name="size"/> bytes, moving trailing raw-bit data up.
    /// New size must hold both the range-coded prefix and the raw-bit suffix.
    /// </summary>
    public void Shrink(int size)
    {
        CheckNotDone();
        if (size < 0) throw new ArgumentOutOfRangeException(nameof(size));
        if (_offs + _endOffs > (uint)size) throw new ArgumentOutOfRangeException(nameof(size), "size too small for already-written data.");
        if ((uint)size > _storage) throw new ArgumentOutOfRangeException(nameof(size), "size must not exceed current capacity.");

        uint newSize = (uint)size;
        if (_endOffs > 0)
        {
            int src = _bufOffset + (int)(_storage - _endOffs);
            int dst = _bufOffset + (int)(newSize - _endOffs);
            Buffer.BlockCopy(_buf, src, _buf, dst, (int)_endOffs);
        }
        _storage = newSize;
    }

    /// <summary>
    /// Finalizes the stream. After this returns, <see cref="EncodedSpan"/> and
    /// <see cref="ToArray"/> yield the complete encoded output. No further encoding is allowed
    /// until a new encoder is created.
    /// </summary>
    public void Done()
    {
        if (_isDone) return;

        // Output the minimum number of bits that keep the stream decodable regardless of trailing bits.
        int l = EC_CODE_BITS - EcIlog(_rng);
        uint msk = (EC_CODE_TOP - 1u) >> l;
        uint end = (_val + msk) & ~msk;
        if ((end | msk) >= _val + _rng)
        {
            l++;
            msk >>= 1;
            end = (_val + msk) & ~msk;
        }
        while (l > 0)
        {
            CarryOut((int)(end >> EC_CODE_SHIFT));
            end = (end << EC_SYM_BITS) & (EC_CODE_TOP - 1u);
            l -= EC_SYM_BITS;
        }

        // Flush any buffered byte.
        if (_rem >= 0 || _ext > 0) CarryOut(0);

        // Flush any buffered end bits as full bytes.
        uint window = _endWindow;
        int used = _nendBits;
        while (used >= EC_SYM_BITS)
        {
            _error |= WriteByteAtEnd(window & EC_SYM_MAX);
            window >>= EC_SYM_BITS;
            used -= EC_SYM_BITS;
        }

        if (_error == 0)
        {
            // Clear the gap between the two regions.
            uint gapStart = _offs;
            uint gapEnd = _storage - _endOffs;
            if (gapEnd > gapStart)
            {
                Array.Clear(_buf, _bufOffset + (int)gapStart, (int)(gapEnd - gapStart));
            }

            if (used > 0)
            {
                if (_endOffs >= _storage)
                {
                    _error = -1;
                }
                else
                {
                    l = -l;
                    if (_offs + _endOffs >= _storage && l < used)
                    {
                        window &= (uint)((1 << l) - 1);
                        _error = -1;
                    }
                    int pos = _bufOffset + (int)(_storage - _endOffs - 1u);
                    _buf[pos] = (byte)(_buf[pos] | window);
                }
            }
        }

        _isDone = true;
    }

    // -------- Private helpers --------

    private void Reset()
    {
        _endOffs = 0;
        _endWindow = 0;
        _nendBits = 0;
        _nbitsTotal = EC_CODE_BITS + 1;
        _offs = 0;
        _rng = EC_CODE_TOP;
        _rem = -1;
        _val = 0;
        _ext = 0;
        _error = 0;
        _isDone = false;
    }

    private void CheckNotDone()
    {
        if (_isDone) throw new InvalidOperationException("Cannot encode more symbols after Done().");
    }

    private int WriteByte(uint value)
    {
        if (_offs + _endOffs >= _storage) return -1;
        _buf[_bufOffset + (int)_offs++] = (byte)value;
        return 0;
    }

    private int WriteByteAtEnd(uint value)
    {
        if (_offs + _endOffs >= _storage) return -1;
        _buf[_bufOffset + (int)(_storage - ++_endOffs)] = (byte)value;
        return 0;
    }

    /// <summary>
    /// Outputs a symbol with carry-propagation buffering. Values of EC_SYM_MAX stack up in
    /// ext until a non-max value resolves the carry direction.
    /// </summary>
    private void CarryOut(int c)
    {
        if (c != EC_SYM_MAX)
        {
            int carry = c >> EC_SYM_BITS;
            if (_rem >= 0) _error |= WriteByte((uint)(_rem + carry));
            if (_ext > 0)
            {
                uint sym = (EC_SYM_MAX + (uint)carry) & EC_SYM_MAX;
                do
                {
                    _error |= WriteByte(sym);
                }
                while (--_ext > 0);
            }
            _rem = (int)((uint)c & EC_SYM_MAX);
        }
        else
        {
            _ext++;
        }
    }

    private void Normalize()
    {
        while (_rng <= EC_CODE_BOT)
        {
            CarryOut((int)(_val >> EC_CODE_SHIFT));
            _val = (_val << EC_SYM_BITS) & (EC_CODE_TOP - 1u);
            _rng <<= EC_SYM_BITS;
            _nbitsTotal += EC_SYM_BITS;
        }
    }

    private static int EcIlog(uint v) =>
        v == 0 ? 0 : 32 - BitOperations.LeadingZeroCount(v);
}
