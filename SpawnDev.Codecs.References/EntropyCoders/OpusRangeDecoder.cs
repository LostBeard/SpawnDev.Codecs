// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// This file is a structural port of libopus celt/entcode.h + celt/entcode.c +
// celt/entdec.h + celt/entdec.c + celt/mfrngcod.h + celt/ecintrin.h to clean C#.
//
// Upstream Copyright (c) 2001-2011 Timothy B. Terriberry
// Upstream Copyright (c) 2008-2009 Xiph.Org Foundation
// Upstream license: BSD 3-Clause (Xiph.Org license). See NOTICE.md.
// Upstream source: https://github.com/xiph/opus
//
// Algorithm: Martin (1979) range coding; a FIFO arithmetic code variant. See
// RFC 6716 section 4.1. Arithmetic coding is inherently sequential - each symbol
// updates state the next symbol depends on - so this implementation is pure C#
// CPU. No GPU kernel would provide any parallelism benefit here; the decoder is
// placed where its nature fits.

using System.Numerics;

namespace SpawnDev.Codecs.EntropyCoders;

/// <summary>
/// Opus range decoder. Decodes symbols from a compressed Opus bitstream per RFC 6716 section 4.1.
/// Stateful and not thread-safe; create one instance per decode session.
/// </summary>
public sealed class OpusRangeDecoder
{
    // Constants from mfrngcod.h
    private const int EC_SYM_BITS = 8;
    private const int EC_CODE_BITS = 32;
    private const uint EC_SYM_MAX = (1u << EC_SYM_BITS) - 1u;            // 0xFF
    private const uint EC_CODE_TOP = 1u << (EC_CODE_BITS - 1);           // 0x80000000
    private const uint EC_CODE_BOT = EC_CODE_TOP >> EC_SYM_BITS;         // 0x00800000
    private const int EC_CODE_EXTRA = (EC_CODE_BITS - 2) % EC_SYM_BITS + 1; // 7

    // Constants from entcode.h
    private const int EC_UINT_BITS = 8;
    private const int BITRES = 3;
    private const int EC_WINDOW_SIZE = 32; // sizeof(uint) * 8

    // Buffer
    private readonly byte[] _buf;
    private readonly int _bufOffset;
    private readonly uint _storage;

    // Mutable range coder state (mirrors libopus ec_ctx)
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

    /// <summary>
    /// Creates a decoder that reads the entire <paramref name="buf"/>.
    /// </summary>
    public OpusRangeDecoder(byte[] buf) : this(buf, 0, buf?.Length ?? 0) { }

    /// <summary>
    /// Creates a decoder reading <paramref name="length"/> bytes starting at <paramref name="offset"/> of <paramref name="buf"/>.
    /// </summary>
    public OpusRangeDecoder(byte[] buf, int offset, int length)
    {
        ArgumentNullException.ThrowIfNull(buf);
        if ((uint)offset > (uint)buf.Length) throw new ArgumentOutOfRangeException(nameof(offset));
        if (length < 0 || offset + length > buf.Length) throw new ArgumentOutOfRangeException(nameof(length));

        _buf = buf;
        _bufOffset = offset;
        _storage = (uint)length;
        _endOffs = 0;
        _endWindow = 0;
        _nendBits = 0;
        // Offset from which ec_tell() subtracts partial bits.
        _nbitsTotal = EC_CODE_BITS + 1 - ((EC_CODE_BITS - EC_CODE_EXTRA) / EC_SYM_BITS) * EC_SYM_BITS;
        _offs = 0;
        _rng = 1u << EC_CODE_EXTRA;
        _rem = ReadByte();
        _val = _rng - 1u - (uint)(_rem >> (EC_SYM_BITS - EC_CODE_EXTRA));
        _error = 0;
        Normalize();
    }

    /// <summary>Number of bytes consumed by the range-coded portion of the stream.</summary>
    public uint RangeBytes => _offs;

    /// <summary>Non-zero if the decoder has encountered an error (e.g. malformed uint).</summary>
    public int Error => _error;

    /// <summary>
    /// Number of whole bits "used" by decoded symbols so far.
    /// Computed the same way in encoder and decoder, so suitable for coding-decision logic.
    /// </summary>
    public int Tell => _nbitsTotal - EcIlog(_rng);

    /// <summary>
    /// Number of bits used scaled by 2^BITRES (1/8 bits). Slightly larger than exact; rounding is biased positive.
    /// </summary>
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
    /// Calculates the cumulative frequency for the next symbol when it was encoded with total
    /// frequency <paramref name="ft"/>. Feed this to a probability model to recover the symbol;
    /// follow with <see cref="Update"/> to advance the decoder past that symbol.
    /// </summary>
    /// <returns>Cumulative frequency in <c>[fl, fh)</c> for the encoded symbol.</returns>
    public uint Decode(uint ft)
    {
        _ext = _rng / ft;
        uint s = _val / _ext;
        return ft - Min(s + 1u, ft);
    }

    /// <summary>
    /// Equivalent to <see cref="Decode"/> with <c>ft = 1u &lt;&lt; bits</c> but avoids the division.
    /// </summary>
    public uint DecodeBin(int bits)
    {
        _ext = _rng >> bits;
        uint s = _val / _ext;
        uint ft = 1u << bits;
        return ft - Min(s + 1u, ft);
    }

    /// <summary>
    /// Advance the decoder past the symbol previously queried via <see cref="Decode"/> or
    /// <see cref="DecodeBin"/> using the cumulative frequencies <paramref name="fl"/>, <paramref name="fh"/>
    /// and total frequency <paramref name="ft"/>.
    /// </summary>
    public void Update(uint fl, uint fh, uint ft)
    {
        uint s = _ext * (ft - fh);
        _val -= s;
        _rng = fl > 0 ? _ext * (fh - fl) : _rng - s;
        Normalize();
    }

    /// <summary>
    /// Decodes a bit that has probability <c>1 / (1 &lt;&lt; logp)</c> of being 1.
    /// </summary>
    public int DecodeBitLogP(int logp)
    {
        uint r = _rng;
        uint d = _val;
        uint s = r >> logp;
        int ret = d < s ? 1 : 0;
        if (ret == 0) _val = d - s;
        _rng = ret != 0 ? s : r - s;
        Normalize();
        return ret;
    }

    /// <summary>
    /// Decodes a symbol given an "inverse" CDF table. The table must be monotonically
    /// non-increasing with a final entry of 0. <paramref name="ftb"/> is the number of bits of precision.
    /// No follow-up <see cref="Update"/> call is required.
    /// </summary>
    public int DecodeIcdf(ReadOnlySpan<byte> icdf, int ftb)
    {
        uint s = _rng;
        uint d = _val;
        uint r = s >> ftb;
        int ret = -1;
        uint t;
        do
        {
            t = s;
            s = r * icdf[++ret];
        }
        while (d < s);
        _val = d - s;
        _rng = t - s;
        Normalize();
        return ret;
    }

    /// <summary>
    /// Decodes a symbol given an "inverse" CDF table with 16-bit entries. Same semantics as <see cref="DecodeIcdf"/>.
    /// </summary>
    public int DecodeIcdf16(ReadOnlySpan<ushort> icdf, int ftb)
    {
        uint s = _rng;
        uint d = _val;
        uint r = s >> ftb;
        int ret = -1;
        uint t;
        do
        {
            t = s;
            s = r * icdf[++ret];
        }
        while (d < s);
        _val = d - s;
        _rng = t - s;
        Normalize();
        return ret;
    }

    /// <summary>
    /// Extracts a raw unsigned integer with a non-power-of-2 range from the stream.
    /// Must have been encoded with the encoder's equivalent. <paramref name="ft"/> is one
    /// more than the maximum legal value (i.e. values in <c>[0, ft)</c>). No follow-up <see cref="Update"/>.
    /// </summary>
    public uint DecodeUint(uint ft)
    {
        // EC_ILOG is undefined for 0; Opus requires ft > 1.
        if (ft <= 1) throw new ArgumentOutOfRangeException(nameof(ft), "ft must be at least 2.");

        uint decoded = ft - 1u;
        int ftb = EcIlog(decoded);
        if (ftb > EC_UINT_BITS)
        {
            ftb -= EC_UINT_BITS;
            uint scaledFt = (decoded >> ftb) + 1u;
            uint s = Decode(scaledFt);
            Update(s, s + 1u, scaledFt);
            uint t = (s << ftb) | DecodeBits(ftb);
            if (t <= decoded) return t;
            _error = 1;
            return decoded;
        }
        else
        {
            uint s = Decode(ft);
            Update(s, s + 1u, ft);
            return s;
        }
    }

    /// <summary>
    /// Extracts a raw sequence of <paramref name="bits"/> from the stream.
    /// The bits must have been written with the encoder's equivalent. Valid range: 0 to 25 inclusive.
    /// No follow-up <see cref="Update"/>.
    /// </summary>
    public uint DecodeBits(int bits)
    {
        uint window = _endWindow;
        int available = _nendBits;
        if (available < bits)
        {
            do
            {
                window |= (uint)ReadByteFromEnd() << available;
                available += EC_SYM_BITS;
            }
            while (available <= EC_WINDOW_SIZE - EC_SYM_BITS);
        }
        uint ret = window & ((1u << bits) - 1u);
        window >>= bits;
        available -= bits;
        _endWindow = window;
        _nendBits = available;
        _nbitsTotal += bits;
        return ret;
    }

    // -------- Private helpers --------

    private int ReadByte()
    {
        return _offs < _storage ? _buf[_bufOffset + (int)_offs++] : 0;
    }

    private int ReadByteFromEnd()
    {
        return _endOffs < _storage ? _buf[_bufOffset + (int)(_storage - ++_endOffs)] : 0;
    }

    /// <summary>
    /// Normalizes val/rng so that rng lies entirely in the high-order symbol.
    /// Reads additional bytes from the front of the buffer as needed.
    /// </summary>
    private void Normalize()
    {
        while (_rng <= EC_CODE_BOT)
        {
            _nbitsTotal += EC_SYM_BITS;
            _rng <<= EC_SYM_BITS;
            int sym = _rem;
            _rem = ReadByte();
            sym = (sym << EC_SYM_BITS | _rem) >> (EC_SYM_BITS - EC_CODE_EXTRA);
            _val = ((_val << EC_SYM_BITS) + (EC_SYM_MAX & (uint)~sym)) & (EC_CODE_TOP - 1u);
        }
    }

    /// <summary>Branchless min for uint. Equivalent to libopus EC_MINI macro.</summary>
    private static uint Min(uint a, uint b) => a < b ? a : b;

    /// <summary>
    /// Integer log2 + 1 for positive values (matches libopus EC_ILOG semantics).
    /// <c>EC_ILOG(0) == 0</c>; <c>EC_ILOG(1) == 1</c>; <c>EC_ILOG(2) == 2</c>; etc.
    /// </summary>
    private static int EcIlog(uint v) =>
        v == 0 ? 0 : 32 - BitOperations.LeadingZeroCount(v);
}
