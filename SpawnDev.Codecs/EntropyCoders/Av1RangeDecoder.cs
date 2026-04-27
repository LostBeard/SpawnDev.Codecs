// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 range decoder. Structural port of libaom aom_dsp/entdec.h + entdec.c
// + entcode.h to clean C#.
//
// Upstream Copyright (c) 2001-2016, Alliance for Open Media. All rights reserved.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.
// Upstream source: https://aomedia.googlesource.com/aom (aom_dsp/entdec.{h,c})
//
// Algorithm: Daala range coder (Martin 1979 range coding family). Same family
// as Opus's range coder but with different state semantics:
//   - 32-bit window of unconsumed bit material (`dif`)
//   - q15 probabilities (CDF range [0, 32768))
//   - EC_PROB_SHIFT = 6 reduces internal prob precision to 8 bits per multiply
//   - EC_MIN_PROB = 4 prevents zero-range collapse on near-degenerate symbols
//   - tell_offs accumulates "virtual zero bits" past end of buffer
//
// Arithmetic coding is inherently sequential - each symbol updates state the
// next symbol depends on - so this implementation is pure C# CPU. No GPU
// kernel would help.

using System.Numerics;

namespace SpawnDev.Codecs.EntropyCoders;

/// <summary>
/// AV1 range decoder. Decodes symbols from an AV1 entropy-coded bitstream
/// (the body of a Frame OBU's tile data). Stateful and not thread-safe;
/// create one instance per tile decode session.
/// </summary>
public sealed class Av1RangeDecoder
{
    /// <summary>EC_PROB_SHIFT from aom_dsp/entcode.h.</summary>
    public const int EcProbShift = 6;
    /// <summary>EC_MIN_PROB from aom_dsp/entcode.h.</summary>
    public const int EcMinProb = 4;
    /// <summary>OD_EC_WINDOW_SIZE from aom_dsp/entcode.h (sizeof(uint32_t) * 8).</summary>
    public const int OdEcWindowSize = 32;
    /// <summary>OD_BITRES from aom_dsp/entcode.h - tell_frac scale factor.</summary>
    public const int OdBitRes = 3;
    /// <summary>OD_EC_LOTS_OF_BITS from aom_dsp/entdec.c.</summary>
    public const int OdEcLotsOfBits = 0x4000;
    /// <summary>q15 CDF top value (CDF_PROB_TOP from aom_dsp/prob.h).</summary>
    public const int CdfProbTop = 1 << 15;

    private readonly byte[] _buf;
    private readonly int _bufStart;
    private readonly int _bufEnd;

    private int _bptr;       // current read pointer (offset into _buf)
    private int _tellOffs;   // virtual-zero bits past end of stream
    private uint _dif;       // bit window (high 16 bits feed the next symbol)
    private uint _rng;       // current range, normalized to [32768, 65535]
    private int _cnt;        // bits remaining in dif

    /// <summary>Construct a decoder over the entire buffer.</summary>
    public Av1RangeDecoder(byte[] buf) : this(buf, 0, buf?.Length ?? 0) { }

    /// <summary>
    /// Construct a decoder reading <paramref name="length"/> bytes starting at
    /// <paramref name="offset"/> of <paramref name="buf"/>. Mirrors libaom
    /// <c>od_ec_dec_init</c>.
    /// </summary>
    public Av1RangeDecoder(byte[] buf, int offset, int length)
    {
        ArgumentNullException.ThrowIfNull(buf);
        if ((uint)offset > (uint)buf.Length) throw new ArgumentOutOfRangeException(nameof(offset));
        if (length < 0 || offset + length > buf.Length) throw new ArgumentOutOfRangeException(nameof(length));

        _buf = buf;
        _bufStart = offset;
        _bufEnd = offset + length;
        _bptr = offset;

        // Init constants from od_ec_dec_init.
        _tellOffs = 10 - (OdEcWindowSize - 8);
        _dif = (1u << (OdEcWindowSize - 1)) - 1u;
        _rng = 0x8000;
        _cnt = -15;
        Refill();
    }

    /// <summary>
    /// Decode a single binary value with probability of 1 = <paramref name="f"/> / 32768.
    /// Mirrors libaom <c>od_ec_decode_bool_q15</c>. <paramref name="f"/> must be in (0, 32768).
    /// </summary>
    public int DecodeBoolQ15(uint f)
    {
        if (f == 0 || f >= CdfProbTop)
            throw new ArgumentOutOfRangeException(nameof(f), "must be in (0, 32768)");
        uint dif = _dif;
        uint r = _rng;
        // v = ((r >> 8) * (f >> EC_PROB_SHIFT) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB
        uint v = ((r >> 8) * (f >> EcProbShift)) >> (7 - EcProbShift);
        v += EcMinProb;
        uint vw = v << (OdEcWindowSize - 16);
        int ret = 1;
        uint rNew = v;
        if (dif >= vw)
        {
            rNew = r - v;
            dif -= vw;
            ret = 0;
        }
        return Normalize(dif, rNew, ret);
    }

    /// <summary>
    /// Decode a symbol given an inverse CDF table in q15. The table must be
    /// monotonically non-increasing with <c>icdf[nsyms - 1] == 0</c>.
    /// Mirrors libaom <c>od_ec_decode_cdf_q15</c>.
    /// </summary>
    public int DecodeCdfQ15(ReadOnlySpan<ushort> icdf, int nsyms)
    {
        if (nsyms < 1 || nsyms > icdf.Length)
            throw new ArgumentOutOfRangeException(nameof(nsyms));
        if (icdf[nsyms - 1] != 0)
            throw new ArgumentException("icdf[nsyms - 1] must be 0", nameof(icdf));

        uint dif = _dif;
        uint r = _rng;
        int N = nsyms - 1;
        uint c = dif >> (OdEcWindowSize - 16);
        uint v = r;
        uint u;
        int ret = -1;
        do
        {
            u = v;
            // libaom: v = ((r >> 8) * (icdf[++ret] >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT) + EC_MIN_PROB * (N - ret)
            ret++;
            uint icdfVal = icdf[ret];
            // CDF entries are stored as `CDF_PROB_TOP - cumprob`; aom's OD_ICDF
            // macro pre-applies that flip at table-build time.
            v = ((r >> 8) * (icdfVal >> EcProbShift)) >> (7 - EcProbShift);
            v += (uint)(EcMinProb * (N - ret));
        } while (c < v);

        r = u - v;
        dif -= v << (OdEcWindowSize - 16);
        return Normalize(dif, r, ret);
    }

    /// <summary>
    /// Read <paramref name="ftb"/> raw bits (uncoded, just packed in MSB-first
    /// order at the back of the bitstream window). Mirrors libaom
    /// <c>od_ec_dec_bits_</c>.
    /// </summary>
    public uint DecodeBits(int ftb)
    {
        if (ftb < 0 || ftb > 25)
            throw new ArgumentOutOfRangeException(nameof(ftb), "must be in [0, 25]");
        if (ftb == 0) return 0;

        // libaom packs raw bits into the entropy stream by treating them as
        // a binary symbol with f = 1 << (15 - ftb): each call peels one bit.
        // The clean equivalent: shift them in via the normalize/refill loop.
        // We emulate libaom by repeated 1-bit decodes with a fixed midpoint
        // probability (uniform); this matches the C reference exactly.
        uint result = 0;
        for (int i = 0; i < ftb; i++)
        {
            // 1-bit "uniform" symbol: f = 16384. Emulates the per-bit branch
            // libaom takes for raw bits packed into the entropy window.
            uint dif = _dif;
            uint r = _rng;
            uint v = ((r >> 8) * (16384u >> EcProbShift)) >> (7 - EcProbShift);
            v += EcMinProb;
            uint vw = v << (OdEcWindowSize - 16);
            uint bit;
            uint rNew;
            if (dif >= vw)
            {
                rNew = r - v;
                dif -= vw;
                bit = 0;
            }
            else
            {
                rNew = v;
                bit = 1;
            }
            Normalize(dif, rNew, 0);
            result = (result << 1) | bit;
        }
        return result;
    }

    /// <summary>
    /// Number of whole bits "used" by symbols decoded so far. Mirrors
    /// libaom <c>od_ec_dec_tell</c>.
    /// </summary>
    public int Tell => (_bptr - _bufStart) * 8 - _cnt + _tellOffs;

    /// <summary>
    /// Bits used by symbols decoded so far, scaled by <c>2^OD_BITRES</c>.
    /// Mirrors libaom <c>od_ec_dec_tell_frac</c>.
    /// </summary>
    public uint TellFrac
    {
        get
        {
            // OD_BITRES = 3
            const int correctionLen = 8;
            ReadOnlySpan<uint> correction = new uint[]
            {
                35733, 38967, 42495, 46340,
                50535, 55109, 60097, 65535
            };
            uint nbits = (uint)Tell << OdBitRes;
            int l = IlogNz(_rng);
            uint r = _rng >> (l - 16);
            uint b = (r >> 12) - 8;
            if (correctionLen > 0 && b < correctionLen && r > correction[(int)b]) b++;
            l = (l << 3) + (int)b;
            return nbits - (uint)l;
        }
    }

    private void Refill()
    {
        int s = OdEcWindowSize - 9 - (_cnt + 15);
        for (; s >= 0 && _bptr < _bufEnd; s -= 8, _bptr++)
        {
            _dif ^= ((uint)_buf[_bptr]) << s;
            _cnt += 8;
        }
        if (_bptr >= _bufEnd)
        {
            // Stuff "virtual zero bits" past end of buffer.
            _tellOffs += OdEcLotsOfBits - _cnt;
            _cnt = OdEcLotsOfBits;
        }
    }

    private int Normalize(uint dif, uint rng, int ret)
    {
        // d = 16 - OD_ILOG_NZ(rng) where rng must be > 0
        int d = 16 - IlogNz(rng);
        _cnt -= d;
        _dif = ((dif + 1u) << d) - 1u;
        _rng = rng << d;
        if (_cnt < 0) Refill();
        return ret;
    }

    /// <summary>OD_ILOG_NZ(v) = position of the highest set bit of v + 1 (v != 0).</summary>
    private static int IlogNz(uint v) => 32 - BitOperations.LeadingZeroCount(v);
}
