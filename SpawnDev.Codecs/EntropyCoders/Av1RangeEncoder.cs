// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 range encoder. Structural port of libaom aom_dsp/entenc.h + entenc.c
// + entcode.h to clean C#.
//
// Upstream Copyright (c) 2001-2016, Alliance for Open Media. All rights reserved.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.
// Upstream source: https://aomedia.googlesource.com/aom (aom_dsp/entenc.{h,c})
//
// Pairs with <see cref="Av1RangeDecoder"/>. Round-trip-tested in
// CodecsTestBase.Av1RangeCoderTests.
//
// Differences from the C reference:
//   - Output buffer is a C# List<byte>; the C reference uses a malloc'd
//     buffer with explicit growth. The carry-propagation logic is identical.
//   - low is uint64 (matching libaom's od_ec_enc_window typedef).
//   - cnt is int32 (vs libaom int16) for simpler arithmetic; only the low
//     7 bits matter at any point so the range fits trivially.

using System.Buffers.Binary;

namespace SpawnDev.Codecs.EntropyCoders;

/// <summary>
/// AV1 range encoder. Encodes symbols into an AV1 entropy-coded bitstream
/// using q15 probabilities and CDFs. Stateful, single-tile lifetime.
/// </summary>
public sealed class Av1RangeEncoder
{
    private readonly List<byte> _buf = new();
    private ulong _low;
    private uint _rng;
    private int _cnt;
    private bool _error;

    /// <summary>q15 CDF top value (CDF_PROB_TOP from aom_dsp/prob.h).</summary>
    public const int CdfProbTop = 1 << 15;

    /// <summary>Construct an encoder with an empty output buffer.</summary>
    public Av1RangeEncoder()
    {
        Reset();
    }

    /// <summary>Reset encoder state for reuse. Mirrors libaom <c>od_ec_enc_reset</c>.</summary>
    public void Reset()
    {
        _buf.Clear();
        _low = 0;
        _rng = 0x8000;
        // -9 so cnt crosses zero after one byte + one carry bit accumulates.
        _cnt = -9;
        _error = false;
    }

    /// <summary>
    /// Encode a single binary value with probability of 1 = <paramref name="f"/> / 32768.
    /// Mirrors libaom <c>od_ec_encode_bool_q15</c>.
    /// </summary>
    public void EncodeBoolQ15(int val, uint f)
    {
        if (f == 0 || f >= Av1RangeDecoder.CdfProbTop)
            throw new ArgumentOutOfRangeException(nameof(f), "must be in (0, 32768)");
        ulong l = _low;
        uint r = _rng;
        uint v = ((r >> 8) * (f >> Av1RangeDecoder.EcProbShift)) >> (7 - Av1RangeDecoder.EcProbShift);
        v += Av1RangeDecoder.EcMinProb;
        if (val != 0) l += r - v;
        r = val != 0 ? v : r - v;
        Normalize(l, r);
    }

    /// <summary>
    /// Encode a symbol via an inverse-CDF table in q15. Mirrors libaom
    /// <c>od_ec_encode_cdf_q15</c>. <paramref name="icdf"/> must be monotonically
    /// non-increasing with <c>icdf[nsyms - 1] == 0</c>.
    /// </summary>
    public void EncodeCdfQ15(int s, ReadOnlySpan<ushort> icdf, int nsyms)
    {
        if (s < 0 || s >= nsyms) throw new ArgumentOutOfRangeException(nameof(s));
        if (icdf[nsyms - 1] != 0)
            throw new ArgumentException("icdf[nsyms - 1] must be 0", nameof(icdf));
        EncodeQ15(s > 0 ? icdf[s - 1] : (ushort)Av1RangeDecoder.CdfProbTop, icdf[s], s, nsyms);
    }

    /// <summary>
    /// Encode <paramref name="ftb"/> raw bits packed at uniform probability.
    /// Mirrors the per-bit emulation pattern Av1RangeDecoder.DecodeBits uses.
    /// </summary>
    public void EncodeBits(uint value, int ftb)
    {
        if (ftb < 0 || ftb > 25) throw new ArgumentOutOfRangeException(nameof(ftb), "must be in [0, 25]");
        for (int i = ftb - 1; i >= 0; i--)
        {
            int bit = (int)((value >> i) & 1u);
            // Same fixed midpoint as DecodeBits for round-trip parity.
            EncodeBoolQ15(bit, 16384u);
        }
    }

    /// <summary>
    /// Number of whole bits "used" so far. Mirrors libaom <c>od_ec_enc_tell</c>.
    /// </summary>
    public int Tell
    {
        get
        {
            // libaom: (offs * 8) + cnt + 10
            // (cnt is offset by -9 so adding 10 brings it positive plus 1 for the carry bit reservation.)
            return _buf.Count * 8 + _cnt + 10;
        }
    }

    /// <summary>
    /// Finalize the bitstream and return the encoded bytes. Mirrors libaom
    /// <c>od_ec_enc_done</c> verbatim (including the m=0x3FFF rounding to the
    /// next 0x4000 boundary and the per-byte high-end emission with carry
    /// propagation). After calling this, the encoder must be Reset before
    /// re-use.
    /// </summary>
    public byte[] Done()
    {
        if (_error) throw new InvalidOperationException("encoder error");

        ulong l = _low;
        int c = _cnt;
        // libaom: s = 10; m = 0x3FFF; e = ((l+m) & ~m) | (m+1); s += c;
        ulong m = 0x3FFFUL;
        ulong e = ((l + m) & ~m) | (m + 1UL);
        int s = 10 + c;

        if (s > 0)
        {
            // n = ((uint64_t)1 << (c + 16)) - 1
            ulong n = (1UL << (c + 16)) - 1UL;
            while (s > 0)
            {
                // val = (uint16_t)(e >> (c + 16));
                int val = (int)((e >> (c + 16)) & 0xFFFFu);
                _buf.Add((byte)(val & 0xFF));
                if ((val & 0x0100) != 0)
                {
                    // Propagate carry backward through preceding bytes.
                    int idx = _buf.Count - 2;
                    while (idx >= 0)
                    {
                        int sum = _buf[idx] + 1;
                        _buf[idx] = (byte)(sum & 0xFF);
                        if ((sum >> 8) == 0) break;
                        idx--;
                    }
                }
                e &= n;
                s -= 8;
                c -= 8;
                n >>= 8;
            }
        }
        return _buf.ToArray();
    }

    private void EncodeQ15(uint fl, uint fh, int s, int nsyms)
    {
        ulong l = _low;
        uint r = _rng;
        uint u, v;
        int N = nsyms - 1;
        if (fl < Av1RangeDecoder.CdfProbTop)
        {
            u = ((r >> 8) * (fl >> Av1RangeDecoder.EcProbShift) >> (7 - Av1RangeDecoder.EcProbShift));
            u += (uint)(Av1RangeDecoder.EcMinProb * (N - (s - 1)));
            v = ((r >> 8) * (fh >> Av1RangeDecoder.EcProbShift) >> (7 - Av1RangeDecoder.EcProbShift));
            v += (uint)(Av1RangeDecoder.EcMinProb * (N - s));
            l += r - u;
            r = u - v;
        }
        else
        {
            uint sub = ((r >> 8) * (fh >> Av1RangeDecoder.EcProbShift) >> (7 - Av1RangeDecoder.EcProbShift));
            sub += (uint)(Av1RangeDecoder.EcMinProb * (N - s));
            r -= sub;
        }
        Normalize(l, r);
    }

    private void Normalize(ulong low, uint rng)
    {
        if (_error) return;
        int c = _cnt;
        int d = 16 - IlogNz(rng);
        int s = c + d;

        if (s >= 40)
        {
            int numBytesReady = (s >> 3) + 1;
            c += 24 - (numBytesReady << 3);

            ulong output = low >> c;
            low &= (1UL << c) - 1UL;

            ulong mask = 1UL << (numBytesReady << 3);
            ulong carry = output & mask;
            mask -= 1UL;
            output &= mask;

            // Write big-endian, MSB-first across numBytesReady bytes, advancing
            // a single output position from the top.
            int writeOffset = _buf.Count;
            // libaom's write_enc_data_to_out_buf emits 8 bytes total, masked
            // by num_bytes_ready, with carry propagation BACKWARD into the
            // already-emitted stream when carry != 0.
            Span<byte> tmp = stackalloc byte[8];
            ulong reg = (output << ((8 - numBytesReady) << 3));
            // Big-endian write
            BinaryPrimitives.WriteUInt64BigEndian(tmp, reg);
            for (int i = 0; i < numBytesReady; i++)
                _buf.Add(tmp[i]);

            if (carry != 0)
            {
                // Propagate carry backward: increment buf[writeOffset - 1] and ripple.
                int idx = writeOffset - 1;
                while (idx >= 0)
                {
                    int sum = _buf[idx] + 1;
                    _buf[idx] = (byte)(sum & 0xFF);
                    if ((sum >> 8) == 0) break;
                    idx--;
                }
            }

            s = c + d - 24;
        }
        _low = low << d;
        _rng = rng << d;
        _cnt = s;
    }

    private static int IlogNz(uint v)
    {
        if (v == 0) return 0;
        return 32 - System.Numerics.BitOperations.LeadingZeroCount(v);
    }
}
