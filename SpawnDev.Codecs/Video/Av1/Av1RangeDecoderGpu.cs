// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 range decoder, GPU-callable form. Bit-exact mirror of
// Av1RangeDecoder. Symmetric companion to Av1RangeEncoderGpu under
// the v3 host-as-pure-coordinator rule.
//
// State holds the running (dif, rng, cnt, bptr) quartet plus the
// buffer end. Static helper methods take state by ref and the input
// buffer as ArrayView<byte>. Same pattern as Vp8BoolDecoderGpu.
//
// AV1 spec note: end-of-stream is handled by stuffing virtual zero
// bits when the buffer runs out (cnt gets bumped by LotsOfBits),
// matching libaom od_ec_dec_init refill semantics.

using ILGPU;
using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// In-kernel state for the AV1 range decoder. Mirrors the internal
/// fields of <see cref="Av1RangeDecoder"/>.
/// </summary>
public struct Av1RangeDecoderGpuState
{
    /// <summary>Current input buffer offset (absolute).</summary>
    public int Bptr;

    /// <summary>Buffer start offset (the value passed to Init).</summary>
    public int BufStart;

    /// <summary>Buffer end offset (exclusive).</summary>
    public int BufEnd;

    /// <summary>Virtual-zero bits past end of stream.</summary>
    public int TellOffs;

    /// <summary>Bit window (high 16 bits feed the next symbol).</summary>
    public uint Dif;

    /// <summary>Current range, normalized to [32768, 65535].</summary>
    public uint Rng;

    /// <summary>Bits remaining in dif.</summary>
    public int Cnt;
}

/// <summary>
/// Static GPU-callable helpers for the AV1 range decoder. Mirrors
/// <see cref="Av1RangeDecoder"/> bit-for-bit.
/// </summary>
public static class Av1RangeDecoderGpu
{
    /// <summary>EC_PROB_SHIFT.</summary>
    public const int EcProbShift = Av1RangeDecoder.EcProbShift;
    /// <summary>EC_MIN_PROB.</summary>
    public const int EcMinProb = Av1RangeDecoder.EcMinProb;
    /// <summary>OD_EC_WINDOW_SIZE.</summary>
    public const int OdEcWindowSize = Av1RangeDecoder.OdEcWindowSize;
    /// <summary>OD_EC_LOTS_OF_BITS.</summary>
    public const int OdEcLotsOfBits = Av1RangeDecoder.OdEcLotsOfBits;
    /// <summary>q15 CDF top.</summary>
    public const int CdfProbTop = Av1RangeDecoder.CdfProbTop;

    /// <summary>
    /// Initialize the decoder state to read <paramref name="length"/>
    /// bytes from <paramref name="buf"/> starting at <paramref name="offset"/>.
    /// Mirrors libaom <c>od_ec_dec_init</c>.
    /// </summary>
    public static Av1RangeDecoderGpuState Init(
        ArrayView<byte> buf, int offset, int length)
    {
        var state = new Av1RangeDecoderGpuState
        {
            BufStart = offset,
            BufEnd = offset + length,
            Bptr = offset,
            TellOffs = 10 - (OdEcWindowSize - 8),
            Dif = (1u << (OdEcWindowSize - 1)) - 1u,
            Rng = 0x8000,
            Cnt = -15,
        };
        Refill(ref state, buf);
        return state;
    }

    /// <summary>
    /// Decode one binary value with q15 probability of 1 = <paramref name="f"/> / 32768.
    /// Mirrors libaom <c>od_ec_decode_bool_q15</c>.
    /// </summary>
    public static int DecodeBoolQ15(
        ref Av1RangeDecoderGpuState state,
        ArrayView<byte> buf,
        uint f)
    {
        uint dif = state.Dif;
        uint r = state.Rng;
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
        Normalize(ref state, buf, dif, rNew);
        return ret;
    }

    /// <summary>
    /// Decode a symbol given an inverse CDF table in q15. Mirrors
    /// <see cref="Av1RangeDecoder.DecodeCdfQ15"/>. The icdf table
    /// must be monotonically non-increasing with
    /// <c>icdf[icdfBase + nsyms - 1] == 0</c>.
    /// </summary>
    public static int DecodeCdfQ15(
        ref Av1RangeDecoderGpuState state,
        ArrayView<byte> buf,
        ArrayView<ushort> icdf, long icdfBase, int nsyms)
    {
        uint dif = state.Dif;
        uint r = state.Rng;
        int N = nsyms - 1;
        uint c = dif >> (OdEcWindowSize - 16);
        uint v = r;
        uint u;
        int ret = -1;
        do
        {
            u = v;
            ret++;
            uint icdfVal = icdf[icdfBase + ret];
            v = ((r >> 8) * (icdfVal >> EcProbShift)) >> (7 - EcProbShift);
            v += (uint)(EcMinProb * (N - ret));
        } while (c < v);

        r = u - v;
        dif -= v << (OdEcWindowSize - 16);
        Normalize(ref state, buf, dif, r);
        return ret;
    }

    /// <summary>
    /// Read <paramref name="ftb"/> raw bits at uniform probability.
    /// Mirrors <see cref="Av1RangeDecoder.DecodeBits"/>.
    /// </summary>
    public static uint DecodeBits(
        ref Av1RangeDecoderGpuState state,
        ArrayView<byte> buf,
        int ftb)
    {
        uint result = 0;
        for (int i = 0; i < ftb; i++)
        {
            uint dif = state.Dif;
            uint r = state.Rng;
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
            Normalize(ref state, buf, dif, rNew);
            result = (result << 1) | bit;
        }
        return result;
    }

    private static void Refill(ref Av1RangeDecoderGpuState state, ArrayView<byte> buf)
    {
        int s = OdEcWindowSize - 9 - (state.Cnt + 15);
        for (; s >= 0 && state.Bptr < state.BufEnd; s -= 8, state.Bptr++)
        {
            state.Dif ^= ((uint)buf[state.Bptr]) << s;
            state.Cnt += 8;
        }
        if (state.Bptr >= state.BufEnd)
        {
            state.TellOffs += OdEcLotsOfBits - state.Cnt;
            state.Cnt = OdEcLotsOfBits;
        }
    }

    private static void Normalize(
        ref Av1RangeDecoderGpuState state,
        ArrayView<byte> buf,
        uint dif, uint rng)
    {
        int d = 16 - IlogNz(rng);
        state.Cnt -= d;
        state.Dif = ((dif + 1u) << d) - 1u;
        state.Rng = rng << d;
        if (state.Cnt < 0) Refill(ref state, buf);
    }

    /// <summary>
    /// OD_ILOG_NZ(v) = position of the highest set bit of v + 1 (v != 0).
    /// Manual loop for cross-backend portability.
    /// </summary>
    private static int IlogNz(uint v)
    {
        int pos = 0;
        while (v != 0)
        {
            pos++;
            v >>= 1;
        }
        return pos;
    }
}
