// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 range encoder, GPU-callable form. Bit-exact mirror of
// Av1RangeEncoder. Used by GPU entropy kernels to keep the AV1
// bitstream output GPU-resident under the v3 host-as-pure-coordinator
// rule.
//
// The CPU version uses List<byte> for the output buffer (grows). GPU
// kernels can't allocate dynamically, so the caller pre-sizes a
// worst-case-bounded ArrayView<byte> and the encoder maintains an
// offset into it. The state struct + static helper methods pattern
// is so ILGPU can pass state by ref into the helper methods - same
// shape as Vp8BoolEncoderGpu / Vp9 entropy work.
//
// Carry propagation scans backward through the already-written
// portion of the output buffer and increments bytes until the carry
// is absorbed - identical algorithm to libaom's
// write_enc_data_to_out_buf when carry != 0.
//
// The previous-Tuvok comment in Av1RangeDecoder.cs ("Arithmetic
// coding is inherently sequential - so this implementation is pure
// C# CPU. No GPU kernel would help") was wrong per Captain's
// cardinal rule: single-thread-on-GPU is still GPU-resident, and
// that's the bar for the SpawnDev.Codecs library. The whole point
// is to keep data on the accelerator end-to-end.

using ILGPU;
using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// In-kernel state for the AV1 range encoder. Mirrors the internal
/// fields of <see cref="Av1RangeEncoder"/>.
/// </summary>
public struct Av1RangeEncoderGpuState
{
    /// <summary>Low value of the range coder (high bits of the current code).</summary>
    public ulong Low;

    /// <summary>Current range, normalized to [32768, 65535] when valid.</summary>
    public uint Rng;

    /// <summary>Bit counter; starts at -9 and increments as bytes are produced.</summary>
    public int Cnt;

    /// <summary>Number of bytes written to the output view so far.</summary>
    public long OutLen;
}

/// <summary>
/// Static GPU-callable helpers for the AV1 range encoder. Mirrors
/// <see cref="Av1RangeEncoder"/> bit-for-bit. Caller passes a
/// <see cref="Av1RangeEncoderGpuState"/> by ref and an
/// <see cref="ArrayView{Byte}"/> output buffer.
/// </summary>
public static class Av1RangeEncoderGpu
{
    /// <summary>EC_PROB_SHIFT from aom_dsp/entcode.h.</summary>
    public const int EcProbShift = Av1RangeDecoder.EcProbShift;
    /// <summary>EC_MIN_PROB from aom_dsp/entcode.h.</summary>
    public const int EcMinProb = Av1RangeDecoder.EcMinProb;
    /// <summary>q15 CDF top value.</summary>
    public const int CdfProbTop = Av1RangeDecoder.CdfProbTop;

    /// <summary>Initial state: low=0, rng=0x8000, cnt=-9, outLen=0.</summary>
    public static Av1RangeEncoderGpuState Init() => new Av1RangeEncoderGpuState
    {
        Low = 0,
        Rng = 0x8000,
        Cnt = -9,
        OutLen = 0,
    };

    /// <summary>
    /// Encode one binary symbol with q15 probability of 1 = <paramref name="f"/> / 32768.
    /// Mirrors <see cref="Av1RangeEncoder.EncodeBoolQ15"/>.
    /// </summary>
    public static void EncodeBoolQ15(
        ref Av1RangeEncoderGpuState state,
        ArrayView<byte> outBuf,
        int val,
        uint f)
    {
        ulong l = state.Low;
        uint r = state.Rng;
        uint v = ((r >> 8) * (f >> EcProbShift)) >> (7 - EcProbShift);
        v += EcMinProb;
        if (val != 0) l += r - v;
        r = val != 0 ? v : r - v;
        Normalize(ref state, outBuf, l, r);
    }

    /// <summary>
    /// Encode a symbol via an inverse-CDF table in q15. Mirrors
    /// <see cref="Av1RangeEncoder.EncodeCdfQ15"/>. <paramref name="icdf"/>
    /// is the icdf table starting at <paramref name="icdfBase"/>; values
    /// must be monotonically non-increasing with
    /// <c>icdf[icdfBase + nsyms - 1] == 0</c>.
    /// </summary>
    public static void EncodeCdfQ15(
        ref Av1RangeEncoderGpuState state,
        ArrayView<byte> outBuf,
        int s,
        ArrayView<ushort> icdf, long icdfBase, int nsyms)
    {
        uint fl = s > 0 ? icdf[icdfBase + (s - 1)] : (uint)CdfProbTop;
        uint fh = icdf[icdfBase + s];
        EncodeQ15(ref state, outBuf, fl, fh, s, nsyms);
    }

    /// <summary>
    /// Encode <paramref name="ftb"/> raw bits at uniform probability
    /// (midpoint 16384 / 32768). Mirrors
    /// <see cref="Av1RangeEncoder.EncodeBits"/>.
    /// </summary>
    public static void EncodeBits(
        ref Av1RangeEncoderGpuState state,
        ArrayView<byte> outBuf,
        uint value, int ftb)
    {
        for (int i = ftb - 1; i >= 0; i--)
        {
            int bit = (int)((value >> i) & 1u);
            EncodeBoolQ15(ref state, outBuf, bit, 16384u);
        }
    }

    /// <summary>
    /// Finalize the bitstream. Emits the closing bytes (with carry
    /// propagation) and updates state.OutLen. Mirrors
    /// <see cref="Av1RangeEncoder.Done"/>.
    /// </summary>
    public static void Done(
        ref Av1RangeEncoderGpuState state,
        ArrayView<byte> outBuf)
    {
        ulong l = state.Low;
        int c = state.Cnt;
        ulong m = 0x3FFFUL;
        ulong e = ((l + m) & ~m) | (m + 1UL);
        int s = 10 + c;

        if (s > 0)
        {
            ulong n = (1UL << (c + 16)) - 1UL;
            while (s > 0)
            {
                int valByte = (int)((e >> (c + 16)) & 0xFFFFu);
                outBuf[state.OutLen] = (byte)(valByte & 0xFF);
                state.OutLen++;
                if ((valByte & 0x0100) != 0)
                {
                    // Propagate carry backward through preceding bytes.
                    long idx = state.OutLen - 2;
                    while (idx >= 0)
                    {
                        int sum = outBuf[idx] + 1;
                        outBuf[idx] = (byte)(sum & 0xFF);
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
    }

    private static void EncodeQ15(
        ref Av1RangeEncoderGpuState state,
        ArrayView<byte> outBuf,
        uint fl, uint fh, int s, int nsyms)
    {
        ulong l = state.Low;
        uint r = state.Rng;
        uint u, v;
        int N = nsyms - 1;
        if (fl < CdfProbTop)
        {
            u = ((r >> 8) * (fl >> EcProbShift) >> (7 - EcProbShift));
            u += (uint)(EcMinProb * (N - (s - 1)));
            v = ((r >> 8) * (fh >> EcProbShift) >> (7 - EcProbShift));
            v += (uint)(EcMinProb * (N - s));
            l += r - u;
            r = u - v;
        }
        else
        {
            uint sub = ((r >> 8) * (fh >> EcProbShift) >> (7 - EcProbShift));
            sub += (uint)(EcMinProb * (N - s));
            r -= sub;
        }
        Normalize(ref state, outBuf, l, r);
    }

    private static void Normalize(
        ref Av1RangeEncoderGpuState state,
        ArrayView<byte> outBuf,
        ulong low, uint rng)
    {
        int c = state.Cnt;
        int d = 16 - IlogNz(rng);
        int sBits = c + d;

        if (sBits >= 40)
        {
            int numBytesReady = (sBits >> 3) + 1;
            c += 24 - (numBytesReady << 3);

            ulong output = low >> c;
            low &= (1UL << c) - 1UL;

            ulong mask = 1UL << (numBytesReady << 3);
            ulong carry = output & mask;
            mask -= 1UL;
            output &= mask;

            long writeOffset = state.OutLen;
            // Big-endian write of the top numBytesReady bytes from
            // (output << (8 - numBytesReady) * 8). We unpack manually
            // since ILGPU doesn't have BinaryPrimitives in kernels.
            ulong reg = output << ((8 - numBytesReady) << 3);
            for (int i = 0; i < numBytesReady; i++)
            {
                int shift = (7 - i) << 3;
                outBuf[state.OutLen] = (byte)((reg >> shift) & 0xFFu);
                state.OutLen++;
            }

            if (carry != 0)
            {
                long idx = writeOffset - 1;
                while (idx >= 0)
                {
                    int sum = outBuf[idx] + 1;
                    outBuf[idx] = (byte)(sum & 0xFF);
                    if ((sum >> 8) == 0) break;
                    idx--;
                }
            }

            sBits = c + d - 24;
        }
        state.Low = low << d;
        state.Rng = rng << d;
        state.Cnt = sBits;
    }

    /// <summary>
    /// OD_ILOG_NZ(v) = position of the highest set bit of v + 1 (v != 0).
    /// Returns 0 for v == 0 to match the CPU helper's degenerate
    /// behaviour (caller never invokes with rng == 0 in practice).
    /// Manual loop instead of CLZ to lower cleanly on every backend
    /// without intrinsic dependencies.
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
