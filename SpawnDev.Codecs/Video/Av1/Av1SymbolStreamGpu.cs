// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 entropy coding in two steps: symbols first, the range coder after.
//
// With CDF updates disabled (the keyframe encoder writes disable_cdf_update = 1) a symbol's probabilities come
// from a fixed table and a context, and the context comes from coefficients, never from the range coder. So
// everything the entropy coder does except the range coder's own state (low, range, the bytes written so far) can
// be worked out for every block at once, in any order. This file is the hand-over between the two: one 32-bit
// entry per symbol, holding exactly what Av1RangeEncoderGpu needs to code it.
//
//   CDF symbol s of an N + 1 symbol alphabet with inverse CDF values fl (below s; 32768 for s = 0) and fh (at s):
//     bits  0 ..  9   fl >> 6   (0 .. 512; 512 = "s is the first symbol")
//     bits 10 .. 19   fh >> 6   (0 .. 511)
//     bits 20 .. 24   N - s     (0 .. 15: what EC_MIN_PROB is multiplied with)
//     bit  31         0
//   The range coder uses fl and fh only shifted right by EC_PROB_SHIFT (6), so nothing is lost.
//
//   A run of raw bits (sign bits, Golomb codes, the low bits of eob_extra), most significant first:
//     bits  0 .. 23   the bits
//     bits 24 .. 28   how many (1 .. 24)
//     bit  31         1
//   A raw bit is the two-symbol CDF { 16384, 0 }; the run only saves entries.
//
// A slot (one 16x16 position of a tile) holds its entry count in its first int, then the entries.

using ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>The symbol entries that carry AV1 entropy symbols from the pass that derives them to the range coder.</summary>
public static class Av1SymbolStreamGpu
{
    /// <summary>The most raw bits one run entry holds.</summary>
    public const int MaxRunBits = 24;

    /// <summary>
    /// Ints of a slot: its count, then the entries of one 16x16 position at the worst case. Per coefficient at most
    /// 8 entries (the base level, 4 range symbols, a sign, and a Golomb code as two runs); 384 coefficients; per
    /// transform block (3, or 12 for four 8x8 leaves) txb_skip, tx_type, eob_pt, eob_extra and its bits: 5; per
    /// leaf 3 mode symbols; 7 partition symbols: 3072 + 60 + 12 + 7 = 3151.
    /// </summary>
    public const int SlotInts = 3200;

    /// <summary>
    /// Append CDF symbol <paramref name="s"/> of the <paramref name="nsyms"/>-symbol inverse CDF at
    /// <paramref name="icdfBase"/> (what <see cref="Av1RangeEncoderGpu.EncodeCdfQ15"/> takes) to the slot at
    /// <paramref name="slotBase"/>, whose entry count so far is <paramref name="n"/>.
    /// </summary>
    public static void Cdf(
        ArrayView<int> sym, long slotBase, ref int n,
        int s, ArrayView<ushort> icdf, long icdfBase, int nsyms)
    {
        int fl = s > 0 ? icdf[icdfBase + (s - 1)] : Av1RangeEncoderGpu.CdfProbTop;
        int fh = icdf[icdfBase + s];
        n++;
        sym[slotBase + n] = (fl >> Av1RangeEncoderGpu.EcProbShift)
            | ((fh >> Av1RangeEncoderGpu.EcProbShift) << 10)
            | ((nsyms - 1 - s) << 20);
    }

    /// <summary>
    /// Append <paramref name="count"/> raw bits of <paramref name="value"/> (most significant first, what
    /// <see cref="Av1RangeEncoderGpu.EncodeBits"/> takes), in runs of at most <see cref="MaxRunBits"/>.
    /// </summary>
    public static void Bits(ArrayView<int> sym, long slotBase, ref int n, uint value, int count)
    {
        while (count > 0)
        {
            int take = count > MaxRunBits ? MaxRunBits : count;
            int shift = count - take;
            int bits = (int)((value >> shift) & ((1u << take) - 1u));
            n++;
            sym[slotBase + n] = unchecked((int)0x80000000) | (take << 24) | bits;
            count = shift;
        }
    }

    /// <summary>Code one entry with the range coder.</summary>
    public static void Encode(ref Av1RangeEncoderGpuState re, ArrayView<byte> outBuf, int entry)
    {
        if (entry < 0)
        {
            int count = (entry >> 24) & 31;
            for (int i = count - 1; i >= 0; i--)
                Av1RangeEncoderGpu.EncodeBoolQ15(ref re, outBuf, (entry >> i) & 1, 16384u);
        }
        else
        {
            Av1RangeEncoderGpu.EncodeShifted(ref re, outBuf,
                (uint)(entry & 1023), (uint)((entry >> 10) & 1023), (uint)((entry >> 20) & 31));
        }
    }
}
