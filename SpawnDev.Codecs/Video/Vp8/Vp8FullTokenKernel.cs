// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Full-syntax VP8 keyframe decoder, kernel 3: DCT token decoding (RFC 6386 section 13, libvpx detokenize.c), ONE
// macroblock row per launch on ONE GPU thread. Macroblock row N is coded in token partition (N mod number of
// partitions); each partition's bool decoder lives in the state buffer between launches. The entropy contexts (does the
// block above / to the left have coefficients) are shared by all partitions, which is why the rows run in order.
//
// Per macroblock: Y2 first (block type 1) when the macroblock is not B_PRED, then 16 luma blocks (type 0 starting at
// position 1 when there is a Y2 block, else type 3 from position 0), then 4 U and 4 V blocks (type 2). Coefficients are
// stored RAW (not dequantized) in zig-zag-undone raster order at coeffs[mb * 400 + block * 16 + position]. A macroblock
// with mb_skip_coeff has no tokens: its contexts are cleared (the Y2 context only when the macroblock has a Y2 block,
// because a skipped B_PRED macroblock leaves the Y2 context of its neighbours alone).

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Video.Vp8;

internal static class Vp8FullTokenKernel
{
    public const int CoeffsPerMb = 25 * 16;

    /// <summary>Macroblocks per launch of the token decode (one thread; ~0.2 ms per busy macroblock on a desktop GPU).</summary>
    public const int MbsPerLaunch = 32;

    /// <summary>
    /// Macroblocks <paramref name="colStart"/> .. <paramref name="colEnd"/> - 1 of macroblock row <paramref name="mbRow"/>. The
    /// host launches a row in chunks of <see cref="MbsPerLaunch"/> so no launch grows with the picture width; the left
    /// non-zero context carries over in <c>aboveCtx[mbCols * 9 ..]</c> (9 bytes after the above context), the bool decoder
    /// state in its slot.
    /// </summary>
    public static void TokenKernel(
        Index1D _,
        ArrayView<byte> enc, ArrayView<int> meta, ArrayView<int> bdStates,
        ArrayView<byte> recs, ArrayView<byte> probs, ArrayView<short> coeffs,
        ArrayView<byte> aboveCtx, ArrayView<byte> tables, int mbRow, int colStart, int colEnd)
    {
        if (meta[Vp8FullMeta.Error] != 0) return;
        int mbCols = meta[Vp8FullMeta.MbCols];
        int nparts = meta[Vp8FullMeta.NumParts];
        int slot = 1 + (mbRow & (nparts - 1));
        var bd = Vp8BdOps.Load(bdStates, slot);
        long leftBase = (long)mbCols * 9;
        var left = LocalMemory.Allocate<int>(9);
        for (int k = 0; k < 9; k++) left[k] = colStart == 0 ? 0 : aboveCtx[leftBase + k];
        int colLast = colEnd < mbCols ? colEnd : mbCols;

        for (int mbCol = colStart; mbCol < colLast; mbCol++)
        {
            long mb = (long)mbRow * mbCols + mbCol;
            long rec = mb * Vp8MbRecord.Size;
            long a = (long)mbCol * 9;
            long cBase = mb * CoeffsPerMb;
            int yMode = recs[rec + Vp8MbRecord.YMode];
            int skip = recs[rec + Vp8MbRecord.Skip];
            bool hasY2 = yMode != 4;

            if (skip != 0)
            {
                for (int k = 0; k < 8; k++) { aboveCtx[a + k] = 0; left[k] = 0; }
                if (hasY2) { aboveCtx[a + 8] = 0; left[8] = 0; }
            }
            else
            {
                int total = 0;
                int first = 0;
                int yType = 3;
                if (hasY2)
                {
                    int ctx = aboveCtx[a + 8] + left[8];
                    int eob = DecodeBlock(ref bd, enc, probs, tables, coeffs, cBase + 24 * 16, 1 * 264, ctx, 0);
                    int flag = eob > 0 ? 1 : 0;
                    aboveCtx[a + 8] = (byte)flag;
                    left[8] = flag;
                    total += eob;
                    first = 1;
                    yType = 0;
                }
                for (int i = 0; i < 16; i++)
                {
                    int ai = i & 3;
                    int li = i >> 2;
                    int ctx = aboveCtx[a + ai] + left[li];
                    int eob = DecodeBlock(ref bd, enc, probs, tables, coeffs, cBase + i * 16, yType * 264, ctx, first);
                    int flag = eob > 0 ? 1 : 0;
                    aboveCtx[a + ai] = (byte)flag;
                    left[li] = flag;
                    total += eob;
                }
                for (int i = 16; i < 24; i++)
                {
                    int j = i - 16;
                    int plane = j >> 2;            // 0 = U, 1 = V
                    int jj = j & 3;
                    int ai = 4 + plane * 2 + (jj & 1);
                    int li = 4 + plane * 2 + (jj >> 1);
                    int ctx = aboveCtx[a + ai] + left[li];
                    int eob = DecodeBlock(ref bd, enc, probs, tables, coeffs, cBase + i * 16, 2 * 264, ctx, 0);
                    int flag = eob > 0 ? 1 : 0;
                    aboveCtx[a + ai] = (byte)flag;
                    left[li] = flag;
                    total += eob;
                }
                recs[rec + Vp8MbRecord.HasCoef] = (byte)(total > 0 ? 1 : 0);
            }
        }

        for (int k = 0; k < 9; k++) aboveCtx[leftBase + k] = (byte)left[k];
        Vp8BdOps.Save(bdStates, slot, bd);
        if (bd.Err != 0) meta[Vp8FullMeta.Error] = (int)Vp8DecodeError.TokenPartitionTruncated;
    }

    /// <summary>Decode one 4x4 block; returns its end-of-block position (0 = no coefficients).</summary>
    private static int DecodeBlock(
        ref Vp8Bd bd, ArrayView<byte> enc, ArrayView<byte> probs, ArrayView<byte> tables,
        ArrayView<short> coeffs, long cOff, int typeBase, int ctx, int first)
    {
        int n = first;
        int p = typeBase + tables[Vp8FullTables.Bands + n] * 33 + ctx * 11;
        int eob = 0;
        int firstBit = Vp8BdOps.Bool(ref bd, enc, probs[p]);
        bool active = firstBit != 0;
        while (active)
        {
            n++;
            int isNonZero = Vp8BdOps.Bool(ref bd, enc, probs[p + 1]);
            if (isNonZero == 0)
            {
                // a zero token: no end-of-block check follows it
                p = typeBase + tables[Vp8FullTables.Bands + n] * 33;
                if (n == 16) { eob = 16; active = false; }
            }
            else
            {
                int v;
                int nextCtx = 2;
                int notOne = Vp8BdOps.Bool(ref bd, enc, probs[p + 2]);
                if (notOne == 0)
                {
                    v = 1;
                    nextCtx = 1;
                }
                else
                {
                    int big = Vp8BdOps.Bool(ref bd, enc, probs[p + 3]);
                    if (big == 0)
                    {
                        int notTwo = Vp8BdOps.Bool(ref bd, enc, probs[p + 4]);
                        if (notTwo == 0) v = 2;
                        else v = 3 + Vp8BdOps.Bool(ref bd, enc, probs[p + 5]);
                    }
                    else
                    {
                        int huge = Vp8BdOps.Bool(ref bd, enc, probs[p + 6]);
                        if (huge == 0)
                        {
                            int cat2 = Vp8BdOps.Bool(ref bd, enc, probs[p + 7]);
                            if (cat2 == 0)
                            {
                                v = 5 + Vp8BdOps.Bool(ref bd, enc, 159);                  // DCT_CAT1
                            }
                            else
                            {
                                v = 7 + 2 * Vp8BdOps.Bool(ref bd, enc, 165);               // DCT_CAT2
                                v += Vp8BdOps.Bool(ref bd, enc, 145);
                            }
                        }
                        else
                        {
                            int bit1 = Vp8BdOps.Bool(ref bd, enc, probs[p + 8]);
                            int bit0 = Vp8BdOps.Bool(ref bd, enc, probs[p + 9 + bit1]);
                            int cat = 2 * bit1 + bit0;                                     // DCT_CAT3..6
                            int tab = Vp8FullTables.Cat3;
                            int width = 3;
                            if (cat == 1) { tab = Vp8FullTables.Cat4; width = 4; }
                            else if (cat == 2) { tab = Vp8FullTables.Cat5; width = 5; }
                            else if (cat == 3) { tab = Vp8FullTables.Cat6; width = 11; }
                            v = 0;
                            for (int t = 0; t < width; t++) v = v + v + Vp8BdOps.Bool(ref bd, enc, tables[tab + t]);
                            v += 3 + (8 << cat);
                        }
                    }
                }
                int sign = Vp8BdOps.Bool(ref bd, enc, 128);
                if (sign != 0) v = -v;
                coeffs[cOff + tables[Vp8FullTables.Zigzag + n - 1]] = (short)v;
                p = typeBase + tables[Vp8FullTables.Bands + n] * 33 + nextCtx * 11;
                eob = n;
                if (n == 16)
                {
                    active = false;
                }
                else
                {
                    int more = Vp8BdOps.Bool(ref bd, enc, probs[p]);
                    active = more != 0;
                }
            }
        }
        return eob;
    }
}
