// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 coefficient decoder. Bit-exact port of libaom
// av1/decoder/decodetxb.c <c>av1_read_coeffs_txb</c>.
//
// Upstream Copyright (c) 2017, Alliance for Open Media. All rights reserved.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.
//
// Reads transform coefficients for one transform block from the entropy
// stream. Writes a sparse coefficient buffer (length = txWide * txHigh,
// most entries zero except for indices 0..eob-1 in scan order).
//
// The coefficient decoder is the largest single piece of the AV1 decoder
// after the entropy state itself. The decode path:
//   1. txb_skip CDF -> early-exit if entire block is zero
//   2. tx_type read (intra_ext_tx CDF, conditional on size)
//   3. eob_multi CDF -> coarse EOB position class
//   4. eob_extra (raw bits) -> exact EOB position
//   5. coeff_base_eob CDF for the EOB-position level (1..3)
//   6. coeff_base CDF for non-EOB scan positions (running magnitude tracker)
//   7. coeff_br CDF for high-magnitude refinement (level 4..15)
//   8. read_golomb for very-high magnitudes (level >= 15)
//   9. dc_sign CDF for the DC coefficient sign
//   10. Per-AC sign bit
//   11. Apply dequant scale per (qindex, plane, dc/ac) and write into output
//
// Spec reference: AV1 Bitstream and Decoding Process Specification
//   sec 5.11.39 Coefficients syntax
//   sec 9.4.2  Initialization process for the coefficient decoder
//   sec 7.4    Coefficient decoder process

using SpawnDev.Codecs.EntropyCoders;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 coefficient (token) decoder.</summary>
internal static class Av1CoefDecoder
{
    /// <summary>Result of decoding one transform block's coefficients.</summary>
    public sealed class CoefBlock
    {
        /// <summary>End-of-block scan index (exclusive); 0 means all-zero.</summary>
        public int Eob;
        /// <summary>Decoded transform type (after av1_read_tx_type or default).</summary>
        public Av1TxType TxType;
        /// <summary>Dequantized coefficients in raster (row-major, length = w*h).</summary>
        public int[] DqCoeffs = Array.Empty<int>();
        /// <summary>Cumulative absolute sum (libaom cul_level + dc_sign packed in top bits).</summary>
        public int CulLevel;
    }

    /// <summary>
    /// libaom <c>get_q_ctx</c>. Token CDFs are quantized into 4 buckets by qindex:
    ///   qindex &lt;= 20  -> 0
    ///   qindex &lt;= 60  -> 1
    ///   qindex &lt;= 120 -> 2
    ///   else            -> 3
    /// </summary>
    public static int GetQctx(int qindex)
    {
        if (qindex <= 20) return 0;
        if (qindex <= 60) return 1;
        if (qindex <= 120) return 2;
        return 3;
    }

    /// <summary>
    /// libaom <c>av1_get_ext_tx_set_type</c>. EXT_TX_SET_TYPE enum:
    ///   0 = DCTONLY, 1 = DCT_IDTX, 2 = DTT4_IDTX, 3 = DTT4_IDTX_1DDCT,
    ///   4 = DTT9_IDTX_1DDCT, 5 = ALL16.
    /// </summary>
    private static int GetExtTxSetType(Av1TxSize txSize, bool isInter, bool useReducedSet)
    {
        int w = Av1TxSizeInfo.TxWide[(int)txSize];
        int h = Av1TxSizeInfo.TxHigh[(int)txSize];
        int sqrUp = Math.Max(w, h);
        if (sqrUp > 32) return 0;
        if (sqrUp == 32) return isInter ? 1 : 0;
        if (useReducedSet) return isInter ? 1 : 2;
        int sqr = Math.Min(w, h);
        bool is16 = (sqr == 16);
        if (!isInter) return is16 ? 2 : 3;
        return is16 ? 4 : 5;
    }

    /// <summary>libaom <c>get_ext_tx_set</c>: ext_tx_set_index[is_inter][set_type].</summary>
    private static int GetExtTxSetIndex(int setType, bool isInter)
    {
        if (!isInter)
        {
            return setType switch { 0 => 0, 2 => 2, 3 => 1, _ => -1 };
        }
        return setType switch { 0 => 0, 1 => 3, 4 => 2, 5 => 1, _ => -1 };
    }

    /// <summary>libaom <c>av1_num_ext_tx_set[EXT_TX_SET_TYPES]</c>.</summary>
    private static readonly int[] NumExtTxSet = new int[] { 1, 2, 5, 7, 12, 16 };

    /// <summary>
    /// libaom <c>av1_ext_tx_inv[EXT_TX_SET_TYPES][TX_TYPES]</c>: maps a CDF
    /// symbol (read index) back to the actual TX_TYPE for the active set.
    /// </summary>
    private static readonly int[][] ExtTxInv = new int[][]
    {
        new int[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        new int[] { 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        new int[] { 9, 0, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        new int[] { 9, 0, 10, 11, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        new int[] { 9, 10, 11, 0, 1, 2, 4, 5, 3, 6, 7, 8, 0, 0, 0, 0 },
        new int[] { 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 4, 5, 3, 6, 7, 8 },
    };

    /// <summary>
    /// Map a tx_size to the 0..3 "square TX size" index used by intra_ext_tx CDF
    /// (4x4=0, 8x8=1, 16x16=2, 32x32=3, capping by max(w,h)).
    /// </summary>
    private static int SquareTxSizeIndex(Av1TxSize txSize)
    {
        int w = Av1TxSizeInfo.TxWide[(int)txSize];
        int h = Av1TxSizeInfo.TxHigh[(int)txSize];
        int s = Math.Min(Math.Max(w, h), 32);
        return s switch { 4 => 0, 8 => 1, 16 => 2, 32 => 3, _ => 3 };
    }

    /// <summary>
    /// Decode one transform block. Mirrors libaom <c>av1_read_coeffs_txb</c>.
    /// </summary>
    public static CoefBlock ReadCoeffsTxb(
        Av1RangeDecoder rd,
        Av1TxSize txSize,
        int plane,        // 0 = Y, 1 = U, 2 = V
        Av1IntraMode intraMode,
        int qindex,
        Av1QuantParams quant,
        int bitDepth,
        bool reducedTxSet,
        int txbSkipCtx,
        int dcSignCtx,
        Av1TxType? forcedTxType = null)
    {
        // Plane type: 0 = Y, 1 = UV.
        int planeType = plane == 0 ? 0 : 1;
        int txsCtx = Av1TxbCommon.GetTxSizeEntropyCtx(txSize);
        int width = Av1TxbCommon.GetTxbWide(txSize);
        int height = Av1TxbCommon.GetTxbHigh(txSize);
        int bhl = Av1TxbCommon.GetTxbBhl(txSize);
        // Full output dimensions (the actual TX block size, not the
        // entropy-coded subset). 64x64 carries only 32x32 coefs but the
        // residual buffer is full 64x64.
        int outW = Av1TxSizeInfo.TxWide[(int)txSize];
        int outH = Av1TxSizeInfo.TxHigh[(int)txSize];
        int n = outW * outH;

        // qctx: token CDFs are quantized by the per-frame qindex (libaom get_q_ctx).
        int qctx = GetQctx(qindex);

        // Step 1: txb_skip - one bit per tx block.
        var skipCdf = Av1DefaultCoefCdfs.DefaultTxbSkipCdf[qctx][txsCtx][txbSkipCtx];
        int allZero = rd.DecodeCdfQ15(skipCdf, 2);
        var result = new CoefBlock
        {
            Eob = 0,
            TxType = Av1TxType.DctDct,
            DqCoeffs = new int[n],
            CulLevel = 0,
        };
        if (allZero != 0)
        {
            return result;
        }

        // Step 2: tx_type read (Y plane only, qindex > 0). Mirrors libaom
        // av1_read_tx_type: reads a symbol from intra_ext_tx_cdf and remaps
        // through av1_ext_tx_inv to a TX_TYPE enum value.
        Av1TxType txType = Av1TxType.DctDct;
        if (forcedTxType.HasValue)
        {
            txType = forcedTxType.Value;
        }
        else if (plane == 0 && qindex > 0)
        {
            int setType = GetExtTxSetType(txSize, false, reducedTxSet);
            int numSyms = NumExtTxSet[setType];
            if (numSyms > 1)
            {
                int eset = GetExtTxSetIndex(setType, false);
                if (eset > 0)
                {
                    int squareIdx = SquareTxSizeIndex(txSize);
                    var extCdf = Av1DefaultTxfmCdfs.DefaultIntraExtTxCdf[eset][squareIdx][(int)intraMode];
                    int sym = rd.DecodeCdfQ15(extCdf, numSyms);
                    txType = (Av1TxType)ExtTxInv[setType][sym];
                }
            }
        }
        result.TxType = txType;

        int txClass = Av1TxbCommon.TxTypeToClass[(int)txType];
        var scan = Av1ScanTables.Scan[(int)txSize][(int)txType];

        // Step 3: eob_multi - coarse EOB position class.
        int eobPt = ReadEobMulti(rd, txSize, planeType, txClass, qctx);

        // Step 4: eob_extra - raw bits to refine. libaom uses
        // av1_eob_offset_bits[eob_pt] (0-indexed lookup).
        int eobExtra = 0;
        int eobOffsetBits = Av1TxbCommon.EobOffsetBits[eobPt];
        if (eobOffsetBits > 0)
        {
            int eobCtx = eobPt - 3;
            var extraCdf = Av1DefaultCoefCdfs.DefaultEobExtraCdf[qctx][txsCtx][planeType][eobCtx];
            int bit = rd.DecodeCdfQ15(extraCdf, 2);
            if (bit != 0) eobExtra += 1 << (eobOffsetBits - 1);
            for (int i = 1; i < eobOffsetBits; i++)
            {
                bit = (int)rd.DecodeBits(1);
                if (bit != 0) eobExtra += 1 << (eobOffsetBits - 1 - i);
            }
        }
        int eob = Av1TxbCommon.RecEobPos(eobPt, eobExtra);
        if (eob <= 0 || eob > n)
        {
            // Defensive: malformed stream would corrupt our state. Bail out.
            return result;
        }
        result.Eob = eob;

        // Step 5: levels[] padded buffer (libaom's set_levels layout).
        int paddedStride = (1 << bhl) + Av1TxbCommon.TxPadHor;
        int paddedRows = height + Av1TxbCommon.TxPadVer;
        var levelsBuf = new byte[paddedStride * paddedRows + Av1TxbCommon.TxPadEnd];
        int levelsOff = Av1TxbCommon.SetLevelsOffset(height);

        // Step 6: read the EOB-position coefficient (level only, no sign yet).
        {
            int c = eob - 1;
            int pos = scan[c];
            int coefCtx = Av1TxbCommon.GetLowerLevelsCtxEob(bhl, width, c);
            var baseEobCdf = Av1DefaultCoefCdfs.DefaultCoeffBaseEobMultiCdf[qctx][txsCtx][planeType][coefCtx];
            int level = rd.DecodeCdfQ15(baseEobCdf, 3) + 1;
            if (level > Av1TxbCommon.NumBaseLevels)
            {
                int brCtx = Av1TxbCommon.GetBrCtxEob(pos, bhl, txClass);
                var brCdf = Av1DefaultCoefCdfs.DefaultCoeffLpsMultiCdf[qctx][Math.Min(txsCtx, 3)][planeType][brCtx];
                for (int idx = 0; idx < Av1TxbCommon.CoeffBaseRange; idx += Av1TxbCommon.BrCdfSize - 1)
                {
                    int k = rd.DecodeCdfQ15(brCdf, Av1TxbCommon.BrCdfSize);
                    level += k;
                    if (k < Av1TxbCommon.BrCdfSize - 1) break;
                }
            }
            int padIdx = Av1TxbCommon.GetPaddedIdx(pos, bhl);
            levelsBuf[levelsOff + padIdx] = (byte)Math.Min(level, 255);
        }

        // Step 7: read non-EOB coefficients in REVERSE scan order (eob-2 .. 0).
        if (eob > 1)
        {
            for (int c = eob - 2; c >= 0; c--)
            {
                int pos = scan[c];
                int coefCtx = c == 0
                    ? Av1TxbCommon.GetLowerLevelsCtxEob(bhl, width, 0)
                    : (txClass == Av1TxbCommon.TxClass2d
                        ? Av1TxbCommon.GetLowerLevelsCtx2d(levelsBuf, pos, bhl, txSize)
                        : Av1TxbCommon.GetLowerLevelsCtx(levelsBuf, pos, bhl, txSize, txClass));

                var baseCdf = Av1DefaultCoefCdfs.DefaultCoeffBaseMultiCdf[qctx][txsCtx][planeType][coefCtx];
                int level = rd.DecodeCdfQ15(baseCdf, 4);
                if (level > Av1TxbCommon.NumBaseLevels)
                {
                    int brCtx = Av1TxbCommon.GetBrCtx(levelsBuf, pos, bhl, txClass);
                    var brCdf = Av1DefaultCoefCdfs.DefaultCoeffLpsMultiCdf[qctx][Math.Min(txsCtx, 3)][planeType][brCtx];
                    for (int idx = 0; idx < Av1TxbCommon.CoeffBaseRange; idx += Av1TxbCommon.BrCdfSize - 1)
                    {
                        int k = rd.DecodeCdfQ15(brCdf, Av1TxbCommon.BrCdfSize);
                        level += k;
                        if (k < Av1TxbCommon.BrCdfSize - 1) break;
                    }
                }
                int padIdx = Av1TxbCommon.GetPaddedIdx(pos, bhl);
                levelsBuf[levelsOff + padIdx] = (byte)Math.Min(level, 255);
            }
        }

        // Step 8: signs + dequant. Walk scan 0..eob-1.
        int culLevel = 0;
        int dcVal = 0;
        // Dequant scale per plane: DC for c==0, AC for c>0.
        short qDc = plane == 0
            ? Av1DequantTables.DcQuantQtx(qindex, quant.YDcDeltaQ, bitDepth)
            : (plane == 1
                ? Av1DequantTables.DcQuantQtx(qindex, quant.UDcDeltaQ, bitDepth)
                : Av1DequantTables.DcQuantQtx(qindex, quant.VDcDeltaQ, bitDepth));
        short qAc = plane == 0
            ? Av1DequantTables.AcQuantQtx(qindex, 0, bitDepth)
            : (plane == 1
                ? Av1DequantTables.AcQuantQtx(qindex, quant.UAcDeltaQ, bitDepth)
                : Av1DequantTables.AcQuantQtx(qindex, quant.VAcDeltaQ, bitDepth));

        int shift = Av1TxbCommon.GetTxScale(txSize);
        int maxValue = (1 << (7 + bitDepth)) - 1;
        int minValue = -(1 << (7 + bitDepth));

        for (int c = 0; c < eob; c++)
        {
            int pos = scan[c];
            int padIdx = Av1TxbCommon.GetPaddedIdx(pos, bhl);
            int level = levelsBuf[levelsOff + padIdx];
            if (level == 0) continue;

            int sign;
            if (c == 0)
            {
                var dcSignCdf = Av1DefaultCoefCdfs.DefaultDcSignCdf[qctx][planeType][dcSignCtx];
                sign = rd.DecodeCdfQ15(dcSignCdf, 2);
            }
            else
            {
                sign = (int)rd.DecodeBits(1);
            }

            if (level >= Av1TxbCommon.MaxBaseBrRange)
            {
                level += ReadGolomb(rd);
            }

            if (c == 0) dcVal = sign != 0 ? -level : level;
            culLevel += level;

            short dq = c == 0 ? qDc : qAc;
            long dqCoeff = ((long)level * dq) & 0xFFFFFF;
            int dqInt = (int)dqCoeff;
            dqInt = dqInt >> shift;
            if (sign != 0) dqInt = -dqInt;
            // Clamp to valid range.
            if (dqInt > maxValue) dqInt = maxValue;
            if (dqInt < minValue) dqInt = minValue;
            // Write into the raster output. The scan array uses the libaom
            // bhl-stride convention: pos = col * (1 << bhl) + row. Convert
            // to row-major (pos_raster = row * outW + col). Decoded coefs
            // only land in the top-left (width x height) corner; rest stays 0.
            int col = pos >> bhl;
            int row = pos - (col << bhl);
            if (col < outW && row < outH)
            {
                result.DqCoeffs[row * outW + col] = dqInt;
            }
        }

        culLevel = Math.Min(Av1TxbCommon.CoeffContextMask, culLevel);
        culLevel = Av1TxbCommon.SetDcSign(culLevel, dcVal);
        result.CulLevel = culLevel;
        return result;
    }

    /// <summary>libaom <c>read_golomb</c>: unconstrained Golomb-Rice for high magnitudes.</summary>
    private static int ReadGolomb(Av1RangeDecoder rd)
    {
        int x = 1;
        int length = 0;
        int i = 0;
        while (i == 0)
        {
            i = (int)rd.DecodeBits(1);
            length++;
            if (length > 20) throw new InvalidDataException("AV1 read_golomb length overflow");
        }
        for (i = 0; i < length - 1; i++)
        {
            x <<= 1;
            x += (int)rd.DecodeBits(1);
        }
        return x - 1;
    }

    /// <summary>
    /// libaom EOB multi-CDF reader. Picks the appropriate per-tx-size CDF
    /// (cdf16 / cdf32 / .. / cdf1024) and adds 1 to convert from 0-based to
    /// the 1-based eob_pt expected by RecEobPos.
    /// </summary>
    private static int ReadEobMulti(Av1RangeDecoder rd, Av1TxSize txSize, int planeType, int txClass, int qctx)
    {
        int eobMultiCtx = (txClass == Av1TxbCommon.TxClass2d) ? 0 : 1;
        int eobMultiSize = Av1TxbCommon.TxSizeLog2Minus4[(int)txSize];
        switch (eobMultiSize)
        {
            case 0:
                return rd.DecodeCdfQ15(
                    Av1DefaultCoefCdfs.DefaultEobMulti16Cdf[qctx][planeType][eobMultiCtx], 5) + 1;
            case 1:
                return rd.DecodeCdfQ15(
                    Av1DefaultCoefCdfs.DefaultEobMulti32Cdf[qctx][planeType][eobMultiCtx], 6) + 1;
            case 2:
                return rd.DecodeCdfQ15(
                    Av1DefaultCoefCdfs.DefaultEobMulti64Cdf[qctx][planeType][eobMultiCtx], 7) + 1;
            case 3:
                return rd.DecodeCdfQ15(
                    Av1DefaultCoefCdfs.DefaultEobMulti128Cdf[qctx][planeType][eobMultiCtx], 8) + 1;
            case 4:
                return rd.DecodeCdfQ15(
                    Av1DefaultCoefCdfs.DefaultEobMulti256Cdf[qctx][planeType][eobMultiCtx], 9) + 1;
            case 5:
                return rd.DecodeCdfQ15(
                    Av1DefaultCoefCdfs.DefaultEobMulti512Cdf[qctx][planeType][eobMultiCtx], 10) + 1;
            default:
                return rd.DecodeCdfQ15(
                    Av1DefaultCoefCdfs.DefaultEobMulti1024Cdf[qctx][planeType][eobMultiCtx], 11) + 1;
        }
    }
}
