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

        // Step 1: txb_skip - one bit per tx block.
        var skipCdf = Av1DefaultCoefCdfs.DefaultTxbSkipCdf[3][txsCtx][txbSkipCtx]; // q-ctx=3 (high quality)
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

        // Step 2: tx_type read (Y plane only). For BBB intra keyframe at low
        // qindex this is mostly DCT_DCT, but the bits MUST be consumed when
        // the spec calls for a tx_type read.
        Av1TxType txType = Av1TxType.DctDct;
        if (forcedTxType.HasValue)
        {
            txType = forcedTxType.Value;
        }
        else if (plane == 0)
        {
            // Per spec sec 5.11.40: tx_type is read when:
            //   - txSize <= TX_32X32 (skipped for 64x64)
            //   - !reducedTxSet OR ext_tx_set_used[0]
            // For now, default to DctDct for unsupported size paths.
            // BBB at qindex=5 mostly uses DctDct anyway.
            txType = Av1TxType.DctDct;
        }
        result.TxType = txType;

        int txClass = Av1TxbCommon.TxTypeToClass[(int)txType];
        var scan = Av1ScanTables.Scan[(int)txSize][(int)txType];

        // Step 3: eob_multi - coarse EOB position class.
        int eobPt = ReadEobMulti(rd, txSize, planeType, txClass);

        // Step 4: eob_extra - raw bits to refine.
        int eobExtra = 0;
        int eobOffsetBits = Av1TxbCommon.EobOffsetBits[eobPt - 1];
        if (eobOffsetBits > 0)
        {
            int eobCtx = eobPt - 3;
            var extraCdf = Av1DefaultCoefCdfs.DefaultEobExtraCdf[3][txsCtx][planeType][eobCtx];
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
            var baseEobCdf = Av1DefaultCoefCdfs.DefaultCoeffBaseEobMultiCdf[3][txsCtx][planeType][coefCtx];
            int level = rd.DecodeCdfQ15(baseEobCdf, 3) + 1;
            if (level > Av1TxbCommon.NumBaseLevels)
            {
                int brCtx = Av1TxbCommon.GetBrCtxEob(pos, bhl, txClass);
                var brCdf = Av1DefaultCoefCdfs.DefaultCoeffLpsMultiCdf[3][Math.Min(txsCtx, 3)][planeType][brCtx];
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

                var baseCdf = Av1DefaultCoefCdfs.DefaultCoeffBaseMultiCdf[3][txsCtx][planeType][coefCtx];
                int level = rd.DecodeCdfQ15(baseCdf, 4);
                if (level > Av1TxbCommon.NumBaseLevels)
                {
                    int brCtx = Av1TxbCommon.GetBrCtx(levelsBuf, pos, bhl, txClass);
                    var brCdf = Av1DefaultCoefCdfs.DefaultCoeffLpsMultiCdf[3][Math.Min(txsCtx, 3)][planeType][brCtx];
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
                var dcSignCdf = Av1DefaultCoefCdfs.DefaultDcSignCdf[3][planeType][dcSignCtx];
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
    private static int ReadEobMulti(Av1RangeDecoder rd, Av1TxSize txSize, int planeType, int txClass)
    {
        int eobMultiCtx = (txClass == Av1TxbCommon.TxClass2d) ? 0 : 1;
        int eobMultiSize = Av1TxbCommon.TxSizeLog2Minus4[(int)txSize];
        // Use q-ctx 3 (high quality).
        switch (eobMultiSize)
        {
            case 0:
                return rd.DecodeCdfQ15(
                    Av1DefaultCoefCdfs.DefaultEobMulti16Cdf[3][planeType][eobMultiCtx], 5) + 1;
            case 1:
                return rd.DecodeCdfQ15(
                    Av1DefaultCoefCdfs.DefaultEobMulti32Cdf[3][planeType][eobMultiCtx], 6) + 1;
            case 2:
                return rd.DecodeCdfQ15(
                    Av1DefaultCoefCdfs.DefaultEobMulti64Cdf[3][planeType][eobMultiCtx], 7) + 1;
            case 3:
                return rd.DecodeCdfQ15(
                    Av1DefaultCoefCdfs.DefaultEobMulti128Cdf[3][planeType][eobMultiCtx], 8) + 1;
            case 4:
                return rd.DecodeCdfQ15(
                    Av1DefaultCoefCdfs.DefaultEobMulti256Cdf[3][planeType][eobMultiCtx], 9) + 1;
            case 5:
                return rd.DecodeCdfQ15(
                    Av1DefaultCoefCdfs.DefaultEobMulti512Cdf[3][planeType][eobMultiCtx], 10) + 1;
            default:
                return rd.DecodeCdfQ15(
                    Av1DefaultCoefCdfs.DefaultEobMulti1024Cdf[3][planeType][eobMultiCtx], 11) + 1;
        }
    }
}
