// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 coefficient encoder. Bit-exact mirror of <see cref="Av1CoefDecoder"/>;
// structural port of libaom av1/encoder/encodetxb.c
// <c>av1_write_coeffs_txb</c>.
//
// Upstream Copyright (c) 2017, Alliance for Open Media. All rights reserved.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.
//
// Writes the coefficient (token) entropy stream for ONE transform block.
// The bit-flow exactly matches what <see cref="Av1CoefDecoder.ReadCoeffsTxb"/>
// reads back, so encoder + decoder round-trip on the same default CDFs and
// the same per-block context inputs (txbSkipCtx + dcSignCtx).
//
// Steps (mirrors libaom + the decoder for a single tx block):
//   1. txb_skip       (1 bin) - emits the all-zero shortcut.
//   2. tx_type        (per intra_ext_tx CDF; only when set has &gt; 1 type)
//   3. eob_pt         (4..11-sym EOB classification)
//   4. eob_extra      (1 CDF bin + raw bits to refine).
//   5. coeff_base_eob (3-sym level for EOB position)
//   6. coeff_base     (4-sym levels for non-EOB scan positions, REVERSE order)
//   7. coeff_lps      (Br increments for level &gt;= NUM_BASE_LEVELS)
//   8. dc_sign + per-AC sign bits + Golomb tail for level &gt; MAX_BASE_BR_RANGE.
//
// The coefficient input is in raster (row-major: coeffs[row * w + col]). We
// internally reshape into the libaom "levels[]" padded buffer using the
// (col*height + row) layout the encoder uses, mirroring
// <c>av1_txb_init_levels_c</c>.

using SpawnDev.Codecs.EntropyCoders;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 coefficient (token) encoder - mirror of <see cref="Av1CoefDecoder"/>.</summary>
internal static class Av1CoefEncoder
{
    /// <summary>
    /// Result of encoding one transform block's coefficients - for caller
    /// neighbor-context bookkeeping (mirrors the CulLevel + Eob the decoder
    /// returns so the caller can update <see cref="Av1EntropyContext"/> in
    /// the same way the decoder does).
    /// </summary>
    public sealed class EncodedCoefBlock
    {
        /// <summary>End-of-block scan index (exclusive); 0 means all-zero block.</summary>
        public int Eob;
        /// <summary>Cumulative absolute coef sum capped at COEFF_CONTEXT_MASK,
        /// with DC sign packed into the top bits per libaom <c>set_dc_sign</c>.</summary>
        public int CulLevel;
    }

    /// <summary>
    /// Write one transform block's coefficients. Coeffs are quantized integer
    /// values in raster (row-major) layout; the encoder converts internally to
    /// the libaom (col*height + row) layout used by <c>av1_txb_init_levels</c>.
    /// </summary>
    /// <param name="re">Per-tile entropy encoder (state mutated as bits emit).</param>
    /// <param name="txSize">TX size of this block.</param>
    /// <param name="plane">0 = Y, 1 = U, 2 = V.</param>
    /// <param name="intraMode">Y intra prediction mode (drives intra_ext_tx CDF row).</param>
    /// <param name="reducedTxSet">reduced_tx_set_used flag from frame header.</param>
    /// <param name="txbSkipCtx">txb_skip CDF context (caller-tracked).</param>
    /// <param name="dcSignCtx">dc_sign CDF context (caller-tracked).</param>
    /// <param name="coeffsRaster">
    /// Quantized coefficients in raster (row-major) order; length must equal
    /// outW * outH where outW/outH are <see cref="Av1TxSizeInfo.TxWide"/> /
    /// <see cref="Av1TxSizeInfo.TxHigh"/>. Coefs outside the entropy-coded
    /// (max 32x32) corner are ignored.
    /// </param>
    /// <param name="txType">Transform type to encode (driver chooses; for v1 always DCT_DCT).</param>
    public static EncodedCoefBlock WriteCoeffsTxb(
        Av1RangeEncoder re,
        Av1TxSize txSize,
        int plane,
        Av1IntraMode intraMode,
        bool reducedTxSet,
        int txbSkipCtx,
        int dcSignCtx,
        ReadOnlySpan<int> coeffsRaster,
        int qindex,
        Av1TxType txType = Av1TxType.DctDct)
    {
        ArgumentNullException.ThrowIfNull(re);

        int planeType = plane == 0 ? 0 : 1;
        int txsCtx = Av1TxbCommon.GetTxSizeEntropyCtx(txSize);
        int width = Av1TxbCommon.GetTxbWide(txSize);
        int height = Av1TxbCommon.GetTxbHigh(txSize);
        int bhl = Av1TxbCommon.GetTxbBhl(txSize);
        int outW = Av1TxSizeInfo.TxWide[(int)txSize];
        int outH = Av1TxSizeInfo.TxHigh[(int)txSize];
        if (coeffsRaster.Length < outW * outH)
            throw new ArgumentException(
                $"coeffsRaster too short: need {outW * outH}, got {coeffsRaster.Length}",
                nameof(coeffsRaster));

        // qctx: token CDFs are quantized by the per-frame qindex (libaom get_q_ctx).
        int qctx = Av1CoefDecoder.GetQctx(qindex);

        // Compute EOB by scanning entropy-coded corner in scan order.
        int klassDct = Av1TxbCommon.TxTypeToClass[(int)txType];
        var scan = Av1ScanTables.Scan[(int)txSize][(int)txType];
        int eob = ComputeEob(coeffsRaster, scan, bhl, outW, outH, width, height);

        var result = new EncodedCoefBlock { Eob = eob, CulLevel = 0 };

        // Step 1: txb_skip - one bin per tx block. Emit 1 if all-zero.
        var skipCdf = Av1DefaultCoefCdfs.DefaultTxbSkipCdf[qctx][txsCtx][txbSkipCtx];
        int allZero = eob == 0 ? 1 : 0;
        re.EncodeCdfQ15(allZero, skipCdf, 2);
        if (eob == 0) return result;

        // Step 2: tx_type for Y plane only; only when ext tx set has &gt; 1 type.
        if (plane == 0 && qindex > 0)
        {
            int extTxSetType = GetExtTxSetType(txSize, reducedTxSet);
            int numSet = ExtTxNumSet[extTxSetType];
            if (numSet > 1)
            {
                int squareTxSize = SquareTxSize(txSize);
                int eset = ExtTxSetIndexIntra[extTxSetType];
                var cdf = Av1DefaultTxfmCdfs.DefaultIntraExtTxCdf[eset][squareTxSize][(int)intraMode];
                int sym = ExtTxInd[extTxSetType, (int)txType];
                if (sym < 0 || sym >= numSet)
                    throw new InvalidOperationException(
                        $"tx_type {txType} not allowed in ext tx set {extTxSetType}.");
                re.EncodeCdfQ15(sym, cdf, numSet);
            }
        }

        int txClass = Av1TxbCommon.TxTypeToClass[(int)txType];

        // Step 3: eob_pt classification (libaom-token, 0..11). Symbol written
        // is (eobPt - 1) to match libaom's CDF indexing.
        int eobPt = GetEobPosToken(eob, out int eobExtra);
        int eobMultiSize = Av1TxbCommon.TxSizeLog2Minus4[(int)txSize];
        int eobMultiCtx = (txClass == Av1TxbCommon.TxClass2d) ? 0 : 1;
        WriteEobMulti(re, eobMultiSize, planeType, eobMultiCtx, eobPt, qctx);

        // Step 4: eob_extra - first bit via CDF, rest as raw bits. libaom
        // uses av1_eob_offset_bits[eob_pt] directly (the libaom convention
        // where eob_pt is the group index, NOT eob_pt - 1).
        int eobOffsetBits = Av1TxbCommon.EobOffsetBits[eobPt];
        if (eobOffsetBits > 0)
        {
            int eobCtx = eobPt - 3;
            var extraCdf = Av1DefaultCoefCdfs.DefaultEobExtraCdf[qctx][txsCtx][planeType][eobCtx];
            int shift = eobOffsetBits - 1;
            int firstBit = ((eobExtra >> shift) & 1);
            re.EncodeCdfQ15(firstBit, extraCdf, 2);
            for (int i = 1; i < eobOffsetBits; i++)
            {
                shift = eobOffsetBits - 1 - i;
                int b = ((eobExtra >> shift) & 1);
                re.EncodeBits((uint)b, 1);
            }
        }

        // Build the libaom-layout levels[] padded buffer (col*height + row).
        int paddedStride = (1 << bhl) + Av1TxbCommon.TxPadHor;
        int paddedRows = height + Av1TxbCommon.TxPadVer;
        var levelsBuf = new byte[paddedStride * paddedRows + Av1TxbCommon.TxPadEnd];
        int levelsOff = Av1TxbCommon.SetLevelsOffset(height);
        InitLevels(coeffsRaster, outW, outH, width, height, bhl, levelsBuf, levelsOff);

        // Step 5+6: write base CDFs in REVERSE scan order (eob-1 .. 0).
        // c == eob-1 uses coeff_base_eob (3 syms). Others use coeff_base (4 syms).
        for (int c = eob - 1; c >= 0; c--)
        {
            int pos = scan[c];
            int padIdx = Av1TxbCommon.GetPaddedIdx(pos, bhl);
            int level = levelsBuf[levelsOff + padIdx];

            int coefCtx;
            if (c == eob - 1)
            {
                coefCtx = Av1TxbCommon.GetLowerLevelsCtxEob(bhl, width, c);
                var baseEobCdf = Av1DefaultCoefCdfs.DefaultCoeffBaseEobMultiCdf[qctx][txsCtx][planeType][coefCtx];
                int sym = Math.Min(level, 3) - 1;
                re.EncodeCdfQ15(sym, baseEobCdf, 3);
            }
            else
            {
                coefCtx = (txClass == Av1TxbCommon.TxClass2d
                    ? (c == 0
                        ? Av1TxbCommon.GetLowerLevelsCtxEob(bhl, width, 0)
                        : Av1TxbCommon.GetLowerLevelsCtx2d(levelsBuf, pos, bhl, txSize))
                    : Av1TxbCommon.GetLowerLevelsCtx(levelsBuf, pos, bhl, txSize, txClass));
                var baseCdf = Av1DefaultCoefCdfs.DefaultCoeffBaseMultiCdf[qctx][txsCtx][planeType][coefCtx];
                int sym = Math.Min(level, 3);
                re.EncodeCdfQ15(sym, baseCdf, 4);
            }

            // Step 7: coeff_lps for level &gt; NUM_BASE_LEVELS.
            if (level > Av1TxbCommon.NumBaseLevels)
            {
                int baseRange = level - 1 - Av1TxbCommon.NumBaseLevels;
                int brCtx = (c == eob - 1)
                    ? Av1TxbCommon.GetBrCtxEob(pos, bhl, txClass)
                    : Av1TxbCommon.GetBrCtx(levelsBuf, pos, bhl, txClass);
                var brCdf = Av1DefaultCoefCdfs.DefaultCoeffLpsMultiCdf[qctx][Math.Min(txsCtx, 3)][planeType][brCtx];
                for (int idx = 0; idx < Av1TxbCommon.CoeffBaseRange; idx += Av1TxbCommon.BrCdfSize - 1)
                {
                    int k = Math.Min(baseRange - idx, Av1TxbCommon.BrCdfSize - 1);
                    re.EncodeCdfQ15(k, brCdf, Av1TxbCommon.BrCdfSize);
                    if (k < Av1TxbCommon.BrCdfSize - 1) break;
                }
            }
        }

        // Step 8: signs + golomb tails. Walk scan FORWARD (0..eob-1).
        int culLevel = 0;
        int dcVal = 0;
        for (int c = 0; c < eob; c++)
        {
            int pos = scan[c];
            int padIdx = Av1TxbCommon.GetPaddedIdx(pos, bhl);
            int level = levelsBuf[levelsOff + padIdx];
            if (level == 0) continue;

            int signedVal = ReadCoeff(coeffsRaster, pos, bhl, outW, outH);
            int sign = signedVal < 0 ? 1 : 0;

            if (c == 0)
            {
                var dcSignCdf = Av1DefaultCoefCdfs.DefaultDcSignCdf[qctx][planeType][dcSignCtx];
                re.EncodeCdfQ15(sign, dcSignCdf, 2);
            }
            else
            {
                re.EncodeBits((uint)sign, 1);
            }

            // Recover the FULL absolute level (clipped to byte in levelsBuf).
            int absLevel = signedVal < 0 ? -signedVal : signedVal;
            if (absLevel > Av1TxbCommon.CoeffBaseRange + Av1TxbCommon.NumBaseLevels)
            {
                int golombVal = absLevel - Av1TxbCommon.CoeffBaseRange - 1 - Av1TxbCommon.NumBaseLevels;
                WriteGolomb(re, golombVal);
            }

            if (c == 0) dcVal = signedVal;
            culLevel += absLevel;
        }

        culLevel = Math.Min(Av1TxbCommon.CoeffContextMask, culLevel);
        culLevel = Av1TxbCommon.SetDcSign(culLevel, dcVal);
        result.CulLevel = culLevel;
        return result;
    }

    /// <summary>
    /// Compute EOB (exclusive scan index of last non-zero) by walking the
    /// scan order. Returns 0 if all coefficients in the entropy-coded corner
    /// are zero.
    /// </summary>
    private static int ComputeEob(ReadOnlySpan<int> coeffsRaster, short[] scan,
        int bhl, int outW, int outH, int width, int height)
    {
        int n = scan.Length;
        for (int c = n - 1; c >= 0; c--)
        {
            int pos = scan[c];
            int v = ReadCoeff(coeffsRaster, pos, bhl, outW, outH);
            if (v != 0) return c + 1;
        }
        return 0;
    }

    /// <summary>
    /// Read coeffsRaster at scan-position <paramref name="pos"/>. <paramref name="pos"/>
    /// uses libaom's bhl-stride layout: <c>pos = col * (1 &lt;&lt; bhl) + row</c>.
    /// Coeffs outside the entropy-coded corner are zero.
    /// </summary>
    private static int ReadCoeff(ReadOnlySpan<int> coeffsRaster, int pos, int bhl, int outW, int outH)
    {
        int col = pos >> bhl;
        int row = pos - (col << bhl);
        if (col >= outW || row >= outH) return 0;
        return coeffsRaster[row * outW + col];
    }

    /// <summary>
    /// libaom <c>av1_txb_init_levels_c</c>: fill the padded levels[] buffer from
    /// the coefficient block. Coeffs are laid out in (col*height + row) order
    /// in the source buffer libaom passes; we adapt from our raster order.
    /// </summary>
    private static void InitLevels(ReadOnlySpan<int> coeffsRaster, int outW, int outH,
        int width, int height, int bhl, byte[] levels, int levelsOffsetWithinBuf)
    {
        // libaom layout: ls[col * (height + TX_PAD_HOR) + row] = clamp(|coeff[col*height + row]|, 0, 127)
        int stride = (1 << bhl) + Av1TxbCommon.TxPadHor;
        for (int col = 0; col < width; col++)
        {
            for (int row = 0; row < height; row++)
            {
                int v = ReadCoeff(coeffsRaster, col * (1 << bhl) + row, bhl, outW, outH);
                int abs = v < 0 ? -v : v;
                if (abs > sbyte.MaxValue) abs = sbyte.MaxValue;
                levels[levelsOffsetWithinBuf + col * stride + row] = (byte)abs;
            }
            // TX_PAD_HOR zero pad after each column run.
            for (int p = 0; p < Av1TxbCommon.TxPadHor; p++)
                levels[levelsOffsetWithinBuf + col * stride + height + p] = 0;
        }
    }

    /// <summary>
    /// libaom <c>av1_get_eob_pos_token</c>: classify EOB into a group index and
    /// return the (eob - eob_group_start[t]) refinement extra. The returned
    /// value is the LIBAOM EOB token (group index in 0..11). The caller writes
    /// (token - 1) to the eob_flag_cdf and uses [token] for offset_bits /
    /// group_start lookups - matching libaom's convention exactly so the
    /// resulting bytes are the bit-stream a dav1d decoder expects.
    /// </summary>
    private static int GetEobPosToken(int eob, out int extra)
    {
        int t;
        if (eob < 33)
        {
            t = EobToPosSmall[eob];
        }
        else
        {
            int e = Math.Min((eob - 1) >> 5, 16);
            t = EobToPosLarge[e];
        }
        extra = eob - Av1TxbCommon.EobGroupStart[t];
        return t;
    }

    /// <summary>libaom <c>eob_to_pos_small[33]</c>.</summary>
    private static readonly sbyte[] EobToPosSmall = new sbyte[]
    {
        0, 1, 2,                                        // 0-2
        3, 3,                                           // 3-4
        4, 4, 4, 4,                                     // 5-8
        5, 5, 5, 5, 5, 5, 5, 5,                         // 9-16
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, // 17-32
    };

    /// <summary>libaom <c>eob_to_pos_large[17]</c>.</summary>
    private static readonly sbyte[] EobToPosLarge = new sbyte[]
    {
        6,
        7,
        8, 8,
        9, 9, 9, 9,
        10, 10, 10, 10, 10, 10, 10, 10,
        11,
    };

    private static void WriteEobMulti(Av1RangeEncoder re, int eobMultiSize,
        int planeType, int eobMultiCtx, int eobPt, int qctx)
    {
        int sym = eobPt - 1;
        switch (eobMultiSize)
        {
            case 0:
                re.EncodeCdfQ15(sym, Av1DefaultCoefCdfs.DefaultEobMulti16Cdf[qctx][planeType][eobMultiCtx], 5);
                break;
            case 1:
                re.EncodeCdfQ15(sym, Av1DefaultCoefCdfs.DefaultEobMulti32Cdf[qctx][planeType][eobMultiCtx], 6);
                break;
            case 2:
                re.EncodeCdfQ15(sym, Av1DefaultCoefCdfs.DefaultEobMulti64Cdf[qctx][planeType][eobMultiCtx], 7);
                break;
            case 3:
                re.EncodeCdfQ15(sym, Av1DefaultCoefCdfs.DefaultEobMulti128Cdf[qctx][planeType][eobMultiCtx], 8);
                break;
            case 4:
                re.EncodeCdfQ15(sym, Av1DefaultCoefCdfs.DefaultEobMulti256Cdf[qctx][planeType][eobMultiCtx], 9);
                break;
            case 5:
                re.EncodeCdfQ15(sym, Av1DefaultCoefCdfs.DefaultEobMulti512Cdf[qctx][planeType][eobMultiCtx], 10);
                break;
            default:
                re.EncodeCdfQ15(sym, Av1DefaultCoefCdfs.DefaultEobMulti1024Cdf[qctx][planeType][eobMultiCtx], 11);
                break;
        }
    }

    /// <summary>
    /// libaom <c>write_golomb</c>: emit (length-1) leading zero bits, then
    /// the bits of (level + 1) MSB-first - so total length = 2 * floor(log2(level+1)) + 1.
    /// </summary>
    private static void WriteGolomb(Av1RangeEncoder re, int level)
    {
        int x = level + 1;
        int length = 0;
        int i = x;
        while (i != 0) { i >>= 1; length++; }
        if (length <= 0) throw new InvalidOperationException("WriteGolomb: length must be > 0");

        for (i = 0; i < length - 1; i++) re.EncodeBits(0u, 1);
        for (i = length - 1; i >= 0; i--) re.EncodeBits((uint)((x >> i) & 1), 1);
    }

    // ------------------------------------------------------------------
    // libaom ext_tx tables - DUPLICATED from av1/common/blockd.h +
    // av1/common/entropymode.h.
    // ------------------------------------------------------------------

    /// <summary>libaom <c>av1_num_ext_tx_set</c>: per-set active symbol count.</summary>
    private static readonly int[] ExtTxNumSet = { 1, 2, 5, 7, 12, 16 };

    /// <summary>libaom <c>ext_tx_set_index[2][EXT_TX_SET_TYPES]</c> - intra row only.</summary>
    private static readonly int[] ExtTxSetIndexIntra = { 0, -1, 2, 1, -1, -1 };

    /// <summary>libaom <c>av1_ext_tx_ind[EXT_TX_SET_TYPES][TX_TYPES]</c>.</summary>
    private static readonly int[,] ExtTxInd = new int[6, 16]
    {
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 1, 3, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        { 1, 5, 6, 4, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0 },
        { 3, 4, 5, 8, 6, 7, 9, 10, 11, 0, 1, 2, 0, 0, 0, 0 },
        { 7, 8, 9, 12, 10, 11, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6 },
    };

    /// <summary>
    /// libaom <c>av1_get_ext_tx_set_type(tx_size, is_inter, use_reduced_set)</c>
    /// for is_inter = 0 (intra path).
    /// </summary>
    private static int GetExtTxSetType(Av1TxSize txSize, bool reducedTxSet)
    {
        // Square upmap: rect tx sizes use the larger dim for the "sqr_up" check.
        var sqrUp = TxSizeSqrUpMap(txSize);
        // 64x64+ -&gt; DCTONLY (set 0)
        if (TxSizeIsLargerThan32(sqrUp)) return 0; // EXT_TX_SET_DCTONLY
        if (sqrUp == Av1TxSize.Tx32x32) return 0;  // intra at 32x32 -&gt; DCTONLY
        if (reducedTxSet) return 2;                // EXT_TX_SET_DTT4_IDTX
        // Non-reduced intra: set_lookup[0][tx_size_sqr == TX_16X16]
        // intra row: { DTT4_IDTX_1DDCT, DTT4_IDTX } -&gt; { 3, 2 }
        var sqr = TxSizeSqrMap(txSize);
        return sqr == Av1TxSize.Tx16x16 ? 2 : 3;
    }

    /// <summary>libaom <c>txsize_sqr_up_map</c>: the size's bounding square (rounded up).</summary>
    private static Av1TxSize TxSizeSqrUpMap(Av1TxSize txSize) => (int)txSize switch
    {
        0 => Av1TxSize.Tx4x4,
        1 => Av1TxSize.Tx8x8,
        2 => Av1TxSize.Tx16x16,
        3 => Av1TxSize.Tx32x32,
        4 => Av1TxSize.Tx64x64,
        5 => Av1TxSize.Tx8x8,    // 4x8
        6 => Av1TxSize.Tx8x8,    // 8x4
        7 => Av1TxSize.Tx16x16,  // 8x16
        8 => Av1TxSize.Tx16x16,  // 16x8
        9 => Av1TxSize.Tx32x32,  // 16x32
        10 => Av1TxSize.Tx32x32, // 32x16
        11 => Av1TxSize.Tx64x64, // 32x64
        12 => Av1TxSize.Tx64x64, // 64x32
        13 => Av1TxSize.Tx16x16, // 4x16
        14 => Av1TxSize.Tx16x16, // 16x4
        15 => Av1TxSize.Tx32x32, // 8x32
        16 => Av1TxSize.Tx32x32, // 32x8
        17 => Av1TxSize.Tx64x64, // 16x64
        18 => Av1TxSize.Tx64x64, // 64x16
        _ => throw new ArgumentOutOfRangeException(nameof(txSize)),
    };

    /// <summary>libaom <c>txsize_sqr_map</c>: the size's bounding square (rounded down).</summary>
    private static Av1TxSize TxSizeSqrMap(Av1TxSize txSize) => (int)txSize switch
    {
        0 => Av1TxSize.Tx4x4,
        1 => Av1TxSize.Tx8x8,
        2 => Av1TxSize.Tx16x16,
        3 => Av1TxSize.Tx32x32,
        4 => Av1TxSize.Tx64x64,
        5 => Av1TxSize.Tx4x4,
        6 => Av1TxSize.Tx4x4,
        7 => Av1TxSize.Tx8x8,
        8 => Av1TxSize.Tx8x8,
        9 => Av1TxSize.Tx16x16,
        10 => Av1TxSize.Tx16x16,
        11 => Av1TxSize.Tx32x32,
        12 => Av1TxSize.Tx32x32,
        13 => Av1TxSize.Tx4x4,
        14 => Av1TxSize.Tx4x4,
        15 => Av1TxSize.Tx8x8,
        16 => Av1TxSize.Tx8x8,
        17 => Av1TxSize.Tx16x16,
        18 => Av1TxSize.Tx16x16,
        _ => throw new ArgumentOutOfRangeException(nameof(txSize)),
    };

    private static bool TxSizeIsLargerThan32(Av1TxSize ts) =>
        ts == Av1TxSize.Tx64x64;

    /// <summary>libaom <c>txsize_sqr_map[tx_size]</c> for picking <c>square_tx_size</c> in CDF row index.</summary>
    private static int SquareTxSize(Av1TxSize txSize)
    {
        var s = TxSizeSqrMap(txSize);
        // intra_ext_tx_cdf is indexed by EXT_TX_SIZES = 4 (TX_4X4..TX_32X32).
        return s switch
        {
            Av1TxSize.Tx4x4 => 0,
            Av1TxSize.Tx8x8 => 1,
            Av1TxSize.Tx16x16 => 2,
            Av1TxSize.Tx32x32 => 3,
            _ => 3,
        };
    }
}
