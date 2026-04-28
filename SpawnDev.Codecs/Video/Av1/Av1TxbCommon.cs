// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 transform-block context helpers. Bit-exact port of libaom
// av1/common/txb_common.{h,c} inline functions + lookup tables.
//
// Upstream Copyright (c) 2017, Alliance for Open Media. All rights reserved.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.
// Upstream source: aomedia.googlesource.com/aom
//   av1/common/txb_common.c       (av1_nz_map_ctx_offset_*, av1_eob_*)
//   av1/common/txb_common.h       (get_lower_levels_ctx_*, get_br_ctx, helpers)
//   av1/common/entropy.h          (TX_CLASS, NUM_BASE_LEVELS, COEFF_BASE_RANGE)
//
// These constants and helpers are used by Av1CoefDecoder to read transform
// coefficients from the entropy stream. The padded-grid helpers here use the
// same TX_PAD_* layout libaom uses for the SIMD-friendly "levels" buffer that
// caches decoded coefficient magnitudes during a tx-block read.
//
// Spec reference: AV1 Bitstream and Decoding Process Specification
//   sec 5.11.39 Coefficients syntax
//   sec 9.4.2  Initialization process for the coefficient decoder

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 transform-block (TXB) context helpers - constants, lookup tables, inline helpers.</summary>
internal static class Av1TxbCommon
{
    /// <summary>libaom NUM_BASE_LEVELS = 2.</summary>
    public const int NumBaseLevels = 2;
    /// <summary>libaom BR_CDF_SIZE = 4.</summary>
    public const int BrCdfSize = 4;
    /// <summary>libaom COEFF_BASE_RANGE = 4 * (BR_CDF_SIZE - 1) = 12.</summary>
    public const int CoeffBaseRange = 4 * (BrCdfSize - 1);
    /// <summary>libaom MAX_BASE_BR_RANGE = COEFF_BASE_RANGE + NUM_BASE_LEVELS + 1 = 15.</summary>
    public const int MaxBaseBrRange = CoeffBaseRange + NumBaseLevels + 1;
    /// <summary>libaom COEFF_CONTEXT_BITS = 6 (per entropy.h).</summary>
    public const int CoeffContextBits = 6;
    /// <summary>libaom COEFF_CONTEXT_MASK = (1 less than (1 shifted by COEFF_CONTEXT_BITS)) = 63.</summary>
    public const int CoeffContextMask = (1 << CoeffContextBits) - 1;
    /// <summary>libaom DC_SIGN_CONTEXTS = 3.</summary>
    public const int DcSignContexts = 3;
    /// <summary>libaom TXB_SKIP_CONTEXTS = 13.</summary>
    public const int TxbSkipContexts = 13;
    /// <summary>libaom SIG_COEF_CONTEXTS_2D = 26.</summary>
    public const int SigCoefContexts2d = 26;
    /// <summary>libaom SIG_COEF_CONTEXTS_1D = 16.</summary>
    public const int SigCoefContexts1d = 16;
    /// <summary>libaom SIG_COEF_CONTEXTS_EOB = 4.</summary>
    public const int SigCoefContextsEob = 4;
    /// <summary>libaom LEVEL_CONTEXTS = 21.</summary>
    public const int LevelContexts = 21;
    /// <summary>libaom EOB_COEF_CONTEXTS = 9.</summary>
    public const int EobCoefContexts = 9;

    /// <summary>libaom TX_PAD_HOR_LOG2 = 2.</summary>
    public const int TxPadHorLog2 = 2;
    /// <summary>libaom TX_PAD_HOR = 4.</summary>
    public const int TxPadHor = 4;
    /// <summary>libaom TX_PAD_TOP = 2.</summary>
    public const int TxPadTop = 2;
    /// <summary>libaom TX_PAD_BOTTOM = 4.</summary>
    public const int TxPadBottom = 4;
    /// <summary>libaom TX_PAD_VER = 6.</summary>
    public const int TxPadVer = TxPadTop + TxPadBottom;
    /// <summary>libaom TX_PAD_END = 16.</summary>
    public const int TxPadEnd = 16;
    /// <summary>libaom TX_PAD_2D = (32 + 4) * (32 + 6) + 16 = 1384.</summary>
    public const int TxPad2D = (32 + TxPadHor) * (32 + TxPadVer) + TxPadEnd;

    /// <summary>libaom <c>av1_eob_group_start[12]</c>.</summary>
    public static readonly short[] EobGroupStart = new short[]
    {
        0, 1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513,
    };

    /// <summary>libaom <c>av1_eob_offset_bits[12]</c>.</summary>
    public static readonly short[] EobOffsetBits = new short[]
    {
        0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    };

    /// <summary>
    /// libaom <c>txsize_log2_minus4[TX_SIZES_ALL]</c> from av1/common/common_data.h.
    /// Selects the eob_multi CDF (0=cdf16, 1=cdf32, 2=cdf64, 3=cdf128,
    /// 4=cdf256, 5=cdf512, 6=cdf1024). Computed as log2(num pixels) - 4
    /// where the pixel count is capped at 1024 (32x32) for the 64x64+ family.
    /// </summary>
    public static readonly int[] TxSizeLog2Minus4 = new int[]
    {
        0, // TX_4X4    (16 px  -> cdf16)
        2, // TX_8X8    (64 px  -> cdf64)
        4, // TX_16X16  (256 px -> cdf256)
        6, // TX_32X32  (1024 px-> cdf1024)
        6, // TX_64X64  (capped to 1024 for entropy-coded coefs)
        1, // TX_4X8    (32 px  -> cdf32)
        1, // TX_8X4    (32 px)
        3, // TX_8X16   (128 px -> cdf128)
        3, // TX_16X8   (128 px)
        5, // TX_16X32  (512 px -> cdf512)
        5, // TX_32X16  (512 px)
        6, // TX_32X64  (capped 1024)
        6, // TX_64X32  (capped 1024)
        2, // TX_4X16   (64 px  -> cdf64)
        2, // TX_16X4   (64 px)
        4, // TX_8X32   (256 px -> cdf256)
        4, // TX_32X8   (256 px)
        5, // TX_16X64  (capped 512)
        5, // TX_64X16  (capped 512)
    };

    /// <summary>
    /// Returns the txsize_entropy_ctx for a TX_SIZE. Mirrors libaom
    /// <c>get_txsize_entropy_ctx()</c>: square equivalent capped at TX_32X32.
    /// </summary>
    public static int GetTxSizeEntropyCtx(Av1TxSize txSize)
    {
        // libaom: (av1_get_tx_size(plane, xd) -> txsize_sqr_map[tx_size]) min TX_32X32.
        // For square sizes returns the size itself (clipped). For rect, uses sqr_map.
        int s = (int)txSize switch
        {
            0 => 0,  // 4x4
            1 => 1,  // 8x8
            2 => 2,  // 16x16
            3 => 3,  // 32x32
            4 => 3,  // 64x64 -> 32x32 entropy ctx
            5 => 1,  // 4x8 -> 8x8
            6 => 1,  // 8x4 -> 8x8
            7 => 2,  // 8x16
            8 => 2,  // 16x8
            9 => 3,  // 16x32
            10 => 3, // 32x16
            11 => 3, // 32x64 -> 32x32
            12 => 3, // 64x32 -> 32x32
            13 => 2, // 4x16 -> 16x16
            14 => 2, // 16x4
            15 => 3, // 8x32 -> 32x32
            16 => 3, // 32x8
            17 => 3, // 16x64 -> 32x32
            18 => 3, // 64x16 -> 32x32
            _ => throw new ArgumentOutOfRangeException(nameof(txSize)),
        };
        return s;
    }

    /// <summary>
    /// Returns the av1_get_tx_scale() shift value: 1 for 64x64 / 32x64 / 64x32
    /// / 16x64 / 64x16, 0 otherwise. Mirrors libaom <c>av1_get_tx_scale()</c>.
    /// </summary>
    public static int GetTxScale(Av1TxSize txSize)
    {
        int w = Av1TxSizeInfo.TxWide[(int)txSize];
        int h = Av1TxSizeInfo.TxHigh[(int)txSize];
        if (w == 64 || h == 64) return 1;
        return 0;
    }

    /// <summary>libaom <c>get_txb_bhl(tx_size)</c> = txsize_high_log2[tx_size] (clipped at 5).</summary>
    public static int GetTxbBhl(Av1TxSize txSize)
    {
        int h = Av1TxSizeInfo.TxHigh[(int)txSize];
        // 64-tall transforms only carry 32 rows of coefs.
        h = Math.Min(h, 32);
        return Log2(h);
    }

    /// <summary>libaom <c>get_txb_wide(tx_size)</c> = clamp(txsize_wide, .., 32).</summary>
    public static int GetTxbWide(Av1TxSize txSize)
    {
        return Math.Min(Av1TxSizeInfo.TxWide[(int)txSize], 32);
    }

    /// <summary>libaom <c>get_txb_high(tx_size)</c> = clamp(txsize_high, .., 32).</summary>
    public static int GetTxbHigh(Av1TxSize txSize)
    {
        return Math.Min(Av1TxSizeInfo.TxHigh[(int)txSize], 32);
    }

    /// <summary>libaom <c>get_padded_idx(idx, bhl)</c>.</summary>
    public static int GetPaddedIdx(int idx, int bhl)
    {
        return idx + ((idx >> bhl) << TxPadHorLog2);
    }

    /// <summary>libaom <c>set_levels(levels_buf, height)</c>.</summary>
    public static int SetLevelsOffset(int height)
    {
        return TxPadTop * (height + TxPadHor);
    }

    /// <summary>
    /// libaom <c>get_lower_levels_ctx_eob(bhl, width, scan_idx)</c>.
    /// Returns the small-context bucket for the EOB position itself.
    /// </summary>
    public static int GetLowerLevelsCtxEob(int bhl, int width, int scanIdx)
    {
        if (scanIdx == 0) return 0;
        if (scanIdx <= (width << bhl) / 8) return 1;
        if (scanIdx <= (width << bhl) / 4) return 2;
        return 3;
    }

    /// <summary>
    /// libaom <c>get_br_ctx_eob(c, bhl, tx_class)</c>. EOB-position Br context.
    /// </summary>
    public static int GetBrCtxEob(int c, int bhl, int txClass)
    {
        int col = c >> bhl;
        int row = c - (col << bhl);
        if (c == 0) return 0;
        if ((txClass == TxClass2d && row < 2 && col < 2)
            || (txClass == TxClassHoriz && col == 0)
            || (txClass == TxClassVert && row == 0))
            return 7;
        return 14;
    }

    /// <summary>libaom TX_CLASS enum: TX_CLASS_2D = 0, TX_CLASS_HORIZ = 1, TX_CLASS_VERT = 2.</summary>
    public const int TxClass2d = 0;
    public const int TxClassHoriz = 1;
    public const int TxClassVert = 2;

    /// <summary>libaom <c>tx_type_to_class[TX_TYPES]</c>.</summary>
    public static readonly int[] TxTypeToClass = new int[]
    {
        TxClass2d,    // DCT_DCT
        TxClass2d,    // ADST_DCT
        TxClass2d,    // DCT_ADST
        TxClass2d,    // ADST_ADST
        TxClass2d,    // FLIPADST_DCT
        TxClass2d,    // DCT_FLIPADST
        TxClass2d,    // FLIPADST_FLIPADST
        TxClass2d,    // ADST_FLIPADST
        TxClass2d,    // FLIPADST_ADST
        TxClass2d,    // IDTX
        TxClassVert,  // V_DCT
        TxClassHoriz, // H_DCT
        TxClassVert,  // V_ADST
        TxClassHoriz, // H_ADST
        TxClassVert,  // V_FLIPADST
        TxClassHoriz, // H_FLIPADST
    };

    /// <summary>libaom <c>clip_max3</c> table: min(x, 3) for x in 0..255.</summary>
    public static readonly byte[] ClipMax3 = BuildClipMax3();

    private static byte[] BuildClipMax3()
    {
        var t = new byte[256];
        for (int i = 0; i < 256; i++) t[i] = (byte)Math.Min(i, 3);
        return t;
    }

    /// <summary>
    /// libaom <c>get_nz_mag(levels, bhl, tx_class)</c>. Computes the 5-tap
    /// neighbor magnitude sum used to bucket the level CDF context.
    /// </summary>
    public static int GetNzMag(byte[] levels, int basePadIdx, int bhl, int txClass)
    {
        int stride = (1 << bhl) + TxPadHor;
        int mag = ClipMax3[levels[basePadIdx + stride]];
        mag += ClipMax3[levels[basePadIdx + 1]];
        if (txClass == TxClass2d)
        {
            mag += ClipMax3[levels[basePadIdx + stride + 1]];
            mag += ClipMax3[levels[basePadIdx + (2 * stride)]];
            mag += ClipMax3[levels[basePadIdx + 2]];
        }
        else if (txClass == TxClassVert)
        {
            mag += ClipMax3[levels[basePadIdx + 2]];
            mag += ClipMax3[levels[basePadIdx + 3]];
            mag += ClipMax3[levels[basePadIdx + 4]];
        }
        else // TxClassHoriz
        {
            mag += ClipMax3[levels[basePadIdx + (2 * stride)]];
            mag += ClipMax3[levels[basePadIdx + (3 * stride)]];
            mag += ClipMax3[levels[basePadIdx + (4 * stride)]];
        }
        return mag;
    }

    /// <summary>
    /// libaom <c>get_nz_map_ctx_from_stats(stats, coeff_idx, bhl, tx_size, tx_class)</c>.
    /// Combines the 5-tap mag stats with positional offsets to produce the
    /// CDF row index used by coeff_base_multi.
    /// </summary>
    public static int GetNzMapCtxFromStats(int stats, int coeffIdx, int bhl,
        Av1TxSize txSize, int txClass)
    {
        int ctx;
        if (txClass == TxClass2d)
        {
            ctx = Math.Min((stats + 1) >> 1, 4);
            return ctx + Av1ScanTables.NzMapCtxOffset[(int)txSize][coeffIdx];
        }
        // 1D classes (HORIZ / VERT) use a positional offset relative to row/col.
        int col = coeffIdx >> bhl;
        int row = coeffIdx - (col << bhl);
        int posOffset;
        if (txClass == TxClassHoriz)
        {
            posOffset = (col == 0) ? 0 : (col == 1 ? 7 : 14);
        }
        else // TxClassVert
        {
            posOffset = (row == 0) ? 0 : (row == 1 ? 7 : 14);
        }
        ctx = Math.Min((stats + 1) >> 1, 4);
        return SigCoefContexts2d + ctx + posOffset;
    }

    /// <summary>libaom <c>get_lower_levels_ctx</c>: full-context 5-tap path used inside the scan loop.</summary>
    public static int GetLowerLevelsCtx(byte[] levels, int coeffIdx, int bhl,
        Av1TxSize txSize, int txClass)
    {
        int basePadIdx = GetPaddedIdx(coeffIdx, bhl);
        int stats = GetNzMag(levels, basePadIdx, bhl, txClass);
        return GetNzMapCtxFromStats(stats, coeffIdx, bhl, txSize, txClass);
    }

    /// <summary>libaom <c>get_lower_levels_ctx_2d</c>: optimized 5-tap path for TX_CLASS_2D.</summary>
    public static int GetLowerLevelsCtx2d(byte[] levels, int coeffIdx, int bhl, Av1TxSize txSize)
    {
        if (coeffIdx == 0) throw new InvalidOperationException("coeff_idx must be >0 for 2D path");
        int basePadIdx = GetPaddedIdx(coeffIdx, bhl);
        int stride = (1 << bhl) + TxPadHor;
        int mag = Math.Min((int)levels[basePadIdx + stride], 3);
        mag += Math.Min((int)levels[basePadIdx + 1], 3);
        mag += Math.Min((int)levels[basePadIdx + stride + 1], 3);
        mag += Math.Min((int)levels[basePadIdx + (2 * stride)], 3);
        mag += Math.Min((int)levels[basePadIdx + 2], 3);
        int ctx = Math.Min((mag + 1) >> 1, 4);
        return ctx + Av1ScanTables.NzMapCtxOffset[(int)txSize][coeffIdx];
    }

    /// <summary>libaom <c>get_br_ctx(levels, c, bhl, tx_class)</c>: Br-CDF context for high-magnitude coefs.</summary>
    public static int GetBrCtx(byte[] levels, int c, int bhl, int txClass)
    {
        int col = c >> bhl;
        int row = c - (col << bhl);
        int stride = (1 << bhl) + TxPadHor;
        int posIdx = TxPadTop * stride + col * stride + row;
        int mag = levels[posIdx + 1];
        mag += levels[posIdx + stride];
        switch (txClass)
        {
            case TxClass2d:
                mag += levels[posIdx + stride + 1];
                mag = Math.Min((mag + 1) >> 1, 6);
                if (c == 0) return mag;
                if (row < 2 && col < 2) return mag + 7;
                break;
            case TxClassHoriz:
                mag += levels[posIdx + (stride << 1)];
                mag = Math.Min((mag + 1) >> 1, 6);
                if (c == 0) return mag;
                if (col == 0) return mag + 7;
                break;
            case TxClassVert:
                mag += levels[posIdx + 2];
                mag = Math.Min((mag + 1) >> 1, 6);
                if (c == 0) return mag;
                if (row == 0) return mag + 7;
                break;
        }
        return mag + 14;
    }

    /// <summary>
    /// libaom <c>av1_get_skip_txfm_context</c>: above + left skip count.
    /// </summary>
    public static int GetSkipTxfmContext(bool aboveSkip, bool leftSkip)
    {
        return (aboveSkip ? 1 : 0) + (leftSkip ? 1 : 0);
    }

    /// <summary>
    /// libaom <c>get_dc_sign_context</c>: dc_sign neighbor accumulator -> 0..2.
    /// </summary>
    public static int GetDcSignContext(int dcSignAccum)
    {
        if (dcSignAccum < 0) return 1;
        if (dcSignAccum > 0) return 2;
        return 0;
    }

    /// <summary>libaom <c>set_dc_sign(cul_level, dc_val)</c>: encodes DC sign into the 7th bit of cul_level.</summary>
    public static int SetDcSign(int culLevel, int dcVal)
    {
        if (dcVal < 0) culLevel |= (1 << CoeffContextBits);
        else if (dcVal > 0) culLevel |= (2 << CoeffContextBits);
        return culLevel;
    }

    /// <summary>
    /// libaom <c>rec_eob_pos(eob_pt, eob_extra)</c>: reconstruct exact EOB scan
    /// position from the (size class, low-bits) decomposition. <paramref name="eobPt"/>
    /// is the libaom-convention eob_pt (group index 0..11), matching the
    /// value the decoder computes as (CDF symbol + 1).
    /// </summary>
    public static int RecEobPos(int eobPt, int eobExtra)
    {
        int eob = EobGroupStart[eobPt];
        if (eob > 2)
        {
            eob += eobExtra;
        }
        return eob;
    }

    private static int Log2(int x)
    {
        // Returns floor(log2(x)) for x in {4, 8, 16, 32, 64}.
        return x switch
        {
            1 => 0,
            2 => 1,
            4 => 2,
            8 => 3,
            16 => 4,
            32 => 5,
            64 => 6,
            _ => throw new ArgumentException($"unsupported size {x}", nameof(x)),
        };
    }
}
