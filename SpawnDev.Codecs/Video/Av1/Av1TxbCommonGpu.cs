// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 transform-block (TXB) context helpers, GPU-callable form.
// Bit-exact mirror of Av1TxbCommon.* for in-kernel use by the AV1
// coef encoder + decoder GPU kernels.
//
// All helpers are pure int functions over byte/int views - no
// allocations, no LocalMemory, no exceptions thrown. Designed for
// the v3 host-as-pure-coordinator pattern: the per-block walker
// kernel calls these inline while building per-block context for
// the entropy stage.
//
// Constants are duplicated as `public const int` so kernel code can
// reference them directly without crossing the CPU/GPU boundary.

using ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>GPU-callable AV1 transform-block context helpers.</summary>
public static class Av1TxbCommonGpu
{
    /// <summary>libaom NUM_BASE_LEVELS = 2.</summary>
    public const int NumBaseLevels = Av1TxbCommon.NumBaseLevels;
    /// <summary>libaom BR_CDF_SIZE = 4.</summary>
    public const int BrCdfSize = Av1TxbCommon.BrCdfSize;
    /// <summary>libaom COEFF_BASE_RANGE = 12.</summary>
    public const int CoeffBaseRange = Av1TxbCommon.CoeffBaseRange;
    /// <summary>libaom MAX_BASE_BR_RANGE = COEFF_BASE_RANGE + NUM_BASE_LEVELS + 1 = 15.</summary>
    public const int MaxBaseBrRange = Av1TxbCommon.MaxBaseBrRange;
    /// <summary>libaom COEFF_CONTEXT_BITS = 6.</summary>
    public const int CoeffContextBits = Av1TxbCommon.CoeffContextBits;
    /// <summary>libaom COEFF_CONTEXT_MASK = 63.</summary>
    public const int CoeffContextMask = Av1TxbCommon.CoeffContextMask;
    /// <summary>libaom SIG_COEF_CONTEXTS_2D = 26.</summary>
    public const int SigCoefContexts2d = Av1TxbCommon.SigCoefContexts2d;

    /// <summary>libaom TX_PAD_HOR_LOG2 = 2.</summary>
    public const int TxPadHorLog2 = Av1TxbCommon.TxPadHorLog2;
    /// <summary>libaom TX_PAD_HOR = 4.</summary>
    public const int TxPadHor = Av1TxbCommon.TxPadHor;
    /// <summary>libaom TX_PAD_TOP = 2.</summary>
    public const int TxPadTop = Av1TxbCommon.TxPadTop;
    /// <summary>libaom TX_PAD_VER = 6.</summary>
    public const int TxPadVer = Av1TxbCommon.TxPadVer;
    /// <summary>libaom TX_PAD_END = 16.</summary>
    public const int TxPadEnd = Av1TxbCommon.TxPadEnd;

    /// <summary>libaom TX_CLASS_2D = 0.</summary>
    public const int TxClass2d = Av1TxbCommon.TxClass2d;
    /// <summary>libaom TX_CLASS_HORIZ = 1.</summary>
    public const int TxClassHoriz = Av1TxbCommon.TxClassHoriz;
    /// <summary>libaom TX_CLASS_VERT = 2.</summary>
    public const int TxClassVert = Av1TxbCommon.TxClassVert;

    /// <summary>libaom <c>get_padded_idx(idx, bhl)</c>.</summary>
    public static int GetPaddedIdx(int idx, int bhl)
    {
        return idx + ((idx >> bhl) << TxPadHorLog2);
    }

    /// <summary>libaom <c>set_levels(levels_buf, height)</c> - returns the offset.</summary>
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

    /// <summary>
    /// libaom <c>get_lower_levels_ctx_2d</c> - optimized 5-tap path for
    /// TX_CLASS_2D. Reads from a padded levels buffer (ArrayView&lt;byte&gt;)
    /// starting at <paramref name="levelsBase"/> (the buffer start, before
    /// the TxPadTop row offset). The nzMapCtxOffset table is supplied
    /// flat: <paramref name="nzMapCtxOffset"/> is the table for the
    /// current tx_size, indexed by coeffIdx.
    /// </summary>
    public static int GetLowerLevelsCtx2d(
        ArrayView<byte> levels, long levelsBase,
        int coeffIdx, int bhl,
        ArrayView<byte> nzMapCtxOffset, long nzMapCtxOffsetBase)
    {
        int basePadIdx = GetPaddedIdx(coeffIdx, bhl);
        int stride = (1 << bhl) + TxPadHor;
        long baseIdx = levelsBase + TxPadTop * stride + basePadIdx;
        int mag = ClipMax3(levels[baseIdx + stride]);
        mag += ClipMax3(levels[baseIdx + 1]);
        mag += ClipMax3(levels[baseIdx + stride + 1]);
        mag += ClipMax3(levels[baseIdx + (2 * stride)]);
        mag += ClipMax3(levels[baseIdx + 2]);
        int ctx = (mag + 1) >> 1;
        if (ctx > 4) ctx = 4;
        return ctx + nzMapCtxOffset[nzMapCtxOffsetBase + coeffIdx];
    }

    /// <summary>
    /// libaom <c>get_br_ctx(levels, c, bhl, tx_class)</c>. Reads from the
    /// padded levels buffer (caller passes raw buffer + base + bhl + class).
    /// </summary>
    public static int GetBrCtx(
        ArrayView<byte> levels, long levelsBase,
        int c, int bhl, int txClass)
    {
        int col = c >> bhl;
        int row = c - (col << bhl);
        int stride = (1 << bhl) + TxPadHor;
        long posIdx = levelsBase + TxPadTop * stride + col * stride + row;
        int mag = levels[posIdx + 1];
        mag += levels[posIdx + stride];
        if (txClass == TxClass2d)
        {
            mag += levels[posIdx + stride + 1];
            mag = (mag + 1) >> 1;
            if (mag > 6) mag = 6;
            if (c == 0) return mag;
            if (row < 2 && col < 2) return mag + 7;
        }
        else if (txClass == TxClassHoriz)
        {
            mag += levels[posIdx + (stride << 1)];
            mag = (mag + 1) >> 1;
            if (mag > 6) mag = 6;
            if (c == 0) return mag;
            if (col == 0) return mag + 7;
        }
        else // TxClassVert
        {
            mag += levels[posIdx + 2];
            mag = (mag + 1) >> 1;
            if (mag > 6) mag = 6;
            if (c == 0) return mag;
            if (row == 0) return mag + 7;
        }
        return mag + 14;
    }

    /// <summary>
    /// libaom <c>set_dc_sign(cul_level, dc_val)</c>: encodes DC sign into
    /// the 7th bit of cul_level.
    /// </summary>
    public static int SetDcSign(int culLevel, int dcVal)
    {
        if (dcVal < 0) culLevel |= (1 << CoeffContextBits);
        else if (dcVal > 0) culLevel |= (2 << CoeffContextBits);
        return culLevel;
    }

    /// <summary>
    /// libaom <c>av1_get_eob_pos_token</c>: classify EOB into a group index
    /// and return the (eob - eob_group_start[t]) refinement extra. Returned
    /// token matches libaom convention (group index in 0..11). Caller writes
    /// (token - 1) to the eob_flag_cdf.
    /// <para>
    /// <paramref name="eobGroupStart"/> is the libaom <c>av1_eob_group_start[12]</c>
    /// table passed as ArrayView&lt;short&gt; for kernel-side lookup.
    /// </para>
    /// </summary>
    public static int GetEobPosToken(int eob, ArrayView<short> eobGroupStart, out int extra)
    {
        int t;
        if (eob < 33)
        {
            t = EobToPosSmall(eob);
        }
        else
        {
            int e = (eob - 1) >> 5;
            if (e > 16) e = 16;
            t = EobToPosLarge(e);
        }
        extra = eob - eobGroupStart[t];
        return t;
    }

    /// <summary>
    /// libaom write_golomb length: returns 2 * floor(log2(level + 1)) + 1
    /// (the total number of bits the golomb tail emits for this level).
    /// Used by callers that need to size their output bitstream worst-case
    /// before the encoder runs.
    /// </summary>
    public static int GolombBitLength(int level)
    {
        int x = level + 1;
        int length = 0;
        while (x != 0) { x >>= 1; length++; }
        return 2 * length - 1;
    }

    /// <summary>
    /// Clip value to [0, 3] - replaces the byte ClipMax3 lookup table for
    /// in-kernel use (saves a table fetch).
    /// </summary>
    private static int ClipMax3(byte v)
    {
        int x = v;
        return x > 3 ? 3 : x;
    }

    /// <summary>libaom eob_to_pos_small[33] - inlined as branches.</summary>
    private static int EobToPosSmall(int eob)
    {
        if (eob <= 2) return eob;          // 0, 1, 2
        if (eob <= 4) return 3;            // 3, 4
        if (eob <= 8) return 4;            // 5..8
        if (eob <= 16) return 5;           // 9..16
        return 6;                          // 17..32
    }

    /// <summary>libaom eob_to_pos_large[17] - inlined as branches.</summary>
    private static int EobToPosLarge(int e)
    {
        if (e == 0) return 6;
        if (e == 1) return 7;
        if (e <= 3) return 8;              // 2, 3
        if (e <= 7) return 9;              // 4..7
        if (e <= 15) return 10;            // 8..15
        return 11;                         // 16
    }
}
