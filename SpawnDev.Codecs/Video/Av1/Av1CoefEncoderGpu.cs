// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 coefficient (token) encoder, GPU-callable form. Bit-exact mirror
// of Av1CoefEncoder.WriteCoeffsTxb (libaom av1_write_coeffs_txb port).
//
// Scope (matches Av1KeyframeEncoder v1 use cases):
//   - Tx8x8 + DCT_DCT (chroma)
//   - Tx16x16 + DCT_DCT (luma)
//   - Plane 0 = Y, planes 1/2 = UV
//   - DC_PRED only (intraMode = Av1IntraMode.Dc)
//   - reducedTxSet = false
//
// All CDF tables are read from the unified Av1KeyframeConstantsGpu
// ushort buffer using its layout offsets. Caller passes the constants
// buffers + raw byte buffer for the libaom-layout levels[] scratch
// (pre-sized worst-case to (32+TxPadHor)*(32+TxPadVer)+TxPadEnd = 1384
// bytes per concurrent block).
//
// Output is written via Av1RangeEncoderGpu state + outBuf (the
// per-tile range encoder buffer the caller manages).
//
// Returns Eob + CulLevel via two ArrayView&lt;int&gt; outputs (one entry per
// block index) - mirrors what Av1CoefEncoder.EncodedCoefBlock returns
// to its CPU caller for neighbor-context bookkeeping.

using ILGPU;
using SpawnDev.Codecs.EntropyCoders;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// GPU-callable AV1 coefficient (token) encoder. Bit-exact mirror of
/// <see cref="Av1CoefEncoder"/> for the v1 keyframe encoder's
/// (Tx8x8/Tx16x16, DCT_DCT, DC_PRED) configurations.
/// </summary>
public static class Av1CoefEncoderGpu
{
    /// <summary>
    /// Write one transform block's coefficients to the range encoder
    /// state. Mirrors Av1CoefEncoder.WriteCoeffsTxb bit-for-bit.
    /// Returns Eob via <paramref name="eobOut"/> and CulLevel via
    /// <paramref name="culLevelOut"/> (both stored at index <paramref name="blockIdx"/>).
    /// </summary>
    /// <param name="re">Range encoder state (mutated as bits emit).</param>
    /// <param name="outBuf">Range encoder output byte buffer.</param>
    /// <param name="constsByte">Av1KeyframeConstantsGpu byte buffer.</param>
    /// <param name="constsUshort">Av1KeyframeConstantsGpu ushort buffer.</param>
    /// <param name="coefsRaster">Quantized coefs (raster layout); per-block
    /// section starting at <paramref name="coefBase"/>.</param>
    /// <param name="coefBase">Starting offset of the block's coefs in
    /// <paramref name="coefsRaster"/>.</param>
    /// <param name="levelsBuf">Padded levels[] scratch (worst-case 1384
    /// bytes per concurrent block); contents replaced.</param>
    /// <param name="levelsBase">Starting offset within
    /// <paramref name="levelsBuf"/> for this block's scratch.</param>
    /// <param name="txSize">Tx size: pass (int)Av1TxSize.Tx8x8 = 1 or
    /// (int)Av1TxSize.Tx16x16 = 2.</param>
    /// <param name="plane">0 = Y, 1 = U, 2 = V.</param>
    /// <param name="qctx">Quantizer-bin index (0..3).</param>
    /// <param name="txbSkipCtx">txb_skip CDF context (0..12).</param>
    /// <param name="dcSignCtx">dc_sign CDF context (0..2).</param>
    /// <param name="qindex">Frame base q-index (1..255).</param>
    /// <param name="eobOut">Output: Eob written at index blockIdx.</param>
    /// <param name="culLevelOut">Output: CulLevel written at index blockIdx.</param>
    /// <param name="blockIdx">Index in eobOut/culLevelOut where this
    /// block's results land.</param>
    public static void WriteCoeffsTxb(
        ref Av1RangeEncoderGpuState re,
        ArrayView<byte> outBuf,
        ArrayView<byte> constsByte,
        ArrayView<ushort> constsUshort,
        ArrayView<int> coefsRaster, long coefBase,
        ArrayView<byte> levelsBuf, long levelsBase,
        int txSize, int plane, int qctx,
        int txbSkipCtx, int dcSignCtx, int qindex,
        ArrayView<int> eobOut, ArrayView<int> culLevelOut, int blockIdx)
    {
        int planeType = plane == 0 ? 0 : 1;
        int txsLocal = txSize == 1 ? 0 : 1; // Tx8x8 -> 0, Tx16x16 -> 1.
        int width = txSize == 1 ? 8 : 16;
        int height = txSize == 1 ? 8 : 16;
        int bhl = txSize == 1 ? 3 : 4;
        int outW = width;
        int outH = height;

        // Scan offset in the packed buffer.
        int scanOffset = txSize == 1
            ? Av1KeyframeConstantsGpu.Scan8x8Offset
            : Av1KeyframeConstantsGpu.Scan16x16Offset;
        int nCoefs = width * height;

        // ---- Compute EOB by walking the scan in reverse. ----
        int eob = ComputeEob(coefsRaster, coefBase, constsUshort, scanOffset,
            nCoefs, bhl, outW, outH);

        // ---- Step 1: txb_skip - one bin. ----
        int allZero = eob == 0 ? 1 : 0;
        long skipCdfBase = Av1KeyframeConstantsGpu.TxbSkipCdfOffset
            + ((long)((qctx * 2 + txsLocal) * Av1KeyframeConstantsGpu.TxbSkipContexts + txbSkipCtx)) * 3;
        Av1RangeEncoderGpu.EncodeCdfQ15(ref re, outBuf, allZero, constsUshort, skipCdfBase, 2);

        if (eob == 0)
        {
            eobOut[blockIdx] = 0;
            culLevelOut[blockIdx] = 0;
            return;
        }

        // ---- Step 2: tx_type. Y plane only with qindex > 0; for v1 we
        // always emit DCT_DCT. The CDF row + sym index depend on tx size:
        //   Tx8x8  -> set 3, 7 syms, DCT_DCT sym = 1
        //   Tx16x16 -> set 2, 5 syms, DCT_DCT sym = 1
        // ----
        if (plane == 0 && qindex > 0)
        {
            long extTxCdfBase;
            int numSet;
            if (txSize == 1)
            {
                extTxCdfBase = Av1KeyframeConstantsGpu.IntraExtTxCdfTx8DcOffset;
                numSet = 7;
            }
            else
            {
                extTxCdfBase = Av1KeyframeConstantsGpu.IntraExtTxCdfTx16DcOffset;
                numSet = 5;
            }
            Av1RangeEncoderGpu.EncodeCdfQ15(ref re, outBuf, 1, constsUshort, extTxCdfBase, numSet);
        }

        // tx_class for DCT_DCT is 2D = 0.
        int txClass = Av1TxbCommonGpu.TxClass2d;

        // ---- Step 3: eob_pt classification. ----
        int eobPt = GetEobPosToken(eob, constsByte, out int eobExtra);
        int eobMultiSize = txSize == 1 ? 2 : 4; // libaom TxSizeLog2Minus4 - 8x8=2, 16x16=4.
        int eobMultiCtx = (txClass == Av1TxbCommonGpu.TxClass2d) ? 0 : 1;
        WriteEobMulti(ref re, outBuf, constsUshort, eobMultiSize, planeType, eobMultiCtx, eobPt, qctx);

        // ---- Step 4: eob_extra - first bit via CDF, rest as raw bits. ----
        int eobOffsetBits = ReadEobOffsetBits(constsByte, eobPt);
        if (eobOffsetBits > 0)
        {
            int eobCtx = eobPt - 3;
            long extraCdfBase = Av1KeyframeConstantsGpu.EobExtraCdfOffset
                + ((long)(((qctx * 2 + txsLocal) * Av1KeyframeConstantsGpu.PlaneTypes + planeType)
                    * Av1KeyframeConstantsGpu.EobCoefContexts + eobCtx)) * 3;
            int shift = eobOffsetBits - 1;
            int firstBit = (eobExtra >> shift) & 1;
            Av1RangeEncoderGpu.EncodeCdfQ15(ref re, outBuf, firstBit, constsUshort, extraCdfBase, 2);
            for (int i = 1; i < eobOffsetBits; i++)
            {
                shift = eobOffsetBits - 1 - i;
                int b = (eobExtra >> shift) & 1;
                Av1RangeEncoderGpu.EncodeBits(ref re, outBuf, (uint)b, 1);
            }
        }

        // ---- Build the libaom-layout levels[] padded buffer. ----
        InitLevels(coefsRaster, coefBase, outW, outH, width, height, bhl,
            levelsBuf, levelsBase);
        int levelsOff = Av1TxbCommonGpu.SetLevelsOffset(height);

        // NzMapCtxOffset table for this tx size (used by GetLowerLevelsCtx2d).
        long nzMapCtxOffsetBase = txSize == 1
            ? Av1KeyframeConstantsGpu.NzMapCtxOffset8x8Offset
            : Av1KeyframeConstantsGpu.NzMapCtxOffset16x16Offset;

        // ---- Step 5+6: write base CDFs in REVERSE scan order. ----
        for (int c = eob - 1; c >= 0; c--)
        {
            int pos = constsUshort[scanOffset + c];
            int padIdx = Av1TxbCommonGpu.GetPaddedIdx(pos, bhl);
            int level = levelsBuf[levelsBase + levelsOff + padIdx];

            int coefCtx;
            if (c == eob - 1)
            {
                // EOB position: 3-sym CDF (coeff_base_eob).
                coefCtx = Av1TxbCommonGpu.GetLowerLevelsCtxEob(bhl, width, c);
                long baseEobCdfBase = Av1KeyframeConstantsGpu.CoeffBaseEobMultiCdfOffset
                    + ((long)(((qctx * 2 + txsLocal) * Av1KeyframeConstantsGpu.PlaneTypes + planeType)
                        * Av1KeyframeConstantsGpu.SigCoefContextsEob + coefCtx)) * 4;
                int sym = level - 1;
                if (sym > 2) sym = 2;
                Av1RangeEncoderGpu.EncodeCdfQ15(ref re, outBuf, sym, constsUshort, baseEobCdfBase, 3);
            }
            else
            {
                // Non-EOB: 4-sym CDF (coeff_base). 2D class always.
                // GetLowerLevelsCtx2d adds TxPadTop*stride internally, so
                // pass the raw block base (NOT levelsBase + levelsOff -
                // that would double the pad-top skip).
                coefCtx = (c == 0)
                    ? Av1TxbCommonGpu.GetLowerLevelsCtxEob(bhl, width, 0)
                    : Av1TxbCommonGpu.GetLowerLevelsCtx2d(levelsBuf, levelsBase,
                        pos, bhl, constsByte, nzMapCtxOffsetBase);
                long baseCdfBase = Av1KeyframeConstantsGpu.CoeffBaseMultiCdfOffset
                    + ((long)(((qctx * 2 + txsLocal) * Av1KeyframeConstantsGpu.PlaneTypes + planeType)
                        * Av1KeyframeConstantsGpu.SigCoefContexts + coefCtx)) * 5;
                int sym = level;
                if (sym > 3) sym = 3;
                Av1RangeEncoderGpu.EncodeCdfQ15(ref re, outBuf, sym, constsUshort, baseCdfBase, 4);
            }

            // ---- Step 7: coeff_lps for level > NumBaseLevels (= 2). ----
            if (level > Av1TxbCommonGpu.NumBaseLevels)
            {
                int baseRange = level - 1 - Av1TxbCommonGpu.NumBaseLevels;
                // GetBrCtx adds TxPadTop*stride internally; pass raw block base.
                int brCtx = (c == eob - 1)
                    ? Av1TxbCommonGpu.GetBrCtxEob(pos, bhl, txClass)
                    : Av1TxbCommonGpu.GetBrCtx(levelsBuf, levelsBase, pos, bhl, txClass);
                int lpsTxsLocal = txsLocal; // both 0 (Tx8x8) and 1 (Tx16x16) are below the libaom min(txsCtx, 3) cap.
                long brCdfBase = Av1KeyframeConstantsGpu.CoeffLpsMultiCdfOffset
                    + ((long)(((qctx * 2 + lpsTxsLocal) * Av1KeyframeConstantsGpu.PlaneTypes + planeType)
                        * Av1KeyframeConstantsGpu.LevelContexts + brCtx)) * (Av1TxbCommonGpu.BrCdfSize + 1);
                for (int idx = 0; idx < Av1TxbCommonGpu.CoeffBaseRange; idx += Av1TxbCommonGpu.BrCdfSize - 1)
                {
                    int k = baseRange - idx;
                    int kCap = Av1TxbCommonGpu.BrCdfSize - 1;
                    if (k > kCap) k = kCap;
                    Av1RangeEncoderGpu.EncodeCdfQ15(ref re, outBuf, k, constsUshort, brCdfBase, Av1TxbCommonGpu.BrCdfSize);
                    if (k < kCap) break;
                }
            }
        }

        // ---- Step 8: signs + golomb tails. Walk scan FORWARD. ----
        int culLevel = 0;
        int dcVal = 0;
        for (int c = 0; c < eob; c++)
        {
            int pos = constsUshort[scanOffset + c];
            int padIdx = Av1TxbCommonGpu.GetPaddedIdx(pos, bhl);
            int level = levelsBuf[levelsBase + levelsOff + padIdx];
            if (level == 0) continue;

            int signedVal = ReadCoeff(coefsRaster, coefBase, pos, bhl, outW, outH);
            int sign = signedVal < 0 ? 1 : 0;

            if (c == 0)
            {
                long dcSignCdfBase = Av1KeyframeConstantsGpu.DcSignCdfOffset
                    + ((long)((qctx * Av1KeyframeConstantsGpu.PlaneTypes + planeType)
                        * Av1KeyframeConstantsGpu.DcSignContexts + dcSignCtx)) * 3;
                Av1RangeEncoderGpu.EncodeCdfQ15(ref re, outBuf, sign, constsUshort, dcSignCdfBase, 2);
            }
            else
            {
                Av1RangeEncoderGpu.EncodeBits(ref re, outBuf, (uint)sign, 1);
            }

            int absLevel = signedVal < 0 ? -signedVal : signedVal;
            if (absLevel > Av1TxbCommonGpu.CoeffBaseRange + Av1TxbCommonGpu.NumBaseLevels)
            {
                int golombVal = absLevel - Av1TxbCommonGpu.CoeffBaseRange - 1 - Av1TxbCommonGpu.NumBaseLevels;
                WriteGolomb(ref re, outBuf, golombVal);
            }

            if (c == 0) dcVal = signedVal;
            culLevel += absLevel;
        }

        if (culLevel > Av1TxbCommonGpu.CoeffContextMask)
            culLevel = Av1TxbCommonGpu.CoeffContextMask;
        culLevel = Av1TxbCommonGpu.SetDcSign(culLevel, dcVal);

        eobOut[blockIdx] = eob;
        culLevelOut[blockIdx] = culLevel;
    }

    // ----------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------

    private static int ComputeEob(
        ArrayView<int> coefs, long coefBase,
        ArrayView<ushort> scanBuf, int scanOffset,
        int nCoefs, int bhl, int outW, int outH)
    {
        for (int c = nCoefs - 1; c >= 0; c--)
        {
            int pos = scanBuf[scanOffset + c];
            int v = ReadCoeff(coefs, coefBase, pos, bhl, outW, outH);
            if (v != 0) return c + 1;
        }
        return 0;
    }

    private static int ReadCoeff(
        ArrayView<int> coefs, long coefBase, int pos, int bhl, int outW, int outH)
    {
        int col = pos >> bhl;
        int row = pos - (col << bhl);
        if (col >= outW || row >= outH) return 0;
        return coefs[coefBase + row * outW + col];
    }

    private static void InitLevels(
        ArrayView<int> coefs, long coefBase,
        int outW, int outH, int width, int height, int bhl,
        ArrayView<byte> levels, long levelsBase)
    {
        // Zero the buffer first - libaom zeroes the levels buffer
        // before the per-coef writes. Buffer size fits in int (max
        // ~456 bytes for Tx16x16); use int counter for OpenCL backend
        // codegen safety.
        int paddedStride = (1 << bhl) + Av1TxbCommonGpu.TxPadHor;
        int paddedRows = height + Av1TxbCommonGpu.TxPadVer;
        int bufBytes = paddedStride * paddedRows + Av1TxbCommonGpu.TxPadEnd;
        for (int i = 0; i < bufBytes; i++) levels[levelsBase + i] = 0;

        int levelsOff = Av1TxbCommonGpu.SetLevelsOffset(height);
        for (int col = 0; col < width; col++)
        {
            for (int row = 0; row < height; row++)
            {
                int v = ReadCoeff(coefs, coefBase, col * (1 << bhl) + row, bhl, outW, outH);
                int abs = v < 0 ? -v : v;
                if (abs > sbyte.MaxValue) abs = sbyte.MaxValue;
                levels[levelsBase + levelsOff + col * paddedStride + row] = (byte)abs;
            }
            // TX_PAD_HOR zero pad after each column run (already zero
            // from the InitLevels memset above).
        }
    }

    private static int GetEobPosToken(int eob, ArrayView<byte> constsByte, out int extra)
    {
        // Inlined logic of Av1TxbCommonGpu.GetEobPosToken but using the
        // packed EobGroupStart byte table (little-endian ushort).
        int t;
        if (eob < 33)
        {
            if (eob <= 2) t = eob;
            else if (eob <= 4) t = 3;
            else if (eob <= 8) t = 4;
            else if (eob <= 16) t = 5;
            else t = 6;
        }
        else
        {
            int e = (eob - 1) >> 5;
            if (e > 16) e = 16;
            if (e == 0) t = 6;
            else if (e == 1) t = 7;
            else if (e <= 3) t = 8;
            else if (e <= 7) t = 9;
            else if (e <= 15) t = 10;
            else t = 11;
        }
        // EobGroupStart[t] from packed buffer (little-endian ushort).
        int egsLo = constsByte[Av1KeyframeConstantsGpu.EobGroupStartOffset + t * 2];
        int egsHi = constsByte[Av1KeyframeConstantsGpu.EobGroupStartOffset + t * 2 + 1];
        int egs = egsLo | (egsHi << 8);
        extra = eob - egs;
        return t;
    }

    private static int ReadEobOffsetBits(ArrayView<byte> constsByte, int eobPt)
    {
        int lo = constsByte[Av1KeyframeConstantsGpu.EobOffsetBitsOffset + eobPt * 2];
        int hi = constsByte[Av1KeyframeConstantsGpu.EobOffsetBitsOffset + eobPt * 2 + 1];
        return lo | (hi << 8);
    }

    private static void WriteEobMulti(
        ref Av1RangeEncoderGpuState re, ArrayView<byte> outBuf,
        ArrayView<ushort> constsUshort,
        int eobMultiSize, int planeType, int eobMultiCtx, int eobPt, int qctx)
    {
        int sym = eobPt - 1;
        if (eobMultiSize == 2)
        {
            // 64 (Tx8x8) - 7 syms.
            long cdfBase = Av1KeyframeConstantsGpu.EobMulti64CdfOffset
                + ((long)((qctx * Av1KeyframeConstantsGpu.PlaneTypes + planeType) * 2 + eobMultiCtx)) * 8;
            Av1RangeEncoderGpu.EncodeCdfQ15(ref re, outBuf, sym, constsUshort, cdfBase, 7);
        }
        else if (eobMultiSize == 4)
        {
            // 256 (Tx16x16) - 9 syms.
            long cdfBase = Av1KeyframeConstantsGpu.EobMulti256CdfOffset
                + ((long)((qctx * Av1KeyframeConstantsGpu.PlaneTypes + planeType) * 2 + eobMultiCtx)) * 10;
            Av1RangeEncoderGpu.EncodeCdfQ15(ref re, outBuf, sym, constsUshort, cdfBase, 9);
        }
        // eobMultiSize 0/1/3/5/6 unused in v1 (Tx16/Tx32 chroma + Tx32+
        // Tx64 luma not produced by Av1KeyframeEncoder.EncodeSingleTile).
    }

    private static void WriteGolomb(
        ref Av1RangeEncoderGpuState re, ArrayView<byte> outBuf, int level)
    {
        int x = level + 1;
        int length = 0;
        int i = x;
        while (i != 0) { i >>= 1; length++; }

        for (i = 0; i < length - 1; i++)
            Av1RangeEncoderGpu.EncodeBits(ref re, outBuf, 0u, 1);
        for (i = length - 1; i >= 0; i--)
            Av1RangeEncoderGpu.EncodeBits(ref re, outBuf, (uint)((x >> i) & 1), 1);
    }
}
