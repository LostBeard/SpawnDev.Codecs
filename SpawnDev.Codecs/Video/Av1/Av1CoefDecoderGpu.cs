// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 coefficient (token) decoder, GPU-callable form. Bit-exact mirror
// of Av1CoefDecoder.ReadCoeffsTxb (libaom av1_read_coeffs_txb port).
// Symmetric companion to Av1CoefEncoderGpu.
//
// Scope (matches Av1KeyframeEncoder v1):
//   - Tx8x8 + DCT_DCT (chroma)
//   - Tx16x16 + DCT_DCT (luma)
//   - DC_PRED only (intra mode)
//   - 8-bit (bitDepth = 8); txScale = 0 for both Tx8x8 / Tx16x16.
//
// Caller passes the precomputed dequant scales (qDc + qAc) per block,
// so we don't need to pack the DcQuantQtx / AcQuantQtx tables. The
// dequantized coefs are written to dqCoeffsRaster in row-major
// (raster) layout matching the CPU reference. EOB + CulLevel are
// returned via the parallel output ArrayViews.

using ILGPU;
using SpawnDev.Codecs.EntropyCoders;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// GPU-callable AV1 coefficient (token) decoder. Bit-exact mirror of
/// <see cref="Av1CoefDecoder"/> for the v1 keyframe decoder's
/// (Tx8x8/Tx16x16, DCT_DCT, DC_PRED) configurations.
/// </summary>
public static class Av1CoefDecoderGpu
{
    /// <summary>
    /// Decode one transform block from the range decoder state. Mirrors
    /// Av1CoefDecoder.ReadCoeffsTxb bit-for-bit.
    /// Returns Eob via <paramref name="eobOut"/> and CulLevel via
    /// <paramref name="culLevelOut"/> (both stored at index <paramref name="blockIdx"/>);
    /// dequantized coefs written to dqCoeffsRaster starting at coefBase.
    /// </summary>
    public static void ReadCoeffsTxb(
        ref Av1RangeDecoderGpuState rd,
        ArrayView<byte> inBuf,
        ArrayView<byte> constsByte,
        ArrayView<ushort> constsUshort,
        ArrayView<int> dqCoeffsRaster, long coefBase,
        ArrayView<byte> levelsBuf, long levelsBase,
        int txSize, int plane, int qctx,
        int txbSkipCtx, int dcSignCtx, int qindex,
        int qDc, int qAc, // dequant scales (caller-precomputed shorts)
        ArrayView<int> eobOut, ArrayView<int> culLevelOut, int blockIdx)
    {
        int planeType = plane == 0 ? 0 : 1;
        int txsLocal = txSize == 1 ? 0 : 1;
        int width = txSize == 1 ? 8 : 16;
        int height = txSize == 1 ? 8 : 16;
        int bhl = txSize == 1 ? 3 : 4;
        int outW = width;
        int outH = height;
        int n = outW * outH;

        int scanOffset = txSize == 1
            ? Av1KeyframeConstantsGpu.Scan8x8Offset
            : Av1KeyframeConstantsGpu.Scan16x16Offset;

        // Zero output coefs first - libaom CoefBlock allocates a fresh
        // int[n] which is zero-initialized; non-emitted positions stay 0.
        for (int i = 0; i < n; i++) dqCoeffsRaster[coefBase + i] = 0;

        // ---- Step 1: txb_skip ----
        long skipCdfBase = Av1KeyframeConstantsGpu.TxbSkipCdfOffset
            + ((long)((qctx * 2 + txsLocal) * Av1KeyframeConstantsGpu.TxbSkipContexts + txbSkipCtx)) * 3;
        int allZero = Av1RangeDecoderGpu.DecodeCdfQ15(ref rd, inBuf, constsUshort, skipCdfBase, 2);
        if (allZero != 0)
        {
            eobOut[blockIdx] = 0;
            culLevelOut[blockIdx] = 0;
            return;
        }

        // ---- Step 2: tx_type. v1 always DCT_DCT. ----
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
            // Decoded sym is consumed (we don't need to remap it back to
            // tx_type since v1 always emits DCT_DCT and the encoder
            // wrote sym=1 for DCT_DCT; here we read it back for entropy
            // state symmetry but ignore the value).
            int unused = Av1RangeDecoderGpu.DecodeCdfQ15(ref rd, inBuf, constsUshort, extTxCdfBase, numSet);
        }

        int txClass = Av1TxbCommonGpu.TxClass2d;

        // ---- Step 3: eob_multi ----
        int eobPt = ReadEobMulti(ref rd, inBuf, constsUshort, txSize, planeType, txClass, qctx);

        // ---- Step 4: eob_extra ----
        int eobExtra = 0;
        int eobOffsetBits = ReadEobOffsetBits(constsByte, eobPt);
        if (eobOffsetBits > 0)
        {
            int eobCtx = eobPt - 3;
            long extraCdfBase = Av1KeyframeConstantsGpu.EobExtraCdfOffset
                + ((long)(((qctx * 2 + txsLocal) * Av1KeyframeConstantsGpu.PlaneTypes + planeType)
                    * Av1KeyframeConstantsGpu.EobCoefContexts + eobCtx)) * 3;
            int bit = Av1RangeDecoderGpu.DecodeCdfQ15(ref rd, inBuf, constsUshort, extraCdfBase, 2);
            if (bit != 0) eobExtra += 1 << (eobOffsetBits - 1);
            for (int i = 1; i < eobOffsetBits; i++)
            {
                bit = (int)Av1RangeDecoderGpu.DecodeBits(ref rd, inBuf, 1);
                if (bit != 0) eobExtra += 1 << (eobOffsetBits - 1 - i);
            }
        }

        // EobGroupStart[eobPt] from packed buffer.
        int egsLo = constsByte[Av1KeyframeConstantsGpu.EobGroupStartOffset + eobPt * 2];
        int egsHi = constsByte[Av1KeyframeConstantsGpu.EobGroupStartOffset + eobPt * 2 + 1];
        int egs = egsLo | (egsHi << 8);
        int eob = egs;
        if (eob > 2) eob += eobExtra;

        if (eob <= 0 || eob > n)
        {
            eobOut[blockIdx] = 0;
            culLevelOut[blockIdx] = 0;
            return;
        }

        // ---- Build padded levels[] - zero first, then write in-loop. ----
        // Buffer size for Tx16x16 = (16+4) * (16+6) + 16 = 456 bytes; for
        // Tx8x8 = (8+4) * (8+6) + 16 = 184. Both fit in int easily; use
        // int counter to avoid OpenCL backend trip-ups on long loops.
        int paddedStride = (1 << bhl) + Av1TxbCommonGpu.TxPadHor;
        int paddedRows = height + Av1TxbCommonGpu.TxPadVer;
        int bufBytes = paddedStride * paddedRows + Av1TxbCommonGpu.TxPadEnd;
        for (int i = 0; i < bufBytes; i++) levelsBuf[levelsBase + i] = 0;
        int levelsOff = Av1TxbCommonGpu.SetLevelsOffset(height);

        long nzMapCtxOffsetBase = txSize == 1
            ? Av1KeyframeConstantsGpu.NzMapCtxOffset8x8Offset
            : Av1KeyframeConstantsGpu.NzMapCtxOffset16x16Offset;

        // ---- Step 6: read EOB-position coefficient ----
        {
            int c = eob - 1;
            int pos = constsUshort[scanOffset + c];
            int coefCtx = Av1TxbCommonGpu.GetLowerLevelsCtxEob(bhl, width, c);
            long baseEobCdfBase = Av1KeyframeConstantsGpu.CoeffBaseEobMultiCdfOffset
                + ((long)(((qctx * 2 + txsLocal) * Av1KeyframeConstantsGpu.PlaneTypes + planeType)
                    * Av1KeyframeConstantsGpu.SigCoefContextsEob + coefCtx)) * 4;
            int level = Av1RangeDecoderGpu.DecodeCdfQ15(ref rd, inBuf, constsUshort, baseEobCdfBase, 3) + 1;
            if (level > Av1TxbCommonGpu.NumBaseLevels)
            {
                int brCtx = Av1TxbCommonGpu.GetBrCtxEob(pos, bhl, txClass);
                long brCdfBase = Av1KeyframeConstantsGpu.CoeffLpsMultiCdfOffset
                    + ((long)(((qctx * 2 + txsLocal) * Av1KeyframeConstantsGpu.PlaneTypes + planeType)
                        * Av1KeyframeConstantsGpu.LevelContexts + brCtx)) * (Av1TxbCommonGpu.BrCdfSize + 1);
                for (int idx = 0; idx < Av1TxbCommonGpu.CoeffBaseRange; idx += Av1TxbCommonGpu.BrCdfSize - 1)
                {
                    int k = Av1RangeDecoderGpu.DecodeCdfQ15(ref rd, inBuf, constsUshort, brCdfBase, Av1TxbCommonGpu.BrCdfSize);
                    level += k;
                    if (k < Av1TxbCommonGpu.BrCdfSize - 1) break;
                }
            }
            int padIdx = Av1TxbCommonGpu.GetPaddedIdx(pos, bhl);
            int lvl = level > 255 ? 255 : level;
            levelsBuf[levelsBase + levelsOff + padIdx] = (byte)lvl;
        }

        // ---- Step 7: read non-EOB coefficients in REVERSE scan ----
        if (eob > 1)
        {
            for (int c = eob - 2; c >= 0; c--)
            {
                int pos = constsUshort[scanOffset + c];
                int coefCtx = (c == 0)
                    ? Av1TxbCommonGpu.GetLowerLevelsCtxEob(bhl, width, 0)
                    : Av1TxbCommonGpu.GetLowerLevelsCtx2d(levelsBuf, levelsBase, pos, bhl, constsByte, nzMapCtxOffsetBase);

                long baseCdfBase = Av1KeyframeConstantsGpu.CoeffBaseMultiCdfOffset
                    + ((long)(((qctx * 2 + txsLocal) * Av1KeyframeConstantsGpu.PlaneTypes + planeType)
                        * Av1KeyframeConstantsGpu.SigCoefContexts + coefCtx)) * 5;
                int level = Av1RangeDecoderGpu.DecodeCdfQ15(ref rd, inBuf, constsUshort, baseCdfBase, 4);
                if (level > Av1TxbCommonGpu.NumBaseLevels)
                {
                    int brCtx = Av1TxbCommonGpu.GetBrCtx(levelsBuf, levelsBase, pos, bhl, txClass);
                    long brCdfBase = Av1KeyframeConstantsGpu.CoeffLpsMultiCdfOffset
                        + ((long)(((qctx * 2 + txsLocal) * Av1KeyframeConstantsGpu.PlaneTypes + planeType)
                            * Av1KeyframeConstantsGpu.LevelContexts + brCtx)) * (Av1TxbCommonGpu.BrCdfSize + 1);
                    for (int idx = 0; idx < Av1TxbCommonGpu.CoeffBaseRange; idx += Av1TxbCommonGpu.BrCdfSize - 1)
                    {
                        int k = Av1RangeDecoderGpu.DecodeCdfQ15(ref rd, inBuf, constsUshort, brCdfBase, Av1TxbCommonGpu.BrCdfSize);
                        level += k;
                        if (k < Av1TxbCommonGpu.BrCdfSize - 1) break;
                    }
                }
                int padIdx = Av1TxbCommonGpu.GetPaddedIdx(pos, bhl);
                int lvl = level > 255 ? 255 : level;
                levelsBuf[levelsBase + levelsOff + padIdx] = (byte)lvl;
            }
        }

        // ---- Step 8: signs + dequant. v1 always 8-bit, txScale=0. ----
        int culLevel = 0;
        int dcVal = 0;
        const int bitDepth = 8;
        const int shift = 0;
        const int maxValue = (1 << (7 + bitDepth)) - 1; // 32767
        const int minValue = -(1 << (7 + bitDepth));    // -32768

        for (int c = 0; c < eob; c++)
        {
            int pos = constsUshort[scanOffset + c];
            int padIdx = Av1TxbCommonGpu.GetPaddedIdx(pos, bhl);
            int level = levelsBuf[levelsBase + levelsOff + padIdx];
            if (level == 0) continue;

            int sign;
            if (c == 0)
            {
                long dcSignCdfBase = Av1KeyframeConstantsGpu.DcSignCdfOffset
                    + ((long)((qctx * Av1KeyframeConstantsGpu.PlaneTypes + planeType)
                        * Av1KeyframeConstantsGpu.DcSignContexts + dcSignCtx)) * 3;
                sign = Av1RangeDecoderGpu.DecodeCdfQ15(ref rd, inBuf, constsUshort, dcSignCdfBase, 2);
            }
            else
            {
                sign = (int)Av1RangeDecoderGpu.DecodeBits(ref rd, inBuf, 1);
            }

            if (level >= Av1TxbCommonGpu.MaxBaseBrRange)
            {
                level += ReadGolomb(ref rd, inBuf);
            }

            if (c == 0) dcVal = sign != 0 ? -level : level;
            culLevel += level;

            int dq = c == 0 ? qDc : qAc;
            long dqCoeff = ((long)level * dq) & 0xFFFFFF;
            int dqInt = (int)dqCoeff;
            dqInt = dqInt >> shift;
            if (sign != 0) dqInt = -dqInt;
            if (dqInt > maxValue) dqInt = maxValue;
            if (dqInt < minValue) dqInt = minValue;

            int col = pos >> bhl;
            int row = pos - (col << bhl);
            if (col < outW && row < outH)
            {
                dqCoeffsRaster[coefBase + row * outW + col] = dqInt;
            }
        }

        if (culLevel > Av1TxbCommonGpu.CoeffContextMask) culLevel = Av1TxbCommonGpu.CoeffContextMask;
        culLevel = Av1TxbCommonGpu.SetDcSign(culLevel, dcVal);
        eobOut[blockIdx] = eob;
        culLevelOut[blockIdx] = culLevel;
    }

    // ----------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------

    private static int ReadEobOffsetBits(ArrayView<byte> constsByte, int eobPt)
    {
        int lo = constsByte[Av1KeyframeConstantsGpu.EobOffsetBitsOffset + eobPt * 2];
        int hi = constsByte[Av1KeyframeConstantsGpu.EobOffsetBitsOffset + eobPt * 2 + 1];
        return lo | (hi << 8);
    }

    private static int ReadEobMulti(
        ref Av1RangeDecoderGpuState rd, ArrayView<byte> inBuf,
        ArrayView<ushort> constsUshort, int txSize, int planeType, int txClass, int qctx)
    {
        int eobMultiCtx = (txClass == Av1TxbCommonGpu.TxClass2d) ? 0 : 1;
        if (txSize == 1)
        {
            // Tx8x8 - 7 syms.
            long cdfBase = Av1KeyframeConstantsGpu.EobMulti64CdfOffset
                + ((long)((qctx * Av1KeyframeConstantsGpu.PlaneTypes + planeType) * 2 + eobMultiCtx)) * 8;
            return Av1RangeDecoderGpu.DecodeCdfQ15(ref rd, inBuf, constsUshort, cdfBase, 7) + 1;
        }
        else
        {
            // Tx16x16 - 9 syms.
            long cdfBase = Av1KeyframeConstantsGpu.EobMulti256CdfOffset
                + ((long)((qctx * Av1KeyframeConstantsGpu.PlaneTypes + planeType) * 2 + eobMultiCtx)) * 10;
            return Av1RangeDecoderGpu.DecodeCdfQ15(ref rd, inBuf, constsUshort, cdfBase, 9) + 1;
        }
    }

    private static int ReadGolomb(ref Av1RangeDecoderGpuState rd, ArrayView<byte> inBuf)
    {
        int x = 1;
        int length = 0;
        int i = 0;
        while (i == 0)
        {
            i = (int)Av1RangeDecoderGpu.DecodeBits(ref rd, inBuf, 1);
            length++;
            if (length > 20) break; // defensive against malformed stream
        }
        for (i = 0; i < length - 1; i++)
        {
            x <<= 1;
            x += (int)Av1RangeDecoderGpu.DecodeBits(ref rd, inBuf, 1);
        }
        return x - 1;
    }
}

