// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 keyframe macroblock walker. Drives the per-MB decode pipeline:
//   1. Decode MB mode info (Y mode, sub-block modes, UV mode, segment, skip)
//   2. Compute per-MB dequantization values (Y1/Y2/UV DC+AC)
//   3. Decode coefficient blocks (Y2 first if not B_PRED, then 16 Y4, then 4 U, 4 V)
//   4. Inverse transform + intra-predict + clamp-add into the recon frame buffer
//   5. Update above/left entropy contexts
//
// Order of coef block decode (libvpx vp8_decode_mb_tokens):
//   - For non-B_PRED MBs:
//       Y2 block (block_type=PLANE_TYPE_Y2=1, ctx=a[8]+l[8], firstCoef=0)
//       Then 16 Y4 (block_type=PLANE_TYPE_Y_NO_DC=0, firstCoef=1 - DC came from Y2)
//   - For B_PRED MBs:
//       16 Y4 (block_type=PLANE_TYPE_Y_WITH_DC=3, firstCoef=0)
//   Then in both cases:
//       8 UV (block_type=PLANE_TYPE_UV=2, firstCoef=0) - 4 U then 4 V
//
// Macroblock entropy context (9 bytes per MB column, 9 per row state):
//   slots 0..3 = Y4 above (one per 4x4 col)
//   slots 4..5 = U above
//   slots 6..7 = V above
//   slot 8     = Y2 above
// The "left" array is the current row's per-plane left context (same layout).
//
// Out of scope for this slice:
//   - Loop filter (skipped; output will be slightly blocky vs ffmpeg)
//   - Multiple token partitions (Log2NumPartitions != 0 throws)
//   - Inter frames (non-key throws)

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// VP8 keyframe macroblock walker. Decodes one keyframe through the
/// already-shipped VP8 inverse-pipeline pieces and writes reconstructed
/// pixels into a Vp8FrameBuffer.
/// </summary>
public static class Vp8KeyframeWalker
{
    /// <summary>
    /// Decode a VP8 keyframe end-to-end into <paramref name="frameBuffer"/>.
    /// </summary>
    /// <param name="frameTag">Parsed 3+7 byte frame tag (must be IsKeyFrame).</param>
    /// <param name="frameHeader">Parsed compressed header (key path).</param>
    /// <param name="modeInfoReader">
    /// Bool decoder positioned at the FIRST mode-info bit of the first
    /// macroblock (i.e., immediately after Vp8FrameHeaderParser.ParseKeyFrameHeader
    /// returned). Used for mode-info decode only.
    /// </param>
    /// <param name="tokenPartitionBytes">
    /// Token partition bytes (the per-frame coefficient data following the
    /// first partition). For Log2NumPartitions == 0 this is the SINGLE token
    /// partition. The walker creates its own bool decoder over these bytes.
    /// </param>
    /// <param name="frameBuffer">Pre-allocated reconstruction frame buffer.</param>
    /// <param name="entropyContexts">Pre-allocated per-frame above/left entropy contexts.</param>
    public static void Decode(
        Vp8FrameTag frameTag,
        Vp8FrameHeader frameHeader,
        Vp8BoolDecoder modeInfoReader,
        byte[] tokenPartitionBytes,
        Vp8FrameBuffer frameBuffer,
        Vp8EntropyContexts entropyContexts)
    {
        ArgumentNullException.ThrowIfNull(frameTag);
        ArgumentNullException.ThrowIfNull(frameHeader);
        ArgumentNullException.ThrowIfNull(modeInfoReader);
        ArgumentNullException.ThrowIfNull(tokenPartitionBytes);
        ArgumentNullException.ThrowIfNull(frameBuffer);
        ArgumentNullException.ThrowIfNull(entropyContexts);

        if (!frameTag.IsKeyFrame)
            throw new NotImplementedException("Vp8KeyframeWalker only handles key frames; inter frames are out of scope for this slice.");
        if (frameHeader.Log2NumPartitions != 0)
            throw new NotImplementedException(
                $"Vp8KeyframeWalker only supports single-partition streams (Log2NumPartitions=0); got {frameHeader.Log2NumPartitions}.");
        if (entropyContexts.MbCols != frameBuffer.MbCols)
            throw new ArgumentException($"entropyContexts.MbCols ({entropyContexts.MbCols}) must equal frameBuffer.MbCols ({frameBuffer.MbCols})");

        // Token partition has its OWN bool decoder (libvpx pbi->mbc[0]).
        var tokenReader = new Vp8BoolDecoder(tokenPartitionBytes);

        entropyContexts.ClearAll();

        int mbRows = frameBuffer.MbRows;
        int mbCols = frameBuffer.MbCols;

        // Persistent context: the bottom row of sub-block modes from the MB row above
        // (one entry per 4x4 column = 4 * mbCols total). For the first MB row, init to
        // BDcPred to match libvpx's "no above MB available" convention.
        var subModeAboveRowOfFrame = new Vp8IntraMode4x4[mbCols * 4];
        for (int i = 0; i < subModeAboveRowOfFrame.Length; i++) subModeAboveRowOfFrame[i] = Vp8IntraMode4x4.BDcPred;

        // Track each MB-above's full Y mode (used to map non-B_PRED neighbors to a sub-mode).
        var aboveMbYMode = new Vp8YMode[mbCols];
        for (int i = 0; i < aboveMbYMode.Length; i++) aboveMbYMode[i] = Vp8YMode.DcPred;

        // Hoisted scratch buffers: stackalloc inside the inner loop trips CA2014 + actually
        // would allocate per-MB. Heap allocations once at the top of Decode is the right shape.
        var qcoeffBuf = new short[25 * 16]; // 25 4x4 blocks per MB, 16 coeffs each
        var eobsBuf = new int[25];
        var y2CoeffsBuf = new short[16];
        var y2RawBuf = new short[16];
        var y2DqBuf = new short[16];
        var dq16Buf = new short[16];
        var predCopyBuf = new byte[16];
        var yAboveBuf = new byte[16];
        var yLeftBuf = new byte[16];
        var uAboveBuf = new byte[8];
        var uLeftBuf = new byte[8];
        var vAboveBuf = new byte[8];
        var vLeftBuf = new byte[8];
        // 4x4 above-extended buffer: above[-1] + above[0..7] = 9 bytes useful; allocate 12 for headroom.
        var b4AboveExtBuf = new byte[12];
        var b4LeftBuf = new byte[4];
        var bModeProbsBuf = new byte[9];

        for (int mbRow = 0; mbRow < mbRows; mbRow++)
        {
            entropyContexts.ClearLeft();

            // Persistent "left" sub-modes for the current MB row (4 entries, top-to-bottom).
            // Reset to BDcPred at start of each row; updated as we walk MB columns.
            var subModeLeft = new Vp8IntraMode4x4[4];
            for (int i = 0; i < 4; i++) subModeLeft[i] = Vp8IntraMode4x4.BDcPred;

            // Track the previous MB's Y mode for left-edge sub-mode context resolution.
            var leftMbYMode = Vp8YMode.DcPred;

            for (int mbCol = 0; mbCol < mbCols; mbCol++)
            {
                bool haveAbove = mbRow > 0;
                bool haveLeft = mbCol > 0;

                // ---------- 1. Mode info ----------
                // Capture loop variables for the closure.
                int curMbCol = mbCol;
                bool curHaveAbove = haveAbove;
                bool curHaveLeft = haveLeft;
                Vp8YMode curLeftMbYMode = leftMbYMode;
                var curSubModeLeft = subModeLeft;

                Vp8KeyFrameMbModeInfo modeInfo = Vp8MbModeInfoDecoder.DecodeKeyFrameMb(
                    modeInfoReader,
                    frameHeader,
                    (subIdx, alreadyDecoded) =>
                    {
                        int sr = subIdx >> 2;       // sub-row 0..3
                        int sc = subIdx & 3;        // sub-col 0..3
                        Vp8IntraMode4x4 a = sr == 0
                            ? AboveModeFromMbAbove(curHaveAbove, aboveMbYMode[curMbCol], subModeAboveRowOfFrame, curMbCol, sc)
                            : alreadyDecoded[(sr - 1) * 4 + sc];
                        Vp8IntraMode4x4 l = sc == 0
                            ? LeftModeFromMbLeft(curHaveLeft, curLeftMbYMode, curSubModeLeft, sr)
                            : alreadyDecoded[sr * 4 + (sc - 1)];

                        Vp8KfBmodeProbs.GetProbs(a, l, bModeProbsBuf);
                        return bModeProbsBuf;
                    });

                // ---------- 2. Per-MB dequantization ----------
                Vp8MbDequant dq = Vp8MbDequantizer.Compute(modeInfo.SegmentId, frameHeader.Quantizer, frameHeader.Segmentation);

                // ---------- 3. Coefficient decode ----------
                Span<short> qcoeff = qcoeffBuf.AsSpan();
                Span<int> eobs = eobsBuf.AsSpan();
                Span<byte> mbAbove = entropyContexts.GetAbove(mbCol);
                Span<byte> mbLeft = entropyContexts.Left;

                if (modeInfo.SkipCoeff)
                {
                    // skip_coeff: zero out qcoeff & reset MB's entropy context to all-zero
                    // for slots we don't otherwise touch in this MB.
                    qcoeff.Clear();
                    eobs.Clear();
                    // Reset per-plane above/left for this MB (libvpx vp8_reset_mb_tokens_context).
                    // Y2 + Y4 + UV slots all clear to 0.
                    mbAbove.Clear();
                    mbLeft.Clear();
                }
                else
                {
                    DecodeMbCoefficients(tokenReader, frameHeader.CoefProbs, modeInfo.YMode != Vp8YMode.BPred, mbAbove, mbLeft, qcoeff, eobs);
                }

                // ---------- 4. Reconstruction ----------
                ReconstructMb(modeInfo, dq, qcoeff, eobs, frameBuffer, mbCol, mbRow, haveAbove, haveLeft,
                              y2RawBuf, y2DqBuf, y2CoeffsBuf, dq16Buf, predCopyBuf,
                              yAboveBuf, yLeftBuf, uAboveBuf, uLeftBuf, vAboveBuf, vLeftBuf,
                              b4AboveExtBuf, b4LeftBuf);

                // ---------- 5. Update sub-mode context ----------
                Vp8IntraMode4x4[]? mbSubModes = modeInfo.SubBlockModes;
                if (mbSubModes != null)
                {
                    // Save bottom row of sub-modes into subModeAboveRowOfFrame for the next MB row.
                    for (int sc = 0; sc < 4; sc++)
                        subModeAboveRowOfFrame[mbCol * 4 + sc] = mbSubModes[12 + sc];
                    // Save right column of sub-modes into subModeLeft for the next MB column.
                    for (int sr = 0; sr < 4; sr++)
                        subModeLeft[sr] = mbSubModes[sr * 4 + 3];
                }
                else
                {
                    // Non-B_PRED MB: project the Y mode into all 4x4 slots.
                    Vp8IntraMode4x4 implied = MapMbModeToSubMode(modeInfo.YMode);
                    for (int sc = 0; sc < 4; sc++)
                        subModeAboveRowOfFrame[mbCol * 4 + sc] = implied;
                    for (int sr = 0; sr < 4; sr++)
                        subModeLeft[sr] = implied;
                }

                aboveMbYMode[mbCol] = modeInfo.YMode;
                leftMbYMode = modeInfo.YMode;
            }
        }
    }

    /// <summary>
    /// Decode all coefficient blocks for one MB. Order:
    ///   - if hasY2: Y2 (block_type=1, ctx=a[8]+l[8], firstCoef=0)
    ///   - 16 Y4: block_type = 0 if hasY2 else 3, firstCoef = 1 if hasY2 else 0
    ///   - 4 U + 4 V (block_type=2, firstCoef=0)
    /// </summary>
    private static void DecodeMbCoefficients(
        Vp8BoolDecoder reader,
        byte[,,,] coefProbs,
        bool hasY2,
        Span<byte> mbAbove,
        Span<byte> mbLeft,
        Span<short> qcoeff,
        Span<int> eobs)
    {
        qcoeff.Clear();
        eobs.Clear();

        int firstCoef = hasY2 ? 1 : 0;
        int yBlockType = hasY2 ? 0 : 3;

        // ---- Y2 (when present) ----
        if (hasY2)
        {
            int ctx = mbAbove[Vp8EntropyContexts.Plane.Y2Slot] + mbLeft[Vp8EntropyContexts.Plane.Y2Slot];
            byte[,,] probsY2 = SliceBlockType(coefProbs, 1);
            int eob = Vp8CoefBlockDecoder.Decode(reader, probsY2, ctx, 0, qcoeff.Slice(24 * 16, 16));
            eobs[24] = eob;
            byte v = (byte)(eob > 0 ? 1 : 0);
            mbAbove[Vp8EntropyContexts.Plane.Y2Slot] = v;
            mbLeft[Vp8EntropyContexts.Plane.Y2Slot] = v;
        }

        // ---- 16 Y4 blocks ----
        byte[,,] probsY = SliceBlockType(coefProbs, yBlockType);
        for (int i = 0; i < 16; i++)
        {
            int aIdx = Vp8EntropyContexts.Plane.YBase + (i & 3);
            int lIdx = Vp8EntropyContexts.Plane.YBase + ((i & 0xC) >> 2);
            int ctx = mbAbove[aIdx] + mbLeft[lIdx];
            int eob = Vp8CoefBlockDecoder.Decode(reader, probsY, ctx, firstCoef, qcoeff.Slice(i * 16, 16));
            eobs[i] = eob;
            byte v = (byte)(eob > 0 ? 1 : 0);
            mbAbove[aIdx] = v;
            mbLeft[lIdx] = v;
        }

        // ---- 8 UV blocks (4 U + 4 V) ----
        // Above context: a = a_uv_base + ((i > 19) << 1) + (i & 1)
        // Left  context: l = l_uv_base + ((i > 19) << 1) + ((i & 3) > 1)
        byte[,,] probsUV = SliceBlockType(coefProbs, 2);
        for (int i = 16; i < 24; i++)
        {
            int aIdx = Vp8EntropyContexts.Plane.UBase + ((i > 19 ? 1 : 0) << 1) + (i & 1);
            int lIdx = Vp8EntropyContexts.Plane.UBase + ((i > 19 ? 1 : 0) << 1) + (((i & 3) > 1) ? 1 : 0);
            int ctx = mbAbove[aIdx] + mbLeft[lIdx];
            int eob = Vp8CoefBlockDecoder.Decode(reader, probsUV, ctx, 0, qcoeff.Slice(i * 16, 16));
            eobs[i] = eob;
            byte v = (byte)(eob > 0 ? 1 : 0);
            mbAbove[aIdx] = v;
            mbLeft[lIdx] = v;
        }
    }

    /// <summary>
    /// Extract the [band, ctx, node] slice for a given block type from the
    /// 4D coef probs table. Vp8CoefBlockDecoder takes [band, ctx, node].
    /// </summary>
    private static byte[,,] SliceBlockType(byte[,,,] coefProbs, int blockType)
    {
        int b = coefProbs.GetLength(1);
        int c = coefProbs.GetLength(2);
        int e = coefProbs.GetLength(3);
        var slice = new byte[b, c, e];
        for (int j = 0; j < b; j++)
            for (int k = 0; k < c; k++)
                for (int l = 0; l < e; l++)
                    slice[j, k, l] = coefProbs[blockType, j, k, l];
        return slice;
    }

    /// <summary>
    /// Reconstruct one MB into the frame buffer. For non-B_PRED: predict the
    /// entire 16x16 luma + 8x8 UV blocks first, then dequant + IDCT + add per
    /// 4x4 sub-block. For B_PRED: per-sub-block predict + IDCT + add in raster
    /// order so each sub-block sees the just-reconstructed left/above context.
    /// </summary>
    private static void ReconstructMb(
        Vp8KeyFrameMbModeInfo modeInfo,
        Vp8MbDequant dq,
        Span<short> qcoeff,
        Span<int> eobs,
        Vp8FrameBuffer frameBuffer,
        int mbCol, int mbRow,
        bool haveAbove, bool haveLeft,
        short[] y2RawBuf, short[] y2DqBuf, short[] y2CoeffsBuf, short[] dq16Buf, byte[] predCopyBuf,
        byte[] yAboveBuf, byte[] yLeftBuf, byte[] uAboveBuf, byte[] uLeftBuf, byte[] vAboveBuf, byte[] vLeftBuf,
        byte[] b4AboveExtBuf, byte[] b4LeftBuf)
    {
        Span<byte> yAbove = yAboveBuf;
        Span<byte> yLeft = yLeftBuf;
        Span<byte> uAbove = uAboveBuf;
        Span<byte> uLeft = uLeftBuf;
        Span<byte> vAbove = vAboveBuf;
        Span<byte> vLeft = vLeftBuf;
        Span<short> dq16 = dq16Buf;
        Span<byte> predCopy = predCopyBuf;
        Span<short> y2Coeffs = y2CoeffsBuf;

        byte yTopLeft = ReadYAboveLeft(frameBuffer, mbCol, mbRow, haveAbove, haveLeft);
        FillYAbove(frameBuffer, mbCol, mbRow, haveAbove, yAbove);
        FillYLeft(frameBuffer, mbCol, mbRow, haveLeft, yLeft);

        byte uTopLeft = ReadUvAboveLeft(frameBuffer.UPlane, frameBuffer.UvStride, mbCol, mbRow, haveAbove, haveLeft);
        byte vTopLeft = ReadUvAboveLeft(frameBuffer.VPlane, frameBuffer.UvStride, mbCol, mbRow, haveAbove, haveLeft);
        FillUvAbove(frameBuffer.UPlane, frameBuffer.UvStride, mbCol, mbRow, haveAbove, uAbove);
        FillUvLeft(frameBuffer.UPlane, frameBuffer.UvStride, mbCol, mbRow, haveLeft, uLeft);
        FillUvAbove(frameBuffer.VPlane, frameBuffer.UvStride, mbCol, mbRow, haveAbove, vAbove);
        FillUvLeft(frameBuffer.VPlane, frameBuffer.UvStride, mbCol, mbRow, haveLeft, vLeft);

        // -------- Predict + reconstruct UV (always - same for all Y modes) --------
        Vp8IntraMode16x16 uvMode = ConvertUvMode(modeInfo.UvMode);
        Vp8IntraPredictor8x8.Predict(
            uvMode, uAbove, uLeft, uTopLeft, haveAbove, haveLeft,
            frameBuffer.GetUMb(mbCol, mbRow), frameBuffer.UvStride);
        Vp8IntraPredictor8x8.Predict(
            uvMode, vAbove, vLeft, vTopLeft, haveAbove, haveLeft,
            frameBuffer.GetVMb(mbCol, mbRow), frameBuffer.UvStride);

        bool hasY2 = modeInfo.YMode != Vp8YMode.BPred;

        // -------- Dequantize + Inverse-Walsh Y2 (when present) --------
        if (hasY2)
        {
            Span<short> y2Raw = y2RawBuf;
            Span<short> y2Dq = y2DqBuf;
            qcoeff.Slice(24 * 16, 16).CopyTo(y2Raw);

            // Dequantize: slot 0 uses Y2Dc, slots 1..15 use Y2Ac.
            y2Dq[0] = (short)(y2Raw[0] * dq.Y2Dc);
            for (int i = 1; i < 16; i++) y2Dq[i] = (short)(y2Raw[i] * dq.Y2Ac);

            if (eobs[24] > 1)
            {
                Vp8InverseTransform.ShortInvWalsh4x4(y2Dq, y2Coeffs);
            }
            else
            {
                // libvpx vp8_short_inv_walsh4x4_1: when only Y2 DC is non-zero,
                // each output is (input_dc + 3) >> 3, broadcast to all 16 slots.
                short v = (short)((y2Dq[0] + 3) >> 3);
                for (int i = 0; i < 16; i++) y2Coeffs[i] = v;
            }
        }
        else
        {
            y2Coeffs.Clear();
        }

        // -------- Predict + reconstruct Y --------
        if (modeInfo.YMode == Vp8YMode.BPred)
        {
            // B_PRED path: per-sub-block predict + dequant + IDCT + add in raster order.
            // Each sub-block reads 'above' and 'left' from the recon buffer (already
            // populated by the previous sub-block / the MB to the left/above).
            Vp8IntraMode4x4[] subModes = modeInfo.SubBlockModes!;
            Span<byte> yMb = frameBuffer.GetYMb(mbCol, mbRow);
            Span<byte> b4Left = b4LeftBuf;
            Span<byte> b4AboveExt = b4AboveExtBuf;

            for (int i = 0; i < 16; i++)
            {
                int sr = i >> 2;
                int sc = i & 3;
                int dstOffset = sr * 4 * frameBuffer.YStride + sc * 4;
                Span<byte> dst = yMb.Slice(dstOffset);

                // Build above (8 bytes) and left (4 bytes) for this 4x4 sub-block.
                BuildB4Above(frameBuffer, mbCol, mbRow, sr, sc, yAbove, b4AboveExt);
                BuildB4Left(frameBuffer, mbCol, mbRow, sr, sc, yLeft, b4Left);
                byte b4TopLeft = GetB4TopLeft(frameBuffer, mbCol, mbRow, sr, sc, yTopLeft, yAbove, yLeft);
                // Place top-left sample at b4AboveExt[0] (so b4AboveExt[1..8] is the above row).
                b4AboveExt[0] = b4TopLeft;

                Vp8IntraPredictor4x4.Predict(
                    subModes[i],
                    b4AboveExt, /*aboveOffset*/ 1,
                    b4Left,
                    dst, frameBuffer.YStride);

                // Now add the residual via dequant + IDCT + clamp+add.
                Span<short> raw = qcoeff.Slice(i * 16, 16);
                dq16[0] = (short)(raw[0] * dq.Y1Dc);
                for (int j = 1; j < 16; j++) dq16[j] = (short)(raw[j] * dq.Y1Ac);

                if (eobs[i] > 0)
                {
                    // Copy 4x4 prediction into a contiguous buffer to feed IDCT.
                    for (int r = 0; r < 4; r++)
                        for (int c = 0; c < 4; c++)
                            predCopy[r * 4 + c] = dst[r * frameBuffer.YStride + c];
                    Vp8InverseTransform.ShortIdct4x4Llm(dq16, predCopy, 4, dst, frameBuffer.YStride);
                }
            }
        }
        else
        {
            // Non-B_PRED: predict the whole 16x16 luma block, then add residuals per 4x4.
            Vp8IntraMode16x16 yMode16 = ConvertYMode(modeInfo.YMode);
            Vp8IntraPredictor16x16.Predict(
                yMode16, yAbove, yLeft, yTopLeft, haveAbove, haveLeft,
                frameBuffer.GetYMb(mbCol, mbRow), frameBuffer.YStride);

            Span<byte> yMb = frameBuffer.GetYMb(mbCol, mbRow);
            for (int i = 0; i < 16; i++)
            {
                int sr = i >> 2;
                int sc = i & 3;
                int dstOffset = sr * 4 * frameBuffer.YStride + sc * 4;
                Span<byte> dst = yMb.Slice(dstOffset);

                Span<short> raw = qcoeff.Slice(i * 16, 16);
                // With Y2 present, the DC for Y4 was carried by Y2; coef decode used
                // firstCoef=1 so raw[0] is unused (always 0). We inject the inv-Walsh
                // DC into slot 0 and use Y1Dc=1 (no further dequant on it).
                dq16[0] = y2Coeffs[i];
                for (int j = 1; j < 16; j++) dq16[j] = (short)(raw[j] * dq.Y1Ac);

                bool hasResidual = eobs[i] > 0 || y2Coeffs[i] != 0;
                if (hasResidual)
                {
                    if (eobs[i] == 0 && y2Coeffs[i] != 0)
                    {
                        // DC-only fast path: only the inv-Walsh DC is present.
                        for (int r = 0; r < 4; r++)
                            for (int c = 0; c < 4; c++)
                                predCopy[r * 4 + c] = dst[r * frameBuffer.YStride + c];
                        Vp8InverseTransform.DcOnlyIdctAdd(y2Coeffs[i], predCopy, 4, dst, frameBuffer.YStride);
                    }
                    else
                    {
                        for (int r = 0; r < 4; r++)
                            for (int c = 0; c < 4; c++)
                                predCopy[r * 4 + c] = dst[r * frameBuffer.YStride + c];
                        Vp8InverseTransform.ShortIdct4x4Llm(dq16, predCopy, 4, dst, frameBuffer.YStride);
                    }
                }
            }
        }

        // -------- Dequantize + IDCT + add for each UV sub-block --------
        Span<byte> uMb = frameBuffer.GetUMb(mbCol, mbRow);
        Span<byte> vMb = frameBuffer.GetVMb(mbCol, mbRow);
        for (int i = 16; i < 24; i++)
        {
            int blkInPlane = i - 16; // 0..7
            bool isV = blkInPlane >= 4;
            int b = isV ? blkInPlane - 4 : blkInPlane;
            int sr = b >> 1;
            int sc = b & 1;
            Span<byte> chromaPlane = isV ? vMb : uMb;
            int dstOffset = sr * 4 * frameBuffer.UvStride + sc * 4;
            Span<byte> dst = chromaPlane.Slice(dstOffset);

            Span<short> raw = qcoeff.Slice(i * 16, 16);
            dq16[0] = (short)(raw[0] * dq.UvDc);
            for (int j = 1; j < 16; j++) dq16[j] = (short)(raw[j] * dq.UvAc);

            if (eobs[i] > 0)
            {
                if (eobs[i] == 1)
                {
                    // DC-only fast path.
                    for (int r = 0; r < 4; r++)
                        for (int c = 0; c < 4; c++)
                            predCopy[r * 4 + c] = dst[r * frameBuffer.UvStride + c];
                    Vp8InverseTransform.DcOnlyIdctAdd(dq16[0], predCopy, 4, dst, frameBuffer.UvStride);
                }
                else
                {
                    for (int r = 0; r < 4; r++)
                        for (int c = 0; c < 4; c++)
                            predCopy[r * 4 + c] = dst[r * frameBuffer.UvStride + c];
                    Vp8InverseTransform.ShortIdct4x4Llm(dq16, predCopy, 4, dst, frameBuffer.UvStride);
                }
            }
        }
    }

    // ---------------- 4x4 sub-block above + left helpers ----------------

    /// <summary>
    /// Build the 8-byte above-row buffer (slots 0..7) for a 4x4 sub-block at
    /// (sr, sc) within the current MB. The first 4 bytes are above[0..3] (the
    /// row directly above the sub-block); the next 4 bytes are above[4..7]
    /// (the above-right sub-block's row). For sub-blocks at sc == 3 (right
    /// edge of MB), the above-right is the next MB's above, which for the
    /// rightmost MB column doesn't exist - libvpx handles this with
    /// intra_prediction_down_copy: the above-right of the entire MB is filled
    /// by replicating the above row's last byte (mbAboveRow[15]) into rows
    /// 4, 8, 12 above the current MB. We replicate that here.
    /// </summary>
    private static void BuildB4Above(
        Vp8FrameBuffer fb, int mbCol, int mbRow, int sr, int sc,
        Span<byte> mbAboveRow, Span<byte> dstAboveExt /* >= 9 bytes; [0]=top-left, [1..8]=above */)
    {
        Span<byte> dstAbove = dstAboveExt.Slice(1, 8);
        // Build the 4 above pixels (just-above the sub-block).
        if (sr == 0)
        {
            // Top of MB - read from mbAboveRow at this sub-col (which is the row above the MB).
            int colStart = sc * 4;
            for (int i = 0; i < 4; i++) dstAbove[i] = mbAboveRow[colStart + i];
        }
        else
        {
            // Inside MB - read from the recon buffer one row up.
            Span<byte> yMb = fb.GetYMb(mbCol, mbRow);
            int rowAbove = sr * 4 - 1;
            int colStart = sc * 4;
            for (int i = 0; i < 4; i++) dstAbove[i] = yMb[rowAbove * fb.YStride + colStart + i];
        }

        // Build the 4 above-right pixels.
        // VP8 quirk: for sub-blocks at (sr=1,2,3, sc=3) - the rightmost column of the MB except top -
        // the "above-right" is NOT the next sub-block's top row (which doesn't exist yet); instead
        // libvpx uses intra_prediction_down_copy to provide the *MB* above-right row's bytes.
        // For top row (sr=0, sc=3), the above-right comes from the next MB's above row.
        // For middle/bottom (sr>=1, sc=3), the above-right is the MB above-right, fed via down-copy.
        if (sc < 3)
        {
            // Above-right is just the next sub-col's above row.
            if (sr == 0)
            {
                int colStart = (sc + 1) * 4;
                // Watch for the right edge of the MB (sc == 3 case is handled below; for sc < 3 we're fine).
                for (int i = 0; i < 4; i++) dstAbove[4 + i] = mbAboveRow[colStart + i];
            }
            else
            {
                Span<byte> yMb = fb.GetYMb(mbCol, mbRow);
                int rowAbove = sr * 4 - 1;
                int colStart = (sc + 1) * 4;
                for (int i = 0; i < 4; i++) dstAbove[4 + i] = yMb[rowAbove * fb.YStride + colStart + i];
            }
        }
        else
        {
            // sc == 3: rightmost column of the MB.
            if (sr == 0)
            {
                // Top row - above-right is the NEXT MB's above row's first 4 bytes.
                // For the rightmost MB column there is no next MB; libvpx fills with the above row's last byte.
                if (mbCol == fb.MbCols - 1 || mbRow == 0)
                {
                    byte fill = mbRow == 0 ? Vp8IntraEdgeFill.AboveDefault : mbAboveRow[15];
                    for (int i = 0; i < 4; i++) dstAbove[4 + i] = fill;
                }
                else
                {
                    // Read from the next MB's above row (one row up, columns 16..19).
                    int rowAbove = mbRow * 16 - 1;
                    int colStart = (mbCol + 1) * 16;
                    for (int i = 0; i < 4; i++) dstAbove[4 + i] = fb.YPlane[rowAbove * fb.YStride + colStart + i];
                }
            }
            else
            {
                // Middle/bottom of MB at right edge: libvpx uses the down-copied 4 bytes
                // from the MB above-right row (or the MB above's own above-right).
                // We replicate that from mbAboveRow's last 4 bytes (cols 12..15) since
                // intra_prediction_down_copy replicates above_right_src into rows 4,8,12.
                // For mbCol == last col, the above-right doesn't exist; use mbAboveRow[15].
                if (mbRow == 0)
                {
                    for (int i = 0; i < 4; i++) dstAbove[4 + i] = Vp8IntraEdgeFill.AboveDefault;
                }
                else if (mbCol == fb.MbCols - 1)
                {
                    byte fill = mbAboveRow[15];
                    for (int i = 0; i < 4; i++) dstAbove[4 + i] = fill;
                }
                else
                {
                    // Read 4 bytes from row above the MB at cols 16..19 (the MB above-right's row).
                    int rowAbove = mbRow * 16 - 1;
                    int colStart = (mbCol + 1) * 16;
                    for (int i = 0; i < 4; i++) dstAbove[4 + i] = fb.YPlane[rowAbove * fb.YStride + colStart + i];
                }
            }
        }
    }

    private static void BuildB4Left(
        Vp8FrameBuffer fb, int mbCol, int mbRow, int sr, int sc,
        Span<byte> mbLeftCol, Span<byte> dstLeft /* 4 bytes */)
    {
        if (sc == 0)
        {
            // Left edge of MB - read from mbLeftCol at this sub-row.
            int rowStart = sr * 4;
            for (int i = 0; i < 4; i++) dstLeft[i] = mbLeftCol[rowStart + i];
        }
        else
        {
            // Inside MB - read from recon buffer one column to the left.
            Span<byte> yMb = fb.GetYMb(mbCol, mbRow);
            int colLeft = sc * 4 - 1;
            int rowStart = sr * 4;
            for (int i = 0; i < 4; i++) dstLeft[i] = yMb[(rowStart + i) * fb.YStride + colLeft];
        }
    }

    private static byte GetB4TopLeft(
        Vp8FrameBuffer fb, int mbCol, int mbRow, int sr, int sc,
        byte mbTopLeft, Span<byte> mbAboveRow, Span<byte> mbLeftCol)
    {
        if (sr == 0 && sc == 0) return mbTopLeft;
        if (sr == 0)
        {
            // Top of MB, not left edge - top-left is the byte one above-left, reading from mbAboveRow.
            return mbAboveRow[sc * 4 - 1];
        }
        if (sc == 0)
        {
            // Left of MB, not top edge - top-left is the byte one above-left, reading from mbLeftCol.
            return mbLeftCol[sr * 4 - 1];
        }
        // Inside MB - read from recon buffer.
        Span<byte> yMb = fb.GetYMb(mbCol, mbRow);
        return yMb[(sr * 4 - 1) * fb.YStride + sc * 4 - 1];
    }

    // ---------------- Edge sample helpers ----------------

    private static byte ReadYAboveLeft(Vp8FrameBuffer fb, int mbCol, int mbRow, bool haveAbove, bool haveLeft)
    {
        if (!haveAbove) return Vp8IntraEdgeFill.AboveDefault;
        if (!haveLeft) return Vp8IntraEdgeFill.LeftDefault;
        return fb.YPlane[(mbRow * 16 - 1) * fb.YStride + (mbCol * 16 - 1)];
    }

    private static byte ReadUvAboveLeft(byte[] plane, int stride, int mbCol, int mbRow, bool haveAbove, bool haveLeft)
    {
        if (!haveAbove) return Vp8IntraEdgeFill.AboveDefault;
        if (!haveLeft) return Vp8IntraEdgeFill.LeftDefault;
        return plane[(mbRow * 8 - 1) * stride + (mbCol * 8 - 1)];
    }

    private static void FillYAbove(Vp8FrameBuffer fb, int mbCol, int mbRow, bool haveAbove, Span<byte> dst)
    {
        if (!haveAbove)
        {
            dst.Slice(0, 16).Fill(Vp8IntraEdgeFill.AboveDefault);
            return;
        }
        int srcRow = mbRow * 16 - 1;
        int srcCol = mbCol * 16;
        for (int i = 0; i < 16; i++) dst[i] = fb.YPlane[srcRow * fb.YStride + srcCol + i];
    }

    private static void FillYLeft(Vp8FrameBuffer fb, int mbCol, int mbRow, bool haveLeft, Span<byte> dst)
    {
        if (!haveLeft)
        {
            dst.Slice(0, 16).Fill(Vp8IntraEdgeFill.LeftDefault);
            return;
        }
        int srcCol = mbCol * 16 - 1;
        int srcRow = mbRow * 16;
        for (int i = 0; i < 16; i++) dst[i] = fb.YPlane[(srcRow + i) * fb.YStride + srcCol];
    }

    private static void FillUvAbove(byte[] plane, int stride, int mbCol, int mbRow, bool haveAbove, Span<byte> dst)
    {
        if (!haveAbove)
        {
            dst.Slice(0, 8).Fill(Vp8IntraEdgeFill.AboveDefault);
            return;
        }
        int srcRow = mbRow * 8 - 1;
        int srcCol = mbCol * 8;
        for (int i = 0; i < 8; i++) dst[i] = plane[srcRow * stride + srcCol + i];
    }

    private static void FillUvLeft(byte[] plane, int stride, int mbCol, int mbRow, bool haveLeft, Span<byte> dst)
    {
        if (!haveLeft)
        {
            dst.Slice(0, 8).Fill(Vp8IntraEdgeFill.LeftDefault);
            return;
        }
        int srcCol = mbCol * 8 - 1;
        int srcRow = mbRow * 8;
        for (int i = 0; i < 8; i++) dst[i] = plane[(srcRow + i) * stride + srcCol];
    }

    // ---------------- Sub-mode context resolution (RFC 6386 + libvpx findnearmv.h) ----------------

    private static Vp8IntraMode4x4 AboveModeFromMbAbove(
        bool haveAbove, Vp8YMode mbAboveMode,
        Vp8IntraMode4x4[] aboveRowSubModes, int mbCol, int subCol)
    {
        if (!haveAbove) return Vp8IntraMode4x4.BDcPred;
        // libvpx above_block_mode: when above MB had B_PRED, look up its bmi[12 + subCol].
        if (mbAboveMode == Vp8YMode.BPred) return aboveRowSubModes[mbCol * 4 + subCol];
        return MapMbModeToSubMode(mbAboveMode);
    }

    private static Vp8IntraMode4x4 LeftModeFromMbLeft(
        bool haveLeft, Vp8YMode mbLeftMode,
        Vp8IntraMode4x4[] leftSubModes, int subRow)
    {
        if (!haveLeft) return Vp8IntraMode4x4.BDcPred;
        if (mbLeftMode == Vp8YMode.BPred) return leftSubModes[subRow];
        return MapMbModeToSubMode(mbLeftMode);
    }

    private static Vp8IntraMode4x4 MapMbModeToSubMode(Vp8YMode mode) => mode switch
    {
        Vp8YMode.DcPred => Vp8IntraMode4x4.BDcPred,
        Vp8YMode.VPred => Vp8IntraMode4x4.BVePred,
        Vp8YMode.HPred => Vp8IntraMode4x4.BHePred,
        Vp8YMode.TmPred => Vp8IntraMode4x4.BTmPred,
        _ => Vp8IntraMode4x4.BDcPred,
    };

    private static Vp8IntraMode16x16 ConvertYMode(Vp8YMode m) => m switch
    {
        Vp8YMode.DcPred => Vp8IntraMode16x16.DcPred,
        Vp8YMode.VPred => Vp8IntraMode16x16.VPred,
        Vp8YMode.HPred => Vp8IntraMode16x16.HPred,
        Vp8YMode.TmPred => Vp8IntraMode16x16.TmPred,
        _ => throw new ArgumentOutOfRangeException(nameof(m), $"Unsupported 16x16 Y mode {m}"),
    };

    private static Vp8IntraMode16x16 ConvertUvMode(Vp8UvMode m) => m switch
    {
        Vp8UvMode.DcPred => Vp8IntraMode16x16.DcPred,
        Vp8UvMode.VPred => Vp8IntraMode16x16.VPred,
        Vp8UvMode.HPred => Vp8IntraMode16x16.HPred,
        Vp8UvMode.TmPred => Vp8IntraMode16x16.TmPred,
        _ => throw new ArgumentOutOfRangeException(nameof(m), $"Unsupported UV mode {m}"),
    };
}
