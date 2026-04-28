// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 keyframe encoder. Takes a YUV420 source frame and emits a valid
// VP8 IVF-payload keyframe that libvpx / ffmpeg can decode.
//
// Simplifications (v1):
//   - All MBs use Y_PRED = DC_PRED, UV_PRED = DC_PRED (no mode selection)
//   - No segmentation
//   - Single token partition (Log2NumPartitions = 0)
//   - Loop filter disabled (filter_level = 0)
//   - mb_no_skip_coeff disabled
//
// More sophisticated mode selection / RD-optimized quantization /
// loop filtering can layer on top of this; the bitstream produced is
// already a fully-valid VP8 keyframe.

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>VP8 keyframe encoder (DC-prediction-only, single partition, no LF).</summary>
public static class Vp8KeyframeEncoder
{
    /// <summary>
    /// Encode a single VP8 keyframe from YUV420 source.
    /// </summary>
    /// <param name="ySrc">Y plane bytes (rowStride * height).</param>
    /// <param name="uSrc">U plane bytes (rowStride/2 * height/2).</param>
    /// <param name="vSrc">V plane bytes (rowStride/2 * height/2).</param>
    /// <param name="width">Frame width in pixels (multiple of 16 for v1).</param>
    /// <param name="height">Frame height in pixels (multiple of 16 for v1).</param>
    /// <param name="ySrcStride">Y plane row stride in bytes.</param>
    /// <param name="uvSrcStride">UV plane row stride in bytes.</param>
    /// <param name="baseQIndex">Base quantizer index 0..127 (lower = higher quality).</param>
    /// <returns>Complete VP8 frame bytes ready to wrap in IVF or webm.</returns>
    public static byte[] EncodeKeyFrame(
        ReadOnlySpan<byte> ySrc, int ySrcStride,
        ReadOnlySpan<byte> uSrc, int uvSrcStride,
        ReadOnlySpan<byte> vSrc,
        int width, int height,
        int baseQIndex = 30)
    {
        if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException();
        if ((width & 15) != 0 || (height & 15) != 0)
            throw new ArgumentException("Width and height must be multiples of 16 (v1)");
        if (baseQIndex < 0 || baseQIndex > 127)
            throw new ArgumentOutOfRangeException(nameof(baseQIndex),
                "VP8 BaseQIndex is a 7-bit field (0..127); larger values wrap modulo 128 in the bitstream " +
                "while the encoder still uses the original quantizer, producing decode garbage.");

        int mbRows = height / 16;
        int mbCols = width / 16;

        // Per-frame state.
        var dequant = Vp8MbDequantizer.Compute(0,
            new Vp8QuantizerIndices
            {
                BaseQIndex = baseQIndex,
                Y1DcDeltaQ = 0, Y2DcDeltaQ = 0, Y2AcDeltaQ = 0, UvDcDeltaQ = 0, UvAcDeltaQ = 0,
            },
            new Vp8SegmentationParams
            {
                Enabled = false, UpdateMap = false, UpdateData = false, AbsDelta = false,
                FeatureData = new int[2, 4],
                SegmentTreeProbs = new byte[3] { 255, 255, 255 },
            });

        var recon = new Vp8FrameBuffer(width, height);
        var entropyContexts = new Vp8EntropyContexts(mbCols);

        // Encode the first partition (frame header + mode info) into one
        // bool encoder, then the second partition (coefficient tokens) into
        // another. Concatenate at the end.
        var partition0 = new Vp8BoolEncoder();
        var tokenPartition = new Vp8BoolEncoder();

        // Build frame header.
        var hdr = new Vp8FrameHeader
        {
            ColorSpace = 0,
            ClampingType = 0,
            Segmentation = new Vp8SegmentationParams
            {
                Enabled = false, UpdateMap = false, UpdateData = false, AbsDelta = false,
                FeatureData = new int[2, 4],
                SegmentTreeProbs = new byte[3] { 255, 255, 255 },
            },
            LoopFilter = new Vp8LoopFilterParams
            {
                FilterType = 0, FilterLevel = 0, SharpnessLevel = 0,
                ModeRefLfDeltaEnabled = false,
                RefLfDeltas = new int[4], ModeLfDeltas = new int[4],
            },
            Log2NumPartitions = 0,
            Quantizer = new Vp8QuantizerIndices
            {
                BaseQIndex = baseQIndex,
                Y1DcDeltaQ = 0, Y2DcDeltaQ = 0, Y2AcDeltaQ = 0, UvDcDeltaQ = 0, UvAcDeltaQ = 0,
            },
            RefreshEntropyProbs = true,
            CoefProbs = (byte[,,,])Vp8DefaultCoefProbs.DefaultProbs.Clone(),
            MbNoSkipCoeffEnabled = false,
            ProbSkipFalse = 0,
        };
        Vp8FrameHeaderWriter.WriteKeyFrameHeader(partition0, hdr);

        // Slice the coef probs to a 3D shape per block type.
        var coefProbsByType = new byte[Vp8DefaultCoefProbs.BlockTypes][,,];
        for (int t = 0; t < Vp8DefaultCoefProbs.BlockTypes; t++)
        {
            var slice = new byte[Vp8DefaultCoefProbs.CoefBands, Vp8DefaultCoefProbs.PrevCoefContexts, Vp8DefaultCoefProbs.EntropyNodes];
            for (int b = 0; b < Vp8DefaultCoefProbs.CoefBands; b++)
                for (int c = 0; c < Vp8DefaultCoefProbs.PrevCoefContexts; c++)
                    for (int n = 0; n < Vp8DefaultCoefProbs.EntropyNodes; n++)
                        slice[b, c, n] = hdr.CoefProbs[t, b, c, n];
            coefProbsByType[t] = slice;
        }

        // Per-MB iteration. Mode info goes into partition0; coef tokens into tokenPartition.
        for (int mbRow = 0; mbRow < mbRows; mbRow++)
        {
            entropyContexts.ClearLeft();
            for (int mbCol = 0; mbCol < mbCols; mbCol++)
            {
                EncodeMb(
                    mbRow, mbCol, mbCols,
                    ySrc, ySrcStride, uSrc, vSrc, uvSrcStride,
                    recon,
                    dequant, coefProbsByType,
                    entropyContexts,
                    partition0, tokenPartition);
            }
        }

        // Finalize both partitions.
        var partition0Bytes = partition0.Stop();
        var tokenBytes = tokenPartition.Stop();

        // Build the frame: tag + partition0 + tokenPartition.
        var tag = new Vp8FrameTag
        {
            IsKeyFrame = true,
            Version = Vp8Version.Bicubic,
            ShowFrame = true,
            FirstPartitionSize = partition0Bytes.Length,
            Width = width, Height = height,
            HorizontalScale = 0, VerticalScale = 0,
        };
        var tagBytes = Vp8FrameTagWriter.WriteTag(tag);

        var output = new byte[tagBytes.Length + partition0Bytes.Length + tokenBytes.Length];
        Buffer.BlockCopy(tagBytes, 0, output, 0, tagBytes.Length);
        Buffer.BlockCopy(partition0Bytes, 0, output, tagBytes.Length, partition0Bytes.Length);
        Buffer.BlockCopy(tokenBytes, 0, output, tagBytes.Length + partition0Bytes.Length, tokenBytes.Length);
        return output;
    }

    private static void EncodeMb(
        int mbRow, int mbCol, int mbCols,
        ReadOnlySpan<byte> ySrc, int ySrcStride,
        ReadOnlySpan<byte> uSrc, ReadOnlySpan<byte> vSrc, int uvSrcStride,
        Vp8FrameBuffer recon,
        Vp8MbDequant dequant,
        byte[][,,] coefProbsByType,
        Vp8EntropyContexts contexts,
        Vp8BoolEncoder partition0,
        Vp8BoolEncoder tokenPartition)
    {
        // === 1. Predict Y 16x16 with DC_PRED ===
        // Read above/left from recon. For first row/col, use 127/129 fills.
        Span<byte> yAbove = stackalloc byte[16];
        Span<byte> yLeft = stackalloc byte[16];
        bool haveAbove = mbRow > 0;
        bool haveLeft = mbCol > 0;
        if (haveAbove)
        {
            for (int c = 0; c < 16; c++)
                yAbove[c] = recon.YPlane[(mbRow * 16 - 1) * recon.YStride + mbCol * 16 + c];
        }
        else Vp8IntraEdgeFill.FillAboveRow16(yAbove);
        if (haveLeft)
        {
            for (int r = 0; r < 16; r++)
                yLeft[r] = recon.YPlane[(mbRow * 16 + r) * recon.YStride + mbCol * 16 - 1];
        }
        else Vp8IntraEdgeFill.FillLeftColumn16(yLeft);

        Span<byte> yPredMb = stackalloc byte[16 * 16];
        Vp8IntraPredictor16x16.Predict(
            Vp8IntraMode16x16.DcPred, yAbove, yLeft, 0, haveAbove, haveLeft, yPredMb, 16);

        // === 2. Compute residual + forward DCT for each Y4 block ===
        // Hold all 16 transformed quantized blocks + the 16 Y4-DC values.
        Span<short> y4Coefs = stackalloc short[16 * 16]; // [Y4 block index * 16 + raster pos]
        Span<short> y2DcVals = stackalloc short[16];
        // Hoist per-iteration scratch out of the loop (CA2014).
        Span<short> residual = stackalloc short[16];
        Span<short> coefs = stackalloc short[16];

        for (int by = 0; by < 4; by++)
        {
            for (int bx = 0; bx < 4; bx++)
            {
                int blockIdx = by * 4 + bx;
                for (int r = 0; r < 4; r++)
                {
                    for (int c = 0; c < 4; c++)
                    {
                        int srcOff = (mbRow * 16 + by * 4 + r) * ySrcStride + mbCol * 16 + bx * 4 + c;
                        int predOff = (by * 4 + r) * 16 + bx * 4 + c;
                        residual[r * 4 + c] = (short)(ySrc[srcOff] - yPredMb[predOff]);
                    }
                }
                Vp8ForwardTransform.ShortFdct4x4(residual, 4, coefs);
                // Save the DC value for Y2.
                y2DcVals[blockIdx] = coefs[0];
                // Y4 stores AC only (DC will come from Y2 inverse).
                coefs[0] = 0;
                // Quantize Y4 AC values.
                Vp8ForwardQuantizer.QuantizeY1Block(coefs, dequant);
                for (int i = 0; i < 16; i++) y4Coefs[blockIdx * 16 + i] = coefs[i];
            }
        }

        // === 3. Forward Walsh-Hadamard the 16 Y2 DCs, quantize ===
        Span<short> y2Coefs = stackalloc short[16];
        Vp8ForwardTransform.ShortWalsh4x4(y2DcVals, 4, y2Coefs);
        Vp8ForwardQuantizer.QuantizeY2Block(y2Coefs, dequant);

        // === 4. UV planes ===
        Span<byte> uAbove = stackalloc byte[8];
        Span<byte> uLeft = stackalloc byte[8];
        Span<byte> vAbove = stackalloc byte[8];
        Span<byte> vLeft = stackalloc byte[8];

        if (haveAbove)
        {
            for (int c = 0; c < 8; c++)
            {
                uAbove[c] = recon.UPlane[(mbRow * 8 - 1) * recon.UvStride + mbCol * 8 + c];
                vAbove[c] = recon.VPlane[(mbRow * 8 - 1) * recon.UvStride + mbCol * 8 + c];
            }
        }
        else { Vp8IntraEdgeFill.FillAboveRow8(uAbove); Vp8IntraEdgeFill.FillAboveRow8(vAbove); }

        if (haveLeft)
        {
            for (int r = 0; r < 8; r++)
            {
                uLeft[r] = recon.UPlane[(mbRow * 8 + r) * recon.UvStride + mbCol * 8 - 1];
                vLeft[r] = recon.VPlane[(mbRow * 8 + r) * recon.UvStride + mbCol * 8 - 1];
            }
        }
        else { Vp8IntraEdgeFill.FillLeftColumn8(uLeft); Vp8IntraEdgeFill.FillLeftColumn8(vLeft); }

        Span<byte> uPredMb = stackalloc byte[8 * 8];
        Span<byte> vPredMb = stackalloc byte[8 * 8];
        Vp8IntraPredictor8x8.Predict(Vp8IntraMode16x16.DcPred, uAbove, uLeft, 0, haveAbove, haveLeft, uPredMb, 8);
        Vp8IntraPredictor8x8.Predict(Vp8IntraMode16x16.DcPred, vAbove, vLeft, 0, haveAbove, haveLeft, vPredMb, 8);

        Span<short> uCoefs = stackalloc short[4 * 16];
        Span<short> vCoefs = stackalloc short[4 * 16];
        for (int by = 0; by < 2; by++)
        {
            for (int bx = 0; bx < 2; bx++)
            {
                int blockIdx = by * 2 + bx;
                EncodeUvBlock(uSrc, uvSrcStride, uPredMb, mbRow, mbCol, by, bx, dequant, uCoefs.Slice(blockIdx * 16, 16));
                EncodeUvBlock(vSrc, uvSrcStride, vPredMb, mbRow, mbCol, by, bx, dequant, vCoefs.Slice(blockIdx * 16, 16));
            }
        }

        // === 5. Encode mode info into partition0 ===
        // Y mode = DC_PRED (4 in inter ordering, 0 in keyframe ordering).
        int yModeLeaf = (int)Vp8YMode.DcPred;
        EncodeYModeKf(partition0, yModeLeaf);
        EncodeUvMode(partition0, (int)Vp8UvMode.DcPred);

        // === 6. Encode coefficients into tokenPartition ===
        // VP8 PLANE_TYPE constants (libvpx vp8/common/blockd.h):
        //   PLANE_TYPE_Y_NO_DC    = 0  (Y4 when Y2 present, firstCoef=1)
        //   PLANE_TYPE_Y2         = 1  (the 16 Y4-DC values via Walsh)
        //   PLANE_TYPE_UV         = 2
        //   PLANE_TYPE_Y_WITH_DC  = 3  (Y4 when Y2 absent / B_PRED, firstCoef=0)
        // Y2 first (block type 1).
        var aboveCtx = contexts.GetAbove(mbCol);
        int y2Ctx = aboveCtx[Vp8EntropyContexts.Plane.Y2Slot] + contexts.Left[Vp8EntropyContexts.Plane.Y2Slot];
        int y2Eob = Vp8CoefBlockEncoder.Encode(tokenPartition, coefProbsByType[1], y2Ctx, 0, y2Coefs);
        byte y2HasCoef = (byte)(y2Eob > 0 ? 1 : 0);
        aboveCtx[Vp8EntropyContexts.Plane.Y2Slot] = y2HasCoef;
        contexts.Left[Vp8EntropyContexts.Plane.Y2Slot] = y2HasCoef;

        // Y4 (block type 0 = PLANE_TYPE_Y_NO_DC, firstCoef=1 since Y2 carries DC).
        for (int by = 0; by < 4; by++)
        {
            for (int bx = 0; bx < 4; bx++)
            {
                int blockIdx = by * 4 + bx;
                int aboveSlot = Vp8EntropyContexts.Plane.YBase + bx;
                int leftSlot = Vp8EntropyContexts.Plane.YBase + by;
                int ctx = aboveCtx[aboveSlot] + contexts.Left[leftSlot];
                int eob = Vp8CoefBlockEncoder.Encode(tokenPartition, coefProbsByType[0], ctx, 1, y4Coefs.Slice(blockIdx * 16, 16));
                // Match decoder: any non-zero coefficient flips the context to 1.
                // For firstCoef=1, Encode returns the EOB position which is >=1 if any AC coef
                // is non-zero (eob == 0 means "block emitted EOB at position firstCoef = 1, no coefs").
                // Use eob > firstCoef test: equivalent to "any coef present".
                byte hasCoef = (byte)(eob > 0 ? 1 : 0);
                aboveCtx[aboveSlot] = hasCoef;
                contexts.Left[leftSlot] = hasCoef;
            }
        }

        // U (block type 2).
        for (int by = 0; by < 2; by++)
        {
            for (int bx = 0; bx < 2; bx++)
            {
                int blockIdx = by * 2 + bx;
                int aboveSlot = Vp8EntropyContexts.Plane.UBase + bx;
                int leftSlot = Vp8EntropyContexts.Plane.UBase + by;
                int ctx = aboveCtx[aboveSlot] + contexts.Left[leftSlot];
                int eob = Vp8CoefBlockEncoder.Encode(tokenPartition, coefProbsByType[2], ctx, 0, uCoefs.Slice(blockIdx * 16, 16));
                byte hasCoef = (byte)(eob > 0 ? 1 : 0);
                aboveCtx[aboveSlot] = hasCoef;
                contexts.Left[leftSlot] = hasCoef;
            }
        }
        // V (block type 2).
        for (int by = 0; by < 2; by++)
        {
            for (int bx = 0; bx < 2; bx++)
            {
                int blockIdx = by * 2 + bx;
                int aboveSlot = Vp8EntropyContexts.Plane.VBase + bx;
                int leftSlot = Vp8EntropyContexts.Plane.VBase + by;
                int ctx = aboveCtx[aboveSlot] + contexts.Left[leftSlot];
                int eob = Vp8CoefBlockEncoder.Encode(tokenPartition, coefProbsByType[2], ctx, 0, vCoefs.Slice(blockIdx * 16, 16));
                byte hasCoef = (byte)(eob > 0 ? 1 : 0);
                aboveCtx[aboveSlot] = hasCoef;
                contexts.Left[leftSlot] = hasCoef;
            }
        }

        // === 7. Reconstruct: dequantize, inverse transform, write back to recon ===
        //
        // The encoder MUST reproduce exactly what the decoder will reconstruct
        // for this MB - subsequent MBs predict from these recon pixels, and
        // any drift between encoder and decoder views compounds across the frame.
        //
        // Steps mirror the decoder (Vp8KeyframeWalker.ReconstructMb):
        //   - Dequantize Y2 -> inverse Walsh (or DC-only fast path)
        //   - For each Y4 block: dequantize AC, inject Y2 inverse-derived DC,
        //     IDCT, add to predictor, clamp, write recon
        //   - For each UV block: dequantize, IDCT, add to UV predictor, write recon

        // ---- Y2 dequant + inverse Walsh ----
        Span<short> y2Dq = stackalloc short[16];
        Span<short> y2Inv = stackalloc short[16];
        // Dequantize Y2: slot 0 uses Y2Dc, slots 1..15 use Y2Ac.
        y2Dq[0] = (short)(y2Coefs[0] * dequant.Y2Dc);
        for (int i = 1; i < 16; i++) y2Dq[i] = (short)(y2Coefs[i] * dequant.Y2Ac);
        // Match decoder's two paths: full Walsh when AC present, broadcast DC otherwise.
        if (y2Eob > 1)
        {
            Vp8InverseTransform.ShortInvWalsh4x4(y2Dq, y2Inv);
        }
        else
        {
            // libvpx vp8_short_inv_walsh4x4_1: when only Y2 DC is non-zero,
            // each output is (input_dc + 3) >> 3, broadcast to all 16 slots.
            short v = (short)((y2Dq[0] + 3) >> 3);
            for (int i = 0; i < 16; i++) y2Inv[i] = v;
        }

        // ---- Per-Y4 block: dequantize + IDCT + add to predictor ----
        Span<short> dq16 = stackalloc short[16];
        Span<byte> predCopy = stackalloc byte[16];
        for (int by = 0; by < 4; by++)
        {
            for (int bx = 0; bx < 4; bx++)
            {
                int blockIdx = by * 4 + bx;
                Span<short> raw = y4Coefs.Slice(blockIdx * 16, 16);
                // Y4 carries DC=0 at slot 0 (cleared after fdct); decoder injects
                // y2Inv[blockIdx] there and uses the rest as AC.
                dq16[0] = y2Inv[blockIdx];
                for (int i = 1; i < 16; i++) dq16[i] = (short)(raw[i] * dequant.Y1Ac);

                // Build the 4x4 predictor patch from the per-MB pred buffer.
                for (int r = 0; r < 4; r++)
                    for (int c = 0; c < 4; c++)
                        predCopy[r * 4 + c] = yPredMb[(by * 4 + r) * 16 + bx * 4 + c];

                // Recon write target: directly into the frame buffer at this MB.
                int reconOff = (mbRow * 16 + by * 4) * recon.YStride + mbCol * 16 + bx * 4;
                Span<byte> reconDst = recon.YPlane.AsSpan(reconOff);

                // Decode-side has a DC-only fast path when eobs[i]==0 && y2Inv!=0.
                // For correctness we don't need that branch on the encoder side -
                // the full IDCT with AC=0 and DC=y2Inv produces the same result
                // as DcOnlyIdctAdd. Use the DC-only fast path when AC is empty
                // to match the decoder path bit-for-bit (avoids any rounding drift).
                bool acAllZero = true;
                for (int i = 1; i < 16; i++) { if (dq16[i] != 0) { acAllZero = false; break; } }
                if (acAllZero && y2Inv[blockIdx] == 0)
                {
                    // No residual at all - just write the predictor.
                    for (int r = 0; r < 4; r++)
                        for (int c = 0; c < 4; c++)
                            reconDst[r * recon.YStride + c] = predCopy[r * 4 + c];
                }
                else if (acAllZero)
                {
                    Vp8InverseTransform.DcOnlyIdctAdd(y2Inv[blockIdx], predCopy, 4, reconDst, recon.YStride);
                }
                else
                {
                    Vp8InverseTransform.ShortIdct4x4Llm(dq16, predCopy, 4, reconDst, recon.YStride);
                }
            }
        }

        // ---- Per-UV block: dequantize + IDCT + add to predictor ----
        for (int by = 0; by < 2; by++)
        {
            for (int bx = 0; bx < 2; bx++)
            {
                int blockIdx = by * 2 + bx;
                ReconUvBlock(uCoefs.Slice(blockIdx * 16, 16), uPredMb, by, bx, dequant.UvDc, dequant.UvAc,
                             recon.UPlane, recon.UvStride, mbRow, mbCol, predCopy, dq16);
                ReconUvBlock(vCoefs.Slice(blockIdx * 16, 16), vPredMb, by, bx, dequant.UvDc, dequant.UvAc,
                             recon.VPlane, recon.UvStride, mbRow, mbCol, predCopy, dq16);
            }
        }
    }

    /// <summary>
    /// Dequantize a single 4x4 UV block, IDCT it, add to the predictor patch,
    /// write into the recon plane. Used by the encoder reconstruction step.
    /// </summary>
    private static void ReconUvBlock(
        Span<short> raw, ReadOnlySpan<byte> uvPred, int by, int bx,
        int uvDc, int uvAc,
        byte[] reconPlane, int uvStride, int mbRow, int mbCol,
        Span<byte> predCopy, Span<short> dq16)
    {
        dq16[0] = (short)(raw[0] * uvDc);
        for (int i = 1; i < 16; i++) dq16[i] = (short)(raw[i] * uvAc);

        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
                predCopy[r * 4 + c] = uvPred[(by * 4 + r) * 8 + bx * 4 + c];

        int reconOff = (mbRow * 8 + by * 4) * uvStride + mbCol * 8 + bx * 4;
        Span<byte> dst = reconPlane.AsSpan(reconOff);

        bool acAllZero = true;
        for (int i = 1; i < 16; i++) { if (dq16[i] != 0) { acAllZero = false; break; } }
        if (acAllZero && dq16[0] == 0)
        {
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                    dst[r * uvStride + c] = predCopy[r * 4 + c];
        }
        else if (acAllZero)
        {
            Vp8InverseTransform.DcOnlyIdctAdd(dq16[0], predCopy, 4, dst, uvStride);
        }
        else
        {
            Vp8InverseTransform.ShortIdct4x4Llm(dq16, predCopy, 4, dst, uvStride);
        }
    }

    private static void EncodeUvBlock(
        ReadOnlySpan<byte> src, int srcStride,
        ReadOnlySpan<byte> pred,
        int mbRow, int mbCol, int by, int bx,
        Vp8MbDequant dequant,
        Span<short> outCoefs)
    {
        Span<short> residual = stackalloc short[16];
        for (int r = 0; r < 4; r++)
        {
            for (int c = 0; c < 4; c++)
            {
                int srcOff = (mbRow * 8 + by * 4 + r) * srcStride + mbCol * 8 + bx * 4 + c;
                int predOff = (by * 4 + r) * 8 + bx * 4 + c;
                residual[r * 4 + c] = (short)(src[srcOff] - pred[predOff]);
            }
        }
        Vp8ForwardTransform.ShortFdct4x4(residual, 4, outCoefs);
        Vp8ForwardQuantizer.QuantizeUvBlock(outCoefs, dequant);
    }

    /// <summary>Encode Y mode for keyframe: kf_ymode_tree walk, leaves are DcPred=0/VPred=1/HPred=2/TmPred=3/BPred=4.</summary>
    private static void EncodeYModeKf(Vp8BoolEncoder writer, int yMode)
    {
        // KfYModeTree shape: leaves are { -BPred, -DcPred, -VPred, -HPred, -TmPred }
        // tree = [-BPred, 2,  4, 6,  -DcPred, -VPred,  -HPred, -TmPred]
        // For DcPred (yMode=0): bits 1, 0, 0
        // For VPred  (yMode=1): bits 1, 0, 1
        // For HPred  (yMode=2): bits 1, 1, 0
        // For TmPred (yMode=3): bits 1, 1, 1
        // For BPred  (yMode=4): bit 0
        var probs = Vp8ModeTrees.DefaultKfYModeProb;
        if (yMode == (int)Vp8YMode.BPred)
        {
            writer.EncodeBool(0, probs[0]);
            return;
        }
        writer.EncodeBool(1, probs[0]);
        // DcPred=0/VPred=1: bit 0; HPred=2/TmPred=3: bit 1
        if (yMode <= (int)Vp8YMode.VPred)
        {
            writer.EncodeBool(0, probs[1]);
            // bit 0 = DcPred, bit 1 = VPred
            writer.EncodeBool(yMode == (int)Vp8YMode.VPred ? 1 : 0, probs[2]);
        }
        else
        {
            writer.EncodeBool(1, probs[1]);
            // bit 0 = HPred, bit 1 = TmPred
            writer.EncodeBool(yMode == (int)Vp8YMode.TmPred ? 1 : 0, probs[3]);
        }
    }

    private static void EncodeUvMode(Vp8BoolEncoder writer, int uvMode)
    {
        // UvModeTree: leaves are { -DcPred, -VPred, -HPred, -TmPred }
        // tree = [-DcPred, 2, -VPred, 4, -HPred, -TmPred]
        // DcPred (0): bit 0
        // VPred  (1): bits 1, 0
        // HPred  (2): bits 1, 1, 0
        // TmPred (3): bits 1, 1, 1
        var probs = Vp8ModeTrees.DefaultKfUvModeProb;
        if (uvMode == (int)Vp8UvMode.DcPred) { writer.EncodeBool(0, probs[0]); return; }
        writer.EncodeBool(1, probs[0]);
        if (uvMode == (int)Vp8UvMode.VPred) { writer.EncodeBool(0, probs[1]); return; }
        writer.EncodeBool(1, probs[1]);
        writer.EncodeBool(uvMode == (int)Vp8UvMode.TmPred ? 1 : 0, probs[2]);
    }
}
