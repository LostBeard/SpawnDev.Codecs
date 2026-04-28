// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 keyframe decode kernel. Single-thread-per-frame GPU kernel that
// reads encoded bytes from input + decodes the entire keyframe to
// Y/U/V recon planes. Symmetric companion to
// Vp8FrameSequentialEncodeKernel.
//
// v1 simplifications (matches the v3 encoder defaults):
//   - All MBs Y_PRED = DC_PRED, UV_PRED = DC_PRED
//   - Single token partition (npart = 1)
//   - No segmentation, no loop filter, no skip-coef flag
//   - Default coef probs
//
// Input layout:
//   inBuf[0..tagLen)         : 10-byte uncompressed tag (skipped here -
//                              host parses width/height/baseQIndex
//                              from the tag and passes via parameters)
//   inBuf[tagLen..tagLen+p0Len)
//                            : partition0 bool stream (header + modes)
//   inBuf[tagLen+p0Len..)    : tokenP0 bool stream (coefs)
//
// Output: yRecon, uRecon, vRecon (writes per-MB recon pixels in
// row-major order; same dimensions as the encoder's yRecon planes).
//
// Per Captain's directive: host is pure coordinator. This kernel does
// ALL the per-frame decode work - bool decoding the header, reading
// per-MB modes + coefs, dequantizing, inverse transforming, adding
// predictor, clipping, writing recon.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// VP8 keyframe decode kernel. Single thread per frame; reads bool
/// streams + writes recon planes. Mirror of
/// <see cref="Vp8FrameSequentialEncodeKernel"/>.
/// </summary>
public sealed class Vp8KeyframeDecodeKernel : IDisposable
{
    private const int CospiSqrt2Minus1 = 20091;
    private const int SinpiSqrt2 = 35468;

    private readonly Accelerator _accelerator;
    private readonly Action<
        Index1D,
        ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
        ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
        ArrayView<int>, ArrayView<byte>, ArrayView<int>,
        int, int> _kernel;

    /// <summary>Compile.</summary>
    public Vp8KeyframeDecodeKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            ArrayView<int>, ArrayView<byte>, ArrayView<int>,
            int, int>(DecodeFrameKernel);
    }

    /// <summary>
    /// Decode a VP8 keyframe.
    /// </summary>
    /// <param name="partition0">Partition0 bytes (header + per-MB modes).</param>
    /// <param name="tokenP0">TokenP0 bytes (coefs).</param>
    /// <param name="coefProbsByType">4 block types * 264 bytes flat.</param>
    /// <param name="constsExtended">62-byte combined consts buffer (zigzag + bands + cat3-6 + mode probs).</param>
    /// <param name="yRecon">Output Y plane.</param>
    /// <param name="uRecon">Output U plane.</param>
    /// <param name="vRecon">Output V plane.</param>
    /// <param name="dequant">6 ints: [Y1Dc, Y1Ac, Y2Dc, Y2Ac, UvDc, UvAc].</param>
    /// <param name="aboveCtx">Frame-wide above-context buffer (mbCols * 9 bytes). Caller zero-initializes.</param>
    /// <param name="mbCols">Macroblock columns.</param>
    /// <param name="mbRows">Macroblock rows.</param>
    /// <param name="p0Offset">Offset into partition0 where the bool stream starts (after the frame header bytes the encoder emitted).</param>
    /// <param name="p0Len">Length of partition0 bytes.</param>
    /// <param name="tp0Offset">Offset into tokenP0 (typically 0).</param>
    /// <param name="tp0Len">Length of tokenP0 bytes.</param>
    /// <param name="streamRanges">4 ints: [p0Offset, p0Len, tp0Offset, tp0Len].</param>
    public void Run(
        ArrayView<byte> partition0,
        ArrayView<byte> tokenP0,
        ArrayView<byte> coefProbsByType,
        ArrayView<byte> constsExtended,
        ArrayView<byte> yRecon,
        ArrayView<byte> uRecon,
        ArrayView<byte> vRecon,
        ArrayView<int> dequant,
        ArrayView<byte> aboveCtx,
        ArrayView<int> streamRanges,
        int mbCols, int mbRows)
    {
        if (mbCols <= 0 || mbRows <= 0) throw new ArgumentOutOfRangeException();
        if (dequant.Length < 6) throw new ArgumentException("dequant must hold 6 ints.", nameof(dequant));
        if (aboveCtx.Length < mbCols * 9L)
            throw new ArgumentException("aboveCtx must hold mbCols*9 bytes.", nameof(aboveCtx));
        if (streamRanges.Length < 4)
            throw new ArgumentException("streamRanges must hold 4 ints.", nameof(streamRanges));
        _kernel(1,
            partition0, tokenP0, coefProbsByType, constsExtended,
            yRecon, uRecon, vRecon,
            dequant, aboveCtx, streamRanges,
            mbCols, mbRows);
    }

    private static void DecodeFrameKernel(
        Index1D _,
        ArrayView<byte> partition0,
        ArrayView<byte> tokenP0,
        ArrayView<byte> coefProbsByType,
        ArrayView<byte> constsExtended,
        ArrayView<byte> yRecon,
        ArrayView<byte> uRecon,
        ArrayView<byte> vRecon,
        ArrayView<int> dequant,
        ArrayView<byte> aboveCtx,
        ArrayView<int> streamRanges,
        int mbCols, int mbRows)
    {
        int p0Offset = streamRanges[0];
        int p0Len = streamRanges[1];
        int tp0Offset = streamRanges[2];
        int tp0Len = streamRanges[3];
        int yStride = mbCols * 16;
        int uvStride = mbCols * 8;
        int y1Dc = dequant[0]; int y1Ac = dequant[1];
        int y2Dc = dequant[2]; int y2Ac = dequant[3];
        int uvDc = dequant[4]; int uvAc = dequant[5];

        const int probsPerType = 8 * 33;
        var probsY4 = coefProbsByType.SubView(0L * probsPerType, probsPerType);
        var probsY2 = coefProbsByType.SubView(1L * probsPerType, probsPerType);
        var probsUv = coefProbsByType.SubView(2L * probsPerType, probsPerType);

        // Mode probabilities (extended consts buffer: bytes 56..61).
        byte kfYProb0 = constsExtended[56];
        byte kfYProb1 = constsExtended[57];
        byte kfYProb2 = constsExtended[58];
        byte kfUvProb0 = constsExtended[59];

        // Initialize bool decoders.
        var p0State = Vp8BoolDecoderGpu.Init(partition0, p0Offset, p0Len);
        var tpState = Vp8BoolDecoderGpu.Init(tokenP0, tp0Offset, tp0Len);

        // Per-row left context.
        int leftY4_0 = 0, leftY4_1 = 0, leftY4_2 = 0, leftY4_3 = 0;
        int leftU_0 = 0, leftU_1 = 0;
        int leftV_0 = 0, leftV_1 = 0;
        int leftY2 = 0;

        for (int mbRow = 0; mbRow < mbRows; mbRow++)
        {
            leftY4_0 = 0; leftY4_1 = 0; leftY4_2 = 0; leftY4_3 = 0;
            leftU_0 = 0; leftU_1 = 0;
            leftV_0 = 0; leftV_1 = 0;
            leftY2 = 0;

            for (int mbCol = 0; mbCol < mbCols; mbCol++)
            {
                long aboveBase = (long)mbCol * 9;
                bool haveAbove = mbRow > 0;
                bool haveLeft = mbCol > 0;

                // Read Y mode tree. Validate it is DC_PRED in v1
                // (decoder tolerates any leaf but we only support DC).
                int yMode0 = Vp8BoolDecoderGpu.DecodeBool(ref p0State, partition0, kfYProb0);
                if (yMode0 != 0)
                {
                    int yMode1 = Vp8BoolDecoderGpu.DecodeBool(ref p0State, partition0, kfYProb1);
                    int yMode2 = Vp8BoolDecoderGpu.DecodeBool(ref p0State, partition0, kfYProb2);
                    // We don't act on yMode1/yMode2 - v1 assumes DC.
                }

                // Read UV mode tree (DC_PRED leaf is bit 0).
                int uvMode0 = Vp8BoolDecoderGpu.DecodeBool(ref p0State, partition0, kfUvProb0);
                // v1 assumes DC.

                // Decode coefs into per-MB scratch buffers (in registers / locals).
                // First Y2 (block type 1, firstCoef=0), then 16 Y4 blocks (block
                // type 0, firstCoef=1), then 4 U blocks (type 2, firstCoef=0),
                // then 4 V blocks (type 2, firstCoef=0).

                // Y2 coefs - 16 shorts.
                short y2_0=0, y2_1=0, y2_2=0, y2_3=0;
                short y2_4=0, y2_5=0, y2_6=0, y2_7=0;
                short y2_8=0, y2_9=0, y2_10=0, y2_11=0;
                short y2_12=0, y2_13=0, y2_14=0, y2_15=0;

                int y2Ctx = aboveCtx[aboveBase + 8] + leftY2;
                int y2Eob = DecodeOneBlock(
                    ref tpState, tokenP0, probsY2, constsExtended, y2Ctx, 0,
                    out y2_0, out y2_1, out y2_2, out y2_3,
                    out y2_4, out y2_5, out y2_6, out y2_7,
                    out y2_8, out y2_9, out y2_10, out y2_11,
                    out y2_12, out y2_13, out y2_14, out y2_15);
                int y2HasCoef = y2Eob > 0 ? 1 : 0;
                aboveCtx[aboveBase + 8] = (byte)y2HasCoef;
                leftY2 = y2HasCoef;

                // Dequantize Y2 + inverse Walsh.
                int y2dq0 = y2_0 * y2Dc;
                int y2dq1 = y2_1 * y2Ac, y2dq2 = y2_2 * y2Ac, y2dq3 = y2_3 * y2Ac;
                int y2dq4 = y2_4 * y2Ac, y2dq5 = y2_5 * y2Ac, y2dq6 = y2_6 * y2Ac;
                int y2dq7 = y2_7 * y2Ac, y2dq8 = y2_8 * y2Ac, y2dq9 = y2_9 * y2Ac;
                int y2dq10 = y2_10 * y2Ac, y2dq11 = y2_11 * y2Ac, y2dq12 = y2_12 * y2Ac;
                int y2dq13 = y2_13 * y2Ac, y2dq14 = y2_14 * y2Ac, y2dq15 = y2_15 * y2Ac;

                int y2InvAc = y2dq1 | y2dq2 | y2dq3 | y2dq4 | y2dq5 | y2dq6 | y2dq7
                           | y2dq8 | y2dq9 | y2dq10 | y2dq11 | y2dq12 | y2dq13 | y2dq14 | y2dq15;
                int yInv0, yInv1, yInv2, yInv3, yInv4, yInv5, yInv6, yInv7;
                int yInv8, yInv9, yInv10, yInv11, yInv12, yInv13, yInv14, yInv15;
                if (y2InvAc == 0)
                {
                    int v = (y2dq0 + 3) >> 3;
                    yInv0 = yInv1 = yInv2 = yInv3 = yInv4 = yInv5 = yInv6 = yInv7 =
                        yInv8 = yInv9 = yInv10 = yInv11 = yInv12 = yInv13 = yInv14 = yInv15 = v;
                }
                else
                {
                    InvWalsh4x4(
                        (short)y2dq0, (short)y2dq1, (short)y2dq2, (short)y2dq3,
                        (short)y2dq4, (short)y2dq5, (short)y2dq6, (short)y2dq7,
                        (short)y2dq8, (short)y2dq9, (short)y2dq10, (short)y2dq11,
                        (short)y2dq12, (short)y2dq13, (short)y2dq14, (short)y2dq15,
                        out yInv0, out yInv1, out yInv2, out yInv3,
                        out yInv4, out yInv5, out yInv6, out yInv7,
                        out yInv8, out yInv9, out yInv10, out yInv11,
                        out yInv12, out yInv13, out yInv14, out yInv15);
                }

                // Compute Y predictor (DC mode) from recon neighbours.
                int yPred = ComputeYDcPredictor(mbRow, mbCol, yStride, yRecon, haveAbove, haveLeft);
                int uPred = ComputeUvDcPredictor(mbRow, mbCol, uvStride, uRecon, haveAbove, haveLeft);
                int vPred = ComputeUvDcPredictor(mbRow, mbCol, uvStride, vRecon, haveAbove, haveLeft);

                // 16 Y4 blocks: read coefs, dequant, inject Y2 inv DC, IDCT, add pred, write recon.
                for (int by = 0; by < 4; by++)
                {
                    for (int bx = 0; bx < 4; bx++)
                    {
                        int blockIdx = by * 4 + bx;
                        int aboveVal;
                        if (bx == 0) aboveVal = aboveCtx[aboveBase + 0];
                        else if (bx == 1) aboveVal = aboveCtx[aboveBase + 1];
                        else if (bx == 2) aboveVal = aboveCtx[aboveBase + 2];
                        else aboveVal = aboveCtx[aboveBase + 3];
                        int leftVal;
                        if (by == 0) leftVal = leftY4_0;
                        else if (by == 1) leftVal = leftY4_1;
                        else if (by == 2) leftVal = leftY4_2;
                        else leftVal = leftY4_3;
                        int y4Ctx = aboveVal + leftVal;

                        short y4_0=0, y4_1=0, y4_2=0, y4_3=0;
                        short y4_4=0, y4_5=0, y4_6=0, y4_7=0;
                        short y4_8=0, y4_9=0, y4_10=0, y4_11=0;
                        short y4_12=0, y4_13=0, y4_14=0, y4_15=0;
                        int y4Eob = DecodeOneBlock(
                            ref tpState, tokenP0, probsY4, constsExtended, y4Ctx, 1,
                            out y4_0, out y4_1, out y4_2, out y4_3,
                            out y4_4, out y4_5, out y4_6, out y4_7,
                            out y4_8, out y4_9, out y4_10, out y4_11,
                            out y4_12, out y4_13, out y4_14, out y4_15);
                        int y4HasCoef = y4Eob > 0 ? 1 : 0;

                        if (bx == 0) aboveCtx[aboveBase + 0] = (byte)y4HasCoef;
                        else if (bx == 1) aboveCtx[aboveBase + 1] = (byte)y4HasCoef;
                        else if (bx == 2) aboveCtx[aboveBase + 2] = (byte)y4HasCoef;
                        else aboveCtx[aboveBase + 3] = (byte)y4HasCoef;
                        if (by == 0) leftY4_0 = y4HasCoef;
                        else if (by == 1) leftY4_1 = y4HasCoef;
                        else if (by == 2) leftY4_2 = y4HasCoef;
                        else leftY4_3 = y4HasCoef;

                        int injectDc;
                        if (blockIdx == 0) injectDc = yInv0;
                        else if (blockIdx == 1) injectDc = yInv1;
                        else if (blockIdx == 2) injectDc = yInv2;
                        else if (blockIdx == 3) injectDc = yInv3;
                        else if (blockIdx == 4) injectDc = yInv4;
                        else if (blockIdx == 5) injectDc = yInv5;
                        else if (blockIdx == 6) injectDc = yInv6;
                        else if (blockIdx == 7) injectDc = yInv7;
                        else if (blockIdx == 8) injectDc = yInv8;
                        else if (blockIdx == 9) injectDc = yInv9;
                        else if (blockIdx == 10) injectDc = yInv10;
                        else if (blockIdx == 11) injectDc = yInv11;
                        else if (blockIdx == 12) injectDc = yInv12;
                        else if (blockIdx == 13) injectDc = yInv13;
                        else if (blockIdx == 14) injectDc = yInv14;
                        else injectDc = yInv15;

                        IdctAddBlock(
                            y4_0, y4_1, y4_2, y4_3,
                            y4_4, y4_5, y4_6, y4_7,
                            y4_8, y4_9, y4_10, y4_11,
                            y4_12, y4_13, y4_14, y4_15,
                            y1Ac, injectDc,
                            yPred, yRecon,
                            mbRow, mbCol, by, bx, yStride, isUv: false);
                    }
                }

                // 4 U blocks.
                for (int by = 0; by < 2; by++)
                {
                    for (int bx = 0; bx < 2; bx++)
                    {
                        int aboveVal = (bx == 0) ? aboveCtx[aboveBase + 4] : aboveCtx[aboveBase + 5];
                        int leftVal = (by == 0) ? leftU_0 : leftU_1;
                        int uCtx = aboveVal + leftVal;

                        short u_0=0, u_1=0, u_2=0, u_3=0;
                        short u_4=0, u_5=0, u_6=0, u_7=0;
                        short u_8=0, u_9=0, u_10=0, u_11=0;
                        short u_12=0, u_13=0, u_14=0, u_15=0;
                        int uEob = DecodeOneBlock(
                            ref tpState, tokenP0, probsUv, constsExtended, uCtx, 0,
                            out u_0, out u_1, out u_2, out u_3,
                            out u_4, out u_5, out u_6, out u_7,
                            out u_8, out u_9, out u_10, out u_11,
                            out u_12, out u_13, out u_14, out u_15);
                        int uHasCoef = uEob > 0 ? 1 : 0;
                        if (bx == 0) aboveCtx[aboveBase + 4] = (byte)uHasCoef;
                        else aboveCtx[aboveBase + 5] = (byte)uHasCoef;
                        if (by == 0) leftU_0 = uHasCoef;
                        else leftU_1 = uHasCoef;

                        IdctAddBlock(
                            u_0, u_1, u_2, u_3, u_4, u_5, u_6, u_7,
                            u_8, u_9, u_10, u_11, u_12, u_13, u_14, u_15,
                            uvAc, ((int)u_0) * uvDc,  // dcQ-multiplied since no Y2 injection
                            uPred, uRecon,
                            mbRow, mbCol, by, bx, uvStride, isUv: true);
                    }
                }

                // 4 V blocks.
                for (int by = 0; by < 2; by++)
                {
                    for (int bx = 0; bx < 2; bx++)
                    {
                        int aboveVal = (bx == 0) ? aboveCtx[aboveBase + 6] : aboveCtx[aboveBase + 7];
                        int leftVal = (by == 0) ? leftV_0 : leftV_1;
                        int vCtx = aboveVal + leftVal;

                        short v_0=0, v_1=0, v_2=0, v_3=0;
                        short v_4=0, v_5=0, v_6=0, v_7=0;
                        short v_8=0, v_9=0, v_10=0, v_11=0;
                        short v_12=0, v_13=0, v_14=0, v_15=0;
                        int vEob = DecodeOneBlock(
                            ref tpState, tokenP0, probsUv, constsExtended, vCtx, 0,
                            out v_0, out v_1, out v_2, out v_3,
                            out v_4, out v_5, out v_6, out v_7,
                            out v_8, out v_9, out v_10, out v_11,
                            out v_12, out v_13, out v_14, out v_15);
                        int vHasCoef = vEob > 0 ? 1 : 0;
                        if (bx == 0) aboveCtx[aboveBase + 6] = (byte)vHasCoef;
                        else aboveCtx[aboveBase + 7] = (byte)vHasCoef;
                        if (by == 0) leftV_0 = vHasCoef;
                        else leftV_1 = vHasCoef;

                        IdctAddBlock(
                            v_0, v_1, v_2, v_3, v_4, v_5, v_6, v_7,
                            v_8, v_9, v_10, v_11, v_12, v_13, v_14, v_15,
                            uvAc, ((int)v_0) * uvDc,
                            vPred, vRecon,
                            mbRow, mbCol, by, bx, uvStride, isUv: true);
                    }
                }
            }
        }
    }

    /// <summary>
    /// Decode one coef block via Vp8CoefBlockDecoderGpu, returning the 16
    /// coefs as out parameters. Wraps the SubView-into-temp-buffer
    /// pattern with thread-local register storage so the decoder doesn't
    /// need a global buffer for per-MB scratch.
    /// </summary>
    private static int DecodeOneBlock(
        ref Vp8BoolDecoderGpuState state,
        ArrayView<byte> bitstream,
        ArrayView<byte> probsFlat,
        ArrayView<byte> constsFlat,
        int ctx, int firstCoef,
        out short c0, out short c1, out short c2, out short c3,
        out short c4, out short c5, out short c6, out short c7,
        out short c8, out short c9, out short c10, out short c11,
        out short c12, out short c13, out short c14, out short c15)
    {
        // The Vp8CoefBlockDecoderGpu writes into an ArrayView<short>
        // slice. Allocate per-thread local memory which IS already an
        // ArrayView<short> in ILGPU.
        var localCoefs = LocalMemory.Allocate<short>(16);
        int eob = Vp8CoefBlockDecoderGpu.Decode(
            ref state, bitstream, probsFlat, constsFlat,
            ctx, firstCoef, localCoefs, 0);
        c0 = localCoefs[0]; c1 = localCoefs[1]; c2 = localCoefs[2]; c3 = localCoefs[3];
        c4 = localCoefs[4]; c5 = localCoefs[5]; c6 = localCoefs[6]; c7 = localCoefs[7];
        c8 = localCoefs[8]; c9 = localCoefs[9]; c10 = localCoefs[10]; c11 = localCoefs[11];
        c12 = localCoefs[12]; c13 = localCoefs[13]; c14 = localCoefs[14]; c15 = localCoefs[15];
        return eob;
    }

    private static int ComputeYDcPredictor(
        int mbRow, int mbCol, int yStride,
        ArrayView<byte> yRecon, bool haveAbove, bool haveLeft)
    {
        if (haveAbove && haveLeft)
        {
            int sum = 0;
            for (int c = 0; c < 16; c++)
                sum += yRecon[(long)(mbRow * 16 - 1) * yStride + mbCol * 16 + c];
            for (int r = 0; r < 16; r++)
                sum += yRecon[(long)(mbRow * 16 + r) * yStride + mbCol * 16 - 1];
            return (sum + 16) >> 5;
        }
        else if (haveAbove)
        {
            int sum = 0;
            for (int c = 0; c < 16; c++)
                sum += yRecon[(long)(mbRow * 16 - 1) * yStride + mbCol * 16 + c];
            return (sum + 8) >> 4;
        }
        else if (haveLeft)
        {
            int sum = 0;
            for (int r = 0; r < 16; r++)
                sum += yRecon[(long)(mbRow * 16 + r) * yStride + mbCol * 16 - 1];
            return (sum + 8) >> 4;
        }
        return 128;
    }

    private static int ComputeUvDcPredictor(
        int mbRow, int mbCol, int uvStride,
        ArrayView<byte> recon, bool haveAbove, bool haveLeft)
    {
        if (haveAbove && haveLeft)
        {
            int sum = 0;
            for (int c = 0; c < 8; c++)
                sum += recon[(long)(mbRow * 8 - 1) * uvStride + mbCol * 8 + c];
            for (int r = 0; r < 8; r++)
                sum += recon[(long)(mbRow * 8 + r) * uvStride + mbCol * 8 - 1];
            return (sum + 8) >> 4;
        }
        else if (haveAbove)
        {
            int sum = 0;
            for (int c = 0; c < 8; c++)
                sum += recon[(long)(mbRow * 8 - 1) * uvStride + mbCol * 8 + c];
            return (sum + 4) >> 3;
        }
        else if (haveLeft)
        {
            int sum = 0;
            for (int r = 0; r < 8; r++)
                sum += recon[(long)(mbRow * 8 + r) * uvStride + mbCol * 8 - 1];
            return (sum + 4) >> 3;
        }
        return 128;
    }

    /// <summary>Inverse Walsh on 16 Y2 dequantized values.</summary>
    private static void InvWalsh4x4(
        short i00, short i01, short i02, short i03,
        short i10, short i11, short i12, short i13,
        short i20, short i21, short i22, short i23,
        short i30, short i31, short i32, short i33,
        out int o00, out int o01, out int o02, out int o03,
        out int o10, out int o11, out int o12, out int o13,
        out int o20, out int o21, out int o22, out int o23,
        out int o30, out int o31, out int o32, out int o33)
    {
        InvWalshCol(i00, i10, i20, i30, out short s00, out short s10, out short s20, out short s30);
        InvWalshCol(i01, i11, i21, i31, out short s01, out short s11, out short s21, out short s31);
        InvWalshCol(i02, i12, i22, i32, out short s02, out short s12, out short s22, out short s32);
        InvWalshCol(i03, i13, i23, i33, out short s03, out short s13, out short s23, out short s33);

        InvWalshRowFinal(s00, s01, s02, s03, out o00, out o01, out o02, out o03);
        InvWalshRowFinal(s10, s11, s12, s13, out o10, out o11, out o12, out o13);
        InvWalshRowFinal(s20, s21, s22, s23, out o20, out o21, out o22, out o23);
        InvWalshRowFinal(s30, s31, s32, s33, out o30, out o31, out o32, out o33);
    }

    private static void InvWalshCol(
        short i0, short i1, short i2, short i3,
        out short o0, out short o1, out short o2, out short o3)
    {
        int a1 = i0 + i3;
        int b1 = i1 + i2;
        int c1 = i1 - i2;
        int d1 = i0 - i3;
        o0 = (short)(a1 + b1);
        o1 = (short)(c1 + d1);
        o2 = (short)(a1 - b1);
        o3 = (short)(d1 - c1);
    }

    private static void InvWalshRowFinal(
        short s0, short s1, short s2, short s3,
        out int o0, out int o1, out int o2, out int o3)
    {
        int a1 = s0 + s3;
        int b1 = s1 + s2;
        int c1 = s1 - s2;
        int d1 = s0 - s3;
        int a2 = a1 + b1;
        int b2 = c1 + d1;
        int c2 = a1 - b1;
        int d2 = d1 - c1;
        o0 = (a2 + 3) >> 3;
        o1 = (b2 + 3) >> 3;
        o2 = (c2 + 3) >> 3;
        o3 = (d2 + 3) >> 3;
    }

    /// <summary>
    /// Dequantize block coefs (skipping coef[0] if injectDcOverride
    /// is the Y2-derived DC for a Y4 block; passing q0*dcQ for UV/Y2
    /// where there's no injection), run IDCT, add predictor, clip,
    /// write recon.
    /// </summary>
    private static void IdctAddBlock(
        short q0, short q1, short q2, short q3,
        short q4, short q5, short q6, short q7,
        short q8, short q9, short q10, short q11,
        short q12, short q13, short q14, short q15,
        int acQ, int dcOverride,
        int pred, ArrayView<byte> reconPlane,
        int mbRow, int mbCol, int by, int bx, int stride, bool isUv)
    {
        // Dequantize. dcOverride is either Y2-derived DC (Y4 block) or
        // q0*dcQ (UV / Y2 standalone).
        int dq0 = dcOverride;
        int dq1 = q1 * acQ, dq2 = q2 * acQ, dq3 = q3 * acQ;
        int dq4 = q4 * acQ, dq5 = q5 * acQ, dq6 = q6 * acQ, dq7 = q7 * acQ;
        int dq8 = q8 * acQ, dq9 = q9 * acQ, dq10 = q10 * acQ, dq11 = q11 * acQ;
        int dq12 = q12 * acQ, dq13 = q13 * acQ, dq14 = q14 * acQ, dq15 = q15 * acQ;

        IdctCol((short)dq0, (short)dq4, (short)dq8, (short)dq12,
            out short s00, out short s10, out short s20, out short s30);
        IdctCol((short)dq1, (short)dq5, (short)dq9, (short)dq13,
            out short s01, out short s11, out short s21, out short s31);
        IdctCol((short)dq2, (short)dq6, (short)dq10, (short)dq14,
            out short s02, out short s12, out short s22, out short s32);
        IdctCol((short)dq3, (short)dq7, (short)dq11, (short)dq15,
            out short s03, out short s13, out short s23, out short s33);

        IdctRowAddRecon(s00, s01, s02, s03, pred, reconPlane,
            mbRow, mbCol, by, bx, 0, stride, isUv);
        IdctRowAddRecon(s10, s11, s12, s13, pred, reconPlane,
            mbRow, mbCol, by, bx, 1, stride, isUv);
        IdctRowAddRecon(s20, s21, s22, s23, pred, reconPlane,
            mbRow, mbCol, by, bx, 2, stride, isUv);
        IdctRowAddRecon(s30, s31, s32, s33, pred, reconPlane,
            mbRow, mbCol, by, bx, 3, stride, isUv);
    }

    private static void IdctCol(
        short i0, short i1, short i2, short i3,
        out short o0, out short o1, out short o2, out short o3)
    {
        int a1 = i0 + i2;
        int b1 = i0 - i2;
        int temp1 = (i1 * SinpiSqrt2) >> 16;
        int temp2 = i3 + ((i3 * CospiSqrt2Minus1) >> 16);
        int c1 = temp1 - temp2;
        temp1 = i1 + ((i1 * CospiSqrt2Minus1) >> 16);
        temp2 = (i3 * SinpiSqrt2) >> 16;
        int d1 = temp1 + temp2;
        o0 = (short)(a1 + d1);
        o3 = (short)(a1 - d1);
        o1 = (short)(b1 + c1);
        o2 = (short)(b1 - c1);
    }

    private static void IdctRowAddRecon(
        short i0, short i1, short i2, short i3,
        int pred, ArrayView<byte> reconPlane,
        int mbRow, int mbCol, int by, int bx, int rowInBlock,
        int stride, bool isUv)
    {
        int a1 = i0 + i2;
        int b1 = i0 - i2;
        int temp1 = (i1 * SinpiSqrt2) >> 16;
        int temp2 = i3 + ((i3 * CospiSqrt2Minus1) >> 16);
        int c1 = temp1 - temp2;
        temp1 = i1 + ((i1 * CospiSqrt2Minus1) >> 16);
        temp2 = (i3 * SinpiSqrt2) >> 16;
        int d1 = temp1 + temp2;
        int r0 = (a1 + d1 + 4) >> 3;
        int r3 = (a1 - d1 + 4) >> 3;
        int r1 = (b1 + c1 + 4) >> 3;
        int r2 = (b1 - c1 + 4) >> 3;

        long reconRow = isUv
            ? (long)(mbRow * 8 + by * 4 + rowInBlock) * stride + mbCol * 8 + bx * 4
            : (long)(mbRow * 16 + by * 4 + rowInBlock) * stride + mbCol * 16 + bx * 4;
        reconPlane[reconRow + 0] = ClipAdd(pred, r0);
        reconPlane[reconRow + 1] = ClipAdd(pred, r1);
        reconPlane[reconRow + 2] = ClipAdd(pred, r2);
        reconPlane[reconRow + 3] = ClipAdd(pred, r3);
    }

    private static byte ClipAdd(int p, int r)
    {
        int a = p + r;
        if (a < 0) return 0;
        if (a > 255) return 255;
        return (byte)a;
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
