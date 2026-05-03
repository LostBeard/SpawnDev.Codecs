// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 Chroma-from-Luma (CFL) prediction. Bit-exact port of libaom
// av1/common/cfl.c <c>av1_cfl_predict_block</c> + <c>cfl_predict_lbd_c</c>
// for 8-bit (lbd) decode.
//
// Upstream Copyright (c) 2017, Alliance for Open Media. All rights reserved.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.
//
// CFL pipeline:
//   1. Sub-sample the luma reconstruction to chroma resolution (4:2:0 = 2x2 sum).
//      Per libaom cfl_luma_subsampling_420_lbd: ac_q3[i] = sum of 4 luma px shifted left 3.
//   2. Subtract the average across the chroma block: ac_q3 -= avg_q3.
//   3. For each chroma plane: dst[i] = clip(get_scaled_luma_q0(alpha_q3, ac_q3[i]) + dc_pred[i]).
//      get_scaled_luma_q0(alpha_q3, ac_q3) = round_shift(alpha_q3 * ac_q3, 6).
//
// The DC predictor (UV DC_PRED on the chroma block) is computed by the standard
// intra path BEFORE this routine is called; CFL just adds the AC component.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 Chroma-from-Luma (CFL) prediction (8-bit / lbd).</summary>
public static class Av1CflPredictor
{
    /// <summary>libaom: convert (joint_sign, plane) to per-plane signed alpha_q3.</summary>
    private static int CflIdxToAlpha(byte alphaIdx, sbyte jointSign, int plane)
    {
        // CFL_SIGN_U(js) = ((js + 1) * 11) >> 5; CFL_SIGN_V(js) = (js + 1) - 3 * CFL_SIGN_U.
        int signU = ((jointSign + 1) * 11) >> 5;
        int signV = (jointSign + 1) - 3 * signU;
        int alphaSign = (plane == 0) ? signU : signV;  // plane: 0 = U, 1 = V (CFL_PRED_U/V)
        if (alphaSign == 0) return 0;
        // CFL_IDX_U(idx) = idx >> 4; CFL_IDX_V(idx) = idx & 0xF.
        int absAlphaQ3 = (plane == 0) ? (alphaIdx >> 4) : (alphaIdx & 0xF);
        return (alphaSign == 2) ? (absAlphaQ3 + 1) : (-absAlphaQ3 - 1);
    }

    /// <summary>
    /// Apply CFL alpha to <paramref name="dst"/> (which already contains DC_PRED).
    /// <paramref name="lumaRecon"/> is the reconstructed luma plane buffer (full frame).
    /// </summary>
    public static void Apply(
        byte[] lumaRecon, int lumaStride,
        int lumaXPx, int lumaYPx,
        int subX, int subY,
        Span<byte> dst, int dstStride,
        int chromaW, int chromaH,
        byte alphaIdx, sbyte jointSign, int plane /* 0=U, 1=V */)
    {
        int alphaQ3 = CflIdxToAlpha(alphaIdx, jointSign, plane);
        if (alphaQ3 == 0)
        {
            // Nothing to add; dst already has DC predictor.
            return;
        }

        // Step 1: subsample luma reconstruction to chroma resolution.
        // For 4:2:0 (subX = subY = 1), each chroma sample sums 2x2 luma samples
        // and shifts left 3 (q3 fixed-point). For 4:2:2 (subX=1,subY=0) sum 2 vert.
        // For 4:4:4 (subX=subY=0) just shift the single luma sample left 3.
        var acQ3 = new int[chromaW * chromaH];
        long sum = 0;
        for (int r = 0; r < chromaH; r++)
        {
            int lumaRow = lumaYPx + (r << subY);
            int dstRowBase = r * chromaW;
            for (int c = 0; c < chromaW; c++)
            {
                int lumaCol = lumaXPx + (c << subX);
                int q3;
                if (subX == 1 && subY == 1)
                {
                    int s = lumaRecon[lumaRow * lumaStride + lumaCol]
                          + lumaRecon[lumaRow * lumaStride + lumaCol + 1]
                          + lumaRecon[(lumaRow + 1) * lumaStride + lumaCol]
                          + lumaRecon[(lumaRow + 1) * lumaStride + lumaCol + 1];
                    q3 = s << 1; // sum of 4 = avg*4, then shift left 1 = sum<<1 (q3 = avg<<3)
                }
                else if (subX == 1 && subY == 0)
                {
                    int s = lumaRecon[lumaRow * lumaStride + lumaCol]
                          + lumaRecon[lumaRow * lumaStride + lumaCol + 1];
                    q3 = s << 2; // (avg of 2) << 3 = sum << 2
                }
                else
                {
                    q3 = lumaRecon[lumaRow * lumaStride + lumaCol] << 3;
                }
                acQ3[dstRowBase + c] = q3;
                sum += q3;
            }
        }
        // Step 2: subtract average. libaom uses round-to-nearest for the avg.
        int numPel = chromaW * chromaH;
        int numPelLog2 = IntegerLog2(numPel);
        int round = numPel >> 1;
        int avgQ3 = (int)((sum + round) >> numPelLog2);
        for (int i = 0; i < numPel; i++)
        {
            acQ3[i] -= avgQ3;
        }

        // Step 3: dst[i] = clip(round_shift(alpha_q3 * ac_q3, 6) + dst[i]).
        for (int r = 0; r < chromaH; r++)
        {
            for (int c = 0; c < chromaW; c++)
            {
                int idx = r * chromaW + c;
                int scaled = alphaQ3 * acQ3[idx];
                // round_shift(x, 6) = (x + 32) >> 6 if x >= 0, else with sign-aware rounding.
                int rounded = (scaled + (1 << 5)) >> 6;
                int v = rounded + dst[r * dstStride + c];
                if (v < 0) v = 0;
                if (v > 255) v = 255;
                dst[r * dstStride + c] = (byte)v;
            }
        }
    }

    private static int IntegerLog2(int x)
    {
        int log = 0;
        while (x > 1) { x >>= 1; log++; }
        return log;
    }
}
