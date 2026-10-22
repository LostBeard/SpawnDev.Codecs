// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Full-syntax VP8 keyframe decoder, kernel 4: reconstruction (RFC 6386 sections 12, 14): intra prediction + inverse
// transforms, ONE GPU thread per macroblock, macroblocks on one WAVEFRONT DIAGONAL per launch.
//
// Schedule. Macroblock (row r, column c) needs its left neighbour (r, c-1), the one above (r-1, c), the one above-left
// (r-1, c-1) and the one above-right (r-1, c+1) (B_PRED reads 4 pixels above and to the right of the macroblock). With the
// diagonal index d = c + 2 * r those are all on diagonals d-1, d-2 or d-3, so a launch per diagonal, in stream order, is
// correct, and every macroblock on one diagonal is independent of the others (they are two columns apart, one row apart).
// Reconstruction reads and writes UNFILTERED pixels: the loop filter (kernel 5) runs after the last diagonal.
//
// Per macroblock: dequantize with the segment's factors; Y2 + inverse WHT when the macroblock is not B_PRED; 16x16 luma
// prediction (DC / V / H / TM) or sixteen sequential 4x4 predictions (B_PRED, ten modes), residual added per 4x4 block
// by the 4x4 inverse DCT; 8x8 chroma prediction for U and V. Intra edge rules: the row above the frame reads as 127, the
// column left of it as 129; the corner is 127 on the first row, else 129 on the first column; B_PRED above-right of a
// sub-block in the right column comes from the macroblock row above (127 on row 0; the last pixel of that row's
// above-neighbour replicated for the last macroblock column), never from the current macroblock.
// Intermediate values are truncated to 16 bits exactly like libvpx's short arrays.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Video.Vp8;

internal static class Vp8FullReconKernel
{
    private const int SinPi8Sqrt2 = 35468;
    private const int CosPi8Sqrt2Minus1 = 20091;

    // scratch layout (int[64], thread private)
    private const int ScDc = 0;        // 16 inverse-WHT outputs
    private const int ScTmp = 16;      // 16 IDCT first-pass values
    private const int ScEdge = 32;     // 32 = top-left X, 33..40 = above A..H
    private const int ScLeft = 41;     // 41..44 = left I..L
    private const int ScPred = 48;     // 16 predicted 4x4 pixels

    public static void Kernel(
        Index1D idx,
        ArrayView<byte> yP, ArrayView<byte> uP, ArrayView<byte> vP,
        ArrayView<byte> recs, ArrayView<short> coeffs, ArrayView<int> meta,
        int diag, int rowLo)
    {
        if (meta[Vp8FullMeta.Error] != 0) return;
        int mbCols = meta[Vp8FullMeta.MbCols];
        int r = rowLo + idx.X;
        int c = diag - Vp8FullMeta.WaveRowWeight * r;
        long mb = (long)r * mbCols + c;
        long rec = mb * Vp8MbRecord.Size;
        long cBase = mb * Vp8FullTokenKernel.CoeffsPerMb;
        int yStride = mbCols * 16;
        int uvStride = mbCols * 8;

        int yMode = recs[rec + Vp8MbRecord.YMode];
        int uvMode = recs[rec + Vp8MbRecord.UvMode];
        int seg = recs[rec + Vp8MbRecord.Segment];
        int hasCoef = recs[rec + Vp8MbRecord.HasCoef];
        int dq = Vp8FullMeta.Dequant + seg * 6;
        int y1Dc = meta[dq + 0], y1Ac = meta[dq + 1], y2Dc = meta[dq + 2], y2Ac = meta[dq + 3];
        int uvDc = meta[dq + 4], uvAc = meta[dq + 5];

        var s = LocalMemory.Allocate<int>(64);

        // ---------------- luma ----------------
        if (yMode != 4)
        {
            PredictBlock(yP, yStride, r, c, 16, yMode);
            if (hasCoef != 0)
            {
                InverseWalsh(coeffs, cBase + 24 * 16, y2Dc, y2Ac, s);
                for (int i = 0; i < 16; i++)
                {
                    int off = (r * 16 + (i >> 2) * 4) * yStride + c * 16 + (i & 3) * 4;
                    IdctAdd(coeffs, cBase + i * 16, s[ScDc + i], y1Ac, s, yP, off, yStride);
                }
            }
        }
        else
        {
            for (int i = 0; i < 16; i++)
            {
                int sr = i >> 2;
                int sc = i & 3;
                int mode = recs[rec + Vp8MbRecord.BModes + i];
                int x = c * 16 + sc * 4;
                int y = r * 16 + sr * 4;
                BuildEdges(yP, yStride, mbCols, r, c, sr, sc, s);
                Predict4x4(s, mode);
                int off = y * yStride + x;
                for (int k = 0; k < 16; k++) yP[off + (k >> 2) * yStride + (k & 3)] = (byte)s[ScPred + k];
                if (hasCoef != 0)
                {
                    int dc = (short)(coeffs[cBase + i * 16] * y1Dc);
                    IdctAdd(coeffs, cBase + i * 16, dc, y1Ac, s, yP, off, yStride);
                }
            }
        }

        // ---------------- chroma ----------------
        PredictBlock(uP, uvStride, r, c, 8, uvMode);
        PredictBlock(vP, uvStride, r, c, 8, uvMode);
        if (hasCoef != 0)
        {
            for (int j = 0; j < 4; j++)
            {
                int off = (r * 8 + (j >> 1) * 4) * uvStride + c * 8 + (j & 1) * 4;
                int dcU = (short)(coeffs[cBase + (16 + j) * 16] * uvDc);
                IdctAdd(coeffs, cBase + (16 + j) * 16, dcU, uvAc, s, uP, off, uvStride);
                int dcV = (short)(coeffs[cBase + (20 + j) * 16] * uvDc);
                IdctAdd(coeffs, cBase + (20 + j) * 16, dcV, uvAc, s, vP, off, uvStride);
            }
        }
    }

    // ------------------------------------------------------------------------------------------
    // 16x16 luma / 8x8 chroma prediction. mode: 0 DC, 1 V, 2 H, 3 TM.
    // ------------------------------------------------------------------------------------------
    private static void PredictBlock(ArrayView<byte> plane, int stride, int r, int c, int n, int mode)
    {
        int x0 = c * n;
        int y0 = r * n;
        bool haveA = r > 0;
        bool haveL = c > 0;
        if (mode == 0)
        {
            int log2n = n == 16 ? 4 : 3;
            int sum = 0;
            if (haveA) for (int i = 0; i < n; i++) sum += plane[(y0 - 1) * stride + x0 + i];
            if (haveL) for (int i = 0; i < n; i++) sum += plane[(y0 + i) * stride + x0 - 1];
            int dc = 128;
            bool both = haveA & haveL;
            bool either = haveA | haveL;
            if (both) dc = (sum + n) >> (log2n + 1);
            else if (either) dc = (sum + (n >> 1)) >> log2n;
            for (int yy = 0; yy < n; yy++)
                for (int xx = 0; xx < n; xx++)
                    plane[(y0 + yy) * stride + x0 + xx] = (byte)dc;
        }
        else if (mode == 1)
        {
            for (int xx = 0; xx < n; xx++)
            {
                int a = 127;
                if (haveA) a = plane[(y0 - 1) * stride + x0 + xx];
                for (int yy = 0; yy < n; yy++) plane[(y0 + yy) * stride + x0 + xx] = (byte)a;
            }
        }
        else if (mode == 2)
        {
            for (int yy = 0; yy < n; yy++)
            {
                int l = 129;
                if (haveL) l = plane[(y0 + yy) * stride + x0 - 1];
                for (int xx = 0; xx < n; xx++) plane[(y0 + yy) * stride + x0 + xx] = (byte)l;
            }
        }
        else
        {
            int tl = 127;
            if (haveA)
            {
                tl = 129;
                if (haveL) tl = plane[(y0 - 1) * stride + x0 - 1];
            }
            for (int yy = 0; yy < n; yy++)
            {
                int l = 129;
                if (haveL) l = plane[(y0 + yy) * stride + x0 - 1];
                for (int xx = 0; xx < n; xx++)
                {
                    int a = 127;
                    if (haveA) a = plane[(y0 - 1) * stride + x0 + xx];
                    int p = l + a - tl;
                    p = p < 0 ? 0 : (p > 255 ? 255 : p);
                    plane[(y0 + yy) * stride + x0 + xx] = (byte)p;
                }
            }
        }
    }

    // ------------------------------------------------------------------------------------------
    // 4x4 sub-block edges (B_PRED): top-left X, above A..H, left I..L, into the scratch.
    // ------------------------------------------------------------------------------------------
    private static void BuildEdges(ArrayView<byte> plane, int stride, int mbCols, int r, int c, int sr, int sc, ArrayView<int> s)
    {
        int x = c * 16 + sc * 4;
        int y = r * 16 + sr * 4;
        bool haveRowAbove = (sr > 0) | (r > 0);          // is the pixel row y - 1 inside the frame?

        for (int i = 0; i < 4; i++)
        {
            int a = 127;
            if (haveRowAbove) a = plane[(y - 1) * stride + x + i];
            s[ScEdge + 1 + i] = a;
        }
        // above-right
        for (int i = 0; i < 4; i++)
        {
            int a = 127;
            if (sc < 3)
            {
                if (haveRowAbove) a = plane[(y - 1) * stride + x + 4 + i];
            }
            else if (r > 0)
            {
                // right column of the macroblock: the row above the MACROBLOCK, columns 16..19 to the right
                if (c == mbCols - 1) a = plane[(r * 16 - 1) * stride + c * 16 + 15];
                else a = plane[(r * 16 - 1) * stride + (c + 1) * 16 + i];
            }
            s[ScEdge + 5 + i] = a;
        }
        // top-left
        int tl;
        if (sr == 0)
        {
            if (r == 0) tl = 127;
            else if ((sc == 0) & (c == 0)) tl = 129;
            else tl = plane[(y - 1) * stride + x - 1];
        }
        else
        {
            if ((sc == 0) & (c == 0)) tl = 129;
            else tl = plane[(y - 1) * stride + x - 1];
        }
        s[ScEdge] = tl;
        // left
        for (int i = 0; i < 4; i++)
        {
            int l = 129;
            bool haveLeft = (sc > 0) | (c > 0);
            if (haveLeft) l = plane[(y + i) * stride + x - 1];
            s[ScLeft + i] = l;
        }
    }

    private static int Avg3(int a, int b, int c) => (a + 2 * b + c + 2) >> 2;
    private static int Avg2(int a, int b) => (a + b + 1) >> 1;

    // The ten 4x4 modes, RFC 6386 section 12.3 (libvpx vp8_intra4x4_predict). Output into s[ScPred..+15] (row-major).
    private static void Predict4x4(ArrayView<int> s, int mode)
    {
        int X = s[ScEdge];
        int A = s[ScEdge + 1], B = s[ScEdge + 2], C = s[ScEdge + 3], D = s[ScEdge + 4];
        int E = s[ScEdge + 5], F = s[ScEdge + 6], G = s[ScEdge + 7], H = s[ScEdge + 8];
        int I = s[ScLeft], J = s[ScLeft + 1], K = s[ScLeft + 2], L = s[ScLeft + 3];

        int p00 = 0, p01 = 0, p02 = 0, p03 = 0;
        int p10 = 0, p11 = 0, p12 = 0, p13 = 0;
        int p20 = 0, p21 = 0, p22 = 0, p23 = 0;
        int p30 = 0, p31 = 0, p32 = 0, p33 = 0;

        if (mode == 0)          // B_DC_PRED
        {
            int dc = (A + B + C + D + I + J + K + L + 4) >> 3;
            p00 = p01 = p02 = p03 = p10 = p11 = p12 = p13 = p20 = p21 = p22 = p23 = p30 = p31 = p32 = p33 = dc;
        }
        else if (mode == 1)     // B_TM_PRED
        {
            p00 = Clamp(I + A - X); p01 = Clamp(I + B - X); p02 = Clamp(I + C - X); p03 = Clamp(I + D - X);
            p10 = Clamp(J + A - X); p11 = Clamp(J + B - X); p12 = Clamp(J + C - X); p13 = Clamp(J + D - X);
            p20 = Clamp(K + A - X); p21 = Clamp(K + B - X); p22 = Clamp(K + C - X); p23 = Clamp(K + D - X);
            p30 = Clamp(L + A - X); p31 = Clamp(L + B - X); p32 = Clamp(L + C - X); p33 = Clamp(L + D - X);
        }
        else if (mode == 2)     // B_VE_PRED
        {
            int v0 = Avg3(X, A, B), v1 = Avg3(A, B, C), v2 = Avg3(B, C, D), v3 = Avg3(C, D, E);
            p00 = p10 = p20 = p30 = v0; p01 = p11 = p21 = p31 = v1;
            p02 = p12 = p22 = p32 = v2; p03 = p13 = p23 = p33 = v3;
        }
        else if (mode == 3)     // B_HE_PRED
        {
            int r0 = Avg3(X, I, J), r1 = Avg3(I, J, K), r2 = Avg3(J, K, L), r3 = Avg3(K, L, L);
            p00 = p01 = p02 = p03 = r0; p10 = p11 = p12 = p13 = r1;
            p20 = p21 = p22 = p23 = r2; p30 = p31 = p32 = p33 = r3;
        }
        else if (mode == 4)     // B_LD_PRED
        {
            p00 = Avg3(A, B, C);
            p01 = p10 = Avg3(B, C, D);
            p02 = p11 = p20 = Avg3(C, D, E);
            p03 = p12 = p21 = p30 = Avg3(D, E, F);
            p13 = p22 = p31 = Avg3(E, F, G);
            p23 = p32 = Avg3(F, G, H);
            p33 = Avg3(G, H, H);
        }
        else if (mode == 5)     // B_RD_PRED
        {
            p03 = Avg3(D, C, B);
            p02 = p13 = Avg3(C, B, A);
            p01 = p12 = p23 = Avg3(B, A, X);
            p00 = p11 = p22 = p33 = Avg3(A, X, I);
            p10 = p21 = p32 = Avg3(X, I, J);
            p20 = p31 = Avg3(I, J, K);
            p30 = Avg3(J, K, L);
        }
        else if (mode == 6)     // B_VR_PRED
        {
            p00 = p21 = Avg2(X, A);
            p01 = p22 = Avg2(A, B);
            p02 = p23 = Avg2(B, C);
            p03 = Avg2(C, D);
            p10 = p31 = Avg3(I, X, A);
            p11 = p32 = Avg3(X, A, B);
            p12 = p33 = Avg3(A, B, C);
            p13 = Avg3(B, C, D);
            p20 = Avg3(J, I, X);
            p30 = Avg3(K, J, I);
        }
        else if (mode == 7)     // B_VL_PRED
        {
            p00 = Avg2(A, B);
            p01 = p20 = Avg2(B, C);
            p02 = p21 = Avg2(C, D);
            p03 = p22 = Avg2(D, E);
            p10 = Avg3(A, B, C);
            p11 = p30 = Avg3(B, C, D);
            p12 = p31 = Avg3(C, D, E);
            p13 = p32 = Avg3(D, E, F);
            p23 = Avg3(E, F, G);
            p33 = Avg3(F, G, H);
        }
        else if (mode == 8)     // B_HD_PRED
        {
            p00 = p12 = Avg2(I, X);
            p10 = p22 = Avg2(J, I);
            p20 = p32 = Avg2(K, J);
            p30 = Avg2(L, K);
            p03 = Avg3(A, B, C);
            p02 = Avg3(X, A, B);
            p01 = p13 = Avg3(I, X, A);
            p11 = p23 = Avg3(J, I, X);
            p21 = p33 = Avg3(K, J, I);
            p31 = Avg3(L, K, J);
        }
        else                    // B_HU_PRED
        {
            p00 = Avg2(I, J);
            p02 = p10 = Avg2(J, K);
            p12 = p20 = Avg2(K, L);
            p01 = Avg3(I, J, K);
            p03 = p11 = Avg3(J, K, L);
            p13 = p21 = Avg3(K, L, L);
            p22 = p23 = p30 = p31 = p32 = p33 = L;
        }

        s[ScPred + 0] = p00; s[ScPred + 1] = p01; s[ScPred + 2] = p02; s[ScPred + 3] = p03;
        s[ScPred + 4] = p10; s[ScPred + 5] = p11; s[ScPred + 6] = p12; s[ScPred + 7] = p13;
        s[ScPred + 8] = p20; s[ScPred + 9] = p21; s[ScPred + 10] = p22; s[ScPred + 11] = p23;
        s[ScPred + 12] = p30; s[ScPred + 13] = p31; s[ScPred + 14] = p32; s[ScPred + 15] = p33;
    }

    private static int Clamp(int v) => v < 0 ? 0 : (v > 255 ? 255 : v);

    // ------------------------------------------------------------------------------------------
    // Transforms. Intermediates are truncated to 16 bits like libvpx's short arrays.
    // ------------------------------------------------------------------------------------------

    /// <summary>Inverse Walsh-Hadamard of the Y2 block (dequantized here) into s[ScDc..ScDc+15].</summary>
    private static void InverseWalsh(ArrayView<short> coeffs, long y2Base, int y2Dc, int y2Ac, ArrayView<int> s)
    {
        for (int i = 0; i < 4; i++)
        {
            int in0 = (short)(coeffs[y2Base + i] * (i == 0 ? y2Dc : y2Ac));
            int in4 = (short)(coeffs[y2Base + 4 + i] * y2Ac);
            int in8 = (short)(coeffs[y2Base + 8 + i] * y2Ac);
            int in12 = (short)(coeffs[y2Base + 12 + i] * y2Ac);
            int a1 = in0 + in12;
            int b1 = in4 + in8;
            int c1 = in4 - in8;
            int d1 = in0 - in12;
            s[ScTmp + i] = (short)(a1 + b1);
            s[ScTmp + 4 + i] = (short)(c1 + d1);
            s[ScTmp + 8 + i] = (short)(a1 - b1);
            s[ScTmp + 12 + i] = (short)(d1 - c1);
        }
        for (int i = 0; i < 4; i++)
        {
            int row = ScTmp + i * 4;
            int a1 = s[row] + s[row + 3];
            int b1 = s[row + 1] + s[row + 2];
            int c1 = s[row + 1] - s[row + 2];
            int d1 = s[row] - s[row + 3];
            int a2 = a1 + b1;
            int b2 = c1 + d1;
            int c2 = a1 - b1;
            int d2 = d1 - c1;
            s[ScDc + i * 4 + 0] = (short)((a2 + 3) >> 3);
            s[ScDc + i * 4 + 1] = (short)((b2 + 3) >> 3);
            s[ScDc + i * 4 + 2] = (short)((c2 + 3) >> 3);
            s[ScDc + i * 4 + 3] = (short)((d2 + 3) >> 3);
        }
    }

    private static int Dq(ArrayView<short> coeffs, long cOff, int k, int dc, int ac)
    {
        if (k == 0) return (short)dc;
        return (short)(coeffs[cOff + k] * ac);
    }

    /// <summary>4x4 inverse DCT of a block (dequantized here, DC given) added onto the plane with clamping.</summary>
    private static void IdctAdd(ArrayView<short> coeffs, long cOff, int dc, int ac, ArrayView<int> s,
                                ArrayView<byte> plane, int dstOff, int stride)
    {
        for (int i = 0; i < 4; i++)
        {
            int i0 = Dq(coeffs, cOff, i, dc, ac);
            int i4 = Dq(coeffs, cOff, 4 + i, dc, ac);
            int i8 = Dq(coeffs, cOff, 8 + i, dc, ac);
            int i12 = Dq(coeffs, cOff, 12 + i, dc, ac);
            int a1 = i0 + i8;
            int b1 = i0 - i8;
            int temp1 = (i4 * SinPi8Sqrt2) >> 16;
            int temp2 = i12 + ((i12 * CosPi8Sqrt2Minus1) >> 16);
            int c1 = temp1 - temp2;
            temp1 = i4 + ((i4 * CosPi8Sqrt2Minus1) >> 16);
            temp2 = (i12 * SinPi8Sqrt2) >> 16;
            int d1 = temp1 + temp2;
            s[ScTmp + i] = (short)(a1 + d1);
            s[ScTmp + 12 + i] = (short)(a1 - d1);
            s[ScTmp + 4 + i] = (short)(b1 + c1);
            s[ScTmp + 8 + i] = (short)(b1 - c1);
        }
        for (int i = 0; i < 4; i++)
        {
            int row = ScTmp + i * 4;
            int t0 = s[row], t1 = s[row + 1], t2 = s[row + 2], t3 = s[row + 3];
            int a1 = t0 + t2;
            int b1 = t0 - t2;
            int temp1 = (t1 * SinPi8Sqrt2) >> 16;
            int temp2 = t3 + ((t3 * CosPi8Sqrt2Minus1) >> 16);
            int c1 = temp1 - temp2;
            temp1 = t1 + ((t1 * CosPi8Sqrt2Minus1) >> 16);
            temp2 = (t3 * SinPi8Sqrt2) >> 16;
            int d1 = temp1 + temp2;
            int o0 = (short)((a1 + d1 + 4) >> 3);
            int o3 = (short)((a1 - d1 + 4) >> 3);
            int o1 = (short)((b1 + c1 + 4) >> 3);
            int o2 = (short)((b1 - c1 + 4) >> 3);
            int p = dstOff + i * stride;
            plane[p + 0] = (byte)Clamp(plane[p + 0] + o0);
            plane[p + 1] = (byte)Clamp(plane[p + 1] + o1);
            plane[p + 2] = (byte)Clamp(plane[p + 2] + o2);
            plane[p + 3] = (byte)Clamp(plane[p + 3] + o3);
        }
    }
}
