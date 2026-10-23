// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Lossy WebP encoder, kernel 3: libwebp's VP8EncTokenLoop body for ONE macroblock row per launch (methods 3 and 4:
// RD_OPT_BASIC, no trellis), ONE thread. Per macroblock, in raster order: import the source block, refresh the token
// probabilities / level costs every `max_count` macroblocks (FinalizeTokenProbas + VP8CalculateLevelCosts), VP8Decimate
// (PickBestIntra16 / PickBestIntra4 / PickBestUV with the DC error diffusion), RecordTokens (token buffer + statistics),
// VP8IteratorSaveBoundary. The decisions of macroblock k depend on the statistics of every macroblock before it, so the
// walk is serial by construction; this kernel is the correctness-first form (one thread), checked byte for byte against
// SpawnDev.Codecs.References' Vp8LossyEncoderReference, whose structure it follows statement by statement. Parallelism
// inside a macroblock (the candidate modes) is the next step.
//
// State that crosses macroblocks lives in device buffers: the reconstructed top rows (yTop: luma mb_w*16 then u/v 8+8 per
// macroblock), the non-zero context row (nz, index 0 = nz[-1] = 0), the 4x4 prediction-mode map, the macroblock records,
// the top DC diffusion errors, probabilities / statistics / level costs, the token buffer, and the `cnt` countdown and
// partition-0 size in the meta block. Left state restarts every row (VP8IteratorSetRow -> InitLeft), so it is local.

using ILGPU;
using ILGPU.Runtime;
using M = SpawnDev.Codecs.Images.WebP.Lossy.WebPLossyEncMeta;
using T = SpawnDev.Codecs.Images.WebP.Lossy.WebPLossyEncTables;

namespace SpawnDev.Codecs.Images.WebP.Lossy;

internal static partial class WebPLossyEncodeRowKernel
{
    public const int BPS = 32;
    // byte scratch
    public const int BIn = 0, BOutA = 512, BOutB = 1024, BPred = 1536, BLeft = 3328, BI4 = 3400;
    public const int ByteScratchSize = 3456;
    // prediction cache layout inside BPred (vp8i_enc.h)
    private const int I16DC16 = 0, I16TM16 = 16, I16VE16 = 16 * BPS, I16HE16 = 16 * BPS + 16;
    private const int C8DC8 = 2 * 16 * BPS, C8TM8 = C8DC8 + 16, C8VE8 = 2 * 16 * BPS + 8 * BPS, C8HE8 = C8VE8 + 16;
    private const int I4DC4 = 3 * 16 * BPS, I4HD4 = 3 * 16 * BPS + 4 * BPS, I4TMP = I4HD4 + 8;
    // left memory inside BLeft: y_left at +1 (index -1 at +0), u_left = y_left + 32, v_left = u_left + 16
    private const int YLeft = 1, ULeft = 33, VLeft = 49;
    // int scratch
    public const int IRdA = 0;          // levels of ModeScore A: ydc 16 | yac 256 | uv 128  (400)
    public const int IRdB = 400;        // ModeScore B (I16 rd_tmp)
    public const int II4Best = 800;     // PickBestIntra4 rd_best.y_ac_levels (256)
    public const int ITmpLv = 1056;     // tmp_levels (16)
    public const int IUvLv = 1072;      // PickBestUV rd_uv.uv_levels (128)
    public const int ITmp = 1200;       // transform tmp (256)
    public const int IDcTmp = 1456;     // dc_tmp (16)
    public const int IUvTmp = 1472;     // uv tmp (128)
    public const int IT16 = 1600;       // small tmp (16)
    public const int IT16b = 1616;      // small tmp (16)
    public const int ITopNz = 1632, ILeftNz = 1641;   // 9 each
    public const int ILeftDerr = 1650;  // 4
    public const int IDerrA = 1654, IDerrB = 1660, IDerrUv = 1666;   // 6 each
    public const int IModesI4 = 1672;   // rd->modes_i4 (16)
    public const int IntScratchSize = 1696;
    // level-array offsets inside a ModeScore slot
    private const int LDc = 0, LAc = 16, LUv = 272;

    private const long MaxCost = 0x7fffffffffffffL;
    private const int RdDistoMult = 256;
    private const int FixedProbaBit = 1 << 14;

    // ================================================================== small helpers

    private static int Min(int a, int b) => a < b ? a : b;
    private static int Abs(int v) => v < 0 ? -v : v;
    private static byte Clip8b(int v) => (v & ~0xff) == 0 ? (byte)v : (v < 0 ? (byte)0 : (byte)255);
    private static int Mul1(int a) => ((a * 20091) >> 16) + a;
    private static int Mul2(int a) => (a * 35468) >> 16;

    private static int ScanY(int n) => (n & 3) * 4 + (n >> 2) * 4 * BPS;
    private static int ScanUV(int n) => (n & 1) * 4 + ((n >> 1) & 1) * 4 * BPS + (n >> 2) * 8;
    private static int I16Off(int mode) => mode == 0 ? I16DC16 : (mode == 1 ? I16TM16 : (mode == 2 ? I16VE16 : I16HE16));
    private static int UVOff(int mode) => mode == 0 ? C8DC8 : (mode == 1 ? C8TM8 : (mode == 2 ? C8VE8 : C8HE8));
    private static int I4Off(int mode) => mode < 8 ? I4DC4 + 4 * mode : I4HD4 + 4 * (mode - 8);

    private static int SegF(int seg, int field) => M.Seg + seg * M.SegStride + field;

    // ================================================================== dsp (libwebp dsp/enc.c C versions)

    private static void FTransform(ArrayView<byte> s, int so, ArrayView<byte> r, int ro, ArrayView<int> o, int oo, ArrayView<int> tmp)
    {
        for (int i = 0; i < 4; ++i)
        {
            int d0 = s[so + i * BPS + 0] - r[ro + i * BPS + 0];
            int d1 = s[so + i * BPS + 1] - r[ro + i * BPS + 1];
            int d2 = s[so + i * BPS + 2] - r[ro + i * BPS + 2];
            int d3 = s[so + i * BPS + 3] - r[ro + i * BPS + 3];
            int a0 = d0 + d3, a1 = d1 + d2, a2 = d1 - d2, a3 = d0 - d3;
            tmp[0 + i * 4] = (a0 + a1) * 8;
            tmp[1 + i * 4] = (a2 * 2217 + a3 * 5352 + 1812) >> 9;
            tmp[2 + i * 4] = (a0 - a1) * 8;
            tmp[3 + i * 4] = (a3 * 2217 - a2 * 5352 + 937) >> 9;
        }
        for (int i = 0; i < 4; ++i)
        {
            int a0 = tmp[0 + i] + tmp[12 + i];
            int a1 = tmp[4 + i] + tmp[8 + i];
            int a2 = tmp[4 + i] - tmp[8 + i];
            int a3 = tmp[0 + i] - tmp[12 + i];
            int nz = a3 != 0 ? 1 : 0;
            o[oo + 0 + i] = (short)((a0 + a1 + 7) >> 4);
            o[oo + 4 + i] = (short)(((a2 * 2217 + a3 * 5352 + 12000) >> 16) + nz);
            o[oo + 8 + i] = (short)((a0 - a1 + 7) >> 4);
            o[oo + 12 + i] = (short)((a3 * 2217 - a2 * 5352 + 51000) >> 16);
        }
    }

    private static void ITransformOne(ArrayView<byte> r, int ro, ArrayView<int> input, int io, ArrayView<byte> d, int dO, ArrayView<int> c)
    {
        for (int i = 0; i < 4; ++i)
        {
            int a = input[io + i] + input[io + i + 8];
            int b = input[io + i] - input[io + i + 8];
            int cc = Mul2(input[io + i + 4]) - Mul1(input[io + i + 12]);
            int dd = Mul1(input[io + i + 4]) + Mul2(input[io + i + 12]);
            c[i * 4 + 0] = a + dd;
            c[i * 4 + 1] = b + cc;
            c[i * 4 + 2] = b - cc;
            c[i * 4 + 3] = a - dd;
        }
        for (int i = 0; i < 4; ++i)
        {
            int dc = c[i] + 4;
            int a = dc + c[i + 8];
            int b = dc - c[i + 8];
            int cc = Mul2(c[i + 4]) - Mul1(c[i + 12]);
            int dd = Mul1(c[i + 4]) + Mul2(c[i + 12]);
            int row = i * BPS;
            d[dO + 0 + row] = Clip8b(r[ro + 0 + row] + ((a + dd) >> 3));
            d[dO + 1 + row] = Clip8b(r[ro + 1 + row] + ((b + cc) >> 3));
            d[dO + 2 + row] = Clip8b(r[ro + 2 + row] + ((b - cc) >> 3));
            d[dO + 3 + row] = Clip8b(r[ro + 3 + row] + ((a - dd) >> 3));
        }
    }

    private static void FTransformWHT(ArrayView<int> input, int io, ArrayView<int> o, int oo, ArrayView<int> tmp)
    {
        for (int i = 0; i < 4; ++i)
        {
            int b = io + i * 64;
            int a0 = input[b + 0 * 16] + input[b + 2 * 16];
            int a1 = input[b + 1 * 16] + input[b + 3 * 16];
            int a2 = input[b + 1 * 16] - input[b + 3 * 16];
            int a3 = input[b + 0 * 16] - input[b + 2 * 16];
            tmp[0 + i * 4] = a0 + a1;
            tmp[1 + i * 4] = a3 + a2;
            tmp[2 + i * 4] = a3 - a2;
            tmp[3 + i * 4] = a0 - a1;
        }
        for (int i = 0; i < 4; ++i)
        {
            int a0 = tmp[0 + i] + tmp[8 + i];
            int a1 = tmp[4 + i] + tmp[12 + i];
            int a2 = tmp[4 + i] - tmp[12 + i];
            int a3 = tmp[0 + i] - tmp[8 + i];
            o[oo + 0 + i] = (short)((a0 + a1) >> 1);
            o[oo + 4 + i] = (short)((a3 + a2) >> 1);
            o[oo + 8 + i] = (short)((a3 - a2) >> 1);
            o[oo + 12 + i] = (short)((a0 - a1) >> 1);
        }
    }

    private static void TransformWHT(ArrayView<int> input, int io, ArrayView<int> o, int oo, ArrayView<int> tmp)
    {
        for (int i = 0; i < 4; ++i)
        {
            int a0 = input[io + 0 + i] + input[io + 12 + i];
            int a1 = input[io + 4 + i] + input[io + 8 + i];
            int a2 = input[io + 4 + i] - input[io + 8 + i];
            int a3 = input[io + 0 + i] - input[io + 12 + i];
            tmp[0 + i] = a0 + a1;
            tmp[8 + i] = a0 - a1;
            tmp[4 + i] = a3 + a2;
            tmp[12 + i] = a3 - a2;
        }
        for (int i = 0; i < 4; ++i)
        {
            int dc = tmp[0 + i * 4] + 3;
            int a0 = dc + tmp[3 + i * 4];
            int a1 = tmp[1 + i * 4] + tmp[2 + i * 4];
            int a2 = tmp[1 + i * 4] - tmp[2 + i * 4];
            int a3 = dc - tmp[3 + i * 4];
            int b = oo + i * 64;
            o[b + 0] = (short)((a0 + a1) >> 3);
            o[b + 16] = (short)((a3 + a2) >> 3);
            o[b + 32] = (short)((a0 - a1) >> 3);
            o[b + 48] = (short)((a3 - a2) >> 3);
        }
    }

    private static int SSE(ArrayView<byte> a, int ao, ArrayView<byte> b, int bo, int w, int h)
    {
        int count = 0;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
            {
                int d = a[ao + y * BPS + x] - b[bo + y * BPS + x];
                count += d * d;
            }
        return count;
    }

    private static int TTransform(ArrayView<byte> input, int o, ArrayView<int> t, ArrayView<int> tmp)
    {
        int sum = 0;
        for (int i = 0; i < 4; ++i)
        {
            int r = o + i * BPS;
            int a0 = input[r + 0] + input[r + 2];
            int a1 = input[r + 1] + input[r + 3];
            int a2 = input[r + 1] - input[r + 3];
            int a3 = input[r + 0] - input[r + 2];
            tmp[0 + i * 4] = a0 + a1;
            tmp[1 + i * 4] = a3 + a2;
            tmp[2 + i * 4] = a3 - a2;
            tmp[3 + i * 4] = a0 - a1;
        }
        for (int i = 0; i < 4; ++i)
        {
            int a0 = tmp[0 + i] + tmp[8 + i];
            int a1 = tmp[4 + i] + tmp[12 + i];
            int a2 = tmp[4 + i] - tmp[12 + i];
            int a3 = tmp[0 + i] - tmp[8 + i];
            sum += t[T.WeightY + i + 0] * Abs(a0 + a1);
            sum += t[T.WeightY + i + 4] * Abs(a3 + a2);
            sum += t[T.WeightY + i + 8] * Abs(a3 - a2);
            sum += t[T.WeightY + i + 12] * Abs(a0 - a1);
        }
        return sum;
    }

    private static int Disto4x4(ArrayView<byte> a, int ao, ArrayView<byte> b, int bo, ArrayView<int> t, ArrayView<int> tmp)
        => Abs(TTransform(b, bo, t, tmp) - TTransform(a, ao, t, tmp)) >> 5;

    private static int Disto16x16(ArrayView<byte> a, int ao, ArrayView<byte> b, int bo, ArrayView<int> t, ArrayView<int> tmp)
    {
        int d = 0;
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x)
                d += Disto4x4(a, ao + x * 4 + y * 4 * BPS, b, bo + x * 4 + y * 4 * BPS, t, tmp);
        return d;
    }

    /// <summary>QuantizeBlock_C with the matrix of segment `seg`, kind 0 = y1, 1 = y2, 2 = uv. Levels in zigzag order.</summary>
    private static int QuantizeBlock(ArrayView<int> input, int io, ArrayView<int> output, int oo,
        ArrayView<int> meta, int seg, int kind, ArrayView<int> t)
    {
        int sb = M.Seg + seg * M.SegStride;
        int qS = kind == 0 ? M.SQY1 : (kind == 1 ? M.SQY2 : M.SQUV);
        int iqS = kind == 0 ? M.SIQY1 : (kind == 1 ? M.SIQY2 : M.SIQUV);
        int bS = kind == 0 ? M.SBiasY1 : (kind == 1 ? M.SBiasY2 : M.SBiasUV);
        int zS = kind == 0 ? M.SZtY1 : (kind == 1 ? M.SZtY2 : M.SZtUV);
        int last = -1;
        for (int n = 0; n < 16; ++n)
        {
            int j = t[T.Zigzag + n];
            int k = j == 0 ? 0 : 1;
            int v = input[io + j];
            bool sign = v < 0;
            int sharpen = kind == 0 ? meta[sb + M.SSharpY1 + j] : 0;
            uint coeff = (uint)((sign ? -v : v) + sharpen);
            uint zthresh = (uint)meta[sb + zS + k];
            if (coeff > zthresh)
            {
                uint Q = (uint)meta[sb + qS + k], iQ = (uint)meta[sb + iqS + k], B = (uint)meta[sb + bS + k];
                int level = (int)((coeff * iQ + B) >> 17);
                if (level > 2047) level = 2047;
                if (sign) level = -level;
                input[io + j] = (short)(level * (int)Q);
                output[oo + n] = level;
                if (level != 0) last = n;
            }
            else
            {
                output[oo + n] = 0;
                input[io + j] = 0;
            }
        }
        return last >= 0 ? 1 : 0;
    }

    // ================================================================== predictions (into BPred)

    private static void Fill(ArrayView<byte> b, int o, int v, int size)
    {
        for (int j = 0; j < size; ++j)
            for (int i = 0; i < size; ++i) b[o + j * BPS + i] = (byte)v;
    }

    /// <summary>One prediction family (DC / V / H / TM) of size 8 or 16 at `o`; left / top: offsets into `b` / `top`, or -1 when absent.</summary>
    private static void PredictBlock(ArrayView<byte> b, int o, int size, int round, int shift,
        int leftOff, ArrayView<byte> top, int topOff, int mode)
    {
        bool hasLeft = leftOff >= 0, hasTop = topOff >= 0;
        if (mode == 0)
        {
            int dc = 0;
            if (hasTop)
            {
                for (int j = 0; j < size; ++j) dc += top[topOff + j];
                if (hasLeft) { for (int j = 0; j < size; ++j) dc += b[leftOff + j]; }
                else dc += dc;
                dc = (dc + round) >> shift;
            }
            else if (hasLeft)
            {
                for (int j = 0; j < size; ++j) dc += b[leftOff + j];
                dc += dc;
                dc = (dc + round) >> shift;
            }
            else dc = 0x80;
            Fill(b, o, dc, size);
        }
        else if (mode == 2)   // VE
        {
            if (hasTop) { for (int j = 0; j < size; ++j) for (int i = 0; i < size; ++i) b[o + j * BPS + i] = top[topOff + i]; }
            else Fill(b, o, 127, size);
        }
        else if (mode == 3)   // HE
        {
            if (hasLeft) { for (int j = 0; j < size; ++j) for (int i = 0; i < size; ++i) b[o + j * BPS + i] = b[leftOff + j]; }
            else Fill(b, o, 129, size);
        }
        else                  // TM
        {
            if (hasLeft)
            {
                if (hasTop)
                {
                    int tl = b[leftOff - 1];
                    for (int y = 0; y < size; ++y)
                        for (int x = 0; x < size; ++x) b[o + y * BPS + x] = Clip8b(b[leftOff + y] + top[topOff + x] - tl);
                }
                else { for (int j = 0; j < size; ++j) for (int i = 0; i < size; ++i) b[o + j * BPS + i] = b[leftOff + j]; }
            }
            else
            {
                if (hasTop) { for (int j = 0; j < size; ++j) for (int i = 0; i < size; ++i) b[o + j * BPS + i] = top[topOff + i]; }
                else Fill(b, o, 129, size);
            }
        }
    }

    private static byte Avg3(int a, int b, int c) => (byte)((a + 2 * b + c + 2) >> 2);
    private static byte Avg2(int a, int b) => (byte)((a + b + 1) >> 1);
    private static void P(ArrayView<byte> b, int o, int x, int y, byte v) => b[o + x + y * BPS] = v;

    /// <summary>Intra4Preds_C: the ten 4x4 predictions from the boundary at sb[to-5 .. to+7] (left L K J I, corner X, top A..H).</summary>
    private static void Intra4Preds(ArrayView<byte> b, int to)
    {
        int X = b[to - 1], I = b[to - 2], J = b[to - 3], K = b[to - 4], L = b[to - 5];
        int A = b[to], B = b[to + 1], C = b[to + 2], D = b[to + 3], E = b[to + 4], F = b[to + 5], G = b[to + 6], H = b[to + 7];
        int o;
        // DC4
        int dc = 4 + A + B + C + D + L + K + J + I;
        Fill(b, BPred + I4DC4, dc >> 3, 4);
        // TM4
        o = BPred + I4DC4 + 4;
        for (int y = 0; y < 4; ++y)
        {
            int l = b[to - 2 - y];
            for (int x = 0; x < 4; ++x) b[o + y * BPS + x] = Clip8b(l + b[to + x] - X);
        }
        // VE4
        o = BPred + I4DC4 + 8;
        byte v0 = Avg3(X, A, B), v1 = Avg3(A, B, C), v2 = Avg3(B, C, D), v3 = Avg3(C, D, E);
        for (int i = 0; i < 4; ++i) { P(b, o, 0, i, v0); P(b, o, 1, i, v1); P(b, o, 2, i, v2); P(b, o, 3, i, v3); }
        // HE4
        o = BPred + I4DC4 + 12;
        byte r0 = Avg3(X, I, J), r1 = Avg3(I, J, K), r2 = Avg3(J, K, L), r3 = Avg3(K, L, L);
        for (int x = 0; x < 4; ++x) { P(b, o, x, 0, r0); P(b, o, x, 1, r1); P(b, o, x, 2, r2); P(b, o, x, 3, r3); }
        byte v;
        // RD4
        o = BPred + I4DC4 + 16;
        P(b, o, 0, 3, Avg3(J, K, L));
        v = Avg3(I, J, K); P(b, o, 0, 2, v); P(b, o, 1, 3, v);
        v = Avg3(X, I, J); P(b, o, 0, 1, v); P(b, o, 1, 2, v); P(b, o, 2, 3, v);
        v = Avg3(A, X, I); P(b, o, 0, 0, v); P(b, o, 1, 1, v); P(b, o, 2, 2, v); P(b, o, 3, 3, v);
        v = Avg3(B, A, X); P(b, o, 1, 0, v); P(b, o, 2, 1, v); P(b, o, 3, 2, v);
        v = Avg3(C, B, A); P(b, o, 2, 0, v); P(b, o, 3, 1, v);
        P(b, o, 3, 0, Avg3(D, C, B));
        // VR4
        o = BPred + I4DC4 + 20;
        v = Avg2(X, A); P(b, o, 0, 0, v); P(b, o, 1, 2, v);
        v = Avg2(A, B); P(b, o, 1, 0, v); P(b, o, 2, 2, v);
        v = Avg2(B, C); P(b, o, 2, 0, v); P(b, o, 3, 2, v);
        P(b, o, 3, 0, Avg2(C, D));
        P(b, o, 0, 3, Avg3(K, J, I));
        P(b, o, 0, 2, Avg3(J, I, X));
        v = Avg3(I, X, A); P(b, o, 0, 1, v); P(b, o, 1, 3, v);
        v = Avg3(X, A, B); P(b, o, 1, 1, v); P(b, o, 2, 3, v);
        v = Avg3(A, B, C); P(b, o, 2, 1, v); P(b, o, 3, 3, v);
        P(b, o, 3, 1, Avg3(B, C, D));
        // LD4
        o = BPred + I4DC4 + 24;
        P(b, o, 0, 0, Avg3(A, B, C));
        v = Avg3(B, C, D); P(b, o, 1, 0, v); P(b, o, 0, 1, v);
        v = Avg3(C, D, E); P(b, o, 2, 0, v); P(b, o, 1, 1, v); P(b, o, 0, 2, v);
        v = Avg3(D, E, F); P(b, o, 3, 0, v); P(b, o, 2, 1, v); P(b, o, 1, 2, v); P(b, o, 0, 3, v);
        v = Avg3(E, F, G); P(b, o, 3, 1, v); P(b, o, 2, 2, v); P(b, o, 1, 3, v);
        v = Avg3(F, G, H); P(b, o, 3, 2, v); P(b, o, 2, 3, v);
        P(b, o, 3, 3, Avg3(G, H, H));
        // VL4
        o = BPred + I4DC4 + 28;
        P(b, o, 0, 0, Avg2(A, B));
        v = Avg2(B, C); P(b, o, 1, 0, v); P(b, o, 0, 2, v);
        v = Avg2(C, D); P(b, o, 2, 0, v); P(b, o, 1, 2, v);
        v = Avg2(D, E); P(b, o, 3, 0, v); P(b, o, 2, 2, v);
        P(b, o, 0, 1, Avg3(A, B, C));
        v = Avg3(B, C, D); P(b, o, 1, 1, v); P(b, o, 0, 3, v);
        v = Avg3(C, D, E); P(b, o, 2, 1, v); P(b, o, 1, 3, v);
        v = Avg3(D, E, F); P(b, o, 3, 1, v); P(b, o, 2, 3, v);
        P(b, o, 3, 2, Avg3(E, F, G));
        P(b, o, 3, 3, Avg3(F, G, H));
        // HD4
        o = BPred + I4HD4;
        v = Avg2(I, X); P(b, o, 0, 0, v); P(b, o, 2, 1, v);
        v = Avg2(J, I); P(b, o, 0, 1, v); P(b, o, 2, 2, v);
        v = Avg2(K, J); P(b, o, 0, 2, v); P(b, o, 2, 3, v);
        P(b, o, 0, 3, Avg2(L, K));
        P(b, o, 3, 0, Avg3(A, B, C));
        P(b, o, 2, 0, Avg3(X, A, B));
        v = Avg3(I, X, A); P(b, o, 1, 0, v); P(b, o, 3, 1, v);
        v = Avg3(J, I, X); P(b, o, 1, 1, v); P(b, o, 3, 2, v);
        v = Avg3(K, J, I); P(b, o, 1, 2, v); P(b, o, 3, 3, v);
        P(b, o, 1, 3, Avg3(L, K, J));
        // HU4
        o = BPred + I4HD4 + 4;
        P(b, o, 0, 0, Avg2(I, J));
        v = Avg2(J, K); P(b, o, 2, 0, v); P(b, o, 0, 1, v);
        v = Avg2(K, L); P(b, o, 2, 1, v); P(b, o, 0, 2, v);
        P(b, o, 1, 0, Avg3(I, J, K));
        v = Avg3(J, K, L); P(b, o, 3, 0, v); P(b, o, 1, 1, v);
        v = Avg3(K, L, L); P(b, o, 3, 1, v); P(b, o, 1, 2, v);
        byte lb = (byte)L;
        P(b, o, 3, 2, lb); P(b, o, 2, 2, lb); P(b, o, 0, 3, lb); P(b, o, 1, 3, lb); P(b, o, 2, 3, lb); P(b, o, 3, 3, lb);
    }
}
