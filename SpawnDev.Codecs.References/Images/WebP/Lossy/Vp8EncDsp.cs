// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). Port of the C (reference) versions in libwebp 1.6.0
// src/dsp/enc.c and src/dsp/dec.c (TransformWHT), BSD licence, see NOTICE.md. libwebp's SIMD versions are bit-exact with
// these. Pointers become (array, offset); every block lives in a BPS = 32 stride cache exactly like libwebp's yuv_mem.

namespace SpawnDev.Codecs.References.Images.WebP.Lossy;

/// <summary>A libwebp VP8Matrix (quantizer of one coefficient class).</summary>
internal sealed class Vp8Matrix
{
    public readonly ushort[] Q = new ushort[16];
    public readonly ushort[] IQ = new ushort[16];
    public readonly uint[] Bias = new uint[16];
    public readonly uint[] ZThresh = new uint[16];
    public readonly ushort[] Sharpen = new ushort[16];

    public void CopyFrom(Vp8Matrix o)
    {
        o.Q.CopyTo(Q, 0); o.IQ.CopyTo(IQ, 0); o.Bias.CopyTo(Bias, 0); o.ZThresh.CopyTo(ZThresh, 0); o.Sharpen.CopyTo(Sharpen, 0);
    }
}

internal static class Vp8EncDsp
{
    public const int BPS = 32;
    public const int QFIX = 17;
    public const int MAX_LEVEL = 2047;
    public const int MAX_COEFF_THRESH = 31;

    // Prediction cache layout (vp8i_enc.h)
    public const int I16DC16 = 0 * 16 * BPS, I16TM16 = I16DC16 + 16, I16VE16 = 1 * 16 * BPS, I16HE16 = I16VE16 + 16;
    public const int C8DC8 = 2 * 16 * BPS, C8TM8 = C8DC8 + 16, C8VE8 = 2 * 16 * BPS + 8 * BPS, C8HE8 = C8VE8 + 16;
    public const int I4DC4 = 3 * 16 * BPS, I4TM4 = I4DC4 + 4, I4VE4 = I4DC4 + 8, I4HE4 = I4DC4 + 12;
    public const int I4RD4 = I4DC4 + 16, I4VR4 = I4DC4 + 20, I4LD4 = I4DC4 + 24, I4VL4 = I4DC4 + 28;
    public const int I4HD4 = 3 * 16 * BPS + 4 * BPS, I4HU4 = I4HD4 + 4, I4TMP = I4HD4 + 8;
    public const int YUV_SIZE_ENC = BPS * 16;
    public const int PRED_SIZE_ENC = 32 * BPS + 16 * BPS + 8 * BPS;
    public const int Y_OFF_ENC = 0, U_OFF_ENC = 16, V_OFF_ENC = 16 + 8;

    public static readonly int[] I16ModeOffsets = { I16DC16, I16TM16, I16VE16, I16HE16 };
    public static readonly int[] UVModeOffsets = { C8DC8, C8TM8, C8VE8, C8HE8 };
    public static readonly int[] I4ModeOffsets = { I4DC4, I4TM4, I4VE4, I4HE4, I4RD4, I4VR4, I4LD4, I4VL4, I4HD4, I4HU4 };

    public static readonly int[] Scan =
    {
        0 + 0 * BPS, 4 + 0 * BPS, 8 + 0 * BPS, 12 + 0 * BPS,
        0 + 4 * BPS, 4 + 4 * BPS, 8 + 4 * BPS, 12 + 4 * BPS,
        0 + 8 * BPS, 4 + 8 * BPS, 8 + 8 * BPS, 12 + 8 * BPS,
        0 + 12 * BPS, 4 + 12 * BPS, 8 + 12 * BPS, 12 + 12 * BPS,
    };

    public static readonly int[] ScanUV =
    {
        0 + 0 * BPS, 4 + 0 * BPS, 0 + 4 * BPS, 4 + 4 * BPS,
        8 + 0 * BPS, 12 + 0 * BPS, 8 + 4 * BPS, 12 + 4 * BPS,
    };

    /// <summary>VP8DspScan: luma then U then V block offsets (relative to the Y / U plane start).</summary>
    public static readonly int[] DspScan =
    {
        0 + 0 * BPS, 4 + 0 * BPS, 8 + 0 * BPS, 12 + 0 * BPS,
        0 + 4 * BPS, 4 + 4 * BPS, 8 + 4 * BPS, 12 + 4 * BPS,
        0 + 8 * BPS, 4 + 8 * BPS, 8 + 8 * BPS, 12 + 8 * BPS,
        0 + 12 * BPS, 4 + 12 * BPS, 8 + 12 * BPS, 12 + 12 * BPS,
        0 + 0 * BPS, 4 + 0 * BPS, 0 + 4 * BPS, 4 + 4 * BPS,
        8 + 0 * BPS, 12 + 0 * BPS, 8 + 4 * BPS, 12 + 4 * BPS,
    };

    public static byte Clip8b(int v) => (v & ~0xff) == 0 ? (byte)v : (v < 0 ? (byte)0 : (byte)255);

    private static int Mul1(int a) => ((a * 20091) >> 16) + a;
    private static int Mul2(int a) => (a * 35468) >> 16;

    // ------------------------------------------------------------------ transforms

    private static void ITransformOne(byte[] refA, int refO, short[] input, int inO, byte[] dst, int dstO)
    {
        Span<int> C = stackalloc int[16];
        int t = 0;
        for (int i = 0; i < 4; ++i)
        {
            int a = input[inO + i] + input[inO + i + 8];
            int b = input[inO + i] - input[inO + i + 8];
            int c = Mul2(input[inO + i + 4]) - Mul1(input[inO + i + 12]);
            int d = Mul1(input[inO + i + 4]) + Mul2(input[inO + i + 12]);
            C[t + 0] = a + d;
            C[t + 1] = b + c;
            C[t + 2] = b - c;
            C[t + 3] = a - d;
            t += 4;
        }
        for (int i = 0; i < 4; ++i)
        {
            int dc = C[i] + 4;
            int a = dc + C[i + 8];
            int b = dc - C[i + 8];
            int c = Mul2(C[i + 4]) - Mul1(C[i + 12]);
            int d = Mul1(C[i + 4]) + Mul2(C[i + 12]);
            int row = i * BPS;
            dst[dstO + 0 + row] = Clip8b(refA[refO + 0 + row] + ((a + d) >> 3));
            dst[dstO + 1 + row] = Clip8b(refA[refO + 1 + row] + ((b + c) >> 3));
            dst[dstO + 2 + row] = Clip8b(refA[refO + 2 + row] + ((b - c) >> 3));
            dst[dstO + 3 + row] = Clip8b(refA[refO + 3 + row] + ((a - d) >> 3));
        }
    }

    public static void ITransform(byte[] refA, int refO, short[] input, int inO, byte[] dst, int dstO, bool doTwo)
    {
        ITransformOne(refA, refO, input, inO, dst, dstO);
        if (doTwo) ITransformOne(refA, refO + 4, input, inO + 16, dst, dstO + 4);
    }

    public static void FTransform(byte[] src, int srcO, byte[] refA, int refO, short[] output, int outO)
    {
        Span<int> tmp = stackalloc int[16];
        for (int i = 0; i < 4; ++i, srcO += BPS, refO += BPS)
        {
            int d0 = src[srcO + 0] - refA[refO + 0];
            int d1 = src[srcO + 1] - refA[refO + 1];
            int d2 = src[srcO + 2] - refA[refO + 2];
            int d3 = src[srcO + 3] - refA[refO + 3];
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
            output[outO + 0 + i] = (short)((a0 + a1 + 7) >> 4);
            output[outO + 4 + i] = (short)(((a2 * 2217 + a3 * 5352 + 12000) >> 16) + (a3 != 0 ? 1 : 0));
            output[outO + 8 + i] = (short)((a0 - a1 + 7) >> 4);
            output[outO + 12 + i] = (short)((a3 * 2217 - a2 * 5352 + 51000) >> 16);
        }
    }

    public static void FTransform2(byte[] src, int srcO, byte[] refA, int refO, short[] output, int outO)
    {
        FTransform(src, srcO, refA, refO, output, outO);
        FTransform(src, srcO + 4, refA, refO + 4, output, outO + 16);
    }

    /// <summary>FTransformWHT_C: input = the DC of 16 blocks laid out 16 shorts apart (rows of 4 blocks 64 apart).</summary>
    public static void FTransformWHT(short[] input, int inO, short[] output, int outO)
    {
        Span<int> tmp = stackalloc int[16];
        for (int i = 0; i < 4; ++i, inO += 64)
        {
            int a0 = input[inO + 0 * 16] + input[inO + 2 * 16];
            int a1 = input[inO + 1 * 16] + input[inO + 3 * 16];
            int a2 = input[inO + 1 * 16] - input[inO + 3 * 16];
            int a3 = input[inO + 0 * 16] - input[inO + 2 * 16];
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
            int b0 = a0 + a1, b1 = a3 + a2, b2 = a3 - a2, b3 = a0 - a1;
            output[outO + 0 + i] = (short)(b0 >> 1);
            output[outO + 4 + i] = (short)(b1 >> 1);
            output[outO + 8 + i] = (short)(b2 >> 1);
            output[outO + 12 + i] = (short)(b3 >> 1);
        }
    }

    /// <summary>dsp/dec.c TransformWHT_C (inverse): writes the DC of the 16 blocks (16 shorts apart).</summary>
    public static void TransformWHT(short[] input, int inO, short[] output, int outO)
    {
        Span<int> tmp = stackalloc int[16];
        for (int i = 0; i < 4; ++i)
        {
            int a0 = input[inO + 0 + i] + input[inO + 12 + i];
            int a1 = input[inO + 4 + i] + input[inO + 8 + i];
            int a2 = input[inO + 4 + i] - input[inO + 8 + i];
            int a3 = input[inO + 0 + i] - input[inO + 12 + i];
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
            output[outO + 0] = (short)((a0 + a1) >> 3);
            output[outO + 16] = (short)((a3 + a2) >> 3);
            output[outO + 32] = (short)((a0 - a1) >> 3);
            output[outO + 48] = (short)((a3 - a2) >> 3);
            outO += 64;
        }
    }

    // ------------------------------------------------------------------ predictions

    private static void Fill(byte[] dst, int o, int value, int size)
    {
        for (int j = 0; j < size; ++j)
            for (int i = 0; i < size; ++i) dst[o + j * BPS + i] = (byte)value;
    }

    // 'top' / 'left' absent = null array (libwebp's NULL pointers).
    private static void VerticalPred(byte[] dst, int o, byte[]? top, int topO, int size)
    {
        if (top != null)
        {
            for (int j = 0; j < size; ++j)
                for (int i = 0; i < size; ++i) dst[o + j * BPS + i] = top[topO + i];
        }
        else Fill(dst, o, 127, size);
    }

    private static void HorizontalPred(byte[] dst, int o, byte[]? left, int leftO, int size)
    {
        if (left != null)
        {
            for (int j = 0; j < size; ++j)
                for (int i = 0; i < size; ++i) dst[o + j * BPS + i] = left[leftO + j];
        }
        else Fill(dst, o, 129, size);
    }

    private static void TrueMotion(byte[] dst, int o, byte[]? left, int leftO, byte[]? top, int topO, int size)
    {
        if (left != null)
        {
            if (top != null)
            {
                int tl = left[leftO - 1];
                for (int y = 0; y < size; ++y)
                {
                    int l = left[leftO + y];
                    for (int x = 0; x < size; ++x)
                        dst[o + y * BPS + x] = Clip8b(l + top[topO + x] - tl);
                }
            }
            else HorizontalPred(dst, o, left, leftO, size);
        }
        else
        {
            if (top != null) VerticalPred(dst, o, top, topO, size);
            else Fill(dst, o, 129, size);
        }
    }

    private static void DCMode(byte[] dst, int o, byte[]? left, int leftO, byte[]? top, int topO, int size, int round, int shift)
    {
        int DC = 0;
        if (top != null)
        {
            for (int j = 0; j < size; ++j) DC += top[topO + j];
            if (left != null) { for (int j = 0; j < size; ++j) DC += left[leftO + j]; }
            else DC += DC;
            DC = (DC + round) >> shift;
        }
        else if (left != null)
        {
            for (int j = 0; j < size; ++j) DC += left[leftO + j];
            DC += DC;
            DC = (DC + round) >> shift;
        }
        else DC = 0x80;
        Fill(dst, o, DC, size);
    }

    /// <summary>IntraChromaPreds_C: U at +0, V at +8; left: V samples 16 bytes after U's, top: V 8 bytes after U's.</summary>
    public static void IntraChromaPreds(byte[] dst, byte[]? left, int leftO, byte[]? top, int topO)
    {
        DCMode(dst, C8DC8, left, leftO, top, topO, 8, 8, 4);
        VerticalPred(dst, C8VE8, top, topO, 8);
        HorizontalPred(dst, C8HE8, left, leftO, 8);
        TrueMotion(dst, C8TM8, left, leftO, top, topO, 8);
        DCMode(dst, C8DC8 + 8, left, leftO + 16, top, topO + 8, 8, 8, 4);
        VerticalPred(dst, C8VE8 + 8, top, topO + 8, 8);
        HorizontalPred(dst, C8HE8 + 8, left, leftO + 16, 8);
        TrueMotion(dst, C8TM8 + 8, left, leftO + 16, top, topO + 8, 8);
    }

    public static void Intra16Preds(byte[] dst, byte[]? left, int leftO, byte[]? top, int topO)
    {
        DCMode(dst, I16DC16, left, leftO, top, topO, 16, 16, 5);
        VerticalPred(dst, I16VE16, top, topO, 16);
        HorizontalPred(dst, I16HE16, left, leftO, 16);
        TrueMotion(dst, I16TM16, left, leftO, top, topO, 16);
    }

    private static byte Avg3(int a, int b, int c) => (byte)((a + 2 * b + c + 2) >> 2);
    private static byte Avg2(int a, int b) => (byte)((a + b + 1) >> 1);

    /// <summary>Intra4Preds_C: left samples are t[o-5 .. o-2] (bottom to top... as libwebp: L K J I), top-left t[o-1], top t[o..o+3], top-right t[o+4..o+7].</summary>
    public static void Intra4Preds(byte[] dst, byte[] t, int o)
    {
        int X = t[o - 1], I = t[o - 2], J = t[o - 3], K = t[o - 4], L = t[o - 5];
        int A = t[o], B = t[o + 1], C = t[o + 2], D = t[o + 3], E = t[o + 4], F = t[o + 5], G = t[o + 6], H = t[o + 7];

        // DC4
        {
            uint dc = 4;
            for (int i = 0; i < 4; ++i) dc += (uint)(t[o + i] + t[o - 5 + i]);
            Fill(dst, I4DC4, (int)(dc >> 3), 4);
        }
        // TM4
        for (int y = 0; y < 4; ++y)
        {
            int l = t[o - 2 - y];
            for (int x = 0; x < 4; ++x) dst[I4TM4 + y * BPS + x] = Clip8b(l + t[o + x] - X);
        }
        // VE4
        {
            byte v0 = Avg3(X, A, B), v1 = Avg3(A, B, C), v2 = Avg3(B, C, D), v3 = Avg3(C, D, E);
            for (int i = 0; i < 4; ++i)
            {
                int r = I4VE4 + i * BPS;
                dst[r] = v0; dst[r + 1] = v1; dst[r + 2] = v2; dst[r + 3] = v3;
            }
        }
        // HE4
        {
            byte r0 = Avg3(X, I, J), r1 = Avg3(I, J, K), r2 = Avg3(J, K, L), r3 = Avg3(K, L, L);
            for (int x = 0; x < 4; ++x)
            {
                dst[I4HE4 + 0 * BPS + x] = r0; dst[I4HE4 + 1 * BPS + x] = r1;
                dst[I4HE4 + 2 * BPS + x] = r2; dst[I4HE4 + 3 * BPS + x] = r3;
            }
        }
        // RD4
        {
            int b = I4RD4;
            void S(int x, int y, byte v) => dst[b + x + y * BPS] = v;
            S(0, 3, Avg3(J, K, L));
            byte v; v = Avg3(I, J, K); S(0, 2, v); S(1, 3, v);
            v = Avg3(X, I, J); S(0, 1, v); S(1, 2, v); S(2, 3, v);
            v = Avg3(A, X, I); S(0, 0, v); S(1, 1, v); S(2, 2, v); S(3, 3, v);
            v = Avg3(B, A, X); S(1, 0, v); S(2, 1, v); S(3, 2, v);
            v = Avg3(C, B, A); S(2, 0, v); S(3, 1, v);
            S(3, 0, Avg3(D, C, B));
        }
        // VR4
        {
            int b = I4VR4;
            void S(int x, int y, byte v) => dst[b + x + y * BPS] = v;
            byte v;
            v = Avg2(X, A); S(0, 0, v); S(1, 2, v);
            v = Avg2(A, B); S(1, 0, v); S(2, 2, v);
            v = Avg2(B, C); S(2, 0, v); S(3, 2, v);
            S(3, 0, Avg2(C, D));
            S(0, 3, Avg3(K, J, I));
            S(0, 2, Avg3(J, I, X));
            v = Avg3(I, X, A); S(0, 1, v); S(1, 3, v);
            v = Avg3(X, A, B); S(1, 1, v); S(2, 3, v);
            v = Avg3(A, B, C); S(2, 1, v); S(3, 3, v);
            S(3, 1, Avg3(B, C, D));
        }
        // LD4
        {
            int b = I4LD4;
            void S(int x, int y, byte v) => dst[b + x + y * BPS] = v;
            byte v;
            S(0, 0, Avg3(A, B, C));
            v = Avg3(B, C, D); S(1, 0, v); S(0, 1, v);
            v = Avg3(C, D, E); S(2, 0, v); S(1, 1, v); S(0, 2, v);
            v = Avg3(D, E, F); S(3, 0, v); S(2, 1, v); S(1, 2, v); S(0, 3, v);
            v = Avg3(E, F, G); S(3, 1, v); S(2, 2, v); S(1, 3, v);
            v = Avg3(F, G, H); S(3, 2, v); S(2, 3, v);
            S(3, 3, Avg3(G, H, H));
        }
        // VL4
        {
            int b = I4VL4;
            void S(int x, int y, byte v) => dst[b + x + y * BPS] = v;
            byte v;
            S(0, 0, Avg2(A, B));
            v = Avg2(B, C); S(1, 0, v); S(0, 2, v);
            v = Avg2(C, D); S(2, 0, v); S(1, 2, v);
            v = Avg2(D, E); S(3, 0, v); S(2, 2, v);
            S(0, 1, Avg3(A, B, C));
            v = Avg3(B, C, D); S(1, 1, v); S(0, 3, v);
            v = Avg3(C, D, E); S(2, 1, v); S(1, 3, v);
            v = Avg3(D, E, F); S(3, 1, v); S(2, 3, v);
            S(3, 2, Avg3(E, F, G));
            S(3, 3, Avg3(F, G, H));
        }
        // HD4
        {
            int b = I4HD4;
            void S(int x, int y, byte v) => dst[b + x + y * BPS] = v;
            byte v;
            v = Avg2(I, X); S(0, 0, v); S(2, 1, v);
            v = Avg2(J, I); S(0, 1, v); S(2, 2, v);
            v = Avg2(K, J); S(0, 2, v); S(2, 3, v);
            S(0, 3, Avg2(L, K));
            S(3, 0, Avg3(A, B, C));
            S(2, 0, Avg3(X, A, B));
            v = Avg3(I, X, A); S(1, 0, v); S(3, 1, v);
            v = Avg3(J, I, X); S(1, 1, v); S(3, 2, v);
            v = Avg3(K, J, I); S(1, 2, v); S(3, 3, v);
            S(1, 3, Avg3(L, K, J));
        }
        // HU4
        {
            int b = I4HU4;
            void S(int x, int y, byte v) => dst[b + x + y * BPS] = v;
            byte v;
            S(0, 0, Avg2(I, J));
            v = Avg2(J, K); S(2, 0, v); S(0, 1, v);
            v = Avg2(K, L); S(2, 1, v); S(0, 2, v);
            S(1, 0, Avg3(I, J, K));
            v = Avg3(J, K, L); S(3, 0, v); S(1, 1, v);
            v = Avg3(K, L, L); S(3, 1, v); S(1, 2, v);
            byte l = (byte)L;
            S(3, 2, l); S(2, 2, l); S(0, 3, l); S(1, 3, l); S(2, 3, l); S(3, 3, l);
        }
    }

    // ------------------------------------------------------------------ metrics

    public static int SSE(byte[] a, int ao, byte[] b, int bo, int w, int h)
    {
        int count = 0;
        for (int y = 0; y < h; ++y, ao += BPS, bo += BPS)
            for (int x = 0; x < w; ++x)
            {
                int d = a[ao + x] - b[bo + x];
                count += d * d;
            }
        return count;
    }

    private static int TTransform(byte[] input, int o, ushort[] w)
    {
        int sum = 0;
        Span<int> tmp = stackalloc int[16];
        for (int i = 0; i < 4; ++i, o += BPS)
        {
            int a0 = input[o + 0] + input[o + 2];
            int a1 = input[o + 1] + input[o + 3];
            int a2 = input[o + 1] - input[o + 3];
            int a3 = input[o + 0] - input[o + 2];
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
            int b0 = a0 + a1, b1 = a3 + a2, b2 = a3 - a2, b3 = a0 - a1;
            sum += w[i + 0] * Math.Abs(b0);
            sum += w[i + 4] * Math.Abs(b1);
            sum += w[i + 8] * Math.Abs(b2);
            sum += w[i + 12] * Math.Abs(b3);
        }
        return sum;
    }

    public static int Disto4x4(byte[] a, int ao, byte[] b, int bo, ushort[] w)
        => Math.Abs(TTransform(b, bo, w) - TTransform(a, ao, w)) >> 5;

    public static int Disto16x16(byte[] a, int ao, byte[] b, int bo, ushort[] w)
    {
        int D = 0;
        for (int y = 0; y < 16 * BPS; y += 4 * BPS)
            for (int x = 0; x < 16; x += 4)
                D += Disto4x4(a, ao + x + y, b, bo + x + y, w);
        return D;
    }

    public static void Mean16x4(byte[] r, int o, Span<uint> dc)
    {
        for (int k = 0; k < 4; ++k)
        {
            uint avg = 0;
            for (int y = 0; y < 4; ++y)
                for (int x = 0; x < 4; ++x) avg += r[o + x + y * BPS];
            dc[k] = avg;
            o += 4;
        }
    }

    // ------------------------------------------------------------------ quantization

    public static int QuantDiv(uint n, uint iQ, uint B) => (int)((n * iQ + B) >> QFIX);

    /// <summary>QuantizeBlock_C (also VP8EncQuantizeBlockWHT): levels in zigzag order, `input` dequantized in place.</summary>
    public static int QuantizeBlock(short[] input, int inO, short[] output, int outO, Vp8Matrix mtx)
    {
        int last = -1;
        for (int n = 0; n < 16; ++n)
        {
            int j = Vp8EncTables.kZigzag[n];
            int v = input[inO + j];
            bool sign = v < 0;
            uint coeff = (uint)((sign ? -v : v) + mtx.Sharpen[j]);
            if (coeff > mtx.ZThresh[j])
            {
                uint Q = mtx.Q[j], iQ = mtx.IQ[j], B = mtx.Bias[j];
                int level = QuantDiv(coeff, iQ, B);
                if (level > MAX_LEVEL) level = MAX_LEVEL;
                if (sign) level = -level;
                input[inO + j] = (short)(level * (int)Q);
                output[outO + n] = (short)level;
                if (level != 0) last = n;
            }
            else
            {
                output[outO + n] = 0;
                input[inO + j] = 0;
            }
        }
        return last >= 0 ? 1 : 0;
    }

    public static int Quantize2Blocks(short[] input, int inO, short[] output, int outO, Vp8Matrix mtx)
    {
        int nz = QuantizeBlock(input, inO, output, outO, mtx) << 0;
        nz |= QuantizeBlock(input, inO + 16, output, outO + 16, mtx) << 1;
        return nz;
    }

    // ------------------------------------------------------------------ analysis histogram

    /// <summary>CollectHistogram_C + VP8SetHistogramData: returns (max_value, last_non_zero).</summary>
    public static (int MaxValue, int LastNonZero) CollectHistogram(byte[] refA, int refO, byte[] pred, int predO, int startBlock, int endBlock)
    {
        Span<int> distribution = stackalloc int[MAX_COEFF_THRESH + 1];
        distribution.Clear();
        var outC = new short[16];
        for (int j = startBlock; j < endBlock; ++j)
        {
            FTransform(refA, refO + DspScan[j], pred, predO + DspScan[j], outC, 0);
            for (int k = 0; k < 16; ++k)
            {
                int v = Math.Abs(outC[k]) >> 3;
                int clipped = v > MAX_COEFF_THRESH ? MAX_COEFF_THRESH : v;
                ++distribution[clipped];
            }
        }
        int maxValue = 0, lastNonZero = 1;
        for (int k = 0; k <= MAX_COEFF_THRESH; ++k)
        {
            int value = distribution[k];
            if (value > 0)
            {
                if (value > maxValue) maxValue = value;
                lastNonZero = k;
            }
        }
        return (maxValue, lastNonZero);
    }

    public static void Copy(byte[] src, int so, byte[] dst, int dO, int w, int h)
    {
        for (int y = 0; y < h; ++y, so += BPS, dO += BPS)
            System.Array.Copy(src, so, dst, dO, w);
    }
}
