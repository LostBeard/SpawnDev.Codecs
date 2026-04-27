// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 4x4 intra prediction. Ten modes per RFC 6386 sec 12.3:
//   B_DC_PRED - average of 4 above + 4 left
//   B_TM_PRED - TrueMotion: above[c] + left[r] - top_left, clamped
//   B_VE_PRED - vertical-with-edge-filter: AVG3 over above row, repeat down
//   B_HE_PRED - horizontal-with-edge-filter: AVG3 over left column, repeat right
//   B_LD_PRED - down-left diagonal (D45e variant - VP8 uses extrapolated H)
//   B_RD_PRED - right-down diagonal (D135)
//   B_VR_PRED - vertical-right (D117)
//   B_VL_PRED - vertical-left (D63e variant - VP8 uses extrapolated above)
//   B_HD_PRED - horizontal-down (D153)
//   B_HU_PRED - horizontal-up (D207)
//
// Inputs:
//   above[-1] = top_left sample (caller passes 0 if unavailable; T8 default)
//   above[0..7] = above row + above-right samples (libvpx VP8 reads up to 8;
//                 the 4 above-right samples are filled with the above row's
//                 last sample when the actual above-right block is missing)
//   left[0..3] = left column samples
//
// Reference: libvpx vpx_dsp/intrapred.c (the per-mode 4x4 functions named
// vpx_dc_predictor_4x4, vpx_tm_predictor_4x4, vpx_ve_predictor_4x4,
// vpx_he_predictor_4x4, vpx_d45e_predictor_4x4, vpx_d135_predictor_4x4,
// vpx_d117_predictor_4x4, vpx_d63e_predictor_4x4, vpx_d153_predictor_4x4,
// vpx_d207_predictor_4x4) plus the dispatch in vp8/common/reconintra4x4.c.

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>VP8 4x4 intra prediction mode (RFC 6386 sec 12.3).</summary>
public enum Vp8IntraMode4x4 : byte
{
    /// <summary>DC predictor - average of 4 above + 4 left samples.</summary>
    BDcPred = 0,
    /// <summary>TrueMotion predictor.</summary>
    BTmPred = 1,
    /// <summary>Vertical with 3-tap edge filter.</summary>
    BVePred = 2,
    /// <summary>Horizontal with 3-tap edge filter.</summary>
    BHePred = 3,
    /// <summary>Down-left diagonal (D45 variant).</summary>
    BLdPred = 4,
    /// <summary>Right-down diagonal (D135).</summary>
    BRdPred = 5,
    /// <summary>Vertical-right (D117).</summary>
    BVrPred = 6,
    /// <summary>Vertical-left (D63 variant).</summary>
    BVlPred = 7,
    /// <summary>Horizontal-down (D153).</summary>
    BHdPred = 8,
    /// <summary>Horizontal-up (D207).</summary>
    BHuPred = 9,
}

/// <summary>VP8 4x4 intra prediction (10 modes).</summary>
public static class Vp8IntraPredictor4x4
{
    private static int Avg3(int a, int b, int c) => (a + 2 * b + c + 2) >> 2;
    private static int Avg2(int a, int b) => (a + b + 1) >> 1;
    private static int Clamp255(int v) => v < 0 ? 0 : v > 255 ? 255 : v;

    /// <summary>
    /// Predict a 4x4 luma block. <paramref name="aboveBuffer"/> must have at least
    /// 9 entries indexable as <c>above[-1..7]</c> (caller-supplied with the
    /// VP8 edge convention applied).
    /// </summary>
    /// <param name="mode">Prediction mode.</param>
    /// <param name="aboveBuffer">Buffer containing 9+ samples; <paramref name="aboveOffset"/> points at above[0].</param>
    /// <param name="aboveOffset">Index in <paramref name="aboveBuffer"/> of above[0]. Must be >= 1 (above[-1] is at offset-1).</param>
    /// <param name="left">4 left-column samples.</param>
    /// <param name="dst">Destination 4x4 block at <paramref name="stride"/>.</param>
    /// <param name="stride">Stride of dst in bytes.</param>
    public static void Predict(
        Vp8IntraMode4x4 mode,
        ReadOnlySpan<byte> aboveBuffer,
        int aboveOffset,
        ReadOnlySpan<byte> left,
        Span<byte> dst, int stride)
    {
        if (aboveOffset < 1) throw new ArgumentOutOfRangeException(nameof(aboveOffset), "must be >= 1 (above[-1] is at offset-1)");
        if (left.Length < 4) throw new ArgumentException("left must have 4 entries", nameof(left));

        switch (mode)
        {
            case Vp8IntraMode4x4.BDcPred: PredictDc(aboveBuffer, aboveOffset, left, dst, stride); break;
            case Vp8IntraMode4x4.BTmPred: PredictTm(aboveBuffer, aboveOffset, left, dst, stride); break;
            case Vp8IntraMode4x4.BVePred: PredictVe(aboveBuffer, aboveOffset, dst, stride); break;
            case Vp8IntraMode4x4.BHePred: PredictHe(aboveBuffer, aboveOffset, left, dst, stride); break;
            case Vp8IntraMode4x4.BLdPred: PredictLd(aboveBuffer, aboveOffset, dst, stride); break;
            case Vp8IntraMode4x4.BRdPred: PredictRd(aboveBuffer, aboveOffset, left, dst, stride); break;
            case Vp8IntraMode4x4.BVrPred: PredictVr(aboveBuffer, aboveOffset, left, dst, stride); break;
            case Vp8IntraMode4x4.BVlPred: PredictVl(aboveBuffer, aboveOffset, dst, stride); break;
            case Vp8IntraMode4x4.BHdPred: PredictHd(aboveBuffer, aboveOffset, left, dst, stride); break;
            case Vp8IntraMode4x4.BHuPred: PredictHu(left, dst, stride); break;
            default: throw new ArgumentOutOfRangeException(nameof(mode));
        }
    }

    private static void PredictDc(ReadOnlySpan<byte> ab, int o, ReadOnlySpan<byte> left, Span<byte> dst, int stride)
    {
        int sum = ab[o] + ab[o + 1] + ab[o + 2] + ab[o + 3]
                + left[0] + left[1] + left[2] + left[3];
        byte dc = (byte)((sum + 4) >> 3);
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++) dst[r * stride + c] = dc;
    }

    private static void PredictTm(ReadOnlySpan<byte> ab, int o, ReadOnlySpan<byte> left, Span<byte> dst, int stride)
    {
        int topLeft = ab[o - 1];
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
                dst[r * stride + c] = (byte)Clamp255(left[r] + ab[o + c] - topLeft);
    }

    private static void PredictVe(ReadOnlySpan<byte> ab, int o, Span<byte> dst, int stride)
    {
        // VE: filtered vertical. Each column gets AVG3(above[c-1], above[c], above[c+1]).
        // libvpx: H = above[-1], I = above[0], ..., L = above[3], M = above[4]
        int H = ab[o - 1];
        int I = ab[o + 0], J = ab[o + 1], K = ab[o + 2], L = ab[o + 3], M = ab[o + 4];
        byte v0 = (byte)Avg3(H, I, J);
        byte v1 = (byte)Avg3(I, J, K);
        byte v2 = (byte)Avg3(J, K, L);
        byte v3 = (byte)Avg3(K, L, M);
        for (int r = 0; r < 4; r++)
        {
            int row = r * stride;
            dst[row + 0] = v0; dst[row + 1] = v1; dst[row + 2] = v2; dst[row + 3] = v3;
        }
    }

    private static void PredictHe(ReadOnlySpan<byte> ab, int o, ReadOnlySpan<byte> left, Span<byte> dst, int stride)
    {
        // HE: filtered horizontal. Each row gets AVG3(left[r-1], left[r], left[r+1]).
        int H = ab[o - 1];
        int I = left[0], J = left[1], K = left[2], L = left[3];
        byte r0 = (byte)Avg3(H, I, J);
        byte r1 = (byte)Avg3(I, J, K);
        byte r2 = (byte)Avg3(J, K, L);
        byte r3 = (byte)Avg3(K, L, L);
        for (int c = 0; c < 4; c++)
        {
            dst[0 * stride + c] = r0;
            dst[1 * stride + c] = r1;
            dst[2 * stride + c] = r2;
            dst[3 * stride + c] = r3;
        }
    }

    private static void PredictLd(ReadOnlySpan<byte> ab, int o, Span<byte> dst, int stride)
    {
        // D45e: down-left diagonal, VP8 variant. Uses above[0..7].
        int A = ab[o + 0], B = ab[o + 1], C = ab[o + 2], D = ab[o + 3];
        int E = ab[o + 4], F = ab[o + 5], G = ab[o + 6], H = ab[o + 7];
        dst[0 * stride + 0] = (byte)Avg3(A, B, C);
        dst[0 * stride + 1] = dst[1 * stride + 0] = (byte)Avg3(B, C, D);
        dst[0 * stride + 2] = dst[1 * stride + 1] = dst[2 * stride + 0] = (byte)Avg3(C, D, E);
        dst[0 * stride + 3] = dst[1 * stride + 2] = dst[2 * stride + 1] = dst[3 * stride + 0] = (byte)Avg3(D, E, F);
        dst[1 * stride + 3] = dst[2 * stride + 2] = dst[3 * stride + 1] = (byte)Avg3(E, F, G);
        dst[2 * stride + 3] = dst[3 * stride + 2] = (byte)Avg3(F, G, H);
        dst[3 * stride + 3] = (byte)Avg3(G, H, H);
    }

    private static void PredictRd(ReadOnlySpan<byte> ab, int o, ReadOnlySpan<byte> left, Span<byte> dst, int stride)
    {
        // D135: right-down diagonal.
        int I = left[0], J = left[1], K = left[2], L = left[3];
        int X = ab[o - 1];
        int A = ab[o + 0], B = ab[o + 1], C = ab[o + 2], D = ab[o + 3];
        dst[0 * stride + 3] = (byte)Avg3(D, C, B);
        dst[0 * stride + 2] = dst[1 * stride + 3] = (byte)Avg3(C, B, A);
        dst[0 * stride + 1] = dst[1 * stride + 2] = dst[2 * stride + 3] = (byte)Avg3(B, A, X);
        dst[0 * stride + 0] = dst[1 * stride + 1] = dst[2 * stride + 2] = dst[3 * stride + 3] = (byte)Avg3(A, X, I);
        dst[1 * stride + 0] = dst[2 * stride + 1] = dst[3 * stride + 2] = (byte)Avg3(X, I, J);
        dst[2 * stride + 0] = dst[3 * stride + 1] = (byte)Avg3(I, J, K);
        dst[3 * stride + 0] = (byte)Avg3(J, K, L);
    }

    private static void PredictVr(ReadOnlySpan<byte> ab, int o, ReadOnlySpan<byte> left, Span<byte> dst, int stride)
    {
        // D117: vertical-right.
        int I = left[0], J = left[1], K = left[2];
        int X = ab[o - 1];
        int A = ab[o + 0], B = ab[o + 1], C = ab[o + 2], D = ab[o + 3];
        dst[0 * stride + 0] = dst[2 * stride + 1] = (byte)Avg2(X, A);
        dst[0 * stride + 1] = dst[2 * stride + 2] = (byte)Avg2(A, B);
        dst[0 * stride + 2] = dst[2 * stride + 3] = (byte)Avg2(B, C);
        dst[0 * stride + 3] = (byte)Avg2(C, D);
        dst[1 * stride + 0] = dst[3 * stride + 1] = (byte)Avg3(I, X, A);
        dst[1 * stride + 1] = dst[3 * stride + 2] = (byte)Avg3(X, A, B);
        dst[1 * stride + 2] = dst[3 * stride + 3] = (byte)Avg3(A, B, C);
        dst[1 * stride + 3] = (byte)Avg3(B, C, D);
        dst[2 * stride + 0] = (byte)Avg3(J, I, X);
        dst[3 * stride + 0] = (byte)Avg3(K, J, I);
    }

    private static void PredictVl(ReadOnlySpan<byte> ab, int o, Span<byte> dst, int stride)
    {
        // D63e: vertical-left, VP8 variant.
        int A = ab[o + 0], B = ab[o + 1], C = ab[o + 2], D = ab[o + 3];
        int E = ab[o + 4], F = ab[o + 5], G = ab[o + 6];
        dst[0 * stride + 0] = (byte)Avg2(A, B);
        dst[0 * stride + 1] = dst[2 * stride + 0] = (byte)Avg2(B, C);
        dst[0 * stride + 2] = dst[2 * stride + 1] = (byte)Avg2(C, D);
        dst[0 * stride + 3] = dst[2 * stride + 2] = (byte)Avg2(D, E);
        dst[2 * stride + 3] = (byte)Avg3(E, F, G);
        dst[1 * stride + 0] = (byte)Avg3(A, B, C);
        dst[1 * stride + 1] = dst[3 * stride + 0] = (byte)Avg3(B, C, D);
        dst[1 * stride + 2] = dst[3 * stride + 1] = (byte)Avg3(C, D, E);
        dst[1 * stride + 3] = dst[3 * stride + 2] = (byte)Avg3(D, E, F);
        dst[3 * stride + 3] = (byte)Avg3(E, F, G);
    }

    private static void PredictHd(ReadOnlySpan<byte> ab, int o, ReadOnlySpan<byte> left, Span<byte> dst, int stride)
    {
        // D153: horizontal-down.
        int I = left[0], J = left[1], K = left[2], L = left[3];
        int X = ab[o - 1];
        int A = ab[o + 0], B = ab[o + 1], C = ab[o + 2];
        dst[0 * stride + 0] = dst[1 * stride + 2] = (byte)Avg2(I, X);
        dst[1 * stride + 0] = dst[2 * stride + 2] = (byte)Avg2(J, I);
        dst[2 * stride + 0] = dst[3 * stride + 2] = (byte)Avg2(K, J);
        dst[3 * stride + 0] = (byte)Avg2(L, K);
        dst[0 * stride + 3] = (byte)Avg3(A, B, C);
        dst[0 * stride + 2] = (byte)Avg3(X, A, B);
        dst[0 * stride + 1] = dst[1 * stride + 3] = (byte)Avg3(I, X, A);
        dst[1 * stride + 1] = dst[2 * stride + 3] = (byte)Avg3(J, I, X);
        dst[2 * stride + 1] = dst[3 * stride + 3] = (byte)Avg3(K, J, I);
        dst[3 * stride + 1] = (byte)Avg3(L, K, J);
    }

    private static void PredictHu(ReadOnlySpan<byte> left, Span<byte> dst, int stride)
    {
        // D207: horizontal-up.
        int I = left[0], J = left[1], K = left[2], L = left[3];
        dst[0 * stride + 0] = (byte)Avg2(I, J);
        dst[0 * stride + 2] = dst[1 * stride + 0] = (byte)Avg2(J, K);
        dst[1 * stride + 2] = dst[2 * stride + 0] = (byte)Avg2(K, L);
        dst[0 * stride + 1] = (byte)Avg3(I, J, K);
        dst[1 * stride + 1] = dst[0 * stride + 3] = (byte)Avg3(J, K, L);
        dst[2 * stride + 1] = dst[1 * stride + 3] = (byte)Avg3(K, L, L);
        dst[2 * stride + 2] = dst[2 * stride + 3] =
            dst[3 * stride + 0] = dst[3 * stride + 1] =
            dst[3 * stride + 2] = dst[3 * stride + 3] = (byte)L;
    }
}
