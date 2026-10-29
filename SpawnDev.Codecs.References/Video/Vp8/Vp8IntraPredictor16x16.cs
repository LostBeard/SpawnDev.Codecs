// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 16x16 intra prediction. Four modes per RFC 6386 sec 12.2:
//   DC_PRED - average of 16 above + 16 left samples (or 128 when neither
//             available; or just-above or just-left when only one is)
//   V_PRED  - vertical: copy above row down to all 16 rows
//   H_PRED  - horizontal: copy left column right to all 16 cols
//   TM_PRED - TrueMotion: pixel[r][c] = above[c] + left[r] - top_left,
//                          clamped to [0, 255]
//
// The same four modes apply to 8x8 chroma intra prediction (with 8 above
// + 8 left); see Vp8IntraPredictor8x8 for that variant.
//
// Reference: libvpx vpx_dsp/intrapred.c (the dc_*_predictor_* /
// h_predictor_* / v_predictor_* / tm_predictor_* functions).

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>VP8 16x16 intra prediction mode (RFC 6386 sec 12.2).</summary>
public enum Vp8IntraMode16x16 : byte
{
    /// <summary>DC predictor - average of available above + left samples.</summary>
    DcPred = 0,
    /// <summary>Vertical predictor - copy the above row down.</summary>
    VPred = 1,
    /// <summary>Horizontal predictor - copy the left column right.</summary>
    HPred = 2,
    /// <summary>TrueMotion predictor - above + left - top-left, clamped.</summary>
    TmPred = 3,
}

/// <summary>VP8 16x16 intra prediction.</summary>
public static class Vp8IntraPredictor16x16
{
    /// <summary>
    /// Predict a 16x16 luma block. Mirrors libvpx <c>vp8_build_intra_predictors_mby_s</c>.
    /// </summary>
    /// <param name="mode">Prediction mode.</param>
    /// <param name="above">16 above samples (caller-supplied; out-of-frame slots filled with 127 per VP8 convention).</param>
    /// <param name="left">16 left samples (caller-supplied; out-of-frame slots filled with 129).</param>
    /// <param name="topLeft">Diagonally above-left sample (used by TmPred only; default 128 when out of frame).</param>
    /// <param name="haveAbove">True if the row above the block is in-frame.</param>
    /// <param name="haveLeft">True if the column left of the block is in-frame.</param>
    /// <param name="dst">Destination 16x16 block at <paramref name="stride"/>.</param>
    /// <param name="stride">Stride of dst in bytes.</param>
    public static void Predict(
        Vp8IntraMode16x16 mode,
        ReadOnlySpan<byte> above,
        ReadOnlySpan<byte> left,
        byte topLeft,
        bool haveAbove,
        bool haveLeft,
        Span<byte> dst, int stride)
    {
        switch (mode)
        {
            case Vp8IntraMode16x16.DcPred: PredictDc(above, left, haveAbove, haveLeft, dst, stride); break;
            case Vp8IntraMode16x16.VPred:  PredictV(above, dst, stride); break;
            case Vp8IntraMode16x16.HPred:  PredictH(left, dst, stride); break;
            case Vp8IntraMode16x16.TmPred: PredictTm(above, left, topLeft, dst, stride); break;
            default: throw new ArgumentOutOfRangeException(nameof(mode));
        }
    }

    private static void PredictDc(
        ReadOnlySpan<byte> above,
        ReadOnlySpan<byte> left,
        bool haveAbove,
        bool haveLeft,
        Span<byte> dst, int stride)
    {
        int dc;
        if (haveAbove && haveLeft)
        {
            int sum = 0;
            for (int i = 0; i < 16; i++) sum += above[i] + left[i];
            dc = (sum + 16) >> 5;
        }
        else if (haveAbove)
        {
            int sum = 0;
            for (int i = 0; i < 16; i++) sum += above[i];
            dc = (sum + 8) >> 4;
        }
        else if (haveLeft)
        {
            int sum = 0;
            for (int i = 0; i < 16; i++) sum += left[i];
            dc = (sum + 8) >> 4;
        }
        else
        {
            dc = 128;
        }

        byte dcByte = (byte)dc;
        for (int r = 0; r < 16; r++)
        {
            int row = r * stride;
            for (int c = 0; c < 16; c++) dst[row + c] = dcByte;
        }
    }

    private static void PredictV(ReadOnlySpan<byte> above, Span<byte> dst, int stride)
    {
        for (int r = 0; r < 16; r++)
        {
            int row = r * stride;
            for (int c = 0; c < 16; c++) dst[row + c] = above[c];
        }
    }

    private static void PredictH(ReadOnlySpan<byte> left, Span<byte> dst, int stride)
    {
        for (int r = 0; r < 16; r++)
        {
            int row = r * stride;
            byte v = left[r];
            for (int c = 0; c < 16; c++) dst[row + c] = v;
        }
    }

    private static void PredictTm(
        ReadOnlySpan<byte> above,
        ReadOnlySpan<byte> left,
        byte topLeft,
        Span<byte> dst, int stride)
    {
        for (int r = 0; r < 16; r++)
        {
            int row = r * stride;
            int leftR = left[r];
            for (int c = 0; c < 16; c++)
            {
                int p = leftR + above[c] - topLeft;
                if (p < 0) p = 0;
                else if (p > 255) p = 255;
                dst[row + c] = (byte)p;
            }
        }
    }
}
