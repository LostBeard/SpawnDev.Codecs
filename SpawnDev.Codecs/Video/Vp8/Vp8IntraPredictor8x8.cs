// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP8 8x8 chroma intra prediction. Same four modes as 16x16 luma (RFC 6386
// sec 12.2):
//   DC_PRED - average of 8 above + 8 left samples
//   V_PRED  - copy above row down
//   H_PRED  - copy left column right
//   TM_PRED - above[c] + left[r] - top_left, clamped
//
// Used for both U and V planes (called twice per macroblock).

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>VP8 8x8 chroma intra prediction.</summary>
public static class Vp8IntraPredictor8x8
{
    /// <summary>
    /// Predict an 8x8 chroma block. Use Vp8IntraMode16x16 for the mode enum
    /// since it's the same alphabet (DC / V / H / TM).
    /// </summary>
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
            for (int i = 0; i < 8; i++) sum += above[i] + left[i];
            dc = (sum + 8) >> 4;
        }
        else if (haveAbove)
        {
            int sum = 0;
            for (int i = 0; i < 8; i++) sum += above[i];
            dc = (sum + 4) >> 3;
        }
        else if (haveLeft)
        {
            int sum = 0;
            for (int i = 0; i < 8; i++) sum += left[i];
            dc = (sum + 4) >> 3;
        }
        else
        {
            dc = 128;
        }

        byte dcByte = (byte)dc;
        for (int r = 0; r < 8; r++)
        {
            int row = r * stride;
            for (int c = 0; c < 8; c++) dst[row + c] = dcByte;
        }
    }

    private static void PredictV(ReadOnlySpan<byte> above, Span<byte> dst, int stride)
    {
        for (int r = 0; r < 8; r++)
        {
            int row = r * stride;
            for (int c = 0; c < 8; c++) dst[row + c] = above[c];
        }
    }

    private static void PredictH(ReadOnlySpan<byte> left, Span<byte> dst, int stride)
    {
        for (int r = 0; r < 8; r++)
        {
            int row = r * stride;
            byte v = left[r];
            for (int c = 0; c < 8; c++) dst[row + c] = v;
        }
    }

    private static void PredictTm(
        ReadOnlySpan<byte> above,
        ReadOnlySpan<byte> left,
        byte topLeft,
        Span<byte> dst, int stride)
    {
        for (int r = 0; r < 8; r++)
        {
            int row = r * stride;
            int leftR = left[r];
            for (int c = 0; c < 8; c++)
            {
                int p = leftR + above[c] - topLeft;
                if (p < 0) p = 0;
                else if (p > 255) p = 255;
                dst[row + c] = (byte)p;
            }
        }
    }
}
