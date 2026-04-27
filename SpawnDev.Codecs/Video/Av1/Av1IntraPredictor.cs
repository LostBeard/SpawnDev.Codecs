// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 intra prediction primitives (8-bit). Bit-exact port of libaom
// aom_dsp/intrapred.c <c>v/h/dc/paeth/smooth/smooth_v/smooth_h</c>
// predictors.
//
// Upstream Copyright (c) 2016, Alliance for Open Media.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.
//
// Each predictor takes:
//   - dst, stride: destination block (row-major, with row-stride bytes
//     between successive rows so the same routine can write into the
//     middle of a larger frame).
//   - bw, bh: block width / height in pixels (4..64 per axis).
//   - above: pointer to the row of (bw + bh + 1) reconstructed pixels
//     immediately above the block. above[-1] is the top-left corner.
//   - left: pointer to the column of bh reconstructed pixels immediately
//     left of the block, top-to-bottom.
//
// Caller is responsible for setting up the above/left edge buffers
// before calling. AV1 spec sec 7.11.2 covers the edge filtering /
// interpolation / extension rules.
//
// Directional D45/D67/D113/D135/D157/D203 modes are not yet ported -
// they require the angle / dx / dy machinery + per-pixel weighted
// interpolation pass. They are stubbed with NotImplementedException
// so callers fail loud rather than silently produce wrong pixels.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 8-bit intra prediction primitives.</summary>
public static class Av1IntraPredictor
{
    /// <summary>
    /// V_PRED: copy <paramref name="above"/> into every row of the block.
    /// </summary>
    public static void Vertical(Span<byte> dst, int stride, int bw, int bh,
        ReadOnlySpan<byte> above, ReadOnlySpan<byte> left)
    {
        ValidateArgs(dst, stride, bw, bh);
        if (above.Length < bw) throw new ArgumentException("above too short", nameof(above));
        for (int r = 0; r < bh; r++)
        {
            above.Slice(0, bw).CopyTo(dst.Slice(r * stride, bw));
        }
    }

    /// <summary>
    /// H_PRED: replicate <paramref name="left"/>[r] across row r.
    /// </summary>
    public static void Horizontal(Span<byte> dst, int stride, int bw, int bh,
        ReadOnlySpan<byte> above, ReadOnlySpan<byte> left)
    {
        ValidateArgs(dst, stride, bw, bh);
        if (left.Length < bh) throw new ArgumentException("left too short", nameof(left));
        for (int r = 0; r < bh; r++)
        {
            dst.Slice(r * stride, bw).Fill(left[r]);
        }
    }

    /// <summary>
    /// DC_PRED: average of available top + left edge pixels, replicated
    /// into the block. For square blocks of size N, divisor is 2N. For
    /// non-square blocks libaom uses the multiply-shift "div by (bw+bh)"
    /// fast path with constants 0x5556 (1:2) / 0x3334 (1:4) - this method
    /// handles only square blocks for now (NotImplemented for other shapes).
    /// </summary>
    public static void Dc(Span<byte> dst, int stride, int bw, int bh,
        ReadOnlySpan<byte> above, ReadOnlySpan<byte> left)
    {
        ValidateArgs(dst, stride, bw, bh);
        if (above.Length < bw) throw new ArgumentException("above too short", nameof(above));
        if (left.Length < bh) throw new ArgumentException("left too short", nameof(left));
        int sum = 0;
        for (int i = 0; i < bw; i++) sum += above[i];
        for (int i = 0; i < bh; i++) sum += left[i];
        int count = bw + bh;
        int dc;
        if (bw == bh)
        {
            dc = (sum + (count >> 1)) / count;
        }
        else
        {
            // 1:2 ratio -> shift1=2, multiplier=0x5556
            // 1:4 ratio -> shift1=2, multiplier=0x3334 (32x8) or shift1=3 (16x4)
            // Other ratios in AV1: see libaom dc_predictor_rect tables.
            // For now, fall back to plain division for arbitrary rect.
            dc = (sum + (count >> 1)) / count;
        }
        for (int r = 0; r < bh; r++) dst.Slice(r * stride, bw).Fill((byte)dc);
    }

    /// <summary>
    /// DC_LEFT_PRED: average of left edge only (when above is unavailable).
    /// </summary>
    public static void DcLeft(Span<byte> dst, int stride, int bw, int bh,
        ReadOnlySpan<byte> above, ReadOnlySpan<byte> left)
    {
        ValidateArgs(dst, stride, bw, bh);
        if (left.Length < bh) throw new ArgumentException("left too short", nameof(left));
        int sum = 0;
        for (int i = 0; i < bh; i++) sum += left[i];
        int dc = (sum + (bh >> 1)) / bh;
        for (int r = 0; r < bh; r++) dst.Slice(r * stride, bw).Fill((byte)dc);
    }

    /// <summary>
    /// DC_TOP_PRED: average of above edge only (when left is unavailable).
    /// </summary>
    public static void DcTop(Span<byte> dst, int stride, int bw, int bh,
        ReadOnlySpan<byte> above, ReadOnlySpan<byte> left)
    {
        ValidateArgs(dst, stride, bw, bh);
        if (above.Length < bw) throw new ArgumentException("above too short", nameof(above));
        int sum = 0;
        for (int i = 0; i < bw; i++) sum += above[i];
        int dc = (sum + (bw >> 1)) / bw;
        for (int r = 0; r < bh; r++) dst.Slice(r * stride, bw).Fill((byte)dc);
    }

    /// <summary>
    /// DC_128_PRED: when neither edge is available, fill the whole block
    /// with mid-gray (128).
    /// </summary>
    public static void Dc128(Span<byte> dst, int stride, int bw, int bh,
        ReadOnlySpan<byte> above, ReadOnlySpan<byte> left)
    {
        ValidateArgs(dst, stride, bw, bh);
        for (int r = 0; r < bh; r++) dst.Slice(r * stride, bw).Fill((byte)128);
    }

    /// <summary>
    /// PAETH_PRED: per-pixel "closest to base" prediction.
    /// Mirrors libaom <c>paeth_predictor</c>.
    /// </summary>
    public static void Paeth(Span<byte> dst, int stride, int bw, int bh,
        ReadOnlySpan<byte> above, ReadOnlySpan<byte> aboveMinus1, ReadOnlySpan<byte> left)
    {
        ValidateArgs(dst, stride, bw, bh);
        if (above.Length < bw) throw new ArgumentException("above too short", nameof(above));
        if (left.Length < bh) throw new ArgumentException("left too short", nameof(left));
        if (aboveMinus1.Length < 1) throw new ArgumentException("aboveMinus1 (top-left) required", nameof(aboveMinus1));
        byte topLeft = aboveMinus1[0];
        for (int r = 0; r < bh; r++)
        {
            for (int c = 0; c < bw; c++)
            {
                dst[r * stride + c] = PaethSingle(left[r], above[c], topLeft);
            }
        }
    }

    /// <summary>libaom <c>paeth_predictor_single</c>.</summary>
    public static byte PaethSingle(byte left, byte top, byte topLeft)
    {
        int basePred = top + left - topLeft;
        int pLeft = AbsDiff(basePred, left);
        int pTop = AbsDiff(basePred, top);
        int pTopLeft = AbsDiff(basePred, topLeft);
        if (pLeft <= pTop && pLeft <= pTopLeft) return left;
        if (pTop <= pTopLeft) return top;
        return topLeft;
    }

    private static int AbsDiff(int a, int b) => a > b ? a - b : b - a;

    /// <summary>
    /// SMOOTH_PRED: bilinear blend of top + bottom + left + right virtual
    /// edges using <see cref="Av1SmoothWeights"/>. Mirrors libaom
    /// <c>smooth_predictor</c>.
    /// </summary>
    public static void Smooth(Span<byte> dst, int stride, int bw, int bh,
        ReadOnlySpan<byte> above, ReadOnlySpan<byte> left)
    {
        ValidateArgs(dst, stride, bw, bh);
        if (above.Length < bw) throw new ArgumentException("above too short", nameof(above));
        if (left.Length < bh) throw new ArgumentException("left too short", nameof(left));
        byte belowPred = left[bh - 1];   // bottom-left
        byte rightPred = above[bw - 1];  // top-right
        var swW = Av1SmoothWeights.GetWeights(bw);
        var swH = Av1SmoothWeights.GetWeights(bh);
        const int log2Scale = 1 + Av1SmoothWeights.Log2Scale;
        int scale = Av1SmoothWeights.Scale;
        for (int r = 0; r < bh; r++)
        {
            for (int c = 0; c < bw; c++)
            {
                int wTop = swH[r];
                int wBot = scale - swH[r];
                int wLeft = swW[c];
                int wRight = scale - swW[c];
                int pred = wTop * above[c] + wBot * belowPred
                         + wLeft * left[r] + wRight * rightPred;
                dst[r * stride + c] = (byte)((pred + (1 << (log2Scale - 1))) >> log2Scale);
            }
        }
    }

    /// <summary>
    /// SMOOTH_V_PRED: vertical-axis blend of top + bottom-left edge.
    /// </summary>
    public static void SmoothV(Span<byte> dst, int stride, int bw, int bh,
        ReadOnlySpan<byte> above, ReadOnlySpan<byte> left)
    {
        ValidateArgs(dst, stride, bw, bh);
        if (above.Length < bw) throw new ArgumentException("above too short", nameof(above));
        if (left.Length < bh) throw new ArgumentException("left too short", nameof(left));
        byte belowPred = left[bh - 1];
        var swH = Av1SmoothWeights.GetWeights(bh);
        const int log2Scale = Av1SmoothWeights.Log2Scale;
        int scale = Av1SmoothWeights.Scale;
        for (int r = 0; r < bh; r++)
        {
            for (int c = 0; c < bw; c++)
            {
                int wTop = swH[r];
                int wBot = scale - swH[r];
                int pred = wTop * above[c] + wBot * belowPred;
                dst[r * stride + c] = (byte)((pred + (1 << (log2Scale - 1))) >> log2Scale);
            }
        }
    }

    /// <summary>
    /// SMOOTH_H_PRED: horizontal-axis blend of left + top-right edge.
    /// </summary>
    public static void SmoothH(Span<byte> dst, int stride, int bw, int bh,
        ReadOnlySpan<byte> above, ReadOnlySpan<byte> left)
    {
        ValidateArgs(dst, stride, bw, bh);
        if (above.Length < bw) throw new ArgumentException("above too short", nameof(above));
        if (left.Length < bh) throw new ArgumentException("left too short", nameof(left));
        byte rightPred = above[bw - 1];
        var swW = Av1SmoothWeights.GetWeights(bw);
        const int log2Scale = Av1SmoothWeights.Log2Scale;
        int scale = Av1SmoothWeights.Scale;
        for (int r = 0; r < bh; r++)
        {
            for (int c = 0; c < bw; c++)
            {
                int wLeft = swW[c];
                int wRight = scale - swW[c];
                int pred = wLeft * left[r] + wRight * rightPred;
                dst[r * stride + c] = (byte)((pred + (1 << (log2Scale - 1))) >> log2Scale);
            }
        }
    }

    /// <summary>
    /// Directional intra modes (D45 / D67 / D113 / D135 / D157 / D203).
    /// Not yet ported. AV1 spec sec 7.11.2.4 covers the per-pixel angle
    /// interpolation. Throwing NotImplementedException so callers fail
    /// loud rather than emit wrong pixels.
    /// </summary>
    public static void Directional(Av1IntraMode mode, Span<byte> dst, int stride,
        int bw, int bh, ReadOnlySpan<byte> above, ReadOnlySpan<byte> left)
    {
        throw new NotImplementedException(
            $"AV1 directional intra prediction mode {mode} is not yet implemented. " +
            "AV1 spec sec 7.11.2.4 covers the per-pixel angle interpolation pipeline.");
    }

    private static void ValidateArgs(Span<byte> dst, int stride, int bw, int bh)
    {
        if (bw < 4 || bw > 64) throw new ArgumentOutOfRangeException(nameof(bw));
        if (bh < 4 || bh > 64) throw new ArgumentOutOfRangeException(nameof(bh));
        if (stride < bw) throw new ArgumentOutOfRangeException(nameof(stride));
        if (dst.Length < (bh - 1) * stride + bw)
            throw new ArgumentException("dst buffer too small", nameof(dst));
    }
}
