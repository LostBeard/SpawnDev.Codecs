// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 TM_PRED ("True Motion") intra predictor - CPU reference. Mirror
// of libvpx vpx_tm_predictor_NxN_c (vpx_dsp/intrapred.c).
//
// TM prediction extrapolates from the corner: each output pixel is
//   dst[r][c] = clip_pixel(left[r] + above[c] - top_left)
// where top_left is the pixel at position (above row, left column)
// of the corner just outside the block. The libvpx convention reads
// top_left as `above[-1]`; this port takes it as an explicit byte
// argument to keep the API span-safe.
//
// VP9 spec sec 8.5.1 "Intra frame prediction process" (TM mode).

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 TM_PRED ("True Motion") intra predictor CPU reference.
/// Bit-exact against libvpx vpx_tm_predictor_NxN.
/// </summary>
public static class Vp9TmPredictor
{
    /// <summary>
    /// True-Motion prediction: each output pixel is
    /// <c>clip_pixel(left[r] + above[c] - <paramref name="topLeft"/>)</c>.
    /// </summary>
    /// <param name="topLeft">
    /// The corner pixel diagonally above-left of the block (libvpx's
    /// <c>above[-1]</c>). Must be supplied separately because the
    /// span-based API doesn't allow negative indexing.
    /// </param>
    /// <param name="above">N samples from the row above the block.</param>
    /// <param name="left">N samples from the column left of the block.</param>
    /// <param name="dst">Destination block (n*stride bytes).</param>
    /// <param name="n">Block size (4, 8, 16, or 32).</param>
    /// <param name="stride">Stride in bytes for <paramref name="dst"/>.</param>
    public static void TmPredict(
        byte topLeft,
        ReadOnlySpan<byte> above,
        ReadOnlySpan<byte> left,
        Span<byte> dst, int n, int stride)
    {
        if (n != 4 && n != 8 && n != 16 && n != 32)
            throw new ArgumentOutOfRangeException(nameof(n), "n must be 4, 8, 16, or 32");
        if (above.Length < n)
            throw new ArgumentException($"above must hold {n} samples", nameof(above));
        if (left.Length < n)
            throw new ArgumentException($"left must hold {n} samples", nameof(left));
        if (stride < n)
            throw new ArgumentException("stride must be >= n", nameof(stride));
        if (dst.Length < (n - 1) * stride + n)
            throw new ArgumentException("dst too small", nameof(dst));

        int tl = topLeft;
        for (int row = 0; row < n; row++)
        {
            int rowStart = row * stride;
            int leftR = left[row];
            for (int col = 0; col < n; col++)
            {
                int v = leftR + above[col] - tl;
                if (v < 0) v = 0;
                else if (v > 255) v = 255;
                dst[rowStart + col] = (byte)v;
            }
        }
    }
}
