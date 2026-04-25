// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 V_PRED + H_PRED intra predictors - CPU references for the two
// pure edge-replication intra modes:
//
//   V_PRED:  every row of the output is a copy of the above row
//   H_PRED:  every column of the output replicates the corresponding
//            element of the left column (i.e. dst[r][c] = left[r])
//
// Both modes work at all four supported transform sizes (4, 8, 16,
// 32). No arithmetic; just memory motion. libvpx reference:
// vpx_dsp/intrapred.c (vpx_v_predictor_NxN_c, vpx_h_predictor_NxN_c).

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 V_PRED + H_PRED intra predictors. Bit-exact against libvpx.
/// </summary>
public static class Vp9VHPredictor
{
    /// <summary>
    /// V_PRED: copy <paramref name="above"/> to every row of
    /// <paramref name="dst"/>. Block size <paramref name="n"/>;
    /// <paramref name="stride"/> bytes per output row.
    /// </summary>
    public static void VPredict(
        ReadOnlySpan<byte> above,
        Span<byte> dst, int n, int stride)
    {
        ValidateSize(n);
        if (above.Length < n)
            throw new ArgumentException($"above must hold {n} samples", nameof(above));
        if (stride < n)
            throw new ArgumentException("stride must be >= n", nameof(stride));
        if (dst.Length < (n - 1) * stride + n)
            throw new ArgumentException("dst too small", nameof(dst));

        for (int row = 0; row < n; row++)
        {
            int rowStart = row * stride;
            for (int col = 0; col < n; col++)
                dst[rowStart + col] = above[col];
        }
    }

    /// <summary>
    /// H_PRED: every column of the output replicates the
    /// corresponding element of <paramref name="left"/> across the
    /// row. Block size <paramref name="n"/>;
    /// <paramref name="stride"/> bytes per output row.
    /// </summary>
    public static void HPredict(
        ReadOnlySpan<byte> left,
        Span<byte> dst, int n, int stride)
    {
        ValidateSize(n);
        if (left.Length < n)
            throw new ArgumentException($"left must hold {n} samples", nameof(left));
        if (stride < n)
            throw new ArgumentException("stride must be >= n", nameof(stride));
        if (dst.Length < (n - 1) * stride + n)
            throw new ArgumentException("dst too small", nameof(dst));

        for (int row = 0; row < n; row++)
        {
            byte v = left[row];
            int rowStart = row * stride;
            for (int col = 0; col < n; col++)
                dst[rowStart + col] = v;
        }
    }

    private static void ValidateSize(int n)
    {
        if (n != 4 && n != 8 && n != 16 && n != 32)
            throw new ArgumentOutOfRangeException(nameof(n), "n must be 4, 8, 16, or 32");
    }
}
