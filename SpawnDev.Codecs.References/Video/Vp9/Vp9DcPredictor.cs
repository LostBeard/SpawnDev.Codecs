// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 DC intra predictor - CPU reference. The simplest of VP9's 10
// intra prediction modes: every pixel of the predicted block is the
// rounded average of the available edge samples. Four variants
// depending on which edges are available:
//
//   Both edges (above + left): DC = (sum_above + sum_left + N) >> log2(2N)
//   Top only:                  DC = (sum_above + N/2) >> log2(N)
//   Left only:                 DC = (sum_left + N/2) >> log2(N)
//   Neither:                   DC = 128
//
// libvpx reference: vpx_dsp/intrapred.c
//   vpx_dc_predictor_4x4_c / 8x8_c / 16x16_c / 32x32_c
//   vpx_dc_top_predictor_*  / vpx_dc_left_predictor_*  / vpx_dc_128_predictor_*
//
// VP9 spec: sec 8.5.1 "Intra frame prediction process" (DC mode).
//
// All four supported transform sizes (4, 8, 16, 32) share the same
// arithmetic shape; only the shift count differs. Implemented as a
// single function with the size + variant as parameters.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 DC intra prediction CPU reference. Bit-exact against libvpx
/// vpx_dc_predictor / vpx_dc_top_predictor / vpx_dc_left_predictor /
/// vpx_dc_128_predictor across all four transform sizes (4, 8, 16, 32).
/// </summary>
public static class Vp9DcPredictor
{
    /// <summary>
    /// DC prediction with both above and left edges available.
    /// </summary>
    /// <param name="above">N samples from the row above the block.</param>
    /// <param name="left">N samples from the column left of the block.</param>
    /// <param name="dst">Destination block (n*stride bytes).</param>
    /// <param name="n">Block size (4, 8, 16, or 32).</param>
    /// <param name="stride">Stride in bytes for <paramref name="dst"/>.</param>
    public static void DcPredict(
        ReadOnlySpan<byte> above,
        ReadOnlySpan<byte> left,
        Span<byte> dst, int n, int stride)
    {
        ValidateSize(n);
        if (above.Length < n)
            throw new ArgumentException($"above must hold {n} samples", nameof(above));
        if (left.Length < n)
            throw new ArgumentException($"left must hold {n} samples", nameof(left));
        if (stride < n)
            throw new ArgumentException("stride must be >= n", nameof(stride));
        if (dst.Length < (n - 1) * stride + n)
            throw new ArgumentException("dst too small for n rows at the given stride", nameof(dst));

        int sum = 0;
        for (int i = 0; i < n; i++) sum += above[i];
        for (int i = 0; i < n; i++) sum += left[i];
        // (sum + N) >> log2(2N) per VP9 spec sec 8.5.1.
        int shift = LogN(n) + 1;
        byte dc = (byte)((sum + n) >> shift);
        FillBlock(dst, dc, n, stride);
    }

    /// <summary>
    /// DC prediction with only the above row available (left edge
    /// out of frame). Matches libvpx vpx_dc_top_predictor_NxN.
    /// </summary>
    public static void DcPredictTop(
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

        int sum = 0;
        for (int i = 0; i < n; i++) sum += above[i];
        int shift = LogN(n);
        byte dc = (byte)((sum + (n >> 1)) >> shift);
        FillBlock(dst, dc, n, stride);
    }

    /// <summary>
    /// DC prediction with only the left column available (top edge
    /// out of frame). Matches libvpx vpx_dc_left_predictor_NxN.
    /// </summary>
    public static void DcPredictLeft(
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

        int sum = 0;
        for (int i = 0; i < n; i++) sum += left[i];
        int shift = LogN(n);
        byte dc = (byte)((sum + (n >> 1)) >> shift);
        FillBlock(dst, dc, n, stride);
    }

    /// <summary>
    /// DC prediction with neither edge available (top-left corner of
    /// the frame). Output is a flat 128 fill. Matches libvpx
    /// vpx_dc_128_predictor_NxN.
    /// </summary>
    public static void DcPredict128(Span<byte> dst, int n, int stride)
    {
        ValidateSize(n);
        if (stride < n)
            throw new ArgumentException("stride must be >= n", nameof(stride));
        if (dst.Length < (n - 1) * stride + n)
            throw new ArgumentException("dst too small", nameof(dst));
        FillBlock(dst, 128, n, stride);
    }

    private static void ValidateSize(int n)
    {
        if (n != 4 && n != 8 && n != 16 && n != 32)
            throw new ArgumentOutOfRangeException(nameof(n), "n must be 4, 8, 16, or 32");
    }

    private static int LogN(int n) => n switch
    {
        4 => 2, 8 => 3, 16 => 4, 32 => 5,
        _ => throw new ArgumentOutOfRangeException(nameof(n)),
    };

    private static void FillBlock(Span<byte> dst, byte value, int n, int stride)
    {
        for (int row = 0; row < n; row++)
        {
            int rowStart = row * stride;
            for (int col = 0; col < n; col++)
                dst[rowStart + col] = value;
        }
    }
}
