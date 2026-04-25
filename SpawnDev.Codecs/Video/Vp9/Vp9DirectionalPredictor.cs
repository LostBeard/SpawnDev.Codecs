// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 directional intra predictors. Mirror of libvpx vpx_dsp/intrapred.c
// (vpx_d45_predictor_NxN_c, vpx_d63_predictor_NxN_c, ...).
//
// VP9's directional intra modes extrapolate edge samples along a
// specific compass direction. Each output pixel is a 2- or 3-tap
// filter of edge samples on the same direction line. libvpx
// pre-computes the diagonal of the first row(s) using AVG2 / AVG3
// and then memcpy/memset the rest of the block.
//
// Filter primitives (libvpx):
//   AVG3(a, b, c) = (a + 2*b + c + 2) >> 2
//   AVG2(a, b)    = (a + b + 1) >> 1
//
// Slice 165 ships the two above-row-only modes (no left edge):
//   D45:  45 degrees, looking up-right from the block.
//   D63:  63 degrees, less steep variant of the same direction.
//
// Both modes need above samples beyond the block width because the
// diagonal extends past column N-1. libvpx convention is to provide
// 2N samples (above[0..2N-1]); this port matches that convention.
//
// Spec: VP9 Bitstream Specification sec 8.5.1.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 directional intra predictors. Bit-exact against libvpx
/// vpx_dsp/intrapred.c.
/// </summary>
public static class Vp9DirectionalPredictor
{
    /// <summary>3-tap low-pass filter (libvpx AVG3).</summary>
    private static byte Avg3(byte a, byte b, byte c) => (byte)((a + 2 * b + c + 2) >> 2);

    /// <summary>2-tap average (libvpx AVG2).</summary>
    private static byte Avg2(byte a, byte b) => (byte)((a + b + 1) >> 1);

    /// <summary>
    /// D45 predictor (45 deg, up-right diagonal). Each output pixel
    /// is AVG3 of the three above samples on the diagonal:
    /// <c>dst[r][c] = AVG3(above[r+c], above[r+c+1], above[r+c+2])</c>
    /// for the in-block diagonal cells; the bottom-right corner uses
    /// the right-most above sample directly.
    /// </summary>
    /// <param name="above">
    /// At least 2*<paramref name="n"/> samples. The diagonal reads
    /// above[r+c+2] for r = c = n-1, so above[2n-1] is the maximum
    /// index touched. Caller pads with the right-most edge sample if
    /// the block sits at the right frame boundary.
    /// </param>
    /// <param name="dst">Destination block (n*stride bytes).</param>
    /// <param name="n">Block size (4, 8, 16, or 32).</param>
    /// <param name="stride">Stride in bytes for <paramref name="dst"/>.</param>
    public static void D45Predict(
        ReadOnlySpan<byte> above,
        Span<byte> dst, int n, int stride)
    {
        ValidateSize(n);
        if (above.Length < 2 * n)
            throw new ArgumentException($"above must hold {2 * n} samples for D45", nameof(above));
        if (stride < n)
            throw new ArgumentException("stride must be >= n", nameof(stride));
        if (dst.Length < (n - 1) * stride + n)
            throw new ArgumentException("dst too small", nameof(dst));

        // First row (r=0): AVG3 along above starting at each column;
        // last column is above[n-1] (the corner sample) directly.
        // libvpx defines above_right as above[n-1] (the rightmost
        // in-block above sample) - NOT one of the extension samples.
        byte aboveRight = above[n - 1];
        for (int x = 0; x < n - 1; x++)
            dst[x] = Avg3(above[x], above[x + 1], above[x + 2]);
        dst[n - 1] = aboveRight;

        // Subsequent rows: each row is the previous row's data shifted
        // left by 1, with the right side padded with above_right.
        // libvpx implements this via memcpy + memset; same semantics
        // here using Span copies.
        for (int row = 1; row < n; row++)
        {
            int rowStart = row * stride;
            int copyLen = n - 1 - row;  // bytes carried from row 0
            // Copy dst[row=0][row..row+copyLen-1] -> dst[row][0..copyLen-1]
            for (int i = 0; i < copyLen; i++)
                dst[rowStart + i] = dst[row + i];
            // Fill the remaining (row + 1) cells with above_right.
            // Note: copyLen could be 0 or negative on the bottom-right
            // corner; clamp to >= 0.
            int fillStart = copyLen < 0 ? 0 : copyLen;
            for (int i = fillStart; i < n; i++)
                dst[rowStart + i] = aboveRight;
        }
    }

    /// <summary>
    /// D63 predictor (63 deg, between vertical and 45 diagonal). Two
    /// different per-row filters drive even and odd output rows; the
    /// rest of the block is constructed by memcpy + memset like D45.
    /// </summary>
    public static void D63Predict(
        ReadOnlySpan<byte> above,
        Span<byte> dst, int n, int stride)
    {
        ValidateSize(n);
        if (above.Length < 2 * n)
            throw new ArgumentException($"above must hold {2 * n} samples for D63", nameof(above));
        if (stride < n)
            throw new ArgumentException("stride must be >= n", nameof(stride));
        if (dst.Length < (n - 1) * stride + n)
            throw new ArgumentException("dst too small", nameof(dst));

        // Rows 0 and 1: per-column filtering of three above samples.
        for (int c = 0; c < n; c++)
        {
            dst[c] = Avg2(above[c], above[c + 1]);
            dst[stride + c] = Avg3(above[c], above[c + 1], above[c + 2]);
        }

        // Subsequent rows: each pair of rows shifts the row-0/row-1
        // patterns left by 1, padding the right edge with above[n-1]
        // (the rightmost sample of the in-block above range).
        byte fillVal = above[n - 1];
        for (int r = 2, size = n - 2; r < n; r += 2, size--)
        {
            int evenRowStart = r * stride;
            int oddRowStart = (r + 1) * stride;
            int srcEvenOffset = r >> 1;          // dst row 0 starting col
            int srcOddOffset = stride + (r >> 1); // dst row 1 starting col

            for (int i = 0; i < size; i++)
                dst[evenRowStart + i] = dst[srcEvenOffset + i];
            for (int i = size < 0 ? 0 : size; i < n; i++)
                dst[evenRowStart + i] = fillVal;

            if (r + 1 < n)
            {
                for (int i = 0; i < size; i++)
                    dst[oddRowStart + i] = dst[srcOddOffset + i];
                for (int i = size < 0 ? 0 : size; i < n; i++)
                    dst[oddRowStart + i] = fillVal;
            }
        }
    }

    private static void ValidateSize(int n)
    {
        if (n != 4 && n != 8 && n != 16 && n != 32)
            throw new ArgumentOutOfRangeException(nameof(n), "n must be 4, 8, 16, or 32");
    }
}
