// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 block reconstruction: dst[r][c] = clip_pixel(dst[r][c] + residual[r][c]).
//
// Final step of intra (and inter) block decode. The predicted block
// is already in dst (output of Vp9IntraPredictor / motion comp); the
// inverse transform produced an int16 residual at the same N*N shape;
// this helper combines them with saturating clip to [0, 255].
//
// libvpx reference: vpx_dsp/inv_txfm.c
//   #define clip_pixel(value) (uint8_t)((value) < 0 ? 0 : ((value) > 255 ? 255 : (value)))
//   for (j = 0; j < n; j++)
//     for (i = 0; i < n; i++)
//       dest[i] = clip_pixel(dest[i] + input[i]);
//
// VP9 spec sec 8.7: inverse transform output is stored as the
// residual signal; add to the predicted sample and clip to the
// pixel range.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 block reconstruction: clipped add of the iDCT residual into
/// the predicted block. Bit-exact against libvpx clip_pixel(dest +
/// input) loop in vpx_dsp/inv_txfm.c.
/// </summary>
public static class Vp9Reconstruct
{
    /// <summary>
    /// In-place add of an int16 residual into a uint8 predicted block,
    /// clipped to [0, 255].
    /// </summary>
    /// <param name="dst">
    /// Predicted block on input; reconstructed block on output. Length
    /// must cover (n-1)*stride + n bytes.
    /// </param>
    /// <param name="residual">
    /// N*N int16 residual in row-major order with stride = n. Inverse
    /// transform output. Caller is responsible for any per-coefficient
    /// rounding (ROUND_POWER_OF_TWO etc) before passing in - this
    /// helper does a straight add.
    /// </param>
    /// <param name="n">Block size (4, 8, 16, or 32).</param>
    /// <param name="stride">Stride in bytes for <paramref name="dst"/>.</param>
    public static void AddResidual(
        Span<byte> dst,
        ReadOnlySpan<short> residual,
        int n, int stride)
    {
        ValidateSize(n);
        if (residual.Length < n * n)
            throw new ArgumentException($"residual must hold {n * n} entries", nameof(residual));
        if (stride < n)
            throw new ArgumentException("stride must be >= n", nameof(stride));
        if (dst.Length < (n - 1) * stride + n)
            throw new ArgumentException("dst too small", nameof(dst));

        for (int r = 0; r < n; r++)
        {
            int dstOff = r * stride;
            int resOff = r * n;
            for (int c = 0; c < n; c++)
            {
                int sum = dst[dstOff + c] + residual[resOff + c];
                dst[dstOff + c] = sum < 0 ? (byte)0 : sum > 255 ? (byte)255 : (byte)sum;
            }
        }
    }

    private static void ValidateSize(int n)
    {
        if (n != 4 && n != 8 && n != 16 && n != 32)
            throw new ArgumentOutOfRangeException(nameof(n), "n must be 4, 8, 16, or 32");
    }
}
