// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 DC intra predictor, GPU-callable form for in-kernel reuse.
// Bit-exact mirror of Vp9DcPredictor (the CPU reference).
//
// Vp9DcPredict4x4Kernel / 8x8Kernel / 16x16Kernel already wrap this
// math as standalone dispatches that handle batches of independent
// blocks. The static helpers here exist so the per-frame sequential
// encode kernel can run DC predict for ONE block at a time without
// dispatching a separate kernel per block - that's the v3
// host-as-pure-coordinator pattern: a single sequential kernel
// inlines the entire forward pipeline.
//
// VP9 spec: sec 8.5.1 "Intra frame prediction process" (DC mode).
// libvpx reference: vpx_dsp/intrapred.c
//   vpx_dc_predictor_NxN_c
//   vpx_dc_top_predictor_NxN_c
//   vpx_dc_left_predictor_NxN_c
//   vpx_dc_128_predictor_NxN_c
//
// All four sizes share the same arithmetic shape; only the shift
// count differs. The helpers expose Vp9DcVariant routing so callers
// can select Both / TopOnly / LeftOnly / Neither without branching
// across four separate methods.

using ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// GPU-callable VP9 DC intra predictor helpers. Bit-exact mirror of
/// <see cref="Vp9DcPredictor"/> for in-kernel use.
/// </summary>
public static class Vp9DcPredictorGpu
{
    /// <summary>
    /// Predict one NxN DC block. <paramref name="n"/> must be 4, 8,
    /// 16, or 32; <paramref name="variant"/> selects which edges are
    /// available. <paramref name="aboveBase"/> and
    /// <paramref name="leftBase"/> are the byte offsets into the
    /// supplied views; the helper reads N samples from each (only the
    /// edges relevant to the variant are accessed).
    /// </summary>
    /// <param name="above">Above-edge samples (only read when variant uses top).</param>
    /// <param name="aboveBase">Byte offset into <paramref name="above"/>.</param>
    /// <param name="left">Left-edge samples (only read when variant uses left).</param>
    /// <param name="leftBase">Byte offset into <paramref name="left"/>.</param>
    /// <param name="dst">Destination block buffer.</param>
    /// <param name="dstBase">Byte offset of the block's top-left corner in <paramref name="dst"/>.</param>
    /// <param name="dstStride">Stride in bytes between rows of <paramref name="dst"/>.</param>
    /// <param name="n">Block size: 4, 8, 16, or 32.</param>
    /// <param name="variant">Which edges to use; cast from <see cref="Vp9DcVariant"/>.</param>
    public static void Predict(
        ArrayView<byte> above, long aboveBase,
        ArrayView<byte> left, long leftBase,
        ArrayView<byte> dst, long dstBase, int dstStride,
        int n, int variant)
    {
        int log2N = LogN(n);
        byte dc;

        if (variant == (int)Vp9DcVariant.Both)
        {
            int sum = 0;
            for (int i = 0; i < n; i++) sum += above[aboveBase + i];
            for (int i = 0; i < n; i++) sum += left[leftBase + i];
            // (sum + N) >> log2(2N) per VP9 spec sec 8.5.1.
            dc = (byte)((sum + n) >> (log2N + 1));
        }
        else if (variant == (int)Vp9DcVariant.TopOnly)
        {
            int sum = 0;
            for (int i = 0; i < n; i++) sum += above[aboveBase + i];
            dc = (byte)((sum + (n >> 1)) >> log2N);
        }
        else if (variant == (int)Vp9DcVariant.LeftOnly)
        {
            int sum = 0;
            for (int i = 0; i < n; i++) sum += left[leftBase + i];
            dc = (byte)((sum + (n >> 1)) >> log2N);
        }
        else // Neither
        {
            dc = 128;
        }

        // Fill block with the computed DC value.
        for (int row = 0; row < n; row++)
        {
            long rowBase = dstBase + (long)row * dstStride;
            for (int col = 0; col < n; col++)
                dst[rowBase + col] = dc;
        }
    }

    /// <summary>
    /// log2(n) for n in {4, 8, 16, 32}. ILGPU compiles the inline
    /// branch chain cleanly on every backend.
    /// </summary>
    private static int LogN(int n)
    {
        if (n == 4) return 2;
        if (n == 8) return 3;
        if (n == 16) return 4;
        return 5; // n == 32
    }
}
