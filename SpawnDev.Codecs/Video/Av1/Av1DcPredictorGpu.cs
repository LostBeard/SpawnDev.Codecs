// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 DC intra-predictor, GPU-callable form for in-kernel reuse.
// Bit-exact mirror of Av1IntraPredictor.Dc / DcLeft / DcTop / Dc128.
//
// V1 keyframe encoder uses DC_PRED only at fixed block sizes (Tx16x16
// for Y, Tx8x8 for UV). This helper covers all four neighbor
// availability cases (above + left, left only, top only, neither)
// for square blocks - matches the libaom dc_predictor + dc_*_predictor.
//
// Edge buffers are passed as flat ArrayView&lt;byte&gt; with explicit
// base offsets. Caller pre-fills them from the recon plane (or uses
// the default 127/129/128 fill for missing neighbors).

using ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// GPU-callable AV1 DC intra-predictor helper. Fills an N x N block
/// with the AV1 DC prediction value computed from the top + left
/// neighbor pixels.
/// </summary>
public static class Av1DcPredictorGpu
{
    /// <summary>
    /// DC_PRED: average of top + left edge pixels, replicated into
    /// the block (square block of size N x N). Mirrors libaom's
    /// dc_predictor for the square case.
    /// <para>
    /// Reads <paramref name="bw"/> bytes from <paramref name="above"/>
    /// starting at <paramref name="aboveBase"/> and <paramref name="bh"/>
    /// bytes from <paramref name="left"/> starting at
    /// <paramref name="leftBase"/>. Writes <paramref name="bw"/> *
    /// <paramref name="bh"/> bytes to <paramref name="dst"/> starting
    /// at <paramref name="dstBase"/> with stride <paramref name="dstStride"/>.
    /// </para>
    /// </summary>
    public static void DcPred(
        ArrayView<byte> dst, long dstBase, int dstStride,
        ArrayView<byte> above, long aboveBase,
        ArrayView<byte> left, long leftBase,
        int bw, int bh,
        bool haveAbove, bool haveLeft)
    {
        int dc;
        if (haveAbove && haveLeft)
        {
            int sum = 0;
            for (int i = 0; i < bw; i++) sum += above[aboveBase + i];
            for (int i = 0; i < bh; i++) sum += left[leftBase + i];
            int count = bw + bh;
            dc = (sum + (count >> 1)) / count;
        }
        else if (haveLeft)
        {
            int sum = 0;
            for (int i = 0; i < bh; i++) sum += left[leftBase + i];
            dc = (sum + (bh >> 1)) / bh;
        }
        else if (haveAbove)
        {
            int sum = 0;
            for (int i = 0; i < bw; i++) sum += above[aboveBase + i];
            dc = (sum + (bw >> 1)) / bw;
        }
        else
        {
            dc = 128;
        }

        byte dcByte = (byte)dc;
        for (int r = 0; r < bh; r++)
        {
            long row = dstBase + (long)r * dstStride;
            for (int c = 0; c < bw; c++) dst[row + c] = dcByte;
        }
    }
}
