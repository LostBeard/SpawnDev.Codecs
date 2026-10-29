// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 intra prediction edge buffer assembly. Mirrors libaom's intra_edge
// availability + fill-in logic from av1/common/reconintra.c
// <c>build_intra_predictors_high</c>.
//
// Upstream Copyright (c) 2016, Alliance for Open Media. All rights reserved.
// Upstream license: BSD 2-Clause + AV1 Patent License 1.0. See NOTICE.md.
//
// For each tx block, the intra prediction needs:
//   - above[]: bw + bh + 1 reconstructed pixels above the block (top row)
//   - left[]:  bw + bh + 1 reconstructed pixels left of the block (left col)
//   - aboveLeft: the corner pixel
//
// When neighbors are unavailable (frame edge / first row / first col), the
// edge is filled with a default value:
//   - top row unavailable: fill with left[0]
//   - left col unavailable: fill with above[0]
//   - both unavailable: fill with 128 (mid-gray) - DC_128_PRED case
//
// Smooth/Paeth use the corner pixel; directional modes use additional
// pixels beyond bw+bh (up to 2*max(bw,bh) + 1) which we just replicate.
//
// Spec reference: AV1 Bitstream and Decoding Process Specification
//   sec 7.11.2 Intra prediction process

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 intra prediction edge buffer.</summary>
public sealed class Av1IntraEdge
{
    /// <summary>Top edge: 2*max(bw,bh) + 1 pixels (extra room for directional extension).</summary>
    public readonly byte[] Above;
    /// <summary>Left edge: 2*max(bw,bh) + 1 pixels.</summary>
    public readonly byte[] Left;
    /// <summary>Top-left corner pixel.</summary>
    public byte AboveLeft;
    /// <summary>True when the row above the block has reconstructed pixels available.</summary>
    public readonly bool HaveAbove;
    /// <summary>True when the column left of the block has reconstructed pixels available.</summary>
    public readonly bool HaveLeft;

    /// <summary>Construct edge buffers sized for a block of (bw, bh) pixels.</summary>
    public Av1IntraEdge(int bw, int bh, bool haveAbove, bool haveLeft)
    {
        int len = 2 * Math.Max(bw, bh) + 1;
        Above = new byte[len];
        Left = new byte[len];
        HaveAbove = haveAbove;
        HaveLeft = haveLeft;
        AboveLeft = 128;
    }

    /// <summary>
    /// Fill the edge buffers from the supplied frame plane buffer at (xPx, yPx) for a block of (bw, bh).
    /// </summary>
    public static Av1IntraEdge Build(
        byte[] plane, int planeStride, int planeWidth, int planeHeight,
        int xPx, int yPx, int bw, int bh)
    {
        bool haveAbove = yPx > 0;
        bool haveLeft = xPx > 0;
        var edge = new Av1IntraEdge(bw, bh, haveAbove, haveLeft);

        // Per AV1 spec: use 129 (slightly above mid-gray) for missing top
        // and 127 for missing left; libaom uses 129/127 respectively to
        // match HEVC's intra prediction conventions.
        const byte missingAbove = 129;
        const byte missingLeft = 127;
        const byte missingCorner = 128;

        // Above row.
        if (haveAbove)
        {
            int rowOff = (yPx - 1) * planeStride + xPx;
            // Available pixels: from xPx to min(planeWidth, xPx + 2*max(bw,bh))
            int len = Math.Min(2 * Math.Max(bw, bh), planeWidth - xPx);
            for (int i = 0; i < len; i++)
            {
                edge.Above[i] = plane[rowOff + i];
            }
            // Pad with last available pixel.
            byte last = len > 0 ? edge.Above[len - 1] : missingAbove;
            for (int i = len; i < edge.Above.Length; i++) edge.Above[i] = last;
        }
        else
        {
            for (int i = 0; i < edge.Above.Length; i++) edge.Above[i] = missingAbove;
        }

        // Left col.
        if (haveLeft)
        {
            int colOff = yPx * planeStride + (xPx - 1);
            int len = Math.Min(2 * Math.Max(bw, bh), planeHeight - yPx);
            for (int i = 0; i < len; i++)
            {
                edge.Left[i] = plane[colOff + i * planeStride];
            }
            byte last = len > 0 ? edge.Left[len - 1] : missingLeft;
            for (int i = len; i < edge.Left.Length; i++) edge.Left[i] = last;
        }
        else
        {
            for (int i = 0; i < edge.Left.Length; i++) edge.Left[i] = missingLeft;
        }

        // Corner pixel.
        if (haveAbove && haveLeft)
        {
            edge.AboveLeft = plane[(yPx - 1) * planeStride + (xPx - 1)];
        }
        else if (haveAbove)
        {
            edge.AboveLeft = edge.Above[0];
        }
        else if (haveLeft)
        {
            edge.AboveLeft = edge.Left[0];
        }
        else
        {
            edge.AboveLeft = missingCorner;
        }

        return edge;
    }
}
