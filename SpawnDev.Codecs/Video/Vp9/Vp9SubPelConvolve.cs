// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 sub-pel 1D convolution kernel. Produces one filtered output
// sample by dotting an 8-tap filter row with 8 consecutive source
// samples, then rounding + shifting + clamping to byte range. The
// 2D motion-compensation kernel composes two 1D passes (horizontal
// then vertical) using this helper.
//
// libvpx reference: vp9/common/vp9_convolve.c convolve_horiz /
// convolve_vert inner loop.
//
// Math:
//   sum = sum_{t=0..7} src[srcStart + t] * filterRow[t]
//   sum += 1 << (FILTER_BITS - 1)         // round to nearest
//   sum >>= FILTER_BITS                    // FILTER_BITS = 7
//   return clip_pixel(sum)                 // clamp to [0, 255]

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 sub-pel 1D convolve helper.</summary>
public static class Vp9SubPelConvolve
{
    /// <summary>libvpx <c>FILTER_BITS</c> = 7. Filter scale factor is
    /// 2^FILTER_BITS = 128 (also the value of an identity kernel's
    /// center tap).</summary>
    public const int FilterBits = Vp9SubPelFilters.FilterBits;

    /// <summary>
    /// Convolve 8 source samples with an 8-tap filter row, rounding
    /// +64 (half of 128 = 1 &lt;&lt; (FILTER_BITS - 1)), shifting right
    /// by <see cref="FilterBits"/>, then clamping to byte range.
    /// </summary>
    /// <param name="src">
    /// Source samples; must contain at least <paramref name="srcStart"/>
    /// + 8 elements.
    /// </param>
    /// <param name="srcStart">Index of the first source sample to multiply.</param>
    /// <param name="filterRow">
    /// 8-tap filter row from <see cref="Vp9SubPelFilters.GetRow"/>.
    /// Length must be exactly 8.
    /// </param>
    public static byte ConvolveSample(
        ReadOnlySpan<byte> src,
        int srcStart,
        ReadOnlySpan<short> filterRow)
    {
        if (filterRow.Length != Vp9SubPelFilters.SubPelTaps)
            throw new ArgumentException(
                $"filterRow must hold exactly {Vp9SubPelFilters.SubPelTaps} taps.",
                nameof(filterRow));
        if ((uint)srcStart > (uint)(src.Length - Vp9SubPelFilters.SubPelTaps))
            throw new ArgumentOutOfRangeException(nameof(srcStart), srcStart,
                "srcStart out of range for an 8-tap window.");

        int sum = 0;
        for (int t = 0; t < Vp9SubPelFilters.SubPelTaps; t++)
            sum += src[srcStart + t] * filterRow[t];
        sum += 1 << (FilterBits - 1);
        sum >>= FilterBits;
        if (sum < 0) sum = 0;
        else if (sum > 255) sum = 255;
        return (byte)sum;
    }

    /// <summary>
    /// Look up the filter row for <paramref name="filter"/> and
    /// <paramref name="subPel"/>, then convolve.
    /// </summary>
    public static byte ConvolveSample(
        ReadOnlySpan<byte> src,
        int srcStart,
        Vp9InterpFilter filter,
        int subPel)
    {
        return ConvolveSample(src, srcStart,
            Vp9SubPelFilters.GetRow(filter, subPel));
    }

    /// <summary>
    /// Horizontal 1D convolve over a w x h block. For each output
    /// position (x, y), advances <paramref name="x0Q4"/> by
    /// <paramref name="xStepQ4"/> and convolves 8 source samples
    /// centered on the current sub-pel position.
    ///
    /// Mirror of libvpx <c>convolve_horiz</c> inner walker. Caller
    /// is responsible for source-buffer padding: the convolution
    /// reads <c>src[(x0_q4 &gt;&gt; 4) - 3 ..  (x0_q4 &gt;&gt; 4) + 4]</c>
    /// at each position, so the source buffer must include 3 pixels
    /// of left padding and at least 4 pixels of right padding past
    /// the last accessed integer column.
    /// </summary>
    /// <param name="src">Source buffer (typically a reference frame).</param>
    /// <param name="srcStart">Top-left index of the source region.</param>
    /// <param name="srcStride">Stride between source rows.</param>
    /// <param name="dst">Destination buffer.</param>
    /// <param name="dstStart">Top-left index of the destination region.</param>
    /// <param name="dstStride">Stride between destination rows.</param>
    /// <param name="filter">Interpolation filter selection.</param>
    /// <param name="x0Q4">Initial x position in q4 fraction (per-pixel sub-pel offset).</param>
    /// <param name="xStepQ4">x increment in q4 fraction (16 = 1.0 = no scaling).</param>
    /// <param name="width">Output width in pixels.</param>
    /// <param name="height">Output height in pixels.</param>
    public static void ConvolveHoriz(
        ReadOnlySpan<byte> src, int srcStart, int srcStride,
        Span<byte> dst, int dstStart, int dstStride,
        Vp9InterpFilter filter, int x0Q4, int xStepQ4,
        int width, int height)
    {
        var filterTable = Vp9SubPelFilters.GetFilter(filter);
        const int leftPadding = Vp9SubPelFilters.SubPelTaps / 2 - 1; // 3
        for (int y = 0; y < height; y++)
        {
            int xOffsetQ4 = x0Q4;
            int srcRowStart = srcStart + y * srcStride;
            int dstRowStart = dstStart + y * dstStride;
            for (int x = 0; x < width; x++)
            {
                int srcCol = xOffsetQ4 >> Vp9SubPelFilters.SubPelBits;
                int subPel = xOffsetQ4 & (Vp9SubPelFilters.SubPelShifts - 1);
                int srcIdx = srcRowStart + srcCol - leftPadding;
                var row = filterTable.AsSpan(
                    subPel * Vp9SubPelFilters.SubPelTaps, Vp9SubPelFilters.SubPelTaps);
                dst[dstRowStart + x] = ConvolveSample(src, srcIdx, row);
                xOffsetQ4 += xStepQ4;
            }
        }
    }

    /// <summary>
    /// Vertical 1D convolve over a w x h block. For each output
    /// position (x, y), advances <paramref name="y0Q4"/> by
    /// <paramref name="yStepQ4"/> and convolves 8 source samples
    /// vertically centered on the current sub-pel position.
    ///
    /// Symmetric to <see cref="ConvolveHoriz"/> but stride-aware:
    /// the 8 source samples come from 8 consecutive ROWS at the
    /// same column. Mirror of libvpx <c>convolve_vert</c>.
    ///
    /// Caller is responsible for source-buffer padding: reads
    /// <c>src[(y0_q4 &gt;&gt; 4) - 3 .. (y0_q4 &gt;&gt; 4) + 4]</c> rows
    /// at each position; source buffer must include 3 rows of top
    /// padding and at least 4 rows of bottom padding past the last
    /// accessed integer row.
    /// </summary>
    public static void ConvolveVert(
        ReadOnlySpan<byte> src, int srcStart, int srcStride,
        Span<byte> dst, int dstStart, int dstStride,
        Vp9InterpFilter filter, int y0Q4, int yStepQ4,
        int width, int height)
    {
        var filterTable = Vp9SubPelFilters.GetFilter(filter);
        const int topPadding = Vp9SubPelFilters.SubPelTaps / 2 - 1; // 3
        for (int x = 0; x < width; x++)
        {
            int yOffsetQ4 = y0Q4;
            for (int y = 0; y < height; y++)
            {
                int srcRow = yOffsetQ4 >> Vp9SubPelFilters.SubPelBits;
                int subPel = yOffsetQ4 & (Vp9SubPelFilters.SubPelShifts - 1);
                int srcColStart = srcStart + (srcRow - topPadding) * srcStride + x;
                var filterRow = filterTable.AsSpan(
                    subPel * Vp9SubPelFilters.SubPelTaps, Vp9SubPelFilters.SubPelTaps);

                int sum = 0;
                for (int t = 0; t < Vp9SubPelFilters.SubPelTaps; t++)
                    sum += src[srcColStart + t * srcStride] * filterRow[t];
                sum += 1 << (FilterBits - 1);
                sum >>= FilterBits;
                if (sum < 0) sum = 0;
                else if (sum > 255) sum = 255;
                dst[dstStart + y * dstStride + x] = (byte)sum;

                yOffsetQ4 += yStepQ4;
            }
        }
    }
}
