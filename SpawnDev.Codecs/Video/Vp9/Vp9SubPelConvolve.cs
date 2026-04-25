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
}
