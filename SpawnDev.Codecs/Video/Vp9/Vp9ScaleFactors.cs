// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 between-frame scale factors. Mirror of libvpx
// vp9/common/vp9_scale.h / .c.
//
// VP9 supports inter prediction across frames of different
// resolutions (e.g. when a smaller frame is upscaled by an
// alternate reference). The convolve walker uses x_step_q4 and
// y_step_q4 to adjust the per-output-pixel step into the source
// reference - 16 (= 1.0 in Q4) for same-size, larger when the
// reference is bigger than the current frame, smaller when the
// reference is smaller.
//
// libvpx Q14 fixed-point scale = ((src << 14) + dst/2) / dst.
// Round-to-nearest semantics from the +dst/2 numerator.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 between-frame Q14 scale factor helpers.</summary>
public static class Vp9ScaleFactors
{
    /// <summary>libvpx <c>REF_SCALE_SHIFT</c>.</summary>
    public const int RefScaleShift = 14;

    /// <summary>libvpx <c>REF_NO_SCALE</c> = 1 &lt;&lt; 14 = 16384 (1.0 in Q14).</summary>
    public const int RefNoScale = 1 << RefScaleShift;

    /// <summary>
    /// Compute the Q14 fixed-point scale that maps the current
    /// frame's coordinate space to the reference frame's coordinate
    /// space along one dimension. Round-to-nearest:
    /// <c>((otherSize &lt;&lt; 14) + thisSize/2) / thisSize</c>.
    /// </summary>
    /// <param name="otherSize">Reference frame dimension (e.g. ref_w).</param>
    /// <param name="thisSize">Current frame dimension (e.g. cur_w).</param>
    public static int FixedPointScale(int otherSize, int thisSize)
    {
        if (thisSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(thisSize), thisSize, "thisSize must be > 0.");
        if (otherSize < 0)
            throw new ArgumentOutOfRangeException(nameof(otherSize), otherSize, "otherSize must be >= 0.");
        long num = ((long)otherSize << RefScaleShift) + thisSize / 2;
        return (int)(num / thisSize);
    }

    /// <summary>
    /// Convert a Q14 scale factor to the convolve walker's
    /// <c>step_q4</c> (16 = 1.0 in Q4). Computed as
    /// <c>(scaleFp * 16) &gt;&gt; 14</c>.
    /// </summary>
    public static int StepQ4FromScale(int scaleFp)
    {
        return (scaleFp * Vp9SubPelFilters.SubPelShifts) >> RefScaleShift;
    }

    /// <summary>
    /// True when <paramref name="scaleFp"/> equals
    /// <see cref="RefNoScale"/> (i.e. ref and current frame are the
    /// same size in this dimension).
    /// </summary>
    public static bool IsNoScale(int scaleFp) => scaleFp == RefNoScale;
}
