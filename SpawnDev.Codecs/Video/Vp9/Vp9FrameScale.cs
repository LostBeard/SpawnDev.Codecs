// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 frame scale pair: precomputed (XScale, YScale, XStep, YStep)
// for a (current, reference) frame pair. Mirror of libvpx
// vp9/common/vp9_scale.c vp9_setup_scale_factors_for_frame.
//
// The scale factors are needed once per (current, reference) pair
// and reused for every block that uses that reference. Caching
// them in a small record avoids recomputing the FixedPointScale /
// StepQ4FromScale arithmetic per block.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 frame scale pair (Q14 scale factors + Q4 walker steps).</summary>
public sealed record Vp9FrameScale(int XScale, int YScale, int XStep, int YStep)
{
    /// <summary>True when this frame pair has identical dimensions in both axes.</summary>
    public bool IsNoScale =>
        Vp9ScaleFactors.IsNoScale(XScale) && Vp9ScaleFactors.IsNoScale(YScale);

    /// <summary>
    /// Compute the scale pair for a (current, reference) frame pair
    /// in pixel dimensions. Both dimensions must be positive.
    /// </summary>
    public static Vp9FrameScale Compute(int currentWidth, int currentHeight, int referenceWidth, int referenceHeight)
    {
        if (currentWidth <= 0 || currentHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(currentWidth),
                "current dimensions must be positive.");
        if (referenceWidth <= 0 || referenceHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(referenceWidth),
                "reference dimensions must be positive.");

        int xScale = Vp9ScaleFactors.FixedPointScale(referenceWidth, currentWidth);
        int yScale = Vp9ScaleFactors.FixedPointScale(referenceHeight, currentHeight);
        int xStep = Vp9ScaleFactors.StepQ4FromScale(xScale);
        int yStep = Vp9ScaleFactors.StepQ4FromScale(yScale);
        return new Vp9FrameScale(xScale, yScale, xStep, yStep);
    }

    /// <summary>The identity scale - same-size current and reference frames.</summary>
    public static readonly Vp9FrameScale Identity =
        new Vp9FrameScale(Vp9ScaleFactors.RefNoScale, Vp9ScaleFactors.RefNoScale, 16, 16);
}
