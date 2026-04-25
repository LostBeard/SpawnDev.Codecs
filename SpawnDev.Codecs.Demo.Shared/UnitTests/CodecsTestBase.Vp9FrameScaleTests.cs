// Tests for Vp9FrameScale (slice 258).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9FrameScale_Identity_HasNoScaleAndStep16()
    {
        var s = Vp9FrameScale.Identity;
        Equal(Vp9ScaleFactors.RefNoScale, s.XScale);
        Equal(Vp9ScaleFactors.RefNoScale, s.YScale);
        Equal(16, s.XStep);
        Equal(16, s.YStep);
        Equal(true, s.IsNoScale);
    }

    [TestMethod]
    public void Vp9FrameScale_Compute_SameDimensions_IsIdentity()
    {
        var s = Vp9FrameScale.Compute(1280, 720, 1280, 720);
        Equal(Vp9ScaleFactors.RefNoScale, s.XScale);
        Equal(Vp9ScaleFactors.RefNoScale, s.YScale);
        Equal(16, s.XStep);
        Equal(16, s.YStep);
        Equal(true, s.IsNoScale);
    }

    [TestMethod]
    public void Vp9FrameScale_Compute_RefBigger_StepLarger()
    {
        // 2x larger reference (current 640x360, ref 1280x720) -> step 32.
        var s = Vp9FrameScale.Compute(640, 360, 1280, 720);
        Equal(2 * Vp9ScaleFactors.RefNoScale, s.XScale);
        Equal(2 * Vp9ScaleFactors.RefNoScale, s.YScale);
        Equal(32, s.XStep);
        Equal(32, s.YStep);
        Equal(false, s.IsNoScale);
    }

    [TestMethod]
    public void Vp9FrameScale_Compute_RefSmaller_StepSmaller()
    {
        // 0.5x reference -> step 8.
        var s = Vp9FrameScale.Compute(1280, 720, 640, 360);
        Equal(8192, s.XScale);
        Equal(8192, s.YScale);
        Equal(8, s.XStep);
        Equal(8, s.YStep);
    }

    [TestMethod]
    public void Vp9FrameScale_Compute_AnamorphicScale()
    {
        // Same vertical, 2x horizontal: x_step = 32, y_step = 16.
        var s = Vp9FrameScale.Compute(640, 720, 1280, 720);
        Equal(32, s.XStep);
        Equal(16, s.YStep);
        Equal(false, s.IsNoScale);
    }

    [TestMethod]
    public void Vp9FrameScale_RecordEquality()
    {
        var a = Vp9FrameScale.Compute(640, 360, 1280, 720);
        var b = Vp9FrameScale.Compute(640, 360, 1280, 720);
        Equal(a, b);
        Equal(a.GetHashCode(), b.GetHashCode());
    }

    [TestMethod]
    public void Vp9FrameScale_Compute_RejectsZeroOrNegativeDimensions()
    {
        Throws<ArgumentOutOfRangeException>(() => Vp9FrameScale.Compute(0, 720, 1280, 720));
        Throws<ArgumentOutOfRangeException>(() => Vp9FrameScale.Compute(640, -1, 1280, 720));
        Throws<ArgumentOutOfRangeException>(() => Vp9FrameScale.Compute(640, 720, 0, 720));
        Throws<ArgumentOutOfRangeException>(() => Vp9FrameScale.Compute(640, 720, 1280, -1));
    }
}
