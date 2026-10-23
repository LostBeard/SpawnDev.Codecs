// Tests for Vp9ScaleFactors (slice 257).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9ScaleFactors_RefNoScale_Is16384()
    {
        CpuOnlyTest();
        Equal(14, Vp9ScaleFactors.RefScaleShift);
        Equal(16384, Vp9ScaleFactors.RefNoScale);
    }

    [TestMethod]
    public void Vp9ScaleFactors_FixedPointScale_SameSize_IsNoScale()
    {
        CpuOnlyTest();
        Equal(Vp9ScaleFactors.RefNoScale, Vp9ScaleFactors.FixedPointScale(640, 640));
        Equal(Vp9ScaleFactors.RefNoScale, Vp9ScaleFactors.FixedPointScale(1080, 1080));
    }

    [TestMethod]
    public void Vp9ScaleFactors_FixedPointScale_DoubleSize_DoublesScale()
    {
        CpuOnlyTest();
        // ref = 1280, cur = 640 -> scale = 2.0 -> Q14 = 32768.
        Equal(2 * Vp9ScaleFactors.RefNoScale, Vp9ScaleFactors.FixedPointScale(1280, 640));
    }

    [TestMethod]
    public void Vp9ScaleFactors_FixedPointScale_HalfSize_HalvesScale()
    {
        CpuOnlyTest();
        // ref = 320, cur = 640 -> scale = 0.5 -> Q14 = 8192.
        Equal(8192, Vp9ScaleFactors.FixedPointScale(320, 640));
    }

    [TestMethod]
    public void Vp9ScaleFactors_StepQ4FromScale_NoScale_Is16()
    {
        CpuOnlyTest();
        Equal(16, Vp9ScaleFactors.StepQ4FromScale(Vp9ScaleFactors.RefNoScale));
    }

    [TestMethod]
    public void Vp9ScaleFactors_StepQ4FromScale_DoubleScale_Is32()
    {
        CpuOnlyTest();
        Equal(32, Vp9ScaleFactors.StepQ4FromScale(2 * Vp9ScaleFactors.RefNoScale));
    }

    [TestMethod]
    public void Vp9ScaleFactors_StepQ4FromScale_HalfScale_Is8()
    {
        CpuOnlyTest();
        Equal(8, Vp9ScaleFactors.StepQ4FromScale(8192));
    }

    [TestMethod]
    public void Vp9ScaleFactors_IsNoScale_DetectsExactly()
    {
        CpuOnlyTest();
        Equal(true, Vp9ScaleFactors.IsNoScale(Vp9ScaleFactors.RefNoScale));
        Equal(false, Vp9ScaleFactors.IsNoScale(Vp9ScaleFactors.RefNoScale - 1));
        Equal(false, Vp9ScaleFactors.IsNoScale(Vp9ScaleFactors.RefNoScale + 1));
    }

    [TestMethod]
    public void Vp9ScaleFactors_FixedPointScale_RejectsZeroDenominator()
    {
        CpuOnlyTest();
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9ScaleFactors.FixedPointScale(640, 0));
    }
}
