// Tests for Vp9ColorRange + Vp9ColorSpaces helpers (slice 269).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Category = "CpuOnly")]
    public void Vp9ColorRange_NumericValues_MatchLibvpx()
    {
        CpuOnlyTest();
        Equal(0, (int)Vp9ColorRange.Studio);
        Equal(1, (int)Vp9ColorRange.Full);
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9ColorSpaces_IsRgb_OnlyForSrgb()
    {
        CpuOnlyTest();
        Equal(false, Vp9ColorSpaces.IsRgb(Vp9ColorSpace.Unknown));
        Equal(false, Vp9ColorSpaces.IsRgb(Vp9ColorSpace.Bt601));
        Equal(false, Vp9ColorSpaces.IsRgb(Vp9ColorSpace.Bt709));
        Equal(false, Vp9ColorSpaces.IsRgb(Vp9ColorSpace.Bt2020));
        Equal(true, Vp9ColorSpaces.IsRgb(Vp9ColorSpace.Srgb));
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9ColorSpaces_ImpliesFullRange_OnlyForSrgb()
    {
        CpuOnlyTest();
        Equal(false, Vp9ColorSpaces.ImpliesFullRange(Vp9ColorSpace.Unknown));
        Equal(false, Vp9ColorSpaces.ImpliesFullRange(Vp9ColorSpace.Bt601));
        Equal(true, Vp9ColorSpaces.ImpliesFullRange(Vp9ColorSpace.Srgb));
    }
}
