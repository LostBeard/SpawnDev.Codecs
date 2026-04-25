// Tests for Vp9ColorRange + Vp9ColorSpaces helpers (slice 269).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9ColorRange_NumericValues_MatchLibvpx()
    {
        Equal(0, (int)Vp9ColorRange.Studio);
        Equal(1, (int)Vp9ColorRange.Full);
    }

    [TestMethod]
    public void Vp9ColorSpaces_IsRgb_OnlyForSrgb()
    {
        Equal(false, Vp9ColorSpaces.IsRgb(Vp9ColorSpace.Unknown));
        Equal(false, Vp9ColorSpaces.IsRgb(Vp9ColorSpace.Bt601));
        Equal(false, Vp9ColorSpaces.IsRgb(Vp9ColorSpace.Bt709));
        Equal(false, Vp9ColorSpaces.IsRgb(Vp9ColorSpace.Bt2020));
        Equal(true, Vp9ColorSpaces.IsRgb(Vp9ColorSpace.Srgb));
    }

    [TestMethod]
    public void Vp9ColorSpaces_ImpliesFullRange_OnlyForSrgb()
    {
        Equal(false, Vp9ColorSpaces.ImpliesFullRange(Vp9ColorSpace.Unknown));
        Equal(false, Vp9ColorSpaces.ImpliesFullRange(Vp9ColorSpace.Bt601));
        Equal(true, Vp9ColorSpaces.ImpliesFullRange(Vp9ColorSpace.Srgb));
    }
}
