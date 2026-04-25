// Tests for Vp9Profile (slice 267).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9Profile_Constants_MatchLibvpx()
    {
        Equal(0, (int)Vp9Profile.Profile0);
        Equal(1, (int)Vp9Profile.Profile1);
        Equal(2, (int)Vp9Profile.Profile2);
        Equal(3, (int)Vp9Profile.Profile3);
        Equal(4, Vp9Profiles.Count);
    }

    [TestMethod]
    public void Vp9Profile_IsHighBitDepth_OnlyProfiles2And3()
    {
        Equal(false, Vp9Profiles.IsHighBitDepth(Vp9Profile.Profile0));
        Equal(false, Vp9Profiles.IsHighBitDepth(Vp9Profile.Profile1));
        Equal(true, Vp9Profiles.IsHighBitDepth(Vp9Profile.Profile2));
        Equal(true, Vp9Profiles.IsHighBitDepth(Vp9Profile.Profile3));
    }

    [TestMethod]
    public void Vp9Profile_AllowsNonYuv420_OnlyProfiles1And3()
    {
        Equal(false, Vp9Profiles.AllowsNonYuv420(Vp9Profile.Profile0));
        Equal(true, Vp9Profiles.AllowsNonYuv420(Vp9Profile.Profile1));
        Equal(false, Vp9Profiles.AllowsNonYuv420(Vp9Profile.Profile2));
        Equal(true, Vp9Profiles.AllowsNonYuv420(Vp9Profile.Profile3));
    }

    [TestMethod]
    public void Vp9Profile_IsMostPermissive_OnlyProfile3()
    {
        Equal(false, Vp9Profiles.IsMostPermissive(Vp9Profile.Profile0));
        Equal(false, Vp9Profiles.IsMostPermissive(Vp9Profile.Profile1));
        Equal(false, Vp9Profiles.IsMostPermissive(Vp9Profile.Profile2));
        Equal(true, Vp9Profiles.IsMostPermissive(Vp9Profile.Profile3));
    }

    [TestMethod]
    public void Vp9Profile_FlagsCombineCorrectly()
    {
        // Profile 3 = high bit depth AND non-420 capable.
        Equal(true, Vp9Profiles.IsHighBitDepth(Vp9Profile.Profile3) &&
                    Vp9Profiles.AllowsNonYuv420(Vp9Profile.Profile3));
        // Profile 0 = neither.
        Equal(false, Vp9Profiles.IsHighBitDepth(Vp9Profile.Profile0) ||
                     Vp9Profiles.AllowsNonYuv420(Vp9Profile.Profile0));
    }
}
