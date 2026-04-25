// Tests for Vp9FrameSizeLimits (slice 272).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9FrameSizeLimits_Constants_MatchSpec()
    {
        Equal(1, Vp9FrameSizeLimits.MinWidth);
        Equal(65536, Vp9FrameSizeLimits.MaxWidth);
        Equal(1, Vp9FrameSizeLimits.MinHeight);
        Equal(65536, Vp9FrameSizeLimits.MaxHeight);
    }

    [TestMethod]
    public void Vp9FrameSizeLimits_IsValid_TypicalSizes()
    {
        Equal(true, Vp9FrameSizeLimits.IsValid(1280, 720));
        Equal(true, Vp9FrameSizeLimits.IsValid(1920, 1080));
        Equal(true, Vp9FrameSizeLimits.IsValid(3840, 2160));
        Equal(true, Vp9FrameSizeLimits.IsValid(7680, 4320));
        Equal(true, Vp9FrameSizeLimits.IsValid(1, 1));
        Equal(true, Vp9FrameSizeLimits.IsValid(65536, 65536));
    }

    [TestMethod]
    public void Vp9FrameSizeLimits_IsValid_RejectsOutOfRange()
    {
        Equal(false, Vp9FrameSizeLimits.IsValid(0, 720));
        Equal(false, Vp9FrameSizeLimits.IsValid(1280, 0));
        Equal(false, Vp9FrameSizeLimits.IsValid(65537, 720));
        Equal(false, Vp9FrameSizeLimits.IsValid(1280, 65537));
        Equal(false, Vp9FrameSizeLimits.IsValid(-1, 720));
    }

    [TestMethod]
    public void Vp9FrameSizeLimits_Validate_DoesNotThrowForValid()
    {
        Vp9FrameSizeLimits.Validate(1280, 720);
        Vp9FrameSizeLimits.Validate(1, 1);
        Vp9FrameSizeLimits.Validate(65536, 65536);
    }

    [TestMethod]
    public void Vp9FrameSizeLimits_Validate_ThrowsForInvalid()
    {
        Throws<ArgumentOutOfRangeException>(() => Vp9FrameSizeLimits.Validate(0, 720));
        Throws<ArgumentOutOfRangeException>(() => Vp9FrameSizeLimits.Validate(1280, 65537));
        Throws<ArgumentOutOfRangeException>(() => Vp9FrameSizeLimits.Validate(-1, 720));
    }
}
