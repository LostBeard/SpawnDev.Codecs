// Tests for Vp9BitDepth (slice 268).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9BitDepth_NumericValues_AreActualBitsCount()
    {
        Equal(8, (int)Vp9BitDepth.Bits8);
        Equal(10, (int)Vp9BitDepth.Bits10);
        Equal(12, (int)Vp9BitDepth.Bits12);
    }

    [TestMethod]
    public void Vp9BitDepth_Resolve_Profile0Or1_AlwaysBits8()
    {
        Equal(Vp9BitDepth.Bits8, Vp9BitDepths.Resolve(Vp9Profile.Profile0, tenOrTwelveBit: false));
        Equal(Vp9BitDepth.Bits8, Vp9BitDepths.Resolve(Vp9Profile.Profile0, tenOrTwelveBit: true));
        Equal(Vp9BitDepth.Bits8, Vp9BitDepths.Resolve(Vp9Profile.Profile1, tenOrTwelveBit: false));
        Equal(Vp9BitDepth.Bits8, Vp9BitDepths.Resolve(Vp9Profile.Profile1, tenOrTwelveBit: true));
    }

    [TestMethod]
    public void Vp9BitDepth_Resolve_Profile2_FlagPicks10Or12()
    {
        Equal(Vp9BitDepth.Bits10, Vp9BitDepths.Resolve(Vp9Profile.Profile2, tenOrTwelveBit: false));
        Equal(Vp9BitDepth.Bits12, Vp9BitDepths.Resolve(Vp9Profile.Profile2, tenOrTwelveBit: true));
    }

    [TestMethod]
    public void Vp9BitDepth_Resolve_Profile3_FlagPicks10Or12()
    {
        Equal(Vp9BitDepth.Bits10, Vp9BitDepths.Resolve(Vp9Profile.Profile3, tenOrTwelveBit: false));
        Equal(Vp9BitDepth.Bits12, Vp9BitDepths.Resolve(Vp9Profile.Profile3, tenOrTwelveBit: true));
    }

    [TestMethod]
    public void Vp9BitDepth_MaxSampleValue()
    {
        Equal(255, Vp9BitDepths.MaxSampleValue(Vp9BitDepth.Bits8));
        Equal(1023, Vp9BitDepths.MaxSampleValue(Vp9BitDepth.Bits10));
        Equal(4095, Vp9BitDepths.MaxSampleValue(Vp9BitDepth.Bits12));
    }

    [TestMethod]
    public void Vp9BitDepth_MaxQuantizerIndex()
    {
        Equal(255, Vp9BitDepths.MaxQuantizerIndex(Vp9BitDepth.Bits8));
        Equal(1023, Vp9BitDepths.MaxQuantizerIndex(Vp9BitDepth.Bits10));
        Equal(4095, Vp9BitDepths.MaxQuantizerIndex(Vp9BitDepth.Bits12));
    }
}
