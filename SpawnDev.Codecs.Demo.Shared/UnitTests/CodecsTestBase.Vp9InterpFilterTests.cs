// Tests for Vp9InterpFilterParser (slice 209).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9InterpFilter_Switchable_FirstBitOne()
    {
        var data = BitsToBytes((1, 1));  // bit 0 = 1 -> Switchable, no further bits
        var f = Vp9InterpFilterParser.Parse(data);
        Equal(Vp9InterpFilter.Switchable, f);
    }

    [TestMethod]
    public void Vp9InterpFilter_EightTap_BitZeroIndex0()
    {
        var data = BitsToBytes((0, 1), (0, 2));  // bit 0 = 0, then 2 bits = 0
        var f = Vp9InterpFilterParser.Parse(data);
        Equal(Vp9InterpFilter.EightTap, f);
    }

    [TestMethod]
    public void Vp9InterpFilter_EightTapSmooth_BitZeroIndex1()
    {
        var data = BitsToBytes((0, 1), (1, 2));
        var f = Vp9InterpFilterParser.Parse(data);
        Equal(Vp9InterpFilter.EightTapSmooth, f);
    }

    [TestMethod]
    public void Vp9InterpFilter_EightTapSharp_BitZeroIndex2()
    {
        var data = BitsToBytes((0, 1), (2, 2));
        var f = Vp9InterpFilterParser.Parse(data);
        Equal(Vp9InterpFilter.EightTapSharp, f);
    }

    [TestMethod]
    public void Vp9InterpFilter_Bilinear_BitZeroIndex3()
    {
        var data = BitsToBytes((0, 1), (3, 2));
        var f = Vp9InterpFilterParser.Parse(data);
        Equal(Vp9InterpFilter.Bilinear, f);
    }
}
