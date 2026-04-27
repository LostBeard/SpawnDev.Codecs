// Tests for Vp8Quantizer - VP8 dequantizer Q-index lookup. RFC 6386 sec
// 9.7 + Annex A. Sample-based bit-exact verification against libvpx
// quant_common.c.

using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp8Quantizer_DcQLookup_HasCorrectLengthAndBoundaryValues()
    {
        Equal(128, Vp8Quantizer.DcQLookup.Length);
        Equal(4, Vp8Quantizer.DcQLookup[0]);
        Equal(157, Vp8Quantizer.DcQLookup[127]);
    }

    [TestMethod]
    public void Vp8Quantizer_AcQLookup_HasCorrectLengthAndBoundaryValues()
    {
        Equal(128, Vp8Quantizer.AcQLookup.Length);
        Equal(4, Vp8Quantizer.AcQLookup[0]);
        Equal(284, Vp8Quantizer.AcQLookup[127]);
    }

    [TestMethod]
    public void Vp8Quantizer_Y1Dc_AppliesDelta()
    {
        // Q=37, delta=0 -> dc_qlookup[37] = 28
        Equal(28, Vp8Quantizer.Y1Dc(37, 0));
        // Q=37, delta=5 -> dc_qlookup[42] = 32
        Equal(32, Vp8Quantizer.Y1Dc(37, 5));
        // Q=37, delta=-5 -> dc_qlookup[32] = 23
        Equal(23, Vp8Quantizer.Y1Dc(37, -5));
    }

    [TestMethod]
    public void Vp8Quantizer_Y1Dc_ClampsAtBoundaries()
    {
        // Negative Q clamps to 0 -> dc_qlookup[0] = 4
        Equal(4, Vp8Quantizer.Y1Dc(-10, 0));
        // Q > 127 clamps to 127 -> dc_qlookup[127] = 157
        Equal(157, Vp8Quantizer.Y1Dc(200, 0));
    }

    [TestMethod]
    public void Vp8Quantizer_Y2Dc_DoublesDcValue()
    {
        // Y2 DC = dc_qlookup[Q+delta] * 2
        Equal(56, Vp8Quantizer.Y2Dc(37, 0)); // 28 * 2
        Equal(64, Vp8Quantizer.Y2Dc(37, 5)); // 32 * 2
    }

    [TestMethod]
    public void Vp8Quantizer_Y2Ac_AppliesScaleAndFloor()
    {
        // Y2 AC = (ac_qlookup[Q+delta] * 101581) >> 16 (= 155/100), floor at 8.
        // ac_qlookup[37] = 41. (41 * 101581) >> 16 = 4163821 >> 16 = 63
        Equal(63, Vp8Quantizer.Y2Ac(37, 0));

        // Verify floor: low Q where the math < 8.
        // ac_qlookup[0] = 4. (4 * 101581) >> 16 = 406324 >> 16 = 6. Floor to 8.
        Equal(8, Vp8Quantizer.Y2Ac(0, 0));
    }

    [TestMethod]
    public void Vp8Quantizer_UvDc_ClampsAt132()
    {
        // ac_qlookup[127] = 157, but UV DC clamps at 132.
        Equal(132, Vp8Quantizer.UvDc(127, 0));
        // Below clamp passes through.
        Equal(28, Vp8Quantizer.UvDc(37, 0));
    }

    [TestMethod]
    public void Vp8Quantizer_UvAc_NoSpecialClamp()
    {
        // UV AC has no clamp; just lookup.
        Equal(284, Vp8Quantizer.UvAc(127, 0));
        Equal(41, Vp8Quantizer.UvAc(37, 0));
    }
}
