// Tests for Vp9SubPelConvolve (slice 241).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9SubPelConvolve_FilterBits_Is7()
    {
        Equal(7, Vp9SubPelConvolve.FilterBits);
    }

    [TestMethod]
    public void Vp9SubPelConvolve_Identity_PassesCenter()
    {
        // Sub-pel 0 is { 0, 0, 0, 128, 0, 0, 0, 0 } across all filters.
        // Dot product picks out src[srcStart + 3], divided by 128 (round).
        // With src[3] = 200, output = (200 * 128 + 64) >> 7 = 25664 >> 7 = 200.
        var src = new byte[] { 1, 5, 50, 200, 7, 42, 0, 64 };
        Equal((byte)200, Vp9SubPelConvolve.ConvolveSample(
            src, 0, Vp9InterpFilter.EightTap, 0));
    }

    [TestMethod]
    public void Vp9SubPelConvolve_Bilinear_HalfPel_AveragesCenters()
    {
        // Bilinear at sub-pel 8 is { 0, 0, -12, 76, 76, -12, 0, 0 }.
        // With src[3] = src[4] = 100 and src[2] = src[5] = 100 (constant
        // sequence), dot product = 100 * (-12 + 76 + 76 - 12) = 100 * 128 = 12800.
        // (12800 + 64) >> 7 = 12864 >> 7 = 100. Filter is normalized.
        var src = new byte[] { 50, 50, 100, 100, 100, 100, 50, 50 };
        Equal((byte)100, Vp9SubPelConvolve.ConvolveSample(
            src, 0, Vp9InterpFilter.Bilinear, 8));
    }

    [TestMethod]
    public void Vp9SubPelConvolve_Constant_PreservedAcrossAllSubPel()
    {
        // For a constant input sequence, the filter produces the same
        // constant (since rows sum to 128 and rounding is symmetric).
        var src = new byte[8];
        for (int i = 0; i < 8; i++) src[i] = 137;

        for (int p = 0; p < Vp9SubPelFilters.SubPelShifts; p++)
        {
            Equal((byte)137, Vp9SubPelConvolve.ConvolveSample(
                src, 0, Vp9InterpFilter.EightTap, p));
            Equal((byte)137, Vp9SubPelConvolve.ConvolveSample(
                src, 0, Vp9InterpFilter.EightTapSmooth, p));
            Equal((byte)137, Vp9SubPelConvolve.ConvolveSample(
                src, 0, Vp9InterpFilter.EightTapSharp, p));
            Equal((byte)137, Vp9SubPelConvolve.ConvolveSample(
                src, 0, Vp9InterpFilter.Bilinear, p));
        }
    }

    [TestMethod]
    public void Vp9SubPelConvolve_Identity_AllZerosReturnZero()
    {
        var src = new byte[8];
        Equal((byte)0, Vp9SubPelConvolve.ConvolveSample(
            src, 0, Vp9InterpFilter.EightTap, 0));
        Equal((byte)0, Vp9SubPelConvolve.ConvolveSample(
            src, 0, Vp9InterpFilter.EightTap, 8));
    }

    [TestMethod]
    public void Vp9SubPelConvolve_Identity_AllMaxReturnMax()
    {
        var src = new byte[8];
        for (int i = 0; i < 8; i++) src[i] = 255;
        for (int p = 0; p < Vp9SubPelFilters.SubPelShifts; p++)
        {
            Equal((byte)255, Vp9SubPelConvolve.ConvolveSample(
                src, 0, Vp9InterpFilter.EightTap, p));
        }
    }

    [TestMethod]
    public void Vp9SubPelConvolve_Clamps_NegativeOvershoot()
    {
        // Construct an input that produces a negative pre-clamp sum.
        // EightTap at sub-pel 1: { 0, 1, -5, 126, 8, -3, 1, 0 }.
        // src[2] = 255, others = 0 -> sum = 255 * (-5) = -1275.
        // After rounding + shift: (-1275 + 64) >> 7 = -1211 >> 7 = -10
        // (or ceil-toward-zero depending on implementation; either way negative).
        // Clamp to 0.
        var src = new byte[] { 0, 0, 255, 0, 0, 0, 0, 0 };
        Equal((byte)0, Vp9SubPelConvolve.ConvolveSample(
            src, 0, Vp9InterpFilter.EightTap, 1));
    }

    [TestMethod]
    public void Vp9SubPelConvolve_Clamps_PositiveOvershoot()
    {
        // EightTapSharp at sub-pel 8: { -4, 11, -23, 80, 80, -23, 11, -4 }.
        // src[3] = src[4] = 255, others = 0
        // -> sum = 255 * 80 + 255 * 80 = 40800.
        // (40800 + 64) >> 7 = 40864 >> 7 = 319. Clamp to 255.
        var src = new byte[] { 0, 0, 0, 255, 255, 0, 0, 0 };
        Equal((byte)255, Vp9SubPelConvolve.ConvolveSample(
            src, 0, Vp9InterpFilter.EightTapSharp, 8));
    }

    [TestMethod]
    public void Vp9SubPelConvolve_RejectsBadFilterRowLength()
    {
        var src = new byte[8];
        Throws<ArgumentException>(() =>
            Vp9SubPelConvolve.ConvolveSample(src, 0, new short[] { 0, 0, 0, 128, 0, 0, 0 }));
    }

    [TestMethod]
    public void Vp9SubPelConvolve_RejectsOutOfRangeSrcStart()
    {
        var src = new byte[8];
        var rowArr = new short[] { 0, 0, 0, 128, 0, 0, 0, 0 };
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9SubPelConvolve.ConvolveSample(src, 1, rowArr));
    }
}
