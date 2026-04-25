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

    [TestMethod]
    public void Vp9SubPelConvolve_Horiz_NoScale_IdentityCopiesPixels()
    {
        // x0_q4 = 0 (sub-pel 0 = identity), x_step_q4 = 16 (1.0 step).
        // Source row of 16 padded pixels (3 left + 8 actual + 5 right).
        var src = new byte[16];
        for (int i = 0; i < 16; i++) src[i] = (byte)((i + 1) * 10);
        var dst = new byte[8];

        // srcStart = 3 means "first output reads src[3..10]" with sub-pel 0 -> src[3+3] = 30+30 = ... wait.
        // sub-pel 0 row is { 0, 0, 0, 128, 0, 0, 0, 0 } -> picks src[idx + 3].
        // With srcStart = 3 + leftPadding=3 = 0... no.
        // Actually horizontal walker reads src[srcStart + 0*stride + 0 - 3 .. + 4]
        // for the first output pixel at x0_q4 = 0 -> srcCol = 0, srcIdx = srcStart - 3.
        // To read src[3..10] for the first output (which is src[6] under identity),
        // srcStart must be 3 - meaning the "source position" is index 3.
        Vp9SubPelConvolve.ConvolveHoriz(
            src, srcStart: 3, srcStride: 16,
            dst, dstStart: 0, dstStride: 8,
            Vp9InterpFilter.EightTap, x0Q4: 0, xStepQ4: 16,
            width: 8, height: 1);

        // Identity at sub-pel 0 with srcStart=3 -> output[x] = src[3 + x + (0>>4) - 3 + 3]
        //                                                 = src[3 + x + 0]
        // Wait, that's not right. Let me trace:
        //   srcCol = (x0_q4 + x*step) >> 4 = (0 + 0*16) >> 4 = 0 for x=0
        //   srcIdx = srcStart + 0*stride + 0 - leftPadding(3) = 3 - 3 = 0
        //   row = subpel-0 = { 0, 0, 0, 128, 0, 0, 0, 0 }
        //   dot picks src[0 + 3] = src[3]
        // For x=1: srcCol = (0 + 16) >> 4 = 1, srcIdx = 3 + 1 - 3 = 1, picks src[1 + 3] = src[4].
        // So output[x] = src[x + 3].
        for (int x = 0; x < 8; x++)
        {
            Equal((byte)((x + 3 + 1) * 10), dst[x]);
        }
    }

    [TestMethod]
    public void Vp9SubPelConvolve_Horiz_ConstantInput_ConstantOutput()
    {
        // Constant input of 100 across all positions; any sub-pel
        // produces 100 (filter rows sum to 128).
        var src = new byte[64];
        for (int i = 0; i < 64; i++) src[i] = 100;
        var dst = new byte[16];

        Vp9SubPelConvolve.ConvolveHoriz(
            src, srcStart: 8, srcStride: 64,
            dst, dstStart: 0, dstStride: 16,
            Vp9InterpFilter.EightTapSharp, x0Q4: 5, xStepQ4: 16,
            width: 16, height: 1);

        for (int x = 0; x < 16; x++)
        {
            Equal((byte)100, dst[x]);
        }
    }

    [TestMethod]
    public void Vp9SubPelConvolve_Vert_NoScale_IdentityCopiesPixels()
    {
        // 1-column source of 16 padded rows, identity sub-pel.
        // Each output row maps to src[srcStart + (3 + y) * stride].
        const int rows = 16, cols = 1, stride = 1;
        var src = new byte[rows * stride];
        for (int y = 0; y < rows; y++) src[y] = (byte)((y + 1) * 10);
        var dst = new byte[8];

        Vp9SubPelConvolve.ConvolveVert(
            src, srcStart: 3, srcStride: stride,
            dst, dstStart: 0, dstStride: cols,
            Vp9InterpFilter.EightTap, y0Q4: 0, yStepQ4: 16,
            width: 1, height: 8);

        for (int y = 0; y < 8; y++)
        {
            Equal((byte)((y + 3 + 1) * 10), dst[y]);
        }
    }

    [TestMethod]
    public void Vp9SubPelConvolve_2D_NoScale_IdentityCopiesPixels()
    {
        // Create a 16x16 source with values src[y, x] = y * 16 + x.
        // Then 2D-convolve a 4x4 region with both sub-pels = 0
        // (identity in both dimensions). Output should match source
        // at the corresponding locations.
        const int sw = 32, sh = 32, stride = sw;
        var src = new byte[sh * stride];
        for (int y = 0; y < sh; y++)
            for (int x = 0; x < sw; x++)
                src[y * stride + x] = (byte)((y * 7 + x * 11) & 0xff);
        var dst = new byte[4 * 4];

        // srcStart = 8*stride + 8 -> source pixel (8, 8) is the "top-left"
        // of the requested output. With identity sub-pel both axes, the
        // output should be src[(8 + y) * stride + (8 + x)] for output (x, y).
        Vp9SubPelConvolve.Convolve2D(
            src, srcStart: 8 * stride + 8, srcStride: stride,
            dst, dstStart: 0, dstStride: 4,
            Vp9InterpFilter.EightTap,
            x0Q4: 0, xStepQ4: 16, y0Q4: 0, yStepQ4: 16,
            width: 4, height: 4);

        for (int y = 0; y < 4; y++)
        {
            for (int x = 0; x < 4; x++)
            {
                Equal((byte)(((8 + y) * 7 + (8 + x) * 11) & 0xff),
                    dst[y * 4 + x]);
            }
        }
    }

    [TestMethod]
    public void Vp9SubPelConvolve_2D_ConstantInput_ConstantOutput()
    {
        // A constant source produces a constant output for any sub-pel
        // pair (both filter passes preserve constants).
        const int stride = 32;
        var src = new byte[32 * stride];
        for (int i = 0; i < src.Length; i++) src[i] = 99;
        var dst = new byte[8 * 8];

        Vp9SubPelConvolve.Convolve2D(
            src, srcStart: 8 * stride + 8, srcStride: stride,
            dst, dstStart: 0, dstStride: 8,
            Vp9InterpFilter.EightTapSharp,
            x0Q4: 5, xStepQ4: 16, y0Q4: 11, yStepQ4: 16,
            width: 8, height: 8);

        for (int i = 0; i < dst.Length; i++)
        {
            Equal((byte)99, dst[i]);
        }
    }

    [TestMethod]
    public void Vp9SubPelConvolve_Vert_ConstantInput_ConstantOutput()
    {
        const int rows = 64, cols = 8, stride = 8;
        var src = new byte[rows * stride];
        for (int i = 0; i < src.Length; i++) src[i] = 137;
        var dst = new byte[8 * 8];

        Vp9SubPelConvolve.ConvolveVert(
            src, srcStart: 8 * stride, srcStride: stride,
            dst, dstStart: 0, dstStride: 8,
            Vp9InterpFilter.EightTapSmooth, y0Q4: 7, yStepQ4: 16,
            width: 8, height: 8);

        for (int i = 0; i < dst.Length; i++)
        {
            Equal((byte)137, dst[i]);
        }
    }

    [TestMethod]
    public void Vp9SubPelConvolve_Horiz_MultiRow_CopiesEachIndependently()
    {
        // 2-row source, identity sub-pel: output rows match source rows
        // (offset by 3 from srcStart per identity tap pattern).
        var src = new byte[2 * 16];
        for (int y = 0; y < 2; y++)
            for (int x = 0; x < 16; x++)
                src[y * 16 + x] = (byte)(y * 100 + x);
        var dst = new byte[2 * 8];

        Vp9SubPelConvolve.ConvolveHoriz(
            src, srcStart: 3, srcStride: 16,
            dst, dstStart: 0, dstStride: 8,
            Vp9InterpFilter.EightTap, x0Q4: 0, xStepQ4: 16,
            width: 8, height: 2);

        for (int y = 0; y < 2; y++)
        {
            for (int x = 0; x < 8; x++)
            {
                Equal((byte)(y * 100 + x + 3), dst[y * 8 + x]);
            }
        }
    }
}
