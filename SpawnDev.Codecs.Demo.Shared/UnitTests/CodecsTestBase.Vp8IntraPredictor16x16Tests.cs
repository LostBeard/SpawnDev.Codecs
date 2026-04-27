// Tests for Vp8IntraPredictor16x16 - VP8 16x16 intra prediction
// (DC / V / H / TM_PRED). RFC 6386 sec 12.2.

using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp8IntraPredictor16x16_DcPred_NoNeighbors_Returns128()
    {
        // No above + no left -> DC = 128 fill.
        Span<byte> above = stackalloc byte[16];
        Span<byte> left = stackalloc byte[16];
        Span<byte> dst = stackalloc byte[16 * 16];

        Vp8IntraPredictor16x16.Predict(
            Vp8IntraMode16x16.DcPred,
            above, left, 0, haveAbove: false, haveLeft: false,
            dst, 16);

        for (int i = 0; i < dst.Length; i++)
            Equal((byte)128, dst[i]);
    }

    [TestMethod]
    public void Vp8IntraPredictor16x16_DcPred_AllOnesNeighbors_ReturnsOne()
    {
        Span<byte> above = stackalloc byte[16];
        Span<byte> left = stackalloc byte[16];
        for (int i = 0; i < 16; i++) { above[i] = 1; left[i] = 1; }
        Span<byte> dst = stackalloc byte[16 * 16];

        Vp8IntraPredictor16x16.Predict(
            Vp8IntraMode16x16.DcPred,
            above, left, 0, haveAbove: true, haveLeft: true,
            dst, 16);

        // sum = 32, dc = (32+16) >> 5 = 1
        for (int i = 0; i < dst.Length; i++)
            Equal((byte)1, dst[i]);
    }

    [TestMethod]
    public void Vp8IntraPredictor16x16_DcPred_OnlyAbove_AveragesAboveOnly()
    {
        Span<byte> above = stackalloc byte[16];
        Span<byte> left = stackalloc byte[16];
        for (int i = 0; i < 16; i++) above[i] = 32;
        Span<byte> dst = stackalloc byte[16 * 16];

        Vp8IntraPredictor16x16.Predict(
            Vp8IntraMode16x16.DcPred,
            above, left, 0, haveAbove: true, haveLeft: false,
            dst, 16);

        // sum = 16*32 = 512, dc = (512 + 8) >> 4 = 32
        for (int i = 0; i < dst.Length; i++)
            Equal((byte)32, dst[i]);
    }

    [TestMethod]
    public void Vp8IntraPredictor16x16_VPred_CopiesAboveRowDown()
    {
        Span<byte> above = stackalloc byte[16];
        Span<byte> left = stackalloc byte[16];
        for (int i = 0; i < 16; i++) above[i] = (byte)(10 + i);
        Span<byte> dst = stackalloc byte[16 * 16];

        Vp8IntraPredictor16x16.Predict(
            Vp8IntraMode16x16.VPred, above, left, 0, true, true, dst, 16);

        for (int r = 0; r < 16; r++)
            for (int c = 0; c < 16; c++)
                Equal(above[c], dst[r * 16 + c]);
    }

    [TestMethod]
    public void Vp8IntraPredictor16x16_HPred_CopiesLeftColumnRight()
    {
        Span<byte> above = stackalloc byte[16];
        Span<byte> left = stackalloc byte[16];
        for (int i = 0; i < 16; i++) left[i] = (byte)(50 + i);
        Span<byte> dst = stackalloc byte[16 * 16];

        Vp8IntraPredictor16x16.Predict(
            Vp8IntraMode16x16.HPred, above, left, 0, true, true, dst, 16);

        for (int r = 0; r < 16; r++)
            for (int c = 0; c < 16; c++)
                Equal(left[r], dst[r * 16 + c]);
    }

    [TestMethod]
    public void Vp8IntraPredictor16x16_TmPred_KnownInput_KnownOutput()
    {
        // TM: pixel[r][c] = above[c] + left[r] - top_left, clamped.
        Span<byte> above = stackalloc byte[16];
        Span<byte> left = stackalloc byte[16];
        for (int i = 0; i < 16; i++) { above[i] = 100; left[i] = 100; }
        byte topLeft = 50;
        Span<byte> dst = stackalloc byte[16 * 16];

        Vp8IntraPredictor16x16.Predict(
            Vp8IntraMode16x16.TmPred, above, left, topLeft, true, true, dst, 16);

        // 100 + 100 - 50 = 150
        for (int i = 0; i < dst.Length; i++)
            Equal((byte)150, dst[i]);
    }

    [TestMethod]
    public void Vp8IntraPredictor16x16_TmPred_ClampsAt255()
    {
        Span<byte> above = stackalloc byte[16];
        Span<byte> left = stackalloc byte[16];
        for (int i = 0; i < 16; i++) { above[i] = 255; left[i] = 255; }
        byte topLeft = 0;
        Span<byte> dst = stackalloc byte[16 * 16];

        Vp8IntraPredictor16x16.Predict(
            Vp8IntraMode16x16.TmPred, above, left, topLeft, true, true, dst, 16);

        // 255 + 255 - 0 = 510, clamp to 255
        for (int i = 0; i < dst.Length; i++)
            Equal((byte)255, dst[i]);
    }

    [TestMethod]
    public void Vp8IntraPredictor16x16_TmPred_ClampsAtZero()
    {
        Span<byte> above = stackalloc byte[16];
        Span<byte> left = stackalloc byte[16];
        // above all 0, left all 0, topLeft 200 -> 0 + 0 - 200 = -200, clamp to 0
        byte topLeft = 200;
        Span<byte> dst = stackalloc byte[16 * 16];

        Vp8IntraPredictor16x16.Predict(
            Vp8IntraMode16x16.TmPred, above, left, topLeft, true, true, dst, 16);

        for (int i = 0; i < dst.Length; i++)
            Equal((byte)0, dst[i]);
    }
}
