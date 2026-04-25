// Tests for Vp9MvBoundsCalculator (slice 260).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9MvBoundsCalculator_Constants_MatchLibvpx()
    {
        Equal(8, Vp9MvBoundsCalculator.MiSize);
        Equal(192, Vp9MvBoundsCalculator.EncBorderInPixels);
        Equal(4, Vp9MvBoundsCalculator.InterpExtend);
        Equal(1504, Vp9MvBoundsCalculator.LeftTopMargin);
        Equal(1504, Vp9MvBoundsCalculator.RightBottomMargin);
    }

    [TestMethod]
    public void Vp9MvBoundsCalculator_TopLeftBlock_IsNegOnlyForTopLeft()
    {
        // mi (0, 0) of an 8x8 block in a 32x32-mi (256x256-pixel) frame.
        // mb_to_top_edge = 0; mb_to_left_edge = 0.
        // mb_to_bottom_edge = (32 - 1 - 0) * 8 << 3 = 248 * 8 = 1984.
        // mb_to_right_edge = same = 1984.
        // bounds = (0 - 1504, 1984 + 1504, 0 - 1504, 1984 + 1504)
        //        = (-1504, 3488, -1504, 3488).
        var b = Vp9MvBoundsCalculator.Compute(
            miRow: 0, miCol: 0,
            blockSize: Vp9BlockSize.Block8x8,
            frameMiRows: 32, frameMiCols: 32);
        Equal(-1504, b.MinRow);
        Equal(3488, b.MaxRow);
        Equal(-1504, b.MinCol);
        Equal(3488, b.MaxCol);
    }

    [TestMethod]
    public void Vp9MvBoundsCalculator_BottomRightBlock_IsZeroOnlyForBottomRight()
    {
        // 32x32 mi frame, 8x8 block at mi (31, 31) - last block.
        // mb_to_top_edge = -31 * 8 << 3 = -1984.
        // mb_to_bottom_edge = (32 - 1 - 31) * 8 << 3 = 0.
        // mb_to_left_edge = -1984; mb_to_right_edge = 0.
        // bounds = (-1984 - 1504, 0 + 1504, -1984 - 1504, 0 + 1504)
        //        = (-3488, 1504, -3488, 1504).
        var b = Vp9MvBoundsCalculator.Compute(
            miRow: 31, miCol: 31,
            blockSize: Vp9BlockSize.Block8x8,
            frameMiRows: 32, frameMiCols: 32);
        Equal(-3488, b.MinRow);
        Equal(1504, b.MaxRow);
        Equal(-3488, b.MinCol);
        Equal(1504, b.MaxCol);
    }

    [TestMethod]
    public void Vp9MvBoundsCalculator_LargerBlock_BoundsShrink()
    {
        // 16x16 block at (0, 0) of 32x32 mi frame.
        // bw = 2, bh = 2. mb_to_bottom = (32 - 2 - 0) * 8 << 3 = 30*64 = 1920.
        // bounds = (-1504, 1920+1504, -1504, 1920+1504) = (-1504, 3424, -1504, 3424).
        var b = Vp9MvBoundsCalculator.Compute(
            miRow: 0, miCol: 0,
            blockSize: Vp9BlockSize.Block16x16,
            frameMiRows: 32, frameMiCols: 32);
        Equal(3424, b.MaxRow);
        Equal(3424, b.MaxCol);
    }

    [TestMethod]
    public void Vp9MvBoundsCalculator_RejectsNegativeMiPosition()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9MvBoundsCalculator.Compute(-1, 0, Vp9BlockSize.Block8x8, 32, 32));
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9MvBoundsCalculator.Compute(0, -1, Vp9BlockSize.Block8x8, 32, 32));
    }

    [TestMethod]
    public void Vp9MvBoundsCalculator_RejectsNonPositiveFrameDimensions()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9MvBoundsCalculator.Compute(0, 0, Vp9BlockSize.Block8x8, 0, 32));
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9MvBoundsCalculator.Compute(0, 0, Vp9BlockSize.Block8x8, 32, 0));
    }
}
