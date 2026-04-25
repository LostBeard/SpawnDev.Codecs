// Tests for Vp9TileLayout (slice 252).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9TileLayout_MiBlockSize_Is8()
    {
        Equal(3, Vp9TileLayout.MiBlockSizeLog2);
        Equal(8, Vp9TileLayout.MiBlockSize);
    }

    [TestMethod]
    public void Vp9TileLayout_MiAligned_RoundsUpToMultipleOf8()
    {
        Equal(0, Vp9TileLayout.MiAlignedToSb(0));
        Equal(8, Vp9TileLayout.MiAlignedToSb(1));
        Equal(8, Vp9TileLayout.MiAlignedToSb(8));
        Equal(16, Vp9TileLayout.MiAlignedToSb(9));
        Equal(16, Vp9TileLayout.MiAlignedToSb(16));
        Equal(80, Vp9TileLayout.MiAlignedToSb(73));
    }

    [TestMethod]
    public void Vp9TileLayout_GetTileOffset_SingleTile_FullFrame()
    {
        // log2 = 0 -> 1 tile total -> offset 0 covers everything.
        Equal(0, Vp9TileLayout.GetTileOffset(0, 64, 0));
        Equal(64, Vp9TileLayout.GetTileOffset(1, 64, 0));
    }

    [TestMethod]
    public void Vp9TileLayout_GetTileOffset_TwoTiles_HalfEach()
    {
        // 64 mi cols, log2 = 1 -> 2 tiles, each ~32 mi.
        // sb_cols = 64 / 8 = 8; offset = (i * 8) >> 1) * 8.
        Equal(0, Vp9TileLayout.GetTileOffset(0, 64, 1));
        Equal(32, Vp9TileLayout.GetTileOffset(1, 64, 1));
        Equal(64, Vp9TileLayout.GetTileOffset(2, 64, 1));
    }

    [TestMethod]
    public void Vp9TileLayout_GetTileOffset_NonAlignedFrame_ClampsToMis()
    {
        // 73 mi cols, log2 = 1 -> aligned to 80, sb_cols=10.
        // tile 0: 0; tile 1: (10 >> 1) * 8 = 40; tile 2: (20 >> 1) * 8 = 80, clamped to 73.
        Equal(0, Vp9TileLayout.GetTileOffset(0, 73, 1));
        Equal(40, Vp9TileLayout.GetTileOffset(1, 73, 1));
        Equal(73, Vp9TileLayout.GetTileOffset(2, 73, 1));
    }

    [TestMethod]
    public void Vp9TileLayout_Compute_FullFrame_OneTileBounds()
    {
        var bounds = Vp9TileLayout.Compute(0, 0, 32, 64, 0, 0);
        Equal(0, bounds.MiRowStart);
        Equal(32, bounds.MiRowEnd);
        Equal(0, bounds.MiColStart);
        Equal(64, bounds.MiColEnd);
        Equal(32, bounds.MiHeight);
        Equal(64, bounds.MiWidth);
    }

    [TestMethod]
    public void Vp9TileLayout_Compute_4ColumnTiles()
    {
        // 64 mi cols, log2_tile_cols = 2 -> 4 tiles of 16 each.
        var t0 = Vp9TileLayout.Compute(0, 0, 32, 64, 0, 2);
        var t1 = Vp9TileLayout.Compute(0, 1, 32, 64, 0, 2);
        var t2 = Vp9TileLayout.Compute(0, 2, 32, 64, 0, 2);
        var t3 = Vp9TileLayout.Compute(0, 3, 32, 64, 0, 2);
        Equal(0, t0.MiColStart); Equal(16, t0.MiColEnd);
        Equal(16, t1.MiColStart); Equal(32, t1.MiColEnd);
        Equal(32, t2.MiColStart); Equal(48, t2.MiColEnd);
        Equal(48, t3.MiColStart); Equal(64, t3.MiColEnd);
    }

    [TestMethod]
    public void Vp9TileLayout_Compute_RejectsNegativeMis()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9TileLayout.GetTileOffset(0, -1, 0));
    }

    [TestMethod]
    public void Vp9TileLayout_Compute_RejectsNegativeLog2()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9TileLayout.GetTileOffset(0, 64, -1));
    }

    [TestMethod]
    public void Vp9TileLayout_TileWidthConstants()
    {
        Equal(4, Vp9TileLayout.MinTileWidthSb);
        Equal(64, Vp9TileLayout.MaxTileWidthSb);
    }

    [TestMethod]
    public void Vp9TileLayout_Log2TileColsBounds_SmallFrame_BothZero()
    {
        // 32 mi cols = 4 SBs. min: 64 << 0 < 4 is false -> min = 0.
        // max: (4 >> 1) >= 4? no, so max = 0.
        var (min, max) = Vp9TileLayout.GetLog2TileColsBounds(32);
        Equal(0, min);
        Equal(0, max);
    }

    [TestMethod]
    public void Vp9TileLayout_Log2TileColsBounds_HD()
    {
        // 1280 px frame -> 1280/8 = 160 mi cols, aligned -> 160, sb_cols = 20.
        // max: how many bits before sb_cols >> bits drops below 4? 20 >> 2 = 5 (>=4 ok), 20 >> 3 = 2 (<4 stop). max=3-1=2.
        // min: 64 << 0 = 64; 64 < 20? no. min = 0.
        var (min, max) = Vp9TileLayout.GetLog2TileColsBounds(160);
        Equal(0, min);
        Equal(2, max);
    }

    [TestMethod]
    public void Vp9TileLayout_Log2TileColsBounds_VeryWide()
    {
        // 8192 mi cols -> sb_cols = 1024.
        // min: 64 << 0 = 64 < 1024 -> ++; 128 -> ++; ... 64 << 4 = 1024, not < 1024 -> stop. min = 4.
        // max: 1024 >> 1 = 512 (>=4); 1024 >> 9 = 2 (<4); ++ went 1..9, stops with maxLog2=9 -> --=8.
        var (min, max) = Vp9TileLayout.GetLog2TileColsBounds(8192);
        Equal(4, min);
        Equal(8, max);
    }

    [TestMethod]
    public void Vp9TileLayout_Log2TileColsBounds_RejectsNegativeMis()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9TileLayout.GetLog2TileColsBounds(-1));
    }
}
