// Tests for Vp9TileInfoParser. Spec sec 6.2.14 calc_min/max_log2_tile_cols
// (commit be10e55 corrected the formulas - they were transposed previously).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9TileInfo_GetTileNBits_640x480Frame()
    {
        // 640 / 8 = 80 mi_cols. sb_cols = round_up(80, 8) / 8 = 80 / 8 = 10.
        // min_log2_tile_cols (forced floor by MAX_TILE_WIDTH=64):
        //   while ((64 << min) < 10): false at min=0. min = 0.
        // max_log2_tile_cols (splitting ceiling by MIN_TILE_WIDTH=4):
        //   max = 1; (10 >> 1)=5 >= 4 -> ++; (10 >> 2)=2 < 4 stop. max = 2; --max = 1.
        var (minLog2, maxLog2) = Vp9TileInfoParser.GetTileNBits(80);
        Equal(0, minLog2);
        Equal(1, maxLog2);
    }

    [TestMethod]
    public void Vp9TileInfo_GetTileNBits_1920x1080Frame()
    {
        // 1920 / 8 = 240. sb_cols = 240 / 8 = 30.
        // min_log2: (64 << 0)=64 >= 30 -> stop at 0.
        // max_log2: 1, 30>>1=15>=4 ++; 30>>2=7>=4 ++; 30>>3=3<4 stop at 3; --=2.
        var (minLog2, maxLog2) = Vp9TileInfoParser.GetTileNBits(240);
        Equal(0, minLog2);
        Equal(2, maxLog2);
    }

    [TestMethod]
    public void Vp9TileInfo_GetTileNBits_4kFrame()
    {
        // 3840 / 8 = 480. sb_cols = 480 / 8 = 60.
        // min_log2: (64 << 0)=64 >= 60 -> 0.
        // max_log2: 1, 60>>1=30>=4 ++; 30>>2=7..; 60>>4=3<4 stop at 4; --=3.
        var (minLog2, maxLog2) = Vp9TileInfoParser.GetTileNBits(480);
        Equal(0, minLog2);
        Equal(3, maxLog2);
    }

    [TestMethod]
    public void Vp9TileInfo_GetTileNBits_8kFrame()
    {
        // 7680 / 8 = 960. sb_cols = 960 / 8 = 120.
        // min_log2: (64 << 0)=64 < 120 -> ++; (64<<1)=128 >= 120 stop at 1.
        // max_log2: 1, 120>>1=60>=4..120>>5=3<4 stop at 5; --=4.
        var (minLog2, maxLog2) = Vp9TileInfoParser.GetTileNBits(960);
        Equal(1, minLog2);
        Equal(4, maxLog2);
    }

    [TestMethod]
    public void Vp9TileInfo_Parse_NoIncrements_NoRowSplit()
    {
        // 1920x1080: min=0, max=2. The decoder reads bits while it sees 1
        // up to (max - min) = 2 bits. We feed 0 0 -> log2_tile_cols stays at 0.
        // tile_rows_log2: first bit 0 -> 0 rows.
        var data = BitsToBytes((0, 1), (0, 1));

        var ti = Vp9TileInfoParser.Parse(data, miCols: 240);

        Equal(0, ti.Log2TileCols);
        Equal(0, ti.Log2TileRows);
        Equal(1, ti.TileCols);
        Equal(1, ti.TileRows);
    }

    [TestMethod]
    public void Vp9TileInfo_Parse_OneTileColIncrement_BumpsToTwoCols()
    {
        // 1920x1080: min=0, max=2. Feed (1, 0) for tile_cols -> log2 climbs to 1.
        // Then tile_rows first bit 0 -> 0 rows.
        var data = BitsToBytes((1, 1), (0, 1), (0, 1));

        var ti = Vp9TileInfoParser.Parse(data, miCols: 240);

        Equal(1, ti.Log2TileCols);
        Equal(2, ti.TileCols);
        Equal(0, ti.Log2TileRows);
    }

    [TestMethod]
    public void Vp9TileInfo_Parse_TwoTileColIncrements_FourCols()
    {
        // 1920x1080: min=0, max=2. Feed (1, 1) -> log2 = 2.
        var data = BitsToBytes((1, 1), (1, 1), (0, 1));

        var ti = Vp9TileInfoParser.Parse(data, miCols: 240);

        Equal(2, ti.Log2TileCols);
        Equal(4, ti.TileCols);
    }

    [TestMethod]
    public void Vp9TileInfo_Parse_TileRowsLog2Equals2_ReadsTwoBits()
    {
        // tile_cols default (no increments) + tile_rows = 2 reads (1, 1).
        var data = BitsToBytes((0, 1), (1, 1), (1, 1));

        var ti = Vp9TileInfoParser.Parse(data, miCols: 240);

        Equal(2, ti.Log2TileRows);
        Equal(4, ti.TileRows);
    }

    [TestMethod]
    public void Vp9TileInfo_Parse_TileRowsLog2Equals1_ReadsTwoBits()
    {
        // tile_cols default + tile_rows = 1 reads (1, 0).
        var data = BitsToBytes((0, 1), (1, 1), (0, 1));

        var ti = Vp9TileInfoParser.Parse(data, miCols: 240);

        Equal(1, ti.Log2TileRows);
        Equal(2, ti.TileRows);
    }

    // ===========================================================================
    // GetTileColRange / GetTileRowRange - per-tile SB-range computation.
    // Mirrors libvpx's `get_tile_offset` from vp9/common/vp9_tile_common.c.
    // ===========================================================================

    [TestMethod]
    public void Vp9TileInfo_GetTileColRange_SingleTile_FullFrame()
    {
        // 1 tile column = full frame.
        var (s, e) = Vp9TileInfoParser.GetTileColRange(tileColIdx: 0, tileCols: 1, sbCols: 30);
        Equal(0, s);
        Equal(30, e);
    }

    [TestMethod]
    public void Vp9TileInfo_GetTileColRange_2Tiles_1920Wide()
    {
        // 1920 wide -> mi_cols=240, sb_cols=30. 2 tile cols (log2=1).
        // libvpx: offset[1] = ((1 * 240) >> 1) = 120 mi -> SB-aligned 120 -> sb 15.
        var (s0, e0) = Vp9TileInfoParser.GetTileColRange(tileColIdx: 0, tileCols: 2, sbCols: 30);
        Equal(0, s0);
        Equal(15, e0);

        var (s1, e1) = Vp9TileInfoParser.GetTileColRange(tileColIdx: 1, tileCols: 2, sbCols: 30);
        Equal(15, s1);
        Equal(30, e1);
    }

    [TestMethod]
    public void Vp9TileInfo_GetTileColRange_4Tiles_1920Wide()
    {
        // sb_cols=30, 4 tile cols (log2=2). Tile starts:
        //   tile 0: ((0 * 240) >> 2) = 0   -> sb 0
        //   tile 1: ((1 * 240) >> 2) = 60  -> sb 7 ... wait 60 mi = sb 7.5, SB-aligned up = 64 mi = sb 8
        //   tile 2: ((2 * 240) >> 2) = 120 -> sb 15
        //   tile 3: ((3 * 240) >> 2) = 180 -> sb 22.5, aligned up = sb 23
        // Half-open ranges: [0,8), [8,15), [15,23), [23,30).
        var (s0, e0) = Vp9TileInfoParser.GetTileColRange(0, 4, 30);
        Equal(0, s0); Equal(8, e0);
        var (s1, e1) = Vp9TileInfoParser.GetTileColRange(1, 4, 30);
        Equal(8, s1); Equal(15, e1);
        var (s2, e2) = Vp9TileInfoParser.GetTileColRange(2, 4, 30);
        Equal(15, s2); Equal(23, e2);
        var (s3, e3) = Vp9TileInfoParser.GetTileColRange(3, 4, 30);
        Equal(23, s3); Equal(30, e3); // last tile always extends to sb_cols
    }

    [TestMethod]
    public void Vp9TileInfo_GetTileColRange_2Tiles_512Wide()
    {
        // 512 wide -> mi_cols=64, sb_cols=8. 2 tile cols.
        //   tile 0 starts: ((0 * 64) >> 1) = 0 mi -> sb 0
        //   tile 1 starts: ((1 * 64) >> 1) = 32 mi -> sb 4
        var (s0, e0) = Vp9TileInfoParser.GetTileColRange(0, 2, 8);
        Equal(0, s0); Equal(4, e0);
        var (s1, e1) = Vp9TileInfoParser.GetTileColRange(1, 2, 8);
        Equal(4, s1); Equal(8, e1);
    }

    [TestMethod]
    public void Vp9TileInfo_GetTileRowRange_2x_1080Tall()
    {
        // 1080 tall -> mi_rows=135, sb_rows = AlignUp(135,8)/8 = 17. 2 tile rows.
        // libvpx: offset[1] = ((1 * (17*8)) >> 1) = 68 mi -> SB-aligned to 72 -> sb 9
        var (s0, e0) = Vp9TileInfoParser.GetTileRowRange(0, 2, 17);
        Equal(0, s0); Equal(9, e0);
        var (s1, e1) = Vp9TileInfoParser.GetTileRowRange(1, 2, 17);
        Equal(9, s1); Equal(17, e1);
    }

    [TestMethod]
    public void Vp9TileInfo_GetTileColRange_LastTileSpansToSbCols()
    {
        // Edge case: sbCols not divisible by tileCols. Last tile extends to sbCols.
        // sb_cols=15 (rare odd dim), 4 tile cols.
        //   tile 0: 0
        //   tile 1: ((1 * 120) >> 2) = 30 mi -> sb 4 (30 SB-aligned to 32 mi)
        //   tile 2: ((2 * 120) >> 2) = 60 mi -> sb 8 (64 mi)
        //   tile 3: ((3 * 120) >> 2) = 90 mi -> sb 12 (aligned to 96 mi)
        var (_, e3) = Vp9TileInfoParser.GetTileColRange(3, 4, 15);
        Equal(15, e3); // last tile must reach sb_cols, not the computed boundary
    }
}
