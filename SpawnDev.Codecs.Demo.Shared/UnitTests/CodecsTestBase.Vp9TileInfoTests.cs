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
}
