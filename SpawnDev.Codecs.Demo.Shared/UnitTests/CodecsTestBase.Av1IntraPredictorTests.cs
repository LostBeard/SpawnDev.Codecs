// AV1 intra predictor tests. Exercises V / H / DC / DC_LEFT / DC_TOP /
// DC_128 / PAETH / SMOOTH / SMOOTH_V / SMOOTH_H modes against
// hand-computed reference values mirroring libaom aom_dsp/intrapred.c.

using SpawnDev.Codecs.Video.Av1;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static (byte[] dst, byte[] above, byte[] left) MakeBlock(int bw, int bh)
    {
        var dst = new byte[bw * bh];
        var above = new byte[bw + bh + 1];
        var left = new byte[bh];
        return (dst, above, left);
    }

    [TestMethod]
    public void Av1IntraPredictor_V_CopiesAboveIntoEveryRow()
    {
        var (dst, above, left) = MakeBlock(8, 8);
        for (int i = 0; i < 8; i++) above[i] = (byte)(50 + i * 10);
        Av1IntraPredictor.Vertical(dst, 8, 8, 8, above, left);
        for (int r = 0; r < 8; r++)
        {
            for (int c = 0; c < 8; c++)
            {
                Equal(above[c], dst[r * 8 + c]);
            }
        }
    }

    [TestMethod]
    public void Av1IntraPredictor_H_ReplicatesLeftPerRow()
    {
        var (dst, above, left) = MakeBlock(8, 8);
        for (int i = 0; i < 8; i++) left[i] = (byte)(40 + i * 5);
        Av1IntraPredictor.Horizontal(dst, 8, 8, 8, above, left);
        for (int r = 0; r < 8; r++)
        {
            for (int c = 0; c < 8; c++)
            {
                Equal(left[r], dst[r * 8 + c]);
            }
        }
    }

    [TestMethod]
    public void Av1IntraPredictor_Dc_AveragesAllAvailableEdges()
    {
        var (dst, above, left) = MakeBlock(4, 4);
        // 4 above pixels: 100, 200, 100, 200 -> sum 600
        // 4 left pixels: 50, 50, 100, 100 -> sum 300
        // count = 8, dc = (900 + 4) / 8 = 113
        above[0] = 100; above[1] = 200; above[2] = 100; above[3] = 200;
        left[0] = 50; left[1] = 50; left[2] = 100; left[3] = 100;
        Av1IntraPredictor.Dc(dst, 4, 4, 4, above, left);
        for (int i = 0; i < 16; i++) Equal(113, dst[i]);
    }

    [TestMethod]
    public void Av1IntraPredictor_DcLeft_AveragesLeftOnly()
    {
        var (dst, above, left) = MakeBlock(4, 4);
        // sum = 50+100+150+200 = 500, dc = (500+2)/4 = 125
        left[0] = 50; left[1] = 100; left[2] = 150; left[3] = 200;
        Av1IntraPredictor.DcLeft(dst, 4, 4, 4, above, left);
        for (int i = 0; i < 16; i++) Equal(125, dst[i]);
    }

    [TestMethod]
    public void Av1IntraPredictor_DcTop_AveragesAboveOnly()
    {
        var (dst, above, left) = MakeBlock(4, 4);
        above[0] = 80; above[1] = 80; above[2] = 80; above[3] = 80;
        Av1IntraPredictor.DcTop(dst, 4, 4, 4, above, left);
        for (int i = 0; i < 16; i++) Equal(80, dst[i]);
    }

    [TestMethod]
    public void Av1IntraPredictor_Dc128_FillsWithMidGray()
    {
        var (dst, above, left) = MakeBlock(8, 8);
        Av1IntraPredictor.Dc128(dst, 8, 8, 8, above, left);
        for (int i = 0; i < 64; i++) Equal(128, dst[i]);
    }

    [TestMethod]
    public void Av1IntraPredictor_Paeth_OnFlatEdges_ProducesFlatBlock()
    {
        var (dst, above, left) = MakeBlock(4, 4);
        for (int i = 0; i < 4; i++) above[i] = 100;
        for (int i = 0; i < 4; i++) left[i] = 100;
        var aboveMinus1 = new byte[1] { 100 }; // top-left = 100
        Av1IntraPredictor.Paeth(dst, 4, 4, 4, above, aboveMinus1, left);
        for (int i = 0; i < 16; i++) Equal(100, dst[i]);
    }

    [TestMethod]
    public void Av1IntraPredictor_Paeth_PicksClosestToBase()
    {
        // top=255, left=0, topLeft=128: base = 255 + 0 - 128 = 127
        // pLeft = |127 - 0| = 127
        // pTop = |127 - 255| = 128
        // pTopLeft = |127 - 128| = 1
        // Result: topLeft (128) since pTopLeft is smallest (and pTopLeft<=pLeft fails since 1<=127, but ranking is left-then-top-then-topLeft).
        // Actual paeth_single rule: if pLeft<=pTop && pLeft<=pTopLeft return left; elif pTop<=pTopLeft return top; else topLeft.
        // pLeft=127 vs pTopLeft=1: 127<=1 false. pTop=128 vs pTopLeft=1: 128<=1 false. -> topLeft = 128.
        Equal((byte)128, Av1IntraPredictor.PaethSingle(0, 255, 128));
        // top=100, left=100, topLeft=50: base = 100 + 100 - 50 = 150
        // pLeft = |150-100| = 50, pTop = 50, pTopLeft = 100. -> left (50<=50 && 50<=100) = 100.
        Equal((byte)100, Av1IntraPredictor.PaethSingle(100, 100, 50));
    }

    [TestMethod]
    public void Av1IntraPredictor_Smooth_BlendsTowardCorners()
    {
        var (dst, above, left) = MakeBlock(4, 4);
        // Edges all 100 -> smooth blend should be ~100
        for (int i = 0; i < 4; i++) { above[i] = 100; left[i] = 100; }
        Av1IntraPredictor.Smooth(dst, 4, 4, 4, above, left);
        for (int i = 0; i < 16; i++) InRange(dst[i], 99, 101);
    }

    [TestMethod]
    public void Av1IntraPredictor_SmoothV_BlendsTopWithBottomLeft()
    {
        var (dst, above, left) = MakeBlock(4, 4);
        for (int i = 0; i < 4; i++) above[i] = 200;
        for (int i = 0; i < 4; i++) left[i] = 100;
        // bottom-left = left[3] = 100; SmoothV blends top -> 100 vertically.
        // First row uses weight[0]=255 -> nearly all top -> ~200
        // Last row uses weight[3]=64 -> heavier on bottom -> closer to 100.
        Av1IntraPredictor.SmoothV(dst, 4, 4, 4, above, left);
        // Top-left pixel: weighted toward above[0]=200 (weight[0]=255 of 256 -> ~200)
        InRange(dst[0], 195, 200);
        // Bottom-left pixel: weight[3]=64 -> 64/256 * 200 + 192/256 * 100 = 50 + 75 = 125
        InRange(dst[3 * 4], 120, 130);
    }

    [TestMethod]
    public void Av1IntraPredictor_SmoothH_BlendsLeftWithTopRight()
    {
        var (dst, above, left) = MakeBlock(4, 4);
        for (int i = 0; i < 4; i++) above[i] = 200;
        for (int i = 0; i < 4; i++) left[i] = 100;
        Av1IntraPredictor.SmoothH(dst, 4, 4, 4, above, left);
        // Leftmost column: weight[0]=255 -> nearly all left -> ~100
        InRange(dst[0], 100, 105);
        // Rightmost column: weight[3]=64 -> closer to top-right (above[3]=200)
        InRange(dst[3], 130, 175);
    }
}
