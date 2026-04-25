// Tests for Vp9TmPredictor (slice 164).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9TmPredictor_4x4_FlatInputs_ProduceCornerExtrapolation()
    {
        // top_left=100, above=all 100, left=all 100 -> dst = 100+100-100 = 100.
        var above = new byte[] { 100, 100, 100, 100 };
        var left = new byte[] { 100, 100, 100, 100 };
        var dst = new byte[16];
        Vp9TmPredictor.TmPredict(topLeft: 100, above, left, dst, n: 4, stride: 4);
        for (int i = 0; i < 16; i++) Equal((byte)100, dst[i]);
    }

    [TestMethod]
    public void Vp9TmPredictor_4x4_KnownPattern()
    {
        // top_left=10, above=[20, 30, 40, 50], left=[60, 70, 80, 90].
        // Row 0: left=60, dst[c] = clip(60 + above[c] - 10) =
        //   clip(70, 80, 90, 100) = 70, 80, 90, 100.
        // Row 1: left=70, dst[c] = clip(70 + above[c] - 10) =
        //   clip(80, 90, 100, 110) = 80, 90, 100, 110.
        // Row 2: left=80 -> 90, 100, 110, 120
        // Row 3: left=90 -> 100, 110, 120, 130
        var above = new byte[] { 20, 30, 40, 50 };
        var left = new byte[] { 60, 70, 80, 90 };
        var dst = new byte[16];
        Vp9TmPredictor.TmPredict(topLeft: 10, above, left, dst, 4, 4);

        Equal((byte)70,  dst[0]);  Equal((byte)80,  dst[1]);  Equal((byte)90,  dst[2]);  Equal((byte)100, dst[3]);
        Equal((byte)80,  dst[4]);  Equal((byte)90,  dst[5]);  Equal((byte)100, dst[6]);  Equal((byte)110, dst[7]);
        Equal((byte)90,  dst[8]);  Equal((byte)100, dst[9]);  Equal((byte)110, dst[10]); Equal((byte)120, dst[11]);
        Equal((byte)100, dst[12]); Equal((byte)110, dst[13]); Equal((byte)120, dst[14]); Equal((byte)130, dst[15]);
    }

    [TestMethod]
    public void Vp9TmPredictor_4x4_ClipsHighToTwoFiveFive()
    {
        // top_left=0, above=all 200, left=all 200 -> dst = 200+200-0 = 400 -> clamps to 255.
        var above = new byte[] { 200, 200, 200, 200 };
        var left = new byte[] { 200, 200, 200, 200 };
        var dst = new byte[16];
        Vp9TmPredictor.TmPredict(topLeft: 0, above, left, dst, 4, 4);
        for (int i = 0; i < 16; i++) Equal((byte)255, dst[i]);
    }

    [TestMethod]
    public void Vp9TmPredictor_4x4_ClipsLowToZero()
    {
        // top_left=200, above=all 50, left=all 50 -> dst = 50+50-200 = -100 -> clamps to 0.
        var above = new byte[] { 50, 50, 50, 50 };
        var left = new byte[] { 50, 50, 50, 50 };
        var dst = new byte[16];
        Vp9TmPredictor.TmPredict(topLeft: 200, above, left, dst, 4, 4);
        for (int i = 0; i < 16; i++) Equal((byte)0, dst[i]);
    }

    [TestMethod]
    public void Vp9TmPredictor_8x8_MixedClipping()
    {
        // Asymmetric inputs that produce both clipped-high and clipped-low
        // pixels, plus in-range middle values. Confirms per-pixel clipping
        // not a wholesale block-level branch.
        var above = new byte[8];
        for (int i = 0; i < 8; i++) above[i] = (byte)(i * 30); // 0, 30, 60, 90, 120, 150, 180, 210
        var left = new byte[8];
        for (int i = 0; i < 8; i++) left[i] = (byte)(i * 30);
        byte topLeft = 100;
        var dst = new byte[64];
        Vp9TmPredictor.TmPredict(topLeft, above, left, dst, 8, 8);

        // Row 0 (left=0): dst[c] = clip(0 + above[c] - 100)
        //   = clip(-100, -70, -40, -10, 20, 50, 80, 110) = 0,0,0,0,20,50,80,110
        Equal((byte)0,   dst[0]);  Equal((byte)0,   dst[1]);  Equal((byte)0,   dst[2]);  Equal((byte)0,   dst[3]);
        Equal((byte)20,  dst[4]);  Equal((byte)50,  dst[5]);  Equal((byte)80,  dst[6]);  Equal((byte)110, dst[7]);

        // Row 7 (left=210): dst[c] = clip(210 + above[c] - 100)
        //   = clip(110, 140, 170, 200, 230, 260, 290, 320) = 110,140,170,200,230,255,255,255
        int r7 = 7 * 8;
        Equal((byte)110, dst[r7 + 0]); Equal((byte)140, dst[r7 + 1]); Equal((byte)170, dst[r7 + 2]); Equal((byte)200, dst[r7 + 3]);
        Equal((byte)230, dst[r7 + 4]); Equal((byte)255, dst[r7 + 5]); Equal((byte)255, dst[r7 + 6]); Equal((byte)255, dst[r7 + 7]);
    }

    [TestMethod]
    public void Vp9TmPredictor_16x16_StridedDest_OnlyTouchesBlock()
    {
        var above = new byte[16];
        var left = new byte[16];
        for (int i = 0; i < 16; i++) { above[i] = 100; left[i] = 100; }
        const int stride = 32;
        var canvas = new byte[stride * 16];
        for (int i = 0; i < canvas.Length; i++) canvas[i] = 222;
        Vp9TmPredictor.TmPredict(topLeft: 100, above, left, canvas, n: 16, stride);

        for (int row = 0; row < 16; row++)
        {
            for (int col = 0; col < 16; col++)
                Equal((byte)100, canvas[row * stride + col]);
            for (int col = 16; col < stride; col++)
                Equal((byte)222, canvas[row * stride + col]);
        }
    }

    [TestMethod]
    public void Vp9TmPredictor_32x32_FlatInputProducesFlatOutput()
    {
        var above = new byte[32];
        var left = new byte[32];
        for (int i = 0; i < 32; i++) { above[i] = 80; left[i] = 80; }
        var dst = new byte[1024];
        Vp9TmPredictor.TmPredict(topLeft: 80, above, left, dst, 32, 32);
        for (int i = 0; i < 1024; i++) Equal((byte)80, dst[i]);
    }

    [TestMethod]
    public void Vp9TmPredictor_RejectsInvalidSize()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9TmPredictor.TmPredict(0, new byte[5], new byte[5], new byte[25], n: 5, stride: 5));
    }

    [TestMethod]
    public void Vp9TmPredictor_RejectsTooSmallEdgeSpans()
    {
        Throws<ArgumentException>(() =>
            Vp9TmPredictor.TmPredict(0, new byte[3], new byte[4], new byte[16], n: 4, stride: 4));
        Throws<ArgumentException>(() =>
            Vp9TmPredictor.TmPredict(0, new byte[4], new byte[3], new byte[16], n: 4, stride: 4));
    }
}
