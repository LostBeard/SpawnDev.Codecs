// Tests for Vp9DcPredictor (slice 160). Exercises all four DC
// prediction variants (both edges / top only / left only / 128 fill)
// across all four transform sizes (4, 8, 16, 32) with hand-pickable
// inputs that produce known outputs.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] FilledArray(int len, byte value)
    {
        var a = new byte[len];
        for (int i = 0; i < len; i++) a[i] = value;
        return a;
    }

    private static void AssertBlockUniformlyFilled(byte[] dst, int n, int stride, byte expected)
    {
        for (int row = 0; row < n; row++)
        for (int col = 0; col < n; col++)
            Equal(expected, dst[row * stride + col]);
    }

    [TestMethod]
    public void Vp9DcPredictor_BothEdges_4x4_FlatInput100_Produces100()
    {
        // sum_above = 4*100 = 400, sum_left = 4*100 = 400, sum = 800.
        // dc = (800 + 4) >> 3 = 804 / 8 = 100 (exact).
        var above = FilledArray(4, 100);
        var left = FilledArray(4, 100);
        var dst = new byte[16];
        Vp9DcPredictor.DcPredict(above, left, dst, n: 4, stride: 4);
        AssertBlockUniformlyFilled(dst, 4, 4, 100);
    }

    [TestMethod]
    public void Vp9DcPredictor_BothEdges_8x8_FlatInput200_Produces200()
    {
        // sum = 8*200 + 8*200 = 3200. dc = (3200 + 8) >> 4 = 3208 / 16 = 200.
        var above = FilledArray(8, 200);
        var left = FilledArray(8, 200);
        var dst = new byte[64];
        Vp9DcPredictor.DcPredict(above, left, dst, n: 8, stride: 8);
        AssertBlockUniformlyFilled(dst, 8, 8, 200);
    }

    [TestMethod]
    public void Vp9DcPredictor_BothEdges_16x16_AsymmetricInputProducesRoundedAverage()
    {
        // above = all 100, left = all 200 -> sum = 16*100 + 16*200 = 4800.
        // dc = (4800 + 16) >> 5 = 4816 / 32 = 150.5 -> 150 (truncating shift).
        var above = FilledArray(16, 100);
        var left = FilledArray(16, 200);
        var dst = new byte[256];
        Vp9DcPredictor.DcPredict(above, left, dst, n: 16, stride: 16);
        AssertBlockUniformlyFilled(dst, 16, 16, 150);
    }

    [TestMethod]
    public void Vp9DcPredictor_BothEdges_32x32_FullDynamicRange()
    {
        // above = all 0, left = all 255 -> sum = 32*0 + 32*255 = 8160.
        // dc = (8160 + 32) >> 6 = 8192 / 64 = 128.
        var above = FilledArray(32, 0);
        var left = FilledArray(32, 255);
        var dst = new byte[1024];
        Vp9DcPredictor.DcPredict(above, left, dst, n: 32, stride: 32);
        AssertBlockUniformlyFilled(dst, 32, 32, 128);
    }

    [TestMethod]
    public void Vp9DcPredictor_TopOnly_4x4_FlatInput80_Produces80()
    {
        // sum = 4*80 = 320. dc = (320 + 2) >> 2 = 322 / 4 = 80.
        var above = FilledArray(4, 80);
        var dst = new byte[16];
        Vp9DcPredictor.DcPredictTop(above, dst, n: 4, stride: 4);
        AssertBlockUniformlyFilled(dst, 4, 4, 80);
    }

    [TestMethod]
    public void Vp9DcPredictor_TopOnly_16x16_FlatInput100_Produces100()
    {
        // sum = 16*100 = 1600. dc = (1600 + 8) >> 4 = 1608 / 16 = 100.
        var above = FilledArray(16, 100);
        var dst = new byte[256];
        Vp9DcPredictor.DcPredictTop(above, dst, n: 16, stride: 16);
        AssertBlockUniformlyFilled(dst, 16, 16, 100);
    }

    [TestMethod]
    public void Vp9DcPredictor_LeftOnly_8x8_FlatInput150_Produces150()
    {
        // sum = 8*150 = 1200. dc = (1200 + 4) >> 3 = 1204 / 8 = 150.
        var left = FilledArray(8, 150);
        var dst = new byte[64];
        Vp9DcPredictor.DcPredictLeft(left, dst, n: 8, stride: 8);
        AssertBlockUniformlyFilled(dst, 8, 8, 150);
    }

    [TestMethod]
    public void Vp9DcPredictor_Dc128_AllSizes_ProduceFlat128()
    {
        foreach (int n in new[] { 4, 8, 16, 32 })
        {
            var dst = new byte[n * n];
            Vp9DcPredictor.DcPredict128(dst, n, stride: n);
            AssertBlockUniformlyFilled(dst, n, n, 128);
        }
    }

    [TestMethod]
    public void Vp9DcPredictor_BothEdges_StrideLargerThanN_OnlyTouchesNxN()
    {
        // 8x8 block at the top of a 16-stride canvas. Only positions
        // [0..7] of each of the first 8 rows should be touched.
        const int n = 8;
        const int stride = 16;
        var above = FilledArray(n, 100);
        var left = FilledArray(n, 100);
        var canvas = new byte[stride * n];
        for (int i = 0; i < canvas.Length; i++) canvas[i] = 77;
        Vp9DcPredictor.DcPredict(above, left, canvas, n, stride);

        // Cols 0-7 should be 100; cols 8-15 should still be 77.
        for (int row = 0; row < n; row++)
        {
            for (int col = 0; col < n; col++)
                Equal((byte)100, canvas[row * stride + col]);
            for (int col = n; col < stride; col++)
                Equal((byte)77, canvas[row * stride + col]);
        }
    }

    [TestMethod]
    public void Vp9DcPredictor_RejectsInvalidSize()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9DcPredictor.DcPredict(new byte[5], new byte[5], new byte[25], n: 5, stride: 5));
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9DcPredictor.DcPredict128(new byte[16], n: 0, stride: 16));
    }

    [TestMethod]
    public void Vp9DcPredictor_BothEdges_RejectsTooSmallEdgeSpans()
    {
        Throws<ArgumentException>(() =>
            Vp9DcPredictor.DcPredict(new byte[3], new byte[4], new byte[16], n: 4, stride: 4));
        Throws<ArgumentException>(() =>
            Vp9DcPredictor.DcPredict(new byte[4], new byte[3], new byte[16], n: 4, stride: 4));
    }
}
