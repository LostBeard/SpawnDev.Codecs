// Tests for Vp9VHPredictor (slice 161). V_PRED and H_PRED at all
// four sizes with hand-pickable inputs.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9VHPredictor_VPredict_4x4_CopiesAboveToEveryRow()
    {
        // above = [10, 20, 30, 40]; output is 4 copies of that row.
        var above = new byte[] { 10, 20, 30, 40 };
        var dst = new byte[16];
        Vp9VHPredictor.VPredict(above, dst, n: 4, stride: 4);
        for (int row = 0; row < 4; row++)
        {
            Equal((byte)10, dst[row * 4 + 0]);
            Equal((byte)20, dst[row * 4 + 1]);
            Equal((byte)30, dst[row * 4 + 2]);
            Equal((byte)40, dst[row * 4 + 3]);
        }
    }

    [TestMethod]
    public void Vp9VHPredictor_VPredict_8x8_AllSamplesAtFullRange()
    {
        var above = new byte[] { 0, 50, 100, 150, 200, 250, 255, 128 };
        var dst = new byte[64];
        Vp9VHPredictor.VPredict(above, dst, n: 8, stride: 8);
        for (int row = 0; row < 8; row++)
        for (int col = 0; col < 8; col++)
            Equal(above[col], dst[row * 8 + col]);
    }

    [TestMethod]
    public void Vp9VHPredictor_HPredict_4x4_ReplicatesLeftColumnAcrossRow()
    {
        // left = [11, 22, 33, 44]; output is each row filled with left[row].
        var left = new byte[] { 11, 22, 33, 44 };
        var dst = new byte[16];
        Vp9VHPredictor.HPredict(left, dst, n: 4, stride: 4);
        for (int row = 0; row < 4; row++)
        for (int col = 0; col < 4; col++)
            Equal(left[row], dst[row * 4 + col]);
    }

    [TestMethod]
    public void Vp9VHPredictor_HPredict_16x16_FlatLeftAtBoundaryValues()
    {
        var left = new byte[16];
        for (int i = 0; i < 16; i++) left[i] = (byte)(i * 17); // 0, 17, 34, ..., 255
        var dst = new byte[256];
        Vp9VHPredictor.HPredict(left, dst, n: 16, stride: 16);
        for (int row = 0; row < 16; row++)
        for (int col = 0; col < 16; col++)
            Equal(left[row], dst[row * 16 + col]);
    }

    [TestMethod]
    public void Vp9VHPredictor_VPredict_32x32_StridedDest_OnlyTouchesBlock()
    {
        var above = new byte[32];
        for (int i = 0; i < 32; i++) above[i] = (byte)i;
        const int stride = 64;
        var canvas = new byte[stride * 32];
        for (int i = 0; i < canvas.Length; i++) canvas[i] = 99;
        Vp9VHPredictor.VPredict(above, canvas, n: 32, stride);

        for (int row = 0; row < 32; row++)
        {
            for (int col = 0; col < 32; col++)
                Equal(above[col], canvas[row * stride + col]);
            for (int col = 32; col < stride; col++)
                Equal((byte)99, canvas[row * stride + col]);
        }
    }

    [TestMethod]
    public void Vp9VHPredictor_HPredict_8x8_WithLargeStride_DoesNotOverwriteOutOfBlock()
    {
        var left = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        const int stride = 16;
        var canvas = new byte[stride * 8];
        for (int i = 0; i < canvas.Length; i++) canvas[i] = 200;
        Vp9VHPredictor.HPredict(left, canvas, n: 8, stride);

        for (int row = 0; row < 8; row++)
        {
            for (int col = 0; col < 8; col++)
                Equal(left[row], canvas[row * stride + col]);
            for (int col = 8; col < stride; col++)
                Equal((byte)200, canvas[row * stride + col]);
        }
    }

    [TestMethod]
    public void Vp9VHPredictor_RejectsInvalidSize()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9VHPredictor.VPredict(new byte[5], new byte[25], n: 5, stride: 5));
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9VHPredictor.HPredict(new byte[5], new byte[25], n: 5, stride: 5));
    }

    [TestMethod]
    public void Vp9VHPredictor_RejectsTooSmallEdgeSpans()
    {
        Throws<ArgumentException>(() =>
            Vp9VHPredictor.VPredict(new byte[3], new byte[16], n: 4, stride: 4));
        Throws<ArgumentException>(() =>
            Vp9VHPredictor.HPredict(new byte[3], new byte[16], n: 4, stride: 4));
    }
}
