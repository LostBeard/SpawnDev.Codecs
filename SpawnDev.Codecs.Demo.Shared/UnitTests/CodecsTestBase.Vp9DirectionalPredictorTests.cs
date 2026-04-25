// Tests for Vp9DirectionalPredictor (slice 165). Hand-traced
// expected outputs for D45 + D63 at 4x4, plus larger-size sanity
// checks.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9DirectionalPredictor_D45_4x4_KnownPattern()
    {
        // above = [10, 20, 30, 40, 50, 60, 70, 80].
        // above_right = above[3] = 40.
        // Row 0: AVG3(10,20,30)=20, AVG3(20,30,40)=30, AVG3(30,40,50)=40, dst[3]=40.
        //        Result: 20, 30, 40, 40.
        // Row 1: copy dst_row0[1..2] = [30, 40], fill 2 with 40.
        //        Result: 30, 40, 40, 40.
        // Row 2: copy dst_row0[2..2] = [40], fill 3 with 40.
        //        Result: 40, 40, 40, 40.
        // Row 3: copy 0 bytes, fill 4 with 40.
        //        Result: 40, 40, 40, 40.
        var above = new byte[] { 10, 20, 30, 40, 50, 60, 70, 80 };
        var dst = new byte[16];
        Vp9DirectionalPredictor.D45Predict(above, dst, n: 4, stride: 4);
        Equal((byte)20, dst[0]);  Equal((byte)30, dst[1]);  Equal((byte)40, dst[2]);  Equal((byte)40, dst[3]);
        Equal((byte)30, dst[4]);  Equal((byte)40, dst[5]);  Equal((byte)40, dst[6]);  Equal((byte)40, dst[7]);
        Equal((byte)40, dst[8]);  Equal((byte)40, dst[9]);  Equal((byte)40, dst[10]); Equal((byte)40, dst[11]);
        Equal((byte)40, dst[12]); Equal((byte)40, dst[13]); Equal((byte)40, dst[14]); Equal((byte)40, dst[15]);
    }

    [TestMethod]
    public void Vp9DirectionalPredictor_D45_4x4_FlatInput_ProducesFlatOutput()
    {
        // above = all 100. AVG3(100,100,100) = (100+200+100+2)>>2 = 100.
        // above_right = 100. Whole block = 100.
        var above = new byte[8];
        for (int i = 0; i < 8; i++) above[i] = 100;
        var dst = new byte[16];
        Vp9DirectionalPredictor.D45Predict(above, dst, n: 4, stride: 4);
        for (int i = 0; i < 16; i++) Equal((byte)100, dst[i]);
    }

    [TestMethod]
    public void Vp9DirectionalPredictor_D45_8x8_FlatInputProducesFlatOutput()
    {
        var above = new byte[16];
        for (int i = 0; i < 16; i++) above[i] = 50;
        var dst = new byte[64];
        Vp9DirectionalPredictor.D45Predict(above, dst, n: 8, stride: 8);
        for (int i = 0; i < 64; i++) Equal((byte)50, dst[i]);
    }

    [TestMethod]
    public void Vp9DirectionalPredictor_D63_4x4_KnownPattern()
    {
        // above = [10, 20, 30, 40, 50, 60, 70, 80].
        // Row 0: AVG2 of consecutive pairs:
        //   AVG2(10,20)=(10+20+1)/2=15
        //   AVG2(20,30)=25, AVG2(30,40)=35, AVG2(40,50)=45
        // Row 0: 15, 25, 35, 45.
        // Row 1: AVG3 of triples:
        //   AVG3(10,20,30)=(10+40+30+2)>>2=20
        //   AVG3(20,30,40)=30, AVG3(30,40,50)=40, AVG3(40,50,60)=50
        // Row 1: 20, 30, 40, 50.
        // Row 2 (r=2, size=2): copy row 0 starting at col r/2=1 -> [25, 35], fill 2 with above[3]=40.
        //   Row 2: 25, 35, 40, 40.
        // Row 3 (r=3, paired with r=2, size=2): copy row 1 starting at col 1 -> [30, 40], fill 2 with 40.
        //   Row 3: 30, 40, 40, 40.
        var above = new byte[] { 10, 20, 30, 40, 50, 60, 70, 80 };
        var dst = new byte[16];
        Vp9DirectionalPredictor.D63Predict(above, dst, n: 4, stride: 4);
        Equal((byte)15, dst[0]);  Equal((byte)25, dst[1]);  Equal((byte)35, dst[2]);  Equal((byte)45, dst[3]);
        Equal((byte)20, dst[4]);  Equal((byte)30, dst[5]);  Equal((byte)40, dst[6]);  Equal((byte)50, dst[7]);
        Equal((byte)25, dst[8]);  Equal((byte)35, dst[9]);  Equal((byte)40, dst[10]); Equal((byte)40, dst[11]);
        Equal((byte)30, dst[12]); Equal((byte)40, dst[13]); Equal((byte)40, dst[14]); Equal((byte)40, dst[15]);
    }

    [TestMethod]
    public void Vp9DirectionalPredictor_D63_FlatInputProducesFlatOutput()
    {
        var above = new byte[16];
        for (int i = 0; i < 16; i++) above[i] = 200;
        var dst = new byte[64];
        Vp9DirectionalPredictor.D63Predict(above, dst, n: 8, stride: 8);
        for (int i = 0; i < 64; i++) Equal((byte)200, dst[i]);
    }

    [TestMethod]
    public void Vp9DirectionalPredictor_D45_StridedDest_OnlyTouchesBlock()
    {
        var above = new byte[16];
        for (int i = 0; i < 16; i++) above[i] = 80;
        const int stride = 16;
        var canvas = new byte[stride * 8];
        for (int i = 0; i < canvas.Length; i++) canvas[i] = 222;
        Vp9DirectionalPredictor.D45Predict(above, canvas, n: 8, stride);
        for (int row = 0; row < 8; row++)
        {
            for (int col = 0; col < 8; col++)
                Equal((byte)80, canvas[row * stride + col]);
            for (int col = 8; col < stride; col++)
                Equal((byte)222, canvas[row * stride + col]);
        }
    }

    [TestMethod]
    public void Vp9DirectionalPredictor_D45_RejectsInvalidArgs()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9DirectionalPredictor.D45Predict(new byte[10], new byte[25], n: 5, stride: 5));
        Throws<ArgumentException>(() =>
            Vp9DirectionalPredictor.D45Predict(new byte[7], new byte[16], n: 4, stride: 4));
    }

    [TestMethod]
    public void Vp9DirectionalPredictor_D63_RejectsInvalidArgs()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9DirectionalPredictor.D63Predict(new byte[10], new byte[25], n: 5, stride: 5));
        Throws<ArgumentException>(() =>
            Vp9DirectionalPredictor.D63Predict(new byte[7], new byte[16], n: 4, stride: 4));
    }

    [TestMethod]
    public void Vp9DirectionalPredictor_D135_4x4_KnownPattern()
    {
        // above = [10, 20, 30, 40]; left = [50, 60, 70, 80]; topLeft = 5.
        // border[0] = AVG3(left[1], left[2], left[3]) = AVG3(60, 70, 80) = 70
        // border[1] = AVG3(left[0], left[1], left[2]) = AVG3(50, 60, 70) = 60
        // border[2] = AVG3(topLeft, left[0], left[1]) = AVG3(5, 50, 60) = 41
        // border[3] = AVG3(left[0], topLeft, above[0]) = AVG3(50, 5, 10) = 18
        // border[4] = AVG3(topLeft, above[0], above[1]) = AVG3(5, 10, 20) = 11
        // border[5] = AVG3(above[0], above[1], above[2]) = AVG3(10, 20, 30) = 20
        // border[6] = AVG3(above[1], above[2], above[3]) = AVG3(20, 30, 40) = 30
        // Row 0 (srcStart=3): border[3..6] = 18, 11, 20, 30
        // Row 1 (srcStart=2): border[2..5] = 41, 18, 11, 20
        // Row 2 (srcStart=1): border[1..4] = 60, 41, 18, 11
        // Row 3 (srcStart=0): border[0..3] = 70, 60, 41, 18
        var above = new byte[] { 10, 20, 30, 40 };
        var left = new byte[] { 50, 60, 70, 80 };
        var dst = new byte[16];
        Vp9DirectionalPredictor.D135Predict(topLeft: 5, above, left, dst, n: 4, stride: 4);

        Equal((byte)18, dst[0]);  Equal((byte)11, dst[1]);  Equal((byte)20, dst[2]);  Equal((byte)30, dst[3]);
        Equal((byte)41, dst[4]);  Equal((byte)18, dst[5]);  Equal((byte)11, dst[6]);  Equal((byte)20, dst[7]);
        Equal((byte)60, dst[8]);  Equal((byte)41, dst[9]);  Equal((byte)18, dst[10]); Equal((byte)11, dst[11]);
        Equal((byte)70, dst[12]); Equal((byte)60, dst[13]); Equal((byte)41, dst[14]); Equal((byte)18, dst[15]);
    }

    [TestMethod]
    public void Vp9DirectionalPredictor_D135_FlatInputProducesFlatOutput()
    {
        var above = new byte[8];
        var left = new byte[8];
        for (int i = 0; i < 8; i++) { above[i] = 100; left[i] = 100; }
        var dst = new byte[64];
        Vp9DirectionalPredictor.D135Predict(topLeft: 100, above, left, dst, n: 8, stride: 8);
        for (int i = 0; i < 64; i++) Equal((byte)100, dst[i]);
    }

    [TestMethod]
    public void Vp9DirectionalPredictor_D135_StridedDest_OnlyTouchesBlock()
    {
        var above = new byte[16];
        var left = new byte[16];
        for (int i = 0; i < 16; i++) { above[i] = 80; left[i] = 80; }
        const int stride = 32;
        var canvas = new byte[stride * 16];
        for (int i = 0; i < canvas.Length; i++) canvas[i] = 222;
        Vp9DirectionalPredictor.D135Predict(topLeft: 80, above, left, canvas, n: 16, stride);
        for (int row = 0; row < 16; row++)
        {
            for (int col = 0; col < 16; col++)
                Equal((byte)80, canvas[row * stride + col]);
            for (int col = 16; col < stride; col++)
                Equal((byte)222, canvas[row * stride + col]);
        }
    }

    [TestMethod]
    public void Vp9DirectionalPredictor_D135_RejectsInvalidArgs()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9DirectionalPredictor.D135Predict(0, new byte[5], new byte[5], new byte[25], n: 5, stride: 5));
        Throws<ArgumentException>(() =>
            Vp9DirectionalPredictor.D135Predict(0, new byte[3], new byte[4], new byte[16], n: 4, stride: 4));
        Throws<ArgumentException>(() =>
            Vp9DirectionalPredictor.D135Predict(0, new byte[4], new byte[3], new byte[16], n: 4, stride: 4));
    }

    [TestMethod]
    public void Vp9DirectionalPredictor_D117_4x4_KnownPattern()
    {
        // above = [10, 20, 30, 40]; left = [50, 60, 70, 80]; topLeft = 5.
        // Row 0 (AVG2): AVG2(5,10)=8, AVG2(10,20)=15, AVG2(20,30)=25, AVG2(30,40)=35.
        // Row 1 (AVG3): AVG3(50,5,10)=18, AVG3(5,10,20)=11, AVG3(10,20,30)=20, AVG3(20,30,40)=30.
        // Row 2 col 0: AVG3(5, 50, 60) = (5+100+60+2)>>2 = 41
        //         cols 1..3 = row 0 cols 0..2 = 8, 15, 25.
        // Row 3 col 0: AVG3(50, 60, 70) = 60
        //         cols 1..3 = row 1 cols 0..2 = 18, 11, 20.
        var above = new byte[] { 10, 20, 30, 40 };
        var left = new byte[] { 50, 60, 70, 80 };
        var dst = new byte[16];
        Vp9DirectionalPredictor.D117Predict(topLeft: 5, above, left, dst, n: 4, stride: 4);

        Equal((byte)8,  dst[0]);  Equal((byte)15, dst[1]);  Equal((byte)25, dst[2]);  Equal((byte)35, dst[3]);
        Equal((byte)18, dst[4]);  Equal((byte)11, dst[5]);  Equal((byte)20, dst[6]);  Equal((byte)30, dst[7]);
        Equal((byte)41, dst[8]);  Equal((byte)8,  dst[9]);  Equal((byte)15, dst[10]); Equal((byte)25, dst[11]);
        Equal((byte)60, dst[12]); Equal((byte)18, dst[13]); Equal((byte)11, dst[14]); Equal((byte)20, dst[15]);
    }

    [TestMethod]
    public void Vp9DirectionalPredictor_D117_FlatInputProducesFlatOutput()
    {
        var above = new byte[8];
        var left = new byte[8];
        for (int i = 0; i < 8; i++) { above[i] = 100; left[i] = 100; }
        var dst = new byte[64];
        Vp9DirectionalPredictor.D117Predict(topLeft: 100, above, left, dst, n: 8, stride: 8);
        for (int i = 0; i < 64; i++) Equal((byte)100, dst[i]);
    }

    [TestMethod]
    public void Vp9DirectionalPredictor_D117_RejectsInvalidArgs()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9DirectionalPredictor.D117Predict(0, new byte[5], new byte[5], new byte[25], n: 5, stride: 5));
        Throws<ArgumentException>(() =>
            Vp9DirectionalPredictor.D117Predict(0, new byte[3], new byte[4], new byte[16], n: 4, stride: 4));
        Throws<ArgumentException>(() =>
            Vp9DirectionalPredictor.D117Predict(0, new byte[4], new byte[3], new byte[16], n: 4, stride: 4));
    }
}
