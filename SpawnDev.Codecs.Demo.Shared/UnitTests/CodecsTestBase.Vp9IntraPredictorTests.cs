// Tests for Vp9IntraPredictor (slice 170). Verify the dispatcher
// routes each of the 10 intra modes to the same output the
// per-mode predictor would produce, and that DC variant selection
// honours the haveAbove / haveLeft flags.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] BuildPattern(int len, int seed)
    {
        var arr = new byte[len];
        for (int i = 0; i < len; i++) arr[i] = (byte)((seed + i * 7) & 0xFF);
        return arr;
    }

    [TestMethod]
    public void Vp9IntraPredictor_Dispatches_DcPred_BothEdges()
    {
        var above = BuildPattern(4, 10);
        var left = BuildPattern(4, 50);

        var expected = new byte[16];
        Vp9DcPredictor.DcPredict(above, left, expected, 4, 4);

        var actual = new byte[16];
        Vp9IntraPredictor.Predict(
            Vp9IntraMode.DcPred, topLeft: 0, above, left,
            actual, n: 4, stride: 4);

        for (int i = 0; i < 16; i++) Equal(expected[i], actual[i]);
    }

    [TestMethod]
    public void Vp9IntraPredictor_Dispatches_DcPred_TopOnly_WhenLeftMissing()
    {
        var above = BuildPattern(8, 30);
        var left = new byte[8];

        var expected = new byte[64];
        Vp9DcPredictor.DcPredictTop(above, expected, 8, 8);

        var actual = new byte[64];
        Vp9IntraPredictor.Predict(
            Vp9IntraMode.DcPred, topLeft: 0, above, left,
            actual, n: 8, stride: 8,
            haveAbove: true, haveLeft: false);

        for (int i = 0; i < 64; i++) Equal(expected[i], actual[i]);
    }

    [TestMethod]
    public void Vp9IntraPredictor_Dispatches_DcPred_LeftOnly_WhenAboveMissing()
    {
        var above = new byte[8];
        var left = BuildPattern(8, 70);

        var expected = new byte[64];
        Vp9DcPredictor.DcPredictLeft(left, expected, 8, 8);

        var actual = new byte[64];
        Vp9IntraPredictor.Predict(
            Vp9IntraMode.DcPred, topLeft: 0, above, left,
            actual, n: 8, stride: 8,
            haveAbove: false, haveLeft: true);

        for (int i = 0; i < 64; i++) Equal(expected[i], actual[i]);
    }

    [TestMethod]
    public void Vp9IntraPredictor_Dispatches_DcPred_128_WhenNoEdges()
    {
        var above = new byte[4];
        var left = new byte[4];

        var actual = new byte[16];
        Vp9IntraPredictor.Predict(
            Vp9IntraMode.DcPred, topLeft: 0, above, left,
            actual, n: 4, stride: 4,
            haveAbove: false, haveLeft: false);

        for (int i = 0; i < 16; i++) Equal((byte)128, actual[i]);
    }

    [TestMethod]
    public void Vp9IntraPredictor_Dispatches_VPred_MatchesDirectCall()
    {
        var above = BuildPattern(4, 5);
        var left = new byte[4];

        var expected = new byte[16];
        Vp9VHPredictor.VPredict(above, expected, 4, 4);

        var actual = new byte[16];
        Vp9IntraPredictor.Predict(
            Vp9IntraMode.VPred, topLeft: 0, above, left,
            actual, n: 4, stride: 4);

        for (int i = 0; i < 16; i++) Equal(expected[i], actual[i]);
    }

    [TestMethod]
    public void Vp9IntraPredictor_Dispatches_HPred_MatchesDirectCall()
    {
        var above = new byte[4];
        var left = BuildPattern(4, 99);

        var expected = new byte[16];
        Vp9VHPredictor.HPredict(left, expected, 4, 4);

        var actual = new byte[16];
        Vp9IntraPredictor.Predict(
            Vp9IntraMode.HPred, topLeft: 0, above, left,
            actual, n: 4, stride: 4);

        for (int i = 0; i < 16; i++) Equal(expected[i], actual[i]);
    }

    [TestMethod]
    public void Vp9IntraPredictor_Dispatches_TmPred_MatchesDirectCall()
    {
        var above = BuildPattern(4, 10);
        var left = BuildPattern(4, 30);
        const byte topLeft = 5;

        var expected = new byte[16];
        Vp9TmPredictor.TmPredict(topLeft, above, left, expected, 4, 4);

        var actual = new byte[16];
        Vp9IntraPredictor.Predict(
            Vp9IntraMode.TmPred, topLeft, above, left,
            actual, n: 4, stride: 4);

        for (int i = 0; i < 16; i++) Equal(expected[i], actual[i]);
    }

    [TestMethod]
    public void Vp9IntraPredictor_Dispatches_D45_MatchesDirectCall()
    {
        var above = BuildPattern(8, 10);  // D45 needs 2N samples
        var left = new byte[4];

        var expected = new byte[16];
        Vp9DirectionalPredictor.D45Predict(above, expected, 4, 4);

        var actual = new byte[16];
        Vp9IntraPredictor.Predict(
            Vp9IntraMode.D45Pred, topLeft: 0, above, left,
            actual, n: 4, stride: 4);

        for (int i = 0; i < 16; i++) Equal(expected[i], actual[i]);
    }

    [TestMethod]
    public void Vp9IntraPredictor_Dispatches_D63_MatchesDirectCall()
    {
        var above = BuildPattern(8, 80);
        var left = new byte[4];

        var expected = new byte[16];
        Vp9DirectionalPredictor.D63Predict(above, expected, 4, 4);

        var actual = new byte[16];
        Vp9IntraPredictor.Predict(
            Vp9IntraMode.D63Pred, topLeft: 0, above, left,
            actual, n: 4, stride: 4);

        for (int i = 0; i < 16; i++) Equal(expected[i], actual[i]);
    }

    [TestMethod]
    public void Vp9IntraPredictor_Dispatches_D135_MatchesDirectCall()
    {
        var above = BuildPattern(4, 10);
        var left = BuildPattern(4, 50);
        const byte topLeft = 5;

        var expected = new byte[16];
        Vp9DirectionalPredictor.D135Predict(topLeft, above, left, expected, 4, 4);

        var actual = new byte[16];
        Vp9IntraPredictor.Predict(
            Vp9IntraMode.D135Pred, topLeft, above, left,
            actual, n: 4, stride: 4);

        for (int i = 0; i < 16; i++) Equal(expected[i], actual[i]);
    }

    [TestMethod]
    public void Vp9IntraPredictor_Dispatches_D117_MatchesDirectCall()
    {
        var above = BuildPattern(4, 10);
        var left = BuildPattern(4, 50);
        const byte topLeft = 5;

        var expected = new byte[16];
        Vp9DirectionalPredictor.D117Predict(topLeft, above, left, expected, 4, 4);

        var actual = new byte[16];
        Vp9IntraPredictor.Predict(
            Vp9IntraMode.D117Pred, topLeft, above, left,
            actual, n: 4, stride: 4);

        for (int i = 0; i < 16; i++) Equal(expected[i], actual[i]);
    }

    [TestMethod]
    public void Vp9IntraPredictor_Dispatches_D153_MatchesDirectCall()
    {
        var above = BuildPattern(4, 10);
        var left = BuildPattern(4, 50);
        const byte topLeft = 5;

        var expected = new byte[16];
        Vp9DirectionalPredictor.D153Predict(topLeft, above, left, expected, 4, 4);

        var actual = new byte[16];
        Vp9IntraPredictor.Predict(
            Vp9IntraMode.D153Pred, topLeft, above, left,
            actual, n: 4, stride: 4);

        for (int i = 0; i < 16; i++) Equal(expected[i], actual[i]);
    }

    [TestMethod]
    public void Vp9IntraPredictor_Dispatches_D207_MatchesDirectCall()
    {
        var above = new byte[4];
        var left = BuildPattern(4, 50);

        var expected = new byte[16];
        Vp9DirectionalPredictor.D207Predict(left, expected, 4, 4);

        var actual = new byte[16];
        Vp9IntraPredictor.Predict(
            Vp9IntraMode.D207Pred, topLeft: 0, above, left,
            actual, n: 4, stride: 4);

        for (int i = 0; i < 16; i++) Equal(expected[i], actual[i]);
    }

    [TestMethod]
    public void Vp9IntraPredictor_RejectsUnknownMode()
    {
        var above = new byte[4];
        var left = new byte[4];
        var dst = new byte[16];
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9IntraPredictor.Predict((Vp9IntraMode)99, 0, above, left, dst, 4, 4));
    }
}
