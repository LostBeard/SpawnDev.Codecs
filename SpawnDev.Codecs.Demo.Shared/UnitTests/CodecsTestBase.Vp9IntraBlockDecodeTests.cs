// Tests for Vp9IntraBlockDecode (slice 175). Verifies the composed
// "predict + inverse-transform" output matches the manually-stepped
// equivalent for a few (mode, tx_size) combinations.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] EdgePattern(int len, int seed)
    {
        var arr = new byte[len];
        for (int i = 0; i < len; i++) arr[i] = (byte)((seed + i * 11) & 0xFF);
        return arr;
    }

    private static short[] CoeffPattern(int n)
    {
        var arr = new short[n * n];
        for (int i = 0; i < arr.Length; i++)
            arr[i] = (short)((i * 7) % 23 - 11);  // -11..+11 spread
        return arr;
    }

    [TestMethod]
    public void Vp9IntraBlockDecode_TxSizeToN_AllSizes()
    {
        Equal(4, Vp9IntraBlockDecode.TxSizeToN(Vp9TxSize.Tx4x4));
        Equal(8, Vp9IntraBlockDecode.TxSizeToN(Vp9TxSize.Tx8x8));
        Equal(16, Vp9IntraBlockDecode.TxSizeToN(Vp9TxSize.Tx16x16));
        Equal(32, Vp9IntraBlockDecode.TxSizeToN(Vp9TxSize.Tx32x32));
    }

    [TestMethod]
    public void Vp9IntraBlockDecode_TxSizeToN_RejectsUnknown()
    {
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9IntraBlockDecode.TxSizeToN((Vp9TxSize)99));
    }

    [TestMethod]
    public void Vp9IntraBlockDecode_DcPred_4x4_ComposesPredictAndIht()
    {
        var above = EdgePattern(4, 30);
        var left = EdgePattern(4, 80);
        var coeffs = CoeffPattern(4);

        // Manually composed reference.
        var expected = new byte[16];
        Vp9IntraPredictor.Predict(
            Vp9IntraMode.DcPred, topLeft: 0, above, left,
            expected, n: 4, stride: 4);
        Vp9InverseTransform.Apply(
            Vp9TxType.DctDct, Vp9TxSize.Tx4x4, coeffs, expected, 4);

        // Composed call.
        var actual = new byte[16];
        Vp9IntraBlockDecode.Decode(
            Vp9IntraMode.DcPred, Vp9TxSize.Tx4x4,
            topLeft: 0, above, left,
            coeffs, actual, stride: 4);

        for (int i = 0; i < 16; i++) Equal(expected[i], actual[i]);
    }

    [TestMethod]
    public void Vp9IntraBlockDecode_VPred_8x8_UsesAdstDctTxType()
    {
        var above = EdgePattern(8, 50);
        var left = EdgePattern(8, 90);
        var coeffs = CoeffPattern(8);

        var expected = new byte[64];
        Vp9IntraPredictor.Predict(
            Vp9IntraMode.VPred, topLeft: 0, above, left,
            expected, n: 8, stride: 8);
        // V_PRED -> AdstDct
        Vp9InverseTransform.Apply(
            Vp9TxType.AdstDct, Vp9TxSize.Tx8x8, coeffs, expected, 8);

        var actual = new byte[64];
        Vp9IntraBlockDecode.Decode(
            Vp9IntraMode.VPred, Vp9TxSize.Tx8x8,
            topLeft: 0, above, left,
            coeffs, actual, stride: 8);

        for (int i = 0; i < 64; i++) Equal(expected[i], actual[i]);
    }

    [TestMethod]
    public void Vp9IntraBlockDecode_TmPred_16x16_UsesAdstAdstTxType()
    {
        var above = EdgePattern(16, 10);
        var left = EdgePattern(16, 70);
        var coeffs = CoeffPattern(16);
        const byte topLeft = 100;

        var expected = new byte[256];
        Vp9IntraPredictor.Predict(
            Vp9IntraMode.TmPred, topLeft, above, left,
            expected, n: 16, stride: 16);
        // TM_PRED -> AdstAdst
        Vp9InverseTransform.Apply(
            Vp9TxType.AdstAdst, Vp9TxSize.Tx16x16, coeffs, expected, 16);

        var actual = new byte[256];
        Vp9IntraBlockDecode.Decode(
            Vp9IntraMode.TmPred, Vp9TxSize.Tx16x16,
            topLeft, above, left,
            coeffs, actual, stride: 16);

        for (int i = 0; i < 256; i++) Equal(expected[i], actual[i]);
    }

    [TestMethod]
    public void Vp9IntraBlockDecode_32x32_AlwaysUsesDctDctEvenForTmMode()
    {
        var above = EdgePattern(32, 40);
        var left = EdgePattern(32, 90);
        var coeffs = CoeffPattern(32);
        const byte topLeft = 50;

        var expected = new byte[1024];
        Vp9IntraPredictor.Predict(
            Vp9IntraMode.TmPred, topLeft, above, left,
            expected, n: 32, stride: 32);
        // 32x32 forces DctDct regardless of intra mode.
        Vp9InverseTransform.Apply(
            Vp9TxType.DctDct, Vp9TxSize.Tx32x32, coeffs, expected, 32);

        var actual = new byte[1024];
        Vp9IntraBlockDecode.Decode(
            Vp9IntraMode.TmPred, Vp9TxSize.Tx32x32,
            topLeft, above, left,
            coeffs, actual, stride: 32);

        for (int i = 0; i < 1024; i++) Equal(expected[i], actual[i]);
    }

    [TestMethod]
    public void Vp9IntraBlockDecode_DcPred_NoEdges_Uses128Variant()
    {
        var above = new byte[4];
        var left = new byte[4];
        var coeffs = new short[16];  // zero residual -> output stays at 128

        var dst = new byte[16];
        Vp9IntraBlockDecode.Decode(
            Vp9IntraMode.DcPred, Vp9TxSize.Tx4x4,
            topLeft: 0, above, left,
            coeffs, dst, stride: 4,
            haveAbove: false, haveLeft: false);

        for (int i = 0; i < 16; i++) Equal((byte)128, dst[i]);
    }

    [TestMethod]
    public void Vp9IntraBlockDecode_D207_LeftOnly_4x4_Roundtrips()
    {
        var above = new byte[4];
        var left = EdgePattern(4, 100);
        var coeffs = new short[16];  // zero residual; output should match D207 prediction

        var expected = new byte[16];
        Vp9DirectionalPredictor.D207Predict(left, expected, 4, 4);

        var actual = new byte[16];
        Vp9IntraBlockDecode.Decode(
            Vp9IntraMode.D207Pred, Vp9TxSize.Tx4x4,
            topLeft: 0, above, left,
            coeffs, actual, stride: 4);

        for (int i = 0; i < 16; i++) Equal(expected[i], actual[i]);
    }
}
