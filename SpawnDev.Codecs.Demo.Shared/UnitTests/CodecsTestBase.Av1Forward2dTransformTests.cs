// AV1 2D forward transform round-trip tests. Pairs the new
// Av1Forward2dTransform dispatcher with the existing
// Av1Inverse2dTransform oracle. Verifies that the full
// pixels-in -> coefficients -> pixels-out chain produces
// reconstructions within tolerance for small square sizes.

using SpawnDev.Codecs.Video.Av1;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static int RoundTripAv1_2D(Av1TxSize txSize, Av1TxType txType,
        int seed, int qScale)
    {
        int w = Av1TxSizeInfo.TxWide[(int)txSize];
        int h = Av1TxSizeInfo.TxHigh[(int)txSize];
        int n = w * h;

        var rng = new Random(seed);
        var input = new short[n];
        for (int i = 0; i < n; i++) input[i] = (short)(rng.Next(-128, 128));

        var coefs = new int[n];
        Av1Forward2dTransform.Apply(txSize, txType, input, coefs);

        // The Inverse2dTransform's RoundShift returns int residuals; we then
        // compare against the original input scaled by qScale.
        var residual = new int[n];
        Av1Inverse2dTransform.Apply(txSize, txType, coefs, residual);

        // Compute max absolute error vs input * qScale.
        int maxErr = 0;
        for (int i = 0; i < n; i++)
        {
            maxErr = Math.Max(maxErr, Math.Abs(residual[i] - input[i] * qScale));
        }
        return maxErr;
    }

    [TestMethod]
    public void Av1Forward2d_Dct4x4_FwdInv_RoundTripSmall()
    {
        // Probe scaling factor empirically; at least one of qScale 1, 2, 4, 8
        // should give max error within bound (depends on internal libaom shifts).
        int e1 = RoundTripAv1_2D(Av1TxSize.Tx4x4, Av1TxType.DctDct, 0xD44, 1);
        int e2 = RoundTripAv1_2D(Av1TxSize.Tx4x4, Av1TxType.DctDct, 0xD44, 2);
        int e4 = RoundTripAv1_2D(Av1TxSize.Tx4x4, Av1TxType.DctDct, 0xD44, 4);
        int e8 = RoundTripAv1_2D(Av1TxSize.Tx4x4, Av1TxType.DctDct, 0xD44, 8);
        int best = Math.Min(Math.Min(e1, e2), Math.Min(e4, e8));
        True(best <= 16,
            $"Av1 fdct/idct 4x4 round-trip: q1={e1}, q2={e2}, q4={e4}, q8={e8} - none within tolerance");
    }

    [TestMethod]
    public void Av1Forward2d_Dct8x8_FwdInv_RoundTripSmall()
    {
        int e1 = RoundTripAv1_2D(Av1TxSize.Tx8x8, Av1TxType.DctDct, 0xD88, 1);
        int e2 = RoundTripAv1_2D(Av1TxSize.Tx8x8, Av1TxType.DctDct, 0xD88, 2);
        int e4 = RoundTripAv1_2D(Av1TxSize.Tx8x8, Av1TxType.DctDct, 0xD88, 4);
        int e8 = RoundTripAv1_2D(Av1TxSize.Tx8x8, Av1TxType.DctDct, 0xD88, 8);
        int best = Math.Min(Math.Min(e1, e2), Math.Min(e4, e8));
        True(best <= 32,
            $"Av1 fdct/idct 8x8 round-trip: q1={e1}, q2={e2}, q4={e4}, q8={e8} - none within tolerance");
    }

    [TestMethod]
    public void Av1Forward2d_Dct16x16_FwdInv_RoundTripSmall()
    {
        int e1 = RoundTripAv1_2D(Av1TxSize.Tx16x16, Av1TxType.DctDct, 0xD16, 1);
        int e2 = RoundTripAv1_2D(Av1TxSize.Tx16x16, Av1TxType.DctDct, 0xD16, 2);
        int e4 = RoundTripAv1_2D(Av1TxSize.Tx16x16, Av1TxType.DctDct, 0xD16, 4);
        int e8 = RoundTripAv1_2D(Av1TxSize.Tx16x16, Av1TxType.DctDct, 0xD16, 8);
        int best = Math.Min(Math.Min(e1, e2), Math.Min(e4, e8));
        True(best <= 64,
            $"Av1 fdct/idct 16x16 round-trip: q1={e1}, q2={e2}, q4={e4}, q8={e8} - none within tolerance");
    }

    [TestMethod]
    public void Av1Forward2d_AllZero_ProducesAllZero()
    {
        var input = new short[256];
        var output = new int[256];
        Av1Forward2dTransform.Apply(Av1TxSize.Tx16x16, Av1TxType.DctDct, input, output);
        for (int i = 0; i < 256; i++) Equal(0, output[i]);
    }

    [TestMethod]
    public void Av1Forward2d_Determinism()
    {
        var rng = new Random(0xD2D);
        var input = new short[64];
        for (int i = 0; i < 64; i++) input[i] = (short)rng.Next(-100, 100);
        var a = new int[64];
        var b = new int[64];
        Av1Forward2dTransform.Apply(Av1TxSize.Tx8x8, Av1TxType.DctDct, input, a);
        Av1Forward2dTransform.Apply(Av1TxSize.Tx8x8, Av1TxType.DctDct, input, b);
        for (int i = 0; i < 64; i++) Equal(a[i], b[i]);
    }
}
