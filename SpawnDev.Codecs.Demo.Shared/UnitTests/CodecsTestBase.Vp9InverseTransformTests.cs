// Tests for Vp9InverseTransform (slice 174). Verifies the dispatcher
// produces the same output as a direct call to the per-size
// reference for each (txType, txSize) combination, and that
// invalid combinations throw.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static short[] BuildResidual(int n)
    {
        var coeffs = new short[n * n];
        // Mild residual - keep range tight so dest+residual stays in [0, 255]
        // for trivial tests; the per-size references handle clipping.
        for (int i = 0; i < coeffs.Length; i++)
            coeffs[i] = (short)((i * 13) % 17 - 8);
        return coeffs;
    }

    private static byte[] BuildDest(int n)
    {
        var dest = new byte[n * n];
        for (int i = 0; i < dest.Length; i++)
            dest[i] = (byte)(64 + (i * 7) % 96);
        return dest;
    }

    [TestMethod]
    public void Vp9InverseTransform_4x4_DctDct_MatchesDirectIht()
    {
        var input = BuildResidual(4);
        var expected = BuildDest(4);
        Vp9Iht4x4Reference.Iht4x4_16_Add(Vp9TxType4x4.DctDct, input, expected, 4);

        var actual = BuildDest(4);
        Vp9InverseTransform.Apply(Vp9TxType.DctDct, Vp9TxSize.Tx4x4, input, actual, 4);

        for (int i = 0; i < 16; i++) Equal(expected[i], actual[i]);
    }

    [TestMethod]
    public void Vp9InverseTransform_4x4_AdstAdst_MatchesDirectIht()
    {
        var input = BuildResidual(4);
        var expected = BuildDest(4);
        Vp9Iht4x4Reference.Iht4x4_16_Add(Vp9TxType4x4.AdstAdst, input, expected, 4);

        var actual = BuildDest(4);
        Vp9InverseTransform.Apply(Vp9TxType.AdstAdst, Vp9TxSize.Tx4x4, input, actual, 4);

        for (int i = 0; i < 16; i++) Equal(expected[i], actual[i]);
    }

    [TestMethod]
    public void Vp9InverseTransform_8x8_DctDct_MatchesDirectIht()
    {
        var input = BuildResidual(8);
        var expected = BuildDest(8);
        Vp9Iht8x8Reference.Iht8x8_64_Add(Vp9TxType8x8.DctDct, input, expected, 8);

        var actual = BuildDest(8);
        Vp9InverseTransform.Apply(Vp9TxType.DctDct, Vp9TxSize.Tx8x8, input, actual, 8);

        for (int i = 0; i < 64; i++) Equal(expected[i], actual[i]);
    }

    [TestMethod]
    public void Vp9InverseTransform_8x8_AdstDct_MatchesDirectIht()
    {
        var input = BuildResidual(8);
        var expected = BuildDest(8);
        Vp9Iht8x8Reference.Iht8x8_64_Add(Vp9TxType8x8.AdstDct, input, expected, 8);

        var actual = BuildDest(8);
        Vp9InverseTransform.Apply(Vp9TxType.AdstDct, Vp9TxSize.Tx8x8, input, actual, 8);

        for (int i = 0; i < 64; i++) Equal(expected[i], actual[i]);
    }

    [TestMethod]
    public void Vp9InverseTransform_16x16_DctDct_MatchesDirectIht()
    {
        var input = BuildResidual(16);
        var expected = BuildDest(16);
        Vp9Iht16x16Reference.Iht16x16_256_Add(Vp9TxType16x16.DctDct, input, expected, 16);

        var actual = BuildDest(16);
        Vp9InverseTransform.Apply(Vp9TxType.DctDct, Vp9TxSize.Tx16x16, input, actual, 16);

        for (int i = 0; i < 256; i++) Equal(expected[i], actual[i]);
    }

    [TestMethod]
    public void Vp9InverseTransform_16x16_DctAdst_MatchesDirectIht()
    {
        var input = BuildResidual(16);
        var expected = BuildDest(16);
        Vp9Iht16x16Reference.Iht16x16_256_Add(Vp9TxType16x16.DctAdst, input, expected, 16);

        var actual = BuildDest(16);
        Vp9InverseTransform.Apply(Vp9TxType.DctAdst, Vp9TxSize.Tx16x16, input, actual, 16);

        for (int i = 0; i < 256; i++) Equal(expected[i], actual[i]);
    }

    [TestMethod]
    public void Vp9InverseTransform_32x32_DctDct_MatchesDirectIdct()
    {
        var input = BuildResidual(32);
        var expected = BuildDest(32);
        Vp9Idct32x32Reference.Idct32x32_1024_Add(input, expected, 32);

        var actual = BuildDest(32);
        Vp9InverseTransform.Apply(Vp9TxType.DctDct, Vp9TxSize.Tx32x32, input, actual, 32);

        for (int i = 0; i < 1024; i++) Equal(expected[i], actual[i]);
    }

    [TestMethod]
    public void Vp9InverseTransform_32x32_NonDctDct_Throws()
    {
        var input = new short[1024];
        var dest = new byte[1024];
        Throws<ArgumentException>(() =>
            Vp9InverseTransform.Apply(Vp9TxType.AdstDct, Vp9TxSize.Tx32x32, input, dest, 32));
        Throws<ArgumentException>(() =>
            Vp9InverseTransform.Apply(Vp9TxType.DctAdst, Vp9TxSize.Tx32x32, input, dest, 32));
        Throws<ArgumentException>(() =>
            Vp9InverseTransform.Apply(Vp9TxType.AdstAdst, Vp9TxSize.Tx32x32, input, dest, 32));
    }

    [TestMethod]
    public void Vp9InverseTransform_UnknownTxSize_Throws()
    {
        var input = new short[16];
        var dest = new byte[16];
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9InverseTransform.Apply(Vp9TxType.DctDct, (Vp9TxSize)99, input, dest, 4));
    }
}
