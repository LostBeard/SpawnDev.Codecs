// Tests for Vp9Iht16x16Reference dispatcher.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] MakeIht16Predictor(byte value)
    {
        var dest = new byte[256];
        for (int i = 0; i < 256; i++) dest[i] = value;
        return dest;
    }

    [TestMethod]
    public void Vp9Iht16x16_DctDct_MatchesIdct16x16Reference()
    {
        var rng = new Random(0xF1);
        for (int trial = 0; trial < 10; trial++)
        {
            var coeffs = new short[256];
            for (int i = 0; i < 256; i++) coeffs[i] = (short)rng.Next(-4096, 4096);

            var idctDest = MakeIht16Predictor(100);
            Vp9Idct16x16Reference.Idct16x16_256_Add(coeffs, idctDest, 16);

            var ihtDest = MakeIht16Predictor(100);
            Vp9Iht16x16Reference.Iht16x16_256_Add(Vp9TxType16x16.DctDct, coeffs, ihtDest, 16);

            True(idctDest.AsSpan().SequenceEqual(ihtDest),
                $"DctDct dispatcher must match standalone iDCT 16x16 (trial {trial})");
        }
    }

    [TestMethod]
    public void Vp9Iht16x16_AdstAdst_MatchesIadst16x16Reference()
    {
        var rng = new Random(0xF3);
        for (int trial = 0; trial < 10; trial++)
        {
            var coeffs = new short[256];
            for (int i = 0; i < 256; i++) coeffs[i] = (short)rng.Next(-4096, 4096);

            var iadstDest = MakeIht16Predictor(100);
            Vp9Iadst16x16Reference.IadstAdst16x16_256_Add(coeffs, iadstDest, 16);

            var ihtDest = MakeIht16Predictor(100);
            Vp9Iht16x16Reference.Iht16x16_256_Add(Vp9TxType16x16.AdstAdst, coeffs, ihtDest, 16);

            True(iadstDest.AsSpan().SequenceEqual(ihtDest),
                $"AdstAdst dispatcher must match standalone iADST 16x16 (trial {trial})");
        }
    }

    [TestMethod]
    public void Vp9Iht16x16_AdstDct_VsDctAdst_ProduceDifferentOutput()
    {
        var coeffs = new short[256];
        coeffs[1] = 1200;
        coeffs[17] = -800;
        coeffs[33] = 600;
        coeffs[49] = 400;

        var adstDct = MakeIht16Predictor(100);
        Vp9Iht16x16Reference.Iht16x16_256_Add(Vp9TxType16x16.AdstDct, coeffs, adstDct, 16);

        var dctAdst = MakeIht16Predictor(100);
        Vp9Iht16x16Reference.Iht16x16_256_Add(Vp9TxType16x16.DctAdst, coeffs, dctAdst, 16);

        False(adstDct.AsSpan().SequenceEqual(dctAdst),
            "AdstDct and DctAdst must not produce identical output on asymmetric AC");
    }

    [TestMethod]
    public void Vp9Iht16x16_AllFourTxTypes_ZeroInput_LeavePredictorUnchanged()
    {
        var coeffs = new short[256];
        foreach (var tx in new[] { Vp9TxType16x16.DctDct, Vp9TxType16x16.AdstDct,
                                    Vp9TxType16x16.DctAdst, Vp9TxType16x16.AdstAdst })
        {
            var dest = MakeIht16Predictor(128);
            Vp9Iht16x16Reference.Iht16x16_256_Add(tx, coeffs, dest, 16);
            for (int i = 0; i < 256; i++) Equal((byte)128, dest[i]);
        }
    }

    [TestMethod]
    public void Vp9Iht16x16_AllFourTxTypes_ProduceValidPixels()
    {
        var rng = new Random(0xF7);
        var coeffs = new short[256];
        for (int i = 0; i < 256; i++) coeffs[i] = (short)rng.Next(-4096, 4096);
        foreach (var tx in new[] { Vp9TxType16x16.DctDct, Vp9TxType16x16.AdstDct,
                                    Vp9TxType16x16.DctAdst, Vp9TxType16x16.AdstAdst })
        {
            var a = MakeIht16Predictor(128);
            var b = MakeIht16Predictor(128);
            Vp9Iht16x16Reference.Iht16x16_256_Add(tx, coeffs, a, 16);
            Vp9Iht16x16Reference.Iht16x16_256_Add(tx, coeffs, b, 16);
            True(a.AsSpan().SequenceEqual(b), $"non-deterministic on tx {tx}");
            for (int i = 0; i < 256; i++)
                True(a[i] >= 0 && a[i] <= 255, $"tx {tx} pixel out of range at {i}");
        }
    }

    [TestMethod]
    public void Vp9Iht16x16_Throws_OnShortInput()
    {
        bool threw = false;
        try { Vp9Iht16x16Reference.Iht16x16_256_Add(Vp9TxType16x16.DctDct, new short[128], new byte[256], 16); }
        catch (ArgumentException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void Vp9Iht16x16_Throws_OnBadStride()
    {
        bool threw = false;
        try { Vp9Iht16x16Reference.Iht16x16_256_Add(Vp9TxType16x16.DctDct, new short[256], new byte[256], 8); }
        catch (ArgumentException) { threw = true; }
        True(threw);
    }
}
