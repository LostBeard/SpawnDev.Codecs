// Tests for Vp9Iht8x8Reference dispatcher. Same structure as the iHT 4x4
// tests from slice 122 - verify each tx_type routes to the expected
// row + column 1D transform pair.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] MakeIht8Predictor(byte value)
    {
        var dest = new byte[64];
        for (int i = 0; i < 64; i++) dest[i] = value;
        return dest;
    }

    [TestMethod]
    public void Vp9Iht8x8_DctDct_MatchesIdct8x8Reference()
    {
        var rng = new Random(0xE1);
        for (int trial = 0; trial < 10; trial++)
        {
            var coeffs = new short[64];
            for (int i = 0; i < 64; i++) coeffs[i] = (short)rng.Next(-4096, 4096);

            var idctDest = MakeIht8Predictor(100);
            Vp9Idct8x8Reference.Idct8x8_64_Add(coeffs, idctDest, 8);

            var ihtDest = MakeIht8Predictor(100);
            Vp9Iht8x8Reference.Iht8x8_64_Add(Vp9TxType8x8.DctDct, coeffs, ihtDest, 8);

            True(idctDest.AsSpan().SequenceEqual(ihtDest),
                $"DctDct dispatcher must match standalone iDCT 8x8 (trial {trial})");
        }
    }

    [TestMethod]
    public void Vp9Iht8x8_AdstAdst_MatchesIadst8x8Reference()
    {
        var rng = new Random(0xE3);
        for (int trial = 0; trial < 10; trial++)
        {
            var coeffs = new short[64];
            for (int i = 0; i < 64; i++) coeffs[i] = (short)rng.Next(-4096, 4096);

            var iadstDest = MakeIht8Predictor(100);
            Vp9Iadst8x8Reference.IadstAdst8x8_64_Add(coeffs, iadstDest, 8);

            var ihtDest = MakeIht8Predictor(100);
            Vp9Iht8x8Reference.Iht8x8_64_Add(Vp9TxType8x8.AdstAdst, coeffs, ihtDest, 8);

            True(iadstDest.AsSpan().SequenceEqual(ihtDest),
                $"AdstAdst dispatcher must match standalone iADST 8x8 (trial {trial})");
        }
    }

    [TestMethod]
    public void Vp9Iht8x8_AdstDct_VsDctAdst_ProduceDifferentOutput()
    {
        // Pattern that breaks 2D symmetry: non-zero AC coefficients that
        // aren't symmetric across the main diagonal. If the dispatcher
        // swapped row + col transforms, the two modes would coincide.
        var coeffs = new short[64];
        coeffs[1] = 800;     // row 0, col 1
        coeffs[9] = -400;    // row 1, col 1
        coeffs[17] = 600;    // row 2, col 1
        coeffs[25] = 300;    // row 3, col 1

        var adstDct = MakeIht8Predictor(100);
        Vp9Iht8x8Reference.Iht8x8_64_Add(Vp9TxType8x8.AdstDct, coeffs, adstDct, 8);

        var dctAdst = MakeIht8Predictor(100);
        Vp9Iht8x8Reference.Iht8x8_64_Add(Vp9TxType8x8.DctAdst, coeffs, dctAdst, 8);

        False(adstDct.AsSpan().SequenceEqual(dctAdst),
            "AdstDct and DctAdst must not produce identical output on asymmetric AC");
    }

    [TestMethod]
    public void Vp9Iht8x8_AllFourTxTypes_ZeroInput_LeavePredictorUnchanged()
    {
        var coeffs = new short[64];
        foreach (var tx in new[] { Vp9TxType8x8.DctDct, Vp9TxType8x8.AdstDct,
                                    Vp9TxType8x8.DctAdst, Vp9TxType8x8.AdstAdst })
        {
            var dest = MakeIht8Predictor(128);
            Vp9Iht8x8Reference.Iht8x8_64_Add(tx, coeffs, dest, 8);
            for (int i = 0; i < 64; i++) Equal((byte)128, dest[i]);
        }
    }

    [TestMethod]
    public void Vp9Iht8x8_AllFourTxTypes_ProduceValidPixels()
    {
        var rng = new Random(0xE7);
        var coeffs = new short[64];
        for (int i = 0; i < 64; i++) coeffs[i] = (short)rng.Next(-4096, 4096);
        foreach (var tx in new[] { Vp9TxType8x8.DctDct, Vp9TxType8x8.AdstDct,
                                    Vp9TxType8x8.DctAdst, Vp9TxType8x8.AdstAdst })
        {
            var a = MakeIht8Predictor(128);
            var b = MakeIht8Predictor(128);
            Vp9Iht8x8Reference.Iht8x8_64_Add(tx, coeffs, a, 8);
            Vp9Iht8x8Reference.Iht8x8_64_Add(tx, coeffs, b, 8);
            True(a.AsSpan().SequenceEqual(b), $"non-deterministic on tx {tx}");
            for (int i = 0; i < 64; i++)
                True(a[i] >= 0 && a[i] <= 255, $"tx {tx} pixel out of range at {i}: {a[i]}");
        }
    }

    [TestMethod]
    public void Vp9Iht8x8_Throws_OnShortInput()
    {
        bool threw = false;
        try { Vp9Iht8x8Reference.Iht8x8_64_Add(Vp9TxType8x8.DctDct, new short[32], new byte[64], 8); }
        catch (ArgumentException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void Vp9Iht8x8_Throws_OnBadStride()
    {
        bool threw = false;
        try { Vp9Iht8x8Reference.Iht8x8_64_Add(Vp9TxType8x8.DctDct, new short[64], new byte[64], 4); }
        catch (ArgumentException) { threw = true; }
        True(threw);
    }
}
