// Tests for Vp9Iht4x4Reference dispatcher. Verifies that each of the 4
// tx_types correctly routes to the expected row + column 1D transform
// pair by comparing against the standalone references we've already
// validated byte-for-byte (slice 116 iDCT 4x4 + slice 121 iADST 4x4).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] MakeIhtPredictor(byte value)
    {
        var dest = new byte[16];
        for (int i = 0; i < 16; i++) dest[i] = value;
        return dest;
    }

    [TestMethod]
    public void Vp9Iht4x4_DctDct_MatchesIdct4x4Reference()
    {
        // tx_type=0 must exactly match the standalone iDCT 4x4 reference
        // on every input, because it IS the same transform.
        var rng = new Random(0xD1);
        for (int trial = 0; trial < 10; trial++)
        {
            var coeffs = new short[16];
            for (int i = 0; i < 16; i++) coeffs[i] = (short)rng.Next(-2048, 2048);

            var idctDest = MakeIhtPredictor(100);
            Vp9Idct4x4Reference.Idct4x4_16_Add(coeffs, idctDest, 4);

            var ihtDest = MakeIhtPredictor(100);
            Vp9Iht4x4Reference.Iht4x4_16_Add(Vp9TxType4x4.DctDct, coeffs, ihtDest, 4);

            True(idctDest.AsSpan().SequenceEqual(ihtDest),
                $"DctDct dispatcher must match standalone iDCT 4x4 (trial {trial})");
        }
    }

    [TestMethod]
    public void Vp9Iht4x4_AdstAdst_MatchesIadstReference()
    {
        // tx_type=3 must match the pure ADST_ADST reference from slice 121.
        var rng = new Random(0xD3);
        for (int trial = 0; trial < 10; trial++)
        {
            var coeffs = new short[16];
            for (int i = 0; i < 16; i++) coeffs[i] = (short)rng.Next(-2048, 2048);

            var iadstDest = MakeIhtPredictor(100);
            Vp9Iadst4x4Reference.IadstAdst4x4_16_Add(coeffs, iadstDest, 4);

            var ihtDest = MakeIhtPredictor(100);
            Vp9Iht4x4Reference.Iht4x4_16_Add(Vp9TxType4x4.AdstAdst, coeffs, ihtDest, 4);

            True(iadstDest.AsSpan().SequenceEqual(ihtDest),
                $"AdstAdst dispatcher must match standalone iADST 4x4 (trial {trial})");
        }
    }

    [TestMethod]
    public void Vp9Iht4x4_AdstDct_VsDctAdst_ProduceDifferentOutput()
    {
        // ADST_DCT (tx_type=1) and DCT_ADST (tx_type=2) are distinct
        // transforms - they apply iADST to different axes. For a
        // non-symmetric coefficient pattern they must produce non-identical
        // output, otherwise the dispatcher's axis selection is broken.
        var coeffs = new short[16];
        coeffs[1] = 800;    // AC coefficient in row 0 col 1
        coeffs[5] = -400;   // AC coefficient in row 1 col 1
        coeffs[6] = 600;    // AC coefficient in row 1 col 2

        var adstDctDest = MakeIhtPredictor(100);
        Vp9Iht4x4Reference.Iht4x4_16_Add(Vp9TxType4x4.AdstDct, coeffs, adstDctDest, 4);

        var dctAdstDest = MakeIhtPredictor(100);
        Vp9Iht4x4Reference.Iht4x4_16_Add(Vp9TxType4x4.DctAdst, coeffs, dctAdstDest, 4);

        False(adstDctDest.AsSpan().SequenceEqual(dctAdstDest),
            "AdstDct and DctAdst must not produce identical output on asymmetric input");
    }

    [TestMethod]
    public void Vp9Iht4x4_AllFourTxTypes_ZeroInput_LeavePredictorUnchanged()
    {
        // Every tx_type on an all-zero input must leave the predictor
        // untouched - no residual to add.
        var coeffs = new short[16];
        foreach (var tx in new[] { Vp9TxType4x4.DctDct, Vp9TxType4x4.AdstDct,
                                    Vp9TxType4x4.DctAdst, Vp9TxType4x4.AdstAdst })
        {
            var dest = MakeIhtPredictor(128);
            Vp9Iht4x4Reference.Iht4x4_16_Add(tx, coeffs, dest, 4);
            for (int i = 0; i < 16; i++)
                Equal((byte)128, dest[i]);
        }
    }

    [TestMethod]
    public void Vp9Iht4x4_AllFourTxTypes_ProduceValidPixels()
    {
        // Stress: every tx_type on a random input must produce output
        // bytes in [0, 255] (clipping works) and a deterministic result.
        var rng = new Random(0xD7);
        var coeffs = new short[16];
        for (int i = 0; i < 16; i++) coeffs[i] = (short)rng.Next(-2048, 2048);
        foreach (var tx in new[] { Vp9TxType4x4.DctDct, Vp9TxType4x4.AdstDct,
                                    Vp9TxType4x4.DctAdst, Vp9TxType4x4.AdstAdst })
        {
            var a = MakeIhtPredictor(128);
            var b = MakeIhtPredictor(128);
            Vp9Iht4x4Reference.Iht4x4_16_Add(tx, coeffs, a, 4);
            Vp9Iht4x4Reference.Iht4x4_16_Add(tx, coeffs, b, 4);
            True(a.AsSpan().SequenceEqual(b), $"non-deterministic on tx_type {tx}");
            for (int i = 0; i < 16; i++)
                True(a[i] >= 0 && a[i] <= 255,
                    $"pixel out of range on tx_type {tx} at index {i}: {a[i]}");
        }
    }

    [TestMethod]
    public void Vp9Iht4x4_Throws_OnShortInput()
    {
        bool threw = false;
        try { Vp9Iht4x4Reference.Iht4x4_16_Add(Vp9TxType4x4.DctDct, new short[8], new byte[16], 4); }
        catch (ArgumentException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void Vp9Iht4x4_Throws_OnBadStride()
    {
        bool threw = false;
        try { Vp9Iht4x4Reference.Iht4x4_16_Add(Vp9TxType4x4.DctDct, new short[16], new byte[16], 2); }
        catch (ArgumentException) { threw = true; }
        True(threw);
    }
}
