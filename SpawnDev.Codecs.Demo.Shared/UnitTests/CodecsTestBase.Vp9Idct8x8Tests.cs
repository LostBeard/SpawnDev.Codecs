// Tests for Vp9Idct8x8Reference. Mirrors the slice-116 4x4 test
// structure - zero, DC-only, clipping, determinism, bounds checks.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] MakePredictor8x8(byte value)
    {
        var dest = new byte[64];
        for (int i = 0; i < 64; i++) dest[i] = value;
        return dest;
    }

    [TestMethod]
    public void Vp9Idct8x8_ZeroCoefficients_LeavesPredictorUnchanged()
    {
        var coeffs = new short[64];
        var dest = MakePredictor8x8(128);
        Vp9Idct8x8Reference.Idct8x8_64_Add(coeffs, dest, 8);
        for (int i = 0; i < 64; i++) Equal((byte)128, dest[i]);
    }

    [TestMethod]
    public void Vp9Idct8x8_DcOnlyCoefficient_AppliesUniformResidual()
    {
        // DC = 1024 at [0,0]. Hand-derived:
        //   row 0 iDCT of [1024,0,...,0] -> [724,724,...,724]
        //   other rows -> all zero
        //   col pass on [724,0,...,0] -> [512,512,...,512]
        //   residual = ROUND_POWER_OF_TWO(512, 5) = (512+16)>>5 = 16
        //   All 64 pixels get +16.
        var coeffs = new short[64];
        coeffs[0] = 1024;
        var dest = MakePredictor8x8(100);
        Vp9Idct8x8Reference.Idct8x8_64_Add(coeffs, dest, 8);
        for (int i = 0; i < 64; i++)
            Equal((byte)116, dest[i]); // 100 + 16
    }

    [TestMethod]
    public void Vp9Idct8x8_Clips_HighResidualToUpper255()
    {
        var coeffs = new short[64];
        coeffs[0] = 16384; // push residual well above 256
        var dest = MakePredictor8x8(200);
        Vp9Idct8x8Reference.Idct8x8_64_Add(coeffs, dest, 8);
        for (int i = 0; i < 64; i++) Equal((byte)255, dest[i]);
    }

    [TestMethod]
    public void Vp9Idct8x8_Clips_NegativeResidualToZero()
    {
        var coeffs = new short[64];
        coeffs[0] = -16384;
        var dest = MakePredictor8x8(100);
        Vp9Idct8x8Reference.Idct8x8_64_Add(coeffs, dest, 8);
        for (int i = 0; i < 64; i++) Equal((byte)0, dest[i]);
    }

    [TestMethod]
    public void Vp9Idct8x8_StridedDest_OnlyTouches8x8Block()
    {
        // 8x8 block at top-left of an 8-row x 24-col canvas.
        var canvas = new byte[8 * 24];
        for (int i = 0; i < canvas.Length; i++) canvas[i] = 77;
        var coeffs = new short[64];
        coeffs[0] = 1024;
        Vp9Idct8x8Reference.Idct8x8_64_Add(coeffs, canvas, stride: 24);
        for (int row = 0; row < 8; row++)
        {
            for (int col = 0; col < 8; col++)
                Equal((byte)93, canvas[row * 24 + col]); // 77 + 16
            for (int col = 8; col < 24; col++)
                Equal((byte)77, canvas[row * 24 + col]); // untouched
        }
    }

    [TestMethod]
    public void Vp9Idct8x8_IsDeterministic_ForRandomInputs()
    {
        var rng = new Random(0x8);
        var coeffs = new short[64];
        for (int trial = 0; trial < 30; trial++)
        {
            for (int i = 0; i < 64; i++) coeffs[i] = (short)rng.Next(-4096, 4096);
            var a = MakePredictor8x8(128);
            var b = MakePredictor8x8(128);
            Vp9Idct8x8Reference.Idct8x8_64_Add(coeffs, a, 8);
            Vp9Idct8x8Reference.Idct8x8_64_Add(coeffs, b, 8);
            True(a.AsSpan().SequenceEqual(b), $"non-deterministic at trial {trial}");
        }
    }

    [TestMethod]
    public void Vp9Idct8x8_Throws_OnShortInput()
    {
        var dest = MakePredictor8x8(0);
        bool threw = false;
        try { Vp9Idct8x8Reference.Idct8x8_64_Add(new short[32], dest, 8); }
        catch (ArgumentException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void Vp9Idct8x8_Throws_OnBadStride()
    {
        bool threw = false;
        try { Vp9Idct8x8Reference.Idct8x8_64_Add(new short[64], new byte[64], 4); }
        catch (ArgumentException) { threw = true; }
        True(threw);
    }
}
