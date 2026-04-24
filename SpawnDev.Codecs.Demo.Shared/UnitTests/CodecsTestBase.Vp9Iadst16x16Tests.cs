// Tests for Vp9Iadst16x16Reference. Oracle for the 16x16 iADST kernel.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] MakeIadst16Predictor(byte value)
    {
        var dest = new byte[256];
        for (int i = 0; i < 256; i++) dest[i] = value;
        return dest;
    }

    [TestMethod]
    public void Vp9Iadst16x16_ZeroCoefficients_LeavesPredictorUnchanged()
    {
        var coeffs = new short[256];
        var dest = MakeIadst16Predictor(128);
        Vp9Iadst16x16Reference.IadstAdst16x16_256_Add(coeffs, dest, 16);
        for (int i = 0; i < 256; i++) Equal((byte)128, dest[i]);
    }

    [TestMethod]
    public void Vp9Iadst16x16_IsDeterministic_ForRandomInputs()
    {
        var rng = new Random(0xAD16);
        var coeffs = new short[256];
        for (int trial = 0; trial < 10; trial++)
        {
            for (int i = 0; i < 256; i++) coeffs[i] = (short)rng.Next(-4096, 4096);
            var a = MakeIadst16Predictor(128);
            var b = MakeIadst16Predictor(128);
            Vp9Iadst16x16Reference.IadstAdst16x16_256_Add(coeffs, a, 16);
            Vp9Iadst16x16Reference.IadstAdst16x16_256_Add(coeffs, b, 16);
            True(a.AsSpan().SequenceEqual(b), $"non-deterministic at trial {trial}");
        }
    }

    [TestMethod]
    public void Vp9Iadst16x16_SingleCoefficient_ProducesNonZeroResidual()
    {
        var coeffs = new short[256];
        coeffs[0] = 1024;
        var dest = MakeIadst16Predictor(100);
        Vp9Iadst16x16Reference.IadstAdst16x16_256_Add(coeffs, dest, 16);
        bool changed = false;
        for (int i = 0; i < 256; i++) if (dest[i] != 100) { changed = true; break; }
        True(changed);
    }

    [TestMethod]
    public void Vp9Iadst16x16_HighPositive_OutputsValidPixels()
    {
        var coeffs = new short[256];
        for (int i = 0; i < 256; i++) coeffs[i] = 2048;
        var dest = MakeIadst16Predictor(250);
        Vp9Iadst16x16Reference.IadstAdst16x16_256_Add(coeffs, dest, 16);
        for (int i = 0; i < 256; i++) True(dest[i] <= 255);
    }

    [TestMethod]
    public void Vp9Iadst16x16_LargeNegatives_ClipToZeroSomewhere()
    {
        var coeffs = new short[256];
        for (int i = 0; i < 256; i++) coeffs[i] = -2048;
        var dest = MakeIadst16Predictor(10);
        Vp9Iadst16x16Reference.IadstAdst16x16_256_Add(coeffs, dest, 16);
        for (int i = 0; i < 256; i++) True(dest[i] <= 255);
        bool anyZero = false;
        for (int i = 0; i < 256; i++) if (dest[i] == 0) { anyZero = true; break; }
        True(anyZero);
    }

    [TestMethod]
    public void Vp9Iadst16x16_StridedDest_OnlyTouches16x16Block()
    {
        var canvas = new byte[16 * 32];
        for (int i = 0; i < canvas.Length; i++) canvas[i] = 77;
        var coeffs = new short[256];
        coeffs[0] = 1024;
        Vp9Iadst16x16Reference.IadstAdst16x16_256_Add(coeffs, canvas, stride: 32);
        for (int row = 0; row < 16; row++)
            for (int col = 16; col < 32; col++)
                Equal((byte)77, canvas[row * 32 + col]);
    }

    [TestMethod]
    public void Vp9Iadst16x16_Throws_OnShortInput()
    {
        var dest = MakeIadst16Predictor(0);
        bool threw = false;
        try { Vp9Iadst16x16Reference.IadstAdst16x16_256_Add(new short[128], dest, 16); }
        catch (ArgumentException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void Vp9Iadst16x16_Throws_OnBadStride()
    {
        bool threw = false;
        try { Vp9Iadst16x16Reference.IadstAdst16x16_256_Add(new short[256], new byte[256], 8); }
        catch (ArgumentException) { threw = true; }
        True(threw);
    }
}
