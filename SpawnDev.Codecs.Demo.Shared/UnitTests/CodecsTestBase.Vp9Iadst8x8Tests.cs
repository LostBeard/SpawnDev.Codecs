// Tests for Vp9Iadst8x8Reference. Oracle for the 8x8 iADST kernel
// that lands when Geordi's LocalMemory IR fix rc ships.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] MakeIadstPredictor8x8(byte value)
    {
        var dest = new byte[64];
        for (int i = 0; i < 64; i++) dest[i] = value;
        return dest;
    }

    [TestMethod]
    public void Vp9Iadst8x8_ZeroCoefficients_LeavesPredictorUnchanged()
    {
        var coeffs = new short[64];
        var dest = MakeIadstPredictor8x8(128);
        Vp9Iadst8x8Reference.IadstAdst8x8_64_Add(coeffs, dest, 8);
        for (int i = 0; i < 64; i++) Equal((byte)128, dest[i]);
    }

    [TestMethod]
    public void Vp9Iadst8x8_IsDeterministic_ForRandomInputs()
    {
        var rng = new Random(0xA8);
        var coeffs = new short[64];
        for (int trial = 0; trial < 15; trial++)
        {
            for (int i = 0; i < 64; i++) coeffs[i] = (short)rng.Next(-4096, 4096);
            var a = MakeIadstPredictor8x8(128);
            var b = MakeIadstPredictor8x8(128);
            Vp9Iadst8x8Reference.IadstAdst8x8_64_Add(coeffs, a, 8);
            Vp9Iadst8x8Reference.IadstAdst8x8_64_Add(coeffs, b, 8);
            True(a.AsSpan().SequenceEqual(b),
                $"non-deterministic output at trial {trial}");
        }
    }

    [TestMethod]
    public void Vp9Iadst8x8_SingleCoefficient_ProducesNonZeroResidual()
    {
        // iADST scatters energy non-uniformly across the block. Any single
        // non-zero coefficient should change at least one pixel.
        var coeffs = new short[64];
        coeffs[0] = 1024;
        var dest = MakeIadstPredictor8x8(100);
        Vp9Iadst8x8Reference.IadstAdst8x8_64_Add(coeffs, dest, 8);
        bool changed = false;
        for (int i = 0; i < 64; i++) if (dest[i] != 100) { changed = true; break; }
        True(changed, "single coefficient must produce non-zero residual somewhere");
    }

    [TestMethod]
    public void Vp9Iadst8x8_HighPositive_OutputsClippedPixels()
    {
        var coeffs = new short[64];
        for (int i = 0; i < 64; i++) coeffs[i] = 2048;
        var dest = MakeIadstPredictor8x8(250);
        Vp9Iadst8x8Reference.IadstAdst8x8_64_Add(coeffs, dest, 8);
        for (int i = 0; i < 64; i++)
            True(dest[i] <= 255, "clip violated upper bound");
    }

    [TestMethod]
    public void Vp9Iadst8x8_LargeNegatives_ClipToZeroSomewhere()
    {
        var coeffs = new short[64];
        for (int i = 0; i < 64; i++) coeffs[i] = -2048;
        var dest = MakeIadstPredictor8x8(10);
        Vp9Iadst8x8Reference.IadstAdst8x8_64_Add(coeffs, dest, 8);
        for (int i = 0; i < 64; i++)
            True(dest[i] <= 255, "clip violated upper bound");
        bool anyZero = false;
        for (int i = 0; i < 64; i++) if (dest[i] == 0) { anyZero = true; break; }
        True(anyZero, "at least one pixel must clip to 0 on all-negative input");
    }

    [TestMethod]
    public void Vp9Iadst8x8_StridedDest_OnlyTouches8x8Block()
    {
        var canvas = new byte[8 * 32];
        for (int i = 0; i < canvas.Length; i++) canvas[i] = 77;
        var coeffs = new short[64];
        coeffs[0] = 1024;
        Vp9Iadst8x8Reference.IadstAdst8x8_64_Add(coeffs, canvas, stride: 32);
        for (int row = 0; row < 8; row++)
            for (int col = 8; col < 32; col++)
                Equal((byte)77, canvas[row * 32 + col]);
    }

    [TestMethod]
    public void Vp9Iadst8x8_Throws_OnShortInput()
    {
        var dest = MakeIadstPredictor8x8(0);
        bool threw = false;
        try { Vp9Iadst8x8Reference.IadstAdst8x8_64_Add(new short[32], dest, 8); }
        catch (ArgumentException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void Vp9Iadst8x8_Throws_OnBadStride()
    {
        bool threw = false;
        try { Vp9Iadst8x8Reference.IadstAdst8x8_64_Add(new short[64], new byte[64], 4); }
        catch (ArgumentException) { threw = true; }
        True(threw);
    }
}
