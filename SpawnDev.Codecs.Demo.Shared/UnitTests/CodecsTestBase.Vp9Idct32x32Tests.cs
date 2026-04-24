// Tests for Vp9Idct32x32Reference. Oracle for the 32x32 ILGPU kernel
// that lands across all backends once Geordi's LocalMemory IR fix
// rc ships.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] MakePredictor32x32(byte value)
    {
        var dest = new byte[1024];
        for (int i = 0; i < 1024; i++) dest[i] = value;
        return dest;
    }

    [TestMethod]
    public void Vp9Idct32x32_ZeroCoefficients_LeavesPredictorUnchanged()
    {
        var coeffs = new short[1024];
        var dest = MakePredictor32x32(128);
        Vp9Idct32x32Reference.Idct32x32_1024_Add(coeffs, dest, 32);
        for (int i = 0; i < 1024; i++) Equal((byte)128, dest[i]);
    }

    [TestMethod]
    public void Vp9Idct32x32_Clips_HighResidualToUpper255()
    {
        var coeffs = new short[1024];
        coeffs[0] = 32000;
        var dest = MakePredictor32x32(200);
        Vp9Idct32x32Reference.Idct32x32_1024_Add(coeffs, dest, 32);
        for (int i = 0; i < 1024; i++) Equal((byte)255, dest[i]);
    }

    [TestMethod]
    public void Vp9Idct32x32_Clips_NegativeResidualToZero()
    {
        var coeffs = new short[1024];
        coeffs[0] = -32000;
        var dest = MakePredictor32x32(100);
        Vp9Idct32x32Reference.Idct32x32_1024_Add(coeffs, dest, 32);
        for (int i = 0; i < 1024; i++) Equal((byte)0, dest[i]);
    }

    [TestMethod]
    public void Vp9Idct32x32_StridedDest_OnlyTouches32x32Block()
    {
        // 32-row canvas 64 cols wide.
        var canvas = new byte[32 * 64];
        for (int i = 0; i < canvas.Length; i++) canvas[i] = 77;
        var coeffs = new short[1024];
        coeffs[0] = 4096;
        Vp9Idct32x32Reference.Idct32x32_1024_Add(coeffs, canvas, stride: 64);
        for (int row = 0; row < 32; row++)
            for (int col = 32; col < 64; col++)
                Equal((byte)77, canvas[row * 64 + col]);
    }

    [TestMethod]
    public void Vp9Idct32x32_IsDeterministic_ForRandomInputs()
    {
        var rng = new Random(0x3232);
        var coeffs = new short[1024];
        for (int trial = 0; trial < 8; trial++)
        {
            for (int i = 0; i < 1024; i++) coeffs[i] = (short)rng.Next(-4096, 4096);
            var a = MakePredictor32x32(128);
            var b = MakePredictor32x32(128);
            Vp9Idct32x32Reference.Idct32x32_1024_Add(coeffs, a, 32);
            Vp9Idct32x32Reference.Idct32x32_1024_Add(coeffs, b, 32);
            True(a.AsSpan().SequenceEqual(b), $"non-deterministic at trial {trial}");
        }
    }

    [TestMethod]
    public void Vp9Idct32x32_SingleDcCoefficient_ProducesNonZeroResidual()
    {
        var coeffs = new short[1024];
        coeffs[0] = 1024;
        var dest = MakePredictor32x32(100);
        Vp9Idct32x32Reference.Idct32x32_1024_Add(coeffs, dest, 32);
        bool changed = false;
        for (int i = 0; i < 1024; i++) if (dest[i] != 100) { changed = true; break; }
        True(changed, "DC coefficient must produce non-zero residual somewhere");
    }

    [TestMethod]
    public void Vp9Idct32x32_Throws_OnShortInput()
    {
        var dest = MakePredictor32x32(0);
        bool threw = false;
        try { Vp9Idct32x32Reference.Idct32x32_1024_Add(new short[512], dest, 32); }
        catch (ArgumentException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void Vp9Idct32x32_Throws_OnBadStride()
    {
        bool threw = false;
        try { Vp9Idct32x32Reference.Idct32x32_1024_Add(new short[1024], new byte[1024], 16); }
        catch (ArgumentException) { threw = true; }
        True(threw);
    }
}
