// Tests for Vp9Idct16x16Reference. This CPU reference is the ORACLE
// that the upcoming cross-backend 16x16 ILGPU kernel must match
// byte-for-byte on CPU, CUDA, OpenCL, WebGPU, WebGL, and Wasm. The
// reference itself is validated here through zero / clipping /
// determinism / strided-dest invariants.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] MakePredictor16x16(byte value)
    {
        var dest = new byte[256];
        for (int i = 0; i < 256; i++) dest[i] = value;
        return dest;
    }

    [TestMethod]
    public void Vp9Idct16x16_ZeroCoefficients_LeavesPredictorUnchanged()
    {
        var coeffs = new short[256];
        var dest = MakePredictor16x16(128);
        Vp9Idct16x16Reference.Idct16x16_256_Add(coeffs, dest, 16);
        for (int i = 0; i < 256; i++) Equal((byte)128, dest[i]);
    }

    [TestMethod]
    public void Vp9Idct16x16_Clips_HighResidualToUpper255()
    {
        var coeffs = new short[256];
        coeffs[0] = 32000; // saturate DC; 16-point transform concentrates energy
        var dest = MakePredictor16x16(200);
        Vp9Idct16x16Reference.Idct16x16_256_Add(coeffs, dest, 16);
        for (int i = 0; i < 256; i++) Equal((byte)255, dest[i]);
    }

    [TestMethod]
    public void Vp9Idct16x16_Clips_NegativeResidualToZero()
    {
        var coeffs = new short[256];
        coeffs[0] = -32000;
        var dest = MakePredictor16x16(100);
        Vp9Idct16x16Reference.Idct16x16_256_Add(coeffs, dest, 16);
        for (int i = 0; i < 256; i++) Equal((byte)0, dest[i]);
    }

    [TestMethod]
    public void Vp9Idct16x16_StridedDest_OnlyTouches16x16Block()
    {
        // 16-row canvas 32 cols wide; 16x16 at top-left.
        var canvas = new byte[16 * 32];
        for (int i = 0; i < canvas.Length; i++) canvas[i] = 77;
        var coeffs = new short[256];
        coeffs[0] = 4096;
        Vp9Idct16x16Reference.Idct16x16_256_Add(coeffs, canvas, stride: 32);
        for (int row = 0; row < 16; row++)
            for (int col = 16; col < 32; col++)
                Equal((byte)77, canvas[row * 32 + col]);
    }

    [TestMethod]
    public void Vp9Idct16x16_IsDeterministic_ForRandomInputs()
    {
        var rng = new Random(0x1616);
        var coeffs = new short[256];
        for (int trial = 0; trial < 20; trial++)
        {
            for (int i = 0; i < 256; i++) coeffs[i] = (short)rng.Next(-4096, 4096);
            var a = MakePredictor16x16(128);
            var b = MakePredictor16x16(128);
            Vp9Idct16x16Reference.Idct16x16_256_Add(coeffs, a, 16);
            Vp9Idct16x16Reference.Idct16x16_256_Add(coeffs, b, 16);
            True(a.AsSpan().SequenceEqual(b), $"non-deterministic at trial {trial}");
        }
    }

    [TestMethod]
    public void Vp9Idct16x16_SingleDcCoefficient_ProducesNonZeroResidual()
    {
        var coeffs = new short[256];
        coeffs[0] = 1024;
        var dest = MakePredictor16x16(100);
        Vp9Idct16x16Reference.Idct16x16_256_Add(coeffs, dest, 16);
        bool changed = false;
        for (int i = 0; i < 256; i++) if (dest[i] != 100) { changed = true; break; }
        True(changed, "DC coefficient must produce a non-zero residual somewhere");
    }

    [TestMethod]
    public void Vp9Idct16x16_Throws_OnShortInput()
    {
        var dest = MakePredictor16x16(0);
        bool threw = false;
        try { Vp9Idct16x16Reference.Idct16x16_256_Add(new short[128], dest, 16); }
        catch (ArgumentException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void Vp9Idct16x16_Throws_OnBadStride()
    {
        bool threw = false;
        try { Vp9Idct16x16Reference.Idct16x16_256_Add(new short[256], new byte[256], 8); }
        catch (ArgumentException) { threw = true; }
        True(threw);
    }
}
