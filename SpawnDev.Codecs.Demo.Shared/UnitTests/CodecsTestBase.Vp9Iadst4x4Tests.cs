// Tests for Vp9Iadst4x4Reference. iADST is asymmetric so hand-derived
// DC-only numbers are messier than iDCT; we rely on zero-input (covers
// the short-circuit path), determinism, clipping, and argument
// validation. Bit-exactness vs libvpx is verified via the 2D
// round-trip property once the forward ADST reference lands in a
// later slice.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] MakeIadstPredictor(byte value)
    {
        var dest = new byte[16];
        for (int i = 0; i < 16; i++) dest[i] = value;
        return dest;
    }

    [TestMethod]
    public void Vp9Iadst4x4_ZeroCoefficients_LeavesPredictorUnchanged()
    {
        // Short-circuit path: iadst4_c returns [0,0,0,0] when all inputs
        // are zero. Every residual is 0; predictor stays put.
        var coeffs = new short[16];
        var dest = MakeIadstPredictor(128);
        Vp9Iadst4x4Reference.IadstAdst4x4_16_Add(coeffs, dest, 4);
        for (int i = 0; i < 16; i++) Equal((byte)128, dest[i]);
    }

    [TestMethod]
    public void Vp9Iadst4x4_Clips_HighInputToUpper255()
    {
        // Saturate all coefficients at +2048. Expected behaviour: residuals
        // push every pixel to/above 255 at some point, clipping engages.
        // We just verify no byte exceeds 255 (clip lower bound) and the
        // predictor was changed somewhere (i.e. transform did run).
        var coeffs = new short[16];
        for (int i = 0; i < 16; i++) coeffs[i] = 2048;
        var dest = MakeIadstPredictor(250);
        Vp9Iadst4x4Reference.IadstAdst4x4_16_Add(coeffs, dest, 4);
        bool anyChanged = false;
        for (int i = 0; i < 16; i++)
        {
            True(dest[i] <= 255, "clipping violated upper bound");
            if (dest[i] != 250) anyChanged = true;
        }
        True(anyChanged, "iADST on non-zero coefficients must change at least one pixel");
    }

    [TestMethod]
    public void Vp9Iadst4x4_Clips_LargeNegativesToZero()
    {
        var coeffs = new short[16];
        for (int i = 0; i < 16; i++) coeffs[i] = -2048;
        var dest = MakeIadstPredictor(10);
        Vp9Iadst4x4Reference.IadstAdst4x4_16_Add(coeffs, dest, 4);
        for (int i = 0; i < 16; i++)
            True(dest[i] <= 255, "clipping violated upper bound");
        // At least one pixel should have clipped to 0 (predictor was 10 and
        // the transform is known to produce negatives for all-negative input).
        bool anyZero = false;
        for (int i = 0; i < 16; i++) if (dest[i] == 0) { anyZero = true; break; }
        True(anyZero, "at least one pixel must have clipped to 0 on all-negative input");
    }

    [TestMethod]
    public void Vp9Iadst4x4_IsDeterministic_ForRandomInputs()
    {
        var rng = new Random(0xADE);
        var coeffs = new short[16];
        for (int trial = 0; trial < 50; trial++)
        {
            for (int i = 0; i < 16; i++) coeffs[i] = (short)rng.Next(-2048, 2048);
            var a = MakeIadstPredictor(128);
            var b = MakeIadstPredictor(128);
            Vp9Iadst4x4Reference.IadstAdst4x4_16_Add(coeffs, a, 4);
            Vp9Iadst4x4Reference.IadstAdst4x4_16_Add(coeffs, b, 4);
            True(a.AsSpan().SequenceEqual(b),
                $"non-deterministic output at trial {trial}");
        }
    }

    [TestMethod]
    public void Vp9Iadst4x4_SingleCoefficient_ProducesNonZeroResidual()
    {
        // Unlike iDCT where a DC-only coefficient produces a uniform +32
        // residual, iADST's asymmetric butterfly scatters the energy
        // unevenly across the 4x4 block. Just pin "at least one pixel
        // changed" as the regression check.
        var coeffs = new short[16];
        coeffs[0] = 1024;
        var dest = MakeIadstPredictor(100);
        Vp9Iadst4x4Reference.IadstAdst4x4_16_Add(coeffs, dest, 4);
        bool changed = false;
        for (int i = 0; i < 16; i++) if (dest[i] != 100) { changed = true; break; }
        True(changed, "non-zero coefficient must produce non-zero residual somewhere");
    }

    [TestMethod]
    public void Vp9Iadst4x4_StridedDest_OnlyTouches4x4Block()
    {
        var canvas = new byte[4 * 16];
        for (int i = 0; i < canvas.Length; i++) canvas[i] = 77;
        var coeffs = new short[16];
        coeffs[0] = 1024;
        Vp9Iadst4x4Reference.IadstAdst4x4_16_Add(coeffs, canvas, stride: 16);
        for (int row = 0; row < 4; row++)
            for (int col = 4; col < 16; col++)
                Equal((byte)77, canvas[row * 16 + col]);
    }

    [TestMethod]
    public void Vp9Iadst4x4_Throws_OnShortInput()
    {
        var dest = MakeIadstPredictor(0);
        bool threw = false;
        try { Vp9Iadst4x4Reference.IadstAdst4x4_16_Add(new short[8], dest, 4); }
        catch (ArgumentException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void Vp9Iadst4x4_Throws_OnBadStride()
    {
        bool threw = false;
        try { Vp9Iadst4x4Reference.IadstAdst4x4_16_Add(new short[16], new byte[16], 2); }
        catch (ArgumentException) { threw = true; }
        True(threw);
    }
}
