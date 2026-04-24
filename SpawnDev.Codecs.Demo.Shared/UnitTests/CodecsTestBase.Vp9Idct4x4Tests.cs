// Tests for Vp9Idct4x4Reference. The reference must match libvpx
// vp9_idct4x4_16_add bit-exactly per the VP9 normative bitstream.
// Test strategy:
//   - Zero coefficients produce zero residual (predictor unchanged).
//   - DC-only coefficient produces a mathematically derivable uniform
//     residual (hand-computed against the spec's Q14 constants, pinned).
//   - Every output pixel is clipped to [0, 255].
//   - Output is deterministic for a given input (round-trip sanity).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>Make a 4x4 predictor buffer filled with one value, stride 4.</summary>
    private static byte[] MakePredictor(byte value)
    {
        var dest = new byte[16];
        for (int i = 0; i < 16; i++) dest[i] = value;
        return dest;
    }

    [TestMethod]
    public void Vp9Idct4x4_ZeroCoefficients_LeavesPredictorUnchanged()
    {
        var coeffs = new short[16]; // all zero
        var dest = MakePredictor(128);
        Vp9Idct4x4Reference.Idct4x4_16_Add(coeffs, dest, 4);
        for (int i = 0; i < 16; i++)
            Equal((byte)128, dest[i]);
    }

    [TestMethod]
    public void Vp9Idct4x4_DcOnlyCoefficient_AppliesUniformResidual()
    {
        // Input: DC = 1024 at [0,0], all other coeffs zero.
        // Math (cospi_16_64 = 11585, Q14 constants):
        //   row 0 iDCT of [1024,0,0,0]:
        //     t1 = (1024+0)*11585 = 11863040; step0 = (11863040+8192)>>14 = 724
        //     step1 = step0 = 724; step2 = step3 = 0
        //     output = [724, 724, 724, 724]
        //   rows 1-3 iDCT of [0,0,0,0] = [0,0,0,0]
        //   col pass on [724,0,0,0] column -> [512, 512, 512, 512]
        //     (same Q14 butterfly math with 724 instead of 1024)
        //     t1 = 724 * 11585 = 8387540; step0 = (8387540+8192)>>14 = 512
        //   residual per pixel = ROUND_POWER_OF_TWO(512, 4) = (512+8)>>4 = 32
        //   All 16 pixels receive +32.
        var coeffs = new short[16];
        coeffs[0] = 1024;
        var dest = MakePredictor(100);
        Vp9Idct4x4Reference.Idct4x4_16_Add(coeffs, dest, 4);
        for (int i = 0; i < 16; i++)
            Equal((byte)132, dest[i]); // 100 + 32
    }

    [TestMethod]
    public void Vp9Idct4x4_Clips_HighResidualToUpper255()
    {
        // Push a DC coefficient high enough that predictor + residual > 255
        // and verify the result saturates at 255.
        var coeffs = new short[16];
        coeffs[0] = 8192; // residual ~ 256 at every cell
        var dest = MakePredictor(200);
        Vp9Idct4x4Reference.Idct4x4_16_Add(coeffs, dest, 4);
        for (int i = 0; i < 16; i++)
            Equal((byte)255, dest[i]);
    }

    [TestMethod]
    public void Vp9Idct4x4_Clips_NegativeResidualToZero()
    {
        // Negative DC -> residual pushes below zero, must clamp to 0.
        var coeffs = new short[16];
        coeffs[0] = -8192; // residual ~ -256 at every cell
        var dest = MakePredictor(100);
        Vp9Idct4x4Reference.Idct4x4_16_Add(coeffs, dest, 4);
        for (int i = 0; i < 16; i++)
            Equal((byte)0, dest[i]);
    }

    [TestMethod]
    public void Vp9Idct4x4_StridedDest_OnlyTouches4x4Block()
    {
        // Put the 4x4 block at the top-left of a 16-byte-wide canvas and
        // verify pixels outside the 4x4 are untouched.
        var canvas = new byte[4 * 16]; // 4 rows, 16 cols
        for (int i = 0; i < canvas.Length; i++) canvas[i] = 77;
        var coeffs = new short[16];
        coeffs[0] = 1024;
        Vp9Idct4x4Reference.Idct4x4_16_Add(coeffs, canvas, stride: 16);
        // In-block (cols 0-3): was 77 + 32 = 109.
        for (int row = 0; row < 4; row++)
            for (int col = 0; col < 4; col++)
                Equal((byte)109, canvas[row * 16 + col]);
        // Out-of-block (cols 4-15): untouched.
        for (int row = 0; row < 4; row++)
            for (int col = 4; col < 16; col++)
                Equal((byte)77, canvas[row * 16 + col]);
    }

    [TestMethod]
    public void Vp9Idct4x4_IsDeterministic_ForRandomInputs()
    {
        // Stress: 50 random coeff buffers must produce identical outputs
        // across repeated calls. Catches any hidden state.
        var rng = new Random(0xC0DEC);
        var coeffs = new short[16];
        for (int trial = 0; trial < 50; trial++)
        {
            for (int i = 0; i < 16; i++) coeffs[i] = (short)rng.Next(-2048, 2048);
            var a = MakePredictor(128);
            var b = MakePredictor(128);
            Vp9Idct4x4Reference.Idct4x4_16_Add(coeffs, a, 4);
            Vp9Idct4x4Reference.Idct4x4_16_Add(coeffs, b, 4);
            True(a.AsSpan().SequenceEqual(b),
                $"non-deterministic output at trial {trial}");
            // Also: every output byte must be a valid pixel value (the type
            // already guarantees this, but this documents the intent).
            foreach (var v in a) True(v >= 0 && v <= 255, "pixel out of range");
        }
    }

    [TestMethod]
    public void Vp9Idct4x4_Throws_OnShortInput()
    {
        var dest = MakePredictor(0);
        bool threw = false;
        try { Vp9Idct4x4Reference.Idct4x4_16_Add(new short[8], dest, 4); }
        catch (ArgumentException) { threw = true; }
        True(threw, "short input must throw");
    }

    [TestMethod]
    public void Vp9Idct4x4_Throws_OnBadStride()
    {
        var coeffs = new short[16];
        var dest = new byte[16];
        bool threw = false;
        try { Vp9Idct4x4Reference.Idct4x4_16_Add(coeffs, dest, 2); }
        catch (ArgumentException) { threw = true; }
        True(threw, "stride < 4 must throw");
    }
}
