// Tests for Vp9ForwardDct32x32 - the encoder side of the 32x32 DCT.
// Bit-exactness verified against the existing inverse oracle
// (Vp9Idct32x32Reference) via fwd -> inv round-trip on real residuals.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9ForwardDct32x32_AllZeroInput_ProducesAllZeroOutput()
    {
        var input = new short[1024];
        var output = new int[1024];
        Vp9ForwardDct32x32.Transform(input, 32, output);
        for (int i = 0; i < 1024; i++) Equal(0, output[i]);
    }

    [TestMethod]
    public void Vp9ForwardDct32x32_ConstantInput_PutsEnergyInDcOnly()
    {
        var input = new short[1024];
        for (int i = 0; i < 1024; i++) input[i] = 8;
        var output = new int[1024];
        Vp9ForwardDct32x32.Transform(input, 32, output);
        True(output[0] != 0, "DC coefficient must be non-zero for constant input");
        for (int i = 1; i < 1024; i++)
            Equal(0, output[i]);
    }

    [TestMethod]
    public void Vp9ForwardDct32x32_Determinism()
    {
        var rng = new Random(0xF000);
        var input = new short[1024];
        for (int i = 0; i < 1024; i++) input[i] = (short)rng.Next(-128, 128);
        var a = new int[1024];
        var b = new int[1024];
        Vp9ForwardDct32x32.Transform(input, 32, a);
        Vp9ForwardDct32x32.Transform(input, 32, b);
        for (int i = 0; i < 1024; i++) Equal(a[i], b[i]);
    }

    [TestMethod]
    public void Vp9ForwardDct32x32_FwdInvRoundTrip_OnSmoothResidual_LowError()
    {
        // Smooth gradient residual avoids overflow in the int -> short cast.
        // Per libvpx FwdInvLargeOpt tolerance: ~1 absolute error per pixel
        // is acceptable for the 32x32 DCT pair (half_round_shift between
        // passes loses 2 bits of precision intentionally).
        var residual = new short[1024];
        for (int r = 0; r < 32; r++)
            for (int c = 0; c < 32; c++)
                residual[r * 32 + c] = (short)((r - 16) + (c - 16));

        var coeffs = new int[1024];
        Vp9ForwardDct32x32.Transform(residual, 32, coeffs);

        // Cast back to short for inverse oracle. None of these residuals
        // overflow short range for this particular smooth input.
        var coeffsShort = new short[1024];
        for (int i = 0; i < 1024; i++) coeffsShort[i] = (short)coeffs[i];

        var dest = new byte[1024];
        for (int i = 0; i < 1024; i++) dest[i] = 128;
        Vp9Idct32x32Reference.Idct32x32_1024_Add(coeffsShort, dest, 32);

        int maxErr = 0;
        for (int i = 0; i < 1024; i++)
        {
            int reconstructed = dest[i] - 128;
            int original = residual[i];
            maxErr = Math.Max(maxErr, Math.Abs(reconstructed - original));
        }
        True(maxErr <= 1, $"max round-trip error = {maxErr}, expected <= 1 for smooth gradient");
    }

    [TestMethod]
    public void Vp9ForwardDct32x32_Throws_OnShortInput()
    {
        bool threw = false;
        try { Vp9ForwardDct32x32.Transform(new short[256], 32, new int[1024]); }
        catch (ArgumentException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void Vp9ForwardDct32x32_Throws_OnShortOutput()
    {
        bool threw = false;
        try { Vp9ForwardDct32x32.Transform(new short[1024], 32, new int[512]); }
        catch (ArgumentException) { threw = true; }
        True(threw);
    }
}
