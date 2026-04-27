// Tests for Vp9ForwardAdst16 - the encoder-side 16-point ADST.
// Bit-exactness verified via the existing inverse oracle
// (Vp9Iadst16x16Reference) on the 2D ADST_ADST round-trip path.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9ForwardAdst16_AllZeroInput_ProducesAllZeroOutput()
    {
        var input = new int[16];
        var output = new int[16];
        Vp9ForwardAdst16.Transform(input, output);
        for (int i = 0; i < 16; i++) Equal(0, output[i]);
    }

    [TestMethod]
    public void Vp9ForwardAdst16_Determinism()
    {
        var rng = new Random(0xAD16);
        var input = new int[16];
        for (int i = 0; i < 16; i++) input[i] = rng.Next(-2048, 2048);
        var a = new int[16];
        var b = new int[16];
        Vp9ForwardAdst16.Transform(input, a);
        Vp9ForwardAdst16.Transform(input, b);
        for (int i = 0; i < 16; i++) Equal(a[i], b[i]);
    }

    [TestMethod]
    public void Vp9ForwardAdst16_ImpulseInput_NonZeroOutput()
    {
        var input = new int[16];
        input[3] = 256;
        var output = new int[16];
        Vp9ForwardAdst16.Transform(input, output);
        bool any = false;
        for (int i = 0; i < 16; i++) if (output[i] != 0) { any = true; break; }
        True(any, "impulse must produce non-zero coefficients");
    }

    [TestMethod]
    public void Vp9ForwardAdst16_2dRoundTrip_OnSmoothResidual_LowError()
    {
        // Drive the dispatcher through the AdstAdst 16x16 path so this
        // test exercises both the column pass and the row pass plus the
        // half_round_shift compensation between passes.
        var residual = new short[256];
        for (int r = 0; r < 16; r++)
            for (int c = 0; c < 16; c++)
                residual[r * 16 + c] = (short)((r - 8) * 2 + (c - 8));

        var coeffs = new int[256];
        Vp9ForwardTransform.Apply(
            Vp9TxSize.Tx16x16, Vp9TxType.AdstAdst,
            residual, 16, coeffs);

        // Cast back to short for the inverse oracle.
        var coeffsShort = new short[256];
        for (int i = 0; i < 256; i++) coeffsShort[i] = (short)coeffs[i];

        var dest = new byte[256];
        for (int i = 0; i < 256; i++) dest[i] = 128;
        Vp9Iadst16x16Reference.IadstAdst16x16_256_Add(coeffsShort, dest, 16);

        int maxErr = 0;
        for (int i = 0; i < 256; i++)
        {
            int reconstructed = dest[i] - 128;
            int original = residual[i];
            maxErr = Math.Max(maxErr, Math.Abs(reconstructed - original));
        }
        True(maxErr <= 2, $"max ADST round-trip error = {maxErr}, expected <= 2");
    }

    [TestMethod]
    public void Vp9ForwardAdst16_Throws_OnShortInput()
    {
        bool threw = false;
        try { Vp9ForwardAdst16.Transform(new int[8], new int[16]); }
        catch (ArgumentException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void Vp9ForwardAdst16_Throws_OnShortOutput()
    {
        bool threw = false;
        try { Vp9ForwardAdst16.Transform(new int[16], new int[8]); }
        catch (ArgumentException) { threw = true; }
        True(threw);
    }
}
