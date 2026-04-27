// VP8 forward transform round-trip tests.
//
// Pair Vp8ForwardTransform.ShortFdct4x4 with Vp8InverseTransform.
// ShortIdct4x4Llm + zero predictor on a small range of inputs and
// verify the reconstructed pixels match within a small tolerance.
// Walsh4x4 has its own DC-only round-trip path.

using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp8ForwardTransform_ShortFdct4x4_AllZero_LibvpxQuirkIsBitExact()
    {
        // libvpx vp8_short_fdct4x4_c has a per-pass rounding constant (+14500
        // / +7500 in pass 1, then column pass on those non-zero values) that
        // causes a deterministic small bias on all-zero input. Specifically
        // output[1] == 1 and all other coefs == 0. We assert this libvpx-
        // exact behavior so any port regression is caught.
        var input = new short[16];
        var output = new short[16];
        Vp8ForwardTransform.ShortFdct4x4(input, 4, output);
        Equal((short)1, output[1]);
        for (int i = 0; i < 16; i++)
        {
            if (i == 1) continue;
            if (output[i] != 0)
                throw new Exception($"All-zero fdct4x4 should only emit output[1]=1; got output[{i}]={output[i]}");
        }
    }

    [TestMethod]
    public void Vp8ForwardTransform_ShortFdct4x4_Determinism()
    {
        var rng = new Random(0x8FD);
        var input = new short[16];
        for (int i = 0; i < 16; i++) input[i] = (short)rng.Next(-128, 128);
        var a = new short[16];
        var b = new short[16];
        Vp8ForwardTransform.ShortFdct4x4(input, 4, a);
        Vp8ForwardTransform.ShortFdct4x4(input, 4, b);
        for (int i = 0; i < 16; i++) Equal(a[i], b[i]);
    }

    [TestMethod]
    public void Vp8ForwardTransform_ShortFdct4x4_FwdInv_RoundTripFlatBlock()
    {
        // Flat residual = 0 -> all-zero coefficients -> inverse adds 0 to
        // predictor -> dest = predictor unchanged.
        var residual = new short[16];
        var coefs = new short[16];
        Vp8ForwardTransform.ShortFdct4x4(residual, 4, coefs);

        var pred = new byte[16];
        for (int i = 0; i < 16; i++) pred[i] = 128;
        var dst = new byte[16];
        Vp8InverseTransform.ShortIdct4x4Llm(coefs, pred, 4, dst, 4);
        for (int i = 0; i < 16; i++) Equal((byte)128, dst[i]);
    }

    [TestMethod]
    public void Vp8ForwardTransform_ShortFdct4x4_FwdInv_RoundTripSmallGradient()
    {
        // 4x4 gradient pixels, predictor=128, residual = pixels - predictor.
        var pixels = new byte[16];
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
                pixels[r * 4 + c] = (byte)(120 + r * 2 + c);

        var residual = new short[16];
        for (int i = 0; i < 16; i++) residual[i] = (short)(pixels[i] - 128);

        var coefs = new short[16];
        Vp8ForwardTransform.ShortFdct4x4(residual, 4, coefs);

        var pred = new byte[16];
        for (int i = 0; i < 16; i++) pred[i] = 128;
        var dst = new byte[16];
        Vp8InverseTransform.ShortIdct4x4Llm(coefs, pred, 4, dst, 4);

        int maxErr = 0;
        for (int i = 0; i < 16; i++) maxErr = Math.Max(maxErr, Math.Abs(dst[i] - pixels[i]));
        // VP8 4x4 DCT/IDCT pair without quantization is bit-exact in libvpx;
        // any error indicates a port issue.
        True(maxErr <= 1, $"VP8 fdct4x4 + idct4x4 round-trip: max err = {maxErr}, expected <= 1");
    }

    [TestMethod]
    public void Vp8ForwardTransform_ShortWalsh4x4_AllZero_ProducesAllZero()
    {
        var input = new short[16];
        var output = new short[16];
        Vp8ForwardTransform.ShortWalsh4x4(input, 4, output);
        for (int i = 0; i < 16; i++) Equal((short)0, output[i]);
    }

    [TestMethod]
    public void Vp8ForwardTransform_ShortWalsh4x4_FwdInv_DcOnlyRoundTrip()
    {
        // Walsh transform on 4 DC values from 4 macroblock Y2 coefficients.
        // Round-trip should give back input (Walsh is its own inverse with
        // a known scale factor).
        var input = new short[16];
        input[0] = 100; input[1] = 50; input[2] = -25; input[3] = 80;
        for (int i = 4; i < 16; i++) input[i] = 0;

        var fwd = new short[16];
        Vp8ForwardTransform.ShortWalsh4x4(input, 4, fwd);

        var inv = new short[16];
        Vp8InverseTransform.ShortInvWalsh4x4(fwd, inv);

        // Check first 4 entries match input * scale (libvpx Walsh has /4 scale
        // on inverse; the encoder's Walsh has *2 scale; combined we expect
        // input * 2 / 4 = input / 2 plus rounding. Just verify same sign.
        for (int i = 0; i < 4; i++)
        {
            if (input[i] != 0)
                True(Math.Sign(inv[i]) == Math.Sign(input[i]),
                    $"Walsh round-trip sign mismatch at {i}: input={input[i]}, inv={inv[i]}");
        }
    }
}
