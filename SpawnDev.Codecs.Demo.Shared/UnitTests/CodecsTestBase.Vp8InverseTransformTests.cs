// Tests for Vp8InverseTransform - VP8 4x4 inverse transforms (IDCT,
// DC-only IDCT, inverse Walsh-Hadamard). RFC 6386 sec 14.

using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp8InverseTransform_DcOnlyIdctAdd_ZeroDc_ProducesZeroDeltaPlusPred()
    {
        // input_dc=0 -> a1 = (0+4)>>3 = 0. Output is just pred.
        Span<byte> pred = stackalloc byte[16];
        Span<byte> dst = stackalloc byte[16];
        for (int i = 0; i < 16; i++) pred[i] = (byte)(50 + i);

        Vp8InverseTransform.DcOnlyIdctAdd(0, pred, 4, dst, 4);

        for (int i = 0; i < 16; i++)
            Equal(pred[i], dst[i]);
    }

    [TestMethod]
    public void Vp8InverseTransform_DcOnlyIdctAdd_PositiveDc_AddsConstant()
    {
        // input_dc=8 -> a1 = (8+4)>>3 = 1. Output is pred + 1.
        Span<byte> pred = stackalloc byte[16];
        Span<byte> dst = stackalloc byte[16];
        for (int i = 0; i < 16; i++) pred[i] = 100;

        Vp8InverseTransform.DcOnlyIdctAdd(8, pred, 4, dst, 4);

        for (int i = 0; i < 16; i++)
            Equal((byte)101, dst[i]);
    }

    [TestMethod]
    public void Vp8InverseTransform_DcOnlyIdctAdd_NegativeDc_ClampsAtZero()
    {
        // input_dc=-808 -> a1 = (-808+4)>>3 = -100. pred=50 -> a = -50, clamp to 0.
        Span<byte> pred = stackalloc byte[16];
        Span<byte> dst = stackalloc byte[16];
        for (int i = 0; i < 16; i++) pred[i] = 50;

        Vp8InverseTransform.DcOnlyIdctAdd(-808, pred, 4, dst, 4);

        for (int i = 0; i < 16; i++)
            Equal((byte)0, dst[i]);
    }

    [TestMethod]
    public void Vp8InverseTransform_DcOnlyIdctAdd_LargePositiveDc_ClampsAt255()
    {
        // input_dc=2040 -> a1 = (2040+4)>>3 = 255. pred=200 -> a = 455, clamp to 255.
        Span<byte> pred = stackalloc byte[16];
        Span<byte> dst = stackalloc byte[16];
        for (int i = 0; i < 16; i++) pred[i] = 200;

        Vp8InverseTransform.DcOnlyIdctAdd(2040, pred, 4, dst, 4);

        for (int i = 0; i < 16; i++)
            Equal((byte)255, dst[i]);
    }

    [TestMethod]
    public void Vp8InverseTransform_ShortIdct4x4Llm_AllZeroCoeffs_ProducesZeroDeltaPlusPred()
    {
        // All-zero input -> output = pred (no residual added, all clamped passthrough).
        Span<short> input = stackalloc short[16];
        Span<byte> pred = stackalloc byte[16];
        Span<byte> dst = stackalloc byte[16];
        for (int i = 0; i < 16; i++) pred[i] = (byte)(40 + 5 * i);

        Vp8InverseTransform.ShortIdct4x4Llm(input, pred, 4, dst, 4);

        for (int i = 0; i < 16; i++)
            Equal(pred[i], dst[i]);
    }

    [TestMethod]
    public void Vp8InverseTransform_ShortInvWalsh4x4_KnownInput_KnownOutput()
    {
        // Walsh-Hadamard is an orthogonal transform. WalshFwd then WalshInv
        // applied to integers yields the original scaled by 16. We test the
        // identity case here: input is the all-equal-DC pattern that
        // forward-WHT would produce from a [N, 0, 0...0] DC input.
        // For input = [N, 0, 0, ..., 0] in DC position only, the FORWARD
        // Walsh produces all-N (since WHT[0] sums all entries with +1
        // signs). The INVERSE applied to all-N should recover N (scaled).
        Span<short> input = stackalloc short[16];
        for (int i = 0; i < 16; i++) input[i] = 16; // 16 = uniform DC

        Span<short> output = stackalloc short[16];
        Vp8InverseTransform.ShortInvWalsh4x4(input, output);

        // After inv-Walsh of all-16 input, only output[0] should be non-zero
        // (the "DC of the DC plane"). Specifically: [0] = (sum*+sum* + 3) >> 3
        // For a 4x4 of all 16, the column-pass produces a row of all-32 in
        // the first row and all-0 elsewhere. The row-pass on row 0 gives
        // [0]=64, others 0; with +3>>3 normalize gives [0] = (64+3)>>3 = 8.
        Equal((short)8, output[0]);
        for (int i = 1; i < 16; i++)
            Equal((short)0, output[i]);
    }
}
