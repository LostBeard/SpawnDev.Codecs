// Tests for Vp9Reconstruct (slice 171). Verifies clipped add of the
// iDCT residual into a predicted block.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Category = "CpuOnly")]
    public void Vp9Reconstruct_4x4_AddsResidualToPredicted()
    {
        CpuOnlyTest();
        // dst = 100; residual = +20 -> dst = 120 across the block.
        var dst = new byte[16];
        for (int i = 0; i < 16; i++) dst[i] = 100;
        var residual = new short[16];
        for (int i = 0; i < 16; i++) residual[i] = 20;

        Vp9Reconstruct.AddResidual(dst, residual, n: 4, stride: 4);

        for (int i = 0; i < 16; i++) Equal((byte)120, dst[i]);
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9Reconstruct_4x4_ClipsBelowZeroToZero()
    {
        CpuOnlyTest();
        var dst = new byte[16];
        for (int i = 0; i < 16; i++) dst[i] = 50;
        var residual = new short[16];
        for (int i = 0; i < 16; i++) residual[i] = -100;  // 50 + (-100) = -50 -> 0

        Vp9Reconstruct.AddResidual(dst, residual, n: 4, stride: 4);

        for (int i = 0; i < 16; i++) Equal((byte)0, dst[i]);
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9Reconstruct_4x4_ClipsAbove255To255()
    {
        CpuOnlyTest();
        var dst = new byte[16];
        for (int i = 0; i < 16; i++) dst[i] = 200;
        var residual = new short[16];
        for (int i = 0; i < 16; i++) residual[i] = 100;  // 200 + 100 = 300 -> 255

        Vp9Reconstruct.AddResidual(dst, residual, n: 4, stride: 4);

        for (int i = 0; i < 16; i++) Equal((byte)255, dst[i]);
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9Reconstruct_4x4_PerCellMath()
    {
        CpuOnlyTest();
        // dst[i] = 10*i mod 256, residual[i] = -3*i; expected = clip(10*i - 3*i) = clip(7*i).
        var dst = new byte[16];
        var residual = new short[16];
        for (int i = 0; i < 16; i++)
        {
            dst[i] = (byte)((10 * i) & 0xFF);
            residual[i] = (short)(-3 * i);
        }

        Vp9Reconstruct.AddResidual(dst, residual, n: 4, stride: 4);

        for (int i = 0; i < 16; i++)
        {
            int sum = ((10 * i) & 0xFF) + (-3 * i);
            byte expected = sum < 0 ? (byte)0 : sum > 255 ? (byte)255 : (byte)sum;
            Equal(expected, dst[i]);
        }
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9Reconstruct_8x8_RespectsStridedDst()
    {
        CpuOnlyTest();
        const int stride = 16;
        var canvas = new byte[stride * 8];
        for (int i = 0; i < canvas.Length; i++) canvas[i] = 222;
        // Pre-set the 8x8 block with predicted=100.
        for (int row = 0; row < 8; row++)
            for (int col = 0; col < 8; col++)
                canvas[row * stride + col] = 100;

        var residual = new short[64];
        for (int i = 0; i < 64; i++) residual[i] = 30;  // 100 + 30 = 130

        Vp9Reconstruct.AddResidual(canvas, residual, n: 8, stride);

        for (int row = 0; row < 8; row++)
        {
            for (int col = 0; col < 8; col++)
                Equal((byte)130, canvas[row * stride + col]);
            for (int col = 8; col < stride; col++)
                Equal((byte)222, canvas[row * stride + col]);
        }
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9Reconstruct_16x16_LargestSizeRoundtrip()
    {
        CpuOnlyTest();
        var dst = new byte[256];
        var residual = new short[256];
        for (int i = 0; i < 256; i++)
        {
            dst[i] = (byte)(i & 0xFF);
            residual[i] = (short)((i % 11) - 5);  // -5..+5 spread
        }

        Vp9Reconstruct.AddResidual(dst, residual, n: 16, stride: 16);

        for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++)
        {
            int idx = r * 16 + c;
            int sum = (idx & 0xFF) + ((idx % 11) - 5);
            byte expected = sum < 0 ? (byte)0 : sum > 255 ? (byte)255 : (byte)sum;
            Equal(expected, dst[idx]);
        }
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9Reconstruct_32x32_AllSizes()
    {
        CpuOnlyTest();
        var dst = new byte[32 * 32];
        for (int i = 0; i < dst.Length; i++) dst[i] = 128;
        var residual = new short[32 * 32];
        for (int i = 0; i < residual.Length; i++) residual[i] = (short)((i % 7) - 3);

        Vp9Reconstruct.AddResidual(dst, residual, n: 32, stride: 32);

        for (int i = 0; i < dst.Length; i++)
        {
            int sum = 128 + ((i % 7) - 3);
            byte expected = sum < 0 ? (byte)0 : sum > 255 ? (byte)255 : (byte)sum;
            Equal(expected, dst[i]);
        }
    }

    [TestMethod(Category = "CpuOnly")]
    public void Vp9Reconstruct_RejectsInvalidArgs()
    {
        CpuOnlyTest();
        Throws<ArgumentOutOfRangeException>(() =>
            Vp9Reconstruct.AddResidual(new byte[25], new short[25], n: 5, stride: 5));
        Throws<ArgumentException>(() =>
            Vp9Reconstruct.AddResidual(new byte[16], new short[15], n: 4, stride: 4));
    }
}
