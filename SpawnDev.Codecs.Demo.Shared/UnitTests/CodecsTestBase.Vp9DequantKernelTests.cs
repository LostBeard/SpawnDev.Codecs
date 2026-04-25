// Cross-backend tests for Vp9DequantKernel. Validates byte-for-byte
// parity against Vp9Dequantizer (slice 134's CPU oracle). Per-
// coefficient parallel work, no LocalMemory, no atomics - expected
// to run on every backend including WebGL, unlike the iDCT/iADST/iHT
// kernels.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp9DequantKernel_ZeroCoefficients_StayZero()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9DequantKernel(acc);
            var coeffs = new short[64];
            var quant = new Vp9PlaneQuantizer(Dc: 100, Ac: 50);
            await kernel.RunAsync(coeffs.AsMemory(), quant, blockCount: 1, coeffsPerBlock: 64);
            for (int i = 0; i < 64; i++) Equal((short)0, coeffs[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DequantKernel_SingleBlock_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9DequantKernel(acc);
            var rng = new Random(unchecked((int)0xADA5DE91u));
            // Cover all four block sizes used by VP9.
            int[] sizes = { 16, 64, 256, 1024 };
            foreach (var n in sizes)
            {
                var cpuCoeffs = new short[n];
                var gpuCoeffs = new short[n];
                for (int i = 0; i < n; i++)
                {
                    short v = (short)rng.Next(-2048, 2048);
                    cpuCoeffs[i] = v;
                    gpuCoeffs[i] = v;
                }
                var quant = new Vp9PlaneQuantizer(
                    Dc: (short)rng.Next(4, 1336),
                    Ac: (short)rng.Next(4, 1828));

                Vp9Dequantizer.DequantizeInPlace(cpuCoeffs, quant);
                await kernel.RunAsync(gpuCoeffs.AsMemory(), quant, blockCount: 1, coeffsPerBlock: n);

                for (int i = 0; i < n; i++)
                    Equal(cpuCoeffs[i], gpuCoeffs[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DequantKernel_BatchedBlocks_MatchesCpuReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9DequantKernel(acc);
            var rng = new Random(unchecked((int)0xADA5BA7Cu));
            const int coeffsPerBlock = 64;
            const int blockCount = 32;
            int total = coeffsPerBlock * blockCount;

            var cpuCoeffs = new short[total];
            var gpuCoeffs = new short[total];
            for (int i = 0; i < total; i++)
            {
                short v = (short)rng.Next(-2048, 2048);
                cpuCoeffs[i] = v;
                gpuCoeffs[i] = v;
            }
            var quant = new Vp9PlaneQuantizer(Dc: 71, Ac: 39);

            for (int b = 0; b < blockCount; b++)
            {
                Vp9Dequantizer.DequantizeInPlace(
                    cpuCoeffs.AsSpan(b * coeffsPerBlock, coeffsPerBlock),
                    quant);
            }
            await kernel.RunAsync(gpuCoeffs.AsMemory(), quant, blockCount, coeffsPerBlock);

            for (int i = 0; i < total; i++)
                Equal(cpuCoeffs[i], gpuCoeffs[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9DequantKernel_SaturatesAtInt16Bounds()
    {
        // Deliberately push the coeff*quant product past short range and
        // verify the kernel saturates to short.MinValue / short.MaxValue
        // identically to the CPU oracle.
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9DequantKernel(acc);
            var coeffs = new short[16];
            coeffs[0] = 4096;        // DC: 4096 * 16 = 65536 -> saturates to 32767
            coeffs[1] = -4096;       // AC: -4096 * 16 = -65536 -> saturates to -32768
            coeffs[2] = 100;         // AC: 100 * 16 = 1600 -> in range
            coeffs[3] = -100;        // AC: -100 * 16 = -1600 -> in range
            // The remaining positions (4..15) stay zero.
            var quant = new Vp9PlaneQuantizer(Dc: 16, Ac: 16);

            await kernel.RunAsync(coeffs.AsMemory(), quant, blockCount: 1, coeffsPerBlock: 16);

            Equal(short.MaxValue, coeffs[0]);
            Equal(short.MinValue, coeffs[1]);
            Equal((short)1600, coeffs[2]);
            Equal((short)(-1600), coeffs[3]);
            for (int i = 4; i < 16; i++) Equal((short)0, coeffs[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
