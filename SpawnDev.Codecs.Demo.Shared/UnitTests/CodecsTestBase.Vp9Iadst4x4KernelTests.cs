// Cross-backend tests for Vp9Iadst4x4Kernel. Validates byte-for-byte
// parity against Vp9Iadst4x4Reference on every runner's native ILGPU
// accelerator. Mirrors slice 117's iDCT 4x4 kernel test structure.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp9Iadst4x4Kernel_ZeroCoefficients_LeavesPredictorUnchanged()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9Iadst4x4Kernel(acc);
            var coeffs = new short[16];
            var dest = new byte[16];
            for (int i = 0; i < 16; i++) dest[i] = 128;
            await kernel.RunAsync(coeffs.AsMemory(), dest.AsMemory(), blockCount: 1);
            for (int i = 0; i < 16; i++) Equal((byte)128, dest[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Iadst4x4Kernel_RandomInputs_BitExactMatchReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9Iadst4x4Kernel(acc);
            var rng = new Random(unchecked((int)0xADA574E1u));
            for (int trial = 0; trial < 25; trial++)
            {
                var coeffs = new short[16];
                for (int i = 0; i < 16; i++)
                    coeffs[i] = (short)rng.Next(-2048, 2048);
                var cpuDest = new byte[16];
                var gpuDest = new byte[16];
                for (int i = 0; i < 16; i++)
                {
                    byte p = (byte)rng.Next(0, 256);
                    cpuDest[i] = p;
                    gpuDest[i] = p;
                }

                Vp9Iadst4x4Reference.IadstAdst4x4_16_Add(coeffs, cpuDest, 4);
                await kernel.RunAsync(coeffs.AsMemory(), gpuDest.AsMemory(), blockCount: 1);

                for (int i = 0; i < 16; i++)
                    Equal(cpuDest[i], gpuDest[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Iadst4x4Kernel_BatchedDispatch_AllBlocksMatchReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp9Iadst4x4Kernel(acc);
            const int n = 64;
            var rng = new Random(unchecked((int)0xADA5DEC0u));
            var coeffsFlat = new short[n * 16];
            var predFlat = new byte[n * 16];
            for (int b = 0; b < n; b++)
            {
                for (int i = 0; i < 16; i++)
                    coeffsFlat[b * 16 + i] = (short)rng.Next(-2048, 2048);
                for (int i = 0; i < 16; i++)
                    predFlat[b * 16 + i] = (byte)rng.Next(0, 256);
            }

            var cpuResults = (byte[])predFlat.Clone();
            for (int b = 0; b < n; b++)
            {
                Vp9Iadst4x4Reference.IadstAdst4x4_16_Add(
                    coeffsFlat.AsSpan(b * 16, 16),
                    cpuResults.AsSpan(b * 16, 16),
                    4);
            }

            var gpuResults = (byte[])predFlat.Clone();
            await kernel.RunAsync(coeffsFlat.AsMemory(), gpuResults.AsMemory(), blockCount: n);

            for (int i = 0; i < n * 16; i++)
                Equal(cpuResults[i], gpuResults[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
