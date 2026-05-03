// Cross-backend tests for Vp9Iht4x4Kernel. Validates byte-for-byte
// parity against Vp9Iht4x4Reference across every tx_type and every
// runner's native ILGPU accelerator. Mirrors the slice 130 iADST 4x4
// kernel test shape, harmonized with slice 133's per-dispatch
// scalar-tx_type API.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp9Iht4x4Kernel_ZeroCoefficients_AllTxTypes_LeavesPredictorUnchanged()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9Iht4x4Kernel(acc);
            for (int txType = 0; txType < 4; txType++)
            {
                var coeffs = new short[16];
                var dest = new byte[16];
                for (int i = 0; i < 16; i++) dest[i] = 128;
                await kernel.RunAsync(
                    (Vp9TxType4x4)txType, coeffs.AsMemory(), dest.AsMemory(),
                    blockCount: 1);
                for (int i = 0; i < 16; i++) Equal((byte)128, dest[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Iht4x4Kernel_AllTxTypes_RandomInputs_BitExactMatchReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9Iht4x4Kernel(acc);
            var rng = new Random(unchecked((int)0xADA51B74u));
            for (int txType = 0; txType < 4; txType++)
            {
                for (int trial = 0; trial < 10; trial++)
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

                    Vp9Iht4x4Reference.Iht4x4_16_Add(
                        (Vp9TxType4x4)txType, coeffs, cpuDest, 4);
                    await kernel.RunAsync(
                        (Vp9TxType4x4)txType, coeffs.AsMemory(), gpuDest.AsMemory(),
                        blockCount: 1);

                    for (int i = 0; i < 16; i++)
                        Equal(cpuDest[i], gpuDest[i]);
                }
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Iht4x4Kernel_BatchedDispatch_PerTxType_MatchReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9Iht4x4Kernel(acc);
            const int n = 64;
            var rng = new Random(unchecked((int)0xADA5C70Eu));
            for (int txType = 0; txType < 4; txType++)
            {
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
                    Vp9Iht4x4Reference.Iht4x4_16_Add(
                        (Vp9TxType4x4)txType,
                        coeffsFlat.AsSpan(b * 16, 16),
                        cpuResults.AsSpan(b * 16, 16),
                        4);
                }

                var gpuResults = (byte[])predFlat.Clone();
                await kernel.RunAsync(
                    (Vp9TxType4x4)txType,
                    coeffsFlat.AsMemory(), gpuResults.AsMemory(),
                    blockCount: n);

                for (int i = 0; i < n * 16; i++)
                    Equal(cpuResults[i], gpuResults[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
