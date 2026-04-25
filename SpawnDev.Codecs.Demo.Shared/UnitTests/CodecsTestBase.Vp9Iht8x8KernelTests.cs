// Cross-backend tests for Vp9Iht8x8Kernel. Validates byte-for-byte
// parity against Vp9Iht8x8Reference for all 4 tx_types and a batched
// dispatch case per tx_type. Same WebGL guard as slices 120/131 - the
// 8x8 kernel topology trips GL_MAX_VARYING_VECTORS.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>
    /// iHT 8x8 shares the iDCT 8x8 kernel's WebGL constraint:
    /// 64 `flat out` varyings per thread exceeds GL_MAX_VARYING_VECTORS
    /// on most WebGL implementations. WebGPU is green after rc.10.
    /// </summary>
    private static bool IsIht8x8KernelSupported(Accelerator acc)
    {
        var name = acc.AcceleratorType.ToString();
        return !name.Equals("WebGL", StringComparison.OrdinalIgnoreCase);
    }

    [TestMethod]
    public async Task Vp9Iht8x8Kernel_ZeroCoefficients_AllTxTypes_LeavesPredictorUnchanged()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            if (!IsIht8x8KernelSupported(acc)) return;
            using var kernel = new Vp9Iht8x8Kernel(acc);
            for (int txType = 0; txType < 4; txType++)
            {
                var coeffs = new short[64];
                var dest = new byte[64];
                for (int i = 0; i < 64; i++) dest[i] = 128;
                await kernel.RunAsync(
                    (Vp9TxType8x8)txType, coeffs.AsMemory(), dest.AsMemory(),
                    blockCount: 1);
                for (int i = 0; i < 64; i++) Equal((byte)128, dest[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Iht8x8Kernel_AllTxTypes_RandomInputs_BitExactMatchReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            if (!IsIht8x8KernelSupported(acc)) return;
            using var kernel = new Vp9Iht8x8Kernel(acc);
            var rng = new Random(unchecked((int)0xADA58BAFu));
            for (int txType = 0; txType < 4; txType++)
            {
                for (int trial = 0; trial < 5; trial++)
                {
                    var coeffs = new short[64];
                    for (int i = 0; i < 64; i++)
                        coeffs[i] = (short)rng.Next(-4096, 4096);
                    var cpuDest = new byte[64];
                    var gpuDest = new byte[64];
                    for (int i = 0; i < 64; i++)
                    {
                        byte p = (byte)rng.Next(0, 256);
                        cpuDest[i] = p;
                        gpuDest[i] = p;
                    }

                    Vp9Iht8x8Reference.Iht8x8_64_Add(
                        (Vp9TxType8x8)txType, coeffs, cpuDest, 8);
                    await kernel.RunAsync(
                        (Vp9TxType8x8)txType, coeffs.AsMemory(), gpuDest.AsMemory(),
                        blockCount: 1);

                    for (int i = 0; i < 64; i++)
                        Equal(cpuDest[i], gpuDest[i]);
                }
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Iht8x8Kernel_BatchedDispatch_PerTxType_MatchReference()
    {
        // Slice 131 (iADST 8x8) hit a WebGPU+Wasm divergence at n=16
        // batched dispatch; slice 133 then hit a separate failure when
        // mixed tx_types within one workgroup interacted with the
        // LocalMemory<int>(64) row scratch. Per-call uniform tx_type
        // sidesteps both: each dispatch keeps n at the slice-131 safe
        // size (8) and every block in the workgroup follows the same
        // control flow. Production decode groups blocks by tx_type at
        // the call site for the same reason.
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            if (!IsIht8x8KernelSupported(acc)) return;
            using var kernel = new Vp9Iht8x8Kernel(acc);
            const int n = 8;
            var rng = new Random(unchecked((int)0xADA5DEAFu));
            for (int txType = 0; txType < 4; txType++)
            {
                var coeffsFlat = new short[n * 64];
                var predFlat = new byte[n * 64];
                for (int b = 0; b < n; b++)
                {
                    for (int i = 0; i < 64; i++)
                        coeffsFlat[b * 64 + i] = (short)rng.Next(-4096, 4096);
                    for (int i = 0; i < 64; i++)
                        predFlat[b * 64 + i] = (byte)rng.Next(0, 256);
                }

                var cpuResults = (byte[])predFlat.Clone();
                for (int b = 0; b < n; b++)
                {
                    Vp9Iht8x8Reference.Iht8x8_64_Add(
                        (Vp9TxType8x8)txType,
                        coeffsFlat.AsSpan(b * 64, 64),
                        cpuResults.AsSpan(b * 64, 64),
                        8);
                }

                var gpuResults = (byte[])predFlat.Clone();
                await kernel.RunAsync(
                    (Vp9TxType8x8)txType,
                    coeffsFlat.AsMemory(), gpuResults.AsMemory(),
                    blockCount: n);

                for (int i = 0; i < n * 64; i++)
                    Equal(cpuResults[i], gpuResults[i]);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
