// Cross-backend tests for Av1Inverse2dTransformGpu. Verifies the
// GPU-callable 2D inverse DCT helpers (Tx4x4 + Tx8x8 + Tx16x16 DCT_DCT)
// produce output bit-exact vs Av1Inverse2dTransform.Apply CPU
// reference.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Av1Inverse2dTransformGpu_4x4_ZeroInput_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1Inverse2dTransformGpuKernel(acc);
            int n = 16;
            var coefs = new int[n];
            var cpuOut = new int[n];
            Av1Inverse2dTransform.Apply(Av1TxSize.Tx4x4, Av1TxType.DctDct, coefs, cpuOut);

            using var dCoefs = acc.Allocate1D<int>(n);
            using var dRes = acc.Allocate1D<int>(n);
            using var dScratch = acc.Allocate1D<int>(Av1Inverse2dTransformGpuKernel.ScratchPerBlock(0));
            dCoefs.View.CopyFromCPU(coefs);
            kernel.Run(dCoefs.View, dRes.View, dScratch.View, blockCount: 1, txSize: 0);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dRes);
            for (int i = 0; i < n; i++) Equal(cpuOut[i], readback[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1Inverse2dTransformGpu_4x4_DcOnly_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1Inverse2dTransformGpuKernel(acc);
            int n = 16;
            var coefs = new int[n];
            // DC-only: all energy at (0, 0).
            coefs[0] = 256;
            var cpuOut = new int[n];
            Av1Inverse2dTransform.Apply(Av1TxSize.Tx4x4, Av1TxType.DctDct, coefs, cpuOut);

            using var dCoefs = acc.Allocate1D<int>(n);
            using var dRes = acc.Allocate1D<int>(n);
            using var dScratch = acc.Allocate1D<int>(Av1Inverse2dTransformGpuKernel.ScratchPerBlock(0));
            dCoefs.View.CopyFromCPU(coefs);
            kernel.Run(dCoefs.View, dRes.View, dScratch.View, blockCount: 1, txSize: 0);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dRes);
            for (int i = 0; i < n; i++) Equal(cpuOut[i], readback[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1Inverse2dTransformGpu_4x4_RandomBatch_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1Inverse2dTransformGpuKernel(acc);
            const int blockCount = 64;
            int n = 16;
            int scratchN = Av1Inverse2dTransformGpuKernel.ScratchPerBlock(0);
            var rng = new Random(unchecked((int)0xA124AABEu));
            var coefs = new int[blockCount * n];
            for (int i = 0; i < coefs.Length; i++) coefs[i] = rng.Next(-512, 512);

            var cpuOut = new int[blockCount * n];
            for (int b = 0; b < blockCount; b++)
                Av1Inverse2dTransform.Apply(Av1TxSize.Tx4x4, Av1TxType.DctDct,
                    coefs.AsSpan(b * n, n), cpuOut.AsSpan(b * n, n));

            using var dCoefs = acc.Allocate1D<int>(blockCount * n);
            using var dRes = acc.Allocate1D<int>(blockCount * n);
            using var dScratch = acc.Allocate1D<int>(blockCount * scratchN);
            dCoefs.View.CopyFromCPU(coefs);
            kernel.Run(dCoefs.View, dRes.View, dScratch.View, blockCount, txSize: 0);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dRes);

            int mismatches = 0;
            int firstMismatch = -1;
            for (int i = 0; i < cpuOut.Length; i++)
                if (cpuOut[i] != readback[i])
                {
                    if (firstMismatch < 0) firstMismatch = i;
                    mismatches++;
                }
            if (mismatches > 0)
                throw new Exception($"{mismatches} mismatches; first at i={firstMismatch} cpu={cpuOut[firstMismatch]} gpu={readback[firstMismatch]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1Inverse2dTransformGpu_8x8_ZeroInput_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1Inverse2dTransformGpuKernel(acc);
            int n = 64;
            var coefs = new int[n];
            var cpuOut = new int[n];
            Av1Inverse2dTransform.Apply(Av1TxSize.Tx8x8, Av1TxType.DctDct, coefs, cpuOut);

            using var dCoefs = acc.Allocate1D<int>(n);
            using var dRes = acc.Allocate1D<int>(n);
            using var dScratch = acc.Allocate1D<int>(Av1Inverse2dTransformGpuKernel.ScratchPerBlock(1));
            dCoefs.View.CopyFromCPU(coefs);
            kernel.Run(dCoefs.View, dRes.View, dScratch.View, blockCount: 1, txSize: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dRes);
            for (int i = 0; i < n; i++) Equal(cpuOut[i], readback[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1Inverse2dTransformGpu_8x8_RandomBatch_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1Inverse2dTransformGpuKernel(acc);
            const int blockCount = 32;
            int n = 64;
            var rng = new Random(unchecked((int)0xA12D8BE0u));
            var coefs = new int[blockCount * n];
            for (int i = 0; i < coefs.Length; i++) coefs[i] = rng.Next(-2048, 2048);

            var cpuOut = new int[blockCount * n];
            for (int b = 0; b < blockCount; b++)
                Av1Inverse2dTransform.Apply(Av1TxSize.Tx8x8, Av1TxType.DctDct,
                    coefs.AsSpan(b * n, n), cpuOut.AsSpan(b * n, n));

            using var dCoefs = acc.Allocate1D<int>(blockCount * n);
            using var dRes = acc.Allocate1D<int>(blockCount * n);
            using var dScratch = acc.Allocate1D<int>(blockCount * Av1Inverse2dTransformGpuKernel.ScratchPerBlock(1));
            dCoefs.View.CopyFromCPU(coefs);
            kernel.Run(dCoefs.View, dRes.View, dScratch.View, blockCount, txSize: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dRes);

            int mismatches = 0;
            int firstMismatch = -1;
            for (int i = 0; i < cpuOut.Length; i++)
                if (cpuOut[i] != readback[i])
                {
                    if (firstMismatch < 0) firstMismatch = i;
                    mismatches++;
                }
            if (mismatches > 0)
                throw new Exception($"{mismatches} mismatches; first at i={firstMismatch} cpu={cpuOut[firstMismatch]} gpu={readback[firstMismatch]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1Inverse2dTransformGpu_16x16_ZeroInput_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1Inverse2dTransformGpuKernel(acc);
            int n = 256;
            var coefs = new int[n];
            var cpuOut = new int[n];
            Av1Inverse2dTransform.Apply(Av1TxSize.Tx16x16, Av1TxType.DctDct, coefs, cpuOut);

            using var dCoefs = acc.Allocate1D<int>(n);
            using var dRes = acc.Allocate1D<int>(n);
            using var dScratch = acc.Allocate1D<int>(Av1Inverse2dTransformGpuKernel.ScratchPerBlock(2));
            dCoefs.View.CopyFromCPU(coefs);
            kernel.Run(dCoefs.View, dRes.View, dScratch.View, blockCount: 1, txSize: 2);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dRes);
            for (int i = 0; i < n; i++) Equal(cpuOut[i], readback[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1Inverse2dTransformGpu_16x16_RandomBatch_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1Inverse2dTransformGpuKernel(acc);
            const int blockCount = 16;
            int n = 256;
            var rng = new Random(unchecked((int)0xA1216BE0u));
            var coefs = new int[blockCount * n];
            for (int i = 0; i < coefs.Length; i++) coefs[i] = rng.Next(-2048, 2048);

            var cpuOut = new int[blockCount * n];
            for (int b = 0; b < blockCount; b++)
                Av1Inverse2dTransform.Apply(Av1TxSize.Tx16x16, Av1TxType.DctDct,
                    coefs.AsSpan(b * n, n), cpuOut.AsSpan(b * n, n));

            using var dCoefs = acc.Allocate1D<int>(blockCount * n);
            using var dRes = acc.Allocate1D<int>(blockCount * n);
            using var dScratch = acc.Allocate1D<int>(blockCount * Av1Inverse2dTransformGpuKernel.ScratchPerBlock(2));
            dCoefs.View.CopyFromCPU(coefs);
            kernel.Run(dCoefs.View, dRes.View, dScratch.View, blockCount, txSize: 2);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dRes);

            int mismatches = 0;
            int firstMismatch = -1;
            for (int i = 0; i < cpuOut.Length; i++)
                if (cpuOut[i] != readback[i])
                {
                    if (firstMismatch < 0) firstMismatch = i;
                    mismatches++;
                }
            if (mismatches > 0)
                throw new Exception($"{mismatches} mismatches; first at i={firstMismatch} cpu={cpuOut[firstMismatch]} gpu={readback[firstMismatch]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
