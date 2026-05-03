// Cross-backend tests for Av1Forward2dTransformGpu. Verifies the
// GPU-callable 2D forward DCT helpers (Tx8x8 + Tx16x16 DCT_DCT)
// produce output bit-exact vs Av1Forward2dTransform.Apply CPU
// reference across (a) zero, (b) DC-only, (c) random batches.

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
    public async Task Av1Forward2dTransformGpu_8x8_ZeroInput_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1Forward2dTransformGpuKernel(acc);
            int n = 64;
            var input = new short[n];
            var cpuOut = new int[n];
            Av1Forward2dTransform.Apply(Av1TxSize.Tx8x8, Av1TxType.DctDct, input, cpuOut);

            using var dIn = acc.Allocate1D<short>(n);
            using var dOut = acc.Allocate1D<int>(n);
            using var dScratch = acc.Allocate1D<int>(n);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, dScratch.View, blockCount: 1, txSize: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            for (int i = 0; i < n; i++) Equal(cpuOut[i], readback[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1Forward2dTransformGpu_8x8_DcOnly_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1Forward2dTransformGpuKernel(acc);
            int n = 64;
            var input = new short[n];
            for (int i = 0; i < n; i++) input[i] = 32;
            var cpuOut = new int[n];
            Av1Forward2dTransform.Apply(Av1TxSize.Tx8x8, Av1TxType.DctDct, input, cpuOut);

            using var dIn = acc.Allocate1D<short>(n);
            using var dOut = acc.Allocate1D<int>(n);
            using var dScratch = acc.Allocate1D<int>(n);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, dScratch.View, blockCount: 1, txSize: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            for (int i = 0; i < n; i++) Equal(cpuOut[i], readback[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1Forward2dTransformGpu_8x8_RandomBatch_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1Forward2dTransformGpuKernel(acc);
            const int blockCount = 32;
            int n = 64;
            var rng = new Random(unchecked((int)0xA12D8BADu));
            var input = new short[blockCount * n];
            for (int i = 0; i < input.Length; i++) input[i] = (short)rng.Next(-128, 128);

            var cpuOut = new int[blockCount * n];
            for (int b = 0; b < blockCount; b++)
                Av1Forward2dTransform.Apply(Av1TxSize.Tx8x8, Av1TxType.DctDct,
                    input.AsSpan(b * n, n), cpuOut.AsSpan(b * n, n));

            using var dIn = acc.Allocate1D<short>(blockCount * n);
            using var dOut = acc.Allocate1D<int>(blockCount * n);
            using var dScratch = acc.Allocate1D<int>(blockCount * n);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, dScratch.View, blockCount, txSize: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);

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
    public async Task Av1Forward2dTransformGpu_16x16_ZeroInput_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1Forward2dTransformGpuKernel(acc);
            int n = 256;
            var input = new short[n];
            var cpuOut = new int[n];
            Av1Forward2dTransform.Apply(Av1TxSize.Tx16x16, Av1TxType.DctDct, input, cpuOut);

            using var dIn = acc.Allocate1D<short>(n);
            using var dOut = acc.Allocate1D<int>(n);
            using var dScratch = acc.Allocate1D<int>(n);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, dScratch.View, blockCount: 1, txSize: 2);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            for (int i = 0; i < n; i++) Equal(cpuOut[i], readback[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1Forward2dTransformGpu_16x16_DcOnly_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1Forward2dTransformGpuKernel(acc);
            int n = 256;
            var input = new short[n];
            for (int i = 0; i < n; i++) input[i] = 64;
            var cpuOut = new int[n];
            Av1Forward2dTransform.Apply(Av1TxSize.Tx16x16, Av1TxType.DctDct, input, cpuOut);

            using var dIn = acc.Allocate1D<short>(n);
            using var dOut = acc.Allocate1D<int>(n);
            using var dScratch = acc.Allocate1D<int>(n);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, dScratch.View, blockCount: 1, txSize: 2);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            for (int i = 0; i < n; i++) Equal(cpuOut[i], readback[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1Forward2dTransformGpu_16x16_RandomBatch_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1Forward2dTransformGpuKernel(acc);
            const int blockCount = 16;
            int n = 256;
            var rng = new Random(unchecked((int)0xA1216BADu));
            var input = new short[blockCount * n];
            for (int i = 0; i < input.Length; i++) input[i] = (short)rng.Next(-128, 128);

            var cpuOut = new int[blockCount * n];
            for (int b = 0; b < blockCount; b++)
                Av1Forward2dTransform.Apply(Av1TxSize.Tx16x16, Av1TxType.DctDct,
                    input.AsSpan(b * n, n), cpuOut.AsSpan(b * n, n));

            using var dIn = acc.Allocate1D<short>(blockCount * n);
            using var dOut = acc.Allocate1D<int>(blockCount * n);
            using var dScratch = acc.Allocate1D<int>(blockCount * n);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, dScratch.View, blockCount, txSize: 2);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);

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
