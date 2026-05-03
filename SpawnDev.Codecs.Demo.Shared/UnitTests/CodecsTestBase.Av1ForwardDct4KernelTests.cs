// Tests for Av1ForwardDct4Kernel. Validates the ILGPU kernel produces
// bit-exact output to Av1ForwardDct4.Transform across (a) zero,
// (b) DC-only / structured input, (c) random batches. AV1 forward
// transforms are normative encoder-side operations and the kernel
// must agree with the CPU reference on every backend.

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
    public async Task Av1ForwardDct4Kernel_ZeroInput_ProducesAllZeroCoefs()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1ForwardDct4Kernel(acc);
            var input = new int[4];
            var cpuOut = new int[4];
            Av1ForwardDct4.Transform(input, cpuOut);

            using var dIn = acc.Allocate1D<int>(4);
            using var dOut = acc.Allocate1D<int>(4);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, transformCount: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[4];
            readback.AsSpan(0, 4).CopyTo(gpuOut);
            for (int i = 0; i < 4; i++) Equal(cpuOut[i], gpuOut[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1ForwardDct4Kernel_DcOnlyInput_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1ForwardDct4Kernel(acc);
            // Constant input - the AV1 fdct should put energy at DC only.
            var input = new int[] { 256, 256, 256, 256 };
            var cpuOut = new int[4];
            Av1ForwardDct4.Transform(input, cpuOut);

            using var dIn = acc.Allocate1D<int>(4);
            using var dOut = acc.Allocate1D<int>(4);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, transformCount: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[4];
            readback.AsSpan(0, 4).CopyTo(gpuOut);
            for (int i = 0; i < 4; i++) Equal(cpuOut[i], gpuOut[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1ForwardDct4Kernel_RandomBatch_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1ForwardDct4Kernel(acc);
            const int transformCount = 64;
            var rng = new Random(0xA1F4);
            var input = new int[transformCount * 4];
            for (int i = 0; i < input.Length; i++) input[i] = rng.Next(-2048, 2048);

            // CPU reference: run Transform per 4-element slice.
            var cpuOut = new int[transformCount * 4];
            for (int t = 0; t < transformCount; t++)
                Av1ForwardDct4.Transform(input.AsSpan(t * 4, 4), cpuOut.AsSpan(t * 4, 4));

            using var dIn = acc.Allocate1D<int>(transformCount * 4);
            using var dOut = acc.Allocate1D<int>(transformCount * 4);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, transformCount);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[transformCount * 4];
            readback.AsSpan(0, gpuOut.Length).CopyTo(gpuOut);

            int mismatches = 0;
            for (int i = 0; i < cpuOut.Length; i++)
                if (cpuOut[i] != gpuOut[i]) mismatches++;
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1ForwardDct4Kernel_AllCosBits_MatchCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1ForwardDct4Kernel(acc);
            const int transformCount = 16;
            var rng = new Random(0xFD4B);
            var input = new int[transformCount * 4];
            for (int i = 0; i < input.Length; i++) input[i] = rng.Next(-1024, 1024);

            for (int cosBit = 10; cosBit <= 13; cosBit++)
            {
                var cpuOut = new int[transformCount * 4];
                for (int t = 0; t < transformCount; t++)
                    Av1ForwardDct4.Transform(input.AsSpan(t * 4, 4), cpuOut.AsSpan(t * 4, 4), cosBit);

                using var dIn = acc.Allocate1D<int>(transformCount * 4);
                using var dOut = acc.Allocate1D<int>(transformCount * 4);
                dIn.View.CopyFromCPU(input);
                kernel.Run(dIn.View, dOut.View, transformCount, cosBit);
                await acc.SynchronizeAsync();
                var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
                var gpuOut = new int[transformCount * 4];
                readback.AsSpan(0, gpuOut.Length).CopyTo(gpuOut);

                int mismatches = 0;
                for (int i = 0; i < cpuOut.Length; i++)
                    if (cpuOut[i] != gpuOut[i]) mismatches++;
                Equal(0, mismatches, $"cosBit={cosBit}");
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
