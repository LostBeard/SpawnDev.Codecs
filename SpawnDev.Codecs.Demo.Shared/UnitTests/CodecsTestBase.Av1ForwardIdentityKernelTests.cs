// Tests for Av1ForwardIdentity{4,8,16,32}Kernel. Validates bit-exact
// match with Av1ForwardIdentity.Transform{4,8,16,32} across (a) zero,
// (b) DC-only / structured, (c) random batches.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    // -------- Identity 4 --------

    [TestMethod]
    public async Task Av1ForwardIdentity4Kernel_ZeroInput_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1ForwardIdentity4Kernel(acc);
            var input = new int[4];
            var cpuOut = new int[4];
            Av1ForwardIdentity.Transform4(input, cpuOut);

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
    public async Task Av1ForwardIdentity4Kernel_RandomBatch_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1ForwardIdentity4Kernel(acc);
            const int transformCount = 64;
            var rng = new Random(0xAD14);
            var input = new int[transformCount * 4];
            // Identity transform multiplies by sqrt(2) - keep input bounded so
            // the int output stays in range.
            for (int i = 0; i < input.Length; i++) input[i] = rng.Next(-65536, 65536);

            var cpuOut = new int[transformCount * 4];
            for (int t = 0; t < transformCount; t++)
                Av1ForwardIdentity.Transform4(input.AsSpan(t * 4, 4), cpuOut.AsSpan(t * 4, 4));

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
    public async Task Av1ForwardIdentity4Kernel_DcOnlyInput_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1ForwardIdentity4Kernel(acc);
            var input = new int[] { 1024, 1024, 1024, 1024 };
            var cpuOut = new int[4];
            Av1ForwardIdentity.Transform4(input, cpuOut);

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

    // -------- Identity 8 --------

    [TestMethod]
    public async Task Av1ForwardIdentity8Kernel_ZeroInput_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1ForwardIdentity8Kernel(acc);
            var input = new int[8];
            var cpuOut = new int[8];
            Av1ForwardIdentity.Transform8(input, cpuOut);

            using var dIn = acc.Allocate1D<int>(8);
            using var dOut = acc.Allocate1D<int>(8);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, transformCount: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[8];
            readback.AsSpan(0, 8).CopyTo(gpuOut);
            for (int i = 0; i < 8; i++) Equal(cpuOut[i], gpuOut[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1ForwardIdentity8Kernel_RandomBatch_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1ForwardIdentity8Kernel(acc);
            const int transformCount = 64;
            var rng = new Random(0xAD18);
            var input = new int[transformCount * 8];
            for (int i = 0; i < input.Length; i++) input[i] = rng.Next(-1048576, 1048576);

            var cpuOut = new int[transformCount * 8];
            for (int t = 0; t < transformCount; t++)
                Av1ForwardIdentity.Transform8(input.AsSpan(t * 8, 8), cpuOut.AsSpan(t * 8, 8));

            using var dIn = acc.Allocate1D<int>(transformCount * 8);
            using var dOut = acc.Allocate1D<int>(transformCount * 8);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, transformCount);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[transformCount * 8];
            readback.AsSpan(0, gpuOut.Length).CopyTo(gpuOut);

            int mismatches = 0;
            for (int i = 0; i < cpuOut.Length; i++)
                if (cpuOut[i] != gpuOut[i]) mismatches++;
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1ForwardIdentity8Kernel_DcOnlyInput_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1ForwardIdentity8Kernel(acc);
            var input = new int[8];
            for (int i = 0; i < 8; i++) input[i] = 256;
            var cpuOut = new int[8];
            Av1ForwardIdentity.Transform8(input, cpuOut);

            using var dIn = acc.Allocate1D<int>(8);
            using var dOut = acc.Allocate1D<int>(8);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, transformCount: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[8];
            readback.AsSpan(0, 8).CopyTo(gpuOut);
            for (int i = 0; i < 8; i++) Equal(cpuOut[i], gpuOut[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    // -------- Identity 16 --------

    [TestMethod]
    public async Task Av1ForwardIdentity16Kernel_ZeroInput_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1ForwardIdentity16Kernel(acc);
            var input = new int[16];
            var cpuOut = new int[16];
            Av1ForwardIdentity.Transform16(input, cpuOut);

            using var dIn = acc.Allocate1D<int>(16);
            using var dOut = acc.Allocate1D<int>(16);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, transformCount: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[16];
            readback.AsSpan(0, 16).CopyTo(gpuOut);
            for (int i = 0; i < 16; i++) Equal(cpuOut[i], gpuOut[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1ForwardIdentity16Kernel_RandomBatch_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1ForwardIdentity16Kernel(acc);
            const int transformCount = 32;
            var rng = new Random(0xAD16);
            var input = new int[transformCount * 16];
            for (int i = 0; i < input.Length; i++) input[i] = rng.Next(-65536, 65536);

            var cpuOut = new int[transformCount * 16];
            for (int t = 0; t < transformCount; t++)
                Av1ForwardIdentity.Transform16(input.AsSpan(t * 16, 16), cpuOut.AsSpan(t * 16, 16));

            using var dIn = acc.Allocate1D<int>(transformCount * 16);
            using var dOut = acc.Allocate1D<int>(transformCount * 16);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, transformCount);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[transformCount * 16];
            readback.AsSpan(0, gpuOut.Length).CopyTo(gpuOut);

            int mismatches = 0;
            for (int i = 0; i < cpuOut.Length; i++)
                if (cpuOut[i] != gpuOut[i]) mismatches++;
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1ForwardIdentity16Kernel_DcOnlyInput_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1ForwardIdentity16Kernel(acc);
            var input = new int[16];
            for (int i = 0; i < 16; i++) input[i] = 1024;
            var cpuOut = new int[16];
            Av1ForwardIdentity.Transform16(input, cpuOut);

            using var dIn = acc.Allocate1D<int>(16);
            using var dOut = acc.Allocate1D<int>(16);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, transformCount: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[16];
            readback.AsSpan(0, 16).CopyTo(gpuOut);
            for (int i = 0; i < 16; i++) Equal(cpuOut[i], gpuOut[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    // -------- Identity 32 --------

    [TestMethod]
    public async Task Av1ForwardIdentity32Kernel_ZeroInput_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1ForwardIdentity32Kernel(acc);
            var input = new int[32];
            var cpuOut = new int[32];
            Av1ForwardIdentity.Transform32(input, cpuOut);

            using var dIn = acc.Allocate1D<int>(32);
            using var dOut = acc.Allocate1D<int>(32);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, transformCount: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[32];
            readback.AsSpan(0, 32).CopyTo(gpuOut);
            for (int i = 0; i < 32; i++) Equal(cpuOut[i], gpuOut[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1ForwardIdentity32Kernel_RandomBatch_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1ForwardIdentity32Kernel(acc);
            const int transformCount = 16;
            var rng = new Random(0xAD32);
            var input = new int[transformCount * 32];
            for (int i = 0; i < input.Length; i++) input[i] = rng.Next(-524288, 524288);

            var cpuOut = new int[transformCount * 32];
            for (int t = 0; t < transformCount; t++)
                Av1ForwardIdentity.Transform32(input.AsSpan(t * 32, 32), cpuOut.AsSpan(t * 32, 32));

            using var dIn = acc.Allocate1D<int>(transformCount * 32);
            using var dOut = acc.Allocate1D<int>(transformCount * 32);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, transformCount);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[transformCount * 32];
            readback.AsSpan(0, gpuOut.Length).CopyTo(gpuOut);

            int mismatches = 0;
            for (int i = 0; i < cpuOut.Length; i++)
                if (cpuOut[i] != gpuOut[i]) mismatches++;
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1ForwardIdentity32Kernel_DcOnlyInput_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Av1ForwardIdentity32Kernel(acc);
            var input = new int[32];
            for (int i = 0; i < 32; i++) input[i] = 256;
            var cpuOut = new int[32];
            Av1ForwardIdentity.Transform32(input, cpuOut);

            using var dIn = acc.Allocate1D<int>(32);
            using var dOut = acc.Allocate1D<int>(32);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, transformCount: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[32];
            readback.AsSpan(0, 32).CopyTo(gpuOut);
            for (int i = 0; i < 32; i++) Equal(cpuOut[i], gpuOut[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
