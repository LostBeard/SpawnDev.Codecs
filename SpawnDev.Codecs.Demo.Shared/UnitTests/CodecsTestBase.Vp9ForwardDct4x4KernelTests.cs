// Tests for Vp9ForwardDct4x4Kernel. Validates the ILGPU kernel produces
// byte-for-byte identical output to Vp9ForwardDct4x4.Transform across a
// wide range of inputs. VP9 is a normative bitstream and the kernel runs
// on every backend (CPU/CUDA/OpenCL/WebGPU/WebGL/Wasm) so bit-exact
// agreement on each backend is mandatory.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp9ForwardDct4x4Kernel_ZeroInput_ProducesAllZeroCoefs()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9ForwardDct4x4Kernel(acc);
            var input = new short[16];
            var output = new int[16];
            using var dIn = acc.Allocate1D<short>(16);
            using var dOut = acc.Allocate1D<int>(16);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, blockCount: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            readback.AsSpan(0, 16).CopyTo(output);
            for (int i = 0; i < 16; i++) Equal(0, output[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardDct4x4Kernel_DcOnlyInput_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9ForwardDct4x4Kernel(acc);
            // Constant block: input[*] = 64 (residual after subtracting predictor).
            var input = new short[16];
            for (int i = 0; i < 16; i++) input[i] = 64;
            var cpuOut = new int[16];
            Vp9ForwardDct4x4.Transform(input, rowStrideShorts: 4, cpuOut);

            using var dIn = acc.Allocate1D<short>(16);
            using var dOut = acc.Allocate1D<int>(16);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, blockCount: 1);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[16];
            readback.AsSpan(0, 16).CopyTo(gpuOut);

            for (int i = 0; i < 16; i++) Equal(cpuOut[i], gpuOut[i]);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9ForwardDct4x4Kernel_RandomInput_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new Vp9ForwardDct4x4Kernel(acc);
            const int blockCount = 32;
            var rng = new Random(unchecked((int)0x9F4D4404u));
            var input = new short[blockCount * 16];
            for (int i = 0; i < input.Length; i++) input[i] = (short)rng.Next(-1024, 1024);

            // CPU reference output.
            var cpuOut = new int[blockCount * 16];
            for (int b = 0; b < blockCount; b++)
            {
                Vp9ForwardDct4x4.Transform(
                    input.AsSpan(b * 16, 16), rowStrideShorts: 4,
                    cpuOut.AsSpan(b * 16, 16));
            }

            // GPU kernel output.
            using var dIn = acc.Allocate1D<short>(blockCount * 16);
            using var dOut = acc.Allocate1D<int>(blockCount * 16);
            dIn.View.CopyFromCPU(input);
            kernel.Run(dIn.View, dOut.View, blockCount);
            await acc.SynchronizeAsync();
            var readback = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuOut = new int[blockCount * 16];
            readback.AsSpan(0, gpuOut.Length).CopyTo(gpuOut);

            int mismatches = 0;
            for (int i = 0; i < cpuOut.Length; i++)
                if (cpuOut[i] != gpuOut[i]) mismatches++;
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
