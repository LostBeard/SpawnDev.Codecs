// Cross-backend test for Av1InverseIdentityTransformGpu. Verifies the 4
// GPU per-element inverse-identity helpers (Inverse4At/8At/16At/32At)
// match the CPU reference Av1InverseIdentity.Transform{4,8,16,32}
// bit-exactly.

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
    public async Task Av1InverseIdentityTransformGpu_Inverse4_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var rng = new Random(unchecked((int)0xAA01_DDD4u));
            int[] input = new int[4];
            for (int i = 0; i < 4; i++) input[i] = rng.Next(-16384, 16384);

            int[] cpuOut = new int[4];
            Av1InverseIdentity.Transform4(input, cpuOut);

            await GpuVerifyInv(acc, input, cpuOut, size: 4, kernelKind: 4);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1InverseIdentityTransformGpu_Inverse8_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var rng = new Random(unchecked((int)0xAA01_DDD8u));
            int[] input = new int[8];
            for (int i = 0; i < 8; i++) input[i] = rng.Next(-16384, 16384);

            int[] cpuOut = new int[8];
            Av1InverseIdentity.Transform8(input, cpuOut);

            await GpuVerifyInv(acc, input, cpuOut, size: 8, kernelKind: 8);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1InverseIdentityTransformGpu_Inverse16_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var rng = new Random(unchecked((int)0xAA01_DD16u));
            int[] input = new int[16];
            for (int i = 0; i < 16; i++) input[i] = rng.Next(-16384, 16384);

            int[] cpuOut = new int[16];
            Av1InverseIdentity.Transform16(input, cpuOut);

            await GpuVerifyInv(acc, input, cpuOut, size: 16, kernelKind: 16);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1InverseIdentityTransformGpu_Inverse32_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var rng = new Random(unchecked((int)0xAA01_DD32u));
            int[] input = new int[32];
            for (int i = 0; i < 32; i++) input[i] = rng.Next(-16384, 16384);

            int[] cpuOut = new int[32];
            Av1InverseIdentity.Transform32(input, cpuOut);

            await GpuVerifyInv(acc, input, cpuOut, size: 32, kernelKind: 32);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task GpuVerifyInv(Accelerator acc, int[] input, int[] expected, int size, int kernelKind)
    {
        using var dIn = acc.Allocate1D<int>(size);
        using var dOut = acc.Allocate1D<int>(size);
        dIn.View.CopyFromCPU(input);
        dOut.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, int>(InverseIdentityKernel);
        kernel(new Index1D(size), dIn.View, dOut.View, kernelKind);
        await acc.SynchronizeAsync();

        var gpuOut = await dOut.CopyToHostAsync();

        for (int i = 0; i < size; i++)
        {
            if (expected[i] != gpuOut[i])
                throw new Exception($"output[{i}]: cpu={expected[i]} gpu={gpuOut[i]} (size={size})");
        }
    }

    private static void InverseIdentityKernel(
        Index1D index,
        ArrayView<int> input, ArrayView<int> output, int kernelKind)
    {
        if (kernelKind == 4) Av1InverseIdentityTransformGpu.Inverse4At(input, 0, output, 0, index.X);
        else if (kernelKind == 8) Av1InverseIdentityTransformGpu.Inverse8At(input, 0, output, 0, index.X);
        else if (kernelKind == 16) Av1InverseIdentityTransformGpu.Inverse16At(input, 0, output, 0, index.X);
        else Av1InverseIdentityTransformGpu.Inverse32At(input, 0, output, 0, index.X);
    }
}
