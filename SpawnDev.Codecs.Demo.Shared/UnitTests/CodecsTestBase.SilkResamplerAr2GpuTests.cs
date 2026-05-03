// Cross-backend test for SilkResamplerAr2Gpu.ApplyAt. Verifies the GPU
// AR2 IIR pre-filter matches the CPU reference SilkResampler.Ar2 bit-exactly
// across representative coefficient sets used by libopus down-FIR resampler.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task SilkResamplerAr2Gpu_TypicalCoefs_RandomInput_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Coefs from libopus silk_resampler_down2_3 (AR2 head). Q14 values.
            short[] aQ14 = { 9202, -3271 };
            var rng = new Random(unchecked((int)0xCAFEF00Du));
            short[] input = new short[480];
            for (int i = 0; i < input.Length; i++)
                input[i] = (short)rng.Next(-32768, 32768);
            int[] state = new int[2];
            await ApplyAndVerify(acc, input, aQ14, state);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkResamplerAr2Gpu_NonZeroState_FullScale_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Stress the IIR state path with non-zero seed + full-scale alternating.
            short[] aQ14 = { 16384, -8192 };
            short[] input = new short[256];
            for (int i = 0; i < input.Length; i++)
                input[i] = (short)((i & 1) == 0 ? short.MaxValue : short.MinValue);
            int[] state = { 1 << 22, -1 << 22 };
            await ApplyAndVerify(acc, input, aQ14, state);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkResamplerAr2Gpu_LongStream_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Long sequential stream covering several batch sizes worth.
            short[] aQ14 = { 12288, -4096 };
            var rng = new Random(unchecked((int)0xABADCAFEu));
            short[] input = new short[2048];
            for (int i = 0; i < input.Length; i++)
                input[i] = (short)rng.Next(-32768, 32768);
            int[] state = new int[2];
            await ApplyAndVerify(acc, input, aQ14, state);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task ApplyAndVerify(Accelerator acc, short[] input, short[] aQ14, int[] initialState)
    {
        int len = input.Length;

        // CPU reference.
        int[] cpuState = (int[])initialState.Clone();
        int[] cpuOut = new int[len];
        SilkResampler.Ar2(cpuState, cpuOut, input, aQ14);

        // GPU dispatch: single-thread per stream.
        using var dState = acc.Allocate1D<int>(2);
        using var dOut = acc.Allocate1D<int>(len);
        using var dIn = acc.Allocate1D<short>(len);
        using var dA = acc.Allocate1D<short>(2);
        dState.View.CopyFromCPU(initialState);
        dIn.View.CopyFromCPU(input);
        dA.View.CopyFromCPU(aQ14);
        dOut.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, ArrayView<short>, ArrayView<short>, int>(
            ApplyKernel);
        kernel(new Index1D(1), dState.View, dOut.View, dIn.View, dA.View, len);
        await acc.SynchronizeAsync();

        var gpuOut = await dOut.CopyToHostAsync();
        var gpuState = await dState.CopyToHostAsync();

        for (int i = 0; i < len; i++)
        {
            if (cpuOut[i] != gpuOut[i])
                throw new Exception($"out[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]} (len={len})");
        }
        if (cpuState[0] != gpuState[0] || cpuState[1] != gpuState[1])
            throw new Exception($"state: cpu=({cpuState[0]},{cpuState[1]}) gpu=({gpuState[0]},{gpuState[1]})");
    }

    private static void ApplyKernel(
        Index1D _,
        ArrayView<int> state, ArrayView<int> outQ8, ArrayView<short> input, ArrayView<short> aQ14,
        int len)
    {
        SilkResamplerAr2Gpu.ApplyAt(state, 0, outQ8, 0, input, 0, aQ14, 0, len);
    }
}
