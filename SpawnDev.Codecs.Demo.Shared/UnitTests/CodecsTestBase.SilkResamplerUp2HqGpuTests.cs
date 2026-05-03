// Cross-backend test for SilkResamplerUp2HqGpu.ApplyAt. Verifies the GPU
// 2x HQ upsampler matches the CPU reference SilkResampler.Up2Hq bit-exactly
// across representative input lengths + signal patterns.

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
    public async Task SilkResamplerUp2HqGpu_ZeroState_RandomInput_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Typical SILK 8 kHz frame (160 samples for 20 ms) upsampled to 16 kHz.
            var rng = new Random(unchecked((int)0xFEEDFACEu));
            short[] input = new short[160];
            for (int i = 0; i < input.Length; i++)
                input[i] = (short)rng.Next(-32768, 32768);
            int[] state = new int[6];
            await ResampleAndVerify(acc, input, state);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkResamplerUp2HqGpu_NonZeroState_FullScale_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Stress the SAT16 path with a non-zero IIR state + full-scale input.
            short[] input = new short[320];
            for (int i = 0; i < input.Length; i++)
                input[i] = (short)((i & 1) == 0 ? short.MaxValue : short.MinValue);
            int[] state = { 1 << 20, -1 << 20, 1 << 19, -1 << 19, 1 << 18, -1 << 18 };
            await ResampleAndVerify(acc, input, state);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkResamplerUp2HqGpu_Wb40msFrame_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Wideband 40 ms frame at 16 kHz (640 samples) upsampled to 32 kHz.
            // Covers a full SILK NB-WB analysis window.
            var rng = new Random(unchecked((int)0x12345678u));
            short[] input = new short[640];
            for (int i = 0; i < input.Length; i++)
                input[i] = (short)rng.Next(-32768, 32768);
            int[] state = new int[6];
            await ResampleAndVerify(acc, input, state);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task ResampleAndVerify(Accelerator acc, short[] input, int[] initialState)
    {
        int len = input.Length;

        // CPU reference: copy state, run Up2Hq.
        int[] cpuState = (int[])initialState.Clone();
        short[] cpuOut = new short[2 * len];
        SilkResampler.Up2Hq(cpuState, cpuOut, input, len);

        // GPU dispatch: single-thread per stream (sequential IIR state).
        using var dState = acc.Allocate1D<int>(6);
        using var dIn = acc.Allocate1D<short>(len);
        using var dOut = acc.Allocate1D<short>(2 * len);
        dState.View.CopyFromCPU(initialState);
        dIn.View.CopyFromCPU(input);
        dOut.MemSetToZero();

        var kernel = acc.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<short>, ArrayView<short>, int>(
            ApplyKernel);
        kernel(new Index1D(1), dState.View, dOut.View, dIn.View, len);
        await acc.SynchronizeAsync();

        var gpuOut = await dOut.CopyToHostAsync();
        var gpuState = await dState.CopyToHostAsync();

        for (int i = 0; i < cpuOut.Length; i++)
        {
            if (cpuOut[i] != gpuOut[i])
                throw new Exception($"sample[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]} (len={len})");
        }
        for (int i = 0; i < 6; i++)
        {
            if (cpuState[i] != gpuState[i])
                throw new Exception($"state[{i}]: cpu={cpuState[i]} gpu={gpuState[i]}");
        }
    }

    private static void ApplyKernel(
        Index1D _,
        ArrayView<int> state, ArrayView<short> output, ArrayView<short> input,
        int len)
    {
        SilkResamplerUp2HqGpu.ApplyAt(state, 0, output, 0, input, 0, len);
    }
}
