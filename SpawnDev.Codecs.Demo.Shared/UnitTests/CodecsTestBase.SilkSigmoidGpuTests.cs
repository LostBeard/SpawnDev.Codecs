// Cross-backend tests for SilkSigmoidGpu. Verifies the GPU sigmoid
// approximation is bit-exact with the CPU SilkSigmoid reference.
// First piece of the Opus SILK GPU pipeline build-out.

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
    public async Task SilkSigmoidGpu_ExtremeValues_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var input = new[]
            {
                -10000, -200, -192, -191, -150, -100, -50, -1,
                0, 1, 50, 100, 150, 191, 192, 200, 10000,
            };
            await SigmoidAndVerify(acc, input);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkSigmoidGpu_FullRange_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Test every Q5 value in the active range plus boundaries.
            var input = new int[2 * 6 * 32 + 200];
            int idx = 0;
            for (int v = -6 * 32 - 100; v <= 6 * 32 + 100; v++)
            {
                if (idx < input.Length) input[idx++] = v;
            }
            // Truncate to actual filled length.
            var truncated = new int[idx];
            Array.Copy(input, truncated, idx);
            await SigmoidAndVerify(acc, truncated);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkSigmoidGpu_RandomBatch_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            const int n = 1024;
            var rng = new Random(unchecked((int)0x511C0500u));
            var input = new int[n];
            for (int i = 0; i < n; i++) input[i] = rng.Next(-300, 300);
            await SigmoidAndVerify(acc, input);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task SigmoidAndVerify(Accelerator acc, int[] input)
    {
        // Reflection-based access to the internal SilkSigmoid CPU
        // helper. The InternalsVisibleTo on SpawnDev.Codecs already
        // exposes these to SpawnDev.Codecs.Demo.Shared.
        var cpu = new int[input.Length];
        for (int i = 0; i < input.Length; i++)
            cpu[i] = SpawnDev.Codecs.Audio.Opus.Silk.SilkSigmoid.silk_sigm_Q15(input[i]);

        using var dInput = acc.Allocate1D<int>(input.Length);
        using var dOutput = acc.Allocate1D<int>(input.Length);
        dInput.View.CopyFromCPU(input);

        using var kernel = new SilkSigmoidGpuKernel(acc);
        kernel.Run(dInput.View, dOutput.View, input.Length);
        await acc.SynchronizeAsync();

        var gpu = await dOutput.CopyToHostAsync();
        for (int i = 0; i < input.Length; i++)
            if (cpu[i] != gpu[i])
                throw new Exception($"sigm[{i}] (in={input[i]}): cpu={cpu[i]} gpu={gpu[i]}");
    }
}
