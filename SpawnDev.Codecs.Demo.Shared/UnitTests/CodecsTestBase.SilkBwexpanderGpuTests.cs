// Cross-backend tests for SilkBwexpanderGpu. Verifies that GPU
// chirp expansion of AR filter coefficients matches the CPU
// SilkBwexpander.Expand16 / .Expand32 reference bit-for-bit.

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
    public async Task SilkBwexpanderGpu_Expand16_Order16_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Typical SILK NB AR filter: order 16, Q12 coefficients in [-32768, 32767].
            short[] arInput = { -1234, 5678, -9012, 3456, -7890, 2345, -6789,
                                 1234, -5678, 9012, -3456, 7890, -2345, 6789,
                                 -123, 456 };
            int chirpQ16 = 64225; // ~0.98 in Q16; typical SILK NLSF stabilization chirp.

            var cpu = (short[])arInput.Clone();
            SilkBwexpander.Expand16(cpu, chirpQ16);

            using var dAr = acc.Allocate1D<short>(arInput.Length);
            dAr.View.CopyFromCPU(arInput);

            using var kernel = new SilkBwexpanderGpuKernel(acc);
            kernel.Run16(dAr.View, arInput.Length, chirpQ16);
            await acc.SynchronizeAsync();

            var gpu = await dAr.CopyToHostAsync();
            for (int i = 0; i < arInput.Length; i++)
                if (cpu[i] != gpu[i])
                    throw new Exception($"ar16[{i}]: cpu={cpu[i]} gpu={gpu[i]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkBwexpanderGpu_Expand32_Order10_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // 32-bit coefficients in Q24 - typical for SILK whitening filters.
            int[] arInput = { 100000, -200000, 300000, -400000, 500000,
                              -600000, 700000, -800000, 900000, -1000000 };
            int chirpQ16 = 65000;

            var cpu = (int[])arInput.Clone();
            SilkBwexpander.Expand32(cpu, chirpQ16);

            using var dAr = acc.Allocate1D<int>(arInput.Length);
            dAr.View.CopyFromCPU(arInput);

            using var kernel = new SilkBwexpanderGpuKernel(acc);
            kernel.Run32(dAr.View, arInput.Length, chirpQ16);
            await acc.SynchronizeAsync();

            var gpu = await dAr.CopyToHostAsync();
            for (int i = 0; i < arInput.Length; i++)
                if (cpu[i] != gpu[i])
                    throw new Exception($"ar32[{i}]: cpu={cpu[i]} gpu={gpu[i]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkBwexpanderGpu_Expand16_RandomBatch_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            const int order = 16;
            var rng = new Random(unchecked((int)0x511BBE16u));
            var arInput = new short[order];
            for (int i = 0; i < order; i++) arInput[i] = (short)rng.Next(-32768, 32768);

            // Try multiple chirp values
            int[] chirps = { 32768, 49152, 60000, 64225, 65000, 65500 };
            foreach (int chirp in chirps)
            {
                var cpu = (short[])arInput.Clone();
                SilkBwexpander.Expand16(cpu, chirp);

                using var dAr = acc.Allocate1D<short>(order);
                dAr.View.CopyFromCPU(arInput);

                using var kernel = new SilkBwexpanderGpuKernel(acc);
                kernel.Run16(dAr.View, order, chirp);
                await acc.SynchronizeAsync();

                var gpu = await dAr.CopyToHostAsync();
                for (int i = 0; i < order; i++)
                    if (cpu[i] != gpu[i])
                        throw new Exception($"ar16[{i}] (chirp={chirp}): cpu={cpu[i]} gpu={gpu[i]}");
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
