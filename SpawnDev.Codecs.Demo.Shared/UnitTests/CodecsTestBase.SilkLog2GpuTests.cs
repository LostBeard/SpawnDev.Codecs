// Cross-backend tests for SilkLog2Gpu. Verifies the GPU log2lin /
// lin2log approximations are bit-exact with the CPU SilkLog2 reference.

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
    public async Task SilkLog2Gpu_Log2Lin_FullActiveRange_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Active range: [0, 3967). Plus boundary cases.
            var input = new int[4000];
            for (int i = 0; i < input.Length; i++) input[i] = i;
            await Log2LinAndVerify(acc, input);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkLog2Gpu_Log2Lin_Boundaries_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // Negative + clamp boundary + above-clamp.
            var input = new[] { -10000, -1, 0, 1, 2047, 2048, 3966, 3967, 3968, 10000 };
            await Log2LinAndVerify(acc, input);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkLog2Gpu_Lin2Log_RandomBatch_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            const int n = 1024;
            var rng = new Random(unchecked((int)0x511C0210u));
            var input = new int[n];
            // Lin2log expects positive linear values; libopus uses it on
            // gain magnitudes which are always positive.
            for (int i = 0; i < n; i++) input[i] = rng.Next(1, int.MaxValue);
            await Lin2LogAndVerify(acc, input);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task SilkLog2Gpu_RoundTrip_LinLogLin_NearIdentity()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // log2lin and lin2log are near-inverses. Verify GPU
            // round-trip matches CPU round-trip exactly (the
            // approximation produces the same near-identity error
            // on both, but they should match each other).
            var logQ7Input = new[] { 0, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500 };

            using var dIn = acc.Allocate1D<int>(logQ7Input.Length);
            using var dLin = acc.Allocate1D<int>(logQ7Input.Length);
            using var dRoundTrip = acc.Allocate1D<int>(logQ7Input.Length);
            dIn.View.CopyFromCPU(logQ7Input);

            using var kernel = new SilkLog2GpuKernel(acc);
            kernel.Run(dIn.View, dLin.View, logQ7Input.Length, mode: 0);  // log2lin
            await acc.SynchronizeAsync();
            kernel.Run(dLin.View, dRoundTrip.View, logQ7Input.Length, mode: 1);  // lin2log
            await acc.SynchronizeAsync();

            var lin = await dLin.CopyToHostAsync();
            var roundTrip = await dRoundTrip.CopyToHostAsync();

            // CPU reference round-trip.
            for (int i = 0; i < logQ7Input.Length; i++)
            {
                int cpuLin = SilkLog2.silk_log2lin(logQ7Input[i]);
                int cpuRT = SilkLog2.silk_lin2log(cpuLin);
                if (cpuLin != lin[i])
                    throw new Exception($"log2lin[{i}]: cpu={cpuLin} gpu={lin[i]}");
                if (cpuRT != roundTrip[i])
                    throw new Exception($"roundtrip[{i}]: cpu={cpuRT} gpu={roundTrip[i]}");
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task Log2LinAndVerify(Accelerator acc, int[] input)
    {
        var cpu = new int[input.Length];
        for (int i = 0; i < input.Length; i++) cpu[i] = SilkLog2.silk_log2lin(input[i]);

        using var dIn = acc.Allocate1D<int>(input.Length);
        using var dOut = acc.Allocate1D<int>(input.Length);
        dIn.View.CopyFromCPU(input);

        using var kernel = new SilkLog2GpuKernel(acc);
        kernel.Run(dIn.View, dOut.View, input.Length, mode: 0);
        await acc.SynchronizeAsync();

        var gpu = await dOut.CopyToHostAsync();
        for (int i = 0; i < input.Length; i++)
            if (cpu[i] != gpu[i])
                throw new Exception($"log2lin[{i}] (in={input[i]}): cpu={cpu[i]} gpu={gpu[i]}");
    }

    private static async Task Lin2LogAndVerify(Accelerator acc, int[] input)
    {
        var cpu = new int[input.Length];
        for (int i = 0; i < input.Length; i++) cpu[i] = SilkLog2.silk_lin2log(input[i]);

        using var dIn = acc.Allocate1D<int>(input.Length);
        using var dOut = acc.Allocate1D<int>(input.Length);
        dIn.View.CopyFromCPU(input);

        using var kernel = new SilkLog2GpuKernel(acc);
        kernel.Run(dIn.View, dOut.View, input.Length, mode: 1);
        await acc.SynchronizeAsync();

        var gpu = await dOut.CopyToHostAsync();
        for (int i = 0; i < input.Length; i++)
            if (cpu[i] != gpu[i])
                throw new Exception($"lin2log[{i}] (in={input[i]}): cpu={cpu[i]} gpu={gpu[i]}");
    }
}
