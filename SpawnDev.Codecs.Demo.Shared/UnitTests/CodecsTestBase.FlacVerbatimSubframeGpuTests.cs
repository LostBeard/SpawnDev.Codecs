// Cross-backend tests for FlacVerbatimSubframeGpu encoder + decoder
// pair via FlacVerbatimSubframeRoundTripKernel. Verifies the
// VERBATIM subframe round-trip (encode then decode in same dispatch
// reproduces input samples exactly).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task FlacVerbatimSubframeGpu_RoundTrip_Constant_16bit()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            const int n = 64;
            const int bps = 16;
            var samples = new int[n];
            for (int i = 0; i < n; i++) samples[i] = 1234; // constant value
            await VerbatimRoundTripAndVerify(acc, samples, bps);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacVerbatimSubframeGpu_RoundTrip_Random_16bit()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            const int n = 256;
            const int bps = 16;
            var rng = new Random(unchecked((int)0xF1AC1620u));
            var samples = new int[n];
            for (int i = 0; i < n; i++) samples[i] = rng.Next(-32768, 32768);
            await VerbatimRoundTripAndVerify(acc, samples, bps);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacVerbatimSubframeGpu_RoundTrip_Random_24bit()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            const int n = 128;
            const int bps = 24;
            var rng = new Random(unchecked((int)0xF1AC2400u));
            var samples = new int[n];
            int max = 1 << 23;
            for (int i = 0; i < n; i++) samples[i] = rng.Next(-max, max);
            await VerbatimRoundTripAndVerify(acc, samples, bps);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacVerbatimSubframeGpu_RoundTrip_Random_8bit()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            const int n = 512;
            const int bps = 8;
            var rng = new Random(unchecked((int)0xF1AC0800u));
            var samples = new int[n];
            for (int i = 0; i < n; i++) samples[i] = rng.Next(-128, 128);
            await VerbatimRoundTripAndVerify(acc, samples, bps);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task VerbatimRoundTripAndVerify(Accelerator acc, int[] samples, int bps)
    {
        // Worst case: 8-bit subframe header + bps * sampleCount samples + 7 bits padding.
        int worstCaseBits = 8 + bps * samples.Length + 7;
        int scratchLen = (worstCaseBits + 7) / 8 + 8;

        using var dSamples = acc.Allocate1D<int>(samples.Length);
        using var dDecoded = acc.Allocate1D<int>(samples.Length);
        using var dScratch = acc.Allocate1D<byte>(scratchLen);
        using var dOutLen = acc.Allocate1D<long>(1);
        using var dStatus = acc.Allocate1D<int>(1);

        dSamples.View.CopyFromCPU(samples);
        dScratch.View.CopyFromCPU(new byte[scratchLen]);

        using var kernel = new FlacVerbatimSubframeRoundTripKernel(acc);
        kernel.Run(dSamples.View, dDecoded.View, dScratch.View, dOutLen.View, dStatus.View,
            samples.Length, bps);
        await acc.SynchronizeAsync();

        int status = (await dStatus.CopyToHostAsync())[0];
        Equal(1, status, "decoder status (1=success)");
        var decoded = await dDecoded.CopyToHostAsync();
        for (int i = 0; i < samples.Length; i++)
            if (samples[i] != decoded[i])
                throw new Exception($"sample[{i}]: input={samples[i]} decoded={decoded[i]}");
    }
}
