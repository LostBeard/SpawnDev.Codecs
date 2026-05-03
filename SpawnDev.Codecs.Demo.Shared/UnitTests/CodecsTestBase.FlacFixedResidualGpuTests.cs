// Cross-backend tests for FlacFixedResidualGpu. Verifies the per-
// sample forward-difference computation matches the CPU
// FlacFixedSubframeEncoder reference for orders 1..4.

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
    public async Task FlacFixedResidualGpu_Order1_RandomBatch_MatchesCpu()
        => await OrderResidualAndVerify(1);

    [TestMethod]
    public async Task FlacFixedResidualGpu_Order2_RandomBatch_MatchesCpu()
        => await OrderResidualAndVerify(2);

    [TestMethod]
    public async Task FlacFixedResidualGpu_Order3_RandomBatch_MatchesCpu()
        => await OrderResidualAndVerify(3);

    [TestMethod]
    public async Task FlacFixedResidualGpu_Order4_RandomBatch_MatchesCpu()
        => await OrderResidualAndVerify(4);

    private async Task OrderResidualAndVerify(int order)
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            const int sampleCount = 1024;
            var rng = new Random(unchecked((int)0xF1AC0000u + order));
            var samples = new int[sampleCount];
            for (int i = 0; i < sampleCount; i++) samples[i] = rng.Next(-32768, 32768);

            // CPU reference: forward-difference per the libFLAC predictor
            // coefficients (mirrors FlacFixedSubframeEncoder.ComputeResidual).
            int residualCount = sampleCount - order;
            var cpuResidual = new int[residualCount];
            for (int ri = 0; ri < residualCount; ri++)
            {
                int n = ri + order;
                cpuResidual[ri] = order switch
                {
                    1 => samples[n] - samples[n - 1],
                    2 => samples[n] - 2 * samples[n - 1] + samples[n - 2],
                    3 => samples[n] - 3 * samples[n - 1] + 3 * samples[n - 2] - samples[n - 3],
                    4 => samples[n] - 4 * samples[n - 1] + 6 * samples[n - 2] - 4 * samples[n - 3] + samples[n - 4],
                    _ => throw new ArgumentOutOfRangeException(nameof(order)),
                };
            }

            using var dSamples = acc.Allocate1D<int>(sampleCount);
            using var dResidual = acc.Allocate1D<int>(residualCount);
            dSamples.View.CopyFromCPU(samples);

            using var kernel = new FlacFixedResidualGpuKernel(acc);
            kernel.Run(dSamples.View, dResidual.View, sampleCount, order);
            await acc.SynchronizeAsync();

            var gpuResidual = await dResidual.CopyToHostAsync();
            int mismatches = 0;
            int firstMismatch = -1;
            for (int i = 0; i < residualCount; i++)
                if (cpuResidual[i] != gpuResidual[i])
                {
                    if (firstMismatch < 0) firstMismatch = i;
                    mismatches++;
                }
            if (mismatches > 0)
                throw new Exception(
                    $"order={order} {mismatches} mismatches; first at i={firstMismatch} cpu={cpuResidual[firstMismatch]} gpu={gpuResidual[firstMismatch]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
