// Cross-backend test for VorbisFloorMultiplyGpu.MultiplyAt + ZeroAt.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task VorbisFloorMultiplyGpu_RandomBlock_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            const int n = 1024;
            var rng = new Random(unchecked((int)0xA1BACDFu));
            var floor = new float[n];
            var residue = new float[n];
            for (int i = 0; i < n; i++)
            {
                floor[i] = (float)(rng.NextDouble() * 2 - 1);
                residue[i] = (float)(rng.NextDouble() * 2 - 1);
            }

            // CPU reference.
            var cpuOut = new float[n];
            for (int i = 0; i < n; i++) cpuOut[i] = floor[i] * residue[i];

            // GPU.
            using var dFloor = acc.Allocate1D<float>(n);
            using var dResidue = acc.Allocate1D<float>(n);
            using var dOut = acc.Allocate1D<float>(n);
            dFloor.View.CopyFromCPU(floor);
            dResidue.View.CopyFromCPU(residue);
            dOut.View.CopyFromCPU(new float[n]);

            var kernel = acc.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>(
                MultiplyKernel);
            kernel(new Index1D(n), dFloor.View, dResidue.View, dOut.View, n);
            await acc.SynchronizeAsync();

            var gpuOut = await dOut.CopyToHostAsync();
            for (int i = 0; i < n; i++)
                if (cpuOut[i] != gpuOut[i])
                    throw new Exception($"product[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task VorbisFloorMultiplyGpu_ZeroAt_AllZero()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            const int n = 256;
            using var dOut = acc.Allocate1D<float>(n);
            // Pre-fill with non-zero so we can verify ZeroAt actually clears.
            var prefill = new float[n];
            for (int i = 0; i < n; i++) prefill[i] = 1.0f;
            dOut.View.CopyFromCPU(prefill);

            var kernel = acc.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, int>(ZeroKernel);
            kernel(new Index1D(n), dOut.View, n);
            await acc.SynchronizeAsync();

            var gpuOut = await dOut.CopyToHostAsync();
            for (int i = 0; i < n; i++)
                if (gpuOut[i] != 0f)
                    throw new Exception($"zero[{i}]: {gpuOut[i]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static void MultiplyKernel(
        Index1D idx,
        ArrayView<float> floor, ArrayView<float> residue, ArrayView<float> output,
        int count)
    {
        if (idx >= count) return;
        VorbisFloorMultiplyGpu.MultiplyAt(floor, 0, residue, 0, output, 0, idx);
    }

    private static void ZeroKernel(Index1D idx, ArrayView<float> output, int count)
    {
        if (idx >= count) return;
        VorbisFloorMultiplyGpu.ZeroAt(output, 0, idx);
    }
}
