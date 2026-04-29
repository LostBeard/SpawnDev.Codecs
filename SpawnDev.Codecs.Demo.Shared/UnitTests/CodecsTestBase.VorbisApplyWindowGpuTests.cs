// Cross-backend test for VorbisWindowGpu.ApplyWindowAt.
// Verifies the per-sample window + multiply produces the same float
// output as the CPU reference path used by the Vorbis encoder
// (block[i] * VorbisWindow.GenerateCanonical[i]).

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
    public async Task VorbisWindowGpu_ApplyWindow_RandomBlock_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            const int n = 1024;
            var rng = new Random(unchecked((int)0xA1B0F1u));
            var block = new float[n];
            for (int i = 0; i < n; i++) block[i] = (float)(rng.NextDouble() * 2 - 1);

            // CPU reference.
            var window = VorbisWindow.GenerateCanonical(n);
            var cpuOut = new float[n];
            for (int i = 0; i < n; i++) cpuOut[i] = block[i] * window[i];

            // GPU.
            using var dInput = acc.Allocate1D<float>(n);
            using var dOutput = acc.Allocate1D<float>(n);
            dInput.View.CopyFromCPU(block);
            dOutput.View.CopyFromCPU(new float[n]);

            var kernel = acc.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int>(ApplyWindowKernel);
            kernel(new Index1D(n), dInput.View, dOutput.View, n);
            await acc.SynchronizeAsync();

            var gpuOut = await dOutput.CopyToHostAsync();
            // Tolerance comparison: VorbisWindowGpu.CanonicalSample uses
            // float-precision XMath.Sin while CPU VorbisWindow.GenerateCanonical
            // uses double-precision Math.Sin. Difference is < 1 ULP per sample,
            // matching the existing VorbisWindowGpu canonical-window test pattern.
            const float tol = 1e-6f;
            for (int i = 0; i < n; i++)
            {
                float delta = MathF.Abs(cpuOut[i] - gpuOut[i]);
                if (delta > tol)
                    throw new Exception($"Windowed[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]} delta={delta}");
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static void ApplyWindowKernel(
        Index1D idx, ArrayView<float> input, ArrayView<float> output, int n)
    {
        if (idx >= n) return;
        VorbisWindowGpu.ApplyWindowAt(input, 0, output, 0, idx, n);
    }
}
