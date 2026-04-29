// Cross-backend test for VorbisFwdMdctScaledGpu.ForwardScaledAt.
// Verifies the GPU forward MDCT + 4/N scaling matches what the CPU
// Vorbis encoder produces (MdctReference output * 4f/n) within float
// tolerance (XMath.Cos is float-precision while CPU MdctReference uses
// double-precision Math.Cos -- same tolerance pattern as
// VorbisWindowGpu_ApplyWindow).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Transforms;
using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task VorbisFwdMdctScaledGpu_RandomBlock_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            const int n = 256;       // half-block (output bins)
            const int twoN = 2 * n;  // input length

            var rng = new Random(unchecked((int)0xA1F0A4D1u));
            var input = new float[twoN];
            for (int i = 0; i < twoN; i++)
                input[i] = (float)(rng.NextDouble() * 2 - 1);

            // CPU reference (windowed * MDCT * 4/N - we skip the window and
            // just verify the MDCT + scaling step).
            var cpuSpectrum = new float[n];
            MdctReference.Transform(input, cpuSpectrum);
            float scale = 4f / n;
            for (int i = 0; i < n; i++) cpuSpectrum[i] *= scale;

            // GPU.
            using var dInput = acc.Allocate1D<float>(twoN);
            using var dOutput = acc.Allocate1D<float>(n);
            dInput.View.CopyFromCPU(input);
            dOutput.View.CopyFromCPU(new float[n]);

            var kernel = acc.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int>(FwdMdctKernel);
            kernel(new Index1D(n), dInput.View, dOutput.View, n);
            await acc.SynchronizeAsync();

            var gpuSpectrum = await dOutput.CopyToHostAsync();
            // Allow ~1 ULP per bin for float-cos vs double-cos drift,
            // multiplied by 2N samples summed -> tolerance scales with n.
            // For n=256, 2N=512 cosines summed, max float drift ~1e-4 per bin.
            const float tol = 1e-3f;
            for (int i = 0; i < n; i++)
            {
                float delta = MathF.Abs(cpuSpectrum[i] - gpuSpectrum[i]);
                if (delta > tol)
                    throw new Exception($"spectrum[{i}]: cpu={cpuSpectrum[i]} gpu={gpuSpectrum[i]} delta={delta}");
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static void FwdMdctKernel(
        Index1D idx, ArrayView<float> input, ArrayView<float> output, int n)
    {
        if (idx >= n) return;
        VorbisFwdMdctScaledGpu.ForwardScaledAt(input, 0, output, 0, n, idx);
    }
}
