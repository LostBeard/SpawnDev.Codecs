// Cross-backend tests for VorbisWindowGpu canonical window generation.
// Verifies the per-sample sin-of-sin-squared math matches CPU
// VorbisWindow.GenerateCanonical within float tolerance.

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
    public async Task VorbisWindowGpu_Canonical_N64_MatchesCpu() => await CanonicalWindowAndVerify(64);

    [TestMethod]
    public async Task VorbisWindowGpu_Canonical_N256_MatchesCpu() => await CanonicalWindowAndVerify(256);

    [TestMethod]
    public async Task VorbisWindowGpu_Canonical_N2048_MatchesCpu() => await CanonicalWindowAndVerify(2048);

    private async Task CanonicalWindowAndVerify(int n)
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var cpu = VorbisWindow.GenerateCanonical(n);
            using var dOut = acc.Allocate1D<float>(n);
            using var kernel = new VorbisCanonicalWindowGpuKernel(acc);
            kernel.Run(dOut.View, n);
            await acc.SynchronizeAsync();
            var gpu = await dOut.CopyToHostAsync();
            const float tol = 1e-6f;
            for (int i = 0; i < n; i++)
            {
                float delta = MathF.Abs(cpu[i] - gpu[i]);
                if (delta > tol)
                    throw new Exception($"sample[{i}]: cpu={cpu[i]} gpu={gpu[i]} delta={delta}");
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
