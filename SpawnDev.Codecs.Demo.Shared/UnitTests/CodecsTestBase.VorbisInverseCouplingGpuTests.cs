// Cross-backend tests for VorbisInverseCouplingGpu. Verifies the
// per-coefficient (mag, ang) reconstruction matches the CPU
// VorbisInverseCoupling.Apply reference for one channel pair.

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
    public async Task VorbisInverseCouplingGpu_FourQuadrants_MatchesCpu()
    {
        // Test inputs covering all four sign quadrants of (mag, ang).
        var mag = new[] { 1.5f, 1.5f, -1.5f, -1.5f, 0.5f, -0.5f, 2.0f, -2.0f };
        var ang = new[] { 1.0f, -1.0f, 1.0f, -1.0f, 0.3f, -0.3f, -0.8f, 0.8f };
        await CouplingAndVerify(mag, ang);
    }

    [TestMethod]
    public async Task VorbisInverseCouplingGpu_RandomBatch_MatchesCpu()
    {
        const int n = 256;
        var rng = new Random(unchecked((int)0xC04C0BABu));
        var mag = new float[n];
        var ang = new float[n];
        for (int i = 0; i < n; i++)
        {
            mag[i] = (float)(rng.NextDouble() * 4 - 2);
            ang[i] = (float)(rng.NextDouble() * 4 - 2);
        }
        await CouplingAndVerify(mag, ang);
    }

    private async Task CouplingAndVerify(float[] mag, float[] ang)
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            // CPU reference: inline the per-coefficient reconstruction
            // (matches VorbisInverseCoupling.Apply for one step).
            int n = mag.Length;
            var cpuMag = (float[])mag.Clone();
            var cpuAng = (float[])ang.Clone();
            for (int i = 0; i < n; i++)
            {
                float m = cpuMag[i];
                float a = cpuAng[i];
                float newM, newA;
                if (m > 0)
                {
                    if (a > 0) { newM = m; newA = m - a; }
                    else { newA = m; newM = m + a; }
                }
                else
                {
                    if (a > 0) { newM = m; newA = m + a; }
                    else { newA = m; newM = m - a; }
                }
                cpuMag[i] = newM;
                cpuAng[i] = newA;
            }

            using var dMag = acc.Allocate1D<float>(n);
            using var dAng = acc.Allocate1D<float>(n);
            dMag.View.CopyFromCPU(mag);
            dAng.View.CopyFromCPU(ang);

            using var kernel = new VorbisInverseCouplingGpuKernel(acc);
            kernel.Run(dMag.View, dAng.View, n);
            await acc.SynchronizeAsync();

            var gpuMag = await dMag.CopyToHostAsync();
            var gpuAng = await dAng.CopyToHostAsync();

            for (int i = 0; i < n; i++)
            {
                if (cpuMag[i] != gpuMag[i])
                    throw new Exception($"mag[{i}]: cpu={cpuMag[i]} gpu={gpuMag[i]}");
                if (cpuAng[i] != gpuAng[i])
                    throw new Exception($"ang[{i}]: cpu={cpuAng[i]} gpu={gpuAng[i]}");
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
