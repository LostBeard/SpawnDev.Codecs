// Cross-backend tests for VorbisFloor1ApplyGpuKernel. Verifies the
// per-bin floor1 inverse-dB lookup + multiply produces identical
// output to a CPU reference using the normative InverseDbTable from
// Vorbis I Section 10.1.

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
    public async Task VorbisFloor1ApplyGpu_FullRange_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            const int n = 256;
            var residue = new float[n];
            var curveIdx = new int[n];
            var rng = new Random(unchecked((int)0xCF1A1300u));
            for (int i = 0; i < n; i++)
            {
                residue[i] = (float)(rng.NextDouble() * 2 - 1);
                curveIdx[i] = i; // Cover the full 0..255 index range.
            }
            await ApplyAndVerify(acc, residue, curveIdx);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task VorbisFloor1ApplyGpu_ClampingEdges_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Indices outside [0, 255] should clamp.
            var residue = new[] { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
            var curveIdx = new[] { -1000, -1, 0, 255, 1000 };
            await ApplyAndVerify(acc, residue, curveIdx);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task VorbisFloor1ApplyGpu_HalfBlock1024_RandomBatch_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            const int n = 1024;
            var residue = new float[n];
            var curveIdx = new int[n];
            var rng = new Random(unchecked((int)0xCF1A1024u));
            for (int i = 0; i < n; i++)
            {
                residue[i] = (float)(rng.NextDouble() * 4 - 2);
                curveIdx[i] = rng.Next(0, 256);
            }
            await ApplyAndVerify(acc, residue, curveIdx);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static async Task ApplyAndVerify(Accelerator acc, float[] residue, int[] curveIdx)
    {
        int n = residue.Length;
        var table = VorbisFloor1InverseDbGpu.BuildInverseDbTable();

        // CPU reference.
        var cpu = new float[n];
        for (int i = 0; i < n; i++)
        {
            int idx = curveIdx[i];
            if (idx < 0) idx = 0;
            else if (idx > 255) idx = 255;
            cpu[i] = residue[i] * table[idx];
        }

        using var dResidue = acc.Allocate1D<float>(n);
        using var dCurveIdx = acc.Allocate1D<int>(n);
        using var dTable = acc.Allocate1D<float>(256);
        using var dOutput = acc.Allocate1D<float>(n);

        dResidue.View.CopyFromCPU(residue);
        dCurveIdx.View.CopyFromCPU(curveIdx);
        dTable.View.CopyFromCPU(table);

        using var kernel = new VorbisFloor1ApplyGpuKernel(acc);
        kernel.Run(dResidue.View, dCurveIdx.View, dTable.View, dOutput.View, n);
        await acc.SynchronizeAsync();

        var gpu = await dOutput.CopyToHostAsync();
        for (int i = 0; i < n; i++)
            if (cpu[i] != gpu[i])
                throw new Exception($"bin[{i}]: cpu={cpu[i]} gpu={gpu[i]}");
    }
}
