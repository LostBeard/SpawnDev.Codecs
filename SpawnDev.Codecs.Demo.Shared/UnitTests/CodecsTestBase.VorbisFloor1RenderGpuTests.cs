// Cross-backend tests for VorbisFloor1RenderGpu.
// Verifies the GPU RenderLine + RenderPoint produce bit-exact float
// outputs vs the CPU VorbisFloor1Curve.RenderLine / .RenderPoint
// reference. Floor 1 line rendering is the inner loop of Vorbis spectral
// envelope synthesis (Vorbis I sec 7.2.4).

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
    public async Task VorbisFloor1RenderGpu_RenderPoint_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Test points spanning representative Floor 1 segments.
            var cases = new (int x0, int y0, int x1, int y1, int x)[]
            {
                (0, 50, 100, 150, 25),
                (0, 200, 50, 100, 30),
                (10, 0, 30, 255, 20),
                (5, 128, 25, 64, 15),
            };

            using var dInputs = acc.Allocate1D<int>(cases.Length * 5);
            using var dOutputs = acc.Allocate1D<int>(cases.Length);
            var inputs = new int[cases.Length * 5];
            for (int i = 0; i < cases.Length; i++)
            {
                inputs[i * 5 + 0] = cases[i].x0;
                inputs[i * 5 + 1] = cases[i].y0;
                inputs[i * 5 + 2] = cases[i].x1;
                inputs[i * 5 + 3] = cases[i].y1;
                inputs[i * 5 + 4] = cases[i].x;
            }
            dInputs.View.CopyFromCPU(inputs);

            var kernel = acc.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<int>, ArrayView<int>, int>(RenderPointKernel);
            kernel(new Index1D(cases.Length), dInputs.View, dOutputs.View, cases.Length);
            await acc.SynchronizeAsync();

            var gpuOut = await dOutputs.CopyToHostAsync();
            for (int i = 0; i < cases.Length; i++)
            {
                int cpu = VorbisFloor1Curve.RenderPoint(
                    cases[i].x0, cases[i].y0, cases[i].x1, cases[i].y1, cases[i].x);
                if (cpu != gpuOut[i])
                    throw new Exception($"RenderPoint case[{i}]: cpu={cpu} gpu={gpuOut[i]}");
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task VorbisFloor1RenderGpu_RenderLine_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            const int halfBlock = 256;
            var inverseDb = VorbisFloor1InverseDbGpu.BuildInverseDbTable();

            // Single descending line: x0=0..200, y0=255 -> y1=0.
            int x0 = 0, y0 = 255, x1 = 200, y1 = 0;

            // CPU reference.
            var cpuOut = new float[halfBlock];
            VorbisFloor1Curve.RenderLine(x0, y0, x1, y1, cpuOut, halfBlock);

            // GPU through a tiny single-thread kernel.
            using var dOut = acc.Allocate1D<float>(halfBlock);
            using var dInverseDb = acc.Allocate1D<float>(inverseDb.Length);
            dOut.View.CopyFromCPU(new float[halfBlock]);
            dInverseDb.View.CopyFromCPU(inverseDb);

            var kernel = acc.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>,
                int, int, int, int, int>(RenderLineKernel);
            kernel(new Index1D(1), dOut.View, dInverseDb.View, x0, y0, x1, y1, halfBlock);
            await acc.SynchronizeAsync();

            var gpuOut = await dOut.CopyToHostAsync();
            for (int i = 0; i < halfBlock; i++)
                if (cpuOut[i] != gpuOut[i])
                    throw new Exception($"RenderLine[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static void RenderPointKernel(
        Index1D idx,
        ArrayView<int> inputs, ArrayView<int> outputs, int count)
    {
        if (idx >= count) return;
        int x0 = inputs[idx * 5 + 0];
        int y0 = inputs[idx * 5 + 1];
        int x1 = inputs[idx * 5 + 2];
        int y1 = inputs[idx * 5 + 3];
        int x = inputs[idx * 5 + 4];
        outputs[idx] = VorbisFloor1RenderGpu.RenderPoint(x0, y0, x1, y1, x);
    }

    private static void RenderLineKernel(
        Index1D _,
        ArrayView<float> outBuf, ArrayView<float> inverseDb,
        int x0, int y0, int x1, int y1, int halfBlock)
    {
        VorbisFloor1RenderGpu.RenderLine(x0, y0, x1, y1, outBuf, 0, inverseDb, 0, halfBlock);
    }
}
