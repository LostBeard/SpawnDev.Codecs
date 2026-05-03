// Cross-backend tests for VorbisEncoderHelpersGpu.MagnitudeToFloorY +
// QuantiseResidueValue. Verifies the per-sample encoder helpers match
// the CPU VorbisAudioEncoder reference helpers across all backends.
//
// Both functions use approximate float math (Log10 / Round / Ceiling),
// so tolerance-based comparison is appropriate for the magnitude->Y
// path. The quantise function uses pure integer + float Round so it
// should match exactly.

using System.Reflection;
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
    public async Task VorbisEncoderHelpersGpu_MagnitudeToFloorY_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Sample magnitudes spanning the full Y range:
            //   Y=0   at 1.065e-7
            //   Y=255 at 1.0
            //   ...  exponentially distributed
            // Production encoder magnitudes come from finite spectrum peaks;
            // skip Infinity / NaN here (the GPU primitive's contract is that
            // it expects finite, non-negative input - same as CPU's).
            float[] mags =
            {
                0.0f, 1.0e-8f, 1.065e-7f, 5.0e-7f, 1.0e-6f, 1.0e-5f, 1.0e-4f,
                0.001f, 0.005f, 0.01f, 0.05f, 0.1f, 0.5f, 0.9f, 1.0f, 2.0f,
            };

            // CPU reference (via reflection because MagnitudeToFloorY is private).
            var cpuMethod = typeof(VorbisAudioEncoder)
                .GetMethod("MagnitudeToFloorY", BindingFlags.NonPublic | BindingFlags.Static)
                ?? throw new InvalidOperationException("MagnitudeToFloorY not found.");
            var cpuOut = new int[mags.Length];
            for (int i = 0; i < mags.Length; i++)
                cpuOut[i] = (int)cpuMethod.Invoke(null, new object[] { mags[i] })!;

            // GPU.
            var inverseDb = VorbisFloor1InverseDbGpu.BuildInverseDbTable();
            using var dMags = acc.Allocate1D<float>(mags.Length);
            using var dOut = acc.Allocate1D<int>(mags.Length);
            using var dInverseDb = acc.Allocate1D<float>(inverseDb.Length);
            dMags.View.CopyFromCPU(mags);
            dInverseDb.View.CopyFromCPU(inverseDb);
            var kernel = acc.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, int>(MagToYKernel);
            kernel(new Index1D(mags.Length), dMags.View, dOut.View, dInverseDb.View, mags.Length);
            await acc.SynchronizeAsync();

            var gpuOut = await dOut.CopyToHostAsync();
            // Allow off-by-one because Log10 floats vs doubles can shift Ceiling
            // by one step at boundary magnitudes.
            for (int i = 0; i < mags.Length; i++)
            {
                int delta = Math.Abs(cpuOut[i] - gpuOut[i]);
                if (delta > 1)
                    throw new Exception($"FloorY[{i}] mag={mags[i]}: cpu={cpuOut[i]} gpu={gpuOut[i]}");
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task VorbisEncoderHelpersGpu_QuantiseResidueValue_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Test inputs spanning the full residue range with the v1
            // encoder's defaults (residueRange=2, bookEntries=1024).
            const float residueRange = 2.0f;
            const int bookEntries = 1024;
            float[] values =
            {
                -3.0f, -2.0f, -1.5f, -1.0f, -0.5f, -0.1f, -0.001f, 0.0f,
                0.001f, 0.1f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f,
            };

            // CPU reference (re-implemented inline since the helper is private).
            var cpuOut = new int[values.Length];
            float step = 2f * residueRange / bookEntries;
            int half = bookEntries / 2;
            for (int i = 0; i < values.Length; i++)
            {
                int idx = (int)Math.Round(values[i] / step) + half;
                if (idx < 0) idx = 0;
                if (idx >= bookEntries) idx = bookEntries - 1;
                cpuOut[i] = idx;
            }

            // GPU.
            using var dValues = acc.Allocate1D<float>(values.Length);
            using var dOut = acc.Allocate1D<int>(values.Length);
            dValues.View.CopyFromCPU(values);
            var kernel = acc.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<int>, int, float, int>(QuantiseKernel);
            kernel(new Index1D(values.Length), dValues.View, dOut.View,
                values.Length, residueRange, bookEntries);
            await acc.SynchronizeAsync();

            var gpuOut = await dOut.CopyToHostAsync();
            for (int i = 0; i < values.Length; i++)
                if (cpuOut[i] != gpuOut[i])
                    throw new Exception($"Quantise[{i}] v={values[i]}: cpu={cpuOut[i]} gpu={gpuOut[i]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static void MagToYKernel(
        Index1D idx, ArrayView<float> mags, ArrayView<int> output,
        ArrayView<float> inverseDb, int count)
    {
        if (idx >= count) return;
        output[idx] = VorbisEncoderHelpersGpu.MagnitudeToFloorY(mags[idx], inverseDb, 0);
    }

    private static void QuantiseKernel(
        Index1D idx, ArrayView<float> values, ArrayView<int> output,
        int count, float residueRange, int bookEntries)
    {
        if (idx >= count) return;
        output[idx] = VorbisEncoderHelpersGpu.QuantiseResidueValue(
            values[idx], residueRange, bookEntries);
    }
}
