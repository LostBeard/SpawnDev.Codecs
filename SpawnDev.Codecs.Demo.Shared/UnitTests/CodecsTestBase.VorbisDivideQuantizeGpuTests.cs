// Cross-backend test for VorbisEncoderHelpersGpu.DivideQuantizeAt.
// Verifies the per-bin spectrum/floor divide + quantize matches the
// CPU reference loop in VorbisAudioEncoder.EncodeAudioPacket.

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
    public async Task VorbisEncoderHelpersGpu_DivideQuantize_RandomBlock_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            const int half = 512;
            const float residueRange = 2.0f;
            const int bookEntries = 1024;
            var rng = new Random(unchecked((int)0xA1F0DEFu));
            var spectrum = new float[half];
            var floorCurve = new float[half];
            for (int i = 0; i < half; i++)
            {
                spectrum[i] = (float)(rng.NextDouble() * 2 - 1);
                // Floor values in a realistic encoder range (1e-7 to 1.0).
                floorCurve[i] = (float)(rng.NextDouble() * 0.5 + 1e-6);
            }
            // Test the floor-floor branch with a near-zero entry.
            floorCurve[33] = 1e-15f;

            // CPU reference.
            var cpuOut = new int[half];
            float step = 2f * residueRange / bookEntries;
            int halfEntries = bookEntries / 2;
            for (int i = 0; i < half; i++)
            {
                float floor = floorCurve[i] < 1e-12f ? 1e-12f : floorCurve[i];
                float r = spectrum[i] / floor;
                int idx = (int)Math.Round(r / step) + halfEntries;
                if (idx < 0) idx = 0;
                if (idx >= bookEntries) idx = bookEntries - 1;
                cpuOut[i] = idx;
            }

            // GPU.
            using var dSpectrum = acc.Allocate1D<float>(half);
            using var dFloor = acc.Allocate1D<float>(half);
            using var dOut = acc.Allocate1D<int>(half);
            dSpectrum.View.CopyFromCPU(spectrum);
            dFloor.View.CopyFromCPU(floorCurve);
            dOut.View.CopyFromCPU(new int[half]);

            var kernel = acc.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>,
                int, float, int>(DivideQuantizeKernel);
            kernel(new Index1D(half), dSpectrum.View, dFloor.View, dOut.View,
                half, residueRange, bookEntries);
            await acc.SynchronizeAsync();

            var gpuOut = await dOut.CopyToHostAsync();
            for (int i = 0; i < half; i++)
                if (cpuOut[i] != gpuOut[i])
                    throw new Exception($"residueQ[{i}]: cpu={cpuOut[i]} gpu={gpuOut[i]} (s={spectrum[i]} f={floorCurve[i]})");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static void DivideQuantizeKernel(
        Index1D idx,
        ArrayView<float> spectrum, ArrayView<float> floor, ArrayView<int> output,
        int count, float residueRange, int bookEntries)
    {
        if (idx >= count) return;
        VorbisEncoderHelpersGpu.DivideQuantizeAt(
            spectrum, 0, floor, 0, output, 0,
            idx, residueRange, bookEntries);
    }
}
