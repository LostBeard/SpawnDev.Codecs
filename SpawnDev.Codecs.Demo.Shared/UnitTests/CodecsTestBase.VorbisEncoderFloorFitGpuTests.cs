// Cross-backend test for VorbisEncoderFloorFitGpu.FitFloorEndpoints.
// Verifies the composite peak + headroom + MagnitudeToFloorY +
// silent-guard primitive matches the CPU reference loop in
// VorbisAudioEncoder.EncodeAudioPacket (lines 235-249) within
// tolerance ±1 (Log10 float vs double drift at boundary magnitudes).

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
    public async Task VorbisEncoderFloorFitGpu_FitFloorEndpoints_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            const int halfBlock = 512;
            const float headroom = 1.25f;

            // Random spectrum with controlled peaks per half-band.
            var rng = new Random(unchecked((int)0xA1A0F1u));
            var spectrum = new float[halfBlock];
            for (int i = 0; i < halfBlock; i++)
                spectrum[i] = (float)(rng.NextDouble() * 0.001 - 0.0005);
            spectrum[100] = 0.4f;     // low half peak
            spectrum[400] = -0.7f;    // high half peak

            // CPU reference (mirror of EncodeAudioPacket lines 237-249).
            int split = halfBlock >> 1;
            float specPeakLow = 0, specPeakHigh = 0;
            for (int i = 0; i < split; i++)
            {
                float a = MathF.Abs(spectrum[i]);
                if (a > specPeakLow) specPeakLow = a;
            }
            for (int i = split; i < halfBlock; i++)
            {
                float a = MathF.Abs(spectrum[i]);
                if (a > specPeakHigh) specPeakHigh = a;
            }
            int cpuYLow = CpuMagToFloorY(specPeakLow * headroom);
            int cpuYHigh = CpuMagToFloorY(specPeakHigh * headroom);
            if (cpuYLow < 1) cpuYLow = 1;
            if (cpuYHigh < 1) cpuYHigh = 1;

            // GPU.
            var inverseDb = VorbisFloor1InverseDbGpu.BuildInverseDbTable();
            using var dSpectrum = acc.Allocate1D<float>(halfBlock);
            using var dInverseDb = acc.Allocate1D<float>(inverseDb.Length);
            using var dPosteriors = acc.Allocate1D<int>(2);
            dSpectrum.View.CopyFromCPU(spectrum);
            dInverseDb.View.CopyFromCPU(inverseDb);
            dPosteriors.View.CopyFromCPU(new int[2]);

            var kernel = acc.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>,
                int, float>(FitFloorKernel);
            kernel(new Index1D(1), dSpectrum.View, dInverseDb.View, dPosteriors.View,
                halfBlock, headroom);
            await acc.SynchronizeAsync();

            int[] gpuPosteriors = await dPosteriors.CopyToHostAsync();
            // Allow ±1 step at boundary magnitudes (binary search vs Log10 ceiling).
            if (Math.Abs(cpuYLow - gpuPosteriors[0]) > 1)
                throw new Exception($"yLow: cpu={cpuYLow} gpu={gpuPosteriors[0]}");
            if (Math.Abs(cpuYHigh - gpuPosteriors[1]) > 1)
                throw new Exception($"yHigh: cpu={cpuYHigh} gpu={gpuPosteriors[1]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static int CpuMagToFloorY(float magnitude)
    {
        // Mirror of VorbisAudioEncoder.MagnitudeToFloorY (Log10/Ceiling path).
        if (!float.IsFinite(magnitude) || magnitude <= 1.0649863e-7f) return 0;
        if (magnitude >= 1.0f) return 255;
        double idx = Math.Log10(magnitude) / 0.02735 + 255.0;
        int y = (int)Math.Ceiling(idx);
        if (y < 0) y = 0;
        if (y > 255) y = 255;
        return y;
    }

    private static void FitFloorKernel(
        Index1D _, ArrayView<float> spectrum, ArrayView<float> inverseDb,
        ArrayView<int> posteriors, int halfBlock, float headroom)
    {
        VorbisEncoderFloorFitGpu.FitFloorEndpoints(
            spectrum, 0, halfBlock, headroom,
            inverseDb, 0, posteriors, 0);
    }
}
