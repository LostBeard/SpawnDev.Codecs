// Cross-backend tests for VorbisSpectrumPeakGpu.ComputeHalfBandPeaks.
// Verifies the GPU half-band peak reducer matches the CPU reference
// loops in VorbisAudioEncoder.EncodeAudioPacket bit-exactly.

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
    public async Task VorbisSpectrumPeakGpu_RandomSpectrum_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            const int halfBlock = 512;
            var rng = new Random(unchecked((int)0xA1F0A8u));
            var spectrum = new float[halfBlock];
            for (int i = 0; i < halfBlock; i++)
                spectrum[i] = (float)(rng.NextDouble() * 2 - 1);

            // Inject specific extrema so we know the peaks exactly.
            spectrum[33] = 0.95f;
            spectrum[200] = -0.99f;
            spectrum[300] = 1.5f;
            spectrum[490] = -2.0f;

            // CPU reference.
            int split = halfBlock >> 1;
            float cpuLow = 0, cpuHigh = 0;
            for (int i = 0; i < split; i++) { float a = MathF.Abs(spectrum[i]); if (a > cpuLow) cpuLow = a; }
            for (int i = split; i < halfBlock; i++) { float a = MathF.Abs(spectrum[i]); if (a > cpuHigh) cpuHigh = a; }

            // GPU.
            using var dSpectrum = acc.Allocate1D<float>(halfBlock);
            using var dPeaks = acc.Allocate1D<float>(2);
            dSpectrum.View.CopyFromCPU(spectrum);
            dPeaks.View.CopyFromCPU(new float[2]);
            var kernel = acc.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int>(PeakKernel);
            kernel(new Index1D(1), dSpectrum.View, dPeaks.View, halfBlock);
            await acc.SynchronizeAsync();

            var gpuPeaks = await dPeaks.CopyToHostAsync();
            if (cpuLow != gpuPeaks[0]) throw new Exception($"low peak: cpu={cpuLow} gpu={gpuPeaks[0]}");
            if (cpuHigh != gpuPeaks[1]) throw new Exception($"high peak: cpu={cpuHigh} gpu={gpuPeaks[1]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task VorbisSpectrumPeakGpu_AllZero_ProducesZeroPeaks()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            const int halfBlock = 256;
            var spectrum = new float[halfBlock];

            using var dSpectrum = acc.Allocate1D<float>(halfBlock);
            using var dPeaks = acc.Allocate1D<float>(2);
            dSpectrum.View.CopyFromCPU(spectrum);
            dPeaks.View.CopyFromCPU(new float[2]);
            var kernel = acc.LoadAutoGroupedStreamKernel<
                Index1D, ArrayView<float>, ArrayView<float>, int>(PeakKernel);
            kernel(new Index1D(1), dSpectrum.View, dPeaks.View, halfBlock);
            await acc.SynchronizeAsync();

            var gpuPeaks = await dPeaks.CopyToHostAsync();
            if (gpuPeaks[0] != 0f) throw new Exception($"all-zero low peak: {gpuPeaks[0]}");
            if (gpuPeaks[1] != 0f) throw new Exception($"all-zero high peak: {gpuPeaks[1]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static void PeakKernel(
        Index1D _, ArrayView<float> spectrum, ArrayView<float> peaks, int halfBlock)
    {
        VorbisSpectrumPeakGpu.ComputeHalfBandPeaks(spectrum, 0, halfBlock, peaks, 0);
    }
}
