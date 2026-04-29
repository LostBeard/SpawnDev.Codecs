// Cross-backend end-to-end tests for FlacDecoderGpu. Encodes via
// FlacEncoderGpu (also GPU), then decodes via FlacDecoderGpu, then
// verifies decoded samples match input. Demonstrates the FLAC encoder
// + decoder pair both running 100% on the accelerator.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task FlacDecoderGpu_Mono_RoundTrip_GpuEncodeGpuDecode()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            const int blockSize = FlacEncoderGpu.BlockSize;
            const int channels = 1;

            var samples = new int[blockSize];
            for (int i = 0; i < blockSize; i++)
                samples[i] = (int)(Math.Sin(2 * Math.PI * 440 * i / FlacEncoderGpu.SampleRateHz) * 16384);

            using var enc = new FlacEncoderGpu(acc);
            var bytes = await enc.EncodeStreamAsync(samples, channels);

            using var dec = new FlacDecoderGpu(acc);
            var decoded = await dec.DecodeStreamAsync(bytes);

            Equal(channels, decoded.Channels);
            Equal(FlacEncoderGpu.BitsPerSample, decoded.BitsPerSample);
            Equal(FlacEncoderGpu.SampleRateHz, decoded.SampleRateHz);
            Equal(blockSize, decoded.TotalSamplesPerChannel);

            for (int i = 0; i < blockSize; i++)
                if (samples[i] != decoded.InterleavedSamples[i])
                    throw new Exception($"sample[{i}]: input={samples[i]} decoded={decoded.InterleavedSamples[i]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacDecoderGpu_Stereo_MultiFrame_RoundTrip_GpuEncodeGpuDecode()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            const int blockSize = FlacEncoderGpu.BlockSize;
            const int channels = 2;
            const int frameCount = 3;
            int totalPerChannel = blockSize * frameCount;

            var rng = new Random(unchecked((int)0xF1ACD0DEu));
            var samples = new int[totalPerChannel * channels];
            for (int i = 0; i < samples.Length; i++) samples[i] = rng.Next(-32768, 32768);

            using var enc = new FlacEncoderGpu(acc);
            var bytes = await enc.EncodeStreamAsync(samples, channels);

            using var dec = new FlacDecoderGpu(acc);
            var decoded = await dec.DecodeStreamAsync(bytes);

            Equal(channels, decoded.Channels);
            Equal(FlacEncoderGpu.BitsPerSample, decoded.BitsPerSample);
            Equal(FlacEncoderGpu.SampleRateHz, decoded.SampleRateHz);
            Equal(totalPerChannel, decoded.TotalSamplesPerChannel);

            int mismatches = 0;
            int firstMismatch = -1;
            for (int i = 0; i < samples.Length; i++)
                if (samples[i] != decoded.InterleavedSamples[i])
                {
                    if (firstMismatch < 0) firstMismatch = i;
                    mismatches++;
                }
            if (mismatches > 0)
                throw new Exception($"{mismatches} sample mismatches; first at i={firstMismatch} input={samples[firstMismatch]} decoded={decoded.InterleavedSamples[firstMismatch]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
