// Cross-backend end-to-end tests for FlacEncoderGpu. Encodes a PCM
// stream entirely on the GPU and verifies the resulting .flac bytes
// can be parsed by the existing CPU FLAC decoder back to the
// original samples.

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
    public async Task FlacEncoderGpu_Mono_SinglFrame_RoundTripsViaCpuDecoder()
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

            // Verify "fLaC" marker at start.
            Equal((byte)'f', bytes[0], "marker[0]");
            Equal((byte)'L', bytes[1], "marker[1]");
            Equal((byte)'a', bytes[2], "marker[2]");
            Equal((byte)'C', bytes[3], "marker[3]");

            // Decode via the existing CPU decoder + compare samples.
            var decoded = SpawnDev.Codecs.Audio.Flac.FlacDecoder.Decode(bytes);
            Equal(channels, decoded.StreamInfo.Channels, "channels");
            Equal(FlacEncoderGpu.BitsPerSample, decoded.StreamInfo.BitsPerSample, "bps");
            Equal(FlacEncoderGpu.SampleRateHz, decoded.StreamInfo.SampleRateHz, "sample rate");
            Equal(blockSize, decoded.InterleavedSamples.Length / channels, "samples per channel");

            for (int i = 0; i < blockSize; i++)
                if (samples[i] != decoded.InterleavedSamples[i])
                    throw new Exception($"sample[{i}]: input={samples[i]} decoded={decoded.InterleavedSamples[i]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task FlacEncoderGpu_Stereo_MultiFrame_RoundTripsViaCpuDecoder()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            const int blockSize = FlacEncoderGpu.BlockSize;
            const int channels = 2;
            const int frameCount = 3;
            int totalPerChannel = blockSize * frameCount;

            var rng = new Random(unchecked((int)0xF1AC57E0u));
            var samples = new int[totalPerChannel * channels];
            for (int i = 0; i < samples.Length; i++) samples[i] = rng.Next(-32768, 32768);

            using var enc = new FlacEncoderGpu(acc);
            var bytes = await enc.EncodeStreamAsync(samples, channels);

            var decoded = SpawnDev.Codecs.Audio.Flac.FlacDecoder.Decode(bytes);
            Equal(channels, decoded.StreamInfo.Channels);
            Equal(FlacEncoderGpu.BitsPerSample, decoded.StreamInfo.BitsPerSample);
            Equal(FlacEncoderGpu.SampleRateHz, decoded.StreamInfo.SampleRateHz);
            Equal(samples.Length, decoded.InterleavedSamples.Length);

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
