// Cross-backend test for VorbisAudioDecoderGpu - the v1 mono Vorbis
// audio packet decoder integration class. Pairs with
// VorbisAudioEncoderGpu to close the v1 Vorbis encoder/decoder pair.
//
// Strategy: encode silence packets via VorbisAudioEncoderGpu, then
// decode them via BOTH the CPU reference VorbisAudioDecoder AND the
// GPU VorbisAudioDecoderGpu using the same packet bytes. Compare the
// decoded PCM sample-by-sample with a float tolerance (the post-IMDCT
// chain runs through GPU FMA reordering vs the CPU's separate
// multiply+add, so samples can drift at the ~1e-5 level - same
// tolerance the existing VorbisPostImdctGpu tests use).

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
    public async Task VorbisAudioDecoderGpu_SilenceRoundTrip_MatchesCpuDecoder()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var options = new VorbisAudioEncoderOptions
            {
                SampleRateHz = 44100,
                Channels = 1,
                BlockSize = 1024,
            };

            // Encode a few packets of silence so the decoder's overlap-add
            // state is exercised across packet boundaries (first packet
            // returns empty; subsequent packets produce halfBlock samples).
            using var encoder = new VorbisAudioEncoderGpu(acc, options);
            byte[] packet1 = await encoder.EncodeAudioPacketAsync(new float[options.BlockSize]);
            byte[] packet2 = await encoder.EncodeAudioPacketAsync(new float[options.BlockSize]);
            byte[] packet3 = await encoder.EncodeAudioPacketAsync(new float[options.BlockSize]);

            // CPU reference decode of the same packets.
            var cpuDec = new VorbisAudioDecoder(encoder.Identification, encoder.Setup);
            int halfBlock = options.BlockSize / 2;
            var cpuOut1 = new float[halfBlock * options.Channels];
            int cpuFrames1 = cpuDec.DecodePacket(packet1, cpuOut1);
            var cpuOut2 = new float[halfBlock * options.Channels];
            int cpuFrames2 = cpuDec.DecodePacket(packet2, cpuOut2);
            var cpuOut3 = new float[halfBlock * options.Channels];
            int cpuFrames3 = cpuDec.DecodePacket(packet3, cpuOut3);

            // GPU decode of the same packets.
            using var gpuDec = new VorbisAudioDecoderGpu(acc, encoder.Identification, encoder.Setup);
            float[] gpuOut1 = await gpuDec.DecodePacketAsync(packet1);
            float[] gpuOut2 = await gpuDec.DecodePacketAsync(packet2);
            float[] gpuOut3 = await gpuDec.DecodePacketAsync(packet3);

            // Packet 1 primes the overlap-add buffer; both decoders return zero frames.
            if (cpuFrames1 != 0)
                throw new Exception($"CPU expected 0 frames on first packet, got {cpuFrames1}");
            if (gpuOut1.Length != 0)
                throw new Exception($"GPU expected 0 samples on first packet, got {gpuOut1.Length}");

            // Packets 2 + 3: must match within tolerance.
            CompareSamples(cpuOut2, gpuOut2, cpuFrames2 * options.Channels, "packet2");
            CompareSamples(cpuOut3, gpuOut3, cpuFrames3 * options.Channels, "packet3");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task VorbisAudioDecoderGpu_SilenceOutput_IsSilent()
    {
        // Sanity: silence in -> silence out (sample magnitudes near zero).
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var options = new VorbisAudioEncoderOptions
            {
                SampleRateHz = 44100,
                Channels = 1,
                BlockSize = 1024,
            };

            using var encoder = new VorbisAudioEncoderGpu(acc, options);
            byte[] packet1 = await encoder.EncodeAudioPacketAsync(new float[options.BlockSize]);
            byte[] packet2 = await encoder.EncodeAudioPacketAsync(new float[options.BlockSize]);

            using var gpuDec = new VorbisAudioDecoderGpu(acc, encoder.Identification, encoder.Setup);
            _ = await gpuDec.DecodePacketAsync(packet1);
            float[] gpuOut2 = await gpuDec.DecodePacketAsync(packet2);

            // Silence-path samples should be tiny (residue dequant + IMDCT
            // cannot produce anything but ~zero since residue codebook
            // entries decode to 0 for silent input).
            const float kSilenceTol = 1e-3f;
            for (int i = 0; i < gpuOut2.Length; i++)
            {
                float a = gpuOut2[i];
                if (a < 0) a = -a;
                if (a > kSilenceTol)
                    throw new Exception($"sample[{i}] not silent: {gpuOut2[i]:R}");
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task VorbisAudioDecoderGpu_ToneRoundTrip_MatchesCpuDecoder()
    {
        // Non-silence: 440 Hz mono tone. Same packets feed BOTH the CPU
        // VorbisAudioDecoder reference AND the GPU VorbisAudioDecoderGpu.
        // Output drift comes from the IMDCT float (GPU) vs double (CPU)
        // accumulator difference - the test allows ~1e-3 absolute on
        // sample magnitudes that peak near 0.5.
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var options = new VorbisAudioEncoderOptions
            {
                SampleRateHz = 44100,
                Channels = 1,
                BlockSize = 1024,
            };

            // Generate 4 blocks of a 440 Hz tone at 0.5 amplitude.
            int totalSamples = options.BlockSize * 4;
            var pcm = new float[totalSamples];
            double w = 2.0 * Math.PI * 440.0 / options.SampleRateHz;
            for (int i = 0; i < totalSamples; i++)
                pcm[i] = 0.5f * (float)Math.Sin(w * i);

            using var encoder = new VorbisAudioEncoderGpu(acc, options);
            byte[] packet1 = await encoder.EncodeAudioPacketAsync(
                pcm.AsMemory(0, options.BlockSize).ToArray());
            byte[] packet2 = await encoder.EncodeAudioPacketAsync(
                pcm.AsMemory(options.BlockSize, options.BlockSize).ToArray());
            byte[] packet3 = await encoder.EncodeAudioPacketAsync(
                pcm.AsMemory(2 * options.BlockSize, options.BlockSize).ToArray());

            // CPU reference decode.
            var cpuDec = new VorbisAudioDecoder(encoder.Identification, encoder.Setup);
            int halfBlock = options.BlockSize / 2;
            var cpuOut1 = new float[halfBlock * options.Channels];
            cpuDec.DecodePacket(packet1, cpuOut1);
            var cpuOut2 = new float[halfBlock * options.Channels];
            int cpuFrames2 = cpuDec.DecodePacket(packet2, cpuOut2);
            var cpuOut3 = new float[halfBlock * options.Channels];
            int cpuFrames3 = cpuDec.DecodePacket(packet3, cpuOut3);

            // GPU decode.
            using var gpuDec = new VorbisAudioDecoderGpu(
                acc, encoder.Identification, encoder.Setup);
            _ = await gpuDec.DecodePacketAsync(packet1);
            float[] gpuOut2 = await gpuDec.DecodePacketAsync(packet2);
            float[] gpuOut3 = await gpuDec.DecodePacketAsync(packet3);

            CompareSamples(cpuOut2, gpuOut2, cpuFrames2 * options.Channels,
                "tone-packet2", absTol: 1e-3f);
            CompareSamples(cpuOut3, gpuOut3, cpuFrames3 * options.Channels,
                "tone-packet3", absTol: 1e-3f);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static void CompareSamples(
        float[] expected, float[] actual, int count, string context,
        float absTol = 1e-5f)
    {
        if (actual.Length < count)
            throw new Exception($"[{context}] GPU buffer too short: {actual.Length} < {count}");
        for (int i = 0; i < count; i++)
        {
            float diff = expected[i] - actual[i];
            if (diff < 0) diff = -diff;
            if (diff > absTol)
                throw new Exception($"[{context}] sample[{i}]: cpu={expected[i]:R} gpu={actual[i]:R} diff={diff:R}");
        }
    }
}
