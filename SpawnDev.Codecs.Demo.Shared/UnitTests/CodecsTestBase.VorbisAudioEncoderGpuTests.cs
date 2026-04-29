// Cross-backend test for VorbisAudioEncoderGpu - the v1 mono Vorbis
// audio packet encoder integration class. Verifies that the 6-kernel
// chain produces byte-identical output to VorbisAudioEncoder.EncodeAudioPacket
// for a given block of mono PCM input.

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
    public async Task VorbisAudioEncoderGpu_OnePacketSilence_MatchesCpu()
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

            // Silence: all-zero PCM (worst-case for residue codebook -
            // every bin should quantize to entry 512 = N/2).
            var pcm = new float[options.BlockSize];
            await EncodeAndCompare(acc, options, pcm);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    // Non-silent (tone, random PCM) bit-exact match deferred. CUDA +
    // OpenCL ILGPU backends don't support XMath.Cos(double) /
    // XMath.Log10(double) without EnableAlgorithms, so the GPU encoder
    // can't bit-exactly mirror the CPU MdctReference (double accumulator).
    // Float-precision MDCT produces acoustically identical decoded PCM
    // but a few bitstream bytes can diverge at floor-Y boundaries.
    // The silence path produces byte-identical output regardless.
    // Resolution path: configure ILGPU contexts to call
    // EnableAlgorithms() so XMath double-precision intrinsics are
    // available (test factory change); follow-up.

    private static async Task EncodeAndCompare(
        Accelerator acc, VorbisAudioEncoderOptions options, float[] pcm)
    {
        // CPU reference (internal method, accessible via shared-test InternalsVisibleTo).
        var cpu = new VorbisAudioEncoder(options);
        byte[] cpuBytes = cpu.EncodeAudioPacket(pcm);

        // GPU.
        using var gpu = new VorbisAudioEncoderGpu(acc, options);
        byte[] gpuBytes = await gpu.EncodeAudioPacketAsync(pcm);

        if (cpuBytes.Length != gpuBytes.Length)
            throw new Exception($"packet len: cpu={cpuBytes.Length} gpu={gpuBytes.Length}");
        for (int i = 0; i < cpuBytes.Length; i++)
            if (cpuBytes[i] != gpuBytes[i])
                throw new Exception($"byte[{i}]: cpu={cpuBytes[i]:X2} gpu={gpuBytes[i]:X2}");
    }
}
