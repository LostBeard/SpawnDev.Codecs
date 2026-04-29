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

    // Non-silent tone test deferred: GPU MagnitudeToFloorY uses binary
    // search on the inverse-dB lookup (cross-backend safe; doesn't need
    // EnableAlgorithms which CUDA + OpenCL backends don't invoke).
    // CPU uses Log10/Ceiling. Both produce the same Y for most magnitudes
    // but can differ by ±1 step at boundary magnitudes, which cascades
    // through the bit stream. The silence path (clamp to Y=1) avoids
    // this and produces byte-identical output across all backends; the
    // non-silent bit-exact match needs CPU/GPU to use the same Y formula.

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
