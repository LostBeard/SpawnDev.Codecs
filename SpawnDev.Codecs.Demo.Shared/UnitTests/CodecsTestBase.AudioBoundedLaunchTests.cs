// Audio kernels in bounded launches: inputs long enough that the resumable / chunked paths take several launches
// (resume points crossed), checked against the CPU references (or, for the transforms, against single-block runs:
// the launch split must not change a single bit).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.Codecs.Audio.Transforms;
using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Category = "NotOnWebGL")]
    public async Task OpusRangeDecoderGpu_DecodeIcdf_SpansLaunches_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            int n = 3 * OpusRangeDecoderGpuTestKernel.StepSymbols + 17;
            byte[] icdf = OpusTestIcdf_Uniform4;
            var rng = new Random(0x0B05);
            var symbols = new int[n];
            for (int i = 0; i < n; i++) symbols[i] = rng.Next(4);
            byte[] encoded = OpusEncodeIcdfSequenceCpu(symbols, icdf, 8, n);
            int[] gpu = await OpusDecodeIcdfSequenceGpuAsync(acc, encoded, icdf, 8, n);
            for (int i = 0; i < n; i++)
                if (symbols[i] != gpu[i])
                    throw new Exception($"symbol[{i}] (launch {i / OpusRangeDecoderGpuTestKernel.StepSymbols}): cpu={symbols[i]} gpu={gpu[i]}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Category = "NotOnWebGL")]
    public async Task FlacBitRoundTripGpu_SpansLaunches_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            var (values, bits) = BoundedBitSequence(2 * FlacBitRoundTripKernel.StepSymbols + 5, 0xF1AC);
            byte[] cpu = FlacBitEncodeCpu(values, bits);
            var (gpuBytes, decoded) = await FlacBitRoundTripGpuAsync(acc, values, bits);
            Equal(cpu.Length, gpuBytes.Length, "byte count");
            for (int i = 0; i < cpu.Length; i++)
                if (cpu[i] != gpuBytes[i]) throw new Exception($"byte {i}: cpu={cpu[i]} gpu={gpuBytes[i]}");
            for (int i = 0; i < values.Length; i++)
                if (values[i] != decoded[i]) throw new Exception($"value {i}: in={values[i]} out={decoded[i]}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Category = "NotOnWebGL")]
    public async Task VorbisBitRoundTripGpu_SpansLaunches_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            var (values, bits) = BoundedBitSequence(2 * VorbisBitRoundTripKernel.StepSymbols + 5, 0x0B15);
            byte[] cpu = VorbisBitEncodeCpu(values, bits);
            var (gpuBytes, decoded) = await VorbisBitRoundTripGpuAsync(acc, values, bits);
            Equal(cpu.Length, gpuBytes.Length, "byte count");
            for (int i = 0; i < cpu.Length; i++)
                if (cpu[i] != gpuBytes[i]) throw new Exception($"byte {i}: cpu={cpu[i]} gpu={gpuBytes[i]}");
            for (int i = 0; i < values.Length; i++)
                if (values[i] != decoded[i]) throw new Exception($"value {i}: in={values[i]} out={decoded[i]}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>Random (value, width) pairs, widths 1..32, values masked to their width.</summary>
    private static (uint[] values, int[] bits) BoundedBitSequence(int n, int seed)
    {
        var rng = new Random(seed);
        var values = new uint[n];
        var bits = new int[n];
        for (int i = 0; i < n; i++)
        {
            bits[i] = 1 + rng.Next(32);
            uint v = (uint)rng.Next() ^ ((uint)rng.Next() << 16);
            values[i] = bits[i] == 32 ? v : v & ((1u << bits[i]) - 1u);
        }
        return (values, bits);
    }

    [TestMethod]
    public async Task MdctKernel_SpansLaunches_BitIdenticalToSingleBlockRuns()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            const int n = 1024;
            int blocks = MdctKernel.BlocksPerLaunch(n) + 8; // two launches
            var rng = new Random(0x3D);
            var input = new float[(long)blocks * 2 * n];
            for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
            using var kernel = new MdctKernel(acc);
            var all = new float[(long)blocks * n];
            await kernel.RunAsync(input, all, blocks, n);
            foreach (int b in new[] { 0, MdctKernel.BlocksPerLaunch(n) - 1, MdctKernel.BlocksPerLaunch(n), blocks - 1 })
            {
                var one = new float[n];
                await kernel.RunAsync(input.AsMemory(b * 2 * n, 2 * n).ToArray(), one, 1, n);
                for (int k = 0; k < n; k++)
                    if (BitConverter.SingleToInt32Bits(one[k]) != BitConverter.SingleToInt32Bits(all[(long)b * n + k]))
                        throw new Exception($"block {b} coef {k}: single={one[k]:R} batched={all[(long)b * n + k]:R}");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task ImdctKernel_SpansLaunches_BitIdenticalToSingleBlockRuns()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            const int n = 1024;
            int blocks = ImdctKernel.BlocksPerLaunch(n) + 8; // two launches
            var rng = new Random(0x1D);
            var input = new float[(long)blocks * n];
            for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
            using var kernel = new ImdctKernel(acc);
            var all = new float[(long)blocks * 2 * n];
            await kernel.RunAsync(input, all, blocks, n);
            foreach (int b in new[] { 0, ImdctKernel.BlocksPerLaunch(n) - 1, ImdctKernel.BlocksPerLaunch(n), blocks - 1 })
            {
                var one = new float[2 * n];
                await kernel.RunAsync(input.AsMemory(b * n, n).ToArray(), one, 1, n);
                for (int k = 0; k < 2 * n; k++)
                    if (BitConverter.SingleToInt32Bits(one[k]) != BitConverter.SingleToInt32Bits(all[(long)b * 2 * n + k]))
                        throw new Exception($"block {b} sample {k}: single={one[k]:R} batched={all[(long)b * 2 * n + k]:R}");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>
    /// 2100 frames: frame numbers past 2047 take the 3-byte UTF-8 header (longer header, different piece split of
    /// the CRC-16), and the de-interleave / pack / CRC launches all run over many frames.
    /// </summary>
    [TestMethod(Category = "NotOnWebGL")]
    public async Task FlacEncoderGpu_LongStream_RoundTripsViaCpuDecoder()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            const int frames = 2100;
            const int channels = 1;
            var rng = new Random(0x10F1);
            var samples = new int[frames * FlacEncoderGpu.BlockSize * channels];
            for (int i = 0; i < samples.Length; i++) samples[i] = rng.Next(-32768, 32768);
            using var enc = new FlacEncoderGpu(acc);
            var bytes = await enc.EncodeStreamAsync(samples, channels);
            var decoded = SpawnDev.Codecs.Audio.Flac.FlacDecoder.Decode(bytes);
            Equal(samples.Length, decoded.InterleavedSamples.Length, "samples");
            for (int i = 0; i < samples.Length; i++)
                if (samples[i] != decoded.InterleavedSamples[i])
                    throw new Exception($"sample[{i}]: input={samples[i]} decoded={decoded.InterleavedSamples[i]}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>
    /// The parallel frame reader keeps the serial reader's verdicts: a damaged subframe header is status 4, a damaged
    /// sample (CRC-16) is status 3, a damaged header byte (CRC-8) is status 2.
    /// </summary>
    [TestMethod(Category = "NotOnWebGL")]
    public async Task FlacDecoderGpu_DamagedFrames_ReportSerialStatusCodes()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            const int channels = 2;
            var rng = new Random(0xBAD);
            var samples = new int[3 * FlacEncoderGpu.BlockSize * channels];
            for (int i = 0; i < samples.Length; i++) samples[i] = rng.Next(-32768, 32768);
            using var enc = new FlacEncoderGpu(acc);
            var good = await enc.EncodeStreamAsync(samples, channels);
            using var dec = new FlacDecoderGpu(acc);
            var ok = await dec.DecodeStreamAsync(good);
            for (int i = 0; i < samples.Length; i++)
                if (samples[i] != ok.InterleavedSamples[i]) throw new Exception($"sample[{i}] mismatch on the intact stream");

            // Frame 0 starts at byte 42: 5 header bytes (1-byte frame number), CRC-8, then the subframes.
            const int frame0 = 42;
            await ExpectStatusAsync(dec, good, frame0 + 2, 0x10, 2);                // sample-rate bits: CRC-8 mismatch
            await ExpectStatusAsync(dec, good, frame0 + 6, 0x40, 4);                // channel 0 subframe header
            await ExpectStatusAsync(dec, good, frame0 + 6 + 1 + 1000, 0x01, 3);     // a channel 0 sample: CRC-16
            await ExpectStatusAsync(dec, good, frame0 + 6 + 1 + 2 * FlacEncoderGpu.BlockSize, 0x80, 4); // channel 1 header
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    private static async Task ExpectStatusAsync(FlacDecoderGpu dec, byte[] good, int at, byte xor, int status)
    {
        var bad = (byte[])good.Clone();
        bad[at] ^= xor;
        try
        {
            await dec.DecodeStreamAsync(bad);
        }
        catch (InvalidDataException e)
        {
            if (!e.Message.Contains($"status={status})"))
                throw new Exception($"byte {at} ^ 0x{xor:X2}: expected status {status}, got: {e.Message}");
            return;
        }
        throw new Exception($"byte {at} ^ 0x{xor:X2}: expected status {status}, decode succeeded");
    }

    /// <summary>
    /// An 8192-sample block: 4096 residue codewords of 10 bits each are about 50k work units, so the resumable
    /// packet decode spans several launches; the PCM must still match the CPU decoder.
    /// </summary>
    [TestMethod(Category = "NotOnWebGL")]
    public async Task VorbisAudioDecoderGpu_LongBlock_SpansLaunches_MatchesCpuDecoder()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            var options = new VorbisAudioEncoderOptions { SampleRateHz = 44100, Channels = 1, BlockSize = 8192 };
            int n = options.BlockSize;
            var rng = new Random(0x0B8);
            var pcm = new float[n * 3];
            double w = 2.0 * Math.PI * 440.0 / options.SampleRateHz;
            for (int i = 0; i < pcm.Length; i++) pcm[i] = 0.4f * (float)Math.Sin(w * i) + 0.1f * (float)(rng.NextDouble() - 0.5);
            using var encoder = new VorbisAudioEncoderGpu(acc, options);
            var packets = new byte[3][];
            for (int p = 0; p < 3; p++) packets[p] = await encoder.EncodeAudioPacketAsync(pcm.AsMemory(p * n, n).ToArray());

            var cpuDec = new VorbisAudioDecoder(encoder.Identification, encoder.Setup);
            using var gpuDec = new VorbisAudioDecoderGpu(acc, encoder.Identification, encoder.Setup);
            var cpuOut = new float[n / 2];
            cpuDec.DecodePacket(packets[0], cpuOut);
            _ = await gpuDec.DecodePacketAsync(packets[0]);
            for (int p = 1; p < 3; p++)
            {
                int frames = cpuDec.DecodePacket(packets[p], cpuOut);
                float[] gpuOut = await gpuDec.DecodePacketAsync(packets[p]);
                CompareSamples(cpuOut, gpuOut, frames, $"long-block packet {p}", absTol: 2e-3f);
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
