// Tests for Vp8BoolEncoderGpu - GPU-resident VP8 boolean range
// encoder. Critical foundation: the GPU output bytes MUST match the
// CPU Vp8BoolEncoder bit-for-bit for any input sequence, otherwise
// downstream entropy stages won't produce a decodable bitstream.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task Vp8BoolEncoderGpu_FixedSequence_MatchesCpu()
    {
        // Hand-picked sequence covering: 50/50 prob, very-skewed-toward-0,
        // very-skewed-toward-1, value boundary cases. Any drift between
        // GPU and CPU bool coder shows up in the output bytes.
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp8BoolEncoderTestKernel(acc);

            int[] bitsList = new[]
            {
                0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0,
                1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0,
                0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0,
                1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0,
            };
            int[] probsList = new[]
            {
                128, 128, 128, 128, 128, 128, 128, 128,
                10, 10, 10, 10, 10, 10, 10, 10,
                245, 245, 245, 245, 245, 245, 245, 245,
                64, 64, 192, 192, 64, 64, 192, 192,
                1, 255, 1, 255, 100, 200, 50, 250,
                128, 128, 128, 128, 128, 128, 128, 128,
                64, 96, 128, 160, 192, 224, 32, 16,
                128, 250, 5, 200, 50, 100, 150, 100,
            };
            int streamCount = 1;
            int bitsPerStream = bitsList.Length;
            int outBufStride = 256;

            // CPU reference.
            var cpuEnc = new Vp8BoolEncoder();
            for (int b = 0; b < bitsPerStream; b++) cpuEnc.EncodeBool(bitsList[b], probsList[b]);
            byte[] cpuBytes = cpuEnc.Stop();

            // GPU encoded.
            var bitsBytes = new byte[bitsPerStream];
            var probsBytes = new byte[bitsPerStream];
            for (int b = 0; b < bitsPerStream; b++)
            {
                bitsBytes[b] = (byte)bitsList[b];
                probsBytes[b] = (byte)probsList[b];
            }

            using var dBits = acc.Allocate1D<byte>(bitsPerStream);
            using var dProbs = acc.Allocate1D<byte>(bitsPerStream);
            using var dOut = acc.Allocate1D<byte>(streamCount * outBufStride);
            using var dLens = acc.Allocate1D<long>(streamCount);
            dBits.View.CopyFromCPU(bitsBytes);
            dProbs.View.CopyFromCPU(probsBytes);
            dOut.View.MemSetToZero();
            kernel.Run(dBits.View, dProbs.View, dOut.View, outBufStride, dLens.View,
                streamCount, bitsPerStream);
            await acc.SynchronizeAsync();

            var lensBack = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dLens);
            long gpuLen = lensBack[0];

            // Compare. We could keep this on GPU but the lengths and
            // bytes are tiny (256 bytes max) and this is the foundation
            // verification - read them back so we can produce a clear
            // mismatch diagnostic if it fails.
            var outBack = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            var gpuBytes = new byte[gpuLen];
            outBack.AsSpan(0, (int)gpuLen).CopyTo(gpuBytes);

            Equal((long)cpuBytes.Length, gpuLen, "byte count");
            int mismatches = 0;
            int firstBad = -1;
            for (int i = 0; i < cpuBytes.Length; i++)
            {
                if (cpuBytes[i] != gpuBytes[i])
                {
                    mismatches++;
                    if (firstBad < 0) firstBad = i;
                }
            }
            Equal(0, mismatches, $"first byte mismatch at i={firstBad}: cpu=0x{(firstBad >= 0 ? cpuBytes[firstBad].ToString("X2") : "")}, gpu=0x{(firstBad >= 0 ? gpuBytes[firstBad].ToString("X2") : "")}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp8BoolEncoderGpu_RandomSequence_MatchesCpu()
    {
        // Random sequence stress test. 1024 bits with random probabilities
        // covering the full [1, 255] range.
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            using var kernel = new Vp8BoolEncoderTestKernel(acc);
            const int bitsPerStream = 1024;
            const int outBufStride = 4096;
            const int streamCount = 4;

            var rng = new Random(0xB001);
            var bits = new byte[streamCount * bitsPerStream];
            var probs = new byte[streamCount * bitsPerStream];
            for (int i = 0; i < bits.Length; i++) bits[i] = (byte)(rng.Next(2));
            for (int i = 0; i < probs.Length; i++) probs[i] = (byte)(1 + rng.Next(255));

            // CPU reference per stream.
            var cpuOuts = new byte[streamCount][];
            for (int s = 0; s < streamCount; s++)
            {
                var enc = new Vp8BoolEncoder();
                for (int b = 0; b < bitsPerStream; b++)
                    enc.EncodeBool(bits[s * bitsPerStream + b], probs[s * bitsPerStream + b]);
                cpuOuts[s] = enc.Stop();
            }

            // GPU encode.
            using var dBits = acc.Allocate1D<byte>(bits.Length);
            using var dProbs = acc.Allocate1D<byte>(probs.Length);
            using var dOut = acc.Allocate1D<byte>(streamCount * outBufStride);
            using var dLens = acc.Allocate1D<long>(streamCount);
            dBits.View.CopyFromCPU(bits);
            dProbs.View.CopyFromCPU(probs);
            dOut.View.MemSetToZero();
            kernel.Run(dBits.View, dProbs.View, dOut.View, outBufStride, dLens.View,
                streamCount, bitsPerStream);
            await acc.SynchronizeAsync();

            var lensBack = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dLens);
            var outBack = await SpawnDev.ILGPU.SpawnDevContextExtensions.CopyToHostAsync(dOut);
            int totalMismatches = 0;
            for (int s = 0; s < streamCount; s++)
            {
                long gpuLen = lensBack[s];
                Equal((long)cpuOuts[s].Length, gpuLen, $"stream {s} length");
                for (int i = 0; i < cpuOuts[s].Length; i++)
                {
                    if (cpuOuts[s][i] != outBack[s * outBufStride + i]) totalMismatches++;
                }
            }
            Equal(0, totalMismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
