// Cross-backend tests for the AV1 range coder GPU port. Verifies
// Av1RangeEncoderGpu + Av1RangeDecoderGpu round-trip on the
// accelerator AND that the encoder produces byte-exact output to
// the CPU Av1RangeEncoder reference.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] Av1EncodeCpu(int[] bits, uint[] probs)
    {
        var enc = new Av1RangeEncoder();
        for (int i = 0; i < bits.Length; i++)
            enc.EncodeBoolQ15(bits[i], probs[i]);
        return enc.Done();
    }

    private static async Task<(byte[] gpuBytes, int[] decodedBits)> Av1RoundTripGpuAsync(
        Accelerator acc, int[] bits, uint[] probs)
    {
        int n = bits.Length;
        // Worst case: each bit emits at most ~2 bytes via Normalize. Add slack.
        int scratchLen = Math.Max(64, n * 2 + 64);

        using var dBits = acc.Allocate1D<int>(n);
        using var dProbs = acc.Allocate1D<uint>(n);
        using var dDecoded = acc.Allocate1D<int>(n);
        using var dScratch = acc.Allocate1D<byte>(scratchLen);
        using var dOutLen = acc.Allocate1D<long>(1);

        dBits.View.CopyFromCPU(bits);
        dProbs.View.CopyFromCPU(probs);
        dScratch.View.CopyFromCPU(new byte[scratchLen]);

        using var kernel = new Av1RangeCoderRoundTripKernel(acc);
        kernel.Run(dBits.View, dProbs.View, dDecoded.View, dScratch.View, dOutLen.View, n);
        await acc.SynchronizeAsync();

        long outLen = (await dOutLen.CopyToHostAsync())[0];
        var fullBytes = await dScratch.CopyToHostAsync();
        var bytes = new byte[outLen];
        Array.Copy(fullBytes, bytes, outLen);
        var decoded = await dDecoded.CopyToHostAsync();
        var decodedSlice = new int[n];
        Array.Copy(decoded, decodedSlice, n);
        return (bytes, decodedSlice);
    }

    [TestMethod]
    public async Task Av1RangeCoderGpu_RoundTrip_AllZeros_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            int n = 64;
            var bits = new int[n];
            var probs = new uint[n];
            for (int i = 0; i < n; i++) probs[i] = 16384u;

            var cpuBytes = Av1EncodeCpu(bits, probs);
            var (gpuBytes, decoded) = await Av1RoundTripGpuAsync(acc, bits, probs);

            // Encoder bit-exact vs CPU.
            Equal(cpuBytes.Length, gpuBytes.Length);
            for (int i = 0; i < cpuBytes.Length; i++)
                if (cpuBytes[i] != gpuBytes[i])
                    throw new Exception($"encoded byte mismatch at {i}: cpu=0x{cpuBytes[i]:X2} gpu=0x{gpuBytes[i]:X2}");

            // Round-trip decoded bits match input.
            for (int i = 0; i < n; i++)
                if (bits[i] != decoded[i])
                    throw new Exception($"decoded bit mismatch at {i}: input={bits[i]} decoded={decoded[i]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1RangeCoderGpu_RoundTrip_RandomBits_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            int n = 256;
            var rng = new Random(unchecked((int)0xAB1CDE01u));
            var bits = new int[n];
            var probs = new uint[n];
            for (int i = 0; i < n; i++)
            {
                bits[i] = rng.Next(2);
                // Random q15 prob in (0, 32768)
                probs[i] = (uint)rng.Next(1, 32768);
            }

            var cpuBytes = Av1EncodeCpu(bits, probs);
            var (gpuBytes, decoded) = await Av1RoundTripGpuAsync(acc, bits, probs);

            Equal(cpuBytes.Length, gpuBytes.Length);
            for (int i = 0; i < cpuBytes.Length; i++)
                if (cpuBytes[i] != gpuBytes[i])
                    throw new Exception($"encoded byte mismatch at {i}: cpu=0x{cpuBytes[i]:X2} gpu=0x{gpuBytes[i]:X2}");

            for (int i = 0; i < n; i++)
                if (bits[i] != decoded[i])
                    throw new Exception($"decoded bit mismatch at {i}: input={bits[i]} decoded={decoded[i]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1RangeCoderGpu_RoundTrip_ExtremeProbs_MatchesCpu()
    {
        // Probabilities near the q15 boundaries (1 and 32767) push the
        // range coder into its narrow-range regime where carry
        // propagation gets exercised heavily.
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            int n = 64;
            var rng = new Random(unchecked((int)0xAB1CFEFEu));
            var bits = new int[n];
            var probs = new uint[n];
            for (int i = 0; i < n; i++)
            {
                bits[i] = rng.Next(2);
                // Alternate between extremes: 1, 32767, 1, 32767, ...
                probs[i] = (i & 1) == 0 ? 1u : 32767u;
            }

            var cpuBytes = Av1EncodeCpu(bits, probs);
            var (gpuBytes, decoded) = await Av1RoundTripGpuAsync(acc, bits, probs);

            Equal(cpuBytes.Length, gpuBytes.Length);
            for (int i = 0; i < cpuBytes.Length; i++)
                if (cpuBytes[i] != gpuBytes[i])
                    throw new Exception($"encoded byte mismatch at {i}: cpu=0x{cpuBytes[i]:X2} gpu=0x{gpuBytes[i]:X2}");

            for (int i = 0; i < n; i++)
                if (bits[i] != decoded[i])
                    throw new Exception($"decoded bit mismatch at {i}: input={bits[i]} decoded={decoded[i]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
