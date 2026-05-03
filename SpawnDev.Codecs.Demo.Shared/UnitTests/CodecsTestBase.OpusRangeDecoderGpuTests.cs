// Cross-backend tests for the Opus range decoder GPU port. Encodes
// a sequence of icdf symbols on CPU using the libopus reference
// encoder, decodes them via OpusRangeDecoderGpu on the accelerator,
// and verifies bit-exact agreement.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.Codecs.EntropyCoders;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Encode a sequence of icdf symbols using the CPU reference encoder
    /// + return the encoded byte buffer at the requested storage size.
    /// Callers pass the symbol-stream + the icdf table + ftb; output is
    /// the byte buffer ready for `OpusRangeDecoderGpu.Init` consumption.
    /// </summary>
    private static byte[] OpusEncodeIcdfSequenceCpu(
        int[] symbols, byte[] icdf, int ftb, int bufCapacity)
    {
        var enc = new OpusRangeEncoder(bufCapacity);
        for (int i = 0; i < symbols.Length; i++)
            enc.EncodeIcdf(symbols[i], icdf, ftb);
        enc.Done();
        return enc.ToArray();
    }

    private static async Task<int[]> OpusDecodeIcdfSequenceGpuAsync(
        Accelerator acc,
        byte[] packetBytes,
        byte[] icdf,
        int ftb,
        int symbolCount)
    {
        using var dPacket = acc.Allocate1D<byte>(packetBytes.Length);
        using var dIcdf = acc.Allocate1D<byte>(icdf.Length);
        using var dDecoded = acc.Allocate1D<int>(symbolCount);

        dPacket.View.CopyFromCPU(packetBytes);
        dIcdf.View.CopyFromCPU(icdf);

        using var kernel = new OpusRangeDecoderGpuTestKernel(acc);
        kernel.Run(
            dPacket.View, 0, packetBytes.Length,
            dIcdf.View, 0, ftb,
            dDecoded.View, symbolCount);
        await acc.SynchronizeAsync();

        var decoded = await dDecoded.CopyToHostAsync();
        var slice = new int[symbolCount];
        Array.Copy(decoded, slice, symbolCount);
        return slice;
    }

    /// <summary>
    /// SilkIcdfTables.Uniform4 is the 4-symbol uniform table (used for
    /// the SILK seed decode). Values [3, 2, 1, 0]. Each symbol equally
    /// likely; covers the simplest 2-bit decode path.
    /// </summary>
    private static readonly byte[] OpusTestIcdf_Uniform4 = new byte[] { 192, 128, 64, 0 };

    /// <summary>
    /// SilkIcdfTables.TypeOffsetVad is the 4-symbol VAD-on signal-type
    /// table (used by SilkSideInfoDecoder.DecodeSignalType when the VAD
    /// flag is set). Values mapped to (signalType, quantOffsetType)
    /// pairs. Tests a non-uniform iCDF with realistic SILK bit-stream
    /// shape.
    /// </summary>
    private static readonly byte[] OpusTestIcdf_TypeOffsetVad = new byte[] { 232, 158, 10, 0 };

    [TestMethod]
    public async Task OpusRangeDecoderGpu_DecodeIcdf_Uniform4_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Encode a known sequence of 4-symbol uniform values on CPU,
            // decode on GPU, verify bit-exact.
            int[] symbols = new[] { 0, 1, 2, 3, 1, 0, 3, 2, 0, 2, 1, 3 };
            byte[] icdf = OpusTestIcdf_Uniform4;
            const int ftb = 8;

            byte[] encoded = OpusEncodeIcdfSequenceCpu(symbols, icdf, ftb, 256);
            True(encoded.Length > 0,
                "Opus encoder should produce non-empty output");

            int[] gpuDecoded = await OpusDecodeIcdfSequenceGpuAsync(
                acc, encoded, icdf, ftb, symbols.Length);

            // Sanity: CPU decoder also recovers the same sequence.
            var cpuDec = new OpusRangeDecoder(encoded);
            for (int i = 0; i < symbols.Length; i++)
            {
                int cpu = cpuDec.DecodeIcdf(icdf, ftb);
                if (cpu != symbols[i])
                    throw new Exception(
                        $"CPU oracle mismatch at {i}: input={symbols[i]} cpu={cpu}");
            }

            // GPU bit-exact vs input (and therefore vs CPU).
            for (int i = 0; i < symbols.Length; i++)
            {
                if (gpuDecoded[i] != symbols[i])
                    throw new Exception(
                        $"GPU decode mismatch at {i}: input={symbols[i]} gpu={gpuDecoded[i]}");
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task OpusRangeDecoderGpu_DecodeIcdf_TypeOffsetVad_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Mix of all 4 symbols using a realistic SILK iCDF shape.
            int[] symbols = new[] { 1, 2, 0, 3, 1, 1, 2, 0, 3, 1, 2 };
            byte[] icdf = OpusTestIcdf_TypeOffsetVad;
            const int ftb = 8;

            byte[] encoded = OpusEncodeIcdfSequenceCpu(symbols, icdf, ftb, 256);
            True(encoded.Length > 0,
                "Opus encoder should produce non-empty output");

            int[] gpuDecoded = await OpusDecodeIcdfSequenceGpuAsync(
                acc, encoded, icdf, ftb, symbols.Length);

            for (int i = 0; i < symbols.Length; i++)
            {
                if (gpuDecoded[i] != symbols[i])
                    throw new Exception(
                        $"GPU decode mismatch at {i}: input={symbols[i]} gpu={gpuDecoded[i]}");
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task OpusRangeDecoderGpu_DecodeIcdf_LongSequence_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // 256 symbols stresses Normalize byte-refill + buffer pointer
            // walk + ensures multi-byte normalization fires repeatedly.
            const int n = 256;
            int[] symbols = new int[n];
            byte[] icdf = OpusTestIcdf_TypeOffsetVad;
            const int ftb = 8;

            // Deterministic pseudo-random sequence drawn from icdf shape
            // (simple LCG; tests the realistic distribution path).
            uint seed = 0x1234_5678u;
            for (int i = 0; i < n; i++)
            {
                seed = seed * 1664525u + 1013904223u;
                int rnd = (int)((seed >> 16) & 0xFF); // 0..255
                // Map 0..255 -> 0..3 via the icdf intervals.
                if (rnd < (256 - icdf[0])) symbols[i] = 0;
                else if (rnd < (256 - icdf[1])) symbols[i] = 1;
                else if (rnd < (256 - icdf[2])) symbols[i] = 2;
                else symbols[i] = 3;
            }

            byte[] encoded = OpusEncodeIcdfSequenceCpu(symbols, icdf, ftb, 1024);
            int[] gpuDecoded = await OpusDecodeIcdfSequenceGpuAsync(
                acc, encoded, icdf, ftb, n);

            int firstMismatch = -1;
            for (int i = 0; i < n; i++)
            {
                if (gpuDecoded[i] != symbols[i])
                {
                    firstMismatch = i;
                    break;
                }
            }
            if (firstMismatch >= 0)
            {
                throw new Exception(
                    $"GPU decode mismatch at index {firstMismatch} of {n}: " +
                    $"input={symbols[firstMismatch]} gpu={gpuDecoded[firstMismatch]}. " +
                    $"Encoded byte length: {encoded.Length}.");
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
