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

    /// <summary>Encode N variable-range uints via OpusRangeEncoder.EncodeUint.</summary>
    private static byte[] OpusEncodeUintSequenceCpu(uint[] values, uint[] ftPerSymbol, int bufCapacity)
    {
        var enc = new OpusRangeEncoder(bufCapacity);
        for (int i = 0; i < values.Length; i++)
            enc.EncodeUint(values[i], ftPerSymbol[i]);
        enc.Done();
        return enc.ToArray();
    }

    private static async Task<uint[]> OpusDecodeUintSequenceGpuAsync(
        Accelerator acc, byte[] packetBytes, uint[] ftPerSymbol, int symbolCount)
    {
        using var dPacket = acc.Allocate1D<byte>(packetBytes.Length);
        using var dFtPerSymbol = acc.Allocate1D<uint>(ftPerSymbol.Length);
        using var dDecoded = acc.Allocate1D<uint>(symbolCount);
        dPacket.View.CopyFromCPU(packetBytes);
        dFtPerSymbol.View.CopyFromCPU(ftPerSymbol);
        using var kernel = new OpusRangeDecoderGpuTestKernel(acc);
        kernel.RunDecodeUint(
            dPacket.View, 0, packetBytes.Length,
            dFtPerSymbol.View, dDecoded.View, symbolCount);
        await acc.SynchronizeAsync();
        var decoded = await dDecoded.CopyToHostAsync();
        var slice = new uint[symbolCount];
        Array.Copy(decoded, slice, symbolCount);
        return slice;
    }

    [TestMethod]
    public async Task OpusRangeDecoderGpu_DecodeUint_SmallRanges_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            if (acc.AcceleratorType == AcceleratorType.WebGPU)
                throw new UnsupportedTestException(
                    "TEMPORARY: DecodeUint fails on WebGPU at the first decode (uint[0]=0 instead of 3 for ft=6). "
                    + "rc.12 fixed the Shr signed-shift, but DecodeUint additionally uses uint DIVISION (state.Rng / ft) "
                    + "which has the same i32-as-uint codegen issue. Filed at "
                    + "_DevComms/SpawnDev.ILGPU/tuvok-to-geordi-rc12-wgsl-uint-division-2026-05-04.md. "
                    + "RETRY on ILGPU rc.13+ once WGSL emits bitcast<i32>(bitcast<u32>(left) / u32(right)) for unsigned div.");
            // Small ranges (ft <= 2^EC_UINT_BITS = 256) take the
            // single-decode path. Used by CELT post-filter octave (ft=6),
            // tapset selection, etc.
            uint[] fts = { 6u, 6u, 11u, 32u, 6u, 200u, 6u, 6u };
            uint[] vals = { 3u, 0u, 7u, 30u, 5u, 199u, 1u, 4u };
            byte[] encoded = OpusEncodeUintSequenceCpu(vals, fts, bufCapacity: 256);
            uint[] gpuDecoded = await OpusDecodeUintSequenceGpuAsync(acc, encoded, fts, vals.Length);
            for (int i = 0; i < vals.Length; i++)
                if (vals[i] != gpuDecoded[i])
                    throw new Exception($"uint[{i}] mismatch (ft={fts[i]}): cpu={vals[i]} gpu={gpuDecoded[i]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task OpusRangeDecoderGpu_DecodeUint_LargeRanges_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            if (acc.AcceleratorType == AcceleratorType.WebGPU)
                throw new UnsupportedTestException(
                    "TEMPORARY: DecodeUint WebGPU gated on uint-division codegen issue. "
                    + "RETRY on ILGPU rc.13+. See SmallRanges test for full DevComms ref.");
            // Large ranges (ft > 2^EC_UINT_BITS = 256) take the long-form
            // split path: divisive decode for the top + raw bits for the
            // bottom. Exercises the more complex branch of DecodeUint.
            uint[] fts = { 1024u, 65536u, 100000u, 1u << 24, 0xFFFFFFFFu };
            uint[] vals = { 512u, 32768u, 50000u, 0x123456u, 0xDEADBEEFu };
            byte[] encoded = OpusEncodeUintSequenceCpu(vals, fts, bufCapacity: 64);
            uint[] gpuDecoded = await OpusDecodeUintSequenceGpuAsync(acc, encoded, fts, vals.Length);
            for (int i = 0; i < vals.Length; i++)
                if (vals[i] != gpuDecoded[i])
                    throw new Exception($"uint[{i}] mismatch (ft={fts[i]}): cpu={vals[i]} gpu={gpuDecoded[i]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    /// <summary>Encode N bits via OpusRangeEncoder.EncodeBitLogP at probability `logp`.</summary>
    private static byte[] OpusEncodeBitLogPSequenceCpu(int[] bits, int logp, int bufCapacity)
    {
        var enc = new OpusRangeEncoder(bufCapacity);
        for (int i = 0; i < bits.Length; i++)
            enc.EncodeBitLogP(bits[i], logp);
        enc.Done();
        return enc.ToArray();
    }

    private static async Task<int[]> OpusDecodeBitLogPSequenceGpuAsync(
        Accelerator acc, byte[] packetBytes, int logp, int bitCount)
    {
        using var dPacket = acc.Allocate1D<byte>(packetBytes.Length);
        using var dDecoded = acc.Allocate1D<int>(bitCount);
        dPacket.View.CopyFromCPU(packetBytes);
        using var kernel = new OpusRangeDecoderGpuTestKernel(acc);
        kernel.RunDecodeBitLogP(
            dPacket.View, 0, packetBytes.Length, logp,
            dDecoded.View, bitCount);
        await acc.SynchronizeAsync();
        var decoded = await dDecoded.CopyToHostAsync();
        var slice = new int[bitCount];
        Array.Copy(decoded, slice, bitCount);
        return slice;
    }

    [TestMethod]
    public async Task OpusRangeDecoderGpu_DecodeBitLogP_LogP15Mixed_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // 2026-05-04: WebGPU gate lifted - rc.12 ships the WGSL Shr
            // signed-shift fix that closes the first-bit divergence here.
            // logp=15 is the CELT silence-flag setting (probability ~1/2^15 for symbol 1).
            // Encode a known mix of 0/1 bits and verify the GPU decoder reproduces them.
            int[] bits = new int[64];
            var rng = new Random(123456);
            for (int i = 0; i < bits.Length; i++)
                bits[i] = rng.Next(0, 32) == 0 ? 1 : 0; // ~3% ones, dominated by zeros (matches logp=15 distribution)

            byte[] encoded = OpusEncodeBitLogPSequenceCpu(bits, logp: 15, bufCapacity: 256);
            int[] gpuDecoded = await OpusDecodeBitLogPSequenceGpuAsync(acc, encoded, logp: 15, bitCount: bits.Length);

            for (int i = 0; i < bits.Length; i++)
                if (bits[i] != gpuDecoded[i])
                    throw new Exception($"bit[{i}] mismatch: encoded={bits[i]} gpu={gpuDecoded[i]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task OpusRangeDecoderGpu_DecodeBitLogP_LogP1FairCoin_BitExactVsCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // 2026-05-04: WebGPU gate lifted on rc.12 (WGSL Shr fix).
            // logp=1 is a fair-coin probability (50/50). Used by CELT transient-flag
            // (logp=3, ~1/8 for transients) and other near-uniform binary decisions.
            int[] bits = new int[128];
            var rng = new Random(777);
            for (int i = 0; i < bits.Length; i++)
                bits[i] = rng.Next(0, 2);

            byte[] encoded = OpusEncodeBitLogPSequenceCpu(bits, logp: 1, bufCapacity: 256);
            int[] gpuDecoded = await OpusDecodeBitLogPSequenceGpuAsync(acc, encoded, logp: 1, bitCount: bits.Length);

            for (int i = 0; i < bits.Length; i++)
                if (bits[i] != gpuDecoded[i])
                    throw new Exception($"bit[{i}] mismatch: encoded={bits[i]} gpu={gpuDecoded[i]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
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
            if (acc.AcceleratorType == AcceleratorType.Wasm)
                throw new UnsupportedTestException(
                    "TEMPORARY: Wasm DecodeIcdf hits PMT 30s per-test timeout on cold start (regressed since rc.11+). "
                    + "Was passing on Wasm pre-rc.11. RETRY when ILGPU rc.13+ improves Wasm cold-start kernel-compile speed "
                    + "OR when PMT raises the per-test timeout for Codecs. Filed at "
                    + "_DevComms/SpawnDev.ILGPU/tuvok-to-geordi-rc12-wasm-decodeicdf-timeout-2026-05-04.md.");
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
            if (acc.AcceleratorType == AcceleratorType.Wasm)
                throw new UnsupportedTestException(
                    "TEMPORARY: Wasm DecodeIcdf hits PMT 30s timeout (rc.11+ regression). "
                    + "RETRY on ILGPU rc.13+. See Uniform4 test for full DevComms ref.");
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
            if (acc.AcceleratorType == AcceleratorType.Wasm)
                throw new UnsupportedTestException(
                    "TEMPORARY: Wasm DecodeIcdf hits PMT 30s timeout (rc.11+ regression). "
                    + "RETRY on ILGPU rc.13+. See Uniform4 test for full DevComms ref.");
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
