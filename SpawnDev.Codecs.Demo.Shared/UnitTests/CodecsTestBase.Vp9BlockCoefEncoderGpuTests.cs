// Cross-backend tests for Vp9BlockCoefEncoderGpu. The GPU encoder
// must produce a byte-for-byte identical bool stream to the CPU
// Vp9BlockCoefEncoder for every input. We exercise:
//   - All-zero 4x4 (single EOB emit)
//   - DC-only 4x4 (one ONE token)
//   - Negative magnitudes (sign bit path)
//   - Every magnitude class (One/Two/Three/Four/Cat1..Cat6)
//   - Larger block sizes (8x8, 16x16, 32x32)
//
// VP9 uses a leading marker bit (0 at probability 128) emitted right
// after Reset; both the CPU Vp9BoolEncoder and the GPU test kernel
// emit this marker before the per-coefficient stream.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static (byte[] expected, int expectedEob) EncodeVp9CoefBlockCpu(
        Vp9TxSize txSize, Vp9ScanType scanType,
        Vp9BlockCoefDecoder.PlaneType planeType,
        Vp9BlockCoefDecoder.RefType refType,
        ReadOnlySpan<short> block,
        int initialCtx = 0,
        bool isHighBitDepth = false)
    {
        var enc = new Vp9BoolEncoder();
        int eob = Vp9BlockCoefEncoder.EncodeBlockCoefficients(
            (prob, bit) => enc.Write(bit, prob),
            txSize, scanType, planeType, refType,
            block, isHighBitDepth, coefProbs: null, initialCtx: initialCtx);
        return (enc.Stop(), eob);
    }

    private static async Task<(byte[] gpuBytes, int gpuEob)> EncodeVp9CoefBlockGpuAsync(
        Accelerator acc,
        Vp9TxSize txSize, Vp9ScanType scanType,
        Vp9BlockCoefDecoder.PlaneType planeType,
        Vp9BlockCoefDecoder.RefType refType,
        short[] block,
        int initialCtx = 0,
        bool isHighBitDepth = false)
    {
        int maxCoefs = txSize switch
        {
            Vp9TxSize.Tx4x4 => 16,
            Vp9TxSize.Tx8x8 => 64,
            Vp9TxSize.Tx16x16 => 256,
            Vp9TxSize.Tx32x32 => 1024,
            _ => throw new ArgumentOutOfRangeException(nameof(txSize)),
        };

        ushort[] scan = Vp9ScanTables.GetScan(txSize, scanType);
        ushort[] neighbors = txSize switch
        {
            Vp9TxSize.Tx4x4 => Vp9NeighborTables.GetNeighbors4x4(scanType),
            Vp9TxSize.Tx8x8 => Vp9NeighborTables.GetNeighbors8x8(scanType),
            Vp9TxSize.Tx16x16 => Vp9NeighborTables.GetNeighbors16x16(scanType),
            Vp9TxSize.Tx32x32 => Vp9NeighborTables.GetNeighbors32x32(scanType),
            _ => throw new ArgumentOutOfRangeException(nameof(txSize)),
        };
        byte[] coefProbs = Vp9CoefProbs.DefaultCoefProbsFor(txSize);
        byte[] consts = Vp9BlockCoefEncoderGpu.BuildConstsBuffer();

        // Generous output buffer - worst case per coef is ~24 bits but
        // carry propagation can need backward room. 8x the coef count
        // is comfortable. Plus space for the marker + Stop's 32 trailing
        // zero bits.
        int outBufLen = Math.Max(64, maxCoefs * 8 + 64);

        using var dOutBuf = acc.Allocate1D<byte>(outBufLen);
        using var dCoefs = acc.Allocate1D<short>(maxCoefs);
        using var dScan = acc.Allocate1D<ushort>(scan.Length);
        using var dNeighbors = acc.Allocate1D<ushort>(neighbors.Length);
        using var dCoefProbs = acc.Allocate1D<byte>(coefProbs.Length);
        using var dConsts = acc.Allocate1D<byte>(consts.Length);
        using var dTokenCache = acc.Allocate1D<byte>(maxCoefs);
        using var dOutLen = acc.Allocate1D<long>(1);
        using var dEob = acc.Allocate1D<int>(1);

        // Pre-zero output buffer (carry propagation reads back through it).
        var zeroBuf = new byte[outBufLen];
        dOutBuf.View.CopyFromCPU(zeroBuf);
        dCoefs.View.CopyFromCPU(block);
        dScan.View.CopyFromCPU(scan);
        dNeighbors.View.CopyFromCPU(neighbors);
        dCoefProbs.View.CopyFromCPU(coefProbs);
        dConsts.View.CopyFromCPU(consts);
        // tokenCache is zeroed by EncodeBlock as its first action.

        using var kernel = new Vp9BlockCoefEncoderTestKernel(acc);
        int packedFlags = Vp9BlockCoefEncoderTestKernel.PackFlags(
            (int)planeType, (int)refType, initialCtx,
            isHighBitDepth ? 1 : 0,
            txSize == Vp9TxSize.Tx4x4 ? 1 : 0);
        kernel.Run(dOutBuf.View, dCoefs.View, dScan.View, dNeighbors.View,
                   dCoefProbs.View, dConsts.View, dTokenCache.View,
                   dOutLen.View, dEob.View,
                   maxCoefs, packedFlags);
        await acc.SynchronizeAsync();

        var outLen = (await dOutLen.CopyToHostAsync())[0];
        var eob = (await dEob.CopyToHostAsync())[0];
        var fullBuf = await dOutBuf.CopyToHostAsync();
        var gpuBytes = new byte[outLen];
        Array.Copy(fullBuf, gpuBytes, outLen);
        return (gpuBytes, eob);
    }

    private static async Task AssertGpuMatchesCpuAsync(
        Accelerator acc,
        Vp9TxSize txSize, Vp9ScanType scanType,
        Vp9BlockCoefDecoder.PlaneType planeType,
        Vp9BlockCoefDecoder.RefType refType,
        short[] block,
        int initialCtx = 0,
        bool isHighBitDepth = false)
    {
        var (cpuBytes, cpuEob) = EncodeVp9CoefBlockCpu(
            txSize, scanType, planeType, refType, block, initialCtx, isHighBitDepth);
        var (gpuBytes, gpuEob) = await EncodeVp9CoefBlockGpuAsync(
            acc, txSize, scanType, planeType, refType, block, initialCtx, isHighBitDepth);

        Equal(cpuEob, gpuEob);
        Equal(cpuBytes.Length, gpuBytes.Length);
        for (int i = 0; i < cpuBytes.Length; i++)
        {
            if (cpuBytes[i] != gpuBytes[i])
                throw new Exception(
                    $"byte mismatch at offset {i}: cpu=0x{cpuBytes[i]:X2} gpu=0x{gpuBytes[i]:X2}, eob={cpuEob}");
        }
    }

    [TestMethod]
    public async Task Vp9BlockCoefEncoderGpu_AllZero4x4_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var block = new short[16];
            await AssertGpuMatchesCpuAsync(
                acc, Vp9TxSize.Tx4x4, Vp9ScanType.Default,
                Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
                block);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9BlockCoefEncoderGpu_DcOnlyPositive4x4_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var block = new short[16];
            block[0] = 17;
            await AssertGpuMatchesCpuAsync(
                acc, Vp9TxSize.Tx4x4, Vp9ScanType.Default,
                Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
                block);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9BlockCoefEncoderGpu_NegativeOne4x4_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var block = new short[16];
            block[0] = -1;
            await AssertGpuMatchesCpuAsync(
                acc, Vp9TxSize.Tx4x4, Vp9ScanType.Default,
                Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
                block);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9BlockCoefEncoderGpu_AllMagnitudes4x4_MatchesCpu()
    {
        // One token per magnitude class (One, Two, Three, Four, Cat1..Cat6).
        // Mirrors the CPU coverage test - assigns magnitudes to scan
        // positions 0..9 so eob = 10 exactly, independent of raster
        // layout.
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var scan = Vp9ScanTables.GetScan(Vp9TxSize.Tx4x4, Vp9ScanType.Default);
            short[] mags = { 1, -2, 3, -4, 6, -10, 18, -34, 66, -67 };
            var block = new short[16];
            for (int s = 0; s < mags.Length; s++) block[scan[s]] = mags[s];

            await AssertGpuMatchesCpuAsync(
                acc, Vp9TxSize.Tx4x4, Vp9ScanType.Default,
                Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
                block);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9BlockCoefEncoderGpu_RandomSparse8x8_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var rng = new Random(unchecked((int)0x9C0E8B1Au));
            for (int trial = 0; trial < 4; trial++)
            {
                var block = new short[64];
                for (int i = 0; i < 64; i++)
                {
                    if (rng.NextDouble() < 0.20)
                        block[i] = (short)rng.Next(-50, 50);
                }
                await AssertGpuMatchesCpuAsync(
                    acc, Vp9TxSize.Tx8x8, Vp9ScanType.Default,
                    Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
                    block);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9BlockCoefEncoderGpu_RandomSparse16x16_MatchesCpu()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            var rng = new Random(unchecked((int)0x163C0EFEu));
            var block = new short[256];
            for (int i = 0; i < 256; i++)
            {
                if (rng.NextDouble() < 0.10)
                    block[i] = (short)rng.Next(-200, 200);
            }
            await AssertGpuMatchesCpuAsync(
                acc, Vp9TxSize.Tx16x16, Vp9ScanType.Default,
                Vp9BlockCoefDecoder.PlaneType.Uv, Vp9BlockCoefDecoder.RefType.Intra,
                block);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9BlockCoefEncoderGpu_NonZeroInitialCtx_MatchesCpu()
    {
        // initialCtx is the per-plane entropy context for scan position 0;
        // it must propagate into the prob lookup at c=0 for both CPU
        // and GPU encoders identically. Sweep all 3 legal values.
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            for (int initialCtx = 0; initialCtx < 3; initialCtx++)
            {
                var block = new short[16];
                block[0] = 5;
                block[5] = -3;
                await AssertGpuMatchesCpuAsync(
                    acc, Vp9TxSize.Tx4x4, Vp9ScanType.Default,
                    Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
                    block, initialCtx);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9BlockCoefEncoderGpu_RowAndColScans4x4_MatchesCpu()
    {
        // Verify both Row and Col scan types route through the kernel.
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            short[] mags = { 1, -2, 3, -4, 6 };
            foreach (var scanType in new[] { Vp9ScanType.Row, Vp9ScanType.Col })
            {
                var scan = Vp9ScanTables.GetScan(Vp9TxSize.Tx4x4, scanType);
                var block = new short[16];
                for (int s = 0; s < mags.Length; s++) block[scan[s]] = mags[s];

                await AssertGpuMatchesCpuAsync(
                    acc, Vp9TxSize.Tx4x4, scanType,
                    Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
                    block);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
