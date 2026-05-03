// Cross-backend tests for Vp9BlockCoefDecoderGpu. The GPU decoder
// must produce a coefficient block byte-for-byte identical to the
// CPU Vp9BlockCoefDecoder for every input that the encoder side
// could produce. We verify this via round-trip: encode block on
// the CPU side via Vp9BlockCoefEncoder, then decode on the GPU
// side via Vp9BlockCoefDecoderGpu, then compare against the
// original input.
//
// VP9 uses a leading marker bit (0 at prob 128) emitted by the
// encoder right after Reset. The CPU Vp9BoolDecoder consumes it
// during init; the GPU test kernel mirrors that by calling
// DecodeBool(.., 128) right after Init.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] EncodeVp9BlockForDecode(
        Vp9TxSize txSize, Vp9ScanType scanType,
        Vp9BlockCoefDecoder.PlaneType planeType,
        Vp9BlockCoefDecoder.RefType refType,
        short[] block,
        int initialCtx,
        bool isHighBitDepth)
    {
        var enc = new Vp9BoolEncoder();
        Vp9BlockCoefEncoder.EncodeBlockCoefficients(
            (prob, bit) => enc.Write(bit, prob),
            txSize, scanType, planeType, refType,
            block, isHighBitDepth, coefProbs: null, initialCtx: initialCtx);
        return enc.Stop();
    }

    private static async Task<(short[] decoded, int eob)> DecodeVp9CoefBlockGpuAsync(
        Accelerator acc,
        byte[] encoded,
        Vp9TxSize txSize, Vp9ScanType scanType,
        Vp9BlockCoefDecoder.PlaneType planeType,
        Vp9BlockCoefDecoder.RefType refType,
        int initialCtx,
        bool isHighBitDepth)
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

        using var dInBuf = acc.Allocate1D<byte>(encoded.Length);
        using var dBlock = acc.Allocate1D<short>(maxCoefs);
        using var dScan = acc.Allocate1D<ushort>(scan.Length);
        using var dNeighbors = acc.Allocate1D<ushort>(neighbors.Length);
        using var dCoefProbs = acc.Allocate1D<byte>(coefProbs.Length);
        using var dConsts = acc.Allocate1D<byte>(consts.Length);
        using var dTokenCache = acc.Allocate1D<byte>(maxCoefs);
        using var dEob = acc.Allocate1D<int>(1);

        dInBuf.View.CopyFromCPU(encoded);
        dScan.View.CopyFromCPU(scan);
        dNeighbors.View.CopyFromCPU(neighbors);
        dCoefProbs.View.CopyFromCPU(coefProbs);
        dConsts.View.CopyFromCPU(consts);

        using var kernel = new Vp9BlockCoefDecoderTestKernel(acc);
        int packedFlags = Vp9BlockCoefEncoderTestKernel.PackFlags(
            (int)planeType, (int)refType, initialCtx,
            isHighBitDepth ? 1 : 0,
            txSize == Vp9TxSize.Tx4x4 ? 1 : 0);
        kernel.Run(dInBuf.View, dBlock.View, dScan.View, dNeighbors.View,
                   dCoefProbs.View, dConsts.View, dTokenCache.View, dEob.View,
                   encoded.Length, maxCoefs, packedFlags);
        await acc.SynchronizeAsync();

        int eob = (await dEob.CopyToHostAsync())[0];
        var decoded = await dBlock.CopyToHostAsync();
        return (decoded.AsSpan(0, maxCoefs).ToArray(), eob);
    }

    private static async Task AssertGpuDecodeMatchesCpuEncodeAsync(
        Accelerator acc,
        Vp9TxSize txSize, Vp9ScanType scanType,
        Vp9BlockCoefDecoder.PlaneType planeType,
        Vp9BlockCoefDecoder.RefType refType,
        short[] block,
        int expectedEob,
        int initialCtx = 0,
        bool isHighBitDepth = false)
    {
        byte[] encoded = EncodeVp9BlockForDecode(
            txSize, scanType, planeType, refType, block, initialCtx, isHighBitDepth);
        var (decoded, gpuEob) = await DecodeVp9CoefBlockGpuAsync(
            acc, encoded, txSize, scanType, planeType, refType, initialCtx, isHighBitDepth);

        Equal(expectedEob, gpuEob);
        for (int i = 0; i < block.Length; i++)
        {
            if (block[i] != decoded[i])
                throw new Exception(
                    $"coef mismatch at raster {i}: expected {block[i]}, got {decoded[i]}, eob={gpuEob}");
        }
    }

    [TestMethod]
    public async Task Vp9BlockCoefDecoderGpu_AllZero4x4_RoundTrips()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var block = new short[16];
            await AssertGpuDecodeMatchesCpuEncodeAsync(
                acc, Vp9TxSize.Tx4x4, Vp9ScanType.Default,
                Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
                block, expectedEob: 0);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9BlockCoefDecoderGpu_DcOnly4x4_RoundTrips()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var block = new short[16];
            block[0] = 17;
            await AssertGpuDecodeMatchesCpuEncodeAsync(
                acc, Vp9TxSize.Tx4x4, Vp9ScanType.Default,
                Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
                block, expectedEob: 1);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9BlockCoefDecoderGpu_NegativeOne4x4_RoundTrips()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var block = new short[16];
            block[0] = -1;
            await AssertGpuDecodeMatchesCpuEncodeAsync(
                acc, Vp9TxSize.Tx4x4, Vp9ScanType.Default,
                Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
                block, expectedEob: 1);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9BlockCoefDecoderGpu_AllMagnitudes4x4_RoundTrips()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var scan = Vp9ScanTables.GetScan(Vp9TxSize.Tx4x4, Vp9ScanType.Default);
            short[] mags = { 1, -2, 3, -4, 6, -10, 18, -34, 66, -67 };
            var block = new short[16];
            for (int s = 0; s < mags.Length; s++) block[scan[s]] = mags[s];

            await AssertGpuDecodeMatchesCpuEncodeAsync(
                acc, Vp9TxSize.Tx4x4, Vp9ScanType.Default,
                Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
                block, expectedEob: 10);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9BlockCoefDecoderGpu_RandomSparse8x8_RoundTrips()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var rng = new Random(unchecked((int)0x8C0DEC58u));
            for (int trial = 0; trial < 4; trial++)
            {
                var block = new short[64];
                int lastNonZeroScan = -1;
                var scan = Vp9ScanTables.GetScan(Vp9TxSize.Tx8x8, Vp9ScanType.Default);
                for (int s = 0; s < 64; s++)
                {
                    if (rng.NextDouble() < 0.20)
                    {
                        block[scan[s]] = (short)rng.Next(-50, 50);
                        if (block[scan[s]] != 0) lastNonZeroScan = s;
                    }
                }
                int expectedEob = lastNonZeroScan + 1;
                await AssertGpuDecodeMatchesCpuEncodeAsync(
                    acc, Vp9TxSize.Tx8x8, Vp9ScanType.Default,
                    Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
                    block, expectedEob);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9BlockCoefDecoderGpu_RandomSparse16x16_RoundTrips()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var rng = new Random(unchecked((int)0x16DC0DE1u));
            var block = new short[256];
            int lastNonZeroScan = -1;
            var scan = Vp9ScanTables.GetScan(Vp9TxSize.Tx16x16, Vp9ScanType.Default);
            for (int s = 0; s < 256; s++)
            {
                if (rng.NextDouble() < 0.10)
                {
                    block[scan[s]] = (short)rng.Next(-200, 200);
                    if (block[scan[s]] != 0) lastNonZeroScan = s;
                }
            }
            int expectedEob = lastNonZeroScan + 1;
            await AssertGpuDecodeMatchesCpuEncodeAsync(
                acc, Vp9TxSize.Tx16x16, Vp9ScanType.Default,
                Vp9BlockCoefDecoder.PlaneType.Uv, Vp9BlockCoefDecoder.RefType.Intra,
                block, expectedEob);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9BlockCoefDecoderGpu_NonZeroInitialCtx_RoundTrips()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            for (int initialCtx = 0; initialCtx < 3; initialCtx++)
            {
                var block = new short[16];
                block[0] = 5;
                block[5] = -3;
                // expectedEob: scan position of last non-zero + 1.
                // For Default scan + (raster 0, raster 5) - we need the
                // round-trip oracle to compute it - simpler to just call
                // CPU encoder/decoder once to determine.
                int expectedEob = 0;
                {
                    var enc = new Vp9BoolEncoder();
                    expectedEob = Vp9BlockCoefEncoder.EncodeBlockCoefficients(
                        (prob, bit) => enc.Write(bit, prob),
                        Vp9TxSize.Tx4x4, Vp9ScanType.Default,
                        Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
                        block, false, null, initialCtx);
                }
                await AssertGpuDecodeMatchesCpuEncodeAsync(
                    acc, Vp9TxSize.Tx4x4, Vp9ScanType.Default,
                    Vp9BlockCoefDecoder.PlaneType.Y, Vp9BlockCoefDecoder.RefType.Intra,
                    block, expectedEob, initialCtx);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
