// Cross-backend tests for the AV1 coefficient encoder GPU port.
// Verifies Av1CoefEncoderGpu.WriteCoeffsTxb produces byte-exact output
// to the CPU Av1CoefEncoder.WriteCoeffsTxb reference for the v1
// keyframe encoder's two configurations:
//   - Tx8x8 + DCT_DCT + DC_PRED + plane=1 (chroma U)
//   - Tx16x16 + DCT_DCT + DC_PRED + plane=0 (luma Y)
//
// Each test feeds the same coefficient block to both encoders, runs
// each to completion (Done), and compares output bytes byte-for-byte.

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
    private static byte[] EncodeOneBlockCpu(
        int[] coefs, Av1TxSize txSize, int plane,
        int txbSkipCtx, int dcSignCtx, int qindex,
        out int eob, out int culLevel)
    {
        var enc = new Av1RangeEncoder();
        var result = Av1CoefEncoder.WriteCoeffsTxb(
            enc, txSize, plane, Av1IntraMode.Dc, reducedTxSet: true,
            txbSkipCtx, dcSignCtx, coefs, qindex, Av1TxType.DctDct);
        eob = result.Eob;
        culLevel = result.CulLevel;
        return enc.Done();
    }

    private static int Av1QindexToQctx(int qindex)
    {
        // libaom get_q_ctx (Av1CoefDecoder.GetQctx).
        return Av1CoefDecoder.GetQctx(qindex);
    }

    private static async Task<(byte[] gpuBytes, int eob, int culLevel)> EncodeOneBlockGpuAsync(
        Accelerator acc,
        int[] coefs, Av1TxSize txSize, int plane,
        int txbSkipCtx, int dcSignCtx, int qindex,
        byte[] constsByte, ushort[] constsUshort)
    {
        int qctx = Av1QindexToQctx(qindex);
        // Per-block worst-case bytes (range coder output): each q15 sym
        // can emit up to ~2 bytes after Done; budget generously.
        int outBufLen = Math.Max(64, coefs.Length * 6 + 256);
        // levels[] padded scratch per AV1 worst case (32+TxPadHor)*(32+TxPadVer)+TxPadEnd.
        int levelsLen = (32 + Av1TxbCommon.TxPadHor) * (32 + Av1TxbCommon.TxPadVer) + Av1TxbCommon.TxPadEnd;

        using var dOutBuf = acc.Allocate1D<byte>(outBufLen);
        using var dConstsByte = acc.Allocate1D<byte>(constsByte.Length);
        using var dConstsUshort = acc.Allocate1D<ushort>(constsUshort.Length);
        using var dCoefs = acc.Allocate1D<int>(coefs.Length);
        using var dLevels = acc.Allocate1D<byte>(levelsLen);
        using var dOutLen = acc.Allocate1D<long>(1);
        using var dEob = acc.Allocate1D<int>(1);
        using var dCulLevel = acc.Allocate1D<int>(1);

        dOutBuf.View.CopyFromCPU(new byte[outBufLen]);
        dConstsByte.View.CopyFromCPU(constsByte);
        dConstsUshort.View.CopyFromCPU(constsUshort);
        dCoefs.View.CopyFromCPU(coefs);
        dLevels.View.CopyFromCPU(new byte[levelsLen]);

        using var kernel = new Av1CoefEncoderGpuKernel(acc);
        kernel.Run(dOutBuf.View, dConstsByte.View, dConstsUshort.View,
            dCoefs.View, dLevels.View, dOutLen.View, dEob.View, dCulLevel.View,
            (int)txSize, plane, qctx, txbSkipCtx, dcSignCtx, qindex);
        await acc.SynchronizeAsync();

        long outLen = (await dOutLen.CopyToHostAsync())[0];
        var fullBytes = await dOutBuf.CopyToHostAsync();
        var bytes = new byte[outLen];
        Array.Copy(fullBytes, bytes, outLen);
        int eob = (await dEob.CopyToHostAsync())[0];
        int culLevel = (await dCulLevel.CopyToHostAsync())[0];
        return (bytes, eob, culLevel);
    }

    [TestMethod]
    public async Task Av1CoefEncoderGpu_AllZero_Tx16x16_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var constsByte = Av1KeyframeConstantsGpu.BuildByteConstsBuffer();
            var constsUshort = Av1KeyframeConstantsGpu.BuildUshortConstsBuffer();
            var coefs = new int[16 * 16];
            int qindex = 32;

            var cpuBytes = EncodeOneBlockCpu(coefs, Av1TxSize.Tx16x16, plane: 0,
                txbSkipCtx: 0, dcSignCtx: 0, qindex: qindex,
                out int cpuEob, out int cpuCul);
            var (gpuBytes, gpuEob, gpuCul) = await EncodeOneBlockGpuAsync(acc,
                coefs, Av1TxSize.Tx16x16, plane: 0,
                txbSkipCtx: 0, dcSignCtx: 0, qindex: qindex,
                constsByte, constsUshort);

            Equal(cpuEob, gpuEob, "Eob");
            Equal(cpuCul, gpuCul, "CulLevel");
            Equal(cpuBytes.Length, gpuBytes.Length, "byte length");
            for (int i = 0; i < cpuBytes.Length; i++)
                if (cpuBytes[i] != gpuBytes[i])
                    throw new Exception($"byte {i}: cpu=0x{cpuBytes[i]:X2} gpu=0x{gpuBytes[i]:X2}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1CoefEncoderGpu_DcOnly_Tx8x8_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var constsByte = Av1KeyframeConstantsGpu.BuildByteConstsBuffer();
            var constsUshort = Av1KeyframeConstantsGpu.BuildUshortConstsBuffer();
            var coefs = new int[8 * 8];
            coefs[0] = 5; // DC only.
            int qindex = 64;

            var cpuBytes = EncodeOneBlockCpu(coefs, Av1TxSize.Tx8x8, plane: 1,
                txbSkipCtx: 0, dcSignCtx: 0, qindex: qindex,
                out int cpuEob, out int cpuCul);
            var (gpuBytes, gpuEob, gpuCul) = await EncodeOneBlockGpuAsync(acc,
                coefs, Av1TxSize.Tx8x8, plane: 1,
                txbSkipCtx: 0, dcSignCtx: 0, qindex: qindex,
                constsByte, constsUshort);

            Equal(cpuEob, gpuEob, "Eob");
            Equal(cpuCul, gpuCul, "CulLevel");
            Equal(cpuBytes.Length, gpuBytes.Length, "byte length");
            for (int i = 0; i < cpuBytes.Length; i++)
                if (cpuBytes[i] != gpuBytes[i])
                    throw new Exception($"byte {i}: cpu=0x{cpuBytes[i]:X2} gpu=0x{gpuBytes[i]:X2}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1CoefEncoderGpu_RandomSparse_Tx16x16_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var constsByte = Av1KeyframeConstantsGpu.BuildByteConstsBuffer();
            var constsUshort = Av1KeyframeConstantsGpu.BuildUshortConstsBuffer();
            var rng = new Random(unchecked((int)0xA1C016BAu));
            var coefs = new int[16 * 16];
            // Sparse: only ~10% of positions, low-magnitude.
            for (int i = 0; i < coefs.Length; i++)
                if (rng.Next(10) == 0) coefs[i] = rng.Next(-3, 4);
            int qindex = 32;

            var cpuBytes = EncodeOneBlockCpu(coefs, Av1TxSize.Tx16x16, plane: 0,
                txbSkipCtx: 2, dcSignCtx: 1, qindex: qindex,
                out int cpuEob, out int cpuCul);
            var (gpuBytes, gpuEob, gpuCul) = await EncodeOneBlockGpuAsync(acc,
                coefs, Av1TxSize.Tx16x16, plane: 0,
                txbSkipCtx: 2, dcSignCtx: 1, qindex: qindex,
                constsByte, constsUshort);

            Equal(cpuEob, gpuEob, "Eob");
            Equal(cpuCul, gpuCul, "CulLevel");
            Equal(cpuBytes.Length, gpuBytes.Length, "byte length");
            for (int i = 0; i < cpuBytes.Length; i++)
                if (cpuBytes[i] != gpuBytes[i])
                    throw new Exception($"byte {i}: cpu=0x{cpuBytes[i]:X2} gpu=0x{gpuBytes[i]:X2}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1CoefEncoderGpu_RandomDense_Tx8x8_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var constsByte = Av1KeyframeConstantsGpu.BuildByteConstsBuffer();
            var constsUshort = Av1KeyframeConstantsGpu.BuildUshortConstsBuffer();
            var rng = new Random(unchecked((int)0xA1C08DEBu));
            var coefs = new int[8 * 8];
            // Dense low-magnitude: every position non-zero in [-5, 5].
            for (int i = 0; i < coefs.Length; i++) coefs[i] = rng.Next(-5, 6);
            int qindex = 96;

            var cpuBytes = EncodeOneBlockCpu(coefs, Av1TxSize.Tx8x8, plane: 2,
                txbSkipCtx: 5, dcSignCtx: 2, qindex: qindex,
                out int cpuEob, out int cpuCul);
            var (gpuBytes, gpuEob, gpuCul) = await EncodeOneBlockGpuAsync(acc,
                coefs, Av1TxSize.Tx8x8, plane: 2,
                txbSkipCtx: 5, dcSignCtx: 2, qindex: qindex,
                constsByte, constsUshort);

            Equal(cpuEob, gpuEob, "Eob");
            Equal(cpuCul, gpuCul, "CulLevel");
            Equal(cpuBytes.Length, gpuBytes.Length, "byte length");
            for (int i = 0; i < cpuBytes.Length; i++)
                if (cpuBytes[i] != gpuBytes[i])
                    throw new Exception($"byte {i}: cpu=0x{cpuBytes[i]:X2} gpu=0x{gpuBytes[i]:X2}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1CoefEncoderGpu_HighMagnitude_Tx16x16_GolombPath_MatchesCpu()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var constsByte = Av1KeyframeConstantsGpu.BuildByteConstsBuffer();
            var constsUshort = Av1KeyframeConstantsGpu.BuildUshortConstsBuffer();
            var rng = new Random(unchecked((int)0xA1C016ADu));
            var coefs = new int[16 * 16];
            // Force large magnitudes that exercise the golomb tail
            // path (level > CoeffBaseRange + NumBaseLevels = 14).
            coefs[0] = 50;
            coefs[1] = -75;
            coefs[2] = 30;
            coefs[16] = -42;
            coefs[17] = 200;
            for (int i = 0; i < coefs.Length; i++)
                if (coefs[i] == 0 && rng.Next(20) == 0) coefs[i] = rng.Next(-15, 16);
            int qindex = 24;

            var cpuBytes = EncodeOneBlockCpu(coefs, Av1TxSize.Tx16x16, plane: 0,
                txbSkipCtx: 7, dcSignCtx: 1, qindex: qindex,
                out int cpuEob, out int cpuCul);
            var (gpuBytes, gpuEob, gpuCul) = await EncodeOneBlockGpuAsync(acc,
                coefs, Av1TxSize.Tx16x16, plane: 0,
                txbSkipCtx: 7, dcSignCtx: 1, qindex: qindex,
                constsByte, constsUshort);

            Equal(cpuEob, gpuEob, "Eob");
            Equal(cpuCul, gpuCul, "CulLevel");
            Equal(cpuBytes.Length, gpuBytes.Length, "byte length");
            for (int i = 0; i < cpuBytes.Length; i++)
                if (cpuBytes[i] != gpuBytes[i])
                    throw new Exception($"byte {i}: cpu=0x{cpuBytes[i]:X2} gpu=0x{gpuBytes[i]:X2}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
