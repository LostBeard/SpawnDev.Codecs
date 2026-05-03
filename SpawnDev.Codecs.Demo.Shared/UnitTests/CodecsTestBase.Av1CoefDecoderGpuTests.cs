// Cross-backend tests for the AV1 coefficient decoder GPU port via
// Av1CoefRoundTripKernel. Encodes a coefficient block on the
// accelerator, decodes it back in the same dispatch, and verifies
// the decoded coefs match the input (bit-exact, level cast to byte
// for level > 255 clamping).
//
// CulLevel + Eob for the encoder side are verified separately by
// Av1CoefEncoderGpuTests. The round-trip here exercises both the
// decoder code path AND the symmetric encode/decode property.

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
    private static async Task<(int[] decodedCoefs, long encLen, int encEob, int decEob)>
        Av1CoefRoundTripGpuAsync(
            Accelerator acc,
            int[] coefs, int txSize, int plane,
            int txbSkipCtx, int dcSignCtx, int qindex,
            int qDc, int qAc,
            byte[] constsByte, ushort[] constsUshort)
    {
        int qctx = Av1CoefDecoder.GetQctx(qindex);
        int outBufLen = Math.Max(64, coefs.Length * 6 + 256);
        int levelsLen = (32 + Av1TxbCommon.TxPadHor) * (32 + Av1TxbCommon.TxPadVer) + Av1TxbCommon.TxPadEnd;

        using var dScratchBytes = acc.Allocate1D<byte>(outBufLen);
        using var dConstsByte = acc.Allocate1D<byte>(constsByte.Length);
        using var dConstsUshort = acc.Allocate1D<ushort>(constsUshort.Length);
        using var dCoefs = acc.Allocate1D<int>(coefs.Length);
        using var dDecoded = acc.Allocate1D<int>(coefs.Length);
        using var dLevels = acc.Allocate1D<byte>(levelsLen);
        using var dOutLen = acc.Allocate1D<long>(1);
        // encDecInfo: [0]=encEob, [1]=decEob, [2]=encCul (throwaway), [3]=decCul (throwaway).
        using var dInfo = acc.Allocate1D<int>(4);

        dScratchBytes.View.CopyFromCPU(new byte[outBufLen]);
        dConstsByte.View.CopyFromCPU(constsByte);
        dConstsUshort.View.CopyFromCPU(constsUshort);
        dCoefs.View.CopyFromCPU(coefs);
        dDecoded.View.CopyFromCPU(new int[coefs.Length]);
        dLevels.View.CopyFromCPU(new byte[levelsLen]);

        using var kernel = new Av1CoefRoundTripKernel(acc);
        kernel.Run(dScratchBytes.View, dConstsByte.View, dConstsUshort.View,
            dCoefs.View, dDecoded.View, dLevels.View, dOutLen.View, dInfo.View,
            new Av1CoefRoundTripParams
            {
                TxSize = txSize,
                Plane = plane,
                Qctx = qctx,
                TxbSkipCtx = txbSkipCtx,
                DcSignCtx = dcSignCtx,
                Qindex = qindex,
                QDc = qDc,
                QAc = qAc,
            });
        await acc.SynchronizeAsync();

        long encLen = (await dOutLen.CopyToHostAsync())[0];
        var info = await dInfo.CopyToHostAsync();
        var decoded = await dDecoded.CopyToHostAsync();
        var decodedSlice = new int[coefs.Length];
        Array.Copy(decoded, decodedSlice, coefs.Length);
        return (decodedSlice, encLen, info[0], info[1]);
    }

    [TestMethod]
    public async Task Av1CoefDecoderGpu_RoundTrip_AllZero_Tx16x16()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var constsByte = Av1KeyframeConstantsGpu.BuildByteConstsBuffer();
            var constsUshort = Av1KeyframeConstantsGpu.BuildUshortConstsBuffer();
            var coefs = new int[16 * 16];

            var (decoded, encLen, encEob, decEob) = await Av1CoefRoundTripGpuAsync(
                acc, coefs, txSize: 2, plane: 0,
                txbSkipCtx: 0, dcSignCtx: 0, qindex: 32,
                qDc: 1, qAc: 1,
                constsByte, constsUshort);

            Equal(0, encEob, "encEob");
            Equal(0, decEob, "decEob");
            for (int i = 0; i < coefs.Length; i++)
                if (decoded[i] != 0)
                    throw new Exception($"decoded[{i}] = {decoded[i]} for all-zero input");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1CoefDecoderGpu_RoundTrip_DcOnly_Tx8x8()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var constsByte = Av1KeyframeConstantsGpu.BuildByteConstsBuffer();
            var constsUshort = Av1KeyframeConstantsGpu.BuildUshortConstsBuffer();
            var coefs = new int[8 * 8];
            coefs[0] = 7;

            var (decoded, encLen, encEob, decEob) = await Av1CoefRoundTripGpuAsync(
                acc, coefs, txSize: 1, plane: 1,
                txbSkipCtx: 0, dcSignCtx: 0, qindex: 64,
                qDc: 1, qAc: 1,
                constsByte, constsUshort);

            Equal(1, encEob, "encEob");
            Equal(1, decEob, "decEob");
            Equal(7, decoded[0], "decoded DC value");
            for (int i = 1; i < coefs.Length; i++)
                if (decoded[i] != 0)
                    throw new Exception($"decoded[{i}] = {decoded[i]} for DC-only input");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1CoefDecoderGpu_RoundTrip_NegativeDc_Tx8x8()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var constsByte = Av1KeyframeConstantsGpu.BuildByteConstsBuffer();
            var constsUshort = Av1KeyframeConstantsGpu.BuildUshortConstsBuffer();
            var coefs = new int[8 * 8];
            coefs[0] = -10; // Negative DC exercises the dc_sign CDF + sign reconstruction.
            coefs[1] = 3;

            var (decoded, encLen, encEob, decEob) = await Av1CoefRoundTripGpuAsync(
                acc, coefs, txSize: 1, plane: 2,
                txbSkipCtx: 1, dcSignCtx: 1, qindex: 32,
                qDc: 1, qAc: 1,
                constsByte, constsUshort);

            Equal(2, encEob, "encEob");
            Equal(2, decEob, "decEob");
            Equal(-10, decoded[0], "decoded DC");
            Equal(3, decoded[1], "decoded AC[1]");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1CoefDecoderGpu_RoundTrip_RandomSparse_Tx16x16()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var constsByte = Av1KeyframeConstantsGpu.BuildByteConstsBuffer();
            var constsUshort = Av1KeyframeConstantsGpu.BuildUshortConstsBuffer();
            var rng = new Random(unchecked((int)0xA1D016BAu));
            var coefs = new int[16 * 16];
            for (int i = 0; i < coefs.Length; i++)
                if (rng.Next(8) == 0) coefs[i] = rng.Next(-3, 4);

            var (decoded, encLen, encEob, decEob) = await Av1CoefRoundTripGpuAsync(
                acc, coefs, txSize: 2, plane: 0,
                txbSkipCtx: 2, dcSignCtx: 0, qindex: 32,
                qDc: 1, qAc: 1,
                constsByte, constsUshort);

            Equal(encEob, decEob, "eob round-trip");
            // Decoded coefs match input exactly (qDc=qAc=1).
            for (int i = 0; i < coefs.Length; i++)
                if (coefs[i] != decoded[i])
                    throw new Exception($"coef[{i}]: input={coefs[i]} decoded={decoded[i]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Av1CoefDecoderGpu_RoundTrip_HighMagnitude_Golomb()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var constsByte = Av1KeyframeConstantsGpu.BuildByteConstsBuffer();
            var constsUshort = Av1KeyframeConstantsGpu.BuildUshortConstsBuffer();
            var coefs = new int[16 * 16];
            // High magnitudes that exercise the golomb tail path.
            coefs[0] = 50;
            coefs[1] = -75;
            coefs[2] = 30;
            coefs[16] = -42;
            coefs[17] = 200;

            var (decoded, encLen, encEob, decEob) = await Av1CoefRoundTripGpuAsync(
                acc, coefs, txSize: 2, plane: 0,
                txbSkipCtx: 7, dcSignCtx: 1, qindex: 24,
                qDc: 1, qAc: 1,
                constsByte, constsUshort);

            Equal(encEob, decEob, "eob round-trip");
            for (int i = 0; i < coefs.Length; i++)
                if (coefs[i] != decoded[i])
                    throw new Exception($"coef[{i}]: input={coefs[i]} decoded={decoded[i]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
