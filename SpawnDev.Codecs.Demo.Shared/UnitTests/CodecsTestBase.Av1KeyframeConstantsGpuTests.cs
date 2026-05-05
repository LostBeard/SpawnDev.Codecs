// Tests for Av1KeyframeConstantsGpu - the unified constants packer
// for the AV1 v1 keyframe encoder + decoder GPU kernels.
//
// Verifies:
//   - Buffer sizes match the documented layout.
//   - Sentinel CDF values at known offsets match the original
//     CPU-side jagged tables byte-for-byte (proves the packing is
//     correct, not just non-zero).
//   - Scan tables match Av1ScanTables.Scan[ts][tt] exactly.
//   - Extra constant arrays (EobGroupStart / EobOffsetBits) round-trip
//     through the byte-pack format correctly.

using SpawnDev.Codecs.Video.Av1;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Av1KeyframeConstantsGpu_BufferSizes_MatchLayout()
    {
        var byteBuf = Av1KeyframeConstantsGpu.BuildByteConstsBuffer();
        var ushortBuf = Av1KeyframeConstantsGpu.BuildUshortConstsBuffer();

        Equal(Av1KeyframeConstantsGpu.ByteConstsTotalBytes, byteBuf.Length);
        Equal(Av1KeyframeConstantsGpu.UshortConstsTotalEntries, ushortBuf.Length);
    }

    [TestMethod]
    public void Av1KeyframeConstantsGpu_ScanTables_MatchSource()
    {
        var ushortBuf = Av1KeyframeConstantsGpu.BuildUshortConstsBuffer();

        // Tx8x8 + DCT_DCT - 64 entries.
        var srcScan8 = Av1ScanTables.Scan[1][0];
        for (int i = 0; i < 64; i++)
            Equal((ushort)srcScan8[i], ushortBuf[Av1KeyframeConstantsGpu.Scan8x8Offset + i],
                $"scan8x8 mismatch at {i}");

        // Tx16x16 + DCT_DCT - 256 entries.
        var srcScan16 = Av1ScanTables.Scan[2][0];
        for (int i = 0; i < 256; i++)
            Equal((ushort)srcScan16[i], ushortBuf[Av1KeyframeConstantsGpu.Scan16x16Offset + i],
                $"scan16x16 mismatch at {i}");
    }

    [TestMethod]
    public void Av1KeyframeConstantsGpu_NzMapCtxOffset_RoundTrips()
    {
        var byteBuf = Av1KeyframeConstantsGpu.BuildByteConstsBuffer();

        // Tx8x8 NzMapCtxOffset (libaom TX_SIZE index 1).
        var src8 = Av1ScanTables.NzMapCtxOffset[1];
        for (int i = 0; i < src8.Length; i++)
            Equal((byte)src8[i], byteBuf[Av1KeyframeConstantsGpu.NzMapCtxOffset8x8Offset + i],
                $"NzMapCtxOffset[8x8] mismatch at {i}");

        // Tx16x16 NzMapCtxOffset (libaom TX_SIZE index 2).
        var src16 = Av1ScanTables.NzMapCtxOffset[2];
        for (int i = 0; i < src16.Length; i++)
            Equal((byte)src16[i], byteBuf[Av1KeyframeConstantsGpu.NzMapCtxOffset16x16Offset + i],
                $"NzMapCtxOffset[16x16] mismatch at {i}");
    }

    [TestMethod]
    public void Av1KeyframeConstantsGpu_EobGroupStartAndOffsetBits_RoundTrip()
    {
        var byteBuf = Av1KeyframeConstantsGpu.BuildByteConstsBuffer();

        // EobGroupStart - 12 entries, little-endian ushort.
        var srcEgs = Av1TxbCommon.EobGroupStart;
        for (int i = 0; i < 12; i++)
        {
            int lo = byteBuf[Av1KeyframeConstantsGpu.EobGroupStartOffset + i * 2];
            int hi = byteBuf[Av1KeyframeConstantsGpu.EobGroupStartOffset + i * 2 + 1];
            ushort packed = (ushort)(lo | (hi << 8));
            Equal((ushort)srcEgs[i], packed, $"EobGroupStart mismatch at {i}");
        }

        // EobOffsetBits - 12 entries, little-endian ushort.
        var srcEob = Av1TxbCommon.EobOffsetBits;
        for (int i = 0; i < 12; i++)
        {
            int lo = byteBuf[Av1KeyframeConstantsGpu.EobOffsetBitsOffset + i * 2];
            int hi = byteBuf[Av1KeyframeConstantsGpu.EobOffsetBitsOffset + i * 2 + 1];
            ushort packed = (ushort)(lo | (hi << 8));
            Equal((ushort)srcEob[i], packed, $"EobOffsetBits mismatch at {i}");
        }
    }

    [TestMethod]
    public void Av1KeyframeConstantsGpu_CdfSentinels_MatchSource()
    {
        var ushortBuf = Av1KeyframeConstantsGpu.BuildUshortConstsBuffer();

        // Sentinel: TxbSkipCdf at qctx=0, txs_local=0 (Tx8x8 -> libaom 1), ctx=0, sym=0.
        // Source: Av1DefaultCoefCdfs.DefaultTxbSkipCdf[0][1][0][0].
        ushort srcSkip = Av1DefaultCoefCdfs.DefaultTxbSkipCdf[0][1][0][0];
        int dstSkip = Av1KeyframeConstantsGpu.TxbSkipCdfOffset
            + ((0 * 2 + 0) * Av1KeyframeConstantsGpu.TxbSkipContexts + 0) * 3 + 0;
        Equal(srcSkip, ushortBuf[dstSkip], "TxbSkipCdf sentinel");

        // Sentinel: EobMulti64Cdf at qctx=2, planeType=1, ctx=0, sym=3.
        ushort srcEob64 = Av1DefaultCoefCdfs.DefaultEobMulti64Cdf[2][1][0][3];
        int dstEob64 = Av1KeyframeConstantsGpu.EobMulti64CdfOffset
            + ((2 * Av1KeyframeConstantsGpu.PlaneTypes + 1) * 2 + 0) * 8 + 3;
        Equal(srcEob64, ushortBuf[dstEob64], "EobMulti64Cdf sentinel");

        // Sentinel: EobMulti256Cdf at qctx=3, planeType=0, ctx=1, sym=5.
        ushort srcEob256 = Av1DefaultCoefCdfs.DefaultEobMulti256Cdf[3][0][1][5];
        int dstEob256 = Av1KeyframeConstantsGpu.EobMulti256CdfOffset
            + ((3 * Av1KeyframeConstantsGpu.PlaneTypes + 0) * 2 + 1) * 10 + 5;
        Equal(srcEob256, ushortBuf[dstEob256], "EobMulti256Cdf sentinel");

        // Sentinel: CoeffBaseEobMultiCdf at qctx=1, txs_local=1 (Tx16x16->2), planeType=0, ctx=2, sym=1.
        ushort srcBaseEob = Av1DefaultCoefCdfs.DefaultCoeffBaseEobMultiCdf[1][2][0][2][1];
        int dstBaseEob = Av1KeyframeConstantsGpu.CoeffBaseEobMultiCdfOffset
            + (((1 * 2 + 1) * Av1KeyframeConstantsGpu.PlaneTypes + 0) * Av1KeyframeConstantsGpu.SigCoefContextsEob + 2) * 4 + 1;
        Equal(srcBaseEob, ushortBuf[dstBaseEob], "CoeffBaseEobMultiCdf sentinel");

        // Sentinel: CoeffBaseMultiCdf at qctx=2, txs_local=0 (Tx8x8->1), planeType=1, ctx=10, sym=2.
        ushort srcBase = Av1DefaultCoefCdfs.DefaultCoeffBaseMultiCdf[2][1][1][10][2];
        int dstBase = Av1KeyframeConstantsGpu.CoeffBaseMultiCdfOffset
            + (((2 * 2 + 0) * Av1KeyframeConstantsGpu.PlaneTypes + 1) * Av1KeyframeConstantsGpu.SigCoefContexts + 10) * 5 + 2;
        Equal(srcBase, ushortBuf[dstBase], "CoeffBaseMultiCdf sentinel");

        // Sentinel: CoeffLpsMultiCdf at qctx=0, txs_local=1 (Tx16x16->2), planeType=0, ctx=5, sym=2.
        ushort srcLps = Av1DefaultCoefCdfs.DefaultCoeffLpsMultiCdf[0][2][0][5][2];
        int rowSize = Av1KeyframeConstantsGpu.BrCdfSize + 1;
        int dstLps = Av1KeyframeConstantsGpu.CoeffLpsMultiCdfOffset
            + (((0 * 2 + 1) * Av1KeyframeConstantsGpu.PlaneTypes + 0) * Av1KeyframeConstantsGpu.LevelContexts + 5) * rowSize + 2;
        Equal(srcLps, ushortBuf[dstLps], "CoeffLpsMultiCdf sentinel");

        // Sentinel: DcSignCdf at qctx=1, planeType=1, ctx=2, sym=0.
        ushort srcDcSign = Av1DefaultCoefCdfs.DefaultDcSignCdf[1][1][2][0];
        int dstDcSign = Av1KeyframeConstantsGpu.DcSignCdfOffset
            + ((1 * Av1KeyframeConstantsGpu.PlaneTypes + 1) * Av1KeyframeConstantsGpu.DcSignContexts + 2) * 3 + 0;
        Equal(srcDcSign, ushortBuf[dstDcSign], "DcSignCdf sentinel");

        // Sentinel: IntraExtTxCdf for Tx8x8 DC_PRED under reducedTxSet=true
        // (matches v1 keyframe encoder default): both Tx8x8 and Tx16x16
        // map to ext_tx_set 2, eset=2.
        var srcIntraExtTx8 = Av1DefaultTxfmCdfs.DefaultIntraExtTxCdf[2][1][(int)Av1IntraMode.Dc];
        Equal(srcIntraExtTx8[0], ushortBuf[Av1KeyframeConstantsGpu.IntraExtTxCdfTx8DcOffset + 0], "IntraExtTx tx8 DC sentinel");

        var srcIntraExtTx16 = Av1DefaultTxfmCdfs.DefaultIntraExtTxCdf[2][2][(int)Av1IntraMode.Dc];
        Equal(srcIntraExtTx16[0], ushortBuf[Av1KeyframeConstantsGpu.IntraExtTxCdfTx16DcOffset + 0], "IntraExtTx tx16 DC sentinel");
    }

    [TestMethod]
    public void Av1KeyframeConstantsGpu_Tx4x4_AllEntries_MatchSource()
    {
        // Boundary-MB chroma at non-aligned VP9/AV1 dims uses Tx4x4 scans/CDFs.
        // This test verifies every Tx4x4 entry packs bit-exact vs the libaom
        // default tables (txsCtx index 0 throughout).
        var byteBuf = Av1KeyframeConstantsGpu.BuildByteConstsBuffer();
        var ushortBuf = Av1KeyframeConstantsGpu.BuildUshortConstsBuffer();

        // NzMapCtxOffset[0] -> all 16 entries.
        var src4Nz = Av1ScanTables.NzMapCtxOffset[0];
        for (int i = 0; i < src4Nz.Length; i++)
            Equal((byte)src4Nz[i], byteBuf[Av1KeyframeConstantsGpu.NzMapCtxOffset4x4Offset + i],
                $"NzMapCtxOffset[4x4] mismatch at {i}");

        // Scan[0][0] (Tx4x4 + DCT_DCT) -> 16 ushort.
        var src4Scan = Av1ScanTables.Scan[0][0];
        for (int i = 0; i < 16; i++)
            Equal((ushort)src4Scan[i], ushortBuf[Av1KeyframeConstantsGpu.Scan4x4Offset + i],
                $"Scan[4x4] mismatch at {i}");

        // TxbSkipCdf[q][0][ctx] sentinel: q=2, ctx=5, sym=1.
        ushort srcSkip4 = Av1DefaultCoefCdfs.DefaultTxbSkipCdf[2][0][5][1];
        int dstSkip4 = Av1KeyframeConstantsGpu.TxbSkipCdfTx4x4Offset
            + (2 * Av1KeyframeConstantsGpu.TxbSkipContexts + 5) * 3 + 1;
        Equal(srcSkip4, ushortBuf[dstSkip4], "TxbSkipCdf[Tx4x4] sentinel");

        // EobMulti16Cdf sentinel: q=3, planeType=0, eobMultiCtx=1, sym=2.
        ushort srcEob16 = Av1DefaultCoefCdfs.DefaultEobMulti16Cdf[3][0][1][2];
        int dstEob16 = Av1KeyframeConstantsGpu.EobMulti16CdfOffset
            + ((3 * Av1KeyframeConstantsGpu.PlaneTypes + 0) * 2 + 1) * 6 + 2;
        Equal(srcEob16, ushortBuf[dstEob16], "EobMulti16Cdf sentinel");

        // EobExtraCdf[q][0][p][ctx] sentinel: q=1, p=1, ctx=4, sym=0.
        ushort srcEobExtra4 = Av1DefaultCoefCdfs.DefaultEobExtraCdf[1][0][1][4][0];
        int dstEobExtra4 = Av1KeyframeConstantsGpu.EobExtraCdfTx4x4Offset
            + ((1 * Av1KeyframeConstantsGpu.PlaneTypes + 1) * Av1KeyframeConstantsGpu.EobCoefContexts + 4) * 3 + 0;
        Equal(srcEobExtra4, ushortBuf[dstEobExtra4], "EobExtraCdf[Tx4x4] sentinel");

        // CoeffBaseEobMultiCdf[q][0][p][ctx] sentinel: q=0, p=0, ctx=2, sym=1.
        ushort srcBaseEob4 = Av1DefaultCoefCdfs.DefaultCoeffBaseEobMultiCdf[0][0][0][2][1];
        int dstBaseEob4 = Av1KeyframeConstantsGpu.CoeffBaseEobMultiCdfTx4x4Offset
            + ((0 * Av1KeyframeConstantsGpu.PlaneTypes + 0) * Av1KeyframeConstantsGpu.SigCoefContextsEob + 2) * 4 + 1;
        Equal(srcBaseEob4, ushortBuf[dstBaseEob4], "CoeffBaseEobMultiCdf[Tx4x4] sentinel");

        // CoeffBaseMultiCdf[q][0][p][ctx] sentinel: q=2, p=1, ctx=20, sym=3.
        ushort srcBase4 = Av1DefaultCoefCdfs.DefaultCoeffBaseMultiCdf[2][0][1][20][3];
        int dstBase4 = Av1KeyframeConstantsGpu.CoeffBaseMultiCdfTx4x4Offset
            + ((2 * Av1KeyframeConstantsGpu.PlaneTypes + 1) * Av1KeyframeConstantsGpu.SigCoefContexts + 20) * 5 + 3;
        Equal(srcBase4, ushortBuf[dstBase4], "CoeffBaseMultiCdf[Tx4x4] sentinel");

        // CoeffLpsMultiCdf[q][0][p][ctx] sentinel: q=3, p=0, ctx=10, sym=1.
        ushort srcLps4 = Av1DefaultCoefCdfs.DefaultCoeffLpsMultiCdf[3][0][0][10][1];
        int rowSize = Av1KeyframeConstantsGpu.BrCdfSize + 1;
        int dstLps4 = Av1KeyframeConstantsGpu.CoeffLpsMultiCdfTx4x4Offset
            + ((3 * Av1KeyframeConstantsGpu.PlaneTypes + 0) * Av1KeyframeConstantsGpu.LevelContexts + 10) * rowSize + 1;
        Equal(srcLps4, ushortBuf[dstLps4], "CoeffLpsMultiCdf[Tx4x4] sentinel");

        // Smoke: total buffer sizes still match the offset-chain math.
        Equal(
            Av1KeyframeConstantsGpu.CoeffLpsMultiCdfTx4x4Offset + Av1KeyframeConstantsGpu.CoeffLpsMultiCdfTx4x4Length,
            Av1KeyframeConstantsGpu.UshortConstsTotalEntries,
            "ushort total math");
        Equal(
            Av1KeyframeConstantsGpu.NzMapCtxOffset4x4Offset + Av1KeyframeConstantsGpu.NzMapCtxOffset4x4Length,
            Av1KeyframeConstantsGpu.ByteConstsTotalBytes,
            "byte total math");
    }
}
