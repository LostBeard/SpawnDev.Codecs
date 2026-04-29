// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Unified constant tables for the AV1 v1 keyframe encoder + decoder
// GPU kernels. Packs every CDF / scan / context-offset table the
// per-block walker + entropy stage need into two big buffers (one
// byte, one ushort) that the host uploads ONCE per accelerator and
// reuses across every frame.
//
// Why pack everything: ILGPU's Action&lt;&gt; entry-point arg budget is
// 14-15. A from-scratch AV1 entropy kernel needs ~10 different CDF
// tables + 2 scan tables + 2 NzMapCtxOffset tables + libaom constant
// lookups. Naively that's a dozen separate ArrayViews plus per-table
// length params. Pack them into one byte buffer + one ushort buffer
// with offset constants, and the kernel signature stays in budget.
//
// V1 simplifications match Av1KeyframeEncoder.cs:
//   - Profile 0 (8-bit 4:2:0)
//   - BLOCK_16X16 partitioning, TX_16X16 + DCT_DCT for Y
//   - Chroma uses TX_8X8 + DCT_DCT
//   - DC_PRED only (intra mode)
//   - Single tile (no per-tile size)
//   - No reduced TX set; intra ext-tx CDFs needed for both sizes
//
// Byte buffer layout:
//   [0..63]    NzMapCtxOffset for Tx8x8  (64 sbyte cast to byte)
//   [64..319]  NzMapCtxOffset for Tx16x16 (256 sbyte cast to byte)
//   [320..343] EobGroupStart[12] packed as ushort -> low 24 bytes
//   [344..367] EobOffsetBits[12] packed as ushort -> 24 bytes
//   [368..380] IntraModeContext[13] - libaom intra_mode_context table
//
// Ushort buffer layout:
//   [0..63]    Scan for Tx8x8 DCT_DCT (64 entries, the libaom
//              zigzag for 8x8)
//   [64..319]  Scan for Tx16x16 DCT_DCT (256 entries)
//   [320..631] TxbSkipCdf - [qctx=4][txsCtx={1,2}][txbSkipCtx=13][3]
//              flat 4*2*13*3 = 312 ushort
//   [632..759] EobMulti64Cdf (Tx8x8 EOB classification)
//              [qctx=4][planeType=2][eobMultiCtx=2][8] = 128 ushort
//   [760..919] EobMulti256Cdf (Tx16x16 EOB classification)
//              [qctx=4][planeType=2][eobMultiCtx=2][10] = 160 ushort
//   [920..1351] EobExtraCdf - [qctx=4][txsCtx={1,2}][planeType=2][eobCtx=9][3]
//              flat 4*2*2*9*3 = 432 ushort
//   [1352..1607] CoeffBaseEobMultiCdf - [qctx=4][txsCtx={1,2}][planeType=2][SigCoefContextsEob=4][4]
//              flat 4*2*2*4*4 = 256 ushort
//   [1608..4967] CoeffBaseMultiCdf - [qctx=4][txsCtx={1,2}][planeType=2][SigCoefContexts=42][5]
//              flat 4*2*2*42*5 = 3360 ushort
//   [4968..6647] CoeffLpsMultiCdf - [qctx=4][min(txsCtx,3)={1,2}][planeType=2][LevelContexts=21][BR_CDF_SIZE=4]
//              flat 4*2*2*21*4 = 1344 ushort  (BR_CDF_SIZE=4 not 5)
//   [6648..6719] DcSignCdf - [qctx=4][planeType=2][DcSignContexts=3][3]
//              flat 4*2*3*3 = 72 ushort
//   [6720..6727] IntraExtTxCdf for Tx8x8 DC_PRED set 3 (8 syms)
//   [6728..6733] IntraExtTxCdf for Tx16x16 DC_PRED set 2 (6 syms)
//   [6734..6742] SkipTxfmCdf - [skipCtx=3][3]
//              3*3 = 9 ushort
//   [6743..6962] PartitionCdf - [partitionCtx=20][11]
//              20*11 = 220 ushort (full table; v1 only hits ctx 4..15)
//   [6963..7312] KfYModeCdf - [aboveCtx=5][leftCtx=5][14]
//              5*5*14 = 350 ushort
//   [7313..7327] UvModeCdfV1Row - DefaultUvModeCdf[cflAllowed=1][yMode=DC][15]
//              15 ushort (only the row v1 needs)
//
// All CDFs are stored as inverse CDF (ICDF). Compatible with the
// Av1RangeEncoderGpu / Av1RangeDecoderGpu helpers' EncodeCdfQ15 /
// DecodeCdfQ15 calling convention exactly.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// Builder + offset constants for the unified AV1 v1 keyframe
/// encoder/decoder constant buffers.
/// </summary>
public static class Av1KeyframeConstantsGpu
{
    // ---- libaom constants exposed for kernel-side use ----

    /// <summary>libaom TOKEN_CDF_Q_CTXS = 4.</summary>
    public const int TokenCdfQCtxs = 4;
    /// <summary>libaom PLANE_TYPES = 2 (Y + UV).</summary>
    public const int PlaneTypes = 2;
    /// <summary>libaom DC_SIGN_CONTEXTS = 3.</summary>
    public const int DcSignContexts = 3;
    /// <summary>libaom TXB_SKIP_CONTEXTS = 13.</summary>
    public const int TxbSkipContexts = 13;
    /// <summary>libaom EOB_COEF_CONTEXTS = 9.</summary>
    public const int EobCoefContexts = 9;
    /// <summary>libaom SIG_COEF_CONTEXTS = SIG_COEF_CONTEXTS_2D + SIG_COEF_CONTEXTS_1D = 26 + 16 = 42.</summary>
    public const int SigCoefContexts = 42;
    /// <summary>libaom SIG_COEF_CONTEXTS_EOB = 4.</summary>
    public const int SigCoefContextsEob = 4;
    /// <summary>libaom LEVEL_CONTEXTS = 21.</summary>
    public const int LevelContexts = 21;
    /// <summary>libaom BR_CDF_SIZE = 4.</summary>
    public const int BrCdfSize = 4;

    /// <summary>txs_ctx for Tx8x8 (= GetTxSizeEntropyCtx(Tx8x8)).</summary>
    public const int TxsCtxTx8 = 1;
    /// <summary>txs_ctx for Tx16x16 (= GetTxSizeEntropyCtx(Tx16x16)).</summary>
    public const int TxsCtxTx16 = 2;

    // ---- byte buffer layout ----

    /// <summary>Offset of Tx8x8 NzMapCtxOffset (64 entries cast from sbyte to byte).</summary>
    public const int NzMapCtxOffset8x8Offset = 0;
    /// <summary>Length of Tx8x8 NzMapCtxOffset.</summary>
    public const int NzMapCtxOffset8x8Length = 64;

    /// <summary>Offset of Tx16x16 NzMapCtxOffset (256 entries cast from sbyte to byte).</summary>
    public const int NzMapCtxOffset16x16Offset = NzMapCtxOffset8x8Offset + NzMapCtxOffset8x8Length;
    /// <summary>Length of Tx16x16 NzMapCtxOffset.</summary>
    public const int NzMapCtxOffset16x16Length = 256;

    /// <summary>EobGroupStart[12] packed as 24 bytes (12 ushort little-endian).</summary>
    public const int EobGroupStartOffset = NzMapCtxOffset16x16Offset + NzMapCtxOffset16x16Length;
    /// <summary>Length: 12 entries x 2 bytes each = 24 bytes.</summary>
    public const int EobGroupStartLength = 24;

    /// <summary>EobOffsetBits[12] packed as 24 bytes (12 ushort little-endian).</summary>
    public const int EobOffsetBitsOffset = EobGroupStartOffset + EobGroupStartLength;
    /// <summary>Length: 12 entries x 2 bytes each = 24 bytes.</summary>
    public const int EobOffsetBitsLength = 24;

    /// <summary>libaom <c>intra_mode_context[INTRA_MODES]</c> packed as 13 bytes
    /// (the table values fit in 0..4 so byte storage is safe).</summary>
    public const int IntraModeContextOffset = EobOffsetBitsOffset + EobOffsetBitsLength;
    /// <summary>Length: 13 entries (one per INTRA_MODES enum value).</summary>
    public const int IntraModeContextLength = 13;

    /// <summary>Total byte buffer size.</summary>
    public const int ByteConstsTotalBytes = IntraModeContextOffset + IntraModeContextLength;

    // ---- ushort buffer layout ----

    /// <summary>Tx8x8 DCT_DCT scan order (libaom default zigzag).</summary>
    public const int Scan8x8Offset = 0;
    /// <summary>Length of 8x8 scan.</summary>
    public const int Scan8x8Length = 64;

    /// <summary>Tx16x16 DCT_DCT scan order.</summary>
    public const int Scan16x16Offset = Scan8x8Offset + Scan8x8Length;
    /// <summary>Length of 16x16 scan.</summary>
    public const int Scan16x16Length = 256;

    /// <summary>TxbSkipCdf flat: [qctx][txs_ctx_local][txbSkipCtx][3].
    /// txs_ctx_local: 0 = Tx8x8 (libaom txsCtx=1), 1 = Tx16x16 (txsCtx=2).</summary>
    public const int TxbSkipCdfOffset = Scan16x16Offset + Scan16x16Length;
    /// <summary>Length: 4 * 2 * 13 * 3 = 312.</summary>
    public const int TxbSkipCdfLength = TokenCdfQCtxs * 2 * TxbSkipContexts * 3;

    /// <summary>EobMulti64Cdf (used for Tx8x8) flat: [qctx][planeType][eobMultiCtx][8].</summary>
    public const int EobMulti64CdfOffset = TxbSkipCdfOffset + TxbSkipCdfLength;
    /// <summary>Length: 4 * 2 * 2 * 8 = 128.</summary>
    public const int EobMulti64CdfLength = TokenCdfQCtxs * PlaneTypes * 2 * 8;

    /// <summary>EobMulti256Cdf (used for Tx16x16) flat: [qctx][planeType][eobMultiCtx][10].</summary>
    public const int EobMulti256CdfOffset = EobMulti64CdfOffset + EobMulti64CdfLength;
    /// <summary>Length: 4 * 2 * 2 * 10 = 160.</summary>
    public const int EobMulti256CdfLength = TokenCdfQCtxs * PlaneTypes * 2 * 10;

    /// <summary>EobExtraCdf flat: [qctx][txs_ctx_local][planeType][eobCtx][3].</summary>
    public const int EobExtraCdfOffset = EobMulti256CdfOffset + EobMulti256CdfLength;
    /// <summary>Length: 4 * 2 * 2 * 9 * 3 = 432.</summary>
    public const int EobExtraCdfLength = TokenCdfQCtxs * 2 * PlaneTypes * EobCoefContexts * 3;

    /// <summary>CoeffBaseEobMultiCdf flat: [qctx][txs_ctx_local][planeType][SigCoefContextsEob][4].</summary>
    public const int CoeffBaseEobMultiCdfOffset = EobExtraCdfOffset + EobExtraCdfLength;
    /// <summary>Length: 4 * 2 * 2 * 4 * 4 = 256.</summary>
    public const int CoeffBaseEobMultiCdfLength = TokenCdfQCtxs * 2 * PlaneTypes * SigCoefContextsEob * 4;

    /// <summary>CoeffBaseMultiCdf flat: [qctx][txs_ctx_local][planeType][SigCoefContexts][5].</summary>
    public const int CoeffBaseMultiCdfOffset = CoeffBaseEobMultiCdfOffset + CoeffBaseEobMultiCdfLength;
    /// <summary>Length: 4 * 2 * 2 * 42 * 5 = 3360.</summary>
    public const int CoeffBaseMultiCdfLength = TokenCdfQCtxs * 2 * PlaneTypes * SigCoefContexts * 5;

    /// <summary>CoeffLpsMultiCdf flat: [qctx][txs_ctx_local][planeType][LevelContexts][BR_CDF_SIZE+1].
    /// libaom CDF_SIZE(BR_CDF_SIZE) = BR_CDF_SIZE + 1 = 5 entries per row.</summary>
    public const int CoeffLpsMultiCdfOffset = CoeffBaseMultiCdfOffset + CoeffBaseMultiCdfLength;
    /// <summary>Length: 4 * 2 * 2 * 21 * 5 = 1680.</summary>
    public const int CoeffLpsMultiCdfLength = TokenCdfQCtxs * 2 * PlaneTypes * LevelContexts * (BrCdfSize + 1);

    /// <summary>DcSignCdf flat: [qctx][planeType][DcSignContexts][3].</summary>
    public const int DcSignCdfOffset = CoeffLpsMultiCdfOffset + CoeffLpsMultiCdfLength;
    /// <summary>Length: 4 * 2 * 3 * 3 = 72.</summary>
    public const int DcSignCdfLength = TokenCdfQCtxs * PlaneTypes * DcSignContexts * 3;

    /// <summary>IntraExtTxCdf row for Tx8x8 + DC_PRED + ext_tx_set 2 (DTT4_IDTX,
    /// 5 syms - the v1 encoder uses reducedTxSet=true so both sizes use set 2).
    /// CDF row width = libaom CDF_SIZE(5) = 6.</summary>
    public const int IntraExtTxCdfTx8DcOffset = DcSignCdfOffset + DcSignCdfLength;
    /// <summary>Length: 6 entries.</summary>
    public const int IntraExtTxCdfTx8DcLength = 6;

    /// <summary>IntraExtTxCdf row for Tx16x16 + DC_PRED + ext_tx_set 2 (DTT4_IDTX,
    /// 5 syms). CDF row width = libaom CDF_SIZE(5) = 6.</summary>
    public const int IntraExtTxCdfTx16DcOffset = IntraExtTxCdfTx8DcOffset + IntraExtTxCdfTx8DcLength;
    /// <summary>Length: 6 entries.</summary>
    public const int IntraExtTxCdfTx16DcLength = 6;

    /// <summary>SkipTxfmCdf flat: [skipCtx=3][CDF_SIZE(2)=3].</summary>
    public const int SkipTxfmCdfOffset = IntraExtTxCdfTx16DcOffset + IntraExtTxCdfTx16DcLength;
    /// <summary>Length: 3 contexts * 3 = 9.</summary>
    public const int SkipTxfmCdfLength = 3 * 3;

    /// <summary>libaom PARTITION_CONTEXTS = 20 (5 sizes * 4 ploff combos).
    /// CDF rows are CDF_SIZE(EXT_PARTITION_TYPES) = 11 entries each. Smaller-symbol
    /// contexts (4 syms / 8 syms) zero-pad in the trailing entries; the
    /// kernel passes the actual nsyms based on bsize so the padding is
    /// never read by EncodeCdfQ15.</summary>
    public const int PartitionCdfOffset = SkipTxfmCdfOffset + SkipTxfmCdfLength;
    /// <summary>Length: 20 contexts * 11 = 220.</summary>
    public const int PartitionCdfLength = 20 * 11;

    /// <summary>KfYModeCdf flat: [aboveCtx=5][leftCtx=5][CDF_SIZE(13)=14].</summary>
    public const int KfYModeCdfOffset = PartitionCdfOffset + PartitionCdfLength;
    /// <summary>Length: 5 * 5 * 14 = 350.</summary>
    public const int KfYModeCdfLength = 5 * 5 * 14;

    /// <summary>UvModeCdf row for v1's only emission: cflAllowed=1, yMode=DC.
    /// CDF_SIZE(UV_INTRA_MODES=14) = 15.</summary>
    public const int UvModeCdfV1RowOffset = KfYModeCdfOffset + KfYModeCdfLength;
    /// <summary>Length: 15 entries.</summary>
    public const int UvModeCdfV1RowLength = 15;

    /// <summary>Total ushort buffer entries.</summary>
    public const int UshortConstsTotalEntries = UvModeCdfV1RowOffset + UvModeCdfV1RowLength;

    /// <summary>
    /// Build the byte constants buffer for upload. Caller materialises
    /// once per accelerator and reuses across every frame.
    /// </summary>
    public static byte[] BuildByteConstsBuffer()
    {
        var buf = new byte[ByteConstsTotalBytes];

        // NzMapCtxOffset for Tx8x8 (libaom TX_SIZE index 1).
        var off8 = Av1ScanTables.NzMapCtxOffset[1];
        for (int i = 0; i < off8.Length; i++)
            buf[NzMapCtxOffset8x8Offset + i] = (byte)off8[i];

        // NzMapCtxOffset for Tx16x16 (libaom TX_SIZE index 2).
        var off16 = Av1ScanTables.NzMapCtxOffset[2];
        for (int i = 0; i < off16.Length; i++)
            buf[NzMapCtxOffset16x16Offset + i] = (byte)off16[i];

        // EobGroupStart packed as little-endian ushort.
        var egs = Av1TxbCommon.EobGroupStart;
        for (int i = 0; i < 12; i++)
        {
            ushort v = (ushort)egs[i];
            buf[EobGroupStartOffset + i * 2] = (byte)(v & 0xFF);
            buf[EobGroupStartOffset + i * 2 + 1] = (byte)(v >> 8);
        }

        // EobOffsetBits packed as little-endian ushort.
        var eob = Av1TxbCommon.EobOffsetBits;
        for (int i = 0; i < 12; i++)
        {
            ushort v = (ushort)eob[i];
            buf[EobOffsetBitsOffset + i * 2] = (byte)(v & 0xFF);
            buf[EobOffsetBitsOffset + i * 2 + 1] = (byte)(v >> 8);
        }

        // IntraModeContext (each value fits 0..4, safe in byte storage).
        var imc = Av1ModeInfoReader.IntraModeContext;
        for (int i = 0; i < IntraModeContextLength && i < imc.Length; i++)
            buf[IntraModeContextOffset + i] = (byte)imc[i];

        return buf;
    }

    /// <summary>
    /// Build the ushort constants buffer for upload. Caller materialises
    /// once per accelerator and reuses across every frame.
    /// </summary>
    public static ushort[] BuildUshortConstsBuffer()
    {
        var buf = new ushort[UshortConstsTotalEntries];

        // Scan tables. Av1ScanTables.Scan[ts][tt] is short[]; cast to ushort.
        var scan8 = Av1ScanTables.Scan[1][0]; // Tx8x8 + DCT_DCT
        for (int i = 0; i < 64; i++) buf[Scan8x8Offset + i] = (ushort)scan8[i];
        var scan16 = Av1ScanTables.Scan[2][0]; // Tx16x16 + DCT_DCT
        for (int i = 0; i < 256; i++) buf[Scan16x16Offset + i] = (ushort)scan16[i];

        // TxbSkipCdf - flatten [qctx][txs_ctx_local][txbSkipCtx][3] from
        // libaom layout [qctx][txs_ctx_libaom][txbSkipCtx][3]. txs_ctx_local
        // 0 = libaom 1, 1 = libaom 2.
        PackTxbSkipCdf(buf);
        // EobMulti64Cdf for Tx8x8 EOB classification.
        PackEobMultiCdf(buf, EobMulti64CdfOffset, Av1DefaultCoefCdfs.DefaultEobMulti64Cdf, 7);
        // EobMulti256Cdf for Tx16x16 EOB classification.
        PackEobMultiCdf(buf, EobMulti256CdfOffset, Av1DefaultCoefCdfs.DefaultEobMulti256Cdf, 9);
        // EobExtraCdf - 5D table flatten.
        PackEobExtraCdf(buf);
        // CoeffBaseEobMultiCdf - 5D table flatten.
        PackCoeffBaseEobMultiCdf(buf);
        // CoeffBaseMultiCdf - 5D table flatten.
        PackCoeffBaseMultiCdf(buf);
        // CoeffLpsMultiCdf - 5D table flatten.
        PackCoeffLpsMultiCdf(buf);
        // DcSignCdf - 4D table flatten.
        PackDcSignCdf(buf);
        // IntraExtTxCdf - the two specific rows for Tx8x8 / Tx16x16 DC_PRED
        // under the v1 encoder's reducedTxSet=true config. Per libaom
        // GetExtTxSetType with reducedTxSet=true: BOTH Tx8x8 and Tx16x16
        // map to ext_tx_set 2 (DTT4_IDTX) -> eset = ExtTxSetIndexIntra[2] = 2.
        // Tx8x8 squareTxSize = 1; Tx16x16 squareTxSize = 2.
        var rowTx8 = Av1DefaultTxfmCdfs.DefaultIntraExtTxCdf[2][1][(int)Av1IntraMode.Dc];
        for (int i = 0; i < IntraExtTxCdfTx8DcLength && i < rowTx8.Length; i++)
            buf[IntraExtTxCdfTx8DcOffset + i] = rowTx8[i];
        var rowTx16 = Av1DefaultTxfmCdfs.DefaultIntraExtTxCdf[2][2][(int)Av1IntraMode.Dc];
        for (int i = 0; i < IntraExtTxCdfTx16DcLength && i < rowTx16.Length; i++)
            buf[IntraExtTxCdfTx16DcOffset + i] = rowTx16[i];

        // SkipTxfmCdf: 3 contexts * 3 entries.
        var skipTxfm = Av1DefaultBlockCdfs.DefaultSkipTxfmCdf;
        for (int c = 0; c < 3; c++)
            for (int s = 0; s < 3 && s < skipTxfm[c].Length; s++)
                buf[SkipTxfmCdfOffset + c * 3 + s] = skipTxfm[c][s];

        // PartitionCdf: 20 contexts * 11 entries (zero-padded for short rows).
        var partition = Av1DefaultPartitionCdfs.DefaultPartitionCdf;
        for (int c = 0; c < 20; c++)
        {
            var row = partition[c];
            int dst = PartitionCdfOffset + c * 11;
            for (int s = 0; s < 11 && s < row.Length; s++) buf[dst + s] = row[s];
        }

        // KfYModeCdf: 5 above * 5 left * 14 entries.
        var kfYMode = Av1DefaultIntraModeCdfs.DefaultKfYModeCdf;
        for (int a = 0; a < 5; a++)
            for (int l = 0; l < 5; l++)
            {
                var row = kfYMode[a][l];
                int dst = KfYModeCdfOffset + (a * 5 + l) * 14;
                for (int s = 0; s < 14 && s < row.Length; s++) buf[dst + s] = row[s];
            }

        // UvModeCdfV1Row: cflAllowed=1, yMode=DC, 15 entries.
        var uvRow = Av1DefaultIntraModeCdfs.DefaultUvModeCdf[1][(int)Av1IntraMode.Dc];
        for (int s = 0; s < 15 && s < uvRow.Length; s++)
            buf[UvModeCdfV1RowOffset + s] = uvRow[s];

        return buf;
    }

    private static void PackTxbSkipCdf(ushort[] buf)
    {
        // Flat layout: ((qctx * 2 + txs_local) * TxbSkipContexts + ctx) * 3 + s
        for (int q = 0; q < TokenCdfQCtxs; q++)
        {
            for (int tsLocal = 0; tsLocal < 2; tsLocal++)
            {
                int tsLibaom = tsLocal + 1; // Tx8x8 -> 1, Tx16x16 -> 2
                for (int ctx = 0; ctx < TxbSkipContexts; ctx++)
                {
                    var row = Av1DefaultCoefCdfs.DefaultTxbSkipCdf[q][tsLibaom][ctx];
                    int dst = TxbSkipCdfOffset + ((q * 2 + tsLocal) * TxbSkipContexts + ctx) * 3;
                    for (int s = 0; s < 3 && s < row.Length; s++) buf[dst + s] = row[s];
                }
            }
        }
    }

    private static void PackEobMultiCdf(ushort[] buf, int offset, ushort[][][][] table, int nsyms)
    {
        // libaom layout: table[qctx][planeType][eobMultiCtx][CDF_SIZE(nsyms) = nsyms+1]
        // Flat layout: ((qctx * PlaneTypes + planeType) * 2 + eobMultiCtx) * (nsyms+1) + s
        int rowSize = nsyms + 1;
        for (int q = 0; q < TokenCdfQCtxs; q++)
        for (int p = 0; p < PlaneTypes; p++)
        for (int c = 0; c < 2; c++)
        {
            var row = table[q][p][c];
            int dst = offset + ((q * PlaneTypes + p) * 2 + c) * rowSize;
            for (int s = 0; s < rowSize && s < row.Length; s++) buf[dst + s] = row[s];
        }
    }

    private static void PackEobExtraCdf(ushort[] buf)
    {
        // libaom layout: DefaultEobExtraCdf[qctx][txsCtx][planeType][eobCtx][3]
        // Flat: (((qctx * 2 + txs_local) * PlaneTypes + p) * EobCoefContexts + c) * 3 + s
        for (int q = 0; q < TokenCdfQCtxs; q++)
        for (int tsLocal = 0; tsLocal < 2; tsLocal++)
        {
            int tsLibaom = tsLocal + 1;
            for (int p = 0; p < PlaneTypes; p++)
            for (int c = 0; c < EobCoefContexts; c++)
            {
                var row = Av1DefaultCoefCdfs.DefaultEobExtraCdf[q][tsLibaom][p][c];
                int dst = EobExtraCdfOffset + (((q * 2 + tsLocal) * PlaneTypes + p) * EobCoefContexts + c) * 3;
                for (int s = 0; s < 3 && s < row.Length; s++) buf[dst + s] = row[s];
            }
        }
    }

    private static void PackCoeffBaseEobMultiCdf(ushort[] buf)
    {
        // libaom: DefaultCoeffBaseEobMultiCdf[qctx][txsCtx][planeType][SigCoefContextsEob][CDF_SIZE(3)=4]
        // Flat: (((qctx * 2 + txs_local) * PlaneTypes + p) * SigCoefContextsEob + c) * 4 + s
        for (int q = 0; q < TokenCdfQCtxs; q++)
        for (int tsLocal = 0; tsLocal < 2; tsLocal++)
        {
            int tsLibaom = tsLocal + 1;
            for (int p = 0; p < PlaneTypes; p++)
            for (int c = 0; c < SigCoefContextsEob; c++)
            {
                var row = Av1DefaultCoefCdfs.DefaultCoeffBaseEobMultiCdf[q][tsLibaom][p][c];
                int dst = CoeffBaseEobMultiCdfOffset + (((q * 2 + tsLocal) * PlaneTypes + p) * SigCoefContextsEob + c) * 4;
                for (int s = 0; s < 4 && s < row.Length; s++) buf[dst + s] = row[s];
            }
        }
    }

    private static void PackCoeffBaseMultiCdf(ushort[] buf)
    {
        // libaom: DefaultCoeffBaseMultiCdf[qctx][txsCtx][planeType][SigCoefContexts][CDF_SIZE(4)=5]
        // Flat: (((qctx * 2 + txs_local) * PlaneTypes + p) * SigCoefContexts + c) * 5 + s
        for (int q = 0; q < TokenCdfQCtxs; q++)
        for (int tsLocal = 0; tsLocal < 2; tsLocal++)
        {
            int tsLibaom = tsLocal + 1;
            for (int p = 0; p < PlaneTypes; p++)
            for (int c = 0; c < SigCoefContexts; c++)
            {
                var row = Av1DefaultCoefCdfs.DefaultCoeffBaseMultiCdf[q][tsLibaom][p][c];
                int dst = CoeffBaseMultiCdfOffset + (((q * 2 + tsLocal) * PlaneTypes + p) * SigCoefContexts + c) * 5;
                for (int s = 0; s < 5 && s < row.Length; s++) buf[dst + s] = row[s];
            }
        }
    }

    private static void PackCoeffLpsMultiCdf(ushort[] buf)
    {
        // libaom: DefaultCoeffLpsMultiCdf[qctx][min(txsCtx,3)][planeType][LevelContexts][CDF_SIZE(BR_CDF_SIZE)=5]
        // We pack txs_local 0 = libaom 1, 1 = libaom 2 (both already &lt;= 3 so min cap is no-op for v1).
        int rowSize = BrCdfSize + 1;
        for (int q = 0; q < TokenCdfQCtxs; q++)
        for (int tsLocal = 0; tsLocal < 2; tsLocal++)
        {
            int tsLibaom = tsLocal + 1;
            for (int p = 0; p < PlaneTypes; p++)
            for (int c = 0; c < LevelContexts; c++)
            {
                var row = Av1DefaultCoefCdfs.DefaultCoeffLpsMultiCdf[q][tsLibaom][p][c];
                int dst = CoeffLpsMultiCdfOffset + (((q * 2 + tsLocal) * PlaneTypes + p) * LevelContexts + c) * rowSize;
                for (int s = 0; s < rowSize && s < row.Length; s++) buf[dst + s] = row[s];
            }
        }
    }

    private static void PackDcSignCdf(ushort[] buf)
    {
        // libaom: DefaultDcSignCdf[qctx][planeType][DcSignContexts][CDF_SIZE(2)=3]
        // Flat: ((qctx * PlaneTypes + p) * DcSignContexts + c) * 3 + s
        for (int q = 0; q < TokenCdfQCtxs; q++)
        for (int p = 0; p < PlaneTypes; p++)
        for (int c = 0; c < DcSignContexts; c++)
        {
            var row = Av1DefaultCoefCdfs.DefaultDcSignCdf[q][p][c];
            int dst = DcSignCdfOffset + ((q * PlaneTypes + p) * DcSignContexts + c) * 3;
            for (int s = 0; s < 3 && s < row.Length; s++) buf[dst + s] = row[s];
        }
    }
}
