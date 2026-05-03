// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 v1 keyframe decode kernel. Single-thread per frame. Symmetric
// companion to Vp9FrameEntropyKernel + Vp9FrameSequentialEncodeKernel:
// reads the bool-coded tile bitstream the encoder produced, walks
// the partition tree in the same z-order, decodes per-MB skip /
// modes / coefs, and reconstructs the recon planes via the same
// per-block inverse pipeline (DC_PRED + dequant + iDCT + add).
//
// What the kernel does NOT do: parse the uncompressed header (host
// extracts width / height / baseQIndex / firstPartitionSize as
// scalar metadata), parse the compressed header (v1 uses defaults
// so compressed header bytes are skipped). Both are pure metadata
// extraction, allowed under the host-as-pure-coordinator rule.
//
// Inputs:
//   - tileBytes: bytes from the tile data partition onward (caller
//     positions the buffer at uncompressed_header_size +
//     first_partition_size).
//   - dequant: 4-int [Y_DC, Y_AC, UV_DC, UV_AC] from
//     Vp9DequantizerComputeKernel.
//   - byteConsts + ushortConsts from Vp9KeyframeConstantsGpu.
//
// Outputs:
//   - yRecon, uRecon, vRecon planes filled in-place.
//
// V1 cap: max miColsAligned = 64 (frame width up to 512 pixels).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// VP9 v1 keyframe decode kernel. Single thread per frame; reads
/// the bool-coded tile bitstream and reconstructs the recon planes
/// entirely on the accelerator.
/// </summary>
public sealed class Vp9KeyframeDecodeKernel : IDisposable
{
    /// <summary>Same v1 cap as the entropy kernel.</summary>
    public const int MaxMiColsAligned = Vp9FrameEntropyKernel.MaxMiColsAligned;

    private readonly Accelerator _accelerator;
    private readonly Action<
        Index1D,
        ArrayView<byte>,
        ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
        ArrayView<int>,
        ArrayView<byte>, ArrayView<ushort>,
        int, int, int> _kernel;

    /// <summary>Compile.</summary>
    public Vp9KeyframeDecodeKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>,
            ArrayView<byte>, ArrayView<byte>, ArrayView<byte>,
            ArrayView<int>,
            ArrayView<byte>, ArrayView<ushort>,
            int, int, int>(DecodeFrameKernel);
    }

    /// <summary>
    /// Decode a single VP9 v1 keyframe tile. Caller has already
    /// extracted scalar metadata from the uncompressed header (width,
    /// height, baseQIndex, firstPartitionSize) and positioned
    /// <paramref name="tileBytes"/> at the first byte after the
    /// compressed header.
    /// </summary>
    public void Run(
        ArrayView<byte> tileBytes,
        ArrayView<byte> yRecon, ArrayView<byte> uRecon, ArrayView<byte> vRecon,
        ArrayView<int> dequant,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        int tileBytesLen, int mbCols, int mbRows)
    {
        if (mbCols <= 0 || (mbCols & 3) != 0)
            throw new ArgumentOutOfRangeException(nameof(mbCols), "mbCols must be a positive multiple of 4 (SB-aligned).");
        if (mbRows <= 0 || (mbRows & 3) != 0)
            throw new ArgumentOutOfRangeException(nameof(mbRows), "mbRows must be a positive multiple of 4 (SB-aligned).");
        if (mbCols * 2 > MaxMiColsAligned)
            throw new ArgumentOutOfRangeException(nameof(mbCols),
                $"v1 decode kernel caps mbCols at {MaxMiColsAligned / 2}; got {mbCols}.");
        if (dequant.Length < 4)
            throw new ArgumentException("dequant must hold 4 ints.", nameof(dequant));
        if (byteConsts.Length < Vp9KeyframeConstantsGpu.ByteConstsTotalBytes)
            throw new ArgumentException("byteConsts too short.", nameof(byteConsts));
        if (ushortConsts.Length < Vp9KeyframeConstantsGpu.UshortConstsTotalEntries)
            throw new ArgumentException("ushortConsts too short.", nameof(ushortConsts));
        _kernel(1,
            tileBytes,
            yRecon, uRecon, vRecon,
            dequant,
            byteConsts, ushortConsts,
            tileBytesLen, mbCols, mbRows);
    }

    private static void DecodeFrameKernel(
        Index1D _,
        ArrayView<byte> tileBytes,
        ArrayView<byte> yRecon, ArrayView<byte> uRecon, ArrayView<byte> vRecon,
        ArrayView<int> dequant,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        int tileBytesLen, int mbCols, int mbRows)
    {
        int yStride = mbCols * 16;
        int uvStride = mbCols * 8;

        int yDcQ = dequant[0];
        int yAcQ = dequant[1];
        int uvDcQ = dequant[2];
        int uvAcQ = dequant[3];

        // Per-thread context arrays (same shape as encoder's entropy kernel).
        var aboveYMode = LocalMemory.Allocate<byte>(MaxMiColsAligned * 2);
        var aboveSkip = LocalMemory.Allocate<byte>(MaxMiColsAligned);
        var abovePartCtx = LocalMemory.Allocate<byte>(MaxMiColsAligned);
        var aboveTxSize = LocalMemory.Allocate<byte>(MaxMiColsAligned);
        var aboveYEntropyCtx = LocalMemory.Allocate<byte>(MaxMiColsAligned * 2);
        var aboveUEntropyCtx = LocalMemory.Allocate<byte>(MaxMiColsAligned);
        var aboveVEntropyCtx = LocalMemory.Allocate<byte>(MaxMiColsAligned);

        var leftYMode = LocalMemory.Allocate<byte>(16);
        var leftSkip = LocalMemory.Allocate<byte>(8);
        var leftPartCtx = LocalMemory.Allocate<byte>(8);
        var leftTxSize = LocalMemory.Allocate<byte>(8);
        var leftYEntropyCtx = LocalMemory.Allocate<byte>(16);
        var leftUEntropyCtx = LocalMemory.Allocate<byte>(8);
        var leftVEntropyCtx = LocalMemory.Allocate<byte>(8);

        // Per-block scratch.
        var coefsShort = LocalMemory.Allocate<short>(256);
        var idctScratch = LocalMemory.Allocate<int>(256);
        var tokenCache = LocalMemory.Allocate<byte>(256);
        var aboveLuma = LocalMemory.Allocate<byte>(16);
        var leftLuma = LocalMemory.Allocate<byte>(16);
        var aboveChroma = LocalMemory.Allocate<byte>(8);
        var leftChroma = LocalMemory.Allocate<byte>(8);

        // Init above arrays.
        for (int i = 0; i < MaxMiColsAligned * 2; i++) aboveYMode[i] = 0;
        for (int i = 0; i < MaxMiColsAligned; i++) { aboveSkip[i] = 0; abovePartCtx[i] = 0; aboveTxSize[i] = 0; }
        for (int i = 0; i < MaxMiColsAligned * 2; i++) aboveYEntropyCtx[i] = 0;
        for (int i = 0; i < MaxMiColsAligned; i++) { aboveUEntropyCtx[i] = 0; aboveVEntropyCtx[i] = 0; }

        // Initialize bool decoder + consume VP9 marker bit.
        var state = Vp8BoolDecoderGpu.Init(tileBytes, 0, tileBytesLen);
        Vp8BoolDecoderGpu.DecodeBool(ref state, tileBytes, 128);

        int sbRows = mbRows >> 2;
        int sbCols = mbCols >> 2;

        for (int sbRow = 0; sbRow < sbRows; sbRow++)
        {
            for (int i = 0; i < 16; i++) { leftYMode[i] = 0; leftYEntropyCtx[i] = 0; }
            for (int i = 0; i < 8; i++) { leftSkip[i] = 0; leftPartCtx[i] = 0; leftTxSize[i] = 0; leftUEntropyCtx[i] = 0; leftVEntropyCtx[i] = 0; }

            for (int sbCol = 0; sbCol < sbCols; sbCol++)
            {
                DecodeSb64(
                    ref state, tileBytes,
                    yRecon, uRecon, vRecon,
                    byteConsts, ushortConsts,
                    aboveYMode, aboveSkip, abovePartCtx, aboveTxSize,
                    aboveYEntropyCtx, aboveUEntropyCtx, aboveVEntropyCtx,
                    leftYMode, leftSkip, leftPartCtx, leftTxSize,
                    leftYEntropyCtx, leftUEntropyCtx, leftVEntropyCtx,
                    coefsShort, idctScratch, tokenCache,
                    aboveLuma, leftLuma, aboveChroma, leftChroma,
                    yDcQ, yAcQ, uvDcQ, uvAcQ,
                    sbRow, sbCol, yStride, uvStride, mbCols);
            }
        }
    }

    private static void DecodeSb64(
        ref Vp8BoolDecoderGpuState state, ArrayView<byte> tileBytes,
        ArrayView<byte> yRecon, ArrayView<byte> uRecon, ArrayView<byte> vRecon,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        ArrayView<byte> aboveYMode, ArrayView<byte> aboveSkip,
        ArrayView<byte> abovePartCtx, ArrayView<byte> aboveTxSize,
        ArrayView<byte> aboveYEntropyCtx, ArrayView<byte> aboveUEntropyCtx, ArrayView<byte> aboveVEntropyCtx,
        ArrayView<byte> leftYMode, ArrayView<byte> leftSkip,
        ArrayView<byte> leftPartCtx, ArrayView<byte> leftTxSize,
        ArrayView<byte> leftYEntropyCtx, ArrayView<byte> leftUEntropyCtx, ArrayView<byte> leftVEntropyCtx,
        ArrayView<short> coefsShort, ArrayView<int> idctScratch, ArrayView<byte> tokenCache,
        ArrayView<byte> aboveLuma, ArrayView<byte> leftLuma, ArrayView<byte> aboveChroma, ArrayView<byte> leftChroma,
        int yDcQ, int yAcQ, int uvDcQ, int uvAcQ,
        int sbRow, int sbCol, int yStride, int uvStride, int mbCols)
    {
        int miRow64 = sbRow * 8;
        int miCol64 = sbCol * 8;

        // Block64x64 partition: SPLIT (verify by reading 3 bits).
        ConsumePartitionSplit(ref state, tileBytes, byteConsts, sizeIdx: 3, bsl: 3, miRow: miRow64, miCol: miCol64, abovePartCtx, leftPartCtx);

        for (int q32 = 0; q32 < 4; q32++)
        {
            int miRow32 = miRow64 + ((q32 & 2) >> 1) * 4;
            int miCol32 = miCol64 + (q32 & 1) * 4;
            DecodeBlock32x32(
                ref state, tileBytes,
                yRecon, uRecon, vRecon,
                byteConsts, ushortConsts,
                aboveYMode, aboveSkip, abovePartCtx, aboveTxSize,
                aboveYEntropyCtx, aboveUEntropyCtx, aboveVEntropyCtx,
                leftYMode, leftSkip, leftPartCtx, leftTxSize,
                leftYEntropyCtx, leftUEntropyCtx, leftVEntropyCtx,
                coefsShort, idctScratch, tokenCache,
                aboveLuma, leftLuma, aboveChroma, leftChroma,
                yDcQ, yAcQ, uvDcQ, uvAcQ,
                miRow32, miCol32, yStride, uvStride, mbCols);
        }
    }

    private static void DecodeBlock32x32(
        ref Vp8BoolDecoderGpuState state, ArrayView<byte> tileBytes,
        ArrayView<byte> yRecon, ArrayView<byte> uRecon, ArrayView<byte> vRecon,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        ArrayView<byte> aboveYMode, ArrayView<byte> aboveSkip,
        ArrayView<byte> abovePartCtx, ArrayView<byte> aboveTxSize,
        ArrayView<byte> aboveYEntropyCtx, ArrayView<byte> aboveUEntropyCtx, ArrayView<byte> aboveVEntropyCtx,
        ArrayView<byte> leftYMode, ArrayView<byte> leftSkip,
        ArrayView<byte> leftPartCtx, ArrayView<byte> leftTxSize,
        ArrayView<byte> leftYEntropyCtx, ArrayView<byte> leftUEntropyCtx, ArrayView<byte> leftVEntropyCtx,
        ArrayView<short> coefsShort, ArrayView<int> idctScratch, ArrayView<byte> tokenCache,
        ArrayView<byte> aboveLuma, ArrayView<byte> leftLuma, ArrayView<byte> aboveChroma, ArrayView<byte> leftChroma,
        int yDcQ, int yAcQ, int uvDcQ, int uvAcQ,
        int miRow32, int miCol32, int yStride, int uvStride, int mbCols)
    {
        ConsumePartitionSplit(ref state, tileBytes, byteConsts, sizeIdx: 2, bsl: 2, miRow: miRow32, miCol: miCol32, abovePartCtx, leftPartCtx);

        for (int q16 = 0; q16 < 4; q16++)
        {
            int miRow16 = miRow32 + ((q16 & 2) >> 1) * 2;
            int miCol16 = miCol32 + (q16 & 1) * 2;
            DecodeBlock16x16(
                ref state, tileBytes,
                yRecon, uRecon, vRecon,
                byteConsts, ushortConsts,
                aboveYMode, aboveSkip, abovePartCtx, aboveTxSize,
                aboveYEntropyCtx, aboveUEntropyCtx, aboveVEntropyCtx,
                leftYMode, leftSkip, leftPartCtx, leftTxSize,
                leftYEntropyCtx, leftUEntropyCtx, leftVEntropyCtx,
                coefsShort, idctScratch, tokenCache,
                aboveLuma, leftLuma, aboveChroma, leftChroma,
                yDcQ, yAcQ, uvDcQ, uvAcQ,
                miRow16, miCol16, yStride, uvStride, mbCols);
        }
    }

    private static void DecodeBlock16x16(
        ref Vp8BoolDecoderGpuState state, ArrayView<byte> tileBytes,
        ArrayView<byte> yRecon, ArrayView<byte> uRecon, ArrayView<byte> vRecon,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        ArrayView<byte> aboveYMode, ArrayView<byte> aboveSkip,
        ArrayView<byte> abovePartCtx, ArrayView<byte> aboveTxSize,
        ArrayView<byte> aboveYEntropyCtx, ArrayView<byte> aboveUEntropyCtx, ArrayView<byte> aboveVEntropyCtx,
        ArrayView<byte> leftYMode, ArrayView<byte> leftSkip,
        ArrayView<byte> leftPartCtx, ArrayView<byte> leftTxSize,
        ArrayView<byte> leftYEntropyCtx, ArrayView<byte> leftUEntropyCtx, ArrayView<byte> leftVEntropyCtx,
        ArrayView<short> coefsShort, ArrayView<int> idctScratch, ArrayView<byte> tokenCache,
        ArrayView<byte> aboveLuma, ArrayView<byte> leftLuma, ArrayView<byte> aboveChroma, ArrayView<byte> leftChroma,
        int yDcQ, int yAcQ, int uvDcQ, int uvAcQ,
        int miRow16, int miCol16, int yStride, int uvStride, int mbCols)
    {
        // NONE at 16x16. Consume single bit 0 at probs[0].
        ConsumePartitionNone(ref state, tileBytes, byteConsts, sizeIdx: 1, bsl: 1, miRow: miRow16, miCol: miCol16, abovePartCtx, leftPartCtx);

        int mbR = miRow16 >> 1;
        int mbC = miCol16 >> 1;

        DecodeLeafBlock(
            ref state, tileBytes,
            yRecon, uRecon, vRecon,
            byteConsts, ushortConsts,
            aboveYMode, aboveSkip, aboveTxSize,
            aboveYEntropyCtx, aboveUEntropyCtx, aboveVEntropyCtx,
            leftYMode, leftSkip, leftTxSize,
            leftYEntropyCtx, leftUEntropyCtx, leftVEntropyCtx,
            coefsShort, idctScratch, tokenCache,
            aboveLuma, leftLuma, aboveChroma, leftChroma,
            yDcQ, yAcQ, uvDcQ, uvAcQ,
            mbR, mbC, miRow16, miCol16, yStride, uvStride, mbCols);

        // Update partition context: subsize=Block16x16, lookup (12, 12), bs=2.
        for (int i = 0; i < 2; i++)
        {
            int c = miCol16 + i;
            int r = (miRow16 + i) & 7;
            if (c < MaxMiColsAligned) abovePartCtx[c] = 12;
            leftPartCtx[r] = 12;
        }
    }

    private static void ConsumePartitionSplit(
        ref Vp8BoolDecoderGpuState state, ArrayView<byte> tileBytes,
        ArrayView<byte> byteConsts,
        int sizeIdx, int bsl, int miRow, int miCol,
        ArrayView<byte> abovePartCtx, ArrayView<byte> leftPartCtx)
    {
        int leftIdx = miRow & 7;
        int aboveIdx = miCol;
        int leftBit = (leftPartCtx[leftIdx] >> bsl) & 1;
        int aboveBit = (abovePartCtx[aboveIdx] >> bsl) & 1;
        int splitState = leftBit * 2 + aboveBit;
        long probsBase = Vp9KeyframeConstantsGpu.KfPartitionProbsOffset
                       + ((long)sizeIdx * 4 + splitState) * 3;
        // SPLIT walks 3 bits = 1, 1, 1. We just consume them; we already
        // know the partition is SPLIT per v1 contract.
        Vp8BoolDecoderGpu.DecodeBool(ref state, tileBytes, byteConsts[probsBase + 0]);
        Vp8BoolDecoderGpu.DecodeBool(ref state, tileBytes, byteConsts[probsBase + 1]);
        Vp8BoolDecoderGpu.DecodeBool(ref state, tileBytes, byteConsts[probsBase + 2]);
    }

    private static void ConsumePartitionNone(
        ref Vp8BoolDecoderGpuState state, ArrayView<byte> tileBytes,
        ArrayView<byte> byteConsts,
        int sizeIdx, int bsl, int miRow, int miCol,
        ArrayView<byte> abovePartCtx, ArrayView<byte> leftPartCtx)
    {
        int leftIdx = miRow & 7;
        int aboveIdx = miCol;
        int leftBit = (leftPartCtx[leftIdx] >> bsl) & 1;
        int aboveBit = (abovePartCtx[aboveIdx] >> bsl) & 1;
        int splitState = leftBit * 2 + aboveBit;
        long probsBase = Vp9KeyframeConstantsGpu.KfPartitionProbsOffset
                       + ((long)sizeIdx * 4 + splitState) * 3;
        Vp8BoolDecoderGpu.DecodeBool(ref state, tileBytes, byteConsts[probsBase + 0]);
    }

    private static void DecodeLeafBlock(
        ref Vp8BoolDecoderGpuState state, ArrayView<byte> tileBytes,
        ArrayView<byte> yRecon, ArrayView<byte> uRecon, ArrayView<byte> vRecon,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        ArrayView<byte> aboveYMode, ArrayView<byte> aboveSkip, ArrayView<byte> aboveTxSize,
        ArrayView<byte> aboveYEntropyCtx, ArrayView<byte> aboveUEntropyCtx, ArrayView<byte> aboveVEntropyCtx,
        ArrayView<byte> leftYMode, ArrayView<byte> leftSkip, ArrayView<byte> leftTxSize,
        ArrayView<byte> leftYEntropyCtx, ArrayView<byte> leftUEntropyCtx, ArrayView<byte> leftVEntropyCtx,
        ArrayView<short> coefsShort, ArrayView<int> idctScratch, ArrayView<byte> tokenCache,
        ArrayView<byte> aboveLuma, ArrayView<byte> leftLuma, ArrayView<byte> aboveChroma, ArrayView<byte> leftChroma,
        int yDcQ, int yAcQ, int uvDcQ, int uvAcQ,
        int mbR, int mbC, int miRow, int miCol, int yStride, int uvStride, int mbCols)
    {
        // ---- Skip flag ----
        int leftIdxMi = miRow & 7;
        int leftSkipBit = leftSkip[leftIdxMi];
        int aboveSkipBit = miCol < MaxMiColsAligned ? aboveSkip[miCol] : 0;
        int skipContext = aboveSkipBit + leftSkipBit;
        int skipFlag = Vp8BoolDecoderGpu.DecodeBool(ref state, tileBytes,
            byteConsts[Vp9KeyframeConstantsGpu.SkipProbsOffset + skipContext]);

        // ---- Y mode ----
        // Walk the intra mode tree at probs[0]. For DC_PRED the first
        // bit is 0 and we leaf at -DcPred = 0.
        int b4Col = miCol * 2;
        int leftB4Idx = (miRow & 7) * 2;
        int aboveYCell = b4Col < MaxMiColsAligned * 2 ? aboveYMode[b4Col] : 0;
        int leftYCell = leftYMode[leftB4Idx];
        long yProbBase = Vp9KeyframeConstantsGpu.KfYModeProbsOffset
                       + (long)(aboveYCell * 10 + leftYCell) * 9;
        // For v1 we know the mode is DC_PRED. We still consume the bit
        // to advance the bool decoder state in lock-step with the
        // encoder's emit.
        Vp8BoolDecoderGpu.DecodeBool(ref state, tileBytes, byteConsts[yProbBase + 0]);

        // ---- UV mode ----
        long uvProbBase = Vp9KeyframeConstantsGpu.KfUvModeProbsOffset;
        Vp8BoolDecoderGpu.DecodeBool(ref state, tileBytes, byteConsts[uvProbBase + 0]);

        // ---- Update mode-info contexts ----
        for (int i = 0; i < 4; i++)
        {
            int c = b4Col + i;
            if (c < MaxMiColsAligned * 2) aboveYMode[c] = 0; // DcPred
        }
        for (int i = 0; i < 4; i++)
        {
            int r = (leftB4Idx + i) & 15;
            leftYMode[r] = 0;
        }
        for (int i = 0; i < 2; i++)
        {
            int c = miCol + i;
            if (c < MaxMiColsAligned) { aboveSkip[c] = (byte)skipFlag; aboveTxSize[c] = (byte)Vp9TxSize.Tx16x16; }
        }
        for (int i = 0; i < 2; i++)
        {
            int r = (leftIdxMi + i) & 7;
            leftSkip[r] = (byte)skipFlag;
            leftTxSize[r] = (byte)Vp9TxSize.Tx16x16;
        }

        // ---- Per-plane decode + reconstruction ----
        long yBase = (long)mbR * 16 * yStride + mbC * 16;
        long uvBase = (long)mbR * 8 * uvStride + mbC * 8;

        // Y plane: Tx16x16 at this 16x16 block.
        DecodePlaneCoefsAndRecon(
            ref state, tileBytes,
            ySrc: default, yRecon: yRecon, yBase, yStride,
            byteConsts, ushortConsts,
            mbC * 4, (mbR & 3) * 4, cellsPerTx: 4,
            isTx4x4: 0, txSizeForCoefProbs: (int)Vp9TxSize.Tx16x16,
            n: 16, planeType: (int)Vp9BlockCoefEnums.PlaneType.Y,
            coefsShort, idctScratch, tokenCache,
            aboveLuma, leftLuma,
            aboveYEntropyCtx, leftYEntropyCtx,
            mbR, mbC,
            dcQ: yDcQ, acQ: yAcQ);

        // U plane.
        DecodePlaneCoefsAndRecon(
            ref state, tileBytes,
            ySrc: default, yRecon: uRecon, uvBase, uvStride,
            byteConsts, ushortConsts,
            mbC * 2, (mbR & 3) * 2, cellsPerTx: 2,
            isTx4x4: 0, txSizeForCoefProbs: (int)Vp9TxSize.Tx8x8,
            n: 8, planeType: (int)Vp9BlockCoefEnums.PlaneType.Uv,
            coefsShort, idctScratch, tokenCache,
            aboveChroma, leftChroma,
            aboveUEntropyCtx, leftUEntropyCtx,
            mbR, mbC,
            dcQ: uvDcQ, acQ: uvAcQ);

        // V plane.
        DecodePlaneCoefsAndRecon(
            ref state, tileBytes,
            ySrc: default, yRecon: vRecon, uvBase, uvStride,
            byteConsts, ushortConsts,
            mbC * 2, (mbR & 3) * 2, cellsPerTx: 2,
            isTx4x4: 0, txSizeForCoefProbs: (int)Vp9TxSize.Tx8x8,
            n: 8, planeType: (int)Vp9BlockCoefEnums.PlaneType.Uv,
            coefsShort, idctScratch, tokenCache,
            aboveChroma, leftChroma,
            aboveVEntropyCtx, leftVEntropyCtx,
            mbR, mbC,
            dcQ: uvDcQ, acQ: uvAcQ);
    }

    private static void DecodePlaneCoefsAndRecon(
        ref Vp8BoolDecoderGpuState state, ArrayView<byte> tileBytes,
        ArrayView<byte> ySrc /* unused */, ArrayView<byte> yRecon, long planeBase, int planeStride,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        int aboveCellOff, int leftCellOff, int cellsPerTx,
        int isTx4x4, int txSizeForCoefProbs, int n, int planeType,
        ArrayView<short> coefsShort, ArrayView<int> idctScratch, ArrayView<byte> tokenCache,
        ArrayView<byte> aboveBuf, ArrayView<byte> leftBuf,
        ArrayView<byte> aboveEntropyCtx, ArrayView<byte> leftEntropyCtx,
        int mbR, int mbC,
        int dcQ, int acQ)
    {
        // Initial coef context.
        int aboveAgg = 0;
        int leftAgg = 0;
        for (int i = 0; i < cellsPerTx; i++)
        {
            int aIdx = aboveCellOff + i;
            int lIdx = leftCellOff + i;
            if (aIdx < aboveEntropyCtx.Length) aboveAgg |= aboveEntropyCtx[aIdx];
            if (lIdx < leftEntropyCtx.Length) leftAgg |= leftEntropyCtx[lIdx];
        }
        int initialCtx = (aboveAgg != 0 ? 1 : 0) + (leftAgg != 0 ? 1 : 0);

        long scanBase, neighborsBase, coefProbsBase;
        int coefCount;
        if (txSizeForCoefProbs == (int)Vp9TxSize.Tx8x8)
        {
            scanBase = Vp9KeyframeConstantsGpu.Scan8x8Offset;
            neighborsBase = Vp9KeyframeConstantsGpu.Neighbors8x8Offset;
            coefProbsBase = Vp9KeyframeConstantsGpu.CoefProbs8x8Offset;
            coefCount = 64;
        }
        else
        {
            scanBase = Vp9KeyframeConstantsGpu.Scan16x16Offset;
            neighborsBase = Vp9KeyframeConstantsGpu.Neighbors16x16Offset;
            coefProbsBase = Vp9KeyframeConstantsGpu.CoefProbs16x16Offset;
            coefCount = 256;
        }

        var scanView = ushortConsts.SubView(scanBase, coefCount);
        var neighborsView = ushortConsts.SubView(neighborsBase, (long)coefCount * 2);
        var coefProbsView = byteConsts.SubView(coefProbsBase, 432);
        var coefConstsView = byteConsts.SubView(
            Vp9KeyframeConstantsGpu.CoefConstsOffset,
            Vp9KeyframeConstantsGpu.CoefConstsLength);

        int eob = Vp9BlockCoefDecoderGpu.DecodeBlock(
            ref state, tileBytes,
            coefsShort, scanView, neighborsView,
            coefProbsView, coefConstsView, tokenCache,
            coefCount,
            planeType: planeType,
            refType: (int)Vp9BlockCoefEnums.RefType.Intra,
            initialCtx: initialCtx,
            isHighBitDepth: 0);

        // Update entropy ctx cells.
        byte ec = (byte)(eob > 0 ? 1 : 0);
        for (int i = 0; i < cellsPerTx; i++)
        {
            int aIdx = aboveCellOff + i;
            int lIdx = leftCellOff + i;
            if (aIdx < aboveEntropyCtx.Length) aboveEntropyCtx[aIdx] = ec;
            if (lIdx < leftEntropyCtx.Length) leftEntropyCtx[lIdx] = ec;
        }

        // ---- Build prediction edges from already-decoded neighbors ----
        bool topAvail = mbR > 0 || (n == 8 && false); // for chroma we use same MB-level avail
        bool leftAvail = mbC > 0;
        // Actually edge avail is per-block-position relative to the
        // PLANE: top edge available if any pixel above this block's
        // first row exists in the plane. For the v1 walk that means
        // mbR > 0 for both luma and chroma (4:2:0 makes the chroma
        // grid coincide with the luma MB grid).
        topAvail = mbR > 0;

        if (topAvail)
        {
            for (int i = 0; i < n; i++)
                aboveBuf[i] = yRecon[planeBase - planeStride + i];
        }
        if (leftAvail)
        {
            for (int r = 0; r < n; r++)
                leftBuf[r] = yRecon[planeBase + (long)r * planeStride - 1];
        }

        int variant;
        if (topAvail && leftAvail) variant = (int)Vp9DcVariant.Both;
        else if (topAvail) variant = (int)Vp9DcVariant.TopOnly;
        else if (leftAvail) variant = (int)Vp9DcVariant.LeftOnly;
        else variant = (int)Vp9DcVariant.None;

        Vp9DcPredictorGpu.Predict(
            aboveBuf, 0, leftBuf, 0,
            yRecon, planeBase, planeStride,
            n, variant);

        // Dequantize coefs in place.
        Vp9DequantBlockGpu.DequantizeBlock(coefsShort, 0, coefCount, dcQ, acQ);

        // IDCT + add to recon.
        if (n == 16)
            Vp9Idct16x16Gpu.Idct16x16(coefsShort, 0, yRecon, planeBase, planeStride, idctScratch);
        else
            Vp9Idct8x8Gpu.Idct8x8(coefsShort, 0, yRecon, planeBase, planeStride, idctScratch);
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
