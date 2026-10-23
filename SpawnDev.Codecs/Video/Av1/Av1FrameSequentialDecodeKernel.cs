// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 v1 keyframe per-frame decode walker, 100% ILGPU. Mirror of
// Av1FrameSequentialEncodeKernel but uses Av1RangeDecoderGpu and
// Av1CoefDecoderGpu. One GPU thread runs the keyframe decode
// pipeline bit-exact vs the CPU Av1KeyframeWalker reference
// (for the v1 walker config), in bounded resumable launches of
// LeavesPerLaunch 16x16 leaf slots (a whole-frame launch was 14 ms at
// test sizes on CUDA); the range decoder and the slot cursor are kept in
// scratchInt between launches, the contexts were already in scratchByte.
//
// Pipeline (mirrors Av1KeyframeWalker for v1):
//   - For each 64x64 superblock in raster order:
//       reset left partition + entropy + mode + skip arrays
//       DecodeSuperblock(64x64 at miRow, miCol):
//         decode partition CDF -> SPLIT (sym 3, value not stored)
//         for each 32x32 sub-block:
//           decode partition CDF -> SPLIT
//           for each 16x16 sub-block:
//             decode partition CDF -> NONE
//             DecodeLeaf(16x16 at miRow, miCol):
//               decode skip CDF
//               decode Y mode KF CDF
//               decode UV mode CDF
//               DecodePlane(Y, TX_16X16) -> dq + iDCT + add to predict
//               DecodePlane(U, TX_8X8)
//               DecodePlane(V, TX_8X8)
//               update mode + skip arrays
//             update partition context
//
// V1 simplifications match Av1FrameSequentialEncodeKernel.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// Packed scalar parameters + scratch-buffer offsets for the AV1 v1
/// keyframe decoder walker kernel.
/// </summary>
public struct Av1FrameSeqDecodeParams
{
    /// <summary>Frame luma width in pixels.</summary>
    public int Width;
    /// <summary>Frame luma height in pixels.</summary>
    public int Height;
    /// <summary>Frame base q-index in [1, 255].</summary>
    public int BaseQIndex;

    /// <summary>Y plane offset within the recon byte buffer (always 0).</summary>
    public int YPlaneOff;
    /// <summary>U plane offset within the recon byte buffer.</summary>
    public int UPlaneOff;
    /// <summary>V plane offset within the recon byte buffer.</summary>
    public int VPlaneOff;

    // ScratchByte regions.
    /// <summary>Above entropy context array. Size: 3 planes * frameMiCols bytes.</summary>
    public int AboveEntropyOff;
    /// <summary>Left entropy context array. Size: 3 planes * 32 bytes.</summary>
    public int LeftEntropyOff;
    /// <summary>Above partition context array. Size: frameMiCols bytes.</summary>
    public int AbovePartOff;
    /// <summary>Left partition context array. Size: 32 bytes.</summary>
    public int LeftPartOff;
    /// <summary>Above intra mode context array. Size: frameMiCols bytes.</summary>
    public int AboveYModeOff;
    /// <summary>Left intra mode context array. Size: 32 bytes.</summary>
    public int LeftYModeOff;
    /// <summary>Above skip flag array. Size: frameMiCols bytes.</summary>
    public int AboveSkipOff;
    /// <summary>Left skip flag array. Size: 32 bytes.</summary>
    public int LeftSkipOff;

    /// <summary>Per-block edge buffer (above row). Size: 33 bytes.</summary>
    public int EdgeAboveOff;
    /// <summary>Per-block edge buffer (left col). Size: 33 bytes.</summary>
    public int EdgeLeftOff;
    /// <summary>Per-block predict buffer. Size: 256 bytes (max 16x16).</summary>
    public int PredictOff;
    /// <summary>Per-block libaom-layout levels[] scratch. Size: 1384 bytes.</summary>
    public int LevelsOff;

    /// <summary>Frame mi-cols.</summary>
    public int FrameMiCols;
    /// <summary>Frame mi-rows.</summary>
    public int FrameMiRows;

    /// <summary>Tile bytes start offset within the input byte buffer.</summary>
    public int TileBytesOffset;
    /// <summary>Tile bytes length.</summary>
    public int TileBytesLength;
}

/// <summary>
/// AV1 v1 keyframe per-frame decode walker kernel. Mirror of
/// Av1FrameSequentialEncodeKernel: one GPU thread, bounded resumable launches of
/// <see cref="LeavesPerLaunch"/> leaf slots.
/// </summary>
public sealed class Av1FrameSequentialDecodeKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<
        Index1D,
        ArrayView<byte>,    // tileBytes (input)
        ArrayView<byte>,    // recon (Y + U + V planes, output)
        ArrayView<byte>,    // constsByte
        ArrayView<ushort>,  // constsUshort
        ArrayView<short>,   // dcAcQuant (DC[0..256) + AC[256..512))
        ArrayView<byte>,    // scratchByte (state + edge + predict + levels)
        ArrayView<int>,     // scratchInt (per-block working area + resumable state)
        Av1FrameSeqDecodeParams, int> _kernel;

    /// <summary>Slot in scratchInt where ReadCoeffsTxb writes eob.</summary>
    public const int EobSlot = 1100;
    /// <summary>Slot in scratchInt where ReadCoeffsTxb writes culLevel.</summary>
    public const int CulLevelSlot = 1101;

    // Resumable state in scratchInt (after the per-block area): carried between launches.
    private const int StStarted = 1102;
    private const int StNextSlot = 1103;
    private const int StDec = 1104;   // Av1RangeDecoderGpu.StateInts

    /// <summary>Minimum scratchInt length.</summary>
    public const int MinScratchIntLength = StDec + Av1RangeDecoderGpu.StateInts;

    /// <summary>
    /// Leaf slots (16x16 positions of the superblock z-order walk) one launch decodes: coefficient decode, DC
    /// prediction, 16x16 + two 8x8 inverse transforms and reconstruction each.
    /// </summary>
    public const int LeavesPerLaunch = 4;

    /// <summary>Compile the kernel.</summary>
    public Av1FrameSequentialDecodeKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, ArrayView<byte>,
            ArrayView<byte>, ArrayView<ushort>, ArrayView<short>,
            ArrayView<byte>, ArrayView<int>,
            Av1FrameSeqDecodeParams, int>(Av1DecodeFrameStepKernel);
    }

    /// <summary>
    /// Dispatch the walker (bounded launches, paced with <see cref="LaunchPacer"/>). Zeroes the scratch buffers
    /// first (the walker's contexts start at zero). The work is queued, not awaited to completion.
    /// </summary>
    public async Task RunAsync(
        ArrayView<byte> tileBytes,
        ArrayView<byte> recon,
        ArrayView<byte> constsByte, ArrayView<ushort> constsUshort,
        ArrayView<short> dcAcQuant,
        ArrayView<byte> scratchByte,
        ArrayView<int> scratchInt,
        Av1FrameSeqDecodeParams p)
    {
        if (scratchInt.Length < MinScratchIntLength)
            throw new ArgumentException($"scratchInt must hold at least {MinScratchIntLength} ints.", nameof(scratchInt));
        scratchByte.MemSetToZero();
        scratchInt.MemSetToZero();
        int sbRows = (p.FrameMiRows + 15) >> 4;
        int sbCols = (p.FrameMiCols + 15) >> 4;
        int steps = (sbRows * sbCols * 16 + LeavesPerLaunch - 1) / LeavesPerLaunch;
        var pacer = new LaunchPacer(_accelerator);
        for (int s = 0; s < steps; s++)
        {
            _kernel(new Index1D(1),
                tileBytes, recon,
                constsByte, constsUshort, dcAcQuant,
                scratchByte, scratchInt, p, LeavesPerLaunch);
            await pacer.LaunchedAsync();
        }
    }

    /// <summary>Release resources.</summary>
    public void Dispose() { /* delegate has no cleanup */ }

    // ===========================================================================

    /// <summary>
    /// One bounded step: the next <paramref name="leavesPerLaunch"/> leaf slots (slot = sb * 16 + q32 * 4 + q16)
    /// of the superblock walk. The range decoder and the slot cursor live in scratchInt[StStarted ..].
    /// </summary>
    private static void Av1DecodeFrameStepKernel(
        Index1D _,
        ArrayView<byte> tileBytes,
        ArrayView<byte> recon,
        ArrayView<byte> constsByte, ArrayView<ushort> constsUshort,
        ArrayView<short> dcAcQuant,
        ArrayView<byte> scratchByte,
        ArrayView<int> scratchInt,
        Av1FrameSeqDecodeParams p,
        int leavesPerLaunch)
    {
        Av1RangeDecoderGpuState rd;
        if (scratchInt[StStarted] == 0)
        {
            rd = Av1RangeDecoderGpu.Init(tileBytes, p.TileBytesOffset, p.TileBytesLength);
            scratchInt[StStarted] = 1;
        }
        else
        {
            rd = Av1RangeDecoderGpu.Load(scratchInt, StDec);
        }

        int frameMiCols = p.FrameMiCols;
        int frameMiRows = p.FrameMiRows;
        int sbMi = 16;

        int qDc = (int)dcAcQuant[p.BaseQIndex];
        int qAc = (int)dcAcQuant[256 + p.BaseQIndex];

        int sbCols = (frameMiCols + sbMi - 1) / sbMi;
        int sbRows = (frameMiRows + sbMi - 1) / sbMi;
        int slotCount = sbRows * sbCols * 16;
        int slot = scratchInt[StNextSlot];
        int slotEnd = slot + leavesPerLaunch;
        if (slotEnd > slotCount) slotEnd = slotCount;
        bool go = slot < slotEnd;
        while (go)
        {
            DecodeLeafSlot(
                ref rd, tileBytes, recon,
                constsByte, constsUshort, qDc, qAc,
                scratchByte, scratchInt,
                slot, sbCols, frameMiRows, frameMiCols, p);
            slot++;
            go = slot < slotEnd;
        }

        scratchInt[StNextSlot] = slot;
        Av1RangeDecoderGpu.Save(scratchInt, StDec, ref rd);
    }

    /// <summary>
    /// One 16x16 position of the superblock z-order walk: the left-context reset at the start of a superblock
    /// row, the partition symbols that precede the block (64x64 at the superblock's first slot, 32x32 at a 32x32
    /// block's first slot), then the 16x16 leaf. Out-of-frame blocks are skipped as the recursive walk did.
    /// </summary>
    private static void DecodeLeafSlot(
        ref Av1RangeDecoderGpuState rd, ArrayView<byte> inBuf, ArrayView<byte> recon,
        ArrayView<byte> constsByte, ArrayView<ushort> constsUshort,
        int qDc, int qAc,
        ArrayView<byte> scratchByte, ArrayView<int> scratchInt,
        int slot, int sbCols,
        int frameMiRows, int frameMiCols,
        Av1FrameSeqDecodeParams p)
    {
        int sb = slot >> 4;
        int q = slot & 15;
        int sbRow = sb / sbCols;
        int sbCol = sb - sbRow * sbCols;
        int sbMiRow = sbRow * 16;
        int sbMiCol = sbCol * 16;

        if (q == 0)
        {
            if (sbCol == 0)
            {
                for (int i = 0; i < 32; i++)
                {
                    scratchByte[p.LeftEntropyOff + 0 * 32 + i] = 0;
                    scratchByte[p.LeftEntropyOff + 1 * 32 + i] = 0;
                    scratchByte[p.LeftEntropyOff + 2 * 32 + i] = 0;
                    scratchByte[p.LeftPartOff + i] = 0;
                    scratchByte[p.LeftYModeOff + i] = 0;
                    scratchByte[p.LeftSkipOff + i] = 0;
                }
            }
            // 64x64 partition (decode + discard - we know it's SPLIT for v1).
            DecodePartitionCdf(ref rd, inBuf, constsUshort, scratchByte, p,
                sbMiRow, sbMiCol, BsizeBlock64x64);
        }

        int q32 = q >> 2;
        int q16 = q & 3;
        int miRow32 = sbMiRow + (q32 >> 1) * 8;
        int miCol32 = sbMiCol + (q32 & 1) * 8;
        bool in32 = (miRow32 < frameMiRows) & (miCol32 < frameMiCols);
        if (!in32) return;

        if (q16 == 0)
            DecodePartitionCdf(ref rd, inBuf, constsUshort, scratchByte, p,
                miRow32, miCol32, BsizeBlock32x32);

        int miRow16 = miRow32 + (q16 >> 1) * 4;
        int miCol16 = miCol32 + (q16 & 1) * 4;
        bool in16 = (miRow16 < frameMiRows) & (miCol16 < frameMiCols);
        if (!in16) return;

        DecodePartitionCdf(ref rd, inBuf, constsUshort, scratchByte, p,
            miRow16, miCol16, BsizeBlock16x16);

        DecodeLeafBlock(ref rd, inBuf, recon,
            constsByte, constsUshort, qDc, qAc,
            scratchByte, scratchInt,
            miRow16, miCol16, p);

        UpdatePartitionContext(scratchByte, p, miRow16, miCol16);
    }

    private static void DecodeLeafBlock(
        ref Av1RangeDecoderGpuState rd, ArrayView<byte> inBuf, ArrayView<byte> recon,
        ArrayView<byte> constsByte, ArrayView<ushort> constsUshort,
        int qDc, int qAc,
        ArrayView<byte> scratchByte, ArrayView<int> scratchInt,
        int miRow, int miCol, Av1FrameSeqDecodeParams p)
    {
        int leftMiIdx = miRow & 31;

        // Skip CDF.
        int aboveSkipBit = scratchByte[p.AboveSkipOff + miCol] != 0 ? 1 : 0;
        int leftSkipBit = scratchByte[p.LeftSkipOff + leftMiIdx] != 0 ? 1 : 0;
        int skipCtx = aboveSkipBit + leftSkipBit;
        long skipCdfBase = Av1KeyframeConstantsGpu.SkipTxfmCdfOffset + skipCtx * 3;
        int skipFlag = Av1RangeDecoderGpu.DecodeCdfQ15(ref rd, inBuf, constsUshort, skipCdfBase, 2);
        // For v1 encoder we always emit skip=0 (we have residual). Decoder reads same bit.
        // skipFlag is consumed; per v1 encoder convention we have residual.

        // Y mode CDF.
        int aboveYMode = scratchByte[p.AboveYModeOff + miCol];
        int leftYMode = scratchByte[p.LeftYModeOff + leftMiIdx];
        int aboveCtx = constsByte[Av1KeyframeConstantsGpu.IntraModeContextOffset + aboveYMode];
        int leftCtx = constsByte[Av1KeyframeConstantsGpu.IntraModeContextOffset + leftYMode];
        long yModeCdfBase = Av1KeyframeConstantsGpu.KfYModeCdfOffset + (aboveCtx * 5 + leftCtx) * 14;
        int yMode = Av1RangeDecoderGpu.DecodeCdfQ15(ref rd, inBuf, constsUshort, yModeCdfBase, 13);
        // For v1 yMode should be 0 (DC). Decoder uses it for context updates only.

        // UV mode CDF.
        long uvModeCdfBase = Av1KeyframeConstantsGpu.UvModeCdfV1RowOffset;
        int uvMode = Av1RangeDecoderGpu.DecodeCdfQ15(ref rd, inBuf, constsUshort, uvModeCdfBase, 14);

        // Per-plane decode.
        DecodePlane(ref rd, inBuf, constsByte, constsUshort,
            recon, p.YPlaneOff, p.Width, p.Height,
            qDc, qAc, scratchByte, scratchInt,
            miRow, miCol, plane: 0,
            xPx: miCol * 4, yPx: miRow * 4,
            txW: 16, txH: 16, txSizeIdx: 2,
            txWMi: 4, txHMi: 4, p);

        DecodePlane(ref rd, inBuf, constsByte, constsUshort,
            recon, p.UPlaneOff, p.Width >> 1, p.Height >> 1,
            qDc, qAc, scratchByte, scratchInt,
            miRow, miCol, plane: 1,
            xPx: (miCol * 4) >> 1, yPx: (miRow * 4) >> 1,
            txW: 8, txH: 8, txSizeIdx: 1,
            txWMi: 4, txHMi: 4, p);

        DecodePlane(ref rd, inBuf, constsByte, constsUshort,
            recon, p.VPlaneOff, p.Width >> 1, p.Height >> 1,
            qDc, qAc, scratchByte, scratchInt,
            miRow, miCol, plane: 2,
            xPx: (miCol * 4) >> 1, yPx: (miRow * 4) >> 1,
            txW: 8, txH: 8, txSizeIdx: 1,
            txWMi: 4, txHMi: 4, p);

        // Update mode + skip context arrays.
        for (int i = 0; i < 4; i++)
        {
            if (miCol + i < p.FrameMiCols)
            {
                scratchByte[p.AboveYModeOff + miCol + i] = 0; // DC
                scratchByte[p.AboveSkipOff + miCol + i] = 0;
            }
            int r = (leftMiIdx + i) & 31;
            scratchByte[p.LeftYModeOff + r] = 0;
            scratchByte[p.LeftSkipOff + r] = 0;
        }
    }

    private static void DecodePlane(
        ref Av1RangeDecoderGpuState rd, ArrayView<byte> inBuf,
        ArrayView<byte> constsByte, ArrayView<ushort> constsUshort,
        ArrayView<byte> reconBuf, int reconPlaneOff, int reconStride, int planeH,
        int qDc, int qAc,
        ArrayView<byte> scratchByte, ArrayView<int> scratchInt,
        int miRow, int miCol, int plane,
        int xPx, int yPx,
        int txW, int txH, int txSizeIdx,
        int txWMi, int txHMi,
        Av1FrameSeqDecodeParams p)
    {
        int n = txW * txH;

        // Compute entropy contexts BEFORE updating (matches encoder order).
        int txbSkipCtx = GetTxbSkipContext(scratchByte, p, plane, miRow, miCol, txWMi, txHMi);
        int dcSignCtx = GetDcSignContext(scratchByte, p, plane, miRow, miCol, txWMi, txHMi);

        // Read coefs into scratchInt[0..n) (already-dequantized per Av1CoefDecoderGpu).
        var eobView = scratchInt.SubView(EobSlot, 1);
        var culView = scratchInt.SubView(CulLevelSlot, 1);
        Av1CoefDecoderGpu.ReadCoeffsTxb(
            ref rd, inBuf, constsByte, constsUshort,
            scratchInt, 0,
            scratchByte, p.LevelsOff,
            txSizeIdx, plane, qctx: GetQctx(p.BaseQIndex),
            txbSkipCtx, dcSignCtx, qindex: p.BaseQIndex,
            qDc, qAc,
            eobView, culView, blockIdx: 0);

        int eob = scratchInt[EobSlot];
        int culLevel = scratchInt[CulLevelSlot];

        // Update entropy context.
        UpdateEntropyContext(scratchByte, p, plane, miRow, miCol, txWMi, txHMi, culLevel);

        // Build edge buffer + DC predict.
        BuildEdgeBuffer(reconBuf, reconPlaneOff, reconStride, planeH,
            xPx, yPx, txW, txH, scratchByte, p);
        bool haveAbove = yPx > 0;
        bool haveLeft = xPx > 0;
        Av1DcPredictorGpu.DcPred(
            scratchByte, p.PredictOff, txW,
            scratchByte, p.EdgeAboveOff,
            scratchByte, p.EdgeLeftOff,
            txW, txH, haveAbove, haveLeft);

        // Inverse 2D transform: dq coefs (scratchInt[0..n)) -> residual (scratchInt[n..2n)).
        // iTrans scratch goes in scratchInt[2n..3n+16).
        long iTransOutBase = n;
        long iTransScratchBase = 2 * n;
        for (int i = 0; i < n; i++) scratchInt[iTransOutBase + i] = 0;

        if (eob > 0)
        {
            if (txSizeIdx == 1)
            {
                Av1Inverse2dTransformGpu.Inverse8x8DctDct(
                    scratchInt, 0, scratchInt, iTransOutBase, scratchInt, iTransScratchBase);
            }
            else
            {
                Av1Inverse2dTransformGpu.Inverse16x16DctDct(
                    scratchInt, 0, scratchInt, iTransOutBase, scratchInt, iTransScratchBase);
            }
        }

        // Add residual to predict, clip, write recon.
        long reconOutBase = reconPlaneOff + yPx * reconStride + xPx;
        for (int r = 0; r < txH; r++)
        {
            for (int c = 0; c < txW; c++)
            {
                int v = scratchByte[p.PredictOff + r * txW + c]
                    + scratchInt[iTransOutBase + r * txW + c];
                if (v < 0) v = 0;
                else if (v > 255) v = 255;
                reconBuf[reconOutBase + r * reconStride + c] = (byte)v;
            }
        }
    }

    // ===========================================================================
    // Helpers (mirror of Av1FrameSequentialEncodeKernel)
    // ===========================================================================

    private const int BsizeBlock64x64 = 12;
    private const int BsizeBlock32x32 = 9;
    private const int BsizeBlock16x16 = 6;
    private const int BsizeBlock8x8 = 3;
    private const int Block8x8Log2 = 1;
    private const int PartitionPlaneOffset = 4;

    private static int GetQctx(int qindex)
    {
        if (qindex <= 20) return 0;
        if (qindex <= 60) return 1;
        if (qindex <= 120) return 2;
        return 3;
    }

    private static int MiSizeWideLog2(int bsize)
    {
        if (bsize == BsizeBlock64x64) return 4;
        if (bsize == BsizeBlock32x32) return 3;
        if (bsize == BsizeBlock16x16) return 2;
        return 1;
    }

    private static int PartitionCdfNsyms(int bsize)
    {
        if (bsize <= BsizeBlock8x8) return 4;
        return 10;
    }

    private static int GetPartitionContext(
        ArrayView<byte> scratchByte, Av1FrameSeqDecodeParams p,
        int miRow, int miCol, int bsize)
    {
        int bsl = MiSizeWideLog2(bsize) - Block8x8Log2;
        int above = (scratchByte[p.AbovePartOff + miCol] >> bsl) & 1;
        int left = (scratchByte[p.LeftPartOff + (miRow & 31)] >> bsl) & 1;
        return (left * 2 + above) + bsl * PartitionPlaneOffset;
    }

    private static void DecodePartitionCdf(
        ref Av1RangeDecoderGpuState rd, ArrayView<byte> inBuf,
        ArrayView<ushort> constsUshort,
        ArrayView<byte> scratchByte, Av1FrameSeqDecodeParams p,
        int miRow, int miCol, int bsize)
    {
        int ctx = GetPartitionContext(scratchByte, p, miRow, miCol, bsize);
        int nsyms = PartitionCdfNsyms(bsize);
        long cdfBase = Av1KeyframeConstantsGpu.PartitionCdfOffset + ctx * 11;
        // Decoded sym is consumed (v1 encoder always emits SPLIT at 64/32 and NONE at 16x16).
        int unused = Av1RangeDecoderGpu.DecodeCdfQ15(ref rd, inBuf, constsUshort, cdfBase, nsyms);
    }

    private static void UpdatePartitionContext(
        ArrayView<byte> scratchByte, Av1FrameSeqDecodeParams p,
        int miRow, int miCol)
    {
        // BLOCK_16X16 lookup: above=28, left=28.
        byte aboveVal = 28;
        byte leftVal = 28;
        for (int c = miCol; c < miCol + 4 && c < p.FrameMiCols; c++)
            scratchByte[p.AbovePartOff + c] = aboveVal;
        int leftStart = miRow & 31;
        for (int r = leftStart; r < leftStart + 4 && r < 32; r++)
            scratchByte[p.LeftPartOff + r] = leftVal;
    }

    private static void BuildEdgeBuffer(
        ArrayView<byte> reconBuf, int reconPlaneOff, int reconStride, int planeH,
        int xPx, int yPx, int txW, int txH,
        ArrayView<byte> scratchByte, Av1FrameSeqDecodeParams p)
    {
        int extLen = 2 * (txW > txH ? txW : txH) + 1;
        bool haveAbove = yPx > 0;
        bool haveLeft = xPx > 0;
        int planeW = reconStride;

        if (haveAbove)
        {
            int rowOff = reconPlaneOff + (yPx - 1) * reconStride + xPx;
            int avail = 2 * (txW > txH ? txW : txH);
            int pwAvail = planeW - xPx;
            int len = avail < pwAvail ? avail : pwAvail;
            for (int i = 0; i < len; i++)
                scratchByte[p.EdgeAboveOff + i] = reconBuf[rowOff + i];
            byte last = len > 0 ? scratchByte[p.EdgeAboveOff + len - 1] : (byte)129;
            for (int i = len; i < extLen; i++) scratchByte[p.EdgeAboveOff + i] = last;
        }
        else
        {
            for (int i = 0; i < extLen; i++) scratchByte[p.EdgeAboveOff + i] = 129;
        }

        if (haveLeft)
        {
            int colOff = reconPlaneOff + yPx * reconStride + (xPx - 1);
            int avail = 2 * (txW > txH ? txW : txH);
            int phAvail = planeH - yPx;
            int len = avail < phAvail ? avail : phAvail;
            for (int i = 0; i < len; i++)
                scratchByte[p.EdgeLeftOff + i] = reconBuf[colOff + i * reconStride];
            byte last = len > 0 ? scratchByte[p.EdgeLeftOff + len - 1] : (byte)127;
            for (int i = len; i < extLen; i++) scratchByte[p.EdgeLeftOff + i] = last;
        }
        else
        {
            for (int i = 0; i < extLen; i++) scratchByte[p.EdgeLeftOff + i] = 127;
        }
    }

    private static int GetTxbSkipContext(
        ArrayView<byte> scratchByte, Av1FrameSeqDecodeParams p,
        int plane, int miRow, int miCol, int txWMi, int txHMi)
    {
        if (plane == 0) return 0;
        bool aboveNz = false;
        bool leftNz = false;
        int planeOff = p.AboveEntropyOff + plane * p.FrameMiCols;
        int planeLeftOff = p.LeftEntropyOff + plane * 32;
        for (int i = 0; i < txWMi && miCol + i < p.FrameMiCols; i++)
            if (scratchByte[planeOff + miCol + i] != 0) { aboveNz = true; break; }
        for (int i = 0; i < txHMi; i++)
        {
            int idx = (miRow + i) & 31;
            if (scratchByte[planeLeftOff + idx] != 0) { leftNz = true; break; }
        }
        int ctxBase = (aboveNz ? 1 : 0) + (leftNz ? 1 : 0);
        return ctxBase + 7;
    }

    private static int GetDcSignContext(
        ArrayView<byte> scratchByte, Av1FrameSeqDecodeParams p,
        int plane, int miRow, int miCol, int txWMi, int txHMi)
    {
        int dcSign = 0;
        int planeOff = p.AboveEntropyOff + plane * p.FrameMiCols;
        int planeLeftOff = p.LeftEntropyOff + plane * 32;
        for (int i = 0; i < txWMi && miCol + i < p.FrameMiCols; i++)
        {
            int sign = (scratchByte[planeOff + miCol + i]) >> 6;
            dcSign += DcSignDelta(sign & 0x3);
        }
        for (int i = 0; i < txHMi; i++)
        {
            int idx = (miRow + i) & 31;
            int sign = (scratchByte[planeLeftOff + idx]) >> 6;
            dcSign += DcSignDelta(sign & 0x3);
        }
        int idxFinal = dcSign + 32;
        if (idxFinal < 0) idxFinal = 0;
        else if (idxFinal >= 65) idxFinal = 64;
        if (idxFinal == 32) return 0;
        if (idxFinal < 32) return 1;
        return 2;
    }

    private static int DcSignDelta(int sign)
    {
        if (sign == 1) return -1;
        if (sign == 2) return 1;
        return 0;
    }

    private static void UpdateEntropyContext(
        ArrayView<byte> scratchByte, Av1FrameSeqDecodeParams p,
        int plane, int miRow, int miCol, int txWMi, int txHMi, int culLevelWithSign)
    {
        byte v = (byte)(culLevelWithSign & 0xFF);
        int planeOff = p.AboveEntropyOff + plane * p.FrameMiCols;
        int planeLeftOff = p.LeftEntropyOff + plane * 32;
        for (int i = 0; i < txWMi && miCol + i < p.FrameMiCols; i++)
            scratchByte[planeOff + miCol + i] = v;
        for (int i = 0; i < txHMi; i++)
        {
            int idx = (miRow + i) & 31;
            scratchByte[planeLeftOff + idx] = v;
        }
    }
}
