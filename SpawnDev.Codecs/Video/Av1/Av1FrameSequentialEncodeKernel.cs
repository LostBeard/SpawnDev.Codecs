// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 v1 keyframe per-frame walker, 100% ILGPU. One GPU thread per frame runs
// the EncodeSingleTile pipeline bit-exact vs the CPU
// Av1KeyframeEncoder.EncodeSingleTile reference. The host is a pure
// coordinator: alloc buffers + upload constants + dispatch + read back.
//
// Bounded launches: the walk is linearized into leaf slots (the 16 z-order
// 16x16 positions of each superblock) and a launch encodes at most
// LeavesPerLaunch of them (a whole-frame launch was 15 ms at 64x64 on
// CUDA and grows with the picture). The contexts were already in
// scratchByte; the range encoder and the slot cursor are kept in
// scratchInt[StStarted ..] between launches. The host launches
// ceil(slots / LeavesPerLaunch) steps (a bound it knows; no readback).
//
// Pipeline (mirrors Av1KeyframeEncoder.cs lines 388-438):
//   - For each 64x64 superblock in raster order:
//       reset left partition + entropy + mode + skip arrays
//       EncodeSuperblock(64x64 at miRow, miCol):
//         emit partition CDF -> SPLIT (sym 3)
//         for each 32x32 sub-block:
//           emit partition CDF -> SPLIT (sym 3)
//           for each 16x16 sub-block:
//             emit partition CDF -> NONE (sym 0)
//             EncodeLeaf(16x16 at miRow, miCol):
//               emit skip CDF (=0)
//               emit Y mode KF CDF (=DC, sym 0)
//               emit UV mode CDF (=DC, sym 0)
//               EncodePlane(Y, TX_16X16)
//               EncodePlane(U, TX_8X8)
//               EncodePlane(V, TX_8X8)
//               update mode + skip arrays
//             update partition context
//   - Av1RangeEncoderGpu.Done -> write tileLen
//
// V1 simplifications match Av1KeyframeEncoder.cs:
//   - Width + height multiples of 64 (every block fully present, no
//     forced-split edge cases).
//   - Profile 0, YUV 4:2:0, 8-bit.
//   - DC_PRED only, all blocks BLOCK_16X16 leaves.
//   - tx_mode = LARGEST, reduced_tx_set = 1.
//   - Single tile.
//
// Scratch layout per kernel invocation:
//   scratchByte:
//     [AboveEntropyOff..   ): 3 planes * frameMiCols bytes
//     [LeftEntropyOff..    ): 3 planes * 32 bytes
//     [AbovePartOff..      ): frameMiCols bytes
//     [LeftPartOff..       ): 32 bytes
//     [AboveYModeOff..     ): frameMiCols bytes
//     [LeftYModeOff..      ): 32 bytes
//     [AboveSkipOff..      ): frameMiCols bytes
//     [LeftSkipOff..       ): 32 bytes
//     [EdgeAboveOff..      ): 33 bytes (per-block, reused)
//     [EdgeLeftOff..       ): 33 bytes (per-block, reused)
//     [PredictOff..        ): 256 bytes (per-block predict, reused)
//     [LevelsOff..         ): 1384 bytes (per-block libaom levels, reused)
//   scratchInt (per-block, reused):
//     [0..N)               : forward transform output / quantized coefs (N = txW*txH)
//     [N..2N)              : forward transform scratch / dequant output (overwritten)
//     [2N..3N)             : inverse transform residual output
//     [3N..3N+N+16)        : inverse transform scratch (272 max for Tx16x16)
//     [EobSlot=1024]       : eob output from WriteCoeffsTxb (Tx16x16 N=256, max 3*256+272=1040)
//     [CulLevelSlot=1025]  : culLevel output
//     [1102..)             : resumable state (started, done, next slot, range encoder)
//   scratchShort (per-block, reused):
//     [0..N)               : residual (short)

using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// Packed scalar parameters + scratch-buffer offsets for the AV1 v1
/// keyframe walker kernel. Packed because ILGPU's auto-grouped kernel
/// generic-arg ceiling is 14-15.
/// </summary>
public struct Av1FrameSeqEncodeParams
{
    /// <summary>Frame luma width in pixels.</summary>
    public int Width;
    /// <summary>Frame luma height in pixels.</summary>
    public int Height;
    /// <summary>Frame base q-index in [1, 255].</summary>
    public int BaseQIndex;

    /// <summary>Y plane offset within the src/recon byte buffer (always 0).</summary>
    public int YPlaneOff;
    /// <summary>U plane offset within the src/recon byte buffer.</summary>
    public int UPlaneOff;
    /// <summary>V plane offset within the src/recon byte buffer.</summary>
    public int VPlaneOff;

    // ScratchByte regions. Each is offset into the single scratchByte buffer.
    // The walker initializes the per-frame state arrays to 0 at start.
    /// <summary>Above entropy context array. Size: 3 planes * frameMiCols bytes.</summary>
    public int AboveEntropyOff;
    /// <summary>Left entropy context array. Size: 3 planes * 32 bytes.</summary>
    public int LeftEntropyOff;
    /// <summary>Above partition context array. Size: frameMiCols bytes.</summary>
    public int AbovePartOff;
    /// <summary>Left partition context array. Size: 32 bytes.</summary>
    public int LeftPartOff;
    /// <summary>Above intra mode context array. Size: frameMiCols bytes (one byte per mi).</summary>
    public int AboveYModeOff;
    /// <summary>Left intra mode context array. Size: 32 bytes.</summary>
    public int LeftYModeOff;
    /// <summary>Above skip flag array. Size: frameMiCols bytes (0/1).</summary>
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

    /// <summary>Frame mi-cols (<c>== ((Width + 7) &gt;&gt; 3) &lt;&lt; 1</c>).</summary>
    public int FrameMiCols;
    /// <summary>Frame mi-rows.</summary>
    public int FrameMiRows;
}

/// <summary>Per-frame slot strides for the batch AV1 frame walker.</summary>
public struct Av1FrameBatchStrides
{
    /// <summary>src bytes per frame (yLen + 2*uvLen).</summary>
    public int SrcStride;
    /// <summary>recon bytes per frame (same as src).</summary>
    public int ReconStride;
    /// <summary>tileBytes per frame (worst-case).</summary>
    public int TileStride;
    /// <summary>scratchByte per frame.</summary>
    public int ScratchByteStride;
    /// <summary>scratchInt ints per frame (MinScratchIntLength).</summary>
    public int ScratchIntStride;
    /// <summary>scratchShort shorts per frame (256).</summary>
    public int ScratchShortStride;
}

/// <summary>
/// AV1 v1 keyframe per-frame walker kernel. One GPU thread per frame runs EncodeSingleTile bit-exact vs
/// Av1KeyframeEncoder.EncodeSingleTile, in bounded resumable launches of <see cref="LeavesPerLaunch"/> leaf slots.
/// </summary>
public sealed class Av1FrameSequentialEncodeKernel : IDisposable
{
    private readonly Accelerator _accelerator;

    private readonly Action<
        Index1D,
        ArrayView<byte>,    // src (Y + U + V planes)
        ArrayView<byte>,    // recon (Y + U + V planes)
        ArrayView<byte>,    // tileBytes (output)
        ArrayView<long>,    // tileLen (output, 1 element)
        ArrayView<byte>,    // constsByte (Av1KeyframeConstantsGpu byte buffer)
        ArrayView<ushort>,  // constsUshort (Av1KeyframeConstantsGpu ushort buffer)
        ArrayView<short>,   // dcAcQuant (DC[0..256) + AC[256..512))
        ArrayView<byte>,    // scratchByte (state + edge + predict + levels)
        ArrayView<int>,     // scratchInt (per-block working area + resumable state)
        ArrayView<short>,   // scratchShort (residual)
        Av1FrameSeqEncodeParams, int> _kernel;

    private readonly Action<
        Index1D,
        ArrayView<byte>, ArrayView<byte>,
        ArrayView<byte>, ArrayView<long>,
        ArrayView<byte>, ArrayView<ushort>, ArrayView<short>,
        ArrayView<byte>, ArrayView<int>, ArrayView<short>,
        Av1FrameSeqEncodeParams, Av1FrameBatchStrides, int> _batchKernel;

    /// <summary>Slot in scratchInt where WriteCoeffsTxb writes eob (per call).</summary>
    public const int EobSlot = 1100;
    /// <summary>Slot in scratchInt where WriteCoeffsTxb writes culLevel (per call).</summary>
    public const int CulLevelSlot = 1101;

    // Resumable state in scratchInt (after the per-block area): carried between launches.
    private const int StStarted = 1102;
    private const int StDone = 1103;
    private const int StNextSlot = 1104;
    private const int StEnc = 1105;   // Av1RangeEncoderGpu.StateInts

    /// <summary>Minimum scratchInt length (per-frame; per-block area reused across blocks + the resumable state).</summary>
    public const int MinScratchIntLength = StEnc + Av1RangeEncoderGpu.StateInts;

    /// <summary>
    /// Leaf slots (16x16 positions of the superblock z-order walk, 16 per superblock) one launch encodes per
    /// frame. A leaf is the full predict / 16x16 + two 8x8 forward transforms / quantize / coefficient coding /
    /// reconstruction pipeline (about a millisecond for a busy one on a desktop GPU), so a launch stays at a
    /// few milliseconds whatever the picture size.
    /// </summary>
    public const int LeavesPerLaunch = 4;

    /// <summary>Compile the kernel.</summary>
    public Av1FrameSequentialEncodeKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, ArrayView<byte>,
            ArrayView<byte>, ArrayView<long>,
            ArrayView<byte>, ArrayView<ushort>, ArrayView<short>,
            ArrayView<byte>, ArrayView<int>, ArrayView<short>,
            Av1FrameSeqEncodeParams, int>(Av1EncodeFrameStepKernel);
        _batchKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, ArrayView<byte>,
            ArrayView<byte>, ArrayView<long>,
            ArrayView<byte>, ArrayView<ushort>, ArrayView<short>,
            ArrayView<byte>, ArrayView<int>, ArrayView<short>,
            Av1FrameSeqEncodeParams, Av1FrameBatchStrides, int>(Av1EncodeFrameBatchStepKernel);
    }

    /// <summary>Launches one frame of <paramref name="p"/> takes (ceil(leaf slots / LeavesPerLaunch)).</summary>
    private static int StepCount(Av1FrameSeqEncodeParams p)
    {
        int sbRows = (p.FrameMiRows + 15) >> 4;
        int sbCols = (p.FrameMiCols + 15) >> 4;
        return (sbRows * sbCols * 16 + LeavesPerLaunch - 1) / LeavesPerLaunch;
    }

    /// <summary>
    /// Batch dispatch: extent=frameCount, each thread runs one AV1 EncodeSingleTile on its frame slot,
    /// <see cref="LeavesPerLaunch"/> leaves per launch. Each thread has its own range encoder state, its own
    /// scratch slots, and writes to its own output slot. Zeroes the scratch buffers first (the walker's
    /// contexts start at zero). The work is queued (paced with <see cref="LaunchPacer"/>), not awaited.
    /// </summary>
    public async Task RunBatchAsync(
        ArrayView<byte> src, ArrayView<byte> recon,
        ArrayView<byte> tileBytes, ArrayView<long> tileLen,
        ArrayView<byte> constsByte, ArrayView<ushort> constsUshort,
        ArrayView<short> dcAcQuant,
        ArrayView<byte> scratchByte, ArrayView<int> scratchInt, ArrayView<short> scratchShort,
        Av1FrameSeqEncodeParams p,
        int frameCount,
        Av1FrameBatchStrides strides)
    {
        if (frameCount <= 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
        if (strides.ScratchIntStride < MinScratchIntLength)
            throw new ArgumentException($"ScratchIntStride must be at least {MinScratchIntLength}.", nameof(strides));
        scratchByte.MemSetToZero();
        scratchInt.MemSetToZero();
        int steps = StepCount(p);
        var pacer = new LaunchPacer(_accelerator);
        for (int s = 0; s < steps; s++)
        {
            _batchKernel(new Index1D(frameCount),
                src, recon, tileBytes, tileLen,
                constsByte, constsUshort, dcAcQuant,
                scratchByte, scratchInt, scratchShort, p, strides, LeavesPerLaunch);
            await pacer.LaunchedAsync();
        }
    }

    /// <summary>
    /// Dispatch the walker on one frame (one GPU thread, <see cref="LeavesPerLaunch"/> leaves per launch). Zeroes
    /// the scratch buffers first. The work is queued (paced with <see cref="LaunchPacer"/>), not awaited.
    /// </summary>
    public async Task RunAsync(
        ArrayView<byte> src, ArrayView<byte> recon,
        ArrayView<byte> tileBytes, ArrayView<long> tileLen,
        ArrayView<byte> constsByte, ArrayView<ushort> constsUshort,
        ArrayView<short> dcAcQuant,
        ArrayView<byte> scratchByte,
        ArrayView<int> scratchInt,
        ArrayView<short> scratchShort,
        Av1FrameSeqEncodeParams p)
    {
        if (scratchInt.Length < MinScratchIntLength)
            throw new ArgumentException($"scratchInt must hold at least {MinScratchIntLength} ints.", nameof(scratchInt));
        scratchByte.MemSetToZero();
        scratchInt.MemSetToZero();
        int steps = StepCount(p);
        var pacer = new LaunchPacer(_accelerator);
        for (int s = 0; s < steps; s++)
        {
            _kernel(new Index1D(1),
                src, recon, tileBytes, tileLen,
                constsByte, constsUshort, dcAcQuant,
                scratchByte, scratchInt, scratchShort, p, LeavesPerLaunch);
            await pacer.LaunchedAsync();
        }
    }

    /// <summary>Release the kernel.</summary>
    public void Dispose() { /* delegate is a value type; nothing to release */ }

    // ===========================================================================
    // Kernel entry points
    // ===========================================================================

    private static void Av1EncodeFrameStepKernel(
        Index1D _,
        ArrayView<byte> src, ArrayView<byte> recon,
        ArrayView<byte> tileBytes, ArrayView<long> tileLen,
        ArrayView<byte> constsByte, ArrayView<ushort> constsUshort,
        ArrayView<short> dcAcQuant,
        ArrayView<byte> scratchByte,
        ArrayView<int> scratchInt,
        ArrayView<short> scratchShort,
        Av1FrameSeqEncodeParams p,
        int leavesPerLaunch)
    {
        EncodeFrameStep(src, recon, tileBytes, tileLen,
            constsByte, constsUshort, dcAcQuant,
            scratchByte, scratchInt, scratchShort, p, leavesPerLaunch);
    }

    /// <summary>
    /// Batch kernel: thread = one frame. Per-frame buffer slots SubView'd
    /// at kernel head; body unchanged. Each thread has its own range
    /// encoder state and writes to its own output slot.
    /// </summary>
    private static void Av1EncodeFrameBatchStepKernel(
        Index1D idx,
        ArrayView<byte> src, ArrayView<byte> recon,
        ArrayView<byte> tileBytes, ArrayView<long> tileLen,
        ArrayView<byte> constsByte, ArrayView<ushort> constsUshort,
        ArrayView<short> dcAcQuant,
        ArrayView<byte> scratchByte,
        ArrayView<int> scratchInt,
        ArrayView<short> scratchShort,
        Av1FrameSeqEncodeParams p,
        Av1FrameBatchStrides s,
        int leavesPerLaunch)
    {
        int f = idx.X;
        var fSrc = src.SubView((long)f * s.SrcStride, s.SrcStride);
        var fRecon = recon.SubView((long)f * s.ReconStride, s.ReconStride);
        var fTile = tileBytes.SubView((long)f * s.TileStride, s.TileStride);
        var fTileLen = tileLen.SubView(f, 1);
        var fScratchByte = scratchByte.SubView((long)f * s.ScratchByteStride, s.ScratchByteStride);
        var fScratchInt = scratchInt.SubView((long)f * s.ScratchIntStride, s.ScratchIntStride);
        var fScratchShort = scratchShort.SubView((long)f * s.ScratchShortStride, s.ScratchShortStride);
        EncodeFrameStep(fSrc, fRecon, fTile, fTileLen,
            constsByte, constsUshort, dcAcQuant,
            fScratchByte, fScratchInt, fScratchShort, p, leavesPerLaunch);
    }

    /// <summary>
    /// One bounded step: the next <paramref name="leavesPerLaunch"/> leaf slots of the superblock walk. The
    /// above contexts start at zero (the host zeroes scratchByte); the range encoder and the slot cursor live in
    /// scratchInt[StStarted ..]. The step that encodes the last slot finishes the range coder and writes tileLen.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void EncodeFrameStep(
        ArrayView<byte> src, ArrayView<byte> recon,
        ArrayView<byte> tileBytes, ArrayView<long> tileLen,
        ArrayView<byte> constsByte, ArrayView<ushort> constsUshort,
        ArrayView<short> dcAcQuant,
        ArrayView<byte> scratchByte,
        ArrayView<int> scratchInt,
        ArrayView<short> scratchShort,
        Av1FrameSeqEncodeParams p,
        int leavesPerLaunch)
    {
        if (scratchInt[StDone] != 0) return;

        Av1RangeEncoderGpuState re;
        if (scratchInt[StStarted] == 0)
        {
            re = Av1RangeEncoderGpu.Init();
            scratchInt[StStarted] = 1;
        }
        else
        {
            re = Av1RangeEncoderGpu.Load(scratchInt, StEnc);
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
            EncodeLeafSlot(
                ref re, tileBytes,
                src, recon,
                constsByte, constsUshort,
                qDc, qAc,
                scratchByte, scratchInt, scratchShort,
                slot, sbCols,
                frameMiRows, frameMiCols,
                p);
            slot++;
            go = slot < slotEnd;
        }

        if (slot >= slotCount)
        {
            Av1RangeEncoderGpu.Done(ref re, tileBytes);
            tileLen[0] = re.OutLen;
            scratchInt[StDone] = 1;
        }
        scratchInt[StNextSlot] = slot;
        Av1RangeEncoderGpu.Save(scratchInt, StEnc, ref re);
    }

    // ===========================================================================
    // Partition walk (64x64 -> 32x32 -> 16x16 -> NONE), one 16x16 slot at a time
    // ===========================================================================

    /// <summary>
    /// One 16x16 position of the superblock z-order walk (slot = sb * 16 + q32 * 4 + q16): the left-context
    /// reset at the start of a superblock row, the partition symbols that precede the block (64x64 SPLIT at the
    /// superblock's first slot, 32x32 SPLIT at a 32x32 block's first slot), then the 16x16 NONE leaf. Out-of-frame
    /// blocks are skipped exactly as the recursive walk skipped them.
    /// </summary>
    private static void EncodeLeafSlot(
        ref Av1RangeEncoderGpuState re, ArrayView<byte> outBuf,
        ArrayView<byte> src, ArrayView<byte> recon,
        ArrayView<byte> constsByte, ArrayView<ushort> constsUshort,
        int qDc, int qAc,
        ArrayView<byte> scratchByte, ArrayView<int> scratchInt, ArrayView<short> scratchShort,
        int slot, int sbCols,
        int frameMiRows, int frameMiCols,
        Av1FrameSeqEncodeParams p)
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
                // Reset left context arrays at start of each SB row.
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
            EmitPartitionCdf(ref re, outBuf, constsUshort, scratchByte, p,
                sbMiRow, sbMiCol, BsizeBlock64x64, PartitionSplit);
        }

        int q32 = q >> 2;
        int q16 = q & 3;
        int miRow32 = sbMiRow + (q32 >> 1) * 8;
        int miCol32 = sbMiCol + (q32 & 1) * 8;
        bool in32 = (miRow32 < frameMiRows) & (miCol32 < frameMiCols);
        if (!in32) return;

        if (q16 == 0)
            EmitPartitionCdf(ref re, outBuf, constsUshort, scratchByte, p,
                miRow32, miCol32, BsizeBlock32x32, PartitionSplit);

        int miRow16 = miRow32 + (q16 >> 1) * 4;
        int miCol16 = miCol32 + (q16 & 1) * 4;
        bool in16 = (miRow16 < frameMiRows) & (miCol16 < frameMiCols);
        if (!in16) return;

        EmitPartitionCdf(ref re, outBuf, constsUshort, scratchByte, p,
            miRow16, miCol16, BsizeBlock16x16, PartitionNone);

        EncodeLeafBlock(ref re, outBuf,
            src, recon, constsByte, constsUshort,
            qDc, qAc,
            scratchByte, scratchInt, scratchShort,
            miRow16, miCol16, p);

        UpdatePartitionContext(scratchByte, p, miRow16, miCol16, BsizeBlock16x16);
    }

    // ===========================================================================
    // Leaf block encode (one BLOCK_16X16)
    // ===========================================================================

    private static void EncodeLeafBlock(
        ref Av1RangeEncoderGpuState re, ArrayView<byte> outBuf,
        ArrayView<byte> src, ArrayView<byte> recon,
        ArrayView<byte> constsByte, ArrayView<ushort> constsUshort,
        int qDc, int qAc,
        ArrayView<byte> scratchByte, ArrayView<int> scratchInt, ArrayView<short> scratchShort,
        int miRow, int miCol, Av1FrameSeqEncodeParams p)
    {
        int leftMiIdx = miRow & 31;

        // ---- Skip flag CDF: emit 0 (we have residual). ----
        int aboveSkipBit = scratchByte[p.AboveSkipOff + miCol] != 0 ? 1 : 0;
        int leftSkipBit = scratchByte[p.LeftSkipOff + leftMiIdx] != 0 ? 1 : 0;
        int skipCtx = aboveSkipBit + leftSkipBit;
        long skipCdfBase = Av1KeyframeConstantsGpu.SkipTxfmCdfOffset + skipCtx * 3;
        Av1RangeEncoderGpu.EncodeCdfQ15(ref re, outBuf, 0, constsUshort, skipCdfBase, 2);

        // ---- Y mode CDF: emit DC (sym 0). ----
        int aboveYMode = scratchByte[p.AboveYModeOff + miCol];
        int leftYMode = scratchByte[p.LeftYModeOff + leftMiIdx];
        int aboveCtx = constsByte[Av1KeyframeConstantsGpu.IntraModeContextOffset + aboveYMode];
        int leftCtx = constsByte[Av1KeyframeConstantsGpu.IntraModeContextOffset + leftYMode];
        long yModeCdfBase = Av1KeyframeConstantsGpu.KfYModeCdfOffset + (aboveCtx * 5 + leftCtx) * 14;
        Av1RangeEncoderGpu.EncodeCdfQ15(ref re, outBuf, 0, constsUshort, yModeCdfBase, 13);

        // ---- UV mode CDF: emit DC. cflAllowed=1 -> 14 syms. ----
        long uvModeCdfBase = Av1KeyframeConstantsGpu.UvModeCdfV1RowOffset;
        Av1RangeEncoderGpu.EncodeCdfQ15(ref re, outBuf, 0, constsUshort, uvModeCdfBase, 14);

        // ---- Per-plane: predict + residual + transform + quantize + emit coefs + recon ----
        EncodePlane(ref re, outBuf,
            src, p.YPlaneOff, p.Width, recon, p.YPlaneOff, p.Width,
            constsByte, constsUshort, qDc, qAc,
            scratchByte, scratchInt, scratchShort,
            miRow, miCol, plane: 0,
            xPx: miCol * 4, yPx: miRow * 4,
            txW: 16, txH: 16, txSizeIdx: 2,
            txWMi: 4, txHMi: 4,
            planeH: p.Height,
            p);

        EncodePlane(ref re, outBuf,
            src, p.UPlaneOff, p.Width >> 1, recon, p.UPlaneOff, p.Width >> 1,
            constsByte, constsUshort, qDc, qAc,
            scratchByte, scratchInt, scratchShort,
            miRow, miCol, plane: 1,
            xPx: (miCol * 4) >> 1, yPx: (miRow * 4) >> 1,
            txW: 8, txH: 8, txSizeIdx: 1,
            txWMi: 4, txHMi: 4,
            planeH: p.Height >> 1,
            p);

        EncodePlane(ref re, outBuf,
            src, p.VPlaneOff, p.Width >> 1, recon, p.VPlaneOff, p.Width >> 1,
            constsByte, constsUshort, qDc, qAc,
            scratchByte, scratchInt, scratchShort,
            miRow, miCol, plane: 2,
            xPx: (miCol * 4) >> 1, yPx: (miRow * 4) >> 1,
            txW: 8, txH: 8, txSizeIdx: 1,
            txWMi: 4, txHMi: 4,
            planeH: p.Height >> 1,
            p);

        // ---- Update mode + skip context arrays. BLOCK_16X16 = 4 mi wide / 4 mi high. ----
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

    // ===========================================================================
    // Per-plane encode (one TX block)
    // ===========================================================================

    private static void EncodePlane(
        ref Av1RangeEncoderGpuState re, ArrayView<byte> outBuf,
        ArrayView<byte> srcBuf, int srcPlaneOff, int srcStride,
        ArrayView<byte> reconBuf, int reconPlaneOff, int reconStride,
        ArrayView<byte> constsByte, ArrayView<ushort> constsUshort,
        int qDc, int qAc,
        ArrayView<byte> scratchByte, ArrayView<int> scratchInt, ArrayView<short> scratchShort,
        int miRow, int miCol, int plane,
        int xPx, int yPx,
        int txW, int txH, int txSizeIdx,
        int txWMi, int txHMi,
        int planeH,
        Av1FrameSeqEncodeParams p)
    {
        int n = txW * txH;

        // Build edge buffers from the recon plane.
        BuildEdgeBuffer(reconBuf, reconPlaneOff, reconStride, planeH,
            xPx, yPx, txW, txH, scratchByte, p);
        bool haveAbove = yPx > 0;
        bool haveLeft = xPx > 0;

        // DC predict into scratchByte[PredictOff..+n).
        Av1DcPredictorGpu.DcPred(
            scratchByte, p.PredictOff, txW,
            scratchByte, p.EdgeAboveOff,
            scratchByte, p.EdgeLeftOff,
            txW, txH, haveAbove, haveLeft);

        // Compute residual = src - predict into scratchShort[0..+n).
        for (int r = 0; r < txH; r++)
        {
            int sOff = srcPlaneOff + (yPx + r) * srcStride + xPx;
            int pOff = p.PredictOff + r * txW;
            for (int c = 0; c < txW; c++)
            {
                short s = (short)srcBuf[sOff + c];
                short pp = (short)scratchByte[pOff + c];
                scratchShort[r * txW + c] = (short)(s - pp);
            }
        }

        // Forward 2D transform: residual (scratchShort) -> coefs (scratchInt[0..n)).
        // Forward transform scratch goes at scratchInt[n..2n).
        if (txSizeIdx == 1)
        {
            Av1Forward2dTransformGpu.Forward8x8DctDct(
                scratchShort, 0, scratchInt, 0, scratchInt, n);
        }
        else
        {
            Av1Forward2dTransformGpu.Forward16x16DctDct(
                scratchShort, 0, scratchInt, 0, scratchInt, n);
        }

        // Quantize in place.
        Av1ForwardQuantizerGpu.QuantizeBlock(scratchInt, 0, n, qDc, qAc);

        // Compute entropy contexts.
        // Per CPU contract (Av1KeyframeEncoder.EncodePlane lines 815-822) chroma
        // uses luma-mi indices for ctx lookup; v1 has planeBsizeIsTxsize=true and
        // planeBsizeLargerThanTxBsize=false for both Y and UV.
        int txbSkipCtx = GetTxbSkipContext(scratchByte, p, plane, miRow, miCol, txWMi, txHMi);
        int dcSignCtx = GetDcSignContext(scratchByte, p, plane, miRow, miCol, txWMi, txHMi);

        // Emit coefficient stream. WriteCoeffsTxb writes eob to eobOut[blockIdx]
        // and culLevel to culLevelOut[blockIdx]. Use SubViews so the two outputs
        // land in distinct slots of scratchInt.
        var eobView = scratchInt.SubView(EobSlot, 1);
        var culView = scratchInt.SubView(CulLevelSlot, 1);
        Av1CoefEncoderGpu.WriteCoeffsTxb(
            ref re, outBuf, constsByte, constsUshort,
            scratchInt, 0,
            scratchByte, p.LevelsOff,
            txSizeIdx, plane, qctx: GetQctx(p.BaseQIndex),
            txbSkipCtx, dcSignCtx, qindex: p.BaseQIndex,
            eobView, culView, blockIdx: 0);

        int eob = scratchInt[EobSlot];
        int culLevel = scratchInt[CulLevelSlot];

        // Update entropy context with culLevel (8 LSBs include sign in top 2 bits).
        UpdateEntropyContext(scratchByte, p, plane, miRow, miCol, txWMi, txHMi, culLevel);

        // Reconstruct: dequant + inverse 2D transform + add to predict + clip.
        ReconstructBlock(scratchInt, txSizeIdx, txW, txH, n, qDc, qAc,
            scratchByte, p.PredictOff,
            reconBuf, reconPlaneOff + yPx * reconStride + xPx, reconStride,
            eob);
    }

    // ===========================================================================
    // Reconstruction (dequant + inverse 2D transform + add to predict + clip)
    // ===========================================================================
    //
    // scratchInt slot map:
    //   [0..n)         coefs (input - quantized)
    //   [n..2n)        dq (output of dequant; overwrites forward scratch)
    //   [2n..3n)       inverse transform residual output
    //   [3n..3n+n+16)  inverse transform scratch (272 max for Tx16x16)

    private static void ReconstructBlock(
        ArrayView<int> scratchInt, int txSizeIdx, int txW, int txH, int n,
        int qDc, int qAc,
        ArrayView<byte> predict, long predictBase,
        ArrayView<byte> reconOut, long reconOutBase, int reconStride,
        int eob)
    {
        long coefBase = 0;
        long dqBase = n;
        long iTransOutBase = 2 * n;
        long iTransScratchBase = 3 * n;

        // libaom GetTxScale = 0 for both Tx8x8 and Tx16x16 (only 64-tall/wide
        // sizes get scale = 1).
        int shift = 0;
        int maxV = (1 << 15) - 1;
        int minV = -(1 << 15);

        for (int i = 0; i < n; i++)
        {
            int level = scratchInt[coefBase + i];
            if (level == 0) { scratchInt[dqBase + i] = 0; continue; }
            int absLevel = level < 0 ? -level : level;
            int sign = level < 0 ? 1 : 0;
            int q = (i == 0) ? qDc : qAc;
            int dqInt = (int)(((long)absLevel * q) & 0xFFFFFF);
            dqInt = dqInt >> shift;
            if (sign != 0) dqInt = -dqInt;
            if (dqInt > maxV) dqInt = maxV;
            if (dqInt < minV) dqInt = minV;
            scratchInt[dqBase + i] = dqInt;
        }

        // Zero residual buffer (where iTrans writes); covers eob == 0 case.
        for (int i = 0; i < n; i++) scratchInt[iTransOutBase + i] = 0;

        if (eob > 0)
        {
            if (txSizeIdx == 1)
            {
                Av1Inverse2dTransformGpu.Inverse8x8DctDct(
                    scratchInt, dqBase, scratchInt, iTransOutBase, scratchInt, iTransScratchBase);
            }
            else
            {
                Av1Inverse2dTransformGpu.Inverse16x16DctDct(
                    scratchInt, dqBase, scratchInt, iTransOutBase, scratchInt, iTransScratchBase);
            }
        }

        for (int r = 0; r < txH; r++)
        {
            for (int c = 0; c < txW; c++)
            {
                int v = predict[predictBase + r * txW + c] + scratchInt[iTransOutBase + r * txW + c];
                if (v < 0) v = 0;
                else if (v > 255) v = 255;
                reconOut[reconOutBase + r * reconStride + c] = (byte)v;
            }
        }
    }

    // ===========================================================================
    // Edge buffer + helpers
    // ===========================================================================

    private const int BsizeBlock64x64 = 12;
    private const int BsizeBlock32x32 = 9;
    private const int BsizeBlock16x16 = 6;
    private const int BsizeBlock8x8 = 3;
    private const int Block8x8Log2 = 1;
    private const int PartitionPlaneOffset = 4;

    private const int PartitionNone = 0;
    private const int PartitionSplit = 3;

    /// <summary>Quantizer-bin index for the given baseQIndex (libaom get_q_ctx).
    /// Mirror of Av1CoefDecoder.GetQctx (thresholds 20 / 60 / 120).</summary>
    private static int GetQctx(int qindex)
    {
        if (qindex <= 20) return 0;
        if (qindex <= 60) return 1;
        if (qindex <= 120) return 2;
        return 3;
    }

    /// <summary>Build the above + left edge buffers from the recon plane.</summary>
    private static void BuildEdgeBuffer(
        ArrayView<byte> reconBuf, int reconPlaneOff, int reconStride, int planeH,
        int xPx, int yPx, int txW, int txH,
        ArrayView<byte> scratchByte, Av1FrameSeqEncodeParams p)
    {
        int extLen = 2 * (txW > txH ? txW : txH) + 1;
        bool haveAbove = yPx > 0;
        bool haveLeft = xPx > 0;
        int planeW = reconStride;

        // Above row.
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

        // Left col.
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

    /// <summary>Compute the libaom partition_plane_context for (miRow, miCol, bsize).</summary>
    private static int GetPartitionContext(
        ArrayView<byte> scratchByte, Av1FrameSeqEncodeParams p,
        int miRow, int miCol, int bsize)
    {
        int bsl = MiSizeWideLog2(bsize) - Block8x8Log2;
        int above = (scratchByte[p.AbovePartOff + miCol] >> bsl) & 1;
        int left = (scratchByte[p.LeftPartOff + (miRow & 31)] >> bsl) & 1;
        return (left * 2 + above) + bsl * PartitionPlaneOffset;
    }

    /// <summary>Mi_Width_Log2 lookup for the four bsizes the v1 walker uses.</summary>
    private static int MiSizeWideLog2(int bsize)
    {
        if (bsize == BsizeBlock64x64) return 4;
        if (bsize == BsizeBlock32x32) return 3;
        if (bsize == BsizeBlock16x16) return 2;
        return 1; // 8x8
    }

    /// <summary>Number of active partition CDF symbols for a bsize.</summary>
    private static int PartitionCdfNsyms(int bsize)
    {
        if (bsize <= BsizeBlock8x8) return 4;
        return 10;
    }

    /// <summary>Emit a partition CDF symbol.</summary>
    private static void EmitPartitionCdf(
        ref Av1RangeEncoderGpuState re, ArrayView<byte> outBuf,
        ArrayView<ushort> constsUshort,
        ArrayView<byte> scratchByte, Av1FrameSeqEncodeParams p,
        int miRow, int miCol, int bsize, int partition)
    {
        int ctx = GetPartitionContext(scratchByte, p, miRow, miCol, bsize);
        int nsyms = PartitionCdfNsyms(bsize);
        long cdfBase = Av1KeyframeConstantsGpu.PartitionCdfOffset + ctx * 11;
        Av1RangeEncoderGpu.EncodeCdfQ15(ref re, outBuf, partition, constsUshort, cdfBase, nsyms);
    }

    /// <summary>Update partition context arrays after emitting PARTITION_NONE at subsize.</summary>
    private static void UpdatePartitionContext(
        ArrayView<byte> scratchByte, Av1FrameSeqEncodeParams p,
        int miRow, int miCol, int subsize)
    {
        // libaom partition_context_lookup table values for BLOCK_16X16:
        //   BLOCK_16X16 (subsize=6): above=28, left=28
        // (v1 walker only lands here with subsize=BLOCK_16X16.)
        byte aboveVal = 28;
        byte leftVal = 28;
        int bw = 4; // BLOCK_16X16 mi width
        int bh = 4;

        for (int c = miCol; c < miCol + bw && c < p.FrameMiCols; c++)
            scratchByte[p.AbovePartOff + c] = aboveVal;
        int leftStart = miRow & 31;
        for (int r = leftStart; r < leftStart + bh && r < 32; r++)
            scratchByte[p.LeftPartOff + r] = leftVal;
    }

    // ===========================================================================
    // Entropy context helpers (txbSkipCtx / dcSignCtx / Update)
    // ===========================================================================

    /// <summary>Compute txb_skip CDF context for the given block.</summary>
    private static int GetTxbSkipContext(
        ArrayView<byte> scratchByte, Av1FrameSeqEncodeParams p,
        int plane, int miRow, int miCol, int txWMi, int txHMi)
    {
        if (plane == 0)
        {
            // V1 always has planeBsizeIsTxsize=true for Y -> ctx = 0.
            return 0;
        }
        // Chroma: ctx_offset 7 (planeBsizeLargerThanTxBsize=false) +
        //         (above_nz ? 1 : 0) + (left_nz ? 1 : 0).
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

    /// <summary>Compute dc_sign CDF context for the given block.</summary>
    private static int GetDcSignContext(
        ArrayView<byte> scratchByte, Av1FrameSeqEncodeParams p,
        int plane, int miRow, int miCol, int txWMi, int txHMi)
    {
        int dcSign = 0;
        int planeOff = p.AboveEntropyOff + plane * p.FrameMiCols;
        int planeLeftOff = p.LeftEntropyOff + plane * 32;
        for (int i = 0; i < txWMi && miCol + i < p.FrameMiCols; i++)
        {
            int sign = (scratchByte[planeOff + miCol + i]) >> 6; // CoeffContextBits = 6
            dcSign += DcSignDelta(sign & 0x3);
        }
        for (int i = 0; i < txHMi; i++)
        {
            int idx = (miRow + i) & 31;
            int sign = (scratchByte[planeLeftOff + idx]) >> 6;
            dcSign += DcSignDelta(sign & 0x3);
        }
        // libaom dc_sign_contexts[65]: 32 entries of 1, then 0, then 32 entries of 2.
        int idxFinal = dcSign + 32;
        if (idxFinal < 0) idxFinal = 0;
        else if (idxFinal >= 65) idxFinal = 64;
        if (idxFinal == 32) return 0;
        if (idxFinal < 32) return 1;
        return 2;
    }

    /// <summary>libaom signs[3] = {0, -1, 1, 0} (4 entries with 4th = 0 padding).</summary>
    private static int DcSignDelta(int sign)
    {
        if (sign == 1) return -1;
        if (sign == 2) return 1;
        return 0;
    }

    /// <summary>Update entropy context arrays with the encoded culLevel.</summary>
    private static void UpdateEntropyContext(
        ArrayView<byte> scratchByte, Av1FrameSeqEncodeParams p,
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
