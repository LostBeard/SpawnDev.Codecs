// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 v1 keyframe walker, 100% ILGPU. One GPU thread per TILE runs the tile pipeline (predict, transform, quantize,
// entropy code, reconstruct) bit-exact vs the CPU Av1KeyframeEncoder reference. The host is a pure coordinator:
// alloc buffers + upload constants + dispatch + read back.
//
// Tiles are what makes the encoder parallel. An AV1 tile is independent by definition: its own range coder, its
// own above / left contexts (zero at the tile's top and at the start of each superblock row), and no prediction
// across its boundary. So every tile of every frame is one working thread of the same launches: working thread =
// frame * tileCount + tile. With one 64x64 superblock per tile a frame costs about what one superblock costs
// (CUDA, RTX 4070, 832x480 q40: 799 ms on one thread = 1.25 frames per second; 8.8 ms with 104 tiles; 1.1 ms per
// frame in a batch of 16). The price is bits (contexts restart and prediction stops at every tile edge: +0.42% at
// 832x480), so the tile size is the host's choice (Av1TileLayout).
//
// The working threads are spread over the launch (SerialThreadPlacement: one group per multiprocessor, one thread
// per warp while there is room): packed into the first launch group, the same 104 threads took 20.7 ms.
//
// Bounded launches: a tile's walk is linearized into leaf slots (the 16 z-order
// 16x16 positions of each of its superblocks) and a launch encodes at most
// LeavesPerLaunch of them per tile. The contexts are in the tile's
// scratchByte slot; the range encoder and the slot cursor are kept in
// its scratchInt slot [StStarted ..] between launches. The host launches
// ceil(slots of the largest tile / LeavesPerLaunch) steps (a bound it knows; no readback).
//
// Pipeline (mirrors Av1KeyframeEncoder.EncodeSingleTile / EncodePartition):
//   - For each 64x64 superblock of the tile in raster order:
//       at the first superblock of a row: reset left partition + entropy + mode + skip arrays
//       64x64: SPLIT; each 32x32 inside the picture: SPLIT;
//       each 16x16 inside the picture:
//         whole block inside: partition NONE, one BLOCK_16X16 leaf
//               skip = 0, Y mode DC, UV mode DC
//               EncodePlane(Y, TX_16X16), EncodePlane(U, TX_8X8), EncodePlane(V, TX_8X8)
//         crossing the bottom / right edge: SPLIT, then each 8x8 inside the picture: partition NONE, one
//         BLOCK_8X8 leaf (Y TX_8X8, U / V TX_4X4)
//   - Av1RangeEncoderGpu.Done -> write tileLen
//
// Picture edges (any width and height). The coded picture is MiCols x MiRows 4x4 units, each rounded up to a whole
// number of 8x8 blocks. A SPLIT of a block that crosses the edge is coded as the AV1 syntax requires: the full
// partition symbol when both halves start inside, split_or_horz (lower half outside) or split_or_vert (right half
// outside) from the 2-symbol CDFs in the constants buffer, and nothing when both are outside (the split is implied).
// Source pixels past the picture (up to 7 columns / rows) repeat the last column / row; the reconstruction planes
// have the coded size (ReconWidth x ReconHeight), which is what a decoder predicts from.
//
// V1 simplifications match Av1KeyframeEncoder.cs:
//   - Profile 0, YUV 4:2:0, 8-bit.
//   - DC_PRED only; BLOCK_16X16 leaves, BLOCK_8X8 leaves where a 16x16 crosses the picture edge.
//   - tx_mode = LARGEST, reduced_tx_set = 1.
//   - Single tile.
//
// Scratch layout per tile (one slot per thread; the above arrays are indexed by the mi column inside the tile):
//   scratchByte:
//     [AboveEntropyOff..   ): 3 planes * AboveStride bytes (AboveStride = tile width in mi)
//     [LeftEntropyOff..    ): 3 planes * 32 bytes
//     [AbovePartOff..      ): AboveStride bytes
//     [LeftPartOff..       ): 32 bytes
//     [AboveYModeOff..     ): AboveStride bytes
//     [LeftYModeOff..      ): 32 bytes
//     [AboveSkipOff..      ): AboveStride bytes
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

    /// <summary>Y plane offset within the src byte buffer (always 0; the recon Y plane is at 0 too).</summary>
    public int YPlaneOff;
    /// <summary>U plane offset within the src byte buffer (planes tightly packed: Width x Height, then chroma).</summary>
    public int UPlaneOff;
    /// <summary>V plane offset within the src byte buffer.</summary>
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

    /// <summary>Source chroma plane width, <c>(Width + 1) &gt;&gt; 1</c>.</summary>
    public int ChromaWidth;
    /// <summary>Source chroma plane height, <c>(Height + 1) &gt;&gt; 1</c>.</summary>
    public int ChromaHeight;
    /// <summary>Recon luma width = coded width, <c>FrameMiCols * 4</c> (a multiple of 8, &gt;= Width). Recon chroma
    /// planes are <c>ReconWidth / 2</c> x <c>ReconHeight / 2</c>.</summary>
    public int ReconWidth;
    /// <summary>Recon luma height = coded height, <c>FrameMiRows * 4</c>.</summary>
    public int ReconHeight;
    /// <summary>U plane offset within the recon byte buffer (<c>ReconWidth * ReconHeight</c>).</summary>
    public int ReconUOff;
    /// <summary>V plane offset within the recon byte buffer.</summary>
    public int ReconVOff;

    /// <summary>Tile columns of the frame (<see cref="Av1TileLayout"/>).</summary>
    public int TileCols;
    /// <summary>Tile rows of the frame.</summary>
    public int TileRows;
    /// <summary>Tile width in 64x64 superblocks (the last column of tiles may be narrower).</summary>
    public int TileWidthSb;
    /// <summary>Tile height in 64x64 superblocks (the last row of tiles may be shorter).</summary>
    public int TileHeightSb;
    /// <summary>Per-plane stride of the above entropy context arrays = the tile width in mi (<c>TileWidthSb * 16</c>).</summary>
    public int AboveStride;

    /// <summary>First mi row of the thread's tile. Set by the kernel per thread; the host leaves it 0.</summary>
    public int TileMiRowStart;
    /// <summary>First mi column of the thread's tile. Set by the kernel per thread.</summary>
    public int TileMiColStart;
    /// <summary>One past the last mi row of the thread's tile (clipped to the frame). Set by the kernel per thread.</summary>
    public int TileMiRowEnd;
    /// <summary>One past the last mi column of the thread's tile (clipped to the frame). Set by the kernel per thread.</summary>
    public int TileMiColEnd;
}

/// <summary>Slot strides of the AV1 tile walker: source and recon per FRAME, everything else per TILE (= per thread).</summary>
public struct Av1FrameBatchStrides
{
    /// <summary>src bytes per frame (yLen + 2*uvLen).</summary>
    public int SrcStride;
    /// <summary>recon bytes per frame (coded size: ReconWidth * ReconHeight * 3 / 2).</summary>
    public int ReconStride;
    /// <summary>tileBytes per tile (worst case of the largest tile).</summary>
    public int TileStride;
    /// <summary>scratchByte per tile.</summary>
    public int ScratchByteStride;
    /// <summary>scratchInt ints per tile (MinScratchIntLength).</summary>
    public int ScratchIntStride;
    /// <summary>scratchShort shorts per tile (256).</summary>
    public int ScratchShortStride;

    /// <summary>
    /// Where the tile threads sit in the launch. Set by <see cref="Av1FrameSequentialEncodeKernel.RunAsync"/>; a
    /// caller leaves it default.
    /// </summary>
    public SerialThreadPlacement Placement;
}

/// <summary>
/// AV1 v1 keyframe tile walker kernel. One GPU thread per tile of every frame, bit-exact vs the CPU
/// Av1KeyframeEncoder reference, in bounded resumable launches of <see cref="LeavesPerLaunch"/> leaf slots per tile.
/// </summary>
public sealed class Av1FrameSequentialEncodeKernel : IDisposable
{
    private readonly Accelerator _accelerator;

    // ONE kernel for single frames and batches (a second, near-identical kernel was a second shader of the same
    // size to compile): working thread = frame * tileCount + tile, placed in the launch by a SerialThreadPlacement.
    private readonly Action<
        Index1D,
        ArrayView<byte>,    // src (per frame: Y + U + V planes)
        ArrayView<byte>,    // recon (per frame: Y + U + V planes, coded size)
        ArrayView<byte>,    // tileBytes (output, one slot per tile)
        ArrayView<long>,    // tileLen (output, one element per tile)
        ArrayView<byte>,    // constsByte (Av1KeyframeConstantsGpu byte buffer)
        ArrayView<ushort>,  // constsUshort (Av1KeyframeConstantsGpu ushort buffer)
        ArrayView<short>,   // dcAcQuant (DC[0..256) + AC[256..512))
        ArrayView<byte>,    // scratchByte (per tile: contexts + edge + predict + levels)
        ArrayView<int>,     // scratchInt (per tile: per-block working area + resumable state)
        ArrayView<short>,   // scratchShort (per tile: residual)
        Av1FrameSeqEncodeParams, Av1FrameBatchStrides, int> _tileKernel;

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
        _tileKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, ArrayView<byte>,
            ArrayView<byte>, ArrayView<long>,
            ArrayView<byte>, ArrayView<ushort>, ArrayView<short>,
            ArrayView<byte>, ArrayView<int>, ArrayView<short>,
            Av1FrameSeqEncodeParams, Av1FrameBatchStrides, int>(Av1EncodeTileStepKernel, out var info);
        _groupSize = info?.MinGroupSize ?? 1;
    }

    /// <summary>Threads per launch group of the tile kernel on this accelerator (1 when it reports none).</summary>
    private readonly int _groupSize;

    /// <summary>
    /// Where <paramref name="tileThreads"/> tile threads sit in a launch on this accelerator (a GPU runs them
    /// several times faster spread over its multiprocessors and warps than packed into the first group; see
    /// <see cref="SerialThreadPlacement"/>).
    /// </summary>
    public SerialThreadPlacement GetPlacement(int tileThreads) =>
        SerialThreadPlacement.Create(_accelerator, _groupSize, tileThreads);

    /// <summary>Launches a frame of <paramref name="p"/> takes: ceil(leaf slots of the largest tile / LeavesPerLaunch).</summary>
    private static int StepCount(Av1FrameSeqEncodeParams p)
    {
        int tileSbCols = p.TileWidthSb, tileSbRows = p.TileHeightSb;
        int frameSbCols = (p.FrameMiCols + 15) >> 4, frameSbRows = (p.FrameMiRows + 15) >> 4;
        if (tileSbCols > frameSbCols) tileSbCols = frameSbCols;
        if (tileSbRows > frameSbRows) tileSbRows = frameSbRows;
        return (tileSbRows * tileSbCols * 16 + LeavesPerLaunch - 1) / LeavesPerLaunch;
    }

    /// <summary>
    /// Encode the tiles of <paramref name="frameCount"/> frames: frameCount * tileCount working threads (placed in
    /// the launch by <see cref="GetPlacement"/>), each walks
    /// one tile of one frame, <see cref="LeavesPerLaunch"/> leaves per launch, with its own range encoder state, its
    /// own scratch slots and its own output slot (tile g = frame * tileCount + tile: bytes at
    /// <c>g * strides.TileStride</c>, length in <c>tileLen[g]</c>). Zeroes the scratch buffers first (the walker's
    /// contexts start at zero). The work is queued (paced with <see cref="LaunchPacer"/>), not awaited.
    /// </summary>
    public async Task RunAsync(
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
        if (p.TileCols <= 0 || p.TileRows <= 0 || p.TileWidthSb <= 0 || p.TileHeightSb <= 0)
            throw new ArgumentException("The tile layout is not set (TileCols, TileRows, TileWidthSb, TileHeightSb).", nameof(p));
        if (strides.ScratchIntStride < MinScratchIntLength)
            throw new ArgumentException($"ScratchIntStride must be at least {MinScratchIntLength}.", nameof(strides));
        long threads = (long)frameCount * p.TileCols * p.TileRows;
        if (threads > int.MaxValue) throw new ArgumentException("frameCount * tileCount exceeds the launch extent.", nameof(frameCount));
        if (tileLen.Length < threads) throw new ArgumentException("tileLen needs one element per tile of every frame.", nameof(tileLen));
        scratchByte.MemSetToZero();
        scratchInt.MemSetToZero();
        int steps = StepCount(p);
        strides.Placement = GetPlacement((int)threads);
        long extent = strides.Placement.Extent;
        if (extent > int.MaxValue) throw new ArgumentException("frameCount * tileCount exceeds the launch extent.", nameof(frameCount));
        var pacer = new LaunchPacer(_accelerator);
        for (int s = 0; s < steps; s++)
        {
            _tileKernel(new Index1D((int)extent),
                src, recon, tileBytes, tileLen,
                constsByte, constsUshort, dcAcQuant,
                scratchByte, scratchInt, scratchShort, p, strides, LeavesPerLaunch);
            await pacer.LaunchedAsync();
        }
    }

    /// <summary>Release the kernel.</summary>
    public void Dispose() { /* delegate is a value type; nothing to release */ }

    // ===========================================================================
    // Kernel entry points
    // ===========================================================================

    /// <summary>
    /// Working thread = one tile of one frame (<c>g = frame * tileCount + tile</c>, tiles in raster order). Resolves the
    /// tile's rectangle and its buffer slots, then runs the step. The frame's source and recon are shared by its
    /// tiles (each tile writes only its own rectangle of the recon and reads only inside it).
    /// </summary>
    private static void Av1EncodeTileStepKernel(
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
        // Only the placement's working threads of the launch encode a tile; the others return at once.
        int g = SerialThreadPlacement.WorkingThread(s.Placement, idx.X);
        if (g < 0) return;
        int tileCount = p.TileCols * p.TileRows;
        int f = g / tileCount;
        int t = g - f * tileCount;
        int tileRow = t / p.TileCols;
        int tileCol = t - tileRow * p.TileCols;

        // The thread's own copy of the parameters: its tile rectangle (mi), and the above context arrays rebased so
        // that "offset + miCol" indexes the tile's arrays by the column inside the tile.
        var tp = p;
        tp.TileMiRowStart = tileRow * p.TileHeightSb * 16;
        tp.TileMiColStart = tileCol * p.TileWidthSb * 16;
        int rowEnd = tp.TileMiRowStart + p.TileHeightSb * 16;
        int colEnd = tp.TileMiColStart + p.TileWidthSb * 16;
        tp.TileMiRowEnd = rowEnd < p.FrameMiRows ? rowEnd : p.FrameMiRows;
        tp.TileMiColEnd = colEnd < p.FrameMiCols ? colEnd : p.FrameMiCols;
        tp.AboveEntropyOff = p.AboveEntropyOff - tp.TileMiColStart;
        tp.AbovePartOff = p.AbovePartOff - tp.TileMiColStart;
        tp.AboveYModeOff = p.AboveYModeOff - tp.TileMiColStart;
        tp.AboveSkipOff = p.AboveSkipOff - tp.TileMiColStart;

        var fSrc = src.SubView((long)f * s.SrcStride, s.SrcStride);
        var fRecon = recon.SubView((long)f * s.ReconStride, s.ReconStride);
        var tTile = tileBytes.SubView((long)g * s.TileStride, s.TileStride);
        var tTileLen = tileLen.SubView(g, 1);
        var tScratchByte = scratchByte.SubView((long)g * s.ScratchByteStride, s.ScratchByteStride);
        var tScratchInt = scratchInt.SubView((long)g * s.ScratchIntStride, s.ScratchIntStride);
        var tScratchShort = scratchShort.SubView((long)g * s.ScratchShortStride, s.ScratchShortStride);
        EncodeFrameStep(fSrc, fRecon, tTile, tTileLen,
            constsByte, constsUshort, dcAcQuant,
            tScratchByte, tScratchInt, tScratchShort, tp, leavesPerLaunch);
    }

    /// <summary>
    /// One bounded step of one tile: the next <paramref name="leavesPerLaunch"/> leaf slots of its superblock walk.
    /// The above contexts start at zero (the host zeroes scratchByte); the range encoder and the slot cursor live in
    /// scratchInt[StStarted ..]. The step that encodes the tile's last slot finishes its range coder and writes tileLen.
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

        // superblocks of THIS tile
        int sbCols = (p.TileMiColEnd - p.TileMiColStart + sbMi - 1) / sbMi;
        int sbRows = (p.TileMiRowEnd - p.TileMiRowStart + sbMi - 1) / sbMi;
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
    /// reset at the start of a superblock row, the partition syntax that precedes the block (64x64 SPLIT at the
    /// superblock's first slot, 32x32 SPLIT at a 32x32 block's first slot), then the 16x16 NONE leaf, or, where the
    /// 16x16 crosses the picture edge, its SPLIT and the 8x8 NONE leaves inside the picture. Out-of-frame blocks are
    /// skipped exactly as the recursive walk skips them.
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
        int sbRow = sb / sbCols;                       // superblock row / column inside the tile
        int sbCol = sb - sbRow * sbCols;
        int sbMiRow = p.TileMiRowStart + sbRow * 16;   // frame coordinates
        int sbMiCol = p.TileMiColStart + sbCol * 16;

        if (q == 0)
        {
            if (sbCol == 0)
            {
                // Reset left context arrays at the start of each superblock row of the tile.
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
            EmitSplit(ref re, outBuf, constsUshort, scratchByte, p, sbMiRow, sbMiCol, BsizeBlock64x64, 8);
        }

        int q32 = q >> 2;
        int q16 = q & 3;
        int miRow32 = sbMiRow + (q32 >> 1) * 8;
        int miCol32 = sbMiCol + (q32 & 1) * 8;
        bool in32 = (miRow32 < frameMiRows) & (miCol32 < frameMiCols);
        if (!in32) return;

        if (q16 == 0)
            EmitSplit(ref re, outBuf, constsUshort, scratchByte, p, miRow32, miCol32, BsizeBlock32x32, 4);

        int miRow16 = miRow32 + (q16 >> 1) * 4;
        int miCol16 = miCol32 + (q16 & 1) * 4;
        bool in16 = (miRow16 < frameMiRows) & (miCol16 < frameMiCols);
        if (!in16) return;

        // The picture is a whole number of 8x8 blocks (2 mi), so a 16x16 that starts inside is either whole or has
        // only its upper half, its left half or its top-left 8x8 inside. Whole: PARTITION_NONE and one 16x16 leaf.
        // Crossing the bottom / right edge: PARTITION_NONE is not codable; SPLIT, then the 8x8 leaves inside.
        bool whole16 = (miRow16 + 2 < frameMiRows) & (miCol16 + 2 < frameMiCols);
        if (whole16)
            EmitPartitionCdf(ref re, outBuf, constsUshort, scratchByte, p,
                miRow16, miCol16, BsizeBlock16x16, PartitionNone);
        else
            EmitSplit(ref re, outBuf, constsUshort, scratchByte, p, miRow16, miCol16, BsizeBlock16x16, 2);

        // One loop and ONE EncodeLeaf call site for both cases (a second call site would be a second copy of the
        // whole leaf pipeline in the generated shader): 1 iteration for the 16x16 leaf, 4 for the 8x8 positions.
        int leafCount = whole16 ? 1 : 4;
        int leafBsize = whole16 ? BsizeBlock16x16 : BsizeBlock8x8;
        for (int k = 0; k < leafCount; k++)
        {
            int miRowLeaf = miRow16 + (k >> 1) * 2;
            int miColLeaf = miCol16 + (k & 1) * 2;
            bool inside = (miRowLeaf < frameMiRows) & (miColLeaf < frameMiCols);
            if (inside)
            {
                // An 8x8 block that starts inside is whole: always the full (4-symbol) partition CDF.
                if (!whole16)
                    EmitPartitionCdf(ref re, outBuf, constsUshort, scratchByte, p,
                        miRowLeaf, miColLeaf, BsizeBlock8x8, PartitionNone);

                EncodeLeaf(ref re, outBuf,
                    src, recon, constsByte, constsUshort,
                    qDc, qAc,
                    scratchByte, scratchInt, scratchShort,
                    miRowLeaf, miColLeaf, whole16, p);

                UpdatePartitionContext(scratchByte, p, miRowLeaf, miColLeaf, leafBsize);
            }
        }
    }

    /// <summary>
    /// The partition syntax of a block the walker splits (libaom write_partition / AV1 spec 5.11.4): the full
    /// partition symbol when the block's lower and right halves both start inside the picture, split_or_horz when
    /// only the lower half is outside, split_or_vert when only the right half is outside, nothing when both are
    /// (PARTITION_SPLIT is implied). <paramref name="halfMi"/> is half the block size in mi units.
    /// </summary>
    private static void EmitSplit(
        ref Av1RangeEncoderGpuState re, ArrayView<byte> outBuf,
        ArrayView<ushort> constsUshort,
        ArrayView<byte> scratchByte, Av1FrameSeqEncodeParams p,
        int miRow, int miCol, int bsize, int halfMi)
    {
        bool hasRows = miRow + halfMi < p.FrameMiRows;
        bool hasCols = miCol + halfMi < p.FrameMiCols;
        if (hasRows & hasCols)
        {
            EmitPartitionCdf(ref re, outBuf, constsUshort, scratchByte, p, miRow, miCol, bsize, PartitionSplit);
        }
        else if (hasCols)
        {
            // bottom edge: SPLIT (1) or HORZ (0)
            int ctx = GetPartitionContext(scratchByte, p, miRow, miCol, bsize);
            Av1RangeEncoderGpu.EncodeCdfQ15(ref re, outBuf, 1, constsUshort,
                Av1KeyframeConstantsGpu.PartitionSplitOrHorzCdfOffset + ctx * 3, 2);
        }
        else if (hasRows)
        {
            // right edge: SPLIT (1) or VERT (0)
            int ctx = GetPartitionContext(scratchByte, p, miRow, miCol, bsize);
            Av1RangeEncoderGpu.EncodeCdfQ15(ref re, outBuf, 1, constsUshort,
                Av1KeyframeConstantsGpu.PartitionSplitOrVertCdfOffset + ctx * 3, 2);
        }
    }

    // ===========================================================================
    // Leaf block encode (one BLOCK_16X16, or one BLOCK_8X8 at the picture edge)
    // ===========================================================================

    /// <summary>
    /// One leaf: the mode info (the same for both sizes: CFL is allowed for an 8x8 block too, so the UV mode CDF row
    /// is the same), then the three planes. <paramref name="is16"/>: BLOCK_16X16 (Y TX_16X16, U / V TX_8X8);
    /// otherwise BLOCK_8X8 at the picture edge (Y TX_8X8, U / V TX_4X4).
    /// <para>
    /// Three <see cref="EncodePlane"/> call sites, one per transform size, each with the size a CONSTANT (the body
    /// folds to that size; a run-time size keeps every size's code live and is slower on every backend). The 8x8
    /// site serves the chroma of a 16x16 leaf and the luma of an edge leaf, so the edge leaves add only the 4x4
    /// site to the shader.
    /// </para>
    /// </summary>
    private static void EncodeLeaf(
        ref Av1RangeEncoderGpuState re, ArrayView<byte> outBuf,
        ArrayView<byte> src, ArrayView<byte> recon,
        ArrayView<byte> constsByte, ArrayView<ushort> constsUshort,
        int qDc, int qAc,
        ArrayView<byte> scratchByte, ArrayView<int> scratchInt, ArrayView<short> scratchShort,
        int miRow, int miCol, bool is16, Av1FrameSeqEncodeParams p)
    {
        EmitLeafModeInfo(ref re, outBuf, constsByte, constsUshort, scratchByte, miRow, miCol, p);

        int miSize = is16 ? 4 : 2;                 // leaf size in mi; every transform of the leaf spans it
        int chromaX = (miCol * 4) >> 1, chromaY = (miRow * 4) >> 1;
        int reconChromaW = p.ReconWidth >> 1;

        // ---- TX_16X16: luma of a 16x16 leaf ----
        if (is16)
            EncodePlane(ref re, outBuf,
                src, p.YPlaneOff, p.Width, p.Width, p.Height,
                recon, 0, p.ReconWidth,
                constsByte, constsUshort, qDc, qAc,
                scratchByte, scratchInt, scratchShort,
                miRow, miCol, plane: 0,
                xPx: miCol * 4, yPx: miRow * 4,
                txW: 16, txH: 16, txSizeIdx: 2,
                txWMi: miSize, txHMi: miSize,
                p);

        // ---- TX_8X8: U then V of a 16x16 leaf; luma of an 8x8 leaf ----
        int first8 = is16 ? 1 : 0;
        int last8 = is16 ? 2 : 0;
        for (int plane = first8; plane <= last8; plane++)
        {
            bool luma = plane == 0;
            EncodePlane(ref re, outBuf,
                src, luma ? p.YPlaneOff : (plane == 1 ? p.UPlaneOff : p.VPlaneOff),
                luma ? p.Width : p.ChromaWidth, luma ? p.Width : p.ChromaWidth, luma ? p.Height : p.ChromaHeight,
                recon, luma ? 0 : (plane == 1 ? p.ReconUOff : p.ReconVOff), luma ? p.ReconWidth : reconChromaW,
                constsByte, constsUshort, qDc, qAc,
                scratchByte, scratchInt, scratchShort,
                miRow, miCol, plane,
                xPx: luma ? miCol * 4 : chromaX, yPx: luma ? miRow * 4 : chromaY,
                txW: 8, txH: 8, txSizeIdx: 1,
                txWMi: miSize, txHMi: miSize,
                p);
        }

        // ---- TX_4X4: U then V of an 8x8 leaf ----
        if (!is16)
        {
            for (int plane = 1; plane <= 2; plane++)
            {
                EncodePlane(ref re, outBuf,
                    src, plane == 1 ? p.UPlaneOff : p.VPlaneOff, p.ChromaWidth, p.ChromaWidth, p.ChromaHeight,
                    recon, plane == 1 ? p.ReconUOff : p.ReconVOff, reconChromaW,
                    constsByte, constsUshort, qDc, qAc,
                    scratchByte, scratchInt, scratchShort,
                    miRow, miCol, plane,
                    xPx: chromaX, yPx: chromaY,
                    txW: 4, txH: 4, txSizeIdx: 0,
                    txWMi: miSize, txHMi: miSize,
                    p);
            }
        }

        UpdateLeafModeContext(scratchByte, p, miRow, miCol, miSize);
    }

    /// <summary>The mode info of a leaf (16x16 or 8x8): skip = 0, Y mode DC_PRED, UV mode DC_PRED.</summary>
    private static void EmitLeafModeInfo(
        ref Av1RangeEncoderGpuState re, ArrayView<byte> outBuf,
        ArrayView<byte> constsByte, ArrayView<ushort> constsUshort,
        ArrayView<byte> scratchByte,
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

        // ---- UV mode CDF: emit DC. cflAllowed=1 (16x16 and 8x8 blocks) -> 14 syms. ----
        long uvModeCdfBase = Av1KeyframeConstantsGpu.UvModeCdfV1RowOffset;
        Av1RangeEncoderGpu.EncodeCdfQ15(ref re, outBuf, 0, constsUshort, uvModeCdfBase, 14);
    }

    /// <summary>Mode + skip context arrays after a leaf of <paramref name="miSize"/> mi (4 = 16x16, 2 = 8x8): DC, not skipped.</summary>
    private static void UpdateLeafModeContext(
        ArrayView<byte> scratchByte, Av1FrameSeqEncodeParams p, int miRow, int miCol, int miSize)
    {
        int leftMiIdx = miRow & 31;
        for (int i = 0; i < miSize; i++)
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
        ArrayView<byte> srcBuf, int srcPlaneOff, int srcStride, int srcW, int srcH,
        ArrayView<byte> reconBuf, int reconPlaneOff, int reconStride,
        ArrayView<byte> constsByte, ArrayView<ushort> constsUshort,
        int qDc, int qAc,
        ArrayView<byte> scratchByte, ArrayView<int> scratchInt, ArrayView<short> scratchShort,
        int miRow, int miCol, int plane,
        int xPx, int yPx,
        int txW, int txH, int txSizeIdx,
        int txWMi, int txHMi,
        Av1FrameSeqEncodeParams p)
    {
        // txW / txH / txSizeIdx are CONSTANTS at every call site (see EncodeLeaf).
        // txSizeIdx: 2 = TX_16X16, 1 = TX_8X8, 0 = TX_4X4 (chroma of an edge 8x8 block). srcW x srcH is the source
        // plane; the block may extend past it at the picture's right / bottom edge (the coded size is rounded up to
        // 8), where the last source column / row is repeated.
        int n = txW * txH;

        // Build edge buffers from the recon plane.
        // Neighbours are available only inside the tile (AV1 availability): nothing above the tile's first row,
        // nothing left of its first column, and the above-right / below-left extensions stop at the tile's right /
        // bottom edge (another thread is writing the pixels past it).
        int pxPerMi = plane == 0 ? 4 : 2;
        bool haveAbove = yPx > p.TileMiRowStart * pxPerMi;
        bool haveLeft = xPx > p.TileMiColStart * pxPerMi;
        BuildEdgeBuffer(reconBuf, reconPlaneOff, reconStride,
            p.TileMiColEnd * pxPerMi, p.TileMiRowEnd * pxPerMi,
            xPx, yPx, txW, txH, haveAbove, haveLeft, scratchByte, p);

        // DC predict into scratchByte[PredictOff..+n).
        Av1DcPredictorGpu.DcPred(
            scratchByte, p.PredictOff, txW,
            scratchByte, p.EdgeAboveOff,
            scratchByte, p.EdgeLeftOff,
            txW, txH, haveAbove, haveLeft);

        // Compute residual = src - predict into scratchShort[0..+n).
        for (int r = 0; r < txH; r++)
        {
            int sy = yPx + r;
            if (sy >= srcH) sy = srcH - 1;
            int sRow = srcPlaneOff + sy * srcStride;
            int pOff = p.PredictOff + r * txW;
            for (int c = 0; c < txW; c++)
            {
                int sx = xPx + c;
                if (sx >= srcW) sx = srcW - 1;
                short s = (short)srcBuf[sRow + sx];
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
        else if (txSizeIdx == 0)
        {
            Av1Forward2dTransformGpu.Forward4x4DctDct(
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
        // Per CPU contract (Av1KeyframeEncoder.EncodePlane) chroma uses luma-mi indices for ctx lookup (a chroma
        // 4x4 unit is two luma mi, both written with the same value); the plane block size equals the transform
        // size for every leaf (16x16: Y TX_16X16 / UV TX_8X8; 8x8: Y TX_8X8 / UV TX_4X4), so
        // planeBsizeIsTxsize=true and planeBsizeLargerThanTxBsize=false for both Y and UV.
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

        // libaom GetTxScale = 0 for Tx4x4, Tx8x8 and Tx16x16 (only sizes with more than 256 coefficients
        // get a scale).
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
            else if (txSizeIdx == 0)
            {
                Av1Inverse2dTransformGpu.Inverse4x4DctDct(
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

    /// <summary>
    /// Build the above + left edge buffers from the recon plane. <paramref name="endX"/> / <paramref name="endY"/>:
    /// one past the last column / row of the tile in this plane; no pixel at or past them is read (the last one
    /// inside repeats).
    /// </summary>
    private static void BuildEdgeBuffer(
        ArrayView<byte> reconBuf, int reconPlaneOff, int reconStride, int endX, int endY,
        int xPx, int yPx, int txW, int txH,
        bool haveAbove, bool haveLeft,
        ArrayView<byte> scratchByte, Av1FrameSeqEncodeParams p)
    {
        int extLen = 2 * (txW > txH ? txW : txH) + 1;

        // Above row.
        if (haveAbove)
        {
            int rowOff = reconPlaneOff + (yPx - 1) * reconStride + xPx;
            int avail = 2 * (txW > txH ? txW : txH);
            int pwAvail = endX - xPx;
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
            int phAvail = endY - yPx;
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

    /// <summary>Update partition context arrays after emitting PARTITION_NONE at subsize (BLOCK_16X16 or BLOCK_8X8).</summary>
    private static void UpdatePartitionContext(
        ArrayView<byte> scratchByte, Av1FrameSeqEncodeParams p,
        int miRow, int miCol, int subsize)
    {
        // libaom partition_context_lookup table values:
        //   BLOCK_16X16 (subsize=6): above=28, left=28
        //   BLOCK_8X8   (subsize=3): above=30, left=30
        bool is16 = subsize == BsizeBlock16x16;
        byte aboveVal = is16 ? (byte)28 : (byte)30;
        byte leftVal = aboveVal;
        int bw = is16 ? 4 : 2; // mi width
        int bh = bw;

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
        int planeOff = p.AboveEntropyOff + plane * p.AboveStride;
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
        int planeOff = p.AboveEntropyOff + plane * p.AboveStride;
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
        int planeOff = p.AboveEntropyOff + plane * p.AboveStride;
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
