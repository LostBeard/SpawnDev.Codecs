// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 v1 keyframe walker, 100% ILGPU: the tile pipeline (predict, transform, quantize, reconstruct, entropy code)
// bit-exact vs the CPU Av1KeyframeEncoder reference. The host is a pure coordinator: alloc buffers + upload
// constants + dispatch + read back.
//
// Three passes over every tile (2026-11-13; before, one thread per tile did everything, leaf by leaf):
//   1. Transform pass, parallel INSIDE the tile. A leaf predicts (DC) from the reconstruction above it and to its
//      left, so the 16x16 positions of an anti-diagonal of the tile are independent: a wavefront. One launch group
//      per tile; its lanes take the positions of a diagonal, with a barrier between diagonals. Each position:
//      predict, forward transform, quantize, reconstruct; its quantized coefficients and the entropy context of
//      each of its transform blocks (cul_level + DC sign, a function of the coefficients) go to a device buffer.
//   2. Symbol pass, one thread per 16x16 position, all at once. The keyframe writes disable_cdf_update = 1, so a
//      symbol's probabilities are a fixed table row picked by a context, and every context is known without coding
//      anything: the coefficient contexts are the neighbour blocks' stored contexts, the partition contexts follow
//      from where the picture ends (a leaf is 16x16 unless its position crosses the edge), the skip and mode
//      contexts are constant. The thread writes the position's symbols, in coding order, as entries of a symbol
//      slot (Av1SymbolStreamGpu).
//   3. Range coder pass, one thread per tile: the tile's symbol entries, slot after slot, through its range coder.
//      This is the only part that is serial by nature (one coder state per tile), and it is all a tile thread does.
//   Passes 2 and 3 alternate over waves of slots (WaveSlots per tile) so the symbol memory stays bounded.
//
// Tiles are what makes the encoder parallel. An AV1 tile is independent by definition: its own range coder, its
// own above / left contexts (zero at the tile's top and at the start of each superblock row), and no prediction
// across its boundary. With one 64x64 superblock per tile a frame costs about what one superblock costs. The price
// is bits (contexts restart and prediction stops at every tile edge: +0.42% at 832x480), so the tile size is the
// host's choice (Av1TileLayout).
//
// The serial threads (symbol threads, range coder threads) are spread over the launch (SerialThreadPlacement: one
// group per multiprocessor, one thread per warp while there is room): packed into the first launch group, 104 tile
// threads took 20.7 ms where spread they took 8.8.
//
// Bounded launches. Transform pass: DiagonalsPerLaunch diagonals per launch (the reconstruction carries the
// wavefront between launches). Range coder pass: a tile's walk is linearized into leaf slots (the 16 z-order 16x16
// positions of each of its superblocks) and a launch codes one wave of them; the range encoder is kept in the
// tile's scratchInt slot between launches. The host knows the number of launches (no readback).
//
// The syntax of a tile (mirrors Av1KeyframeEncoder.EncodeSingleTile / EncodePartition):
//   - For each 64x64 superblock of the tile in raster order:
//       64x64: SPLIT; each 32x32 inside the picture: SPLIT;
//       each 16x16 inside the picture:
//         whole block inside: partition NONE, one BLOCK_16X16 leaf
//               skip = 0, Y mode DC, UV mode DC
//               coefficients of Y (TX_16X16), U (TX_8X8), V (TX_8X8)
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
//
// Working memory (device buffers of the kernel class, grow-only):
//   coefficients   per tile: 16x16 slots * SlotInts ints (384 coefficients, then 12 block contexts)
//   transform lane bytes: above edge (33), left edge (33), predict (256); ints: forward output / quantized
//                  coefficients [0..N), dequant [N..2N), inverse output [2N..3N), inverse scratch [3N..4N+16);
//                  shorts: the residual
//   symbol thread  one symbol slot (Av1SymbolStreamGpu.SlotInts ints) and a levels scratch (1384 bytes)
//   tile           scratchInt: the range coder between launches (the caller's buffer, MinScratchIntLength ints)

using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.OpenCL;
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

    // Offsets into a thread's own byte scratch. Set by the kernels per thread; the host leaves them 0.
    /// <summary>Per-block edge buffer (above row), 33 bytes, in a transform lane's bytes.</summary>
    public int EdgeAboveOff;
    /// <summary>Per-block edge buffer (left col), 33 bytes, in a transform lane's bytes.</summary>
    public int EdgeLeftOff;
    /// <summary>Per-block predict buffer, 256 bytes (max 16x16), in a transform lane's bytes.</summary>
    public int PredictOff;
    /// <summary>Per-block libaom-layout levels[] scratch, 1384 bytes, in a symbol thread's bytes.</summary>
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

    /// <summary>First mi row of the thread's tile. Set by the kernel per thread; the host leaves it 0.</summary>
    public int TileMiRowStart;
    /// <summary>First mi column of the thread's tile. Set by the kernel per thread.</summary>
    public int TileMiColStart;
    /// <summary>One past the last mi row of the thread's tile (clipped to the frame). Set by the kernel per thread.</summary>
    public int TileMiRowEnd;
    /// <summary>One past the last mi column of the thread's tile (clipped to the frame). Set by the kernel per thread.</summary>
    public int TileMiColEnd;
}

/// <summary>Slot strides of the AV1 tile walker (source and recon per FRAME, tile output and range coder state per TILE) and the launch shapes of its passes.</summary>
public struct Av1FrameBatchStrides
{
    /// <summary>src bytes per frame (yLen + 2*uvLen).</summary>
    public int SrcStride;
    /// <summary>recon bytes per frame (coded size: ReconWidth * ReconHeight * 3 / 2).</summary>
    public int ReconStride;
    /// <summary>tileBytes per tile (worst case of the largest tile).</summary>
    public int TileStride;
    /// <summary>scratchInt ints per tile (MinScratchIntLength).</summary>
    public int ScratchIntStride;
    /// <summary>Coefficient buffer ints per tile: leaf slots of the largest tile * SlotInts. Set by RunAsync.</summary>
    public int CoefStride;
    /// <summary>Working lanes of a transform launch group (one group per tile): the lanes of the tile's wavefront. Set by RunAsync.</summary>
    public int TransformLanesPerTile;
    /// <summary>
    /// Group threads per working lane: lane L is group thread L * spread, the threads between idle (they only meet
    /// the barriers). The group has TransformLanesPerTile * TransformSpread threads. Set by RunAsync.
    /// </summary>
    public int TransformSpread;
    /// <summary>Rounds a tile's lanes take per diagonal: ceil(widest diagonal / lanes per tile). Set by RunAsync.</summary>
    public int TransformRounds;
    /// <summary>Tiles a transform launch group holds (each with its lanes). Set by RunAsync.</summary>
    public int TransformTilesPerGroup;
    /// <summary>Tiles of the launch (frames * tiles per frame): the last group's lanes past it idle. Set by RunAsync.</summary>
    public int TransformTileCount;
    /// <summary>16x16 slots of a tile the symbol pass and the range coder pass take per wave. Set by RunAsync.</summary>
    public int WaveSlots;

    /// <summary>
    /// Where the tile threads sit in the launch. Set by <see cref="Av1FrameSequentialEncodeKernel.RunAsync"/>; a
    /// caller leaves it default.
    /// </summary>
    public SerialThreadPlacement Placement;
}

/// <summary>
/// AV1 v1 keyframe tile encode kernels, bit-exact vs the CPU Av1KeyframeEncoder reference. Three passes over every
/// tile of every frame: the transform pass (one launch group per tile, the tile's 16x16 positions in anti-diagonal
/// wavefront order: predict, transform, quantize, reconstruct; quantized coefficients and block contexts to a
/// device buffer), the symbol pass (one thread per 16x16 position: its partition, mode and coefficient symbols as
/// entries of a symbol slot) and the range coder pass (one thread per tile: the entries through the tile's range
/// coder).
/// </summary>
public sealed class Av1FrameSequentialEncodeKernel : IDisposable
{
    private readonly Accelerator _accelerator;

    // Transform pass, the GPUs' entry point: explicitly grouped, a group = the lanes of a tile's wavefront (or of a
    // few tiles), with a barrier between diagonals. Null on the accelerators that use the other entry point.
    private readonly Action<
        KernelConfig,
        ArrayView<byte>,    // src (per frame: Y + U + V planes)
        ArrayView<byte>,    // recon (per frame: Y + U + V planes, coded size)
        ArrayView<int>,     // coefs (output, per tile: leaf slots * CoefsPerSlot)
        ArrayView<short>,   // dcAcQuant (DC[0..256) + AC[256..512))
        ArrayView<byte>,    // xfByte (per lane: edges + predict)
        ArrayView<int>,     // xfInt (per lane: transform working area)
        ArrayView<short>,   // xfShort (per lane: residual)
        Av1FrameSeqEncodeParams, Av1FrameBatchStrides, int, int>? _transformKernel;

    // Transform pass, the CPU accelerators' entry point (CPU, Velocity, Wasm): one thread per tile walks the same
    // diagonals alone, no barrier. They have a core per tile, not per lane, and a kernel with barriers runs through
    // their group machinery (Velocity, 832x480: 112 ms for the pass against about 12 this way). Same leaf code
    // (TransformSlot); an accelerator loads one of the two.
    private readonly Action<
        Index1D,
        ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<short>,
        ArrayView<byte>, ArrayView<int>, ArrayView<short>,
        Av1FrameSeqEncodeParams, Av1FrameBatchStrides, int, int>? _transformTileKernel;
    private readonly int _transformTileGroupSize;

    // Symbol pass: working thread = one 16x16 slot of a tile (of the current wave of slots), placed in the launch
    // by a SerialThreadPlacement.
    private readonly Action<
        Index1D,
        ArrayView<byte>,    // constsByte (Av1KeyframeConstantsGpu byte buffer)
        ArrayView<ushort>,  // constsUshort (Av1KeyframeConstantsGpu ushort buffer)
        ArrayView<int>,     // coefs (the transform pass's output: coefficients and block contexts)
        ArrayView<int>,     // sym (output: one symbol slot per thread)
        ArrayView<byte>,    // levels (per thread scratch)
        Av1FrameSeqEncodeParams, Av1FrameBatchStrides, int> _symbolizeKernel;
    private readonly int _symbolizeGroupSize;

    // Range coder pass: ONE kernel for single frames and batches: working thread = frame * tileCount + tile, placed
    // in the launch by a SerialThreadPlacement.
    private readonly Action<
        Index1D,
        ArrayView<byte>,    // tileBytes (output, one slot per tile)
        ArrayView<long>,    // tileLen (output, one element per tile)
        ArrayView<int>,     // sym (the symbol pass's output)
        ArrayView<int>,     // scratchInt (per tile: the range coder between launches)
        Av1FrameSeqEncodeParams, Av1FrameBatchStrides, int> _tileKernel;

    // The symbol pass's device buffers (grow-only): a wave's symbol slots and each thread's levels scratch.
    private MemoryBuffer1D<int, Stride1D.Dense>? _sym;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _levels;

    // The transform pass's own device buffers (grow-only): the quantized coefficients it hands to the entropy pass
    // and each lane's working area. Device memory, not LocalMemory: a long kernel with large private arrays hangs
    // the D3D12 device in Chrome (ILGPU issue 10).
    private MemoryBuffer1D<int, Stride1D.Dense>? _coefs;
    private MemoryBuffer1D<byte, Stride1D.Dense>? _xfByte;
    private MemoryBuffer1D<int, Stride1D.Dense>? _xfInt;
    private MemoryBuffer1D<short, Stride1D.Dense>? _xfShort;

    /// <summary>
    /// Quantized coefficients stored per 16x16 leaf slot: one 16x16 leaf = Y 256 + U 64 + V 64; four 8x8 leaves at
    /// the picture edge = 4 * (Y 64 + U 16 + V 16).
    /// </summary>
    public const int CoefsPerSlot = 384;

    /// <summary>Coefficients of one 8x8 leaf inside a slot (Y 64 + U 16 + V 16).</summary>
    private const int CoefsPerLeaf8 = 96;

    /// <summary>
    /// Ints of a 16x16 slot in the coefficient buffer: its coefficients, then the entropy contexts of its transform
    /// blocks (<see cref="Av1CoefEncoderGpu.ContextOf"/>: leaf * 3 + plane; leaf 0 for a 16x16 leaf, 0 .. 3 for the
    /// 8x8 leaves), which the symbol pass of the leaves below and to the right reads.
    /// </summary>
    public const int SlotInts = CoefsPerSlot + 12;

    /// <summary>Bytes of the levels scratch one symbol thread needs (libaom's padded levels buffer of a 16x16 block and slack).</summary>
    private const int LevelsLength = 1384;

    // A transform lane's working area. Bytes: above edge (33), left edge (33), predict (256).
    private const int XfEdgeAbove = 0;
    private const int XfEdgeLeft = 36;
    private const int XfPredict = 72;
    private const int XfByteLength = XfPredict + 256;
    // Ints: forward transform output / quantized coefs, dequant, inverse transform output and scratch (3n + n + 16).
    private const int XfIntLength = 1040;
    private const int XfShortLength = 256;

    // A tile's range coder between launches, in its scratchInt slot.
    private const int StStarted = 0;
    private const int StDone = 1;
    private const int StEnc = 2;   // Av1RangeEncoderGpu.StateInts

    /// <summary>Minimum scratchInt length per tile (the range coder's state between launches).</summary>
    public const int MinScratchIntLength = StEnc + Av1RangeEncoderGpu.StateInts;

    /// <summary>
    /// The most 16x16 slots of a tile one wave takes: the symbol pass does a wave's slots of every tile at once,
    /// then the range coder pass codes them, tile by tile. A range coder launch then stays at a few milliseconds
    /// (a slot is about a thousand symbols at a fine quantizer).
    /// </summary>
    public const int MaxWaveSlots = 64;

    /// <summary>
    /// Device memory the symbol slots of one wave may take (the wave shrinks to fit; at least one slot per tile).
    /// A slot is reserved at its worst case (<see cref="Av1SymbolStreamGpu.SlotInts"/> ints).
    /// </summary>
    public const long SymbolBudgetBytes = 96L << 20;

    /// <summary>
    /// Wavefront diagonals one launch of the transform pass walks. A diagonal costs one leaf per lane (predict, a
    /// 16x16 and two 8x8 forward transforms, quantize, the inverse transforms), so a launch stays at a few
    /// milliseconds; a tile of one superblock has 7 diagonals and takes one launch.
    /// </summary>
    public const int DiagonalsPerLaunch = 8;

    /// <summary>The most lanes one tile gets (a tile's wavefront is at most as wide as its shorter side).</summary>
    public const int MaxLanesPerTile = 64;

    /// <summary>
    /// Measurements (DemoConsole av1-encode-bench --xf-lanes / --xf-spread): the transform group every instance
    /// uses instead of <see cref="GetTransformGroup"/>'s choice. Lanes 0 = no override.
    /// </summary>
    internal static (int Lanes, int Spread, int TilesPerGroup) TransformGroupOverride;

    /// <summary>
    /// Measurements (av1-encode-bench --passes): when set, RunAsync waits for the device after each pass and reports
    /// (pass name, milliseconds). Null in normal use: the passes are queued without a wait between them.
    /// </summary>
    internal static Action<string, double>? PassTimer;

    /// <summary>Measurements (av1-encode-bench --sym-mb / --wave): the symbol memory budget in bytes and the most slots per wave used instead of the constants. 0 = the constants.</summary>
    internal static (long BudgetBytes, int MaxWaveSlots) WaveOverride;

    /// <summary>Compile the kernels.</summary>
    public Av1FrameSequentialEncodeKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        var type = accelerator.AcceleratorType;
        if (type == AcceleratorType.Cuda || type == AcceleratorType.OpenCL || type == AcceleratorType.WebGPU)
        {
            _transformKernel = accelerator.LoadStreamKernel<
                ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<short>,
                ArrayView<byte>, ArrayView<int>, ArrayView<short>,
                Av1FrameSeqEncodeParams, Av1FrameBatchStrides, int, int>(Av1TransformWavefrontKernel);
        }
        else
        {
            _transformTileKernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<short>,
                ArrayView<byte>, ArrayView<int>, ArrayView<short>,
                Av1FrameSeqEncodeParams, Av1FrameBatchStrides, int, int>(Av1TransformTileKernel, out var tileInfo);
            _transformTileGroupSize = tileInfo?.MinGroupSize ?? 1;
        }
        _symbolizeKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, ArrayView<ushort>, ArrayView<int>,
            ArrayView<int>, ArrayView<byte>,
            Av1FrameSeqEncodeParams, Av1FrameBatchStrides, int>(Av1SymbolizeKernel, out var symInfo);
        _symbolizeGroupSize = symInfo?.MinGroupSize ?? 1;
        _tileKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>, ArrayView<long>,
            ArrayView<int>, ArrayView<int>,
            Av1FrameSeqEncodeParams, Av1FrameBatchStrides, int>(Av1RangeCodeTileKernel, out var info);
        _groupSize = info?.MinGroupSize ?? 1;
    }

    /// <summary>Threads per launch group of the range coder kernel on this accelerator (1 when it reports none).</summary>
    private readonly int _groupSize;

    /// <summary>
    /// Where <paramref name="tileThreads"/> range coder threads sit in a launch on this accelerator (a GPU runs them
    /// several times faster spread over its multiprocessors and warps than packed into the first group; see
    /// <see cref="SerialThreadPlacement"/>).
    /// </summary>
    public SerialThreadPlacement GetPlacement(int tileThreads) =>
        SerialThreadPlacement.Create(_accelerator, _groupSize, tileThreads);

    /// <summary>
    /// The transform launch group of the wavefront entry point for <paramref name="tiles"/> tiles whose wavefront is
    /// at most <paramref name="peak"/> positions wide: working lanes per tile, group threads per working lane, and
    /// tiles per group (the group has tiles * lanes * spread threads).
    /// </summary>
    /// <remarks>
    /// Measured on CUDA (RTX 4070, 832x480 q40 = 104 tiles of 4 x 4 positions per frame; transform pass of one
    /// frame / ms per keyframe in a batch of 16). A leaf's transform code branches on its data, so lanes that run
    /// in step (a warp) pay for each other's branches, and the shape matters:
    ///   one group per tile, its 4 lanes adjacent (lanes 4, spread 1, 1 tile)    1.4 ms / 1.02   (few tiles)
    ///   the same lanes a warp apart (4, 32, 1)                                  1.4    / 1.24
    ///   the lanes of 8 tiles filling a warp (4, 1, 8)                           2.2    / 1.15
    ///   one lane per tile, 2 per warp, 40 tiles per group (1, 16, 40)           4.9    / 0.96   (many tiles)
    ///   one lane per tile, one tile per group (1, 1, 1)                         2.6    / 1.15
    /// Few tiles: latency counts, the wavefront lanes win (up to 8 frames: 1.35 against 1.50 ms per frame). Many
    /// tiles (more than 32 per multiprocessor): the device is full either way, total work counts, and one lane per
    /// tile in the placement the serial tile threads had before the split does the least (a wavefront lane idles at
    /// the barrier while its diagonal's longest leaf finishes). Larger tiles get more lanes (1920x1080 in 32 tiles
    /// of 16 x 20 positions: 16 lanes 8.8 ms, 8 lanes 11.9, 4 lanes 18.9).
    /// </remarks>
    public (int LanesPerTile, int Spread, int TilesPerGroup) GetTransformGroup(int peak, long tiles)
    {
        if (TransformGroupOverride.Lanes > 0)
            return (TransformGroupOverride.Lanes, Math.Max(1, TransformGroupOverride.Spread), Math.Max(1, TransformGroupOverride.TilesPerGroup));
        int max = Math.Max(1, _accelerator.MaxNumThreadsPerGroup);
        var type = _accelerator.AcceleratorType;
        bool nvidia = type == AcceleratorType.Cuda
            || (type == AcceleratorType.OpenCL && _accelerator.Device is CLDevice cl && cl.Vendor == CLDeviceVendor.Nvidia);
        int warp = nvidia ? _accelerator.WarpSize : 0;
        int processors = nvidia ? _accelerator.NumMultiprocessors : 0;
        if (warp >= 2 && processors > 0 && tiles > 32L * processors && 20 * warp <= max)
            return (1, warp / 2, 40);
        int cap = Math.Min(MaxLanesPerTile, max);
        int lanes = 1;
        while (lanes < peak && lanes * 2 <= cap) lanes <<= 1;
        return (lanes, 1, 1);
    }

    /// <summary>
    /// 16x16 slots of a tile the symbol pass and the range coder pass take per wave, for <paramref name="tiles"/>
    /// tiles of at most <paramref name="slotsPerTile"/> slots: all of a tile's slots, at most
    /// <see cref="MaxWaveSlots"/>, and no more than fit <see cref="SymbolBudgetBytes"/> for all tiles together (a
    /// slot is a symbol slot and a levels scratch); at least one.
    /// </summary>
    public static int GetWaveSlots(long tiles, int slotsPerTile)
    {
        long perSlotBytes = Av1SymbolStreamGpu.SlotInts * 4L + LevelsLength;
        long budget = WaveOverride.BudgetBytes > 0 ? WaveOverride.BudgetBytes : SymbolBudgetBytes;
        int maxWave = WaveOverride.MaxWaveSlots > 0 ? WaveOverride.MaxWaveSlots : MaxWaveSlots;
        return (int)Math.Max(1, Math.Min(Math.Min(slotsPerTile, maxWave), budget / (tiles * perSlotBytes)));
    }

    /// <summary>Superblock columns and rows of the largest tile of <paramref name="p"/> (clipped to the frame).</summary>
    private static (int SbCols, int SbRows) LargestTileSb(Av1FrameSeqEncodeParams p)
    {
        int tileSbCols = p.TileWidthSb, tileSbRows = p.TileHeightSb;
        int frameSbCols = (p.FrameMiCols + 15) >> 4, frameSbRows = (p.FrameMiRows + 15) >> 4;
        if (tileSbCols > frameSbCols) tileSbCols = frameSbCols;
        if (tileSbRows > frameSbRows) tileSbRows = frameSbRows;
        return (tileSbCols, tileSbRows);
    }

    /// <summary>
    /// Encode the tiles of <paramref name="frameCount"/> frames (tile g = frame * tileCount + tile). First the
    /// transform pass: one launch group per tile walks the tile's 16x16 positions in wavefront order (a position
    /// needs the reconstruction of the one above it and the one to its left, so the positions of an anti-diagonal are
    /// independent), <see cref="DiagonalsPerLaunch"/> diagonals per launch. Then, a wave of slots at a time (at most
    /// <see cref="MaxWaveSlots"/> per tile, fewer when <see cref="SymbolBudgetBytes"/> asks): the symbol pass, one
    /// thread per 16x16 slot of the wave, and the range coder pass, frameCount * tileCount threads (placed in the
    /// launch by <see cref="GetPlacement"/>), each coding its tile's symbols of the wave with its own range encoder
    /// state into its own output slot (bytes at <c>g * strides.TileStride</c>, length in <c>tileLen[g]</c>). Zeroes
    /// <paramref name="scratchInt"/> first (a range coder that has not started). The work is queued (paced with
    /// <see cref="LaunchPacer"/>), not awaited.
    /// </summary>
    public async Task RunAsync(
        ArrayView<byte> src, ArrayView<byte> recon,
        ArrayView<byte> tileBytes, ArrayView<long> tileLen,
        ArrayView<byte> constsByte, ArrayView<ushort> constsUshort,
        ArrayView<short> dcAcQuant,
        ArrayView<int> scratchInt,
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
        scratchInt.MemSetToZero();

        // ---- Transform pass ----
        var (tileSbCols, tileSbRows) = LargestTileSb(p);
        int slots = tileSbRows * tileSbCols * 16;
        // the largest tile in 16x16 positions (a tile at the picture edge may end inside a superblock)
        int cols16 = Math.Min(tileSbCols * 4, (p.FrameMiCols + 3) >> 2);
        int rows16 = Math.Min(tileSbRows * 4, (p.FrameMiRows + 3) >> 2);
        int diagonals = cols16 + rows16 - 1;
        int peak = Math.Min(cols16, rows16);
        var (lanes, spread, tilesPerGroup) = _transformKernel != null ? GetTransformGroup(peak, threads) : (1, 1, 1);
        int groups = (int)((threads + tilesPerGroup - 1) / tilesPerGroup);
        strides.CoefStride = checked(slots * SlotInts);
        strides.TransformLanesPerTile = lanes;
        strides.TransformSpread = spread;
        strides.TransformRounds = (peak + lanes - 1) / lanes;
        strides.TransformTilesPerGroup = tilesPerGroup;
        strides.TransformTileCount = (int)threads;
        EnsureTransformBuffers(threads, strides.CoefStride, threads * lanes);
        // the tile entry point: one working thread per tile, placed like the entropy threads
        strides.Placement = SerialThreadPlacement.Create(_accelerator, _transformTileGroupSize, (int)threads);
        long tileExtent = strides.Placement.Extent;
        if (tileExtent > int.MaxValue) throw new ArgumentException("frameCount * tileCount exceeds the launch extent.", nameof(frameCount));

        var pacer = new LaunchPacer(_accelerator);
        var passClock = PassTimer != null ? System.Diagnostics.Stopwatch.StartNew() : null;
        if (passClock != null) { await _accelerator.SynchronizeAsync(); passClock.Restart(); }
        for (int d = 0; d < diagonals; d += DiagonalsPerLaunch)
        {
            int dEnd = Math.Min(d + DiagonalsPerLaunch, diagonals);
            if (_transformKernel != null)
                _transformKernel(new KernelConfig(groups, tilesPerGroup * lanes * spread),
                    src, recon, _coefs!.View, dcAcQuant,
                    _xfByte!.View, _xfInt!.View, _xfShort!.View,
                    p, strides, d, dEnd);
            else
                _transformTileKernel!(new Index1D((int)tileExtent),
                    src, recon, _coefs!.View, dcAcQuant,
                    _xfByte!.View, _xfInt!.View, _xfShort!.View,
                    p, strides, d, dEnd);
            await pacer.LaunchedAsync();
        }

        if (passClock != null)
        {
            await _accelerator.SynchronizeAsync();
            PassTimer!("transform", passClock.Elapsed.TotalMilliseconds);
            passClock.Restart();
        }

        // ---- Symbol pass + range coder pass, a wave of slots at a time ----
        int wave = GetWaveSlots(threads, slots);
        long symThreads = threads * wave;
        if (symThreads > int.MaxValue) throw new ArgumentException("frameCount * tileCount * wave exceeds the launch extent.", nameof(frameCount));
        EnsureSymbolBuffers(symThreads);
        strides.WaveSlots = wave;
        var symPlacement = SerialThreadPlacement.Create(_accelerator, _symbolizeGroupSize, (int)symThreads);
        var tilePlacement = GetPlacement((int)threads);
        if (symPlacement.Extent > int.MaxValue || tilePlacement.Extent > int.MaxValue)
            throw new ArgumentException("frameCount * tileCount exceeds the launch extent.", nameof(frameCount));
        double symbolMs = 0, rangeMs = 0;
        for (int w = 0; w < slots; w += wave)
        {
            strides.Placement = symPlacement;
            _symbolizeKernel(new Index1D((int)symPlacement.Extent),
                constsByte, constsUshort, _coefs!.View,
                _sym!.View, _levels!.View, p, strides, w);
            await pacer.LaunchedAsync();
            if (passClock != null)
            {
                await _accelerator.SynchronizeAsync();
                symbolMs += passClock.Elapsed.TotalMilliseconds;
                passClock.Restart();
            }
            strides.Placement = tilePlacement;
            _tileKernel(new Index1D((int)tilePlacement.Extent),
                tileBytes, tileLen,
                _sym!.View, scratchInt, p, strides, w);
            await pacer.LaunchedAsync();
            if (passClock != null)
            {
                await _accelerator.SynchronizeAsync();
                rangeMs += passClock.Elapsed.TotalMilliseconds;
                passClock.Restart();
            }
        }
        if (passClock != null)
        {
            PassTimer!("symbols", symbolMs);
            PassTimer!("range coder", rangeMs);
        }
    }

    /// <summary>The symbol pass's buffers, grown when a run needs more (never shrunk).</summary>
    private void EnsureSymbolBuffers(long symThreads)
    {
        if (_sym == null || _sym.Length < symThreads * Av1SymbolStreamGpu.SlotInts)
        {
            _sym?.Dispose(); _levels?.Dispose();
            _sym = _accelerator.Allocate1D<int>(symThreads * Av1SymbolStreamGpu.SlotInts);
            _levels = _accelerator.Allocate1D<byte>(symThreads * LevelsLength);
        }
    }

    /// <summary>The transform pass's buffers, grown when a run needs more (never shrunk).</summary>
    private void EnsureTransformBuffers(long tiles, int coefStride, long laneCount)
    {
        long coefLen = tiles * coefStride;
        if (_coefs == null || _coefs.Length < coefLen)
        {
            _coefs?.Dispose();
            _coefs = _accelerator.Allocate1D<int>(coefLen);
        }
        if (_xfByte == null || _xfByte.Length < laneCount * XfByteLength)
        {
            _xfByte?.Dispose(); _xfInt?.Dispose(); _xfShort?.Dispose();
            _xfByte = _accelerator.Allocate1D<byte>(laneCount * XfByteLength);
            _xfInt = _accelerator.Allocate1D<int>(laneCount * XfIntLength);
            _xfShort = _accelerator.Allocate1D<short>(laneCount * XfShortLength);
        }
    }

    /// <summary>Release the transform pass's buffers (the kernel delegates hold nothing).</summary>
    public void Dispose()
    {
        _coefs?.Dispose(); _coefs = null;
        _xfByte?.Dispose(); _xfByte = null;
        _xfInt?.Dispose(); _xfInt = null;
        _xfShort?.Dispose(); _xfShort = null;
        _sym?.Dispose(); _sym = null;
        _levels?.Dispose(); _levels = null;
    }

    // ===========================================================================
    // Kernel entry points
    // ===========================================================================

    /// <summary>
    /// A thread's own copy of the parameters for tile <paramref name="t"/> of a frame (raster order): the tile's
    /// rectangle in mi.
    /// </summary>
    private static Av1FrameSeqEncodeParams TileParams(Av1FrameSeqEncodeParams p, int t)
    {
        int tileRow = t / p.TileCols;
        int tileCol = t - tileRow * p.TileCols;
        var tp = p;
        tp.TileMiRowStart = tileRow * p.TileHeightSb * 16;
        tp.TileMiColStart = tileCol * p.TileWidthSb * 16;
        int rowEnd = tp.TileMiRowStart + p.TileHeightSb * 16;
        int colEnd = tp.TileMiColStart + p.TileWidthSb * 16;
        tp.TileMiRowEnd = rowEnd < p.FrameMiRows ? rowEnd : p.FrameMiRows;
        tp.TileMiColEnd = colEnd < p.FrameMiCols ? colEnd : p.FrameMiCols;
        return tp;
    }

    /// <summary>
    /// Transform pass. Group = one tile of one frame (<c>Grid.IdxX = frame * tileCount + tile</c>); its working
    /// lanes (group thread L * TransformSpread is lane L, the threads between only meet the barriers) walk the
    /// tile's 16x16 positions diagonal by diagonal (<paramref name="diagStart"/> .. <paramref name="diagEnd"/>),
    /// with a barrier after each: position (row, col) is on diagonal row + col, and what it predicts from (the
    /// reconstruction above it and to its left) is on diagonal row + col - 1. Lane L takes positions L,
    /// L + lanes, ... of a diagonal. Each position: predict, forward transform, quantize, reconstruct; the
    /// quantized coefficients go to the tile's coefficient slots (the entropy pass's z-order slot index).
    /// </summary>
    private static void Av1TransformWavefrontKernel(
        ArrayView<byte> src, ArrayView<byte> recon,
        ArrayView<int> coefs,
        ArrayView<short> dcAcQuant,
        ArrayView<byte> xfByte, ArrayView<int> xfInt, ArrayView<short> xfShort,
        Av1FrameSeqEncodeParams p,
        Av1FrameBatchStrides s,
        int diagStart, int diagEnd)
    {
        int groupThread = Group.IdxX;
        int unit = groupThread / s.TransformSpread;
        int tileInGroup = unit / s.TransformLanesPerTile;
        int lane = unit - tileInGroup * s.TransformLanesPerTile;
        int g = Grid.IdxX * s.TransformTilesPerGroup + tileInGroup;
        // only the first thread of each spread works, and only for a tile that exists; the others idle on tile 0's
        // (valid) views
        bool live = (unit * s.TransformSpread == groupThread) & (g < s.TransformTileCount);
        if (g >= s.TransformTileCount) g = 0;
        int tileCount = p.TileCols * p.TileRows;
        int f = g / tileCount;
        var tp = TileParams(p, g - f * tileCount);
        // this lane's working area
        tp.EdgeAboveOff = XfEdgeAbove;
        tp.EdgeLeftOff = XfEdgeLeft;
        tp.PredictOff = XfPredict;

        var fSrc = src.SubView((long)f * s.SrcStride, s.SrcStride);
        var fRecon = recon.SubView((long)f * s.ReconStride, s.ReconStride);
        var tCoefs = coefs.SubView((long)g * s.CoefStride, s.CoefStride);
        long thread = (long)g * s.TransformLanesPerTile + lane;
        var tByte = xfByte.SubView(thread * XfByteLength, XfByteLength);
        var tInt = xfInt.SubView(thread * XfIntLength, XfIntLength);
        var tShort = xfShort.SubView(thread * XfShortLength, XfShortLength);

        int qDc = (int)dcAcQuant[p.BaseQIndex];
        int qAc = (int)dcAcQuant[256 + p.BaseQIndex];

        // THIS tile in 16x16 positions and in superblocks
        int cols16 = (tp.TileMiColEnd - tp.TileMiColStart + 3) >> 2;
        int rows16 = (tp.TileMiRowEnd - tp.TileMiRowStart + 3) >> 2;
        int sbCols = (tp.TileMiColEnd - tp.TileMiColStart + 15) >> 4;

        for (int d = diagStart; d < diagEnd; d++)
        {
            int rMin = d - (cols16 - 1);
            if (rMin < 0) rMin = 0;
            int rMax = d < rows16 - 1 ? d : rows16 - 1;
            int count = rMax - rMin + 1;   // positions of this tile on diagonal d (0 or less past its last diagonal)
            for (int k = 0; k < s.TransformRounds; k++)
            {
                int pos = lane + k * s.TransformLanesPerTile;
                if (live & (pos < count))
                {
                    int row = rMin + pos;
                    TransformSlot(
                        fSrc, fRecon, tCoefs,
                        qDc, qAc,
                        tByte, tInt, tShort,
                        row, d - row, sbCols, tp);
                }
            }
            Group.Barrier();
        }
    }

    /// <summary>
    /// Transform pass, one thread per tile (the CPU accelerators): working thread = one tile of one frame, which
    /// walks the same diagonals as <see cref="Av1TransformWavefrontKernel"/>, every position of a diagonal itself
    /// (any order that takes a position after the one above it and the one to its left gives the same result).
    /// </summary>
    private static void Av1TransformTileKernel(
        Index1D idx,
        ArrayView<byte> src, ArrayView<byte> recon,
        ArrayView<int> coefs,
        ArrayView<short> dcAcQuant,
        ArrayView<byte> xfByte, ArrayView<int> xfInt, ArrayView<short> xfShort,
        Av1FrameSeqEncodeParams p,
        Av1FrameBatchStrides s,
        int diagStart, int diagEnd)
    {
        int g = SerialThreadPlacement.WorkingThread(s.Placement, idx.X);
        if (g < 0) return;
        int tileCount = p.TileCols * p.TileRows;
        int f = g / tileCount;
        var tp = TileParams(p, g - f * tileCount);
        tp.EdgeAboveOff = XfEdgeAbove;
        tp.EdgeLeftOff = XfEdgeLeft;
        tp.PredictOff = XfPredict;

        var fSrc = src.SubView((long)f * s.SrcStride, s.SrcStride);
        var fRecon = recon.SubView((long)f * s.ReconStride, s.ReconStride);
        var tCoefs = coefs.SubView((long)g * s.CoefStride, s.CoefStride);
        var tByte = xfByte.SubView((long)g * XfByteLength, XfByteLength);
        var tInt = xfInt.SubView((long)g * XfIntLength, XfIntLength);
        var tShort = xfShort.SubView((long)g * XfShortLength, XfShortLength);

        int qDc = (int)dcAcQuant[p.BaseQIndex];
        int qAc = (int)dcAcQuant[256 + p.BaseQIndex];
        int cols16 = (tp.TileMiColEnd - tp.TileMiColStart + 3) >> 2;
        int rows16 = (tp.TileMiRowEnd - tp.TileMiRowStart + 3) >> 2;
        int sbCols = (tp.TileMiColEnd - tp.TileMiColStart + 15) >> 4;

        for (int d = diagStart; d < diagEnd; d++)
        {
            int rMin = d - (cols16 - 1);
            if (rMin < 0) rMin = 0;
            int rMax = d < rows16 - 1 ? d : rows16 - 1;
            for (int row = rMin; row <= rMax; row++)
                TransformSlot(
                    fSrc, fRecon, tCoefs,
                    qDc, qAc,
                    tByte, tInt, tShort,
                    row, d - row, sbCols, tp);
        }
    }

    /// <summary>
    /// Symbol pass. Working thread t = one 16x16 slot of one tile in the current wave of slots: tile
    /// <c>g = t / WaveSlots</c> (<c>g = frame * tileCount + tile</c>), slot <c>waveStart + t % WaveSlots</c> of the
    /// tile's z-order walk. It writes the slot's symbol entries (count first) to slot t of <paramref name="sym"/>.
    /// Nothing it reads is written by another thread of the pass: the coefficients and the blocks' contexts are the
    /// transform pass's, the partition contexts follow from where the picture ends.
    /// </summary>
    private static void Av1SymbolizeKernel(
        Index1D idx,
        ArrayView<byte> constsByte, ArrayView<ushort> constsUshort,
        ArrayView<int> coefs,
        ArrayView<int> sym,
        ArrayView<byte> levels,
        Av1FrameSeqEncodeParams p,
        Av1FrameBatchStrides s,
        int waveStart)
    {
        int t = SerialThreadPlacement.WorkingThread(s.Placement, idx.X);
        if (t < 0) return;
        int g = t / s.WaveSlots;
        int slot = waveStart + (t - g * s.WaveSlots);
        int tileCount = p.TileCols * p.TileRows;
        int f = g / tileCount;
        var tp = TileParams(p, g - f * tileCount);
        tp.LevelsOff = 0;   // this thread's own levels scratch

        var tCoefs = coefs.SubView((long)g * s.CoefStride, s.CoefStride);
        var tLevels = levels.SubView((long)t * LevelsLength, LevelsLength);
        long slotBase = (long)t * Av1SymbolStreamGpu.SlotInts;

        // superblocks of THIS tile
        int sbCols = (tp.TileMiColEnd - tp.TileMiColStart + 15) >> 4;
        int sbRows = (tp.TileMiRowEnd - tp.TileMiRowStart + 15) >> 4;
        int n = 0;
        if (slot < sbRows * sbCols * 16)
            SymbolizeSlot(sym, slotBase, ref n, constsByte, constsUshort, tCoefs, tLevels, slot, sbCols, tp);
        sym[slotBase] = n;
    }

    /// <summary>
    /// Range coder pass. Working thread = one tile of one frame (<c>g = frame * tileCount + tile</c>): it codes the
    /// symbol entries of the tile's slots of the current wave, in order, into the tile's bytes. The range encoder
    /// lives in the tile's scratchInt slot between launches; the launch that codes the tile's last slot finishes
    /// the coder and writes tileLen.
    /// </summary>
    private static void Av1RangeCodeTileKernel(
        Index1D idx,
        ArrayView<byte> tileBytes, ArrayView<long> tileLen,
        ArrayView<int> sym,
        ArrayView<int> scratchInt,
        Av1FrameSeqEncodeParams p,
        Av1FrameBatchStrides s,
        int waveStart)
    {
        // Only the placement's working threads of the launch code a tile; the others return at once.
        int g = SerialThreadPlacement.WorkingThread(s.Placement, idx.X);
        if (g < 0) return;
        int tileCount = p.TileCols * p.TileRows;
        int f = g / tileCount;
        var tp = TileParams(p, g - f * tileCount);

        var tTile = tileBytes.SubView((long)g * s.TileStride, s.TileStride);
        var tTileLen = tileLen.SubView(g, 1);
        var tState = scratchInt.SubView((long)g * s.ScratchIntStride, s.ScratchIntStride);
        RangeCodeStep(tTile, tTileLen, sym, (long)g * s.WaveSlots * Av1SymbolStreamGpu.SlotInts, tState, tp, s.WaveSlots, waveStart);
    }

    /// <summary>One bounded step of one tile's range coder: the symbol entries of slots waveStart .. waveStart + waveSlots.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void RangeCodeStep(
        ArrayView<byte> tileBytes, ArrayView<long> tileLen,
        ArrayView<int> sym, long symBase,
        ArrayView<int> state,
        Av1FrameSeqEncodeParams p,
        int waveSlots, int waveStart)
    {
        if (state[StDone] != 0) return;

        Av1RangeEncoderGpuState re;
        if (state[StStarted] == 0)
        {
            re = Av1RangeEncoderGpu.Init();
            state[StStarted] = 1;
        }
        else
        {
            re = Av1RangeEncoderGpu.Load(state, StEnc);
        }

        // superblocks of THIS tile
        int sbCols = (p.TileMiColEnd - p.TileMiColStart + 15) >> 4;
        int sbRows = (p.TileMiRowEnd - p.TileMiRowStart + 15) >> 4;
        int slotCount = sbRows * sbCols * 16;
        int end = waveStart + waveSlots;
        if (end > slotCount) end = slotCount;
        for (int slot = waveStart; slot < end; slot++)
        {
            long slotBase = symBase + (long)(slot - waveStart) * Av1SymbolStreamGpu.SlotInts;
            int n = sym[slotBase];
            for (int i = 1; i <= n; i++)
                Av1SymbolStreamGpu.Encode(ref re, tileBytes, sym[slotBase + i]);
        }

        if (end >= slotCount)
        {
            Av1RangeEncoderGpu.Done(ref re, tileBytes);
            tileLen[0] = re.OutLen;
            state[StDone] = 1;
        }
        Av1RangeEncoderGpu.Save(state, StEnc, ref re);
    }

    // ===========================================================================
    // Transform pass: one 16x16 position
    // ===========================================================================

    /// <summary>
    /// The 16x16 position (<paramref name="row"/>, <paramref name="col"/>) of the tile (it starts inside the picture):
    /// its 16x16 leaf, or, where it crosses the picture edge, the 8x8 leaves inside the picture in z-order (the
    /// second predicts from the first, as in the serial walk). Coefficients go to the entropy pass's slot of this
    /// position: superblock * 16 + the z-order index of the position inside its superblock.
    /// </summary>
    private static void TransformSlot(
        ArrayView<byte> src, ArrayView<byte> recon,
        ArrayView<int> coefs,
        int qDc, int qAc,
        ArrayView<byte> xfByte, ArrayView<int> xfInt, ArrayView<short> xfShort,
        int row, int col, int sbCols,
        Av1FrameSeqEncodeParams p)
    {
        int miRow16 = p.TileMiRowStart + row * 4;
        int miCol16 = p.TileMiColStart + col * 4;
        int sb = (row >> 2) * sbCols + (col >> 2);
        int q32 = ((row >> 1) & 1) * 2 + ((col >> 1) & 1);
        int q16 = (row & 1) * 2 + (col & 1);
        int slotBase = (sb * 16 + q32 * 4 + q16) * SlotInts;

        bool whole16 = (miRow16 + 2 < p.FrameMiRows) & (miCol16 + 2 < p.FrameMiCols);
        // One loop and ONE TransformLeaf call site for both cases (a second call site would be a second copy of the
        // whole leaf pipeline in the generated shader): 1 iteration for the 16x16 leaf, 4 for the 8x8 positions.
        int leafCount = whole16 ? 1 : 4;
        for (int k = 0; k < leafCount; k++)
        {
            int miRowLeaf = miRow16 + (k >> 1) * 2;
            int miColLeaf = miCol16 + (k & 1) * 2;
            bool inside = (miRowLeaf < p.FrameMiRows) & (miColLeaf < p.FrameMiCols);
            if (inside)
                TransformLeaf(src, recon, coefs, slotBase + (whole16 ? 0 : k * CoefsPerLeaf8),
                    slotBase + CoefsPerSlot + (whole16 ? 0 : k * 3),
                    qDc, qAc, xfByte, xfInt, xfShort, miRowLeaf, miColLeaf, whole16, p);
        }
    }

    /// <summary>
    /// The three planes of one leaf. <paramref name="is16"/>: BLOCK_16X16 (Y TX_16X16, U / V TX_8X8); otherwise
    /// BLOCK_8X8 at the picture edge (Y TX_8X8, U / V TX_4X4). Coefficients at <paramref name="coefBase"/>: Y, U, V.
    /// <para>
    /// Three <see cref="TransformPlane"/> call sites, one per transform size, each with the size a CONSTANT (the
    /// body folds to that size; a run-time size keeps every size's code live and is slower on every backend). The
    /// 8x8 site serves the chroma of a 16x16 leaf and the luma of an edge leaf, so the edge leaves add only the 4x4
    /// site to the shader.
    /// </para>
    /// </summary>
    private static void TransformLeaf(
        ArrayView<byte> src, ArrayView<byte> recon,
        ArrayView<int> coefs, int coefBase, int ctxBase,
        int qDc, int qAc,
        ArrayView<byte> xfByte, ArrayView<int> xfInt, ArrayView<short> xfShort,
        int miRow, int miCol, bool is16, Av1FrameSeqEncodeParams p)
    {
        int chromaX = (miCol * 4) >> 1, chromaY = (miRow * 4) >> 1;
        int reconChromaW = p.ReconWidth >> 1;

        // ---- TX_16X16: luma of a 16x16 leaf ----
        if (is16)
            TransformPlane(
                src, p.YPlaneOff, p.Width, p.Width, p.Height,
                recon, 0, p.ReconWidth,
                coefs, coefBase, ctxBase,
                qDc, qAc,
                xfByte, xfInt, xfShort,
                plane: 0,
                xPx: miCol * 4, yPx: miRow * 4,
                txW: 16, txH: 16, txSizeIdx: 2,
                p);

        // ---- TX_8X8: U then V of a 16x16 leaf; luma of an 8x8 leaf ----
        int first8 = is16 ? 1 : 0;
        int last8 = is16 ? 2 : 0;
        for (int plane = first8; plane <= last8; plane++)
        {
            bool luma = plane == 0;
            TransformPlane(
                src, luma ? p.YPlaneOff : (plane == 1 ? p.UPlaneOff : p.VPlaneOff),
                luma ? p.Width : p.ChromaWidth, luma ? p.Width : p.ChromaWidth, luma ? p.Height : p.ChromaHeight,
                recon, luma ? 0 : (plane == 1 ? p.ReconUOff : p.ReconVOff), luma ? p.ReconWidth : reconChromaW,
                coefs, luma ? coefBase : coefBase + 256 + (plane - 1) * 64, ctxBase + plane,
                qDc, qAc,
                xfByte, xfInt, xfShort,
                plane,
                xPx: luma ? miCol * 4 : chromaX, yPx: luma ? miRow * 4 : chromaY,
                txW: 8, txH: 8, txSizeIdx: 1,
                p);
        }

        // ---- TX_4X4: U then V of an 8x8 leaf ----
        if (!is16)
        {
            for (int plane = 1; plane <= 2; plane++)
            {
                TransformPlane(
                    src, plane == 1 ? p.UPlaneOff : p.VPlaneOff, p.ChromaWidth, p.ChromaWidth, p.ChromaHeight,
                    recon, plane == 1 ? p.ReconUOff : p.ReconVOff, reconChromaW,
                    coefs, coefBase + 64 + (plane - 1) * 16, ctxBase + plane,
                    qDc, qAc,
                    xfByte, xfInt, xfShort,
                    plane,
                    xPx: chromaX, yPx: chromaY,
                    txW: 4, txH: 4, txSizeIdx: 0,
                    p);
            }
        }
    }

    /// <summary>
    /// One transform block: DC prediction from the reconstruction, residual, forward transform, quantization (the
    /// quantized coefficients to <paramref name="coefs"/> at <paramref name="coefBase"/>, raster order), and the
    /// reconstruction a decoder makes from them. The block's entropy context goes to <paramref name="ctxIndex"/>.
    /// </summary>
    private static void TransformPlane(
        ArrayView<byte> srcBuf, int srcPlaneOff, int srcStride, int srcW, int srcH,
        ArrayView<byte> reconBuf, int reconPlaneOff, int reconStride,
        ArrayView<int> coefs, int coefBase, int ctxIndex,
        int qDc, int qAc,
        ArrayView<byte> xfByte, ArrayView<int> xfInt, ArrayView<short> xfShort,
        int plane,
        int xPx, int yPx,
        int txW, int txH, int txSizeIdx,
        Av1FrameSeqEncodeParams p)
    {
        // txW / txH / txSizeIdx are CONSTANTS at every call site (see TransformLeaf).
        // txSizeIdx: 2 = TX_16X16, 1 = TX_8X8, 0 = TX_4X4 (chroma of an edge 8x8 block). srcW x srcH is the source
        // plane; the block may extend past it at the picture's right / bottom edge (the coded size is rounded up to
        // 8), where the last source column / row is repeated.
        int n = txW * txH;

        // Build edge buffers from the recon plane.
        // Neighbours are available only inside the tile (AV1 availability): nothing above the tile's first row,
        // nothing left of its first column, and the above-right / below-left extensions stop at the tile's right /
        // bottom edge (another group is writing the pixels past it).
        int pxPerMi = plane == 0 ? 4 : 2;
        bool haveAbove = yPx > p.TileMiRowStart * pxPerMi;
        bool haveLeft = xPx > p.TileMiColStart * pxPerMi;
        BuildEdgeBuffer(reconBuf, reconPlaneOff, reconStride,
            p.TileMiColEnd * pxPerMi, p.TileMiRowEnd * pxPerMi,
            xPx, yPx, txW, txH, haveAbove, haveLeft, xfByte, p);

        // DC predict into xfByte[PredictOff..+n).
        Av1DcPredictorGpu.DcPred(
            xfByte, p.PredictOff, txW,
            xfByte, p.EdgeAboveOff,
            xfByte, p.EdgeLeftOff,
            txW, txH, haveAbove, haveLeft);

        // Compute residual = src - predict into xfShort[0..+n).
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
                short pp = (short)xfByte[pOff + c];
                xfShort[r * txW + c] = (short)(s - pp);
            }
        }

        // Forward 2D transform: residual (xfShort) -> coefs (xfInt[0..n)).
        // Forward transform scratch goes at xfInt[n..2n).
        if (txSizeIdx == 1)
        {
            Av1Forward2dTransformGpu.Forward8x8DctDct(
                xfShort, 0, xfInt, 0, xfInt, n);
        }
        else if (txSizeIdx == 0)
        {
            Av1Forward2dTransformGpu.Forward4x4DctDct(
                xfShort, 0, xfInt, 0, xfInt, n);
        }
        else
        {
            Av1Forward2dTransformGpu.Forward16x16DctDct(
                xfShort, 0, xfInt, 0, xfInt, n);
        }

        // Quantize in place, then hand the block to the entropy pass. A block with a nonzero coefficient has eob > 0.
        Av1ForwardQuantizerGpu.QuantizeBlock(xfInt, 0, n, qDc, qAc);
        int any = 0;
        for (int i = 0; i < n; i++)
        {
            int level = xfInt[i];
            coefs[coefBase + i] = level;
            any |= level;
        }
        // the block's entropy context, for the symbol pass of the blocks below and to the right
        coefs[ctxIndex] = Av1CoefEncoderGpu.ContextOf(xfInt, 0, n);

        // Reconstruct: dequant + inverse 2D transform + add to predict + clip.
        ReconstructBlock(xfInt, txSizeIdx, txW, txH, n, qDc, qAc,
            xfByte, p.PredictOff,
            reconBuf, reconPlaneOff + yPx * reconStride + xPx, reconStride,
            any != 0 ? 1 : 0);
    }

    // ===========================================================================
    // Symbol pass: partition walk (64x64 -> 32x32 -> 16x16 -> NONE), one 16x16 slot
    // ===========================================================================

    /// <summary>
    /// The symbols of one 16x16 position of the superblock z-order walk (slot = sb * 16 + q32 * 4 + q16), in coding
    /// order: the partition syntax that precedes the block (64x64 SPLIT at the superblock's first slot, 32x32 SPLIT
    /// at a 32x32 block's first slot), then the 16x16 NONE leaf, or, where the 16x16 crosses the picture edge, its
    /// SPLIT and the 8x8 NONE leaves inside the picture. Out-of-frame blocks are skipped exactly as the recursive
    /// walk skips them. The leaf's quantized coefficients and its neighbours' contexts are in
    /// <paramref name="coefs"/> (the transform pass).
    /// </summary>
    private static void SymbolizeSlot(
        ArrayView<int> sym, long slotBase, ref int n,
        ArrayView<byte> constsByte, ArrayView<ushort> constsUshort,
        ArrayView<int> coefs,
        ArrayView<byte> levels,
        int slot, int sbCols,
        Av1FrameSeqEncodeParams p)
    {
        int frameMiRows = p.FrameMiRows, frameMiCols = p.FrameMiCols;
        int sb = slot >> 4;
        int q = slot & 15;
        int sbRow = sb / sbCols;                       // superblock row / column inside the tile
        int sbCol = sb - sbRow * sbCols;
        int sbMiRow = p.TileMiRowStart + sbRow * 16;   // frame coordinates
        int sbMiCol = p.TileMiColStart + sbCol * 16;

        if (q == 0)
            SymbolizeSplit(sym, slotBase, ref n, constsUshort, p, sbMiRow, sbMiCol, BsizeBlock64x64, 8);

        int q32 = q >> 2;
        int q16 = q & 3;
        int miRow32 = sbMiRow + (q32 >> 1) * 8;
        int miCol32 = sbMiCol + (q32 & 1) * 8;
        bool in32 = (miRow32 < frameMiRows) & (miCol32 < frameMiCols);
        if (!in32) return;

        if (q16 == 0)
            SymbolizeSplit(sym, slotBase, ref n, constsUshort, p, miRow32, miCol32, BsizeBlock32x32, 4);

        int miRow16 = miRow32 + (q16 >> 1) * 4;
        int miCol16 = miCol32 + (q16 & 1) * 4;
        bool in16 = (miRow16 < frameMiRows) & (miCol16 < frameMiCols);
        if (!in16) return;

        // The picture is a whole number of 8x8 blocks (2 mi), so a 16x16 that starts inside is either whole or has
        // only its upper half, its left half or its top-left 8x8 inside. Whole: PARTITION_NONE and one 16x16 leaf.
        // Crossing the bottom / right edge: PARTITION_NONE is not codable; SPLIT, then the 8x8 leaves inside.
        bool whole16 = (miRow16 + 2 < frameMiRows) & (miCol16 + 2 < frameMiCols);
        if (whole16)
            SymbolizePartition(sym, slotBase, ref n, constsUshort, p,
                miRow16, miCol16, BsizeBlock16x16, PartitionNone);
        else
            SymbolizeSplit(sym, slotBase, ref n, constsUshort, p, miRow16, miCol16, BsizeBlock16x16, 2);

        // One loop and ONE SymbolizeLeaf call site for both cases (a second call site would be a second copy of the
        // whole leaf syntax in the generated shader): 1 iteration for the 16x16 leaf, 4 for the 8x8 positions.
        int coefSlotBase = slot * SlotInts;
        int leafCount = whole16 ? 1 : 4;
        for (int k = 0; k < leafCount; k++)
        {
            int miRowLeaf = miRow16 + (k >> 1) * 2;
            int miColLeaf = miCol16 + (k & 1) * 2;
            bool inside = (miRowLeaf < frameMiRows) & (miColLeaf < frameMiCols);
            if (inside)
            {
                // An 8x8 block that starts inside is whole: always the full (4-symbol) partition CDF.
                if (!whole16)
                    SymbolizePartition(sym, slotBase, ref n, constsUshort, p,
                        miRowLeaf, miColLeaf, BsizeBlock8x8, PartitionNone);

                SymbolizeLeaf(sym, slotBase, ref n,
                    constsByte, constsUshort,
                    coefs, coefSlotBase + (whole16 ? 0 : k * CoefsPerLeaf8),
                    levels, sbCols,
                    miRowLeaf, miColLeaf, whole16, p);
            }
        }
    }

    /// <summary>
    /// The partition syntax of a block the walker splits (libaom write_partition / AV1 spec 5.11.4): the full
    /// partition symbol when the block's lower and right halves both start inside the picture, split_or_horz when
    /// only the lower half is outside, split_or_vert when only the right half is outside, nothing when both are
    /// (PARTITION_SPLIT is implied). <paramref name="halfMi"/> is half the block size in mi units.
    /// </summary>
    private static void SymbolizeSplit(
        ArrayView<int> sym, long slotBase, ref int n,
        ArrayView<ushort> constsUshort,
        Av1FrameSeqEncodeParams p,
        int miRow, int miCol, int bsize, int halfMi)
    {
        bool hasRows = miRow + halfMi < p.FrameMiRows;
        bool hasCols = miCol + halfMi < p.FrameMiCols;
        if (hasRows & hasCols)
        {
            SymbolizePartition(sym, slotBase, ref n, constsUshort, p, miRow, miCol, bsize, PartitionSplit);
        }
        else if (hasCols)
        {
            // bottom edge: SPLIT (1) or HORZ (0)
            int ctx = GetPartitionContext(p, miRow, miCol, bsize);
            Av1SymbolStreamGpu.Cdf(sym, slotBase, ref n, 1, constsUshort,
                Av1KeyframeConstantsGpu.PartitionSplitOrHorzCdfOffset + ctx * 3, 2);
        }
        else if (hasRows)
        {
            // right edge: SPLIT (1) or VERT (0)
            int ctx = GetPartitionContext(p, miRow, miCol, bsize);
            Av1SymbolStreamGpu.Cdf(sym, slotBase, ref n, 1, constsUshort,
                Av1KeyframeConstantsGpu.PartitionSplitOrVertCdfOffset + ctx * 3, 2);
        }
    }

    /// <summary>A partition CDF symbol.</summary>
    private static void SymbolizePartition(
        ArrayView<int> sym, long slotBase, ref int n,
        ArrayView<ushort> constsUshort,
        Av1FrameSeqEncodeParams p,
        int miRow, int miCol, int bsize, int partition)
    {
        int ctx = GetPartitionContext(p, miRow, miCol, bsize);
        int nsyms = PartitionCdfNsyms(bsize);
        long cdfBase = Av1KeyframeConstantsGpu.PartitionCdfOffset + ctx * 11;
        Av1SymbolStreamGpu.Cdf(sym, slotBase, ref n, partition, constsUshort, cdfBase, nsyms);
    }

    // ===========================================================================
    // Symbol pass: leaf block (one BLOCK_16X16, or one BLOCK_8X8 at the picture edge)
    // ===========================================================================

    /// <summary>
    /// One leaf: the mode info (the same for both sizes: CFL is allowed for an 8x8 block too, so the UV mode CDF row
    /// is the same), then the coefficients of the three planes (at <paramref name="coefBase"/>: Y, U, V).
    /// <paramref name="is16"/>: BLOCK_16X16 (Y TX_16X16, U / V TX_8X8); otherwise BLOCK_8X8 at the picture edge
    /// (Y TX_8X8, U / V TX_4X4).
    /// <para>
    /// Three <see cref="SymbolizePlane"/> call sites, one per transform size, each with the size a CONSTANT (the
    /// body folds to that size; a run-time size keeps every size's code live and is slower on every backend). The
    /// 8x8 site serves the chroma of a 16x16 leaf and the luma of an edge leaf, so the edge leaves add only the 4x4
    /// site to the shader.
    /// </para>
    /// </summary>
    private static void SymbolizeLeaf(
        ArrayView<int> sym, long slotBase, ref int n,
        ArrayView<byte> constsByte, ArrayView<ushort> constsUshort,
        ArrayView<int> coefs, int coefBase,
        ArrayView<byte> levels, int sbCols,
        int miRow, int miCol, bool is16, Av1FrameSeqEncodeParams p)
    {
        // ---- Mode info: skip = 0, Y mode DC_PRED, UV mode DC_PRED. Every leaf of this encoder is DC and not
        // skipped, so the neighbours' skip flags (0) and Y modes (DC) are the same everywhere: constant contexts. ----
        Av1SymbolStreamGpu.Cdf(sym, slotBase, ref n, 0, constsUshort, Av1KeyframeConstantsGpu.SkipTxfmCdfOffset, 2);
        int dcCtx = constsByte[Av1KeyframeConstantsGpu.IntraModeContextOffset];
        long yModeCdfBase = Av1KeyframeConstantsGpu.KfYModeCdfOffset + (dcCtx * 5 + dcCtx) * 14;
        Av1SymbolStreamGpu.Cdf(sym, slotBase, ref n, 0, constsUshort, yModeCdfBase, 13);
        // UV mode: cflAllowed = 1 (16x16 and 8x8 blocks) -> 14 symbols.
        Av1SymbolStreamGpu.Cdf(sym, slotBase, ref n, 0, constsUshort, Av1KeyframeConstantsGpu.UvModeCdfV1RowOffset, 14);

        int miSize = is16 ? 4 : 2;                 // leaf size in mi; every transform of the leaf spans it

        // ---- TX_16X16: luma of a 16x16 leaf ----
        if (is16)
            SymbolizePlane(sym, slotBase, ref n,
                constsByte, constsUshort,
                coefs, coefBase,
                levels, sbCols,
                miRow, miCol, plane: 0,
                txSizeIdx: 2,
                txWMi: miSize, txHMi: miSize,
                p);

        // ---- TX_8X8: U then V of a 16x16 leaf; luma of an 8x8 leaf ----
        int first8 = is16 ? 1 : 0;
        int last8 = is16 ? 2 : 0;
        for (int plane = first8; plane <= last8; plane++)
        {
            SymbolizePlane(sym, slotBase, ref n,
                constsByte, constsUshort,
                coefs, plane == 0 ? coefBase : coefBase + 256 + (plane - 1) * 64,
                levels, sbCols,
                miRow, miCol, plane,
                txSizeIdx: 1,
                txWMi: miSize, txHMi: miSize,
                p);
        }

        // ---- TX_4X4: U then V of an 8x8 leaf ----
        if (!is16)
        {
            for (int plane = 1; plane <= 2; plane++)
            {
                SymbolizePlane(sym, slotBase, ref n,
                    constsByte, constsUshort,
                    coefs, coefBase + 64 + (plane - 1) * 16,
                    levels, sbCols,
                    miRow, miCol, plane,
                    txSizeIdx: 0,
                    txWMi: miSize, txHMi: miSize,
                    p);
            }
        }
    }

    // ===========================================================================
    // Symbol pass: the coefficients of one transform block
    // ===========================================================================

    private static void SymbolizePlane(
        ArrayView<int> sym, long slotBase, ref int n,
        ArrayView<byte> constsByte, ArrayView<ushort> constsUshort,
        ArrayView<int> coefs, int coefBase,
        ArrayView<byte> levels, int sbCols,
        int miRow, int miCol, int plane,
        int txSizeIdx,
        int txWMi, int txHMi,
        Av1FrameSeqEncodeParams p)
    {
        // txSizeIdx is a CONSTANT at every call site (see SymbolizeLeaf): 2 = TX_16X16, 1 = TX_8X8, 0 = TX_4X4
        // (chroma of an edge 8x8 block). The quantized coefficients are the transform pass's, raster order.

        // Entropy contexts, from the contexts the transform pass stored for the blocks above and to the left.
        // Per CPU contract (Av1KeyframeEncoder.EncodePlane) chroma uses luma-mi indices for ctx lookup (a chroma
        // 4x4 unit is two luma mi, both with the same value); the plane block size equals the transform size for
        // every leaf (16x16: Y TX_16X16 / UV TX_8X8; 8x8: Y TX_8X8 / UV TX_4X4), so planeBsizeIsTxsize=true and
        // planeBsizeLargerThanTxBsize=false for both Y and UV.
        int txbSkipCtx = GetTxbSkipContext(coefs, p, sbCols, plane, miRow, miCol, txWMi, txHMi);
        int dcSignCtx = GetDcSignContext(coefs, p, sbCols, plane, miRow, miCol, txWMi, txHMi);

        Av1CoefEncoderGpu.WriteCoeffsTxbSymbols(
            sym, slotBase, ref n, constsByte, constsUshort,
            coefs, coefBase,
            levels, p.LevelsOff,
            txSizeIdx, plane, qctx: GetQctx(p.BaseQIndex),
            txbSkipCtx, dcSignCtx, qindex: p.BaseQIndex);
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
    /// <summary>libaom partition_context_lookup for BLOCK_16X16 (above and left).</summary>
    private const int PartitionContext16 = 28;
    /// <summary>libaom partition_context_lookup for BLOCK_8X8 (above and left).</summary>
    private const int PartitionContext8 = 30;

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

    // ===========================================================================
    // Contexts without a serial walk
    // ===========================================================================
    //
    // The serial coder keeps "above" and "left" context arrays and updates them leaf by leaf. Every value it reads
    // from them was written by the leaf directly above (same tile; nothing above the tile's first row: zero) or
    // directly to the left (same tile; the left arrays are zeroed at the start of every superblock row of the tile,
    // so nothing left of the tile's first column: zero). So a context is a function of that neighbour:
    //   - partition: what the neighbour's leaf size wrote (libaom partition_context_lookup): 28 for a 16x16 leaf,
    //     30 for an 8x8 leaf, and a leaf is 16x16 exactly when its 16x16 position is whole inside the picture;
    //   - coefficients: the neighbour blocks' contexts (cul_level + DC sign), which the transform pass stored.
    // Skip and Y-mode contexts are constant (every leaf is DC_PRED and not skipped).

    /// <summary>True when the leaf that contains mi (<paramref name="miRow"/>, <paramref name="miCol"/>) is a 16x16 leaf (its 16x16 position is whole inside the picture).</summary>
    private static bool Is16At(Av1FrameSeqEncodeParams p, int miRow, int miCol) =>
        ((miRow & ~3) + 2 < p.FrameMiRows) & ((miCol & ~3) + 2 < p.FrameMiCols);

    /// <summary>The libaom partition_plane_context for (miRow, miCol, bsize).</summary>
    private static int GetPartitionContext(Av1FrameSeqEncodeParams p, int miRow, int miCol, int bsize)
    {
        int bsl = MiSizeWideLog2(bsize) - Block8x8Log2;
        int above = 0, left = 0;
        if (miRow > p.TileMiRowStart)
            above = ((Is16At(p, miRow - 1, miCol) ? PartitionContext16 : PartitionContext8) >> bsl) & 1;
        if (miCol > p.TileMiColStart)
            left = ((Is16At(p, miRow, miCol - 1) ? PartitionContext16 : PartitionContext8) >> bsl) & 1;
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

    /// <summary>
    /// The context (cul_level + DC sign) of the transform block of <paramref name="plane"/> in the leaf that
    /// contains mi (<paramref name="miRow"/>, <paramref name="miCol"/>) of the tile: the transform pass stored it
    /// after the coefficients of the leaf's 16x16 slot.
    /// </summary>
    private static int BlockContextAt(
        ArrayView<int> coefs, Av1FrameSeqEncodeParams p, int sbCols, int plane, int miRow, int miCol)
    {
        int row = (miRow - p.TileMiRowStart) >> 2;
        int col = (miCol - p.TileMiColStart) >> 2;
        int sb = (row >> 2) * sbCols + (col >> 2);
        int q32 = ((row >> 1) & 1) * 2 + ((col >> 1) & 1);
        int q16 = (row & 1) * 2 + (col & 1);
        int leaf = Is16At(p, miRow, miCol) ? 0 : ((miRow >> 1) & 1) * 2 + ((miCol >> 1) & 1);
        return coefs[(sb * 16 + q32 * 4 + q16) * SlotInts + CoefsPerSlot + leaf * 3 + plane];
    }

    /// <summary>The txb_skip CDF context of a block.</summary>
    private static int GetTxbSkipContext(
        ArrayView<int> coefs, Av1FrameSeqEncodeParams p, int sbCols,
        int plane, int miRow, int miCol, int txWMi, int txHMi)
    {
        if (plane == 0)
        {
            // V1 always has planeBsizeIsTxsize=true for Y -> ctx = 0.
            return 0;
        }
        // Chroma: ctx_offset 7 (planeBsizeLargerThanTxBsize=false) +
        //         (above_nz ? 1 : 0) + (left_nz ? 1 : 0).
        int aboveNz = 0, leftNz = 0;
        if (miRow > p.TileMiRowStart)
            for (int i = 0; i < txWMi; i++)
                if (miCol + i < p.FrameMiCols && BlockContextAt(coefs, p, sbCols, plane, miRow - 1, miCol + i) != 0) aboveNz = 1;
        if (miCol > p.TileMiColStart)
            for (int i = 0; i < txHMi; i++)
                if (BlockContextAt(coefs, p, sbCols, plane, miRow + i, miCol - 1) != 0) leftNz = 1;
        return aboveNz + leftNz + 7;
    }

    /// <summary>The dc_sign CDF context of a block: the DC signs of the mi units above and to the left.</summary>
    private static int GetDcSignContext(
        ArrayView<int> coefs, Av1FrameSeqEncodeParams p, int sbCols,
        int plane, int miRow, int miCol, int txWMi, int txHMi)
    {
        int dcSign = 0;
        if (miRow > p.TileMiRowStart)
            for (int i = 0; i < txWMi; i++)
                if (miCol + i < p.FrameMiCols)
                    dcSign += DcSignDelta((BlockContextAt(coefs, p, sbCols, plane, miRow - 1, miCol + i) >> 6) & 0x3); // CoeffContextBits = 6
        if (miCol > p.TileMiColStart)
            for (int i = 0; i < txHMi; i++)
                dcSign += DcSignDelta((BlockContextAt(coefs, p, sbCols, plane, miRow + i, miCol - 1) >> 6) & 0x3);
        // libaom dc_sign_contexts[65]: 32 entries of 1, then 0, then 32 entries of 2.
        if (dcSign == 0) return 0;
        return dcSign < 0 ? 1 : 2;
    }

    /// <summary>libaom signs[3] = {0, -1, 1, 0} (4 entries with 4th = 0 padding).</summary>
    private static int DcSignDelta(int sign)
    {
        if (sign == 1) return -1;
        if (sign == 2) return 1;
        return 0;
    }
}
