// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 v1 keyframe entropy coding kernel. Runs the per-MB tile-data
// entropy stage entirely GPU-resident: walks the partition tree in
// VP9's normative z-order, emits per-MB skip / Y mode / UV mode /
// per-plane coef tokens via Vp9BlockCoefEncoderGpu, and tracks the
// above + left context arrays the partition / mode / coef contexts
// read.
//
// One thread per frame for v1. The bool encoder state stays in the
// kernel; output bytes go to a pre-sized GPU buffer; nothing rounds
// back through the CPU until the final bitstream is read back.
//
// V1 simplifications (mirror Vp9KeyframeEncoder.EncodeKeyFrame):
//   - Width + height multiples of 64 (single tile, integer-SB grid -
//     no boundary partition-forcing).
//   - Profile 0, YUV 4:2:0.
//   - Every leaf is Block16x16 + PARTITION_NONE.
//   - Every leaf encodes Y mode = DC_PRED, UV mode = DC_PRED.
//   - Skip flag hardcoded to 0 (matches CPU encoder v1).
//   - tx_mode = Allow32x32 (no per-block tx_size signalling).
//   - Default coef probs (no compressed-header updates).
//
// Walk order (depth-first z-order, mirrors libvpx encode_sb):
//   for sbRow:
//     reset left arrays
//     for sbCol:
//       Block64x64 -> SPLIT
//         topLeft 32x32 -> SPLIT
//           topLeft 16x16 -> NONE leaf
//           topRight 16x16 -> NONE leaf
//           botLeft 16x16  -> NONE leaf
//           botRight 16x16 -> NONE leaf
//         topRight 32x32 -> SPLIT (4 leaves)
//         botLeft 32x32  -> SPLIT (4 leaves)
//         botRight 32x32 -> SPLIT (4 leaves)
//
// Per leaf: skip_flag bit, intra_y_mode tree (1 bit for DC_PRED),
// intra_uv_mode tree (1 bit for DC_PRED), then 3 coef-token streams
// (Y Tx16x16, U Tx8x8, V Tx8x8) via Vp9BlockCoefEncoderGpu.
//
// Constant tables for prob lookups + scan/neighbor + cat probs +
// pareto8 etc. all come pre-packed via Vp9KeyframeConstantsGpu.
// The kernel signature stays at 10 args (Index1D + 9) which fits
// comfortably under ILGPU's 15-arg Action budget.
//
// LocalMemory caps for v1: max miColsAligned = 64 (frame width up
// to 512 pixels). Beyond that the above-arrays would exceed the
// per-thread budget; the host side throws if mbCols > 32 to surface
// the limit early. Easy to lift later by widening the LocalMemory
// constants.

using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>Per-frame slot strides for the batch entropy kernel.</summary>
public struct Vp9FrameEntropyBatchStrides
{
    /// <summary>Y coefs per frame (mbCount*256).</summary>
    public int YCoefStride;
    /// <summary>UV coefs per frame (mbCount*64).</summary>
    public int UvCoefStride;
    /// <summary>tile output bytes per frame.</summary>
    public int OutBufStride;
    /// <summary>mbCols (working dim).</summary>
    public int MbCols;
    /// <summary>mbRows (working dim).</summary>
    public int MbRows;
    /// <summary>
    /// Display-dim mi cols = (FrameWidth+7)>>3. Used for boundary
    /// forced-partition handling at SBs that straddle the right edge.
    /// </summary>
    public int FrameMiCols;
    /// <summary>Display-dim mi rows = (FrameHeight+7)>>3. Bottom-edge boundary.</summary>
    public int FrameMiRows;
}

/// <summary>
/// Per-tile + frame-shape parameters for the multi-tile entropy kernel.
/// Each kernel thread is one tile (extent = TileCols × TileRows); the kernel
/// derives its tile column/row indices from <see cref="Index1D"/> and computes
/// its SB-range via the libvpx <c>get_tile_offset</c> formula inlined.
/// </summary>
public struct Vp9FrameEntropyTileStrides
{
    /// <summary>mbCols (working dim, multiple of 4 = SB-aligned).</summary>
    public int MbCols;
    /// <summary>mbRows (working dim, multiple of 4 = SB-aligned).</summary>
    public int MbRows;
    /// <summary>Display-dim mi cols = (FrameWidth+7)>>3.</summary>
    public int FrameMiCols;
    /// <summary>Display-dim mi rows = (FrameHeight+7)>>3.</summary>
    public int FrameMiRows;
    /// <summary>1 &lt;&lt; <see cref="Log2TileCols"/>.</summary>
    public int TileCols;
    /// <summary>1 &lt;&lt; <see cref="Log2TileRows"/>.</summary>
    public int TileRows;
    /// <summary>log2 of <see cref="TileCols"/>; used for tile-offset shift math.</summary>
    public int Log2TileCols;
    /// <summary>log2 of <see cref="TileRows"/>; used for tile-offset shift math.</summary>
    public int Log2TileRows;
    /// <summary>Per-tile output buffer slot stride (bytes). Worst-case per tile.</summary>
    public int OutBufStride;
}

/// <summary>
/// VP9 v1 keyframe entropy coding kernel. Single thread per frame;
/// emits the bool-coded tile bitstream from already-quantized per-
/// MB coefs.
/// </summary>
public sealed class Vp9FrameEntropyKernel : IDisposable
{
    /// <summary>
    /// V1 cap on miColsAligned (frame width / 8). At 512, frame width
    /// supports up to 4096px (covers 4K UHD 3840 wide). Throws at
    /// <see cref="Run"/> time if exceeded so the limit surfaces early.
    /// Per-thread local memory grows linearly (~12 × MaxMiColsAligned bytes
    /// = ~6KB at 512, well under CUDA's per-thread budget). WebGPU kernel
    /// compile time grows with this; if PMT's 30s budget is exceeded on
    /// WebGPU, that's a SpawnDev.ILGPU codegen perf issue to fix at the
    /// transpiler level, not a Codecs library workaround.
    /// </summary>
    public const int MaxMiColsAligned = 512;

    private readonly Accelerator _accelerator;
    private readonly Action<
        Index1D,
        ArrayView<short>, ArrayView<short>, ArrayView<short>,
        ArrayView<byte>, ArrayView<long>,
        ArrayView<byte>, ArrayView<ushort>,
        int, int, int> _kernel;

    private readonly Action<
        Index1D,
        ArrayView<short>, ArrayView<short>, ArrayView<short>,
        ArrayView<byte>, ArrayView<long>,
        ArrayView<byte>, ArrayView<ushort>,
        Vp9FrameEntropyBatchStrides> _batchKernel;

    private readonly Action<
        Index1D,
        ArrayView<short>, ArrayView<short>, ArrayView<short>,
        ArrayView<byte>, ArrayView<long>,
        ArrayView<byte>, ArrayView<ushort>,
        Vp9FrameEntropyTileStrides> _multiTileKernel;

    /// <summary>Compile.</summary>
    public Vp9FrameEntropyKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<short>, ArrayView<short>, ArrayView<short>,
            ArrayView<byte>, ArrayView<long>,
            ArrayView<byte>, ArrayView<ushort>,
            int, int, int>(EncodeFrameKernel);
        _batchKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<short>, ArrayView<short>, ArrayView<short>,
            ArrayView<byte>, ArrayView<long>,
            ArrayView<byte>, ArrayView<ushort>,
            Vp9FrameEntropyBatchStrides>(BatchEncodeFrameKernel);
        _multiTileKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<short>, ArrayView<short>, ArrayView<short>,
            ArrayView<byte>, ArrayView<long>,
            ArrayView<byte>, ArrayView<ushort>,
            Vp9FrameEntropyTileStrides>(EncodeMultiTileKernel);
    }

    /// <summary>
    /// Run the entropy kernel.
    /// </summary>
    /// <param name="yCoefs">Per-MB Y quantized coefs (mbCount * 256 shorts).</param>
    /// <param name="uCoefs">Per-MB U quantized coefs (mbCount * 64 shorts).</param>
    /// <param name="vCoefs">Per-MB V quantized coefs (mbCount * 64 shorts).</param>
    /// <param name="outBuf">Tile bytes output (worst-case sized).</param>
    /// <param name="outLen">1 long: actual byte count written.</param>
    /// <param name="byteConsts"><see cref="Vp9KeyframeConstantsGpu.BuildByteConstsBuffer"/> output.</param>
    /// <param name="ushortConsts"><see cref="Vp9KeyframeConstantsGpu.BuildUshortConstsBuffer"/> output.</param>
    /// <param name="mbCols">Macroblock columns. Must be a multiple of 4 (SB-aligned) and <see cref="MaxMiColsAligned"/>/2.</param>
    /// <param name="mbRows">Macroblock rows. Must be a multiple of 4 (SB-aligned).</param>
    public void Run(
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> outBuf, ArrayView<long> outLen,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        int mbCols, int mbRows)
    {
        // Single-arg form: assumes display dims == working dims (no boundary).
        Run(yCoefs, uCoefs, vCoefs, outBuf, outLen, byteConsts, ushortConsts,
            mbCols, mbRows, mbCols * 2, mbRows * 2);
    }

    /// <summary>
    /// Run with explicit display mi dims for spec-compliant boundary
    /// forced-partition handling at SBs that straddle the right/bottom edge.
    /// <paramref name="frameMiCols"/> = (DisplayWidth+7)>>3, similarly rows.
    /// </summary>
    public void Run(
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> outBuf, ArrayView<long> outLen,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        int mbCols, int mbRows,
        int frameMiCols, int frameMiRows)
    {
        if (mbCols <= 0 || (mbCols & 3) != 0)
            throw new ArgumentOutOfRangeException(nameof(mbCols), "mbCols must be a positive multiple of 4 (SB-aligned).");
        if (mbRows <= 0 || (mbRows & 3) != 0)
            throw new ArgumentOutOfRangeException(nameof(mbRows), "mbRows must be a positive multiple of 4 (SB-aligned).");
        if (mbCols * 2 > MaxMiColsAligned)
            throw new ArgumentOutOfRangeException(nameof(mbCols),
                $"v1 entropy kernel caps mbCols at {MaxMiColsAligned / 2}; got {mbCols}. Lift MaxMiColsAligned to grow.");
        if (frameMiCols <= 0 || frameMiRows <= 0
            || frameMiCols > mbCols * 2 || frameMiRows > mbRows * 2)
            throw new ArgumentOutOfRangeException(nameof(frameMiCols),
                "Display mi dims must be in (0, working*2].");
        if (outLen.Length < 1)
            throw new ArgumentException("outLen must hold 1 long.", nameof(outLen));

        // Pack display mi dims into one int: low 16 = miCols, high 16 = miRows.
        int frameMi = (frameMiCols & 0xFFFF) | (frameMiRows << 16);
        _kernel(1,
            yCoefs, uCoefs, vCoefs,
            outBuf, outLen,
            byteConsts, ushortConsts,
            mbCols, mbRows, frameMi);
    }

    /// <summary>
    /// Multi-tile dispatch: extent = TileCols × TileRows. Each thread is one
    /// tile in the same frame, written to its own output buffer slot. Caller
    /// supplies tile-config + per-tile output stride; this method does NOT
    /// emit tile-size-byte prefixes (that's the assembler's job, per VP9
    /// spec sec 6.4 - 4-byte big-endian length prefix before every tile
    /// except the last).
    /// </summary>
    /// <param name="outBuf">Per-tile output bytes, contiguous slots of <c>strides.OutBufStride</c> bytes each (total length = TileCols*TileRows*OutBufStride).</param>
    /// <param name="outLen">Per-tile output lengths; length = TileCols*TileRows.</param>
    /// <param name="strides">Tile config (TileCols/TileRows, log2 + dims, OutBufStride).</param>
    public void RunMultiTile(
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> outBuf, ArrayView<long> outLen,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        Vp9FrameEntropyTileStrides strides)
    {
        if (strides.MbCols <= 0 || (strides.MbCols & 3) != 0)
            throw new ArgumentOutOfRangeException(nameof(strides),
                $"strides.MbCols must be a positive multiple of 4 (SB-aligned); got {strides.MbCols}.");
        if (strides.MbRows <= 0 || (strides.MbRows & 3) != 0)
            throw new ArgumentOutOfRangeException(nameof(strides),
                $"strides.MbRows must be a positive multiple of 4 (SB-aligned); got {strides.MbRows}.");
        if (strides.MbCols * 2 > MaxMiColsAligned)
            throw new ArgumentOutOfRangeException(nameof(strides),
                $"v1 entropy kernel caps mbCols at {MaxMiColsAligned / 2}; got {strides.MbCols}. Lift MaxMiColsAligned to grow.");
        if (strides.TileCols <= 0 || (strides.TileCols & (strides.TileCols - 1)) != 0)
            throw new ArgumentOutOfRangeException(nameof(strides),
                $"strides.TileCols must be a positive power of 2; got {strides.TileCols}.");
        if (strides.TileRows <= 0 || (strides.TileRows & (strides.TileRows - 1)) != 0)
            throw new ArgumentOutOfRangeException(nameof(strides),
                $"strides.TileRows must be a positive power of 2; got {strides.TileRows}.");
        if ((1 << strides.Log2TileCols) != strides.TileCols
            || (1 << strides.Log2TileRows) != strides.TileRows)
            throw new ArgumentException(
                "strides.Log2TileCols / Log2TileRows must satisfy 1 << log2 == count.", nameof(strides));
        if (strides.OutBufStride <= 0)
            throw new ArgumentOutOfRangeException(nameof(strides),
                $"strides.OutBufStride must be positive; got {strides.OutBufStride}.");

        int totalTiles = strides.TileCols * strides.TileRows;
        if (outLen.Length < totalTiles)
            throw new ArgumentException(
                $"outLen must hold at least {totalTiles} entries (one per tile).", nameof(outLen));
        if (outBuf.Length < (long)totalTiles * strides.OutBufStride)
            throw new ArgumentException(
                $"outBuf must hold at least {(long)totalTiles * strides.OutBufStride} bytes " +
                $"({totalTiles} tiles × {strides.OutBufStride} stride).", nameof(outBuf));

        _multiTileKernel(totalTiles,
            yCoefs, uCoefs, vCoefs,
            outBuf, outLen,
            byteConsts, ushortConsts,
            strides);
    }

    /// <summary>Batch entropy: extent=N, each thread walks one frame's MBs.</summary>
    public void RunBatch(
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> outBuf, ArrayView<long> outLen,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        int frameCount, Vp9FrameEntropyBatchStrides strides)
    {
        if (frameCount <= 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
        _batchKernel(frameCount,
            yCoefs, uCoefs, vCoefs,
            outBuf, outLen,
            byteConsts, ushortConsts, strides);
    }

    /// <summary>Batch entropy kernel: thread = one frame's entropy walk.</summary>
    private static void BatchEncodeFrameKernel(
        Index1D idx,
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> outBuf, ArrayView<long> outLen,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        Vp9FrameEntropyBatchStrides s)
    {
        int f = idx.X;
        var fY = yCoefs.SubView((long)f * s.YCoefStride, s.YCoefStride);
        var fU = uCoefs.SubView((long)f * s.UvCoefStride, s.UvCoefStride);
        var fV = vCoefs.SubView((long)f * s.UvCoefStride, s.UvCoefStride);
        var fOut = outBuf.SubView((long)f * s.OutBufStride, s.OutBufStride);
        var fOutLen = outLen.SubView(f, 1);
        int frameMi = (s.FrameMiCols & 0xFFFF) | (s.FrameMiRows << 16);
        // Single-tile: full-frame SB range.
        EncodeFrameBody(fY, fU, fV, fOut, fOutLen, byteConsts, ushortConsts,
            sbRowStart: 0, sbRowEnd: s.MbRows >> 2,
            sbColStart: 0, sbColEnd: s.MbCols >> 2,
            mbCols: s.MbCols, frameMi: frameMi);
    }

    private static void EncodeFrameKernel(
        Index1D _,
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> outBuf, ArrayView<long> outLen,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        int mbCols, int mbRows, int frameMi)
    {
        // Single-tile: full-frame SB range.
        EncodeFrameBody(yCoefs, uCoefs, vCoefs, outBuf, outLen,
            byteConsts, ushortConsts,
            sbRowStart: 0, sbRowEnd: mbRows >> 2,
            sbColStart: 0, sbColEnd: mbCols >> 2,
            mbCols: mbCols, frameMi: frameMi);
    }

    /// <summary>
    /// Multi-tile dispatch: extent = TileCols × TileRows. Each thread is one
    /// tile in the same frame. Per-tile output written to its own slot in
    /// <paramref name="outBuf"/> (stride = <c>s.OutBufStride</c>) and per-tile
    /// length to <paramref name="outLen"/>[tileIdx]. Host-side concatenation
    /// with 4-byte big-endian length prefixes (per VP9 spec sec 6.4) happens
    /// in a follow-up assembler-kernel update.
    /// </summary>
    private static void EncodeMultiTileKernel(
        Index1D idx,
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> outBuf, ArrayView<long> outLen,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        Vp9FrameEntropyTileStrides s)
    {
        int tileIdx = idx.X;
        // tileColIdx + tileRowIdx via integer division. tile_count is power of
        // 2 in VP9, so the column wrap is `& (TileCols - 1)`.
        int tileColIdx = tileIdx & (s.TileCols - 1);
        int tileRowIdx = tileIdx >> s.Log2TileCols;

        int sbCols = s.MbCols >> 2;
        int sbRows = s.MbRows >> 2;

        // Compute this tile's [sbColStart, sbColEnd) × [sbRowStart, sbRowEnd)
        // range. libvpx get_tile_offset inlined: offset = (idx*mi_count) >>
        // log2_tile_count, mi-aligned to MI_BLOCK_SIZE (8), divided by 8.
        // The last tile in each axis extends to the frame edge regardless.
        int sbColStart = TileSbOffset(tileColIdx, s.Log2TileCols, sbCols);
        int sbColEnd = (tileColIdx + 1 == s.TileCols)
            ? sbCols
            : TileSbOffset(tileColIdx + 1, s.Log2TileCols, sbCols);
        int sbRowStart = TileSbOffset(tileRowIdx, s.Log2TileRows, sbRows);
        int sbRowEnd = (tileRowIdx + 1 == s.TileRows)
            ? sbRows
            : TileSbOffset(tileRowIdx + 1, s.Log2TileRows, sbRows);

        // Per-tile output slot. Each tile writes to its own contiguous bytes
        // region; host stitches them together with size prefixes.
        var tileOut = outBuf.SubView((long)tileIdx * s.OutBufStride, s.OutBufStride);
        var tileOutLen = outLen.SubView(tileIdx, 1);

        int frameMi = (s.FrameMiCols & 0xFFFF) | (s.FrameMiRows << 16);
        EncodeFrameBody(yCoefs, uCoefs, vCoefs, tileOut, tileOutLen,
            byteConsts, ushortConsts,
            sbRowStart, sbRowEnd,
            sbColStart, sbColEnd,
            s.MbCols, frameMi);
    }

    /// <summary>
    /// Inlined libvpx <c>get_tile_offset</c>: returns the SB64-column (or row)
    /// where tile <paramref name="idx"/> starts. Called only from
    /// <see cref="EncodeMultiTileKernel"/>; mirror of
    /// <see cref="Vp9TileInfoParser.GetTileOffsetSb"/> in kernel-safe form
    /// (no exceptions, all int math).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int TileSbOffset(int idx, int log2TileCount, int sbCount)
    {
        // mi_count = sbCount × MI_BLOCK_SIZE (= 8); offset = (idx*mi_count) >>
        // log2; SB-aligned; convert back to SB.
        int miCount = sbCount << 3;
        int miOffset = (idx * miCount) >> log2TileCount;
        int miAligned = (miOffset + 7) & ~7;
        return miAligned >> 3;
    }

    /// <summary>
    /// Encode one VP9 tile spanning SB-range [sbRowStart, sbRowEnd) ×
    /// [sbColStart, sbColEnd): own bool encoder (init -&gt; encode -&gt; stop),
    /// fresh above[] context arrays at the column boundary, fresh left[]
    /// reset at every internal sb-row.
    ///
    /// Single-tile callers pass (0, sbRows, 0, sbCols) — full-frame range.
    /// Multi-tile callers (next pass) compute per-tile ranges via
    /// <see cref="Vp9TileInfoParser.GetTileColRange"/> +
    /// <see cref="Vp9TileInfoParser.GetTileRowRange"/> and dispatch one
    /// thread per tile, each writing to its own per-tile output slot.
    ///
    /// Per VP9 spec sec 6.5: every tile carries an independent bool encoder.
    /// Above[] arrays reset at the column boundary (this function's entry);
    /// left[] arrays reset at every sb-row boundary inside the tile (the
    /// inner loop). Tile bytes get a 4-byte big-endian length prefix
    /// concatenated by the assembler (except the last tile per VP9 spec) -
    /// for single-tile mode no prefix is emitted.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void EncodeFrameBody(
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> outBuf, ArrayView<long> outLen,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        int sbRowStart, int sbRowEnd,
        int sbColStart, int sbColEnd,
        int mbCols, int frameMi)
    {
        // Per-thread context arrays. Sizes matter: above arrays are
        // capped at MaxMiColsAligned * 2 (= 128) for b4-cell granular
        // ones (Y entropy + Y mode), MaxMiColsAligned (= 64) for
        // mi-granular ones. Chroma above arrays are subsampled by
        // 2 in the X dimension.
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

        var tokenCache = LocalMemory.Allocate<byte>(256);

        // Init above arrays. Mode = DcPred (0). Everything else 0.
        for (int i = 0; i < MaxMiColsAligned * 2; i++) aboveYMode[i] = 0;
        for (int i = 0; i < MaxMiColsAligned; i++) { aboveSkip[i] = 0; abovePartCtx[i] = 0; aboveTxSize[i] = 0; }
        for (int i = 0; i < MaxMiColsAligned * 2; i++) aboveYEntropyCtx[i] = 0;
        for (int i = 0; i < MaxMiColsAligned; i++) { aboveUEntropyCtx[i] = 0; aboveVEntropyCtx[i] = 0; }

        // Initialize bool encoder + emit VP9 marker bit. Per-tile -
        // every tile (single-tile = the whole frame) has its own bool
        // encoder state.
        var state = Vp8BoolEncoderGpu.Init();
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, 128);

        // Walk SBs row-major within this tile's SB-range.
        // mbCols/mbRows on the FRAME are multiples of 4 by contract so the
        // SB grid is exact; per-tile sub-ranges inherit that property
        // because tile boundaries fall on SB-multiples per VP9 spec sec 6.5.
        for (int sbRow = sbRowStart; sbRow < sbRowEnd; sbRow++)
        {
            // Reset left arrays at the start of each SB row within the tile.
            for (int i = 0; i < 16; i++) { leftYMode[i] = 0; leftYEntropyCtx[i] = 0; }
            for (int i = 0; i < 8; i++) { leftSkip[i] = 0; leftPartCtx[i] = 0; leftTxSize[i] = 0; leftUEntropyCtx[i] = 0; leftVEntropyCtx[i] = 0; }

            for (int sbCol = sbColStart; sbCol < sbColEnd; sbCol++)
            {
                EncodeSb64(
                    ref state, outBuf,
                    yCoefs, uCoefs, vCoefs,
                    byteConsts, ushortConsts,
                    aboveYMode, aboveSkip, abovePartCtx, aboveTxSize,
                    aboveYEntropyCtx, aboveUEntropyCtx, aboveVEntropyCtx,
                    leftYMode, leftSkip, leftPartCtx, leftTxSize,
                    leftYEntropyCtx, leftUEntropyCtx, leftVEntropyCtx,
                    tokenCache,
                    sbRow, sbCol, mbCols, frameMi);
            }
        }

        Vp8BoolEncoderGpu.Stop(ref state, outBuf);
        outLen[0] = state.OutLen;
    }

    private static void EncodeSb64(
        ref Vp8BoolEncoderGpuState state, ArrayView<byte> outBuf,
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        ArrayView<byte> aboveYMode, ArrayView<byte> aboveSkip,
        ArrayView<byte> abovePartCtx, ArrayView<byte> aboveTxSize,
        ArrayView<byte> aboveYEntropyCtx, ArrayView<byte> aboveUEntropyCtx, ArrayView<byte> aboveVEntropyCtx,
        ArrayView<byte> leftYMode, ArrayView<byte> leftSkip,
        ArrayView<byte> leftPartCtx, ArrayView<byte> leftTxSize,
        ArrayView<byte> leftYEntropyCtx, ArrayView<byte> leftUEntropyCtx, ArrayView<byte> leftVEntropyCtx,
        ArrayView<byte> tokenCache,
        int sbRow, int sbCol, int mbCols, int frameMi)
    {
        int frameMiCols = frameMi & 0xFFFF;
        int frameMiRows = frameMi >> 16;
        int miRow64 = sbRow * 8;
        int miCol64 = sbCol * 8;

        // SB64 boundary check: bsl=3, half=4 mi.
        bool hasRows64 = (miRow64 + 4) < frameMiRows;
        bool hasCols64 = (miCol64 + 4) < frameMiCols;

        if (hasRows64 && hasCols64)
        {
            // Full 4-way CDF, emit SPLIT.
            EmitPartitionSplit(ref state, outBuf, byteConsts,
                sizeIdx: 3, bsl: 3, miRow: miRow64, miCol: miCol64,
                abovePartCtx, leftPartCtx);
        }
        else if (hasCols64)
        {
            // Bottom edge: 1-bit at probs[1] = SPLIT (1) vs HORZ (0).
            EmitPartitionBoundary1Bit(ref state, outBuf, byteConsts,
                sizeIdx: 3, bsl: 3, miRow: miRow64, miCol: miCol64,
                abovePartCtx, leftPartCtx, probIdx: 1, value: 1);
        }
        else if (hasRows64)
        {
            // Right edge: 1-bit at probs[2] = SPLIT (1) vs VERT (0).
            EmitPartitionBoundary1Bit(ref state, outBuf, byteConsts,
                sizeIdx: 3, bsl: 3, miRow: miRow64, miCol: miCol64,
                abovePartCtx, leftPartCtx, probIdx: 2, value: 1);
        }
        // else corner: forced split, no emit.

        // Walk 4 children at 32x32 in z-order: TL, TR, BL, BR.
        // Skip out-of-frame children (top-left at or past frame boundary).
        for (int q32 = 0; q32 < 4; q32++)
        {
            int miRow32 = miRow64 + ((q32 & 2) >> 1) * 4;
            int miCol32 = miCol64 + (q32 & 1) * 4;
            if (miRow32 >= frameMiRows || miCol32 >= frameMiCols) continue;
            EncodeBlock32x32(
                ref state, outBuf,
                yCoefs, uCoefs, vCoefs,
                byteConsts, ushortConsts,
                aboveYMode, aboveSkip, abovePartCtx, aboveTxSize,
                aboveYEntropyCtx, aboveUEntropyCtx, aboveVEntropyCtx,
                leftYMode, leftSkip, leftPartCtx, leftTxSize,
                leftYEntropyCtx, leftUEntropyCtx, leftVEntropyCtx,
                tokenCache,
                miRow32, miCol32, mbCols, frameMi);
        }
    }

    private static void EncodeBlock32x32(
        ref Vp8BoolEncoderGpuState state, ArrayView<byte> outBuf,
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        ArrayView<byte> aboveYMode, ArrayView<byte> aboveSkip,
        ArrayView<byte> abovePartCtx, ArrayView<byte> aboveTxSize,
        ArrayView<byte> aboveYEntropyCtx, ArrayView<byte> aboveUEntropyCtx, ArrayView<byte> aboveVEntropyCtx,
        ArrayView<byte> leftYMode, ArrayView<byte> leftSkip,
        ArrayView<byte> leftPartCtx, ArrayView<byte> leftTxSize,
        ArrayView<byte> leftYEntropyCtx, ArrayView<byte> leftUEntropyCtx, ArrayView<byte> leftVEntropyCtx,
        ArrayView<byte> tokenCache,
        int miRow32, int miCol32, int mbCols, int frameMi)
    {
        int frameMiCols = frameMi & 0xFFFF;
        int frameMiRows = frameMi >> 16;

        // SB32 boundary check: bsl=2, half=2 mi.
        bool hasRows32 = (miRow32 + 2) < frameMiRows;
        bool hasCols32 = (miCol32 + 2) < frameMiCols;

        if (hasRows32 && hasCols32)
        {
            EmitPartitionSplit(ref state, outBuf, byteConsts,
                sizeIdx: 2, bsl: 2, miRow: miRow32, miCol: miCol32,
                abovePartCtx, leftPartCtx);
        }
        else if (hasCols32)
        {
            EmitPartitionBoundary1Bit(ref state, outBuf, byteConsts,
                sizeIdx: 2, bsl: 2, miRow: miRow32, miCol: miCol32,
                abovePartCtx, leftPartCtx, probIdx: 1, value: 1);
        }
        else if (hasRows32)
        {
            EmitPartitionBoundary1Bit(ref state, outBuf, byteConsts,
                sizeIdx: 2, bsl: 2, miRow: miRow32, miCol: miCol32,
                abovePartCtx, leftPartCtx, probIdx: 2, value: 1);
        }
        // else corner: forced split, no emit.

        // Walk 4 children at 16x16 in z-order, skipping out-of-frame children.
        for (int q16 = 0; q16 < 4; q16++)
        {
            int miRow16 = miRow32 + ((q16 & 2) >> 1) * 2;
            int miCol16 = miCol32 + (q16 & 1) * 2;
            if (miRow16 >= frameMiRows || miCol16 >= frameMiCols) continue;
            EncodeBlock16x16(
                ref state, outBuf,
                yCoefs, uCoefs, vCoefs,
                byteConsts, ushortConsts,
                aboveYMode, aboveSkip, abovePartCtx, aboveTxSize,
                aboveYEntropyCtx, aboveUEntropyCtx, aboveVEntropyCtx,
                leftYMode, leftSkip, leftPartCtx, leftTxSize,
                leftYEntropyCtx, leftUEntropyCtx, leftVEntropyCtx,
                tokenCache,
                miRow16, miCol16, mbCols, frameMi);
        }
    }

    private static void EncodeBlock16x16(
        ref Vp8BoolEncoderGpuState state, ArrayView<byte> outBuf,
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        ArrayView<byte> aboveYMode, ArrayView<byte> aboveSkip,
        ArrayView<byte> abovePartCtx, ArrayView<byte> aboveTxSize,
        ArrayView<byte> aboveYEntropyCtx, ArrayView<byte> aboveUEntropyCtx, ArrayView<byte> aboveVEntropyCtx,
        ArrayView<byte> leftYMode, ArrayView<byte> leftSkip,
        ArrayView<byte> leftPartCtx, ArrayView<byte> leftTxSize,
        ArrayView<byte> leftYEntropyCtx, ArrayView<byte> leftUEntropyCtx, ArrayView<byte> leftVEntropyCtx,
        ArrayView<byte> tokenCache,
        int miRow16, int miCol16, int mbCols, int frameMi)
    {
        int frameMiCols = frameMi & 0xFFFF;
        int frameMiRows = frameMi >> 16;

        // SB16 boundary check: bsl=1, half=1 mi.
        bool hasRows = (miRow16 + 1) < frameMiRows;
        bool hasCols = (miCol16 + 1) < frameMiCols;

        if (hasRows && hasCols)
        {
            // Standard path: NONE + BLOCK_16X16 leaf.
            EmitPartitionNone(ref state, outBuf, byteConsts,
                sizeIdx: 1, bsl: 1, miRow: miRow16, miCol: miCol16,
                abovePartCtx, leftPartCtx);

            int mbR = miRow16 >> 1;
            int mbC = miCol16 >> 1;
            EncodeLeafBlock(
                ref state, outBuf,
                yCoefs, uCoefs, vCoefs,
                byteConsts, ushortConsts,
                aboveYMode, aboveSkip, aboveTxSize,
                aboveYEntropyCtx, aboveUEntropyCtx, aboveVEntropyCtx,
                leftYMode, leftSkip, leftTxSize,
                leftYEntropyCtx, leftUEntropyCtx, leftVEntropyCtx,
                tokenCache,
                mbR, mbC, miRow16, miCol16, mbCols);
        }
        else if (hasCols)
        {
            // Bottom-edge boundary: emit PARTITION_HORZ (1-bit value 0 at
            // probs[1]). Encode top BLOCK_16X8 leaf only; bottom sub-block
            // top-left is at miRow16+1 which is >= frameMiRows so the
            // decoder skips it per spec.
            EmitPartitionBoundary1Bit(ref state, outBuf, byteConsts,
                sizeIdx: 1, bsl: 1, miRow: miRow16, miCol: miCol16,
                abovePartCtx, leftPartCtx, probIdx: 1, value: 0);

            int mbR = miRow16 >> 1;
            int mbC = miCol16 >> 1;
            EncodeLeafBlock16x8(
                ref state, outBuf,
                yCoefs, uCoefs, vCoefs,
                byteConsts, ushortConsts,
                aboveYMode, aboveSkip, aboveTxSize,
                aboveYEntropyCtx, aboveUEntropyCtx, aboveVEntropyCtx,
                leftYMode, leftSkip, leftTxSize,
                leftYEntropyCtx, leftUEntropyCtx, leftVEntropyCtx,
                tokenCache,
                mbR, mbC, miRow16, miCol16, mbCols);
        }
        // else hasRows-only or corner: not implemented in v1 (1920x1080
        // doesn't hit these cases). Tracked as task #23 for full 4K + arbitrary
        // dim support.

        // UpdatePartitionContext: same for 16x16+NONE and 16x16+HORZ in our path.
        for (int i = 0; i < 2; i++)
        {
            int c = miCol16 + i;
            int r = (miRow16 + i) & 7;
            if (c < MaxMiColsAligned) abovePartCtx[c] = 12;
            leftPartCtx[r] = 12;
        }
    }

    private static void EmitPartitionSplit(
        ref Vp8BoolEncoderGpuState state, ArrayView<byte> outBuf,
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
        // SPLIT walks the partition tree as bits 1, 1, 1.
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, byteConsts[probsBase + 0]);
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, byteConsts[probsBase + 1]);
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 1, byteConsts[probsBase + 2]);
    }

    /// <summary>
    /// Emit the 1-bit partition decision used at SB-grid boundary edges.
    /// At the bottom-only boundary (!hasRows &amp;&amp; hasCols) the decoder reads
    /// probs[1] selecting between PARTITION_HORZ (bit 0) and PARTITION_SPLIT (bit 1).
    /// At the right-only boundary (hasRows &amp;&amp; !hasCols), probs[2] selects
    /// between PARTITION_VERT (0) and PARTITION_SPLIT (1).
    /// At the corner (!hasRows &amp;&amp; !hasCols) no bits are read; partition is
    /// implicitly SPLIT - this helper isn't called for that case.
    /// </summary>
    private static void EmitPartitionBoundary1Bit(
        ref Vp8BoolEncoderGpuState state, ArrayView<byte> outBuf,
        ArrayView<byte> byteConsts,
        int sizeIdx, int bsl, int miRow, int miCol,
        ArrayView<byte> abovePartCtx, ArrayView<byte> leftPartCtx,
        int probIdx, int value)
    {
        int leftIdx = miRow & 7;
        int aboveIdx = miCol;
        int leftBit = (leftPartCtx[leftIdx] >> bsl) & 1;
        int aboveBit = (abovePartCtx[aboveIdx] >> bsl) & 1;
        int splitState = leftBit * 2 + aboveBit;
        long probsBase = Vp9KeyframeConstantsGpu.KfPartitionProbsOffset
                       + ((long)sizeIdx * 4 + splitState) * 3;
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, value, byteConsts[probsBase + probIdx]);
    }

    private static void EmitPartitionNone(
        ref Vp8BoolEncoderGpuState state, ArrayView<byte> outBuf,
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
        // NONE walks the partition tree as bit 0 at probs[0].
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, byteConsts[probsBase + 0]);
    }

    private static void EncodeLeafBlock(
        ref Vp8BoolEncoderGpuState state, ArrayView<byte> outBuf,
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        ArrayView<byte> aboveYMode, ArrayView<byte> aboveSkip, ArrayView<byte> aboveTxSize,
        ArrayView<byte> aboveYEntropyCtx, ArrayView<byte> aboveUEntropyCtx, ArrayView<byte> aboveVEntropyCtx,
        ArrayView<byte> leftYMode, ArrayView<byte> leftSkip, ArrayView<byte> leftTxSize,
        ArrayView<byte> leftYEntropyCtx, ArrayView<byte> leftUEntropyCtx, ArrayView<byte> leftVEntropyCtx,
        ArrayView<byte> tokenCache,
        int mbR, int mbC, int miRow, int miCol, int mbCols)
    {
        // ---- Skip flag ----
        // skip_context = above + left (each is 0 or 1).
        int leftIdxMi = miRow & 7;
        int leftSkipBit = leftSkip[leftIdxMi];
        int aboveSkipBit = miCol < MaxMiColsAligned ? aboveSkip[miCol] : 0;
        int skipContext = aboveSkipBit + leftSkipBit;
        int skipFlag = 0; // v1: hardcoded 0.
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, skipFlag,
            byteConsts[Vp9KeyframeConstantsGpu.SkipProbsOffset + skipContext]);

        // ---- Y mode (DC_PRED, value 0) ----
        // Probs row indexed by (above_y_mode, left_y_mode). For DC_PRED
        // the tree's first decision is at probs[0]; bit 0 means leaf
        // Vp9IntraMode.DcPred. Net cost: 1 bit per leaf for v1.
        int b4Col = miCol * 2;
        int leftB4Idx = (miRow & 7) * 2;
        int aboveYCell = b4Col < MaxMiColsAligned * 2 ? aboveYMode[b4Col] : 0;
        int leftYCell = leftYMode[leftB4Idx];
        long yProbBase = Vp9KeyframeConstantsGpu.KfYModeProbsOffset
                       + (long)(aboveYCell * 10 + leftYCell) * 9;
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, byteConsts[yProbBase + 0]);

        // ---- UV mode (DC_PRED) ----
        // Probs row indexed by Y mode. yMode = DcPred = 0.
        long uvProbBase = Vp9KeyframeConstantsGpu.KfUvModeProbsOffset; // + 0 * 9
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, byteConsts[uvProbBase + 0]);

        // ---- Update mode-info contexts ----
        // For Block16x16 (b4Wide = b4High = 4, miWide = miHigh = 2).
        for (int i = 0; i < 4; i++)
        {
            int c = b4Col + i;
            if (c < MaxMiColsAligned * 2) aboveYMode[c] = 0; // DcPred
        }
        for (int i = 0; i < 4; i++)
        {
            int r = (leftB4Idx + i) & 15;
            leftYMode[r] = 0; // DcPred
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

        // ---- Per-plane coef emission ----
        long mbIdx = (long)mbR * mbCols + mbC;

        // Y plane: Tx16x16 at this 16x16 block. yPx = mbR*16, xPx = mbC*16.
        // cellsPerTx = 16/4 = 4. aboveCellOff = xPx >> 2 = mbC*4.
        // leftCellOff = (yPx & 63) >> 2 = (mbR & 3) * 4.
        int yAboveCell = mbC * 4;
        int yLeftCell = (mbR & 3) * 4;
        EncodePlaneCoefs(
            ref state, outBuf,
            yCoefs, mbIdx * 256, 256,
            byteConsts, ushortConsts,
            yAboveCell, yLeftCell, cellsPerTx: 4,
            isTx4x4: 0, txSizeForCoefProbs: (int)Vp9TxSize.Tx16x16,
            planeType: (int)Vp9BlockCoefEnums.PlaneType.Y,
            tokenCache, aboveYEntropyCtx, leftYEntropyCtx);

        // U plane: Tx8x8 at mbR*8, mbC*8 in chroma plane.
        // cellsPerTx = 8/4 = 2. aboveCellOff = (mbC*8) >> 2 = mbC*2.
        // leftCellOff = ((mbR*8) & 31) >> 2 = (mbR & 3) * 2.
        int uvAboveCell = mbC * 2;
        int uvLeftCell = (mbR & 3) * 2;
        EncodePlaneCoefs(
            ref state, outBuf,
            uCoefs, mbIdx * 64, 64,
            byteConsts, ushortConsts,
            uvAboveCell, uvLeftCell, cellsPerTx: 2,
            isTx4x4: 0, txSizeForCoefProbs: (int)Vp9TxSize.Tx8x8,
            planeType: (int)Vp9BlockCoefEnums.PlaneType.Uv,
            tokenCache, aboveUEntropyCtx, leftUEntropyCtx);

        EncodePlaneCoefs(
            ref state, outBuf,
            vCoefs, mbIdx * 64, 64,
            byteConsts, ushortConsts,
            uvAboveCell, uvLeftCell, cellsPerTx: 2,
            isTx4x4: 0, txSizeForCoefProbs: (int)Vp9TxSize.Tx8x8,
            planeType: (int)Vp9BlockCoefEnums.PlaneType.Uv,
            tokenCache, aboveVEntropyCtx, leftVEntropyCtx);
    }

    /// <summary>
    /// Encode a top-half BLOCK_16X8 leaf at the bottom-edge boundary. Same
    /// shape as <see cref="EncodeLeafBlock"/> but with 2 Tx8x8 luma + 2 Tx4x4
    /// U + 2 Tx4x4 V transforms (per VP9 spec for BLOCK_16X8 with TX_MODE=
    /// ALLOW_32X32). Mode-info contexts updated for a 2-mi-wide x 1-mi-tall
    /// region (vs 2x2 for BLOCK_16X16).
    /// </summary>
    private static void EncodeLeafBlock16x8(
        ref Vp8BoolEncoderGpuState state, ArrayView<byte> outBuf,
        ArrayView<short> yCoefs, ArrayView<short> uCoefs, ArrayView<short> vCoefs,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        ArrayView<byte> aboveYMode, ArrayView<byte> aboveSkip, ArrayView<byte> aboveTxSize,
        ArrayView<byte> aboveYEntropyCtx, ArrayView<byte> aboveUEntropyCtx, ArrayView<byte> aboveVEntropyCtx,
        ArrayView<byte> leftYMode, ArrayView<byte> leftSkip, ArrayView<byte> leftTxSize,
        ArrayView<byte> leftYEntropyCtx, ArrayView<byte> leftUEntropyCtx, ArrayView<byte> leftVEntropyCtx,
        ArrayView<byte> tokenCache,
        int mbR, int mbC, int miRow, int miCol, int mbCols)
    {
        // Skip flag, Y mode, UV mode emit identical to BLOCK_16X16 (DC_PRED,
        // skip=0 in v1).
        int leftIdxMi = miRow & 7;
        int leftSkipBit = leftSkip[leftIdxMi];
        int aboveSkipBit = miCol < MaxMiColsAligned ? aboveSkip[miCol] : 0;
        int skipContext = aboveSkipBit + leftSkipBit;
        int skipFlag = 0;
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, skipFlag,
            byteConsts[Vp9KeyframeConstantsGpu.SkipProbsOffset + skipContext]);

        int b4Col = miCol * 2;
        int leftB4Idx = (miRow & 7) * 2;
        int aboveYCell = b4Col < MaxMiColsAligned * 2 ? aboveYMode[b4Col] : 0;
        int leftYCell = leftYMode[leftB4Idx];
        long yProbBase = Vp9KeyframeConstantsGpu.KfYModeProbsOffset
                       + (long)(aboveYCell * 10 + leftYCell) * 9;
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, byteConsts[yProbBase + 0]);

        long uvProbBase = Vp9KeyframeConstantsGpu.KfUvModeProbsOffset;
        Vp8BoolEncoderGpu.EncodeBool(ref state, outBuf, 0, byteConsts[uvProbBase + 0]);

        // Mode-info context updates for BLOCK_16X8: b4Wide=4, b4High=2,
        // miWide=2, miHigh=1.
        for (int i = 0; i < 4; i++)
        {
            int c = b4Col + i;
            if (c < MaxMiColsAligned * 2) aboveYMode[c] = 0;
        }
        for (int i = 0; i < 2; i++)
        {
            int r = (leftB4Idx + i) & 15;
            leftYMode[r] = 0;
        }
        for (int i = 0; i < 2; i++)
        {
            int c = miCol + i;
            if (c < MaxMiColsAligned) { aboveSkip[c] = (byte)skipFlag; aboveTxSize[c] = (byte)Vp9TxSize.Tx8x8; }
        }
        // miHigh=1: only one left row updated.
        leftSkip[leftIdxMi] = (byte)skipFlag;
        leftTxSize[leftIdxMi] = (byte)Vp9TxSize.Tx8x8;

        long mbIdx = (long)mbR * mbCols + mbC;

        // Y plane: 2 Tx8x8 over the top 16x8 (left 8x8 + right 8x8).
        // Sequential encode stores them at yCoefs[mbIdx*256 + 0..63] and [+64..127].
        // cellsPerTx = 8/4 = 2.
        int yAboveCell = mbC * 4;
        int yLeftCell = (mbR & 3) * 4;
        EncodePlaneCoefs(
            ref state, outBuf,
            yCoefs, mbIdx * 256 + 0, 64,
            byteConsts, ushortConsts,
            yAboveCell, yLeftCell, cellsPerTx: 2,
            isTx4x4: 0, txSizeForCoefProbs: (int)Vp9TxSize.Tx8x8,
            planeType: (int)Vp9BlockCoefEnums.PlaneType.Y,
            tokenCache, aboveYEntropyCtx, leftYEntropyCtx);
        EncodePlaneCoefs(
            ref state, outBuf,
            yCoefs, mbIdx * 256 + 64, 64,
            byteConsts, ushortConsts,
            yAboveCell + 2, yLeftCell, cellsPerTx: 2,
            isTx4x4: 0, txSizeForCoefProbs: (int)Vp9TxSize.Tx8x8,
            planeType: (int)Vp9BlockCoefEnums.PlaneType.Y,
            tokenCache, aboveYEntropyCtx, leftYEntropyCtx);

        // U plane: 2 Tx4x4 over top 8x4 chroma (left 4x4 + right 4x4).
        // Stored at uCoefs[mbIdx*64 + 0..15] and [+16..31].
        // cellsPerTx = 4/4 = 1.
        int uvAboveCell = mbC * 2;
        int uvLeftCell = (mbR & 3) * 2;
        EncodePlaneCoefs(
            ref state, outBuf,
            uCoefs, mbIdx * 64 + 0, 16,
            byteConsts, ushortConsts,
            uvAboveCell, uvLeftCell, cellsPerTx: 1,
            isTx4x4: 1, txSizeForCoefProbs: (int)Vp9TxSize.Tx4x4,
            planeType: (int)Vp9BlockCoefEnums.PlaneType.Uv,
            tokenCache, aboveUEntropyCtx, leftUEntropyCtx);
        EncodePlaneCoefs(
            ref state, outBuf,
            uCoefs, mbIdx * 64 + 16, 16,
            byteConsts, ushortConsts,
            uvAboveCell + 1, uvLeftCell, cellsPerTx: 1,
            isTx4x4: 1, txSizeForCoefProbs: (int)Vp9TxSize.Tx4x4,
            planeType: (int)Vp9BlockCoefEnums.PlaneType.Uv,
            tokenCache, aboveUEntropyCtx, leftUEntropyCtx);

        // V plane: same as U.
        EncodePlaneCoefs(
            ref state, outBuf,
            vCoefs, mbIdx * 64 + 0, 16,
            byteConsts, ushortConsts,
            uvAboveCell, uvLeftCell, cellsPerTx: 1,
            isTx4x4: 1, txSizeForCoefProbs: (int)Vp9TxSize.Tx4x4,
            planeType: (int)Vp9BlockCoefEnums.PlaneType.Uv,
            tokenCache, aboveVEntropyCtx, leftVEntropyCtx);
        EncodePlaneCoefs(
            ref state, outBuf,
            vCoefs, mbIdx * 64 + 16, 16,
            byteConsts, ushortConsts,
            uvAboveCell + 1, uvLeftCell, cellsPerTx: 1,
            isTx4x4: 1, txSizeForCoefProbs: (int)Vp9TxSize.Tx4x4,
            planeType: (int)Vp9BlockCoefEnums.PlaneType.Uv,
            tokenCache, aboveVEntropyCtx, leftVEntropyCtx);
    }

    private static void EncodePlaneCoefs(
        ref Vp8BoolEncoderGpuState state, ArrayView<byte> outBuf,
        ArrayView<short> coefs, long coefBase, int coefCount,
        ArrayView<byte> byteConsts, ArrayView<ushort> ushortConsts,
        int aboveCellOff, int leftCellOff, int cellsPerTx,
        int isTx4x4, int txSizeForCoefProbs, int planeType,
        ArrayView<byte> tokenCache,
        ArrayView<byte> aboveEntropyCtx, ArrayView<byte> leftEntropyCtx)
    {
        // Initial coef context = (aboveAgg != 0) + (leftAgg != 0).
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

        // Pick scan + neighbors based on tx size.
        long scanBase, neighborsBase, coefProbsBase;
        if (txSizeForCoefProbs == (int)Vp9TxSize.Tx4x4)
        {
            scanBase = Vp9KeyframeConstantsGpu.Scan4x4Offset;
            neighborsBase = Vp9KeyframeConstantsGpu.Neighbors4x4Offset;
            coefProbsBase = Vp9KeyframeConstantsGpu.CoefProbs4x4Offset;
        }
        else if (txSizeForCoefProbs == (int)Vp9TxSize.Tx8x8)
        {
            scanBase = Vp9KeyframeConstantsGpu.Scan8x8Offset;
            neighborsBase = Vp9KeyframeConstantsGpu.Neighbors8x8Offset;
            coefProbsBase = Vp9KeyframeConstantsGpu.CoefProbs8x8Offset;
        }
        else // Tx16x16
        {
            scanBase = Vp9KeyframeConstantsGpu.Scan16x16Offset;
            neighborsBase = Vp9KeyframeConstantsGpu.Neighbors16x16Offset;
            coefProbsBase = Vp9KeyframeConstantsGpu.CoefProbs16x16Offset;
        }

        var scanView = ushortConsts.SubView(scanBase, coefCount);
        var neighborsView = ushortConsts.SubView(neighborsBase, (long)coefCount * 2);
        var coefProbsView = byteConsts.SubView(coefProbsBase, 432);
        var coefsView = coefs.SubView(coefBase, coefCount);
        var coefConstsView = byteConsts.SubView(
            Vp9KeyframeConstantsGpu.CoefConstsOffset,
            Vp9KeyframeConstantsGpu.CoefConstsLength);

        int eob = Vp9BlockCoefEncoderGpu.EncodeBlock(
            ref state, outBuf,
            coefsView,
            scanView, neighborsView,
            coefProbsView, coefConstsView, tokenCache,
            coefCount,
            planeType: planeType,
            refType: (int)Vp9BlockCoefEnums.RefType.Intra,
            initialCtx: initialCtx,
            isHighBitDepth: 0,
            isTx4x4: isTx4x4);

        // Update entropy context cells: each cell becomes (eob > 0 ? 1 : 0).
        byte ec = (byte)(eob > 0 ? 1 : 0);
        for (int i = 0; i < cellsPerTx; i++)
        {
            int aIdx = aboveCellOff + i;
            int lIdx = leftCellOff + i;
            if (aIdx < aboveEntropyCtx.Length) aboveEntropyCtx[aIdx] = ec;
            if (lIdx < leftEntropyCtx.Length) leftEntropyCtx[lIdx] = ec;
        }
    }

    /// <summary>Release kernel resources.</summary>
    public void Dispose() { /* auto-grouped */ }
}
