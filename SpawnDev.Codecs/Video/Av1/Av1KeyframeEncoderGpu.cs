// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 v1 keyframe encoder, GPU integration class. Symmetric to Vp8KeyframeEncoderGpu / Vp9KeyframeEncoderGpu. The
// host is a pure coordinator: alloc + upload + dispatch + readback.
//
//   Av1FrameSequentialEncodeKernel : one GPU thread per TILE of every frame runs the tile pipeline (predict, transform,
//                                    quantize, entropy code, reconstruct) bit-exact vs the CPU Av1KeyframeEncoder
//                                    reference, in bounded resumable launches of LeavesPerLaunch 16x16 leaves per tile.
//   Av1ObuAssembleKernel           : writes the Temporal Delimiter + Sequence Header OBUs, the Frame OBU header, the
//                                    leb128 size, the uncompressed frame header and the tile group (size prefixes +
//                                    tile bytes) into the caller's output. The only host-built bytes are a small fixed
//                                    blob (TD + SH + frame header, under 128 bytes of metadata).
//
// Speed comes from tiles (TileSizeSuperblocks). An AV1 tile is independent (own range coder, own contexts, no
// prediction across its edge), so all tiles of a frame are encoded by the same launches, one thread each. The
// default, one 64x64 superblock per tile, makes a frame cost about what one superblock costs; larger tiles spend
// fewer bits (contexts restart and prediction stops at every tile edge) and take proportionally longer. See
// Av1TileLayout for the grid rules and the note on level limits.
//
// API shape (same as the VP8 / VP9 encoders): three overloads per entry point, all funnelling into the ArrayView1D one.
//   - ArrayView1D : planes already on the accelerator, temporal unit assembled into a caller-owned view, returns the length.
//   - TypedArray  : browser. JS planes in via CopyFromJS, JS Uint8Array out; nothing enters the .NET heap.
//   - ReadOnlyMemory<byte> : managed. Desktop and small inputs; a terminal CPU sink/source.
//
// Any picture size. Planes are tightly packed 4:2:0: luma width x height, chroma ((width + 1) / 2) x ((height + 1) / 2).
// The coded size is each dimension rounded up to 8; blocks crossing the bottom / right edge use the AV1 edge partition
// syntax and 8x8 leaves, and source pixels past the picture repeat its last column / row (see the kernel's header).
//
// V1 limits (named, not hidden): 8-bit 4:2:0; DC prediction only; no loop filter / CDEF / restoration.

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// AV1 v1 keyframe encoder integration class.
/// </summary>
/// <remarks>Not thread-safe: one in-flight encode per instance.</remarks>
public sealed class Av1KeyframeEncoderGpu : IDisposable
{
    private const int BlobCapacity = 1024;

    /// <summary>Output bytes reserved per 16x16 leaf of a tile.</summary>
    private const int BytesPerLeaf = 2048;

    private readonly Accelerator _accelerator;
    private readonly Av1FrameSequentialEncodeKernel _frameKernel;
    private readonly Av1ObuAssembleKernel _obu;

    private readonly MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense> _dByteConsts;
    private readonly MemoryBuffer1D<ushort, global::ILGPU.Stride1D.Dense> _dUshortConsts;
    private readonly MemoryBuffer1D<short, global::ILGPU.Stride1D.Dense> _dDcAcQuant;
    private readonly MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense> _blob;
    // What the framing blob on the accelerator was built for (width, height, q, tile grid, frame rate), and where its
    // uncompressed header is.
    private (int, int, int, int, int, int, int) _blobKey = (-1, 0, 0, 0, 0, 0, 0);
    private int _blobBOffset;
    private int _blobBLen;
    private readonly MemoryBuffer1D<int, global::ILGPU.Stride1D.Dense> _outLen;

    // Recon readback (diagnostic entry points): crops the coded-size recon planes to the picture, on the accelerator.
    private Action<Index1D, ArrayView<byte>, ArrayView<byte>, int, int, int, int, int, int, long, long, long>? _packRecon;
    private readonly StageBuffer<byte> _stageRecon;

    // Per-resolution cached buffers - reallocated only when the picture size or the tile grid changes.
    private int _cachedWidth = -1;
    private int _cachedHeight = -1;
    private int _cachedTileCount = -1;
    private int _cachedTileStride = -1;
    private int _cachedScratchByte = -1;
    private MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense>? _dSrc;
    private MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense>? _dRecon;
    private MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense>? _dTile;
    private MemoryBuffer1D<long, global::ILGPU.Stride1D.Dense>? _dTileLen;
    private MemoryBuffer1D<int, global::ILGPU.Stride1D.Dense>? _dTileOff;
    private MemoryBuffer1D<byte, global::ILGPU.Stride1D.Dense>? _dScratchByte;
    private MemoryBuffer1D<int, global::ILGPU.Stride1D.Dense>? _dScratchInt;
    private MemoryBuffer1D<short, global::ILGPU.Stride1D.Dense>? _dScratchShort;

    // Grow-only staging for the managed and TypedArray overloads.
    private readonly StageBuffer<byte> _stageY;
    private readonly StageBuffer<byte> _stageU;
    private readonly StageBuffer<byte> _stageV;
    private readonly StageBuffer<byte> _stageOut;
    private readonly StageBuffer<int> _stageLens;

    private int _tileSizeSuperblocks = 1;
    private int _maxTiles = Av1TileLayout.HardwareDecoderMaxTiles;
    private int _frameRateNum = 30;
    private int _frameRateDen = 1;
    private int _levelLimit = Av1Level.MaximumParameters;

    /// <summary>Construct an encoder bound to <paramref name="accelerator"/>. Uploads the constant tables once.</summary>
    public Av1KeyframeEncoderGpu(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;

        _frameKernel = new Av1FrameSequentialEncodeKernel(accelerator);
        _obu = new Av1ObuAssembleKernel(accelerator);

        _dByteConsts = accelerator.Allocate1D<byte>(Av1KeyframeConstantsGpu.ByteConstsTotalBytes);
        _dUshortConsts = accelerator.Allocate1D<ushort>(Av1KeyframeConstantsGpu.UshortConstsTotalEntries);
        _dByteConsts.View.CopyFromCPU(Av1KeyframeConstantsGpu.BuildByteConstsBuffer());
        _dUshortConsts.View.CopyFromCPU(Av1KeyframeConstantsGpu.BuildUshortConstsBuffer());

        // DC[0..256) + AC[256..512) lookup tables for 8-bit content.
        var dcAc = new short[512];
        for (int i = 0; i < 256; i++)
        {
            dcAc[i] = Av1DequantTables.DcLookup8[i];
            dcAc[256 + i] = Av1DequantTables.AcLookup8[i];
        }
        _dDcAcQuant = accelerator.Allocate1D<short>(512);
        _dDcAcQuant.View.CopyFromCPU(dcAc);

        _blob = accelerator.Allocate1D<byte>(BlobCapacity);
        _outLen = accelerator.Allocate1D<int>(1);
        _stageY = new StageBuffer<byte>(accelerator);
        _stageU = new StageBuffer<byte>(accelerator);
        _stageV = new StageBuffer<byte>(accelerator);
        _stageOut = new StageBuffer<byte>(accelerator);
        _stageLens = new StageBuffer<int>(accelerator);
        _stageRecon = new StageBuffer<byte>(accelerator);
    }

    // ------------------------------------------------------------------
    // Tiles
    // ------------------------------------------------------------------

    /// <summary>
    /// Largest tile size, in 64x64 superblocks per side. Every tile is encoded by its own GPU thread, in the same
    /// launches, so this is the speed / size trade: 1 (the default) = one superblock per tile, the fastest and the
    /// most bits (contexts restart and prediction stops at every tile edge); larger = fewer, bigger tiles;
    /// <see cref="int.MaxValue"/> = the fewest tiles the picture size allows (one tile up to 4096 wide and
    /// 4096 x 2304 in area), the slowest and the smallest. The grid follows <see cref="Av1TileLayout"/>: on a
    /// picture where this size gives more than <see cref="MaxTiles"/> tiles, the tiles are larger.
    /// </summary>
    public int TileSizeSuperblocks
    {
        get => _tileSizeSuperblocks;
        set
        {
            if (value < 1) throw new ArgumentOutOfRangeException(nameof(value), "TileSizeSuperblocks must be at least 1.");
            _tileSizeSuperblocks = value;
        }
    }

    /// <summary>
    /// The most tiles a frame is coded with. Default <see cref="Av1TileLayout.HardwareDecoderMaxTiles"/> (256): the
    /// hardware AV1 decoders measured (NVDEC, D3D11VA, D3D12VA) fail on frames with more, the software decoders
    /// (libdav1d, libaom) do not. Raise it (up to 4096 = 64 x 64) only for streams a software decoder will read: a
    /// 1920x1080 keyframe is 255 tiles of 2 superblocks by default and 510 tiles of one with the limit raised.
    /// </summary>
    public int MaxTiles
    {
        get => _maxTiles;
        set
        {
            if (value < 1) throw new ArgumentOutOfRangeException(nameof(value), "MaxTiles must be at least 1.");
            _maxTiles = value;
        }
    }

    /// <summary>Frames per second the stream is shown at, numerator (default 30). See <see cref="SetFrameRate"/>.</summary>
    public int FrameRateNumerator => _frameRateNum;

    /// <summary>Frames per second the stream is shown at, denominator (default 1). See <see cref="SetFrameRate"/>.</summary>
    public int FrameRateDenominator => _frameRateDen;

    /// <summary>
    /// The rate the encoded frames will be shown at: <paramref name="numerator"/> / <paramref name="denominator"/>
    /// frames per second (default 30). It changes no coded picture; it decides the LEVEL the sequence header signals,
    /// because an AV1 level limits luma samples, frame headers, tiles and bits per second (<see cref="Av1Level"/>).
    /// Every frame signals the lowest level and tier a stream of frames like it, at this rate, conforms to
    /// (seq_level_idx 31, "maximum parameters", when it conforms to none), so set the real rate before encoding.
    /// A file takes ONE sequence header for all its frames: <see cref="Container.Mp4.Av1Isobmff.UnifySequenceHeaders"/>.
    /// </summary>
    public void SetFrameRate(int numerator, int denominator = 1)
    {
        if (numerator <= 0) throw new ArgumentOutOfRangeException(nameof(numerator), "The frame rate must be positive.");
        if (denominator <= 0) throw new ArgumentOutOfRangeException(nameof(denominator), "The frame rate must be positive.");
        _frameRateNum = numerator;
        _frameRateDen = denominator;
    }

    /// <summary>
    /// Keeps the tile grid inside an AV1 level: a seq_level_idx (<see cref="Av1Level.Level4_0"/>,
    /// <see cref="Av1Level.Level5_1"/>, ...), or <see cref="Av1Level.MaximumParameters"/> (31, the default) for no
    /// limit. This is the speed / decoder reach trade. Tiles are this encoder's parallelism, and a level allows few
    /// of them (8 at level 2.0 up to 128 at level 6.x, in at most 4 to 16 columns), so the default grid of one
    /// superblock per tile is the fastest and makes most pictures signal a high level, or 31 (a 1920x1080 frame of
    /// 255 tiles fits no level). With a limit, the grid uses at most the tiles of that level, or of the lowest level
    /// the picture size and frame rate need when that is higher (<see cref="Av1Level.TileLayout"/>): bigger tiles,
    /// a slower encode, and a stream every decoder of that level must decode. The level signalled can still be
    /// higher than the limit when the coded frames are too large for its bit rate (there is no rate control: lower
    /// the quality to lower the level).
    /// </summary>
    public int LevelLimit
    {
        get => _levelLimit;
        set
        {
            if (value != Av1Level.MaximumParameters && !Av1Level.TryGetLimits(value, out _))
                throw new ArgumentOutOfRangeException(nameof(value), $"seq_level_idx {value} is not a defined AV1 level (or 31).");
            _levelLimit = value;
        }
    }

    /// <summary>The tile grid this encoder uses for a <paramref name="width"/> x <paramref name="height"/> picture.</summary>
    public Av1TileLayout GetTileLayout(int width, int height)
    {
        RequireSupportedSize(width, height);
        return Av1Level.TileLayout(width, height, _tileSizeSuperblocks, _maxTiles, _frameRateNum, _frameRateDen, _levelLimit);
    }

    // ------------------------------------------------------------------
    // Geometry
    // ------------------------------------------------------------------

    /// <summary>Largest picture dimension: the AV1 frame size fields hold up to 16 bits.</summary>
    public const int MaxDimension = 65536;

    private static void RequireSupportedSize(int width, int height)
    {
        if (width <= 0 || width > MaxDimension) throw new ArgumentOutOfRangeException(nameof(width), $"width must be in [1, {MaxDimension}].");
        if (height <= 0 || height > MaxDimension) throw new ArgumentOutOfRangeException(nameof(height), $"height must be in [1, {MaxDimension}].");
    }

    /// <summary>Chroma plane width of a 4:2:0 picture <paramref name="width"/> wide: <c>(width + 1) / 2</c>.</summary>
    public static int ChromaWidth(int width) => (width + 1) >> 1;

    /// <summary>Chroma plane height of a 4:2:0 picture <paramref name="height"/> high: <c>(height + 1) / 2</c>.</summary>
    public static int ChromaHeight(int height) => (height + 1) >> 1;

    /// <summary>The coded size of a dimension: rounded up to a whole number of 8x8 blocks.</summary>
    private static int Coded(int dimension) => (dimension + 7) & ~7;

    /// <summary>Bytes of the recon planes (coded size, 4:2:0).</summary>
    private static int ReconLength(int width, int height) => Coded(width) * Coded(height) * 3 / 2;

    /// <summary>Output slot of one tile: <see cref="BytesPerLeaf"/> per 16x16 leaf slot of the largest tile, plus slack.</summary>
    private static int TileCapacity(Av1TileLayout tiles) =>
        checked(tiles.TileWidthSb * tiles.TileHeightSb * 16 * BytesPerLeaf + 256);

    /// <summary>
    /// Size in bytes of the output buffer one encoded keyframe needs for a <paramref name="width"/> x
    /// <paramref name="height"/> picture, whatever the tile size (the batch overload strides its slots by it).
    /// </summary>
    public static int GetMaxEncodedSize(int width, int height)
    {
        RequireSupportedSize(width, height);
        long sbCols = (Coded(width) + 63) >> 6, sbRows = (Coded(height) + 63) >> 6;
        // per superblock: its 16 leaf slots, the slack of a tile of its own and that tile's size prefix
        return checked((int)(sbCols * sbRows * (16L * BytesPerLeaf + 256 + Av1ObuAssembleKernel.TileSizeBytes) + BlobCapacity + 16));
    }

    /// <summary>The walker's parameters and per-tile scratch layout for a frame size and tile grid.</summary>
    private static Av1FrameSeqEncodeParams BuildParams(
        int width, int height, int baseQIndex, Av1TileLayout tiles,
        out int scratchByteLen, out int scratchIntLen, out int scratchShortLen)
    {
        int yLen = width * height;
        int uvLen = ChromaWidth(width) * ChromaHeight(height);
        int frameMiCols = ((width + 7) >> 3) << 1;
        int frameMiRows = ((height + 7) >> 3) << 1;
        int reconW = frameMiCols * 4, reconH = frameMiRows * 4;
        int aboveStride = tiles.TileWidthSb * 16;
        var p = new Av1FrameSeqEncodeParams
        {
            Width = width,
            Height = height,
            BaseQIndex = baseQIndex,
            YPlaneOff = 0,
            UPlaneOff = yLen,
            VPlaneOff = yLen + uvLen,
            FrameMiCols = frameMiCols,
            FrameMiRows = frameMiRows,
            ChromaWidth = ChromaWidth(width),
            ChromaHeight = ChromaHeight(height),
            ReconWidth = reconW,
            ReconHeight = reconH,
            ReconUOff = reconW * reconH,
            ReconVOff = reconW * reconH + (reconW >> 1) * (reconH >> 1),
            TileCols = tiles.TileCols,
            TileRows = tiles.TileRows,
            TileWidthSb = tiles.TileWidthSb,
            TileHeightSb = tiles.TileHeightSb,
            AboveStride = aboveStride,
        };
        int byteOff = 0;
        p.AboveEntropyOff = byteOff; byteOff += 3 * aboveStride;
        p.LeftEntropyOff = byteOff;  byteOff += 3 * 32;
        p.AbovePartOff = byteOff;    byteOff += aboveStride;
        p.LeftPartOff = byteOff;     byteOff += 32;
        p.AboveYModeOff = byteOff;   byteOff += aboveStride;
        p.LeftYModeOff = byteOff;    byteOff += 32;
        p.AboveSkipOff = byteOff;    byteOff += aboveStride;
        p.LeftSkipOff = byteOff;     byteOff += 32;
        p.EdgeAboveOff = byteOff;    byteOff += 33;
        p.EdgeLeftOff = byteOff;     byteOff += 33;
        p.PredictOff = byteOff;      byteOff += 256;
        p.LevelsOff = byteOff;       byteOff += 1384;
        scratchByteLen = byteOff;
        scratchIntLen = Av1FrameSequentialEncodeKernel.MinScratchIntLength;
        scratchShortLen = 256;
        return p;
    }

    private static void ValidatePlanes(
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> y, ArrayView1D<byte, global::ILGPU.Stride1D.Dense> u,
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> v, int width, int height, int baseQIndex, long frames = 1)
    {
        RequireSupportedSize(width, height);
        if (baseQIndex <= 0 || baseQIndex > 255)
            throw new ArgumentOutOfRangeException(nameof(baseQIndex), "baseQIndex must be in [1, 255].");
        long yLen = (long)width * height;
        long uvLen = (long)ChromaWidth(width) * ChromaHeight(height);
        if (y.Length < frames * yLen) throw new ArgumentException("yPlane too short.", nameof(y));
        if (u.Length < frames * uvLen) throw new ArgumentException("uPlane too short.", nameof(u));
        if (v.Length < frames * uvLen) throw new ArgumentException("vPlane too short.", nameof(v));
    }

    // ------------------------------------------------------------------
    // ArrayView1D overloads: the implementation
    // ------------------------------------------------------------------

    /// <summary>
    /// Encode a keyframe from tightly packed YUV 4:2:0 planes that are already on the accelerator into a
    /// complete AV1 temporal unit (TD + SH + Frame OBU) in <paramref name="output"/> (at least
    /// <see cref="GetMaxEncodedSize"/> bytes). Only the length scalar crosses to the host. Returns the length.
    /// </summary>
    /// <param name="yPlane">Luma plane, <c>width * height</c> bytes.</param>
    /// <param name="uPlane">Cb plane, <c>ChromaWidth(width) * ChromaHeight(height)</c> bytes.</param>
    /// <param name="vPlane">Cr plane, same size as <paramref name="uPlane"/>.</param>
    /// <param name="width">Picture width, 1..<see cref="MaxDimension"/>.</param>
    /// <param name="height">Picture height, 1..<see cref="MaxDimension"/>.</param>
    /// <param name="output">Receives the temporal unit.</param>
    /// <param name="baseQIndex">AV1 base quantizer index, 1..255.</param>
    public async Task<int> EncodeKeyFrameAsync(
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> yPlane,
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> uPlane,
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> vPlane,
        int width, int height,
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> output,
        int baseQIndex = 32)
    {
        ValidatePlanes(yPlane, uPlane, vPlane, width, height, baseQIndex);
        int cap = GetMaxEncodedSize(width, height);
        if (output.Length < cap)
            throw new ArgumentException($"output must be at least GetMaxEncodedSize({width}, {height}) bytes.", nameof(output));

        var tiles = GetTileLayout(width, height);
        await RunTilesAsync(yPlane, uPlane, vPlane, width, height, baseQIndex, tiles);
        Assemble(width, height, baseQIndex, tiles, _dTile!.View, _dTileLen!.View, _dTileOff!.View, output, _outLen.View, cap);
        await _accelerator.SynchronizeAsync();
        return (await _outLen.CopyToHostAsync())[0];
    }

    /// <summary>
    /// Dispatches the OBU assembly for one frame (the framing blob's upload when it changed, then the kernels).
    /// No synchronization. <paramref name="cap"/> = GetMaxEncodedSize: no temporal unit is longer.
    /// </summary>
    private void Assemble(
        int width, int height, int baseQIndex, Av1TileLayout tiles,
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> tileBytes, ArrayView1D<long, global::ILGPU.Stride1D.Dense> tileLen,
        ArrayView1D<int, global::ILGPU.Stride1D.Dense> tileOff,
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> output, ArrayView1D<int, global::ILGPU.Stride1D.Dense> outLen,
        int cap)
    {
        // The blob depends only on these (uniform tile spacing: the two log2 values are the grid), so it is built and
        // uploaded when one of them changes, not per frame.
        var key = (width, height, baseQIndex, tiles.TileColsLog2, tiles.TileRowsLog2, _frameRateNum, _frameRateDen);
        if (key != _blobKey)
        {
            byte[] blob = Av1KeyframeFraming.BuildFramingBlob(
                width, height, baseQIndex, tiles, _frameRateNum, _frameRateDen, out _blobBOffset, out _blobBLen);
            if (blob.Length > BlobCapacity) throw new InvalidOperationException("AV1 framing blob exceeds its buffer (internal error).");
            DeviceTransfer.Upload(_blob.View, blob);
            _blobKey = key;
        }
        _obu.Run(_blob.View, tileBytes, tileLen, tileOff, output, outLen, _blobBOffset, _blobBLen, tiles.TileCount, TileCapacity(tiles), cap);
    }

    /// <summary>
    /// Runs the walker on the planes with the tile grid <paramref name="tiles"/> and leaves each tile's raw
    /// range-coder bytes in <c>_dTile</c> (tile t at <c>t * TileCapacity</c>, lengths in <c>_dTileLen</c>) and the
    /// recon planes in <c>_dRecon</c>. Drains the stream before returning.
    /// </summary>
    private async Task RunTilesAsync(
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> yPlane,
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> uPlane,
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> vPlane,
        int width, int height, int baseQIndex, Av1TileLayout tiles)
    {
        int yLen = width * height;
        int uvLen = ChromaWidth(width) * ChromaHeight(height);
        int srcLen = yLen + uvLen + uvLen;
        var p = BuildParams(width, height, baseQIndex, tiles, out int scratchByteLen, out int scratchIntLen, out int scratchShortLen);
        int tileStride = TileCapacity(tiles);
        EnsureBuffers(width, height, srcLen, ReconLength(width, height), tiles.TileCount, tileStride, scratchByteLen, scratchIntLen, scratchShortLen);

        // Planes -> the walker's contiguous Y|U|V source, device to device.
        _dSrc!.View.SubView(0, yLen).CopyFrom(yPlane.SubView(0, yLen));
        _dSrc.View.SubView(yLen, uvLen).CopyFrom(uPlane.SubView(0, uvLen));
        _dSrc.View.SubView(yLen + uvLen, uvLen).CopyFrom(vPlane.SubView(0, uvLen));

        // Pre-zero output buffers per frame (kernel reads them as carry-back state; the walker zeroes its own
        // scratchByte / scratchInt). Every recon pixel a block predicts from was written by an earlier block of
        // the same tile, so recon needs no zero-fill.
        _dTile!.View.MemSetToZero();
        _dTileLen!.View.MemSetToZero();
        _dScratchShort!.View.MemSetToZero();

        var strides = new Av1FrameBatchStrides
        {
            SrcStride = srcLen,
            ReconStride = ReconLength(width, height),
            TileStride = tileStride,
            ScratchByteStride = scratchByteLen,
            ScratchIntStride = scratchIntLen,
            ScratchShortStride = scratchShortLen,
        };
        await _frameKernel.RunAsync(
            _dSrc.View, _dRecon!.View,
            _dTile.View, _dTileLen.View,
            _dByteConsts.View, _dUshortConsts.View,
            _dDcAcQuant.View,
            _dScratchByte!.View, _dScratchInt!.View, _dScratchShort.View,
            p, 1, strides);
        await _accelerator.SynchronizeAsync();
    }

    /// <summary>
    /// Frame-batch AV1 keyframe encoder for frames already on the accelerator: every tile of every frame is one
    /// thread of the same launches (each with its own range encoder state, scratch and output slot). Only the
    /// per-frame lengths (written to <paramref name="lengths"/>) are produced.
    /// </summary>
    /// <param name="yPlanes">Frame-major tightly packed luma planes: frame f at <c>f * width * height</c>.</param>
    /// <param name="uPlanes">Frame-major tightly packed Cb planes: frame f at <c>f * ChromaWidth(width) * ChromaHeight(height)</c>.</param>
    /// <param name="vPlanes">Frame-major tightly packed Cr planes, same layout as <paramref name="uPlanes"/>.</param>
    /// <param name="frameCount">Number of frames.</param>
    /// <param name="width">Picture width, 1..<see cref="MaxDimension"/>.</param>
    /// <param name="height">Picture height, 1..<see cref="MaxDimension"/>.</param>
    /// <param name="outputs">Frame f's temporal unit is written at <c>f * GetMaxEncodedSize(width, height)</c>.</param>
    /// <param name="lengths">Receives each frame's length; at least <paramref name="frameCount"/> ints.</param>
    /// <param name="baseQIndex">AV1 base quantizer index, 1..255.</param>
    public async Task EncodeKeyFramesBatchAsync(
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> yPlanes,
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> uPlanes,
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> vPlanes,
        int frameCount, int width, int height,
        ArrayView1D<byte, global::ILGPU.Stride1D.Dense> outputs,
        ArrayView1D<int, global::ILGPU.Stride1D.Dense> lengths,
        int baseQIndex = 32)
    {
        if (frameCount <= 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
        ValidatePlanes(yPlanes, uPlanes, vPlanes, width, height, baseQIndex, frameCount);
        int cap = GetMaxEncodedSize(width, height);
        if (outputs.Length < (long)frameCount * cap)
            throw new ArgumentException("outputs must hold frameCount * GetMaxEncodedSize bytes.", nameof(outputs));
        if (lengths.Length < frameCount) throw new ArgumentException("lengths must hold frameCount ints.", nameof(lengths));

        int yLen = width * height;
        int uvLen = ChromaWidth(width) * ChromaHeight(height);
        int srcLen = yLen + uvLen + uvLen;
        int reconLen = ReconLength(width, height);
        var tiles = GetTileLayout(width, height);
        int tileCount = tiles.TileCount;
        int tileStride = TileCapacity(tiles);
        long slots = (long)frameCount * tileCount;
        var p = BuildParams(width, height, baseQIndex, tiles, out int scratchByteLen, out int scratchIntLen, out int scratchShortLen);

        // Per-frame source / recon, per-tile everything else.
        using var dAllSrc = _accelerator.Allocate1D<byte>((long)frameCount * srcLen);
        using var dAllRecon = _accelerator.Allocate1D<byte>((long)frameCount * reconLen);
        using var dAllTile = _accelerator.Allocate1D<byte>(slots * tileStride);
        using var dAllTileLen = _accelerator.Allocate1D<long>(slots);
        int offStride = Av1ObuAssembleKernel.OffsetsLength(tileCount);
        using var dAllTileOff = _accelerator.Allocate1D<int>((long)frameCount * offStride);
        using var dAllScratchByte = _accelerator.Allocate1D<byte>(slots * scratchByteLen);
        using var dAllScratchInt = _accelerator.Allocate1D<int>(slots * scratchIntLen);
        using var dAllScratchShort = _accelerator.Allocate1D<short>(slots * scratchShortLen);

        dAllTile.View.MemSetToZero();
        dAllTileLen.View.MemSetToZero();
        dAllScratchShort.View.MemSetToZero();

        // Planes -> per-frame Y|U|V slots, device to device.
        for (int f = 0; f < frameCount; f++)
        {
            long slot = (long)f * srcLen;
            dAllSrc.View.SubView(slot, yLen).CopyFrom(yPlanes.SubView((long)f * yLen, yLen));
            dAllSrc.View.SubView(slot + yLen, uvLen).CopyFrom(uPlanes.SubView((long)f * uvLen, uvLen));
            dAllSrc.View.SubView(slot + yLen + uvLen, uvLen).CopyFrom(vPlanes.SubView((long)f * uvLen, uvLen));
        }

        var strides = new Av1FrameBatchStrides
        {
            SrcStride = srcLen,
            ReconStride = reconLen,
            TileStride = tileStride,
            ScratchByteStride = scratchByteLen,
            ScratchIntStride = scratchIntLen,
            ScratchShortStride = scratchShortLen,
        };

        // Every tile of every frame advances LeavesPerLaunch leaves per launch.
        await _frameKernel.RunAsync(
            dAllSrc.View, dAllRecon.View,
            dAllTile.View, dAllTileLen.View,
            _dByteConsts.View, _dUshortConsts.View, _dDcAcQuant.View,
            dAllScratchByte.View, dAllScratchInt.View, dAllScratchShort.View,
            p, frameCount, strides);

        // One shared framing blob; one assembly per frame straight into the caller's slot (paced).
        var obuPacer = new LaunchPacer(_accelerator);
        for (int f = 0; f < frameCount; f++)
        {
            Assemble(width, height, baseQIndex, tiles,
                dAllTile.View.SubView((long)f * tileCount * tileStride, (long)tileCount * tileStride),
                dAllTileLen.View.SubView((long)f * tileCount, tileCount),
                dAllTileOff.View.SubView((long)f * offStride, offStride),
                outputs.SubView((long)f * cap, cap),
                lengths.SubView(f, 1),
                cap);
            await obuPacer.LaunchedAsync();
        }

        // Drain before the using-scoped slot buffers dispose: on the browser backends a dispatch that still
        // references a freed buffer reads freed memory.
        await _accelerator.SynchronizeAsync();
    }

    // ------------------------------------------------------------------
    // Managed overloads
    // ------------------------------------------------------------------

    private void StagePlanes(ReadOnlyMemory<byte> y, ReadOnlyMemory<byte> u, ReadOnlyMemory<byte> v, int width, int height)
    {
        RequireSupportedSize(width, height);
        int cw = ChromaWidth(width), ch = ChromaHeight(height);
        DeviceTransfer.UploadPlane(_stageY, y, width, width, height, nameof(y));
        DeviceTransfer.UploadPlane(_stageU, u, cw, cw, ch, nameof(u));
        DeviceTransfer.UploadPlane(_stageV, v, cw, cw, ch, nameof(v));
        _stageOut.Ensure(GetMaxEncodedSize(width, height));
    }

    /// <summary>
    /// Encode a full AV1 v1 keyframe (TD OBU + SH OBU + Frame OBU) from managed planes and return the byte stream
    /// as a managed array (a terminal CPU sink; in a browser prefer the <see cref="TypedArray"/> overload).
    /// </summary>
    public async Task<byte[]> EncodeKeyFrameAsync(
        ReadOnlyMemory<byte> yPlane, ReadOnlyMemory<byte> uPlane, ReadOnlyMemory<byte> vPlane,
        int width, int height,
        int baseQIndex = 32)
    {
        StagePlanes(yPlane, uPlane, vPlane, width, height);
        int n = await EncodeKeyFrameAsync(_stageY.View, _stageU.View, _stageV.View, width, height, _stageOut.View, baseQIndex);
        return await DeviceTransfer.DownloadAsync(_stageOut.View, n);
    }

    /// <summary>
    /// Run the GPU walker on managed planes as ONE tile (whatever <see cref="TileSizeSuperblocks"/> is) and return
    /// the raw range-coder tile bytes (the bytes the CPU <c>Av1KeyframeEncoder.EncodeSingleTile</c> produces) - a
    /// test / diagnostic entry point. Pictures too large for a single tile throw.
    /// </summary>
    public async Task<byte[]> EncodeSingleTileAsync(
        ReadOnlyMemory<byte> yPlane, ReadOnlyMemory<byte> uPlane, ReadOnlyMemory<byte> vPlane,
        int width, int height,
        int baseQIndex = 32)
    {
        ValidateManagedQ(baseQIndex);
        StagePlanes(yPlane, uPlane, vPlane, width, height);
        var tiles = Av1TileLayout.SingleTile(width, height);
        if (tiles.TileCount != 1)
            throw new NotSupportedException($"A {width}x{height} picture cannot be a single AV1 tile (it needs {tiles.TileCount}).");
        await RunTilesAsync(_stageY.View, _stageU.View, _stageV.View, width, height, baseQIndex, tiles);
        long tileLen = (await _dTileLen!.CopyToHostAsync())[0];
        return await DeviceTransfer.DownloadAsync(_dTile!.View, tileLen);
    }

    /// <summary>
    /// Like <see cref="EncodeSingleTileAsync"/> but also returns the encoder's internal recon planes - the data the
    /// downstream decoder must reconstruct (self-consistency tests of the encoder / decoder pair).
    /// </summary>
    public async Task<(byte[] tileBytes, byte[] yRecon, byte[] uRecon, byte[] vRecon)>
        EncodeSingleTileWithReconAsync(
        ReadOnlyMemory<byte> yPlane, ReadOnlyMemory<byte> uPlane, ReadOnlyMemory<byte> vPlane,
        int width, int height,
        int baseQIndex = 32)
    {
        var tile = await EncodeSingleTileAsync(yPlane, uPlane, vPlane, width, height, baseQIndex);
        var (y, u, v) = await ReadReconPlanesAsync(width, height);
        return (tile, y, u, v);
    }

    /// <summary>
    /// The reconstruction of the last single-frame encode, cropped to the picture: one tight I420 buffer (Y
    /// <c>width * height</c>, then Cb, then Cr at <c>ChromaWidth * ChromaHeight</c> each) - what a decoder shows for
    /// that keyframe. The crop runs on the accelerator; the result is a managed array (a diagnostic / test sink).
    /// </summary>
    public async Task<byte[]> ReadReconAsync()
    {
        if (_dRecon == null || _cachedWidth <= 0) throw new InvalidOperationException("No frame has been encoded.");
        int width = _cachedWidth, height = _cachedHeight;
        int cw = ChromaWidth(width), ch = ChromaHeight(height);
        int reconW = Coded(width), reconH = Coded(height);
        int total = width * height + 2 * cw * ch;
        _stageRecon.Ensure(total);
        _packRecon ??= await _accelerator.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<byte>, ArrayView<byte>,
            int, int, int, int, int, int, long, long, long>(Vp9.Decoder.Vp9OutputKernels.PackPlanesKernel);
        long uOff = (long)reconW * reconH;
        _packRecon(total, _dRecon.View, _stageRecon.View, width, height, cw, ch, reconW, reconW >> 1,
            0L, uOff, uOff + (long)(reconW >> 1) * (reconH >> 1));
        return await DeviceTransfer.DownloadAsync(_stageRecon.View, total);
    }

    private async Task<(byte[] y, byte[] u, byte[] v)> ReadReconPlanesAsync(int width, int height)
    {
        byte[] i420 = await ReadReconAsync();
        int yLen = width * height, uvLen = ChromaWidth(width) * ChromaHeight(height);
        return (i420.AsSpan(0, yLen).ToArray(), i420.AsSpan(yLen, uvLen).ToArray(), i420.AsSpan(yLen + uvLen, uvLen).ToArray());
    }

    /// <summary>
    /// Encode + return both the encoded keyframe bytes AND the encoder's internal recon planes (managed planes in,
    /// managed out; a diagnostic entry point).
    /// </summary>
    public async Task<(byte[] bytes, byte[] yRecon, byte[] uRecon, byte[] vRecon)>
        EncodeKeyFrameWithReconAsync(
        ReadOnlyMemory<byte> yPlane, ReadOnlyMemory<byte> uPlane, ReadOnlyMemory<byte> vPlane,
        int width, int height,
        int baseQIndex = 32)
    {
        ValidateManagedQ(baseQIndex);
        StagePlanes(yPlane, uPlane, vPlane, width, height);
        int n = await EncodeKeyFrameAsync(_stageY.View, _stageU.View, _stageV.View, width, height, _stageOut.View, baseQIndex);
        var bytes = await DeviceTransfer.DownloadAsync(_stageOut.View, n);
        var (y, u, v) = await ReadReconPlanesAsync(width, height);
        return (bytes, y, u, v);
    }

    private static void ValidateManagedQ(int baseQIndex)
    {
        if (baseQIndex <= 0 || baseQIndex > 255)
            throw new ArgumentOutOfRangeException(nameof(baseQIndex), "baseQIndex must be in [1, 255].");
    }

    /// <summary>
    /// Batch-encode managed keyframes. Every plane is tightly packed. Each plane is uploaded straight into the
    /// device staging buffer (no host-side concatenation); each result is a complete temporal unit.
    /// </summary>
    public async Task<byte[][]> EncodeKeyFramesBatchAsync(
        ReadOnlyMemory<byte>[] yPlanes,
        ReadOnlyMemory<byte>[] uPlanes,
        ReadOnlyMemory<byte>[] vPlanes,
        int width, int height,
        int baseQIndex = 32)
    {
        if (yPlanes is null) throw new ArgumentNullException(nameof(yPlanes));
        if (uPlanes is null) throw new ArgumentNullException(nameof(uPlanes));
        if (vPlanes is null) throw new ArgumentNullException(nameof(vPlanes));
        if (yPlanes.Length != uPlanes.Length || yPlanes.Length != vPlanes.Length)
            throw new ArgumentException("Plane arrays must have equal length.");
        int frameCount = yPlanes.Length;
        if (frameCount == 0) return System.Array.Empty<byte[]>();
        RequireSupportedSize(width, height);

        long yPlane = (long)width * height;
        long uvPlane = (long)ChromaWidth(width) * ChromaHeight(height);
        int cap = GetMaxEncodedSize(width, height);
        _stageY.Ensure(frameCount * yPlane);
        _stageU.Ensure(frameCount * uvPlane);
        _stageV.Ensure(frameCount * uvPlane);
        _stageOut.Ensure((long)frameCount * cap);
        _stageLens.Ensure(frameCount);
        for (int f = 0; f < frameCount; f++)
        {
            DeviceTransfer.UploadPlaneSlot(_stageY, f * yPlane, yPlane, yPlanes[f], nameof(yPlanes));
            DeviceTransfer.UploadPlaneSlot(_stageU, f * uvPlane, uvPlane, uPlanes[f], nameof(uPlanes));
            DeviceTransfer.UploadPlaneSlot(_stageV, f * uvPlane, uvPlane, vPlanes[f], nameof(vPlanes));
        }

        await EncodeKeyFramesBatchAsync(
            _stageY.View, _stageU.View, _stageV.View, frameCount, width, height, _stageOut.View, _stageLens.View, baseQIndex);

        var lens = await _stageLens.View.SubView(0, frameCount).CopyToHostAsync();
        var results = new byte[frameCount][];
        for (int f = 0; f < frameCount; f++)
            results[f] = lens[f] > 0
                ? await DeviceTransfer.DownloadAsync(_stageOut.View.SubView((long)f * cap, lens[f]), lens[f])
                : System.Array.Empty<byte>();
        return results;
    }

    // ------------------------------------------------------------------
    // TypedArray overloads (browser: bulk bytes stay in JS)
    // ------------------------------------------------------------------

    /// <summary>
    /// Encode a keyframe from JS typed arrays (browser accelerators only). The planes go to the GPU with
    /// <c>CopyFromJS</c> and the temporal unit comes back as a JS <see cref="Uint8Array"/> (the caller disposes it);
    /// no bulk byte enters the .NET heap.
    /// </summary>
    /// <exception cref="NotSupportedException">The accelerator is not a browser accelerator.</exception>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> EncodeKeyFrameAsync(
        TypedArray yPlane, TypedArray uPlane, TypedArray vPlane,
        int width, int height,
        int baseQIndex = 32)
    {
        ArgumentNullException.ThrowIfNull(yPlane);
        ArgumentNullException.ThrowIfNull(uPlane);
        ArgumentNullException.ThrowIfNull(vPlane);
        RequireSupportedSize(width, height);
        _stageY.Ensure(yPlane.ByteLength);
        _stageU.Ensure(uPlane.ByteLength);
        _stageV.Ensure(vPlane.ByteLength);
        _stageOut.Ensure(GetMaxEncodedSize(width, height));
        DeviceTransfer.Upload(_stageY.View, yPlane);
        DeviceTransfer.Upload(_stageU.View, uPlane);
        DeviceTransfer.Upload(_stageV.View, vPlane);
        int n = await EncodeKeyFrameAsync(_stageY.View, _stageU.View, _stageV.View, width, height, _stageOut.View, baseQIndex);
        return await DeviceTransfer.DownloadUint8ArrayAsync(_stageOut.View, n);
    }

    /// <summary>
    /// Batch-encode keyframes from JS typed arrays (browser accelerators only), one tightly packed plane per frame
    /// per component. Each returned <see cref="Uint8Array"/> is owned (and disposed) by the caller.
    /// </summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array[]> EncodeKeyFramesBatchAsync(
        TypedArray[] yPlanes, TypedArray[] uPlanes, TypedArray[] vPlanes,
        int width, int height,
        int baseQIndex = 32)
    {
        if (yPlanes is null) throw new ArgumentNullException(nameof(yPlanes));
        if (uPlanes is null) throw new ArgumentNullException(nameof(uPlanes));
        if (vPlanes is null) throw new ArgumentNullException(nameof(vPlanes));
        if (yPlanes.Length != uPlanes.Length || yPlanes.Length != vPlanes.Length)
            throw new ArgumentException("Plane arrays must have equal length.");
        int frameCount = yPlanes.Length;
        if (frameCount == 0) return System.Array.Empty<Uint8Array>();
        RequireSupportedSize(width, height);

        long yPlane = (long)width * height;
        long uvPlane = (long)ChromaWidth(width) * ChromaHeight(height);
        int cap = GetMaxEncodedSize(width, height);
        _stageY.Ensure(frameCount * yPlane);
        _stageU.Ensure(frameCount * uvPlane);
        _stageV.Ensure(frameCount * uvPlane);
        _stageOut.Ensure((long)frameCount * cap);
        _stageLens.Ensure(frameCount);
        for (int f = 0; f < frameCount; f++)
        {
            DeviceTransfer.Upload(_stageY.View.SubView(f * yPlane, yPlane), yPlanes[f]);
            DeviceTransfer.Upload(_stageU.View.SubView(f * uvPlane, uvPlane), uPlanes[f]);
            DeviceTransfer.Upload(_stageV.View.SubView(f * uvPlane, uvPlane), vPlanes[f]);
        }

        await EncodeKeyFramesBatchAsync(
            _stageY.View, _stageU.View, _stageV.View, frameCount, width, height, _stageOut.View, _stageLens.View, baseQIndex);

        var lens = await _stageLens.View.SubView(0, frameCount).CopyToHostAsync();
        var results = new Uint8Array[frameCount];
        for (int f = 0; f < frameCount; f++)
            results[f] = await DeviceTransfer.DownloadUint8ArrayAsync(_stageOut.View.SubView((long)f * cap, lens[f]), lens[f]);
        return results;
    }

    // ------------------------------------------------------------------
    // Working buffers
    // ------------------------------------------------------------------

    /// <summary>
    /// Ensure the single-frame GPU buffers fit the picture size and tile grid. Reallocates only when one of them
    /// changes.
    /// </summary>
    private void EnsureBuffers(int width, int height, int srcLen, int reconLen, int tileCount, int tileStride,
        int scratchByteLen, int scratchIntLen, int scratchShortLen)
    {
        if (_cachedWidth == width && _cachedHeight == height && _cachedTileCount == tileCount
            && _cachedTileStride == tileStride && _cachedScratchByte == scratchByteLen) return;
        DisposeFrameBuffers();
        _cachedWidth = width;
        _cachedHeight = height;
        _cachedTileCount = tileCount;
        _cachedTileStride = tileStride;
        _cachedScratchByte = scratchByteLen;
        _dSrc = _accelerator.Allocate1D<byte>(srcLen);
        _dRecon = _accelerator.Allocate1D<byte>(reconLen);
        _dTile = _accelerator.Allocate1D<byte>((long)tileCount * tileStride);
        _dTileLen = _accelerator.Allocate1D<long>(tileCount);
        _dTileOff = _accelerator.Allocate1D<int>(Av1ObuAssembleKernel.OffsetsLength(tileCount));
        _dScratchByte = _accelerator.Allocate1D<byte>((long)tileCount * scratchByteLen);
        _dScratchInt = _accelerator.Allocate1D<int>((long)tileCount * scratchIntLen);
        _dScratchShort = _accelerator.Allocate1D<short>((long)tileCount * scratchShortLen);
    }

    private void DisposeFrameBuffers()
    {
        _dSrc?.Dispose(); _dSrc = null;
        _dRecon?.Dispose(); _dRecon = null;
        _dTile?.Dispose(); _dTile = null;
        _dTileLen?.Dispose(); _dTileLen = null;
        _dTileOff?.Dispose(); _dTileOff = null;
        _dScratchByte?.Dispose(); _dScratchByte = null;
        _dScratchInt?.Dispose(); _dScratchInt = null;
        _dScratchShort?.Dispose(); _dScratchShort = null;
        _cachedWidth = -1;
        _cachedHeight = -1;
        _cachedTileCount = -1;
        _cachedTileStride = -1;
        _cachedScratchByte = -1;
    }

    /// <summary>Release every resource the encoder owns.</summary>
    public void Dispose()
    {
        _frameKernel.Dispose();
        _obu.Dispose();
        _dByteConsts.Dispose();
        _dUshortConsts.Dispose();
        _dDcAcQuant.Dispose();
        _blob.Dispose();
        _outLen.Dispose();
        DisposeFrameBuffers();
        _stageY.Dispose();
        _stageU.Dispose();
        _stageV.Dispose();
        _stageOut.Dispose();
        _stageLens.Dispose();
        _stageRecon.Dispose();
    }
}
