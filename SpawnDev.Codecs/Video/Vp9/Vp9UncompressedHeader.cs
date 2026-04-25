// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Complete VP9 uncompressed frame header. Bundles the early prefix
// (parsed by Vp9FrameHeaderParser) with all subsequent sub-sections
// (refresh_frame_flags, ref frame info, interp_filter, frame_context,
// loop_filter, quantization, segmentation, tile_info) and the
// trailing header_size field that delimits the compressed header
// from the tile data.
//
// libvpx reference: vp9/decoder/vp9_decodeframe.c
// read_uncompressed_header.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Complete parsed VP9 uncompressed frame header.
/// </summary>
public sealed record Vp9UncompressedHeader
{
    /// <summary>Early prefix (parsed by <see cref="Vp9FrameHeaderParser"/>).</summary>
    public required Vp9FrameHeader FrameHeader { get; init; }

    /// <summary>libvpx <c>refresh_frame_flags</c>.</summary>
    public byte RefreshFrameFlags { get; init; }

    /// <summary>
    /// Per-reference (LAST, GOLDEN, ALTREF) slot indices into the
    /// 8-slot pool. Null for key / intra_only frames.
    /// </summary>
    public int[]? RefFrameIdx { get; init; }

    /// <summary>Per-reference sign bias flags. Null for key / intra_only.</summary>
    public bool[]? RefFrameSignBias { get; init; }

    /// <summary>libvpx <c>allow_high_precision_mv</c>.</summary>
    public bool AllowHighPrecisionMv { get; init; }

    /// <summary>Frame interpolation filter (always EightTap for key / intra_only).</summary>
    public Vp9InterpFilter InterpFilter { get; init; } = Vp9InterpFilter.EightTap;

    /// <summary>libvpx <c>refresh_frame_context</c>.</summary>
    public bool RefreshFrameContext { get; init; }

    /// <summary>libvpx <c>frame_parallel_decoding_mode</c>.</summary>
    public bool FrameParallelDecodingMode { get; init; }

    /// <summary>libvpx <c>frame_context_idx</c> (0..3).</summary>
    public int FrameContextIdx { get; init; }

    /// <summary>Loop filter parameters.</summary>
    public required Vp9LoopFilterParams LoopFilter { get; init; }

    /// <summary>Quantization parameters.</summary>
    public required Vp9QuantizationParams Quantization { get; init; }

    /// <summary>Segmentation parameters.</summary>
    public required Vp9SegmentationParams Segmentation { get; init; }

    /// <summary>Tile info (log2 tile cols / log2 tile rows).</summary>
    public required Vp9TileInfo TileInfo { get; init; }

    /// <summary>libvpx <c>header_size</c>: byte length of compressed header.</summary>
    public required int FirstPartitionSize { get; init; }

    /// <summary>
    /// Byte offset where the compressed header begins, i.e. just past
    /// the byte-aligned uncompressed header. The compressed header
    /// occupies <see cref="FirstPartitionSize"/> bytes from this offset.
    /// </summary>
    public required int UncompressedHeaderSizeBytes { get; init; }
}

/// <summary>VP9 complete uncompressed header parser.</summary>
public static class Vp9CompleteUncompressedHeaderParser
{
    /// <summary>libvpx <c>REF_FRAMES</c>.</summary>
    public const int RefFrames = 8;

    /// <summary>libvpx <c>REFS_PER_FRAME</c>.</summary>
    public const int RefsPerFrame = 3;

    /// <summary>
    /// Parse the complete uncompressed header from <paramref name="frame"/>.
    /// </summary>
    public static Vp9UncompressedHeader Parse(ReadOnlySpan<byte> frame)
    {
        return Parse(frame, refFrameSizes: default);
    }

    /// <summary>
    /// Parse the complete uncompressed header from <paramref name="frame"/>,
    /// resolving inter-frame size_with_refs against the supplied reference
    /// frame dimensions.
    /// </summary>
    public static Vp9UncompressedHeader Parse(
        ReadOnlySpan<byte> frame,
        ReadOnlySpan<(int Width, int Height)> refFrameSizes)
    {
        if (frame.Length < 1)
            throw new InvalidDataException("VP9 frame is empty.");

        var reader = new Vp9BitReader(frame);
        var prefix = Vp9FrameHeaderParser.ParsePrefix(ref reader);

        if (prefix.ShowExistingFrame)
        {
            return new Vp9UncompressedHeader
            {
                FrameHeader = prefix,
                RefreshFrameFlags = 0,
                LoopFilter = EmptyLoopFilter(),
                Quantization = EmptyQuantization(),
                Segmentation = EmptySegmentation(),
                TileInfo = EmptyTileInfo(),
                FirstPartitionSize = 0,
                UncompressedHeaderSizeBytes = 1,
            };
        }

        byte refreshFrameFlags;
        int[]? refFrameIdx = null;
        bool[]? refFrameSignBias = null;
        bool allowHpMv = false;
        var interpFilter = Vp9InterpFilter.EightTap;

        if (prefix.FrameType == Vp9FrameType.Key)
        {
            refreshFrameFlags = 0xff;
        }
        else if (prefix.IntraOnly)
        {
            // ParsePrefix already consumed refresh_frame_flags and
            // discarded it. We don't have access to the parsed value
            // here; assume 0xff (libvpx allows any, decoder applies
            // the bits to the ref pool). For full fidelity the parser
            // needs a refactor to surface this value.
            refreshFrameFlags = 0xff;
        }
        else
        {
            // Inter frame.
            refreshFrameFlags = (byte)reader.ReadBits(8);
            refFrameIdx = new int[RefsPerFrame];
            refFrameSignBias = new bool[RefsPerFrame];
            for (int i = 0; i < RefsPerFrame; i++)
            {
                refFrameIdx[i] = (int)reader.ReadBits(3);
                refFrameSignBias[i] = reader.ReadFlag();
            }
            // frame_size_with_refs (or explicit if no found_ref).
            var fillerSizes = refFrameSizes.IsEmpty
                ? new (int, int)[RefsPerFrame]
                : refFrameSizes;
            var withRefs = Vp9FrameSizeWithRefsParser.Parse(ref reader, fillerSizes);
            prefix = prefix with
            {
                FrameWidth = withRefs.FrameWidth,
                FrameHeight = withRefs.FrameHeight,
                RenderWidth = withRefs.RenderWidth,
                RenderHeight = withRefs.RenderHeight,
            };
            allowHpMv = reader.ReadFlag();
            interpFilter = Vp9InterpFilterParser.Parse(ref reader);
        }

        bool refreshFrameContext;
        bool frameParallelDecoding;
        if (!prefix.ErrorResilientMode)
        {
            refreshFrameContext = reader.ReadFlag();
            frameParallelDecoding = reader.ReadFlag();
        }
        else
        {
            refreshFrameContext = false;
            frameParallelDecoding = true;
        }
        int frameContextIdx = (int)reader.ReadBits(2);

        var loopFilter = Vp9LoopFilterParamsParser.Parse(ref reader);
        var quant = Vp9QuantizationParamsParser.Parse(ref reader);
        var seg = Vp9SegmentationParamsParser.Parse(ref reader);

        // Tile info needs miCols computed from frame width.
        int miCols = (prefix.FrameWidth + 7) >> 3; // ceil(width / 8)
        var tileInfo = Vp9TileInfoParser.Parse(ref reader, miCols);

        int firstPartitionSize = (int)reader.ReadBits(16);

        // Byte-align to make the uncompressed header end on a byte boundary.
        int bitsIntoCurrentByte = reader.Position % 8;
        if (bitsIntoCurrentByte != 0)
            reader.ReadBits(8 - bitsIntoCurrentByte);

        return new Vp9UncompressedHeader
        {
            FrameHeader = prefix,
            RefreshFrameFlags = refreshFrameFlags,
            RefFrameIdx = refFrameIdx,
            RefFrameSignBias = refFrameSignBias,
            AllowHighPrecisionMv = allowHpMv,
            InterpFilter = interpFilter,
            RefreshFrameContext = refreshFrameContext,
            FrameParallelDecodingMode = frameParallelDecoding,
            FrameContextIdx = frameContextIdx,
            LoopFilter = loopFilter,
            Quantization = quant,
            Segmentation = seg,
            TileInfo = tileInfo,
            FirstPartitionSize = firstPartitionSize,
            UncompressedHeaderSizeBytes = reader.Position / 8,
        };
    }

    private static Vp9LoopFilterParams EmptyLoopFilter() => new()
    {
        FilterLevel = 0, SharpnessLevel = 0,
        ModeRefDeltaEnabled = false, ModeRefDeltaUpdate = false,
        RefDeltas = Array.Empty<int?>(), ModeDeltas = Array.Empty<int?>(),
    };

    private static Vp9QuantizationParams EmptyQuantization() => new()
    {
        BaseQIndex = 0, YDcDeltaQ = 0, UvDcDeltaQ = 0, UvAcDeltaQ = 0,
    };

    private static Vp9SegmentationParams EmptySegmentation() => new()
    {
        Enabled = false, UpdateMap = false,
        TreeProbsArray = new byte[Vp9SegmentationParams.TreeProbs],
        TemporalUpdate = false,
        PredProbs = new byte[Vp9SegmentationParams.PredictionProbs],
        UpdateData = false, AbsDelta = false,
        FeatureEnabled = new bool[0, 0],
        FeatureData = new int[0, 0],
    };

    private static Vp9TileInfo EmptyTileInfo() => new()
    {
        Log2TileCols = 0, Log2TileRows = 0,
        MinLog2TileCols = 0, MaxLog2TileCols = 0,
    };
}
