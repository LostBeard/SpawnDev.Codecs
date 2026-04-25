// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 decoder pipeline. Wires together superframe unpacking, frame-header
// parsing, and (incrementally) the per-tile block decode primitives.
//
// Current capability:
//   - Superframe unpacking (Annex B.1)
//   - Uncompressed header parsing (sec 6.2)
//   - Reports frame dimensions back through Width / Height after a key
//     or intra-only frame.
//   - Emits a placeholder mid-gray output for every visible frame so
//     callers can wire end-to-end pipelines while the residual / inter
//     prediction paths are being assembled.
//
// Frame-level decode steps still to wire up:
//   - Compressed header probability updates
//   - Tile data offset extraction
//   - Per-SB partition tree walk + per-block intra/inter decode
//   - Loop filter
//   - Reference frame pool / show_existing_frame replay
//
// Each of those is a downstream slice that plugs into the dispatcher
// without breaking the public IVideoDecoder contract.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>VP9 decoder.</summary>
public sealed class Vp9Decoder : IVideoDecoder
{
    /// <inheritdoc/>
    public VideoCodec Codec => VideoCodec.Vp9;

    /// <inheritdoc/>
    public int Width { get; private set; }

    /// <inheritdoc/>
    public int Height { get; private set; }

    /// <summary>
    /// Subsampling pair learned from the most recent keyframe / intra-only
    /// frame. 4:2:0 by default until the first such frame updates it.
    /// </summary>
    public Vp9SubsamplingPair Subsampling { get; private set; } = Vp9SubsamplingPair.Yuv420;

    /// <summary>Bit depth from the most recent keyframe / intra-only frame.</summary>
    public Vp9BitDepth BitDepth { get; private set; } = Vp9BitDepth.Bits8;

    /// <summary>
    /// Most recently parsed uncompressed frame header prefix; null
    /// before the first successful frame parse.
    /// </summary>
    public Vp9FrameHeader? LastFrameHeader { get; private set; }

    /// <summary>
    /// Most recently parsed COMPLETE uncompressed header (prefix +
    /// loop_filter + quant + segmentation + tile_info + header_size).
    /// Null before the first successful frame parse.
    /// </summary>
    public Vp9UncompressedHeader? LastCompleteHeader { get; private set; }

    /// <summary>
    /// Reference frame dimensions tracked from past keyframes / intra-only
    /// frames. 3 entries (LAST, GOLDEN, ALTREF), zeroed until the first
    /// keyframe lands.
    /// </summary>
    private readonly (int Width, int Height)[] _refFrameSizes = new (int, int)[3];

    /// <inheritdoc/>
    public async ValueTask<int> DecodeFrameAsync(
        ReadOnlyMemory<byte> compressedPacket,
        IVideoFrameSink frameSink,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(frameSink);

        var superframe = Vp9SuperframeParser.Parse(compressedPacket.Span);
        int emitted = 0;

        foreach (var slice in superframe.Frames)
        {
            ct.ThrowIfCancellationRequested();
            var frameBytes = compressedPacket.Slice(slice.Offset, slice.Length);
            var complete = Vp9CompleteUncompressedHeaderParser.Parse(
                frameBytes.Span, _refFrameSizes);
            var header = complete.FrameHeader;
            LastFrameHeader = header;
            LastCompleteHeader = complete;

            // Keyframes / intra-only frames carry color_config + frame_size,
            // so only those update our visible dimensions and chroma format.
            bool sizeCarrying = header.FrameType == Vp9FrameType.Key || header.IntraOnly;
            if (sizeCarrying && !header.ShowExistingFrame)
            {
                Width = header.FrameWidth;
                Height = header.FrameHeight;
                Subsampling = new Vp9SubsamplingPair(
                    SubsamplingX: header.SubsamplingX ? 1 : 0,
                    SubsamplingY: header.SubsamplingY ? 1 : 0);
                BitDepth = header.BitDepth switch
                {
                    8 => Vp9BitDepth.Bits8,
                    10 => Vp9BitDepth.Bits10,
                    12 => Vp9BitDepth.Bits12,
                    _ => Vp9BitDepth.Bits8,
                };
                // Update ref frame size pool from this keyframe.
                _refFrameSizes[0] = (Width, Height);
                _refFrameSizes[1] = (Width, Height);
                _refFrameSizes[2] = (Width, Height);
            }

            // Compressed header (probability updates). Sits between the
            // byte-aligned uncompressed header and the tile data, with
            // length first_partition_size. For show_existing_frame
            // there's no compressed header.
            if (!header.ShowExistingFrame && complete.FirstPartitionSize > 0)
            {
                ParseCompressedHeader(complete, frameBytes);
                LastTileGroup = Vp9TileGroupExtractor.Extract(frameBytes.Span, complete);
            }

            // Hidden alt-ref frames are decoded but not displayed. Until the
            // reference pool exists, skip emission for those.
            if (!header.ShowFrame && !header.ShowExistingFrame) continue;

            // show_existing_frame replays a buffered reference frame. Once
            // the reference pool exists, that's where we replay; for now we
            // fall through to a placeholder at current visible dimensions.
            if (Width <= 0 || Height <= 0) continue;

            await EmitPlaceholderFrameAsync(frameSink, ct).ConfigureAwait(false);
            emitted++;
        }

        return emitted;
    }

    /// <summary>Per-frame compressed-header state. Recreated for keyframes.</summary>
    private Vp9CompressedHeaderState _compressedState = new();

    /// <summary>Last parsed compressed-header tx_mode + reference_mode.</summary>
    public Vp9CompressedHeaderResult? LastCompressedResult { get; private set; }

    /// <summary>Per-tile byte ranges of the most recent visible frame.</summary>
    public Vp9TileGroup? LastTileGroup { get; private set; }

    /// <summary>
    /// Parse the compressed header from the frame at the byte offset
    /// derived from the complete uncompressed header. Updates the
    /// per-frame probability state and exposes tx_mode +
    /// reference_mode via <see cref="LastCompressedResult"/>.
    /// </summary>
    private void ParseCompressedHeader(Vp9UncompressedHeader complete, ReadOnlyMemory<byte> frameBytes)
    {
        // libvpx VP9 frames re-seed prob state on keyframes / intra_only;
        // inter frames carry over previous state. For now we always
        // start with fresh defaults to avoid carrying stale state into
        // mismatching frames.
        if (complete.FrameHeader.FrameType == Vp9FrameType.Key
            || complete.FrameHeader.IntraOnly)
        {
            _compressedState = new Vp9CompressedHeaderState();
        }

        var inputs = new Vp9CompressedHeaderInputs(
            IsLossless: complete.Quantization.BaseQIndex == 0
                && complete.Quantization.YDcDeltaQ == 0
                && complete.Quantization.UvDcDeltaQ == 0
                && complete.Quantization.UvAcDeltaQ == 0,
            IsIntraOnly: complete.FrameHeader.FrameType == Vp9FrameType.Key
                || complete.FrameHeader.IntraOnly,
            InterpFilter: complete.InterpFilter,
            AllowHighPrecisionMv: complete.AllowHighPrecisionMv,
            SignBiasLast: complete.RefFrameSignBias is { } sb && sb.Length > 0 && sb[0],
            SignBiasGolden: complete.RefFrameSignBias is { } sb2 && sb2.Length > 1 && sb2[1],
            SignBiasAltRef: complete.RefFrameSignBias is { } sb3 && sb3.Length > 2 && sb3[2]);

        // Materialize a contiguous buffer covering the compressed header
        // bytes for the bool decoder (Vp9BoolDecoder takes byte[], not Span).
        var headerBytes = frameBytes.Span.Slice(
            complete.UncompressedHeaderSizeBytes,
            complete.FirstPartitionSize).ToArray();
        var reader = new Vp9BoolDecoder(headerBytes, 0, headerBytes.Length);

        LastCompressedResult = Vp9CompressedHeaderParser.Read(_compressedState, inputs, reader);
    }

    /// <summary>
    /// Emit a mid-gray frame at the decoder's current visible dimensions.
    /// This is the placeholder output while the full block-decode pipeline
    /// is being wired up. Equivalent to what an all-skip all-DC intra-only
    /// frame with no neighbors would produce after intra prediction.
    /// </summary>
    private async ValueTask EmitPlaceholderFrameAsync(IVideoFrameSink sink, CancellationToken ct)
    {
        int yW = Width, yH = Height;
        int uW = Subsampling.ChromaWidth(yW);
        int uH = Subsampling.ChromaHeight(yH);

        var y = new byte[yW * yH];
        var u = new byte[uW * uH];
        var v = new byte[uW * uH];

        Array.Fill(y, (byte)128);
        Array.Fill(u, (byte)128);
        Array.Fill(v, (byte)128);

        ct.ThrowIfCancellationRequested();
        await sink.OnFrameAsync(
            y, yW,
            u, uW,
            v, uW,
            pts: 0L).ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}
