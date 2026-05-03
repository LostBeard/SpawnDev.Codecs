// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 decoder pipeline. Currently parses OBU framing + Sequence Header
// metadata. Per-frame block decode (intra/inter prediction, inverse
// transforms, CDEF, loop restoration, film-grain synthesis) is the
// remaining work, scoped across multiple phases following dav1d's
// structure with ILGPU-accelerated kernels across all 6 backends.

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>AV1 decoder.</summary>
public sealed class Av1Decoder : IVideoDecoder
{
    /// <inheritdoc/>
    public VideoCodec Codec => VideoCodec.Av1;

    /// <inheritdoc/>
    public int Width { get; private set; }

    /// <inheritdoc/>
    public int Height { get; private set; }

    /// <summary>Most recently parsed Sequence Header; null before the first SH OBU.</summary>
    public Av1SequenceHeader? LastSequenceHeader { get; private set; }

    /// <summary>
    /// Most recently parsed FrameHeader (prefix-fields only). Updated for
    /// every Frame / FrameHeader / RedundantFrameHeader OBU.
    /// </summary>
    public Av1FrameHeader? LastFrameHeader { get; private set; }

    /// <summary>
    /// Number of OBUs the most recent <see cref="DecodeFrameAsync"/>
    /// invocation parsed, broken down by type.
    /// </summary>
    public IReadOnlyDictionary<Av1ObuType, int> LastFrameObuCounts { get; private set; }
        = new Dictionary<Av1ObuType, int>();

    /// <summary>
    /// Cumulative count of every OBU type observed across all
    /// <see cref="DecodeFrameAsync"/> calls on this decoder instance.
    /// </summary>
    public IReadOnlyDictionary<Av1ObuType, int> CumulativeObuCounts => _cumulativeObuCounts;
    private readonly Dictionary<Av1ObuType, int> _cumulativeObuCounts = new();

    /// <summary>
    /// Cumulative count of every Av1FrameType observed across coded
    /// frames (excludes show_existing_frame replays).
    /// </summary>
    public IReadOnlyDictionary<Av1FrameType, int> CumulativeFrameTypeCounts => _cumulativeFrameTypeCounts;
    private readonly Dictionary<Av1FrameType, int> _cumulativeFrameTypeCounts = new();

    /// <summary>
    /// Cumulative count of show_existing_frame OBUs (frames that just
    /// replay a buffered reference slot, no coded body).
    /// </summary>
    public int ShowExistingFrameCount { get; private set; }

    /// <summary>Total number of Temporal Units (IVF frames) decoded.</summary>
    public int TotalTemporalUnits { get; private set; }

    /// <inheritdoc/>
    public async ValueTask<int> DecodeFrameAsync(
        ReadOnlyMemory<byte> compressedPacket,
        IVideoFrameSink frameSink,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(frameSink);

        TotalTemporalUnits++;

        // Parse all OBUs in this Temporal Unit.
        var counts = new Dictionary<Av1ObuType, int>();
        bool hasFrameData = false;
        ReadOnlyMemory<byte>? frameObuPayload = null;

        foreach (var obu in Av1ObuParser.EnumerateObus(compressedPacket))
        {
            counts.TryGetValue(obu.Type, out int c);
            counts[obu.Type] = c + 1;
            _cumulativeObuCounts.TryGetValue(obu.Type, out int cc);
            _cumulativeObuCounts[obu.Type] = cc + 1;

            if (obu.Type == Av1ObuType.SequenceHeader)
            {
                var sh = Av1SequenceHeaderParser.Parse(
                    compressedPacket.Span.Slice(obu.PayloadOffset, obu.PayloadLength));
                LastSequenceHeader = sh;
                Width = sh.MaxFrameWidth;
                Height = sh.MaxFrameHeight;
            }
            else if (obu.IsCodedFrameData)
            {
                hasFrameData = true;
                if (obu.Type == Av1ObuType.Frame)
                {
                    // Capture the Frame OBU payload for the walker; only
                    // the Frame OBU carries both the complete header and
                    // the tile data the walker needs.
                    frameObuPayload = compressedPacket.Slice(obu.PayloadOffset, obu.PayloadLength);
                }
                if (LastSequenceHeader is not null)
                {
                    var fh = Av1FrameHeaderParser.Parse(
                        compressedPacket.Span.Slice(obu.PayloadOffset, obu.PayloadLength),
                        LastSequenceHeader);
                    LastFrameHeader = fh;
                    if (fh.ShowExistingFrame)
                    {
                        ShowExistingFrameCount++;
                    }
                    else
                    {
                        _cumulativeFrameTypeCounts.TryGetValue(fh.FrameType, out int fc);
                        _cumulativeFrameTypeCounts[fh.FrameType] = fc + 1;
                    }
                }
            }
        }

        LastFrameObuCounts = counts;

        if (hasFrameData && Width > 0 && Height > 0)
        {
            // Drive the walker if we have a Frame OBU + sequence header.
            // Walker handles intra-only frames; non-keyframe paths +
            // multi-tile-group bitstreams fall back to placeholder.
            if (frameObuPayload is { } payload && LastSequenceHeader is { } seq)
            {
                try
                {
                    var complete = Av1CompleteFrameHeaderParser.Parse(payload.Span, seq);
                    var tg = Av1TileGroupExtractor.Extract(payload.Span, complete);
                    var walker = new Av1KeyframeWalker();
                    var fb = walker.DecodeFrame(payload, seq, complete, tg);
                    await frameSink.OnFrameAsync(
                        fb.Y, fb.LumaWidth,
                        fb.U, fb.ChromaWidth,
                        fb.V, fb.ChromaWidth,
                        pts: 0L).ConfigureAwait(false);
                    return 1;
                }
                catch (NotImplementedException)
                {
                    // Walker doesn't support this configuration yet
                    // (inter frames, screen content tools, etc.); fall
                    // back to placeholder.
                }
            }
            await EmitPlaceholderFrameAsync(frameSink, ct).ConfigureAwait(false);
            return 1;
        }
        return 0;
    }

    private async ValueTask EmitPlaceholderFrameAsync(IVideoFrameSink sink, CancellationToken ct)
    {
        // Default to 4:2:0 chroma until the SH update path differentiates.
        int yW = Width, yH = Height;
        int uW = (LastSequenceHeader?.SubsamplingX ?? 1) == 1 ? yW / 2 : yW;
        int uH = (LastSequenceHeader?.SubsamplingY ?? 1) == 1 ? yH / 2 : yH;
        var y = new byte[yW * yH];
        var u = new byte[uW * uH];
        var v = new byte[uW * uH];
        Array.Fill(y, (byte)128);
        Array.Fill(u, (byte)128);
        Array.Fill(v, (byte)128);
        ct.ThrowIfCancellationRequested();
        await sink.OnFrameAsync(y, yW, u, uW, v, uW, pts: 0L).ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}
