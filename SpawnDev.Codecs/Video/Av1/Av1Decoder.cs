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
    /// Cumulative count of every Av1FrameType observed across all
    /// <see cref="DecodeFrameAsync"/> calls on this decoder instance.
    /// Populated for every frame whose header was parsed.
    /// </summary>
    public IReadOnlyDictionary<Av1FrameType, int> CumulativeFrameTypeCounts => _cumulativeFrameTypeCounts;
    private readonly Dictionary<Av1FrameType, int> _cumulativeFrameTypeCounts = new();

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
                if (LastSequenceHeader is not null)
                {
                    var fh = Av1FrameHeaderParser.Parse(
                        compressedPacket.Span.Slice(obu.PayloadOffset, obu.PayloadLength),
                        LastSequenceHeader);
                    LastFrameHeader = fh;
                    _cumulativeFrameTypeCounts.TryGetValue(fh.FrameType, out int fc);
                    _cumulativeFrameTypeCounts[fh.FrameType] = fc + 1;
                }
                // Per-frame block decode goes here once the inverse-transform
                // + prediction + entropy-decode pipeline is wired up.
            }
        }

        LastFrameObuCounts = counts;

        // Emit a placeholder mid-gray frame for every Temporal Unit
        // that carried coded frame data, at current learned dimensions.
        // Once block decode lands these become real pixels.
        if (hasFrameData && Width > 0 && Height > 0)
        {
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
