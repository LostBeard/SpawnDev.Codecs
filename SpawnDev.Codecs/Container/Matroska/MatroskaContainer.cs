// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Thin facade over SpawnDev.EBML that exposes the pieces of a Matroska /
// WebM document that downstream codec work cares about: doc-type, track
// list, and per-track codec IDs. Block/SimpleBlock frame extraction is a
// separate Phase 1b slice and will live alongside this class.

using SpawnDev.EBML;
using SpawnDev.EBML.Elements;
using SpawnDev.EBML.Schemas;

namespace SpawnDev.Codecs.Container.Matroska;

/// <summary>
/// Thin read-only wrapper around a Matroska (.mkv) or WebM (.webm) document.
/// WebM is a strict Matroska profile, so the same parser handles both -
/// check <see cref="IsWebM"/> / <see cref="IsMatroska"/> to distinguish.
/// </summary>
/// <remarks>
/// Backed by <see cref="SpawnDev.EBML.EBMLDocument"/> from the
/// <c>SpawnDev.EBML</c> package, which in turn sits on
/// <c>SpawnDev.PatchStreams</c>. Construction parses the EBML header only;
/// track enumeration and frame extraction happen lazily on demand.
/// </remarks>
public sealed class MatroskaContainer
{
    private readonly EBMLDocument _doc;

    /// <summary>Parse the container header from <paramref name="stream"/>.</summary>
    /// <param name="stream">Source stream. Not disposed by this class.</param>
    /// <param name="parser">
    /// Optional parser override. If null, a default <see cref="EBMLParser"/>
    /// with the built-in ebml + matroska + webm schemas is used.
    /// </param>
    /// <exception cref="InvalidDataException">Stream is not a valid EBML document.</exception>
    public MatroskaContainer(Stream stream, EBMLParser? parser = null)
    {
        ArgumentNullException.ThrowIfNull(stream);
        parser ??= new EBMLParser();
        var doc = parser.ParseDocument(stream)
            ?? throw new InvalidDataException("Stream is not a valid EBML document.");
        _doc = doc;
    }

    /// <summary>Underlying EBML document (for callers that need path-based access).</summary>
    public EBMLDocument Document => _doc;

    /// <summary>Document type string as declared in /EBML/DocType ("webm" or "matroska").</summary>
    public string? DocType => _doc.DocType;

    /// <summary>True when the document's DocType is exactly "webm".</summary>
    public bool IsWebM => string.Equals(DocType, "webm", StringComparison.Ordinal);

    /// <summary>True when the document's DocType is exactly "matroska".</summary>
    public bool IsMatroska => string.Equals(DocType, "matroska", StringComparison.Ordinal);

    /// <summary>
    /// Snapshot of /Segment/Info. Returns default values if the element is
    /// absent; <see cref="MatroskaSegmentInfo.TimestampScaleNs"/> falls back
    /// to the Matroska spec default of 1,000,000 ns/tick (= 1 ms/tick).
    /// </summary>
    public MatroskaSegmentInfo SegmentInfo
    {
        get
        {
            var info = _doc.First<MasterElement>("/Segment/Info");
            var scale = info?.First<UintElement>("TimestampScale")?.Data ?? 1_000_000UL;
            var duration = info?.First<FloatElement>("Duration")?.Data;
            return new MatroskaSegmentInfo
            {
                TimestampScaleNs = scale,
                DurationTicks = duration,
                Title = info?.First<StringElement>("Title")?.Data,
                MuxingApp = info?.First<StringElement>("MuxingApp")?.Data,
                WritingApp = info?.First<StringElement>("WritingApp")?.Data,
            };
        }
    }

    /// <summary>
    /// Enumerate every TrackEntry under /Segment/Tracks with its
    /// TrackNumber + TrackType + CodecID. Consumers route frames to the
    /// matching codec by <see cref="MatroskaTrack.CodecId"/>.
    /// </summary>
    public IEnumerable<MatroskaTrack> Tracks
    {
        get
        {
            // /Segment is the Matroska body root; /Segment/Tracks holds the
            // TrackEntry list. Either may be absent in pathological files -
            // return nothing in that case rather than throwing so callers
            // can safely walk any input stream.
            var tracksMaster = _doc.First<MasterElement>("/Segment/Tracks");
            if (tracksMaster is null) yield break;
            foreach (var entry in tracksMaster.Find<MasterElement>("TrackEntry"))
            {
                var trackNumber = entry.First<UintElement>("TrackNumber")?.Data;
                var trackType = entry.First<UintElement>("TrackType")?.Data;
                var codecId = entry.First<StringElement>("CodecID")?.Data;
                // A TrackEntry without a codec ID is malformed - skip quietly.
                if (trackNumber is null || trackType is null || codecId is null) continue;
                yield return new MatroskaTrack
                {
                    TrackNumber = trackNumber.Value,
                    TrackType = trackType.Value,
                    CodecId = codecId,
                };
            }
        }
    }

    /// <summary>
    /// Enumerate every codec frame in the document in cluster + block order.
    /// Yields one <see cref="MatroskaFrame"/> per SimpleBlock / Block
    /// (or per laced sub-frame when the block uses Xiph / fixed / EBML
    /// lacing). Timestamps are resolved against each enclosing cluster's
    /// Timestamp child.
    /// </summary>
    /// <remarks>
    /// This is the core demux call for downstream codec decoders. Filter
    /// by <see cref="MatroskaFrame.TrackNumber"/> to route packets to the
    /// right codec. Units of the <see cref="MatroskaFrame.Timestamp"/>
    /// field are Matroska ticks - multiply by /Segment/Info/TimestampScale
    /// (ns per tick; 1_000_000 in most WebM files so ticks = ms) to get
    /// nanoseconds.
    /// </remarks>
    public IEnumerable<MatroskaFrame> Frames
    {
        get
        {
            foreach (var cluster in _doc.Find<MasterElement>("/Segment/Cluster"))
            {
                long clusterTs = (long)(cluster.First<UintElement>("Timestamp")?.Data ?? 0UL);
                foreach (var child in cluster.Children)
                {
                    // SimpleBlock (id 0xA3) and BlockGroup/Block (id 0xA1)
                    // are the two places frame bytes live. We dispatch on
                    // element ID rather than name so a future alias doesn't
                    // silently break the walk.
                    if (child.Id == 0xA3UL && child is BinaryElement simple)
                    {
                        foreach (var f in ParseBlockElement(simple, clusterTs, isSimpleBlock: true))
                            yield return f;
                    }
                    else if (child.Id == 0xA0UL && child is MasterElement blockGroup)
                    {
                        var block = blockGroup.First<BinaryElement>("Block");
                        if (block is null) continue;
                        foreach (var f in ParseBlockElement(block, clusterTs, isSimpleBlock: false))
                            yield return f;
                    }
                }
            }
        }
    }

    /// <summary>Read the binary element's bytes and hand them to the parser.</summary>
    private static IEnumerable<MatroskaFrame> ParseBlockElement(
        BinaryElement element, long clusterTimestamp, bool isSimpleBlock)
    {
        // BinaryElement.Data is a PatchStream positioned at 0; ToArray
        // reads the full element body (without the ID+size VINT header).
        var dataStream = element.Data;
        var buf = new byte[dataStream.Length];
        dataStream.Position = 0;
        int read = 0;
        while (read < buf.Length)
        {
            int got = dataStream.Read(buf, read, buf.Length - read);
            if (got <= 0) break;
            read += got;
        }
        return MatroskaBlockParser.Parse(buf.AsSpan(0, read), clusterTimestamp, isSimpleBlock);
    }
}
