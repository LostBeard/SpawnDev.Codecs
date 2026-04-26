// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 stream analyzer - high-level utility that walks an entire AV1
// IVF stream and produces a structured summary: per-stream sequence
// header, per-frame header timeline, OBU type distribution, and
// frame-type histogram.
//
// Useful for codec developers debugging GOP structure, verifying
// bitstream output, or building tooling that needs to introspect
// AV1 streams without writing the parse loop themselves.

using SpawnDev.Codecs.Container.Ivf;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>One coded frame's metadata in an <see cref="Av1StreamSummary"/>.</summary>
public sealed record Av1FrameSummary
{
    /// <summary>1-based Temporal Unit (IVF frame) index.</summary>
    public required int TemporalUnit { get; init; }

    /// <summary>1-based index within this Temporal Unit (frame headers per TU).</summary>
    public required int IndexInTu { get; init; }

    /// <summary>Parsed frame header.</summary>
    public required Av1FrameHeader Header { get; init; }

    /// <summary>Source IVF presentation timestamp.</summary>
    public required long Pts { get; init; }
}

/// <summary>Summary of an entire AV1 IVF stream produced by <see cref="Av1StreamAnalyzer"/>.</summary>
public sealed record Av1StreamSummary
{
    /// <summary>IVF container header.</summary>
    public required IvfHeader IvfHeader { get; init; }

    /// <summary>Sequence header from the stream's first SH OBU. Null only if no SH was found.</summary>
    public Av1SequenceHeader? SequenceHeader { get; init; }

    /// <summary>OBU count distribution across the whole stream.</summary>
    public required IReadOnlyDictionary<Av1ObuType, int> ObuCounts { get; init; }

    /// <summary>Per-coded-frame summary (excludes show_existing_frame entries).</summary>
    public required IReadOnlyList<Av1FrameSummary> CodedFrames { get; init; }

    /// <summary>Per-show-existing-frame summary.</summary>
    public required IReadOnlyList<Av1FrameSummary> ShowExistingFrames { get; init; }

    /// <summary>Frame-type histogram across coded frames.</summary>
    public required IReadOnlyDictionary<Av1FrameType, int> FrameTypeCounts { get; init; }

    /// <summary>Total Temporal Units (IVF frames) walked.</summary>
    public required int TotalTemporalUnits { get; init; }
}

/// <summary>Walks an AV1 IVF stream and produces a structured summary.</summary>
public static class Av1StreamAnalyzer
{
    /// <summary>
    /// Analyze every OBU in <paramref name="ivfBytes"/> and return a
    /// summary. The first SequenceHeader OBU establishes the stream
    /// context; subsequent SH OBUs (if any) are accepted as updates.
    /// </summary>
    public static Av1StreamSummary Analyze(ReadOnlyMemory<byte> ivfBytes)
    {
        var ivfHeader = IvfReader.ParseHeader(ivfBytes.Span);
        var obuCounts = new Dictionary<Av1ObuType, int>();
        var codedFrames = new List<Av1FrameSummary>();
        var showExisting = new List<Av1FrameSummary>();
        var frameTypeCounts = new Dictionary<Av1FrameType, int>();
        Av1SequenceHeader? sh = null;
        int tu = 0;

        foreach (var ivfFrame in IvfReader.EnumerateFrames(ivfBytes))
        {
            tu++;
            int idxInTu = 0;
            foreach (var obu in Av1ObuParser.EnumerateObus(ivfFrame.Data))
            {
                obuCounts.TryGetValue(obu.Type, out int oc);
                obuCounts[obu.Type] = oc + 1;

                if (obu.Type == Av1ObuType.SequenceHeader)
                {
                    sh = Av1SequenceHeaderParser.Parse(
                        ivfFrame.Data.Span.Slice(obu.PayloadOffset, obu.PayloadLength));
                }
                else if (obu.IsCodedFrameData && sh is not null)
                {
                    var fh = Av1FrameHeaderParser.Parse(
                        ivfFrame.Data.Span.Slice(obu.PayloadOffset, obu.PayloadLength), sh);
                    idxInTu++;
                    var summary = new Av1FrameSummary
                    {
                        TemporalUnit = tu,
                        IndexInTu = idxInTu,
                        Header = fh,
                        Pts = ivfFrame.Pts,
                    };
                    if (fh.ShowExistingFrame)
                    {
                        showExisting.Add(summary);
                    }
                    else
                    {
                        codedFrames.Add(summary);
                        frameTypeCounts.TryGetValue(fh.FrameType, out int fc);
                        frameTypeCounts[fh.FrameType] = fc + 1;
                    }
                }
            }
        }

        return new Av1StreamSummary
        {
            IvfHeader = ivfHeader,
            SequenceHeader = sh,
            ObuCounts = obuCounts,
            CodedFrames = codedFrames,
            ShowExistingFrames = showExisting,
            FrameTypeCounts = frameTypeCounts,
            TotalTemporalUnits = tu,
        };
    }
}
