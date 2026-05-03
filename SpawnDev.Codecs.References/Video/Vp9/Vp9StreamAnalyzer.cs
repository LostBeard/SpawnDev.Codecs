// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 stream analyzer - high-level utility that walks an entire VP9
// stream (Matroska/WebM frames or raw IVF) and produces a structured
// summary: per-frame headers, frame-type histogram, dimensions
// timeline. Mirrors the Av1StreamAnalyzer surface for consumer parity.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>One coded VP9 frame's metadata in a <see cref="Vp9StreamSummary"/>.</summary>
public sealed record Vp9FrameSummary
{
    /// <summary>1-based packet index in the input stream.</summary>
    public required int PacketIndex { get; init; }

    /// <summary>1-based slice index within the packet (superframes can carry multiple).</summary>
    public required int SliceIndex { get; init; }

    /// <summary>Parsed frame header.</summary>
    public required Vp9FrameHeader Header { get; init; }

    /// <summary>Compressed-header probability-update result, when one was parsed (null for show_existing).</summary>
    public Vp9CompressedHeaderResult? CompressedResult { get; init; }
}

/// <summary>Summary of a VP9 byte-stream produced by <see cref="Vp9StreamAnalyzer"/>.</summary>
public sealed record Vp9StreamSummary
{
    /// <summary>Per-coded-frame summary (excludes show_existing entries).</summary>
    public required IReadOnlyList<Vp9FrameSummary> CodedFrames { get; init; }

    /// <summary>Per-show-existing-frame summary.</summary>
    public required IReadOnlyList<Vp9FrameSummary> ShowExistingFrames { get; init; }

    /// <summary>Frame-type histogram across coded frames.</summary>
    public required IReadOnlyDictionary<Vp9FrameType, int> FrameTypeCounts { get; init; }

    /// <summary>Distinct (width, height) pairs observed at keyframes / intra-only.</summary>
    public required IReadOnlyList<(int Width, int Height)> SizeChanges { get; init; }

    /// <summary>Total compressed packets walked.</summary>
    public required int TotalPackets { get; init; }

    /// <summary>Total superframe slices walked (>= TotalPackets when superframes carry multiple).</summary>
    public required int TotalSlices { get; init; }

    /// <summary>Format the summary as a human-readable multi-line report.</summary>
    public string ToReport()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"VP9 stream:");
        sb.AppendLine($"  Total packets: {TotalPackets}, total slices: {TotalSlices}");
        sb.AppendLine($"  Coded frames: {CodedFrames.Count}, ShowExisting: {ShowExistingFrames.Count}");
        sb.AppendLine($"  Frame types: " + string.Join(", ",
            FrameTypeCounts.OrderByDescending(kv => kv.Value).Select(kv => $"{kv.Key}={kv.Value}")));
        sb.Append($"  Size changes: " + string.Join(", ",
            SizeChanges.Select(s => $"{s.Width}x{s.Height}")));
        return sb.ToString();
    }
}

/// <summary>Walks a sequence of VP9 frame packets and produces a structured summary.</summary>
public static class Vp9StreamAnalyzer
{
    /// <summary>
    /// Analyze every packet in <paramref name="packets"/> (typically the
    /// SimpleBlock contents of a Matroska/WebM video track) and return a
    /// summary. Each packet may be a VP9 superframe carrying multiple
    /// frame slices; analysis walks every slice.
    /// </summary>
    public static Vp9StreamSummary Analyze(IEnumerable<ReadOnlyMemory<byte>> packets)
    {
        ArgumentNullException.ThrowIfNull(packets);
        var coded = new List<Vp9FrameSummary>();
        var showExisting = new List<Vp9FrameSummary>();
        var frameTypeCounts = new Dictionary<Vp9FrameType, int>();
        var sizeChanges = new List<(int, int)>();
        (int W, int H)? lastSize = null;
        int packetIdx = 0;
        int totalSlices = 0;

        // Reference frame size pool to track keyframe-driven size updates.
        var refFrameSizes = new (int Width, int Height)[3];
        var compState = new Vp9CompressedHeaderState();

        foreach (var pkt in packets)
        {
            packetIdx++;
            var superframe = Vp9SuperframeParser.Parse(pkt.Span);
            int sliceIdx = 0;
            foreach (var slice in superframe.Frames)
            {
                sliceIdx++;
                totalSlices++;
                var frameBytes = pkt.Slice(slice.Offset, slice.Length);
                var complete = Vp9CompleteUncompressedHeaderParser.Parse(
                    frameBytes.Span, refFrameSizes);
                var header = complete.FrameHeader;

                bool sizeCarrying = header.FrameType == Vp9FrameType.Key || header.IntraOnly;
                if (sizeCarrying && !header.ShowExistingFrame)
                {
                    var size = (header.FrameWidth, header.FrameHeight);
                    if (lastSize is null || lastSize.Value.W != size.FrameWidth || lastSize.Value.H != size.FrameHeight)
                    {
                        sizeChanges.Add(size);
                        lastSize = (size.FrameWidth, size.FrameHeight);
                    }
                    refFrameSizes[0] = size;
                    refFrameSizes[1] = size;
                    refFrameSizes[2] = size;
                }

                Vp9CompressedHeaderResult? compRes = null;
                if (!header.ShowExistingFrame && complete.FirstPartitionSize > 0)
                {
                    if (header.FrameType == Vp9FrameType.Key || header.IntraOnly)
                        compState = new Vp9CompressedHeaderState();
                    var inputs = new Vp9CompressedHeaderInputs(
                        IsLossless: complete.Quantization.BaseQIndex == 0
                            && complete.Quantization.YDcDeltaQ == 0
                            && complete.Quantization.UvDcDeltaQ == 0
                            && complete.Quantization.UvAcDeltaQ == 0,
                        IsIntraOnly: header.FrameType == Vp9FrameType.Key || header.IntraOnly,
                        InterpFilter: complete.InterpFilter,
                        AllowHighPrecisionMv: complete.AllowHighPrecisionMv,
                        SignBiasLast: complete.RefFrameSignBias is { } sb && sb.Length > 0 && sb[0],
                        SignBiasGolden: complete.RefFrameSignBias is { } sb2 && sb2.Length > 1 && sb2[1],
                        SignBiasAltRef: complete.RefFrameSignBias is { } sb3 && sb3.Length > 2 && sb3[2]);
                    var headerBytes = frameBytes.Span.Slice(
                        complete.UncompressedHeaderSizeBytes,
                        complete.FirstPartitionSize).ToArray();
                    var reader = new Vp9BoolDecoder(headerBytes, 0, headerBytes.Length);
                    compRes = Vp9CompressedHeaderParser.Read(compState, inputs, reader);
                }

                var summary = new Vp9FrameSummary
                {
                    PacketIndex = packetIdx,
                    SliceIndex = sliceIdx,
                    Header = header,
                    CompressedResult = compRes,
                };
                if (header.ShowExistingFrame)
                {
                    showExisting.Add(summary);
                }
                else
                {
                    coded.Add(summary);
                    frameTypeCounts.TryGetValue(header.FrameType, out int fc);
                    frameTypeCounts[header.FrameType] = fc + 1;
                }
            }
        }

        return new Vp9StreamSummary
        {
            CodedFrames = coded,
            ShowExistingFrames = showExisting,
            FrameTypeCounts = frameTypeCounts,
            SizeChanges = sizeChanges,
            TotalPackets = packetIdx,
            TotalSlices = totalSlices,
        };
    }
}
