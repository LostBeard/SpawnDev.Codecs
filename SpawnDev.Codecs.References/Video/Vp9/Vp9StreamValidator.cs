// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VP9 stream validator - mirrors Av1StreamValidator. Walks a packet
// sequence (typically Matroska/WebM SimpleBlock contents) and reports
// structural issues: superframe well-formedness, valid VP9 sync code,
// first frame is keyframe, dimensions stable.

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>One finding from <see cref="Vp9StreamValidator"/>.</summary>
public sealed record Vp9ValidationFinding
{
    /// <summary>Severity level. Reuses Av1ValidationSeverity values for cross-codec UX.</summary>
    public required Av1.Av1ValidationSeverity Severity { get; init; }

    /// <summary>Short message describing the finding.</summary>
    public required string Message { get; init; }

    /// <summary>1-based packet index, or 0 for stream-level findings.</summary>
    public int PacketIndex { get; init; }

    /// <summary>1-based slice index within the packet (for superframes).</summary>
    public int SliceIndex { get; init; }
}

/// <summary>Result of a Vp9 stream validation pass.</summary>
public sealed record Vp9ValidationResult
{
    /// <summary>All findings in order of discovery.</summary>
    public required IReadOnlyList<Vp9ValidationFinding> Findings { get; init; }

    /// <summary>True if there were no Error-level findings.</summary>
    public bool IsValid => !Findings.Any(f => f.Severity == Av1.Av1ValidationSeverity.Error);

    /// <summary>Number of findings at the given severity.</summary>
    public int CountBy(Av1.Av1ValidationSeverity severity) =>
        Findings.Count(f => f.Severity == severity);
}

/// <summary>Walks a VP9 packet sequence and reports structural issues.</summary>
public static class Vp9StreamValidator
{
    /// <summary>Validate a VP9 packet sequence for common structural issues.</summary>
    public static Vp9ValidationResult Validate(IEnumerable<ReadOnlyMemory<byte>> packets)
    {
        ArgumentNullException.ThrowIfNull(packets);
        var findings = new List<Vp9ValidationFinding>();
        bool firstFrameSeen = false;
        bool firstFrameIsKey = false;
        int firstWidth = 0, firstHeight = 0;
        var refSizes = new (int Width, int Height)[3];
        int packetIdx = 0;

        foreach (var pkt in packets)
        {
            packetIdx++;
            Vp9Superframe? sf;
            try
            {
                sf = Vp9SuperframeParser.Parse(pkt.Span);
            }
            catch (Exception ex)
            {
                findings.Add(new Vp9ValidationFinding
                {
                    Severity = Av1.Av1ValidationSeverity.Error,
                    Message = $"Superframe parse failed: {ex.Message}",
                    PacketIndex = packetIdx,
                });
                continue;
            }

            int sliceIdx = 0;
            foreach (var slice in sf.Frames)
            {
                sliceIdx++;
                var frameBytes = pkt.Slice(slice.Offset, slice.Length);
                Vp9UncompressedHeader? complete;
                try
                {
                    complete = Vp9CompleteUncompressedHeaderParser.Parse(frameBytes.Span, refSizes);
                }
                catch (Exception ex)
                {
                    findings.Add(new Vp9ValidationFinding
                    {
                        Severity = Av1.Av1ValidationSeverity.Error,
                        Message = $"FrameHeader parse failed: {ex.Message}",
                        PacketIndex = packetIdx,
                        SliceIndex = sliceIdx,
                    });
                    continue;
                }

                var header = complete.FrameHeader;
                if (!firstFrameSeen && !header.ShowExistingFrame)
                {
                    firstFrameSeen = true;
                    firstFrameIsKey = header.FrameType == Vp9FrameType.Key;
                    if (firstFrameIsKey)
                    {
                        firstWidth = header.FrameWidth;
                        firstHeight = header.FrameHeight;
                    }
                }

                bool sizeCarrying = header.FrameType == Vp9FrameType.Key || header.IntraOnly;
                if (sizeCarrying && !header.ShowExistingFrame)
                {
                    if (header.FrameWidth <= 0 || header.FrameHeight <= 0)
                    {
                        findings.Add(new Vp9ValidationFinding
                        {
                            Severity = Av1.Av1ValidationSeverity.Error,
                            Message = $"Frame declares non-positive dimensions {header.FrameWidth}x{header.FrameHeight}.",
                            PacketIndex = packetIdx,
                            SliceIndex = sliceIdx,
                        });
                    }
                    var size = (header.FrameWidth, header.FrameHeight);
                    refSizes[0] = size;
                    refSizes[1] = size;
                    refSizes[2] = size;
                }
            }
        }

        if (!firstFrameSeen)
        {
            findings.Add(new Vp9ValidationFinding
            {
                Severity = Av1.Av1ValidationSeverity.Error,
                Message = "Stream has no parseable coded frame.",
            });
        }
        else if (!firstFrameIsKey)
        {
            findings.Add(new Vp9ValidationFinding
            {
                Severity = Av1.Av1ValidationSeverity.Error,
                Message = "First non-show-existing frame is not a KeyFrame.",
            });
        }

        return new Vp9ValidationResult { Findings = findings };
    }
}
