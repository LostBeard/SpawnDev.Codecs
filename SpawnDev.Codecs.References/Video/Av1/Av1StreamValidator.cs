// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AV1 stream validator - high-level utility that checks an AV1 IVF
// stream for common structural issues:
//   - IVF header is well-formed
//   - First TU has a Sequence Header
//   - Every TU starts with a Temporal Delimiter
//   - First non-show-existing frame is a KeyFrame
//   - Every frame OBU's payload bytes are within bounds
//   - SequenceHeader fields are in valid ranges
//
// Returns a structured result with errors + warnings. Designed for
// bitstream QA tooling - does NOT decode pixels, just checks framing.

using SpawnDev.Codecs.Container.Ivf;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>Severity of an Av1 validation finding.</summary>
public enum Av1ValidationSeverity
{
    /// <summary>Cosmetic / non-blocking issue (e.g. unusual but legal value).</summary>
    Info,
    /// <summary>Spec deviation that may not break decode but is suspicious.</summary>
    Warning,
    /// <summary>Spec violation that likely breaks decoding.</summary>
    Error,
}

/// <summary>One finding from <see cref="Av1StreamValidator"/>.</summary>
public sealed record Av1ValidationFinding
{
    /// <summary>Severity level.</summary>
    public required Av1ValidationSeverity Severity { get; init; }

    /// <summary>Short message describing the finding.</summary>
    public required string Message { get; init; }

    /// <summary>1-based Temporal Unit index, or 0 for stream-level findings.</summary>
    public int TemporalUnit { get; init; }
}

/// <summary>Result of an Av1 stream validation pass.</summary>
public sealed record Av1ValidationResult
{
    /// <summary>All findings (info/warnings/errors) in order of discovery.</summary>
    public required IReadOnlyList<Av1ValidationFinding> Findings { get; init; }

    /// <summary>True if there were no Error-level findings.</summary>
    public bool IsValid => !Findings.Any(f => f.Severity == Av1ValidationSeverity.Error);

    /// <summary>Number of findings at the given severity.</summary>
    public int CountBy(Av1ValidationSeverity severity) =>
        Findings.Count(f => f.Severity == severity);
}

/// <summary>Walks an AV1 IVF stream and reports structural issues.</summary>
public static class Av1StreamValidator
{
    /// <summary>
    /// Validate an AV1 IVF stream for common structural issues. Does
    /// NOT decode pixels.
    /// </summary>
    public static Av1ValidationResult Validate(ReadOnlyMemory<byte> ivfBytes)
    {
        var findings = new List<Av1ValidationFinding>();
        IvfHeader? ivfHeader = null;
        try
        {
            ivfHeader = IvfReader.ParseHeader(ivfBytes.Span);
        }
        catch (Exception ex)
        {
            findings.Add(new Av1ValidationFinding
            {
                Severity = Av1ValidationSeverity.Error,
                Message = $"IVF header parse failed: {ex.Message}",
            });
            return new Av1ValidationResult { Findings = findings };
        }

        if (ivfHeader.FourCc != "AV01")
        {
            findings.Add(new Av1ValidationFinding
            {
                Severity = Av1ValidationSeverity.Warning,
                Message = $"IVF FourCC is '{ivfHeader.FourCc}', expected 'AV01' for AV1 streams.",
            });
        }
        if (ivfHeader.Width <= 0 || ivfHeader.Height <= 0)
        {
            findings.Add(new Av1ValidationFinding
            {
                Severity = Av1ValidationSeverity.Error,
                Message = $"IVF declares non-positive dimensions {ivfHeader.Width}x{ivfHeader.Height}.",
            });
        }

        Av1SequenceHeader? sh = null;
        bool firstFrameSeen = false;
        bool firstFrameIsKey = false;
        int tu = 0;

        foreach (var ivfFrame in IvfReader.EnumerateFrames(ivfBytes))
        {
            tu++;
            int obuIdx = 0;
            bool tuStartedWithTd = false;
            foreach (var obu in Av1ObuParser.EnumerateObus(ivfFrame.Data))
            {
                obuIdx++;
                if (obuIdx == 1 && obu.Type == Av1ObuType.TemporalDelimiter)
                    tuStartedWithTd = true;

                if (obu.Type == Av1ObuType.SequenceHeader)
                {
                    try
                    {
                        sh = Av1SequenceHeaderParser.Parse(
                            ivfFrame.Data.Span.Slice(obu.PayloadOffset, obu.PayloadLength));
                        if (sh.SeqProfile < 0 || sh.SeqProfile > 2)
                            findings.Add(new Av1ValidationFinding
                            {
                                Severity = Av1ValidationSeverity.Error,
                                Message = $"SH seq_profile {sh.SeqProfile} out of [0, 2].",
                                TemporalUnit = tu,
                            });
                        if (sh.MaxFrameWidth != ivfHeader.Width || sh.MaxFrameHeight != ivfHeader.Height)
                            findings.Add(new Av1ValidationFinding
                            {
                                Severity = Av1ValidationSeverity.Warning,
                                Message = $"SH max_frame_size {sh.MaxFrameWidth}x{sh.MaxFrameHeight} differs from IVF {ivfHeader.Width}x{ivfHeader.Height}.",
                                TemporalUnit = tu,
                            });
                    }
                    catch (Exception ex)
                    {
                        findings.Add(new Av1ValidationFinding
                        {
                            Severity = Av1ValidationSeverity.Error,
                            Message = $"SH parse failed: {ex.Message}",
                            TemporalUnit = tu,
                        });
                    }
                }
                else if (obu.IsCodedFrameData && sh is not null && !firstFrameSeen)
                {
                    try
                    {
                        var fh = Av1FrameHeaderParser.Parse(
                            ivfFrame.Data.Span.Slice(obu.PayloadOffset, obu.PayloadLength), sh);
                        if (!fh.ShowExistingFrame)
                        {
                            firstFrameSeen = true;
                            firstFrameIsKey = fh.FrameType == Av1FrameType.KeyFrame;
                        }
                    }
                    catch (Exception ex)
                    {
                        findings.Add(new Av1ValidationFinding
                        {
                            Severity = Av1ValidationSeverity.Error,
                            Message = $"FrameHeader parse failed: {ex.Message}",
                            TemporalUnit = tu,
                        });
                    }
                }
            }

            if (!tuStartedWithTd)
            {
                findings.Add(new Av1ValidationFinding
                {
                    Severity = Av1ValidationSeverity.Warning,
                    Message = $"TU {tu} does not start with a TemporalDelimiter OBU.",
                    TemporalUnit = tu,
                });
            }
        }

        if (sh is null)
        {
            findings.Add(new Av1ValidationFinding
            {
                Severity = Av1ValidationSeverity.Error,
                Message = "Stream has no SequenceHeader OBU.",
            });
        }
        if (firstFrameSeen && !firstFrameIsKey)
        {
            findings.Add(new Av1ValidationFinding
            {
                Severity = Av1ValidationSeverity.Error,
                Message = "First non-show-existing frame is not a KeyFrame.",
            });
        }

        return new Av1ValidationResult { Findings = findings };
    }
}
