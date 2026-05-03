// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// FLAC SEEKTABLE metadata block (type 3) per RFC 9639 Section 8.3. Each
// seek point is 18 bytes: 8-byte sample number + 8-byte frame byte offset
// (from the first frame) + 2-byte frame length in samples.

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// One entry in a FLAC SEEKTABLE. <c>SampleNumber</c> is the first sample in
/// the target frame (or <see cref="PlaceholderSampleNumber"/> for an unused
/// entry); <c>StreamOffset</c> is the frame's byte offset from the start of
/// audio data (not from the file start); <c>FrameSamples</c> is the number of
/// samples per channel in that frame.
/// </summary>
public readonly record struct FlacSeekPoint(ulong SampleNumber, ulong StreamOffset, ushort FrameSamples)
{
    /// <summary>Well-known placeholder value used for unused seek points.</summary>
    public static readonly ulong PlaceholderSampleNumber = ulong.MaxValue;

    /// <summary>True when this entry is a placeholder rather than a real seek point.</summary>
    public bool IsPlaceholder => SampleNumber == PlaceholderSampleNumber;
}

/// <summary>Parsed FLAC SEEKTABLE metadata block.</summary>
public sealed record FlacSeekTable
{
    /// <summary>Seek points in the order they appeared in the block.</summary>
    public required FlacSeekPoint[] Points { get; init; }
}

/// <summary>Parser for the SEEKTABLE metadata block.</summary>
public static class FlacSeekTableParser
{
    /// <summary>
    /// Parse a SEEKTABLE payload. The block length must be a multiple of 18
    /// bytes; anything else is a format error per RFC 9639.
    /// </summary>
    public static FlacSeekTable Parse(ReadOnlySpan<byte> payload)
    {
        if (payload.Length % 18 != 0)
            throw new InvalidDataException(
                $"FLAC SEEKTABLE length {payload.Length} must be a multiple of 18.");
        int count = payload.Length / 18;
        var points = new FlacSeekPoint[count];
        for (int i = 0; i < count; i++)
        {
            int offset = i * 18;
            ulong sampleNumber = ReadUInt64Be(payload.Slice(offset, 8));
            ulong streamOffset = ReadUInt64Be(payload.Slice(offset + 8, 8));
            ushort frameSamples = ReadUInt16Be(payload.Slice(offset + 16, 2));
            points[i] = new FlacSeekPoint(sampleNumber, streamOffset, frameSamples);
        }
        return new FlacSeekTable { Points = points };
    }

    /// <summary>
    /// Find the seek point whose <see cref="FlacSeekPoint.SampleNumber"/> is
    /// the largest value not exceeding <paramref name="targetSample"/>, or
    /// <c>null</c> if no non-placeholder point qualifies.
    /// </summary>
    public static FlacSeekPoint? FindNearest(FlacSeekTable table, ulong targetSample)
    {
        FlacSeekPoint? best = null;
        for (int i = 0; i < table.Points.Length; i++)
        {
            var p = table.Points[i];
            if (p.IsPlaceholder) continue;
            if (p.SampleNumber <= targetSample && (!best.HasValue || p.SampleNumber > best.Value.SampleNumber))
                best = p;
        }
        return best;
    }

    private static ushort ReadUInt16Be(ReadOnlySpan<byte> s) => (ushort)((s[0] << 8) | s[1]);

    private static ulong ReadUInt64Be(ReadOnlySpan<byte> s)
    {
        ulong v = 0;
        for (int i = 0; i < 8; i++) v = (v << 8) | s[i];
        return v;
    }
}
