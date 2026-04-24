// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ISOBMFF / MP4 box tree representation. MP4 is a tree of "boxes" (atoms)
// each with a 4-byte size, a 4-byte type (FourCC), and an optional payload.
// A size of 1 signals a 64-bit extended size stored in the next 8 bytes; a
// size of 0 signals "extends to end of file".
//
// We surface the tree structurally here; specific box parsers (ftyp, moov,
// mvhd, trak, stsd, etc.) layer on top for specific codec / container needs.

namespace SpawnDev.Codecs.Container.Mp4;

/// <summary>One MP4 box: type + byte range within the source buffer + optional child list.</summary>
public sealed record Mp4Box
{
    /// <summary>4-byte FourCC type (e.g. "ftyp", "moov", "mdat"). ASCII.</summary>
    public required string Type { get; init; }

    /// <summary>Absolute byte offset of this box's header in the source buffer.</summary>
    public int Offset { get; init; }

    /// <summary>Total box size in bytes including header.</summary>
    public long Size { get; init; }

    /// <summary>Header size in bytes (8 for 32-bit size, 16 for 64-bit extended size).</summary>
    public int HeaderSize { get; init; }

    /// <summary>
    /// Child boxes, if this box is a container box. Null for leaf boxes (media
    /// data, sample tables with known flat layout).
    /// </summary>
    public IReadOnlyList<Mp4Box>? Children { get; init; }
}

/// <summary>MP4 box tree reader.</summary>
public static class Mp4BoxReader
{
    /// <summary>FourCC codes of container boxes that carry child boxes rather than a flat payload.</summary>
    private static readonly HashSet<string> ContainerBoxes = new()
    {
        "moov", "trak", "edts", "mdia", "minf", "stbl", "moof", "traf",
        "mvex", "mfra", "udta", "dinf", "ipro", "sinf", "schi", "meta",
    };

    /// <summary>Enumerate top-level MP4 boxes in <paramref name="data"/>, recursing into known container boxes.</summary>
    public static IReadOnlyList<Mp4Box> ReadAll(ReadOnlySpan<byte> data)
    {
        var result = new List<Mp4Box>();
        int offset = 0;
        while (offset < data.Length)
        {
            var box = ReadBoxAt(data, offset);
            result.Add(box);
            if (box.Size <= 0)
                break; // "rest of file" box - we've consumed everything we can reasonably enumerate.
            offset = (int)(offset + box.Size);
            if (offset < 0) break; // overflow guard
        }
        return result;
    }

    /// <summary>
    /// Parse the box header at <paramref name="offset"/> and, if it's a known
    /// container box, recursively enumerate its children.
    /// </summary>
    public static Mp4Box ReadBoxAt(ReadOnlySpan<byte> data, int offset)
    {
        if (offset + 8 > data.Length)
            throw new InvalidDataException($"MP4 box header truncated at offset {offset}.");
        long size = ReadUInt32Be(data.Slice(offset, 4));
        string type = System.Text.Encoding.ASCII.GetString(data.Slice(offset + 4, 4));
        int headerSize = 8;
        if (size == 1)
        {
            // 64-bit extended size follows the 8-byte header.
            if (offset + 16 > data.Length)
                throw new InvalidDataException($"MP4 64-bit size field truncated at offset {offset}.");
            size = ReadInt64Be(data.Slice(offset + 8, 8));
            headerSize = 16;
        }
        else if (size == 0)
        {
            // Extends to end of file.
            size = data.Length - offset;
        }
        if (size < headerSize)
            throw new InvalidDataException(
                $"MP4 box size {size} at offset {offset} smaller than header {headerSize}.");
        if (offset + size > data.Length)
            throw new InvalidDataException(
                $"MP4 box '{type}' at offset {offset} size {size} extends past buffer length {data.Length}.");

        IReadOnlyList<Mp4Box>? children = null;
        if (ContainerBoxes.Contains(type))
        {
            int childOffset = offset + headerSize;
            int childEnd = (int)(offset + size);
            var childList = new List<Mp4Box>();
            while (childOffset < childEnd)
            {
                var child = ReadBoxAt(data, childOffset);
                childList.Add(child);
                if (child.Size <= 0) break;
                childOffset += (int)child.Size;
            }
            children = childList;
        }

        return new Mp4Box
        {
            Type = type,
            Offset = offset,
            Size = size,
            HeaderSize = headerSize,
            Children = children,
        };
    }

    /// <summary>
    /// Walk the box tree looking for the first box whose type matches
    /// <paramref name="fourcc"/>. Useful for picking out `ftyp`, `moov`, `mvhd`,
    /// etc. without writing a full path traversal.
    /// </summary>
    public static Mp4Box? FindFirst(IEnumerable<Mp4Box> boxes, string fourcc)
    {
        foreach (var box in boxes)
        {
            if (box.Type == fourcc) return box;
            if (box.Children != null)
            {
                var nested = FindFirst(box.Children, fourcc);
                if (nested != null) return nested;
            }
        }
        return null;
    }

    private static uint ReadUInt32Be(ReadOnlySpan<byte> s)
    {
        uint v = 0;
        for (int i = 0; i < 4; i++) v = (v << 8) | s[i];
        return v;
    }

    private static long ReadInt64Be(ReadOnlySpan<byte> s)
    {
        long v = 0;
        for (int i = 0; i < 8; i++) v = (v << 8) | s[i];
        return v;
    }
}
