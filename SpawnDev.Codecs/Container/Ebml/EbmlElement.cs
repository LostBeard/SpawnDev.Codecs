// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// EBML element = (ID VINT) + (Size VINT) + (Data bytes). Same on-wire shape
// is used by Matroska, WebM, DASH-MPD (in some flavours), and CMAF fragment
// metadata. This reader enumerates elements without interpreting their data,
// which is enough to walk the document tree and pick out specific IDs.

namespace SpawnDev.Codecs.Container.Ebml;

/// <summary>Parsed EBML element header (the metadata before the body bytes).</summary>
public sealed record EbmlElement
{
    /// <summary>
    /// Element ID (VINT with the length marker preserved). For Matroska the
    /// top-level "EBML header" ID is <c>0x1A45DFA3</c>, "Segment" is
    /// <c>0x18538067</c>, and so on.
    /// </summary>
    public required ulong Id { get; init; }

    /// <summary>
    /// Element data size in bytes (VINT with marker stripped), or
    /// <see cref="EbmlVint.UnknownSize"/> for streaming elements with no
    /// declared size.
    /// </summary>
    public required ulong Size { get; init; }

    /// <summary>Absolute byte offset of this element's ID VINT in the source buffer.</summary>
    public int Offset { get; init; }

    /// <summary>Total header byte count (ID VINT + Size VINT).</summary>
    public int HeaderBytes { get; init; }

    /// <summary>Byte offset of the element's data payload.</summary>
    public int DataOffset => Offset + HeaderBytes;
}

/// <summary>Enumerates EBML elements out of a byte buffer.</summary>
public static class EbmlElementReader
{
    /// <summary>Read exactly one element starting at <paramref name="offset"/>.</summary>
    public static EbmlElement ReadAt(ReadOnlySpan<byte> data, int offset)
    {
        ulong id = EbmlVint.ReadId(data, offset, out int idBytes);
        ulong size = EbmlVint.ReadSize(data, offset + idBytes, out int sizeBytes);
        int headerBytes = idBytes + sizeBytes;
        return new EbmlElement
        {
            Id = id,
            Size = size,
            Offset = offset,
            HeaderBytes = headerBytes,
        };
    }

    /// <summary>
    /// Enumerate every element at the top level of <paramref name="data"/>
    /// without recursing into container elements. Stops cleanly at end of
    /// buffer or at an element whose declared size extends past the buffer.
    /// </summary>
    public static IEnumerable<EbmlElement> EnumerateTopLevel(byte[] data)
    {
        int offset = 0;
        while (offset < data.Length)
        {
            EbmlElement el;
            try { el = ReadAt(data, offset); }
            catch (InvalidDataException) { yield break; }
            yield return el;
            if (el.Size == EbmlVint.UnknownSize) yield break; // can't safely advance
            long next = (long)offset + el.HeaderBytes + (long)el.Size;
            if (next > data.Length || next < 0) yield break;
            offset = (int)next;
        }
    }
}
