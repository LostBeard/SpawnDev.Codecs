// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Parse the EBML header element (ID 0x1A45DFA3) that introduces every
// Matroska / WebM file. The header's children identify the document type -
// most notably "matroska" vs "webm" - plus the EBML version quartet.

namespace SpawnDev.Codecs.Container.Ebml;

/// <summary>Parsed EBML header.</summary>
public sealed record EbmlDocumentHeader
{
    /// <summary>EBML specification version (default 1).</summary>
    public int EbmlVersion { get; init; } = 1;

    /// <summary>EBML spec read version supported by this document.</summary>
    public int EbmlReadVersion { get; init; } = 1;

    /// <summary>Maximum VINT length used for element IDs (default 4).</summary>
    public int EbmlMaxIdLength { get; init; } = 4;

    /// <summary>Maximum VINT length used for element sizes (default 8).</summary>
    public int EbmlMaxSizeLength { get; init; } = 8;

    /// <summary>Document type string ("matroska", "webm", ...).</summary>
    public required string DocType { get; init; }

    /// <summary>Document-type version.</summary>
    public int DocTypeVersion { get; init; } = 1;

    /// <summary>Document-type read version.</summary>
    public int DocTypeReadVersion { get; init; } = 1;

    /// <summary>True when <see cref="DocType"/> is exactly "webm".</summary>
    public bool IsWebM => DocType == "webm";

    /// <summary>True when <see cref="DocType"/> is "matroska".</summary>
    public bool IsMatroska => DocType == "matroska";
}

/// <summary>Parser for the EBML header element.</summary>
public static class EbmlDocumentHeaderParser
{
    /// <summary>Top-level EBML header element ID.</summary>
    public const ulong EbmlHeaderId = 0x1A45DFA3;

    // Child IDs inside the EBML header.
    private const ulong EBMLVersion         = 0x4286;
    private const ulong EBMLReadVersion     = 0x42F7;
    private const ulong EBMLMaxIDLength     = 0x42F2;
    private const ulong EBMLMaxSizeLength   = 0x42F3;
    private const ulong DocType             = 0x4282;
    private const ulong DocTypeVersion      = 0x4287;
    private const ulong DocTypeReadVersion  = 0x4285;

    /// <summary>
    /// Parse the EBML header starting at offset 0 of <paramref name="data"/>.
    /// <paramref name="data"/> must begin with the EBML header element ID VINT.
    /// </summary>
    public static EbmlDocumentHeader Parse(ReadOnlySpan<byte> data)
    {
        var top = EbmlElementReader.ReadAt(data, 0);
        if (top.Id != EbmlHeaderId)
            throw new InvalidDataException(
                $"Expected EBML header ID 0x{EbmlHeaderId:X}, got 0x{top.Id:X}.");
        if (top.Size == EbmlVint.UnknownSize)
            throw new InvalidDataException("EBML header with unknown size is not supported.");
        int dataOffset = top.DataOffset;
        int dataEnd = (int)(dataOffset + (long)top.Size);
        if (dataEnd > data.Length)
            throw new InvalidDataException("EBML header extends past buffer.");

        int version = 1, readVersion = 1;
        int maxIdLen = 4, maxSizeLen = 8;
        string docType = "";
        int docTypeVersion = 1, docTypeReadVersion = 1;

        int pos = dataOffset;
        while (pos < dataEnd)
        {
            var child = EbmlElementReader.ReadAt(data, pos);
            int childDataOffset = child.DataOffset;
            int childDataEnd = (int)(childDataOffset + (long)child.Size);
            if (child.Size == EbmlVint.UnknownSize || childDataEnd > dataEnd)
                throw new InvalidDataException($"EBML header child 0x{child.Id:X} malformed.");
            switch (child.Id)
            {
                case EBMLVersion: version = ReadUintChild(data.Slice(childDataOffset, (int)child.Size)); break;
                case EBMLReadVersion: readVersion = ReadUintChild(data.Slice(childDataOffset, (int)child.Size)); break;
                case EBMLMaxIDLength: maxIdLen = ReadUintChild(data.Slice(childDataOffset, (int)child.Size)); break;
                case EBMLMaxSizeLength: maxSizeLen = ReadUintChild(data.Slice(childDataOffset, (int)child.Size)); break;
                case DocType: docType = System.Text.Encoding.ASCII.GetString(data.Slice(childDataOffset, (int)child.Size)); break;
                case DocTypeVersion: docTypeVersion = ReadUintChild(data.Slice(childDataOffset, (int)child.Size)); break;
                case DocTypeReadVersion: docTypeReadVersion = ReadUintChild(data.Slice(childDataOffset, (int)child.Size)); break;
                default: break; // unknown children allowed by spec; skip.
            }
            pos = childDataEnd;
        }

        if (string.IsNullOrEmpty(docType))
            throw new InvalidDataException("EBML header missing DocType child.");

        return new EbmlDocumentHeader
        {
            EbmlVersion = version,
            EbmlReadVersion = readVersion,
            EbmlMaxIdLength = maxIdLen,
            EbmlMaxSizeLength = maxSizeLen,
            DocType = docType,
            DocTypeVersion = docTypeVersion,
            DocTypeReadVersion = docTypeReadVersion,
        };
    }

    /// <summary>Read an EBML unsigned-integer payload (1 to 8 bytes, big-endian).</summary>
    private static int ReadUintChild(ReadOnlySpan<byte> bytes)
    {
        if (bytes.Length == 0) return 0;
        if (bytes.Length > 4)
            throw new InvalidDataException($"EBML unsigned integer > 4 bytes ({bytes.Length}) - not supported here.");
        int v = 0;
        for (int i = 0; i < bytes.Length; i++) v = (v << 8) | bytes[i];
        return v;
    }
}
