using SpawnDev.Codecs.Container.Ebml;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="EbmlDocumentHeaderParser"/>. Hand-builds synthetic
/// EBML header bytes and verifies DocType + version fields round-trip.
/// </summary>
public abstract partial class CodecsTestBase
{
    /// <summary>Encode a size-7-or-fewer value as a 1-byte stripped VINT size (marker at bit 7).</summary>
    private static byte EbmlSize1(int n)
    {
        if (n < 0 || n > 0x7E) throw new ArgumentException("out of range for 1-byte VINT");
        return (byte)(0x80 | n);
    }

    /// <summary>Build a child element: 2-byte ID VINT + 1-byte size VINT + body bytes.</summary>
    private static byte[] TwoByteIdChild(ushort id, byte[] body)
    {
        if (body.Length > 0x7E) throw new ArgumentException("body too big for 1-byte VINT size here");
        var bytes = new byte[2 + 1 + body.Length];
        bytes[0] = (byte)(id >> 8);
        bytes[1] = (byte)id;
        bytes[2] = EbmlSize1(body.Length);
        Array.Copy(body, 0, bytes, 3, body.Length);
        return bytes;
    }

    /// <summary>Build a single-byte unsigned-integer child.</summary>
    private static byte[] UintChild(ushort id, int value) => TwoByteIdChild(id, new[] { (byte)value });

    /// <summary>Build a full EBML header element with the given DocType.</summary>
    private static byte[] BuildEbmlHeaderBytes(string docType, int docTypeVersion = 4, int docTypeReadVersion = 2)
    {
        // Children (order matches Matroska/WebM examples).
        var children = new List<byte>();
        children.AddRange(UintChild(0x4286, 1));           // EBMLVersion = 1
        children.AddRange(UintChild(0x42F7, 1));           // EBMLReadVersion = 1
        children.AddRange(UintChild(0x42F2, 4));           // EBMLMaxIDLength = 4
        children.AddRange(UintChild(0x42F3, 8));           // EBMLMaxSizeLength = 8
        children.AddRange(TwoByteIdChild(0x4282,            // DocType
            System.Text.Encoding.ASCII.GetBytes(docType)));
        children.AddRange(UintChild(0x4287, docTypeVersion));
        children.AddRange(UintChild(0x4285, docTypeReadVersion));

        int bodyLen = children.Count;
        // Top element: 4-byte EBML header ID + 1-byte stripped VINT size.
        if (bodyLen > 0x7E) throw new NotSupportedException("body too large for this helper");
        var bytes = new List<byte>();
        bytes.AddRange(new byte[] { 0x1A, 0x45, 0xDF, 0xA3 });
        bytes.Add(EbmlSize1(bodyLen));
        bytes.AddRange(children);
        return bytes.ToArray();
    }

    [TestMethod]
    public void EbmlDocumentHeader_WebMDocType_Detected()
    {
        byte[] bytes = BuildEbmlHeaderBytes("webm");
        var header = EbmlDocumentHeaderParser.Parse(bytes);
        Equal("webm", header.DocType);
        True(header.IsWebM);
        False(header.IsMatroska);
        Equal(4, header.DocTypeVersion);
        Equal(2, header.DocTypeReadVersion);
    }

    [TestMethod]
    public void EbmlDocumentHeader_MatroskaDocType_Detected()
    {
        byte[] bytes = BuildEbmlHeaderBytes("matroska");
        var header = EbmlDocumentHeaderParser.Parse(bytes);
        Equal("matroska", header.DocType);
        True(header.IsMatroska);
        False(header.IsWebM);
    }

    [TestMethod]
    public void EbmlDocumentHeader_VersionFieldsParsed()
    {
        byte[] bytes = BuildEbmlHeaderBytes("webm", docTypeVersion: 7, docTypeReadVersion: 3);
        var header = EbmlDocumentHeaderParser.Parse(bytes);
        Equal(1, header.EbmlVersion);
        Equal(1, header.EbmlReadVersion);
        Equal(4, header.EbmlMaxIdLength);
        Equal(8, header.EbmlMaxSizeLength);
        Equal(7, header.DocTypeVersion);
        Equal(3, header.DocTypeReadVersion);
    }

    [TestMethod]
    public void EbmlDocumentHeader_UnknownChildrenIgnored()
    {
        // Build bytes, then inject an unknown child ID (0x4400) at the end.
        var bytes = BuildEbmlHeaderBytes("webm").ToList();
        // We need to adjust the top-level size to include the new child.
        // For simplicity, rebuild from scratch with an extra unknown child.
        var children = new List<byte>();
        children.AddRange(UintChild(0x4286, 1));
        children.AddRange(UintChild(0x42F7, 1));
        children.AddRange(UintChild(0x42F2, 4));
        children.AddRange(UintChild(0x42F3, 8));
        children.AddRange(TwoByteIdChild(0x4282, System.Text.Encoding.ASCII.GetBytes("webm")));
        children.AddRange(UintChild(0x4287, 4));
        children.AddRange(UintChild(0x4285, 2));
        children.AddRange(UintChild(0x4400, 99));         // unknown child -> should be skipped
        int bodyLen = children.Count;
        var combined = new List<byte> { 0x1A, 0x45, 0xDF, 0xA3, EbmlSize1(bodyLen) };
        combined.AddRange(children);
        var header = EbmlDocumentHeaderParser.Parse(combined.ToArray());
        Equal("webm", header.DocType);
    }

    [TestMethod]
    public void EbmlDocumentHeader_WrongTopLevelId_Throws()
    {
        // Top element ID that is NOT the EBML header.
        var bytes = new byte[] { 0x18, 0x53, 0x80, 0x67, 0x80 };
        bool threw = false;
        try { _ = EbmlDocumentHeaderParser.Parse(bytes); } catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void EbmlDocumentHeader_MissingDocType_Throws()
    {
        // Build a header with only the EBML-version children, no DocType.
        var children = new List<byte>();
        children.AddRange(UintChild(0x4286, 1));
        int bodyLen = children.Count;
        var bytes = new List<byte> { 0x1A, 0x45, 0xDF, 0xA3, EbmlSize1(bodyLen) };
        bytes.AddRange(children);
        bool threw = false;
        try { _ = EbmlDocumentHeaderParser.Parse(bytes.ToArray()); }
        catch (InvalidDataException) { threw = true; }
        True(threw);
    }
}
