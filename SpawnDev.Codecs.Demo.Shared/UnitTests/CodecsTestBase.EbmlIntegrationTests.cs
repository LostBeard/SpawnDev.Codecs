// Integration tests that prove SpawnDev.EBML 3.0.0 is wired into
// SpawnDev.Codecs via PackageReference (with SpawnDev.PatchStreams 1.0.6
// transitively). These replace the earlier hand-rolled thin EBML layer
// that lived at Container/Ebml/* (slices 107-109) and exercise the real
// schema-driven parser that production Codecs demuxers will use.
//
// The tests hand-build synthetic EBML headers so they have no runtime
// dependency on bundled media files, but the code paths exercised here
// are the same ones that will consume real WebM / Matroska streams in
// later slices (Phase 1b WebM demuxer for VP8 / VP9 / AV1 extraction).

using System.IO;
using System.Text;
using SpawnDev.EBML;
using SpawnDev.EBML.Elements;
using SpawnDev.EBML.Schemas;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    // -------- Synthetic EBML header helpers ---------------------------------
    // These build a minimal but spec-valid EBML header that SpawnDev.EBML's
    // schema-driven parser can recognise.

    /// <summary>Encode a value 0..0x7E as a 1-byte stripped VINT size (marker bit 7).</summary>
    private static byte EbmlIntegration_Size1(int n)
    {
        if (n < 0 || n > 0x7E) throw new ArgumentException("out of range for 1-byte VINT");
        return (byte)(0x80 | n);
    }

    /// <summary>Build a child element with a 2-byte ID + 1-byte size + body bytes.</summary>
    private static byte[] EbmlIntegration_TwoByteIdChild(ushort id, byte[] body)
    {
        if (body.Length > 0x7E) throw new ArgumentException("body too big for 1-byte VINT size");
        var bytes = new byte[2 + 1 + body.Length];
        bytes[0] = (byte)(id >> 8);
        bytes[1] = (byte)id;
        bytes[2] = EbmlIntegration_Size1(body.Length);
        Array.Copy(body, 0, bytes, 3, body.Length);
        return bytes;
    }

    private static byte[] EbmlIntegration_UintChild(ushort id, int value)
        => EbmlIntegration_TwoByteIdChild(id, new[] { (byte)value });

    /// <summary>Full EBML header element (ID 0x1A45DFA3) with DocType and version children.</summary>
    private static byte[] EbmlIntegration_BuildHeader(string docType)
    {
        var children = new List<byte>();
        children.AddRange(EbmlIntegration_UintChild(0x4286, 1));                              // EBMLVersion
        children.AddRange(EbmlIntegration_UintChild(0x42F7, 1));                              // EBMLReadVersion
        children.AddRange(EbmlIntegration_UintChild(0x42F2, 4));                              // EBMLMaxIDLength
        children.AddRange(EbmlIntegration_UintChild(0x42F3, 8));                              // EBMLMaxSizeLength
        children.AddRange(EbmlIntegration_TwoByteIdChild(0x4282, Encoding.ASCII.GetBytes(docType))); // DocType
        children.AddRange(EbmlIntegration_UintChild(0x4287, 4));                              // DocTypeVersion
        children.AddRange(EbmlIntegration_UintChild(0x4285, 2));                              // DocTypeReadVersion
        int bodyLen = children.Count;
        var bytes = new List<byte>
        {
            0x1A, 0x45, 0xDF, 0xA3,                 // EBML header element ID
            EbmlIntegration_Size1(bodyLen),         // size VINT (1 byte)
        };
        bytes.AddRange(children);
        return bytes.ToArray();
    }

    // -------- Tests ---------------------------------------------------------

    [TestMethod]
    public void EbmlIntegration_Library_Resolves_FromSpawnDevCodecs()
    {
        // Prove SpawnDev.EBML is accessible through the Codecs project and
        // that its default schemas (ebml + matroska + webm) come loaded.
        var parser = new EBMLParser();
        var keys = parser.Schemas.Keys.ToList();
        True(keys.Count >= 1, "parser must load at least one schema");
        True(keys.Any(k => k.Equals("ebml", System.StringComparison.OrdinalIgnoreCase)),
            $"expected 'ebml' schema, got: {string.Join(",", keys)}");
    }

    [TestMethod]
    public void EbmlIntegration_IsEBML_OnWebMHeader_ReturnsTrue()
    {
        // Feed a hand-built EBML header to the library's IsEBML detector.
        using var stream = new MemoryStream(EbmlIntegration_BuildHeader("webm"));
        var parser = new EBMLParser();
        True(parser.IsEBML(stream), "library must recognise our synthetic EBML header");
    }

    [TestMethod]
    public void EbmlIntegration_IsEBML_OnGarbage_ReturnsFalse()
    {
        using var stream = new MemoryStream(new byte[] { 0, 0, 0, 0, 0 });
        var parser = new EBMLParser();
        False(parser.IsEBML(stream), "garbage must not look like EBML");
    }

    [TestMethod]
    public void EbmlIntegration_ParseDocument_WebMHeader_ReadsDocType()
    {
        // This is the production code path Codecs demuxers will use: hand a
        // Stream to the parser, get back a document, read /EBML/DocType.
        using var stream = new MemoryStream(EbmlIntegration_BuildHeader("webm"));
        var parser = new EBMLParser();
        var doc = parser.ParseDocument(stream);
        True(doc != null, "ParseDocument must succeed on a valid header");
        Equal("webm", doc!.ReadString("/EBML/DocType"));
    }

    [TestMethod]
    public void EbmlIntegration_ParseDocument_MatroskaHeader_ReadsDocType()
    {
        using var stream = new MemoryStream(EbmlIntegration_BuildHeader("matroska"));
        var parser = new EBMLParser();
        var doc = parser.ParseDocument(stream);
        True(doc != null, "ParseDocument must succeed on a valid header");
        Equal("matroska", doc!.ReadString("/EBML/DocType"));
    }

    [TestMethod]
    public void EbmlIntegration_Document_HeaderIsMasterElement()
    {
        using var stream = new MemoryStream(EbmlIntegration_BuildHeader("webm"));
        var parser = new EBMLParser();
        var doc = parser.ParseDocument(stream);
        True(doc != null);
        True(doc!.Header != null, "EBML master element must be navigable");
        var docTypeElem = doc.Header!.First<StringElement>("DocType");
        True(docTypeElem != null, "DocType child must be addressable by name");
        Equal("webm", docTypeElem!.Data);
    }
}
