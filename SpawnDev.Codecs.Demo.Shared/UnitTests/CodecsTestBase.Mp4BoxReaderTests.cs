using SpawnDev.Codecs.Container.Mp4;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="Mp4BoxReader"/>. Hand-builds MP4 box trees and
/// verifies the structural parse including nested container boxes, 64-bit
/// extended sizes, and the size=0 "rest-of-file" convention.
/// </summary>
public abstract partial class CodecsTestBase
{
    /// <summary>Build an MP4 box header (no payload): 4-byte BE size + 4-byte ASCII type.</summary>
    private static byte[] Mp4BoxHeader(string type, uint size)
    {
        if (type.Length != 4) throw new ArgumentException("type must be 4 ASCII characters", nameof(type));
        var bytes = new byte[8];
        bytes[0] = (byte)(size >> 24);
        bytes[1] = (byte)(size >> 16);
        bytes[2] = (byte)(size >> 8);
        bytes[3] = (byte)size;
        for (int i = 0; i < 4; i++) bytes[4 + i] = (byte)type[i];
        return bytes;
    }

    [TestMethod]
    public void Mp4Box_FlatLeafBoxes_Enumerated()
    {
        // ftyp (16 bytes: header + 8 bytes payload) followed by free (8-byte header only).
        var stream = new List<byte>();
        stream.AddRange(Mp4BoxHeader("ftyp", 16));
        stream.AddRange(new byte[] { (byte)'i', (byte)'s', (byte)'o', (byte)'m', 0, 0, 0, 0 });
        stream.AddRange(Mp4BoxHeader("free", 8));
        var boxes = Mp4BoxReader.ReadAll(stream.ToArray());
        Equal(2, boxes.Count);
        Equal("ftyp", boxes[0].Type);
        Equal(0, boxes[0].Offset);
        Equal(16L, boxes[0].Size);
        Equal(8, boxes[0].HeaderSize);
        True(boxes[0].Children is null, "ftyp is not a container box.");
        Equal("free", boxes[1].Type);
        Equal(16, boxes[1].Offset);
    }

    [TestMethod]
    public void Mp4Box_ContainerBox_RecursesIntoChildren()
    {
        // moov (container) { mvhd (8 bytes, empty leaf), trak (container) { tkhd (8 bytes) } }
        // moov outer size = 8 (moov hdr) + 8 (mvhd) + 8 (trak hdr) + 8 (tkhd) = 32.
        // trak inner size = 8 (hdr) + 8 (tkhd) = 16.
        var stream = new List<byte>();
        stream.AddRange(Mp4BoxHeader("moov", 32));
        stream.AddRange(Mp4BoxHeader("mvhd", 8));
        stream.AddRange(Mp4BoxHeader("trak", 16));
        stream.AddRange(Mp4BoxHeader("tkhd", 8));
        var boxes = Mp4BoxReader.ReadAll(stream.ToArray());
        Equal(1, boxes.Count);
        Equal("moov", boxes[0].Type);
        Equal(2, boxes[0].Children!.Count);
        Equal("mvhd", boxes[0].Children[0].Type);
        Equal("trak", boxes[0].Children[1].Type);
        Equal(1, boxes[0].Children[1].Children!.Count);
        Equal("tkhd", boxes[0].Children[1].Children![0].Type);
    }

    [TestMethod]
    public void Mp4Box_FindFirst_FindsNestedBox()
    {
        // moov > trak > mdia > minf > stbl > stsd (all container boxes except stsd).
        // Build a deeply-nested structure and verify FindFirst reaches stsd.
        var tkhd = Mp4BoxHeader("tkhd", 8);
        var stsd = Mp4BoxHeader("stsd", 8);
        var stbl = new List<byte>();
        stbl.AddRange(Mp4BoxHeader("stbl", 16));
        stbl.AddRange(stsd);
        var minf = new List<byte>();
        minf.AddRange(Mp4BoxHeader("minf", (uint)(8 + stbl.Count)));
        minf.AddRange(stbl);
        var mdia = new List<byte>();
        mdia.AddRange(Mp4BoxHeader("mdia", (uint)(8 + minf.Count)));
        mdia.AddRange(minf);
        var trak = new List<byte>();
        trak.AddRange(Mp4BoxHeader("trak", (uint)(8 + tkhd.Length + mdia.Count)));
        trak.AddRange(tkhd);
        trak.AddRange(mdia);
        var moov = new List<byte>();
        moov.AddRange(Mp4BoxHeader("moov", (uint)(8 + trak.Count)));
        moov.AddRange(trak);
        var boxes = Mp4BoxReader.ReadAll(moov.ToArray());
        var found = Mp4BoxReader.FindFirst(boxes, "stsd");
        True(found is not null, "stsd should be found via recursive search.");
        Equal("stsd", found!.Type);
    }

    [TestMethod]
    public void Mp4Box_ExtendedSize_Size1IndicatesLarge()
    {
        // Build a box with size=1 then 8-byte extended size.
        // Total size = 16 (header 8 + ext 8) + 0 payload = 16.
        var bytes = new byte[16];
        bytes[0] = 0; bytes[1] = 0; bytes[2] = 0; bytes[3] = 1; // size = 1 (extended)
        bytes[4] = (byte)'m'; bytes[5] = (byte)'d'; bytes[6] = (byte)'a'; bytes[7] = (byte)'t';
        // 64-bit big-endian 16
        for (int i = 0; i < 7; i++) bytes[8 + i] = 0;
        bytes[15] = 16;
        var boxes = Mp4BoxReader.ReadAll(bytes);
        Equal(1, boxes.Count);
        Equal("mdat", boxes[0].Type);
        Equal(16L, boxes[0].Size);
        Equal(16, boxes[0].HeaderSize);
    }

    [TestMethod]
    public void Mp4Box_SizeZero_ExtendsToEndOfFile()
    {
        // Size=0 means "to end of file"; the box consumes whatever bytes remain.
        var bytes = new byte[32];
        // size 0, type 'mdat'.
        bytes[4] = (byte)'m'; bytes[5] = (byte)'d'; bytes[6] = (byte)'a'; bytes[7] = (byte)'t';
        var boxes = Mp4BoxReader.ReadAll(bytes);
        Equal(1, boxes.Count);
        Equal("mdat", boxes[0].Type);
        Equal(32L, boxes[0].Size);
    }

    [TestMethod]
    public void Mp4Box_TruncatedHeader_Throws()
    {
        var bytes = new byte[] { 0, 0, 0 };
        bool threw = false;
        try { Mp4BoxReader.ReadAll(bytes); } catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void Mp4Box_SizeExceedsBuffer_Throws()
    {
        // size=1000 but only 16 bytes total.
        var bytes = new byte[16];
        bytes[3] = 1000 & 0xFF;
        bytes[2] = (1000 >> 8) & 0xFF;
        bytes[4] = (byte)'f'; bytes[5] = (byte)'t'; bytes[6] = (byte)'y'; bytes[7] = (byte)'p';
        bool threw = false;
        try { Mp4BoxReader.ReadAll(bytes); } catch (InvalidDataException) { threw = true; }
        True(threw);
    }
}
