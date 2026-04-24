using SpawnDev.Codecs.Container.Mp4;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="Mp4FileTypeBoxParser"/>. Hand-builds valid 'ftyp' box
/// bytes and parses them back to verify major brand, minor version, and
/// compatible-brands list.
/// </summary>
public abstract partial class CodecsTestBase
{
    private static byte[] BuildFtypBytes(string majorBrand, uint minorVersion, params string[] compatibleBrands)
    {
        if (majorBrand.Length != 4) throw new ArgumentException(nameof(majorBrand));
        int size = 8 + 4 + 4 + compatibleBrands.Length * 4;
        var bytes = new byte[size];
        // Header: 32-bit BE size, 4-byte type "ftyp".
        bytes[0] = (byte)(size >> 24);
        bytes[1] = (byte)(size >> 16);
        bytes[2] = (byte)(size >> 8);
        bytes[3] = (byte)size;
        bytes[4] = (byte)'f'; bytes[5] = (byte)'t'; bytes[6] = (byte)'y'; bytes[7] = (byte)'p';
        for (int i = 0; i < 4; i++) bytes[8 + i] = (byte)majorBrand[i];
        bytes[12] = (byte)(minorVersion >> 24);
        bytes[13] = (byte)(minorVersion >> 16);
        bytes[14] = (byte)(minorVersion >> 8);
        bytes[15] = (byte)minorVersion;
        int pos = 16;
        foreach (var brand in compatibleBrands)
        {
            if (brand.Length != 4) throw new ArgumentException("compatible brand must be 4 chars");
            for (int i = 0; i < 4; i++) bytes[pos + i] = (byte)brand[i];
            pos += 4;
        }
        return bytes;
    }

    [TestMethod]
    public void Mp4Ftyp_IsomBrand_WithCompatibles_Parses()
    {
        var bytes = BuildFtypBytes("isom", 512, "isom", "iso2", "avc1", "mp41");
        var boxes = Mp4BoxReader.ReadAll(bytes);
        Equal(1, boxes.Count);
        var ftyp = Mp4FileTypeBoxParser.Parse(boxes[0], bytes);
        Equal("isom", ftyp.MajorBrand);
        Equal(512u, ftyp.MinorVersion);
        Equal(4, ftyp.CompatibleBrands.Count);
        Equal("isom", ftyp.CompatibleBrands[0]);
        Equal("iso2", ftyp.CompatibleBrands[1]);
        Equal("avc1", ftyp.CompatibleBrands[2]);
        Equal("mp41", ftyp.CompatibleBrands[3]);
    }

    [TestMethod]
    public void Mp4Ftyp_Av01Brand_Parses()
    {
        // av01 = AV1-in-MP4 per the AOMedia spec.
        var bytes = BuildFtypBytes("av01", 0, "av01", "iso6", "mp41");
        var boxes = Mp4BoxReader.ReadAll(bytes);
        var ftyp = Mp4FileTypeBoxParser.Parse(boxes[0], bytes);
        Equal("av01", ftyp.MajorBrand);
        Equal(0u, ftyp.MinorVersion);
        Equal(3, ftyp.CompatibleBrands.Count);
    }

    [TestMethod]
    public void Mp4Ftyp_NoCompatibleBrands_Parses()
    {
        var bytes = BuildFtypBytes("mp42", 1);
        var boxes = Mp4BoxReader.ReadAll(bytes);
        var ftyp = Mp4FileTypeBoxParser.Parse(boxes[0], bytes);
        Equal("mp42", ftyp.MajorBrand);
        Equal(1u, ftyp.MinorVersion);
        Equal(0, ftyp.CompatibleBrands.Count);
    }

    [TestMethod]
    public void Mp4Ftyp_WrongBoxType_Throws()
    {
        // Build a box whose type is not 'ftyp' and feed it to the parser.
        var moov = new byte[] { 0, 0, 0, 8, (byte)'m', (byte)'o', (byte)'o', (byte)'v' };
        var boxes = Mp4BoxReader.ReadAll(moov);
        bool threw = false;
        try { _ = Mp4FileTypeBoxParser.Parse(boxes[0], moov); }
        catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void Mp4Ftyp_TooSmall_Throws()
    {
        // ftyp box of 12 bytes (header + 4 bytes) - not enough for major + minor.
        var bytes = new byte[12];
        bytes[0] = 0; bytes[1] = 0; bytes[2] = 0; bytes[3] = 12;
        bytes[4] = (byte)'f'; bytes[5] = (byte)'t'; bytes[6] = (byte)'y'; bytes[7] = (byte)'p';
        bytes[8] = (byte)'i'; bytes[9] = (byte)'s'; bytes[10] = (byte)'o'; bytes[11] = (byte)'m';
        var boxes = Mp4BoxReader.ReadAll(bytes);
        bool threw = false;
        try { _ = Mp4FileTypeBoxParser.Parse(boxes[0], bytes); }
        catch (InvalidDataException) { threw = true; }
        True(threw);
    }
}
