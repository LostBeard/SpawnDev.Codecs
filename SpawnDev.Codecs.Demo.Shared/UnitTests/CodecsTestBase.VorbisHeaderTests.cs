using SpawnDev.Codecs.Audio.Vorbis;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="VorbisIdentificationHeaderParser"/> and
/// <see cref="VorbisCommentHeaderParser"/>. Headers are hand-built
/// per Vorbis I Section 4.2.2 and Section 5.
/// </summary>
public abstract partial class CodecsTestBase
{
    // -------- Identification header helpers --------

    private static byte[] BuildIdentHeader(int version, int channels, int sampleRate,
        int bitrateMax, int bitrateNom, int bitrateMin,
        int bs0Log, int bs1Log, bool framingFlag = true)
    {
        var bytes = new byte[30];
        bytes[0] = 0x01; // packet type
        byte[] magic = { (byte)'v', (byte)'o', (byte)'r', (byte)'b', (byte)'i', (byte)'s' };
        Array.Copy(magic, 0, bytes, 1, 6);
        WriteInt32Le(bytes, 7, version);
        bytes[11] = (byte)channels;
        WriteInt32Le(bytes, 12, sampleRate);
        WriteInt32Le(bytes, 16, bitrateMax);
        WriteInt32Le(bytes, 20, bitrateNom);
        WriteInt32Le(bytes, 24, bitrateMin);
        bytes[28] = (byte)((bs1Log << 4) | bs0Log);
        bytes[29] = framingFlag ? (byte)0x01 : (byte)0x00;
        return bytes;
    }

    private static void WriteInt32Le(byte[] dest, int offset, int value)
    {
        for (int i = 0; i < 4; i++) dest[offset + i] = (byte)(value >> (8 * i));
    }

    private static void WriteUInt32Le(byte[] dest, int offset, uint value)
    {
        for (int i = 0; i < 4; i++) dest[offset + i] = (byte)(value >> (8 * i));
    }

    // -------- Ident header tests --------

    [TestMethod]
    public void VorbisIdent_Canonical_StereoFullband_Parses()
    {
        var data = BuildIdentHeader(
            version: 0, channels: 2, sampleRate: 48000,
            bitrateMax: 0, bitrateNom: 192000, bitrateMin: 0,
            bs0Log: 8, bs1Log: 11);
        var h = VorbisIdentificationHeaderParser.Parse(data);
        Equal(0, h.VorbisVersion);
        Equal(2, h.AudioChannels);
        Equal(48000, h.SampleRateHz);
        Equal(192000, h.BitrateNominal);
        Equal(256, h.BlockSize0);  // 2^8
        Equal(2048, h.BlockSize1); // 2^11
    }

    [TestMethod]
    public void VorbisIdent_FreqAtNarrowband_Parses()
    {
        var data = BuildIdentHeader(0, 1, 8000, 0, 0, 0, 6, 13);
        var h = VorbisIdentificationHeaderParser.Parse(data);
        Equal(1, h.AudioChannels);
        Equal(8000, h.SampleRateHz);
        Equal(64, h.BlockSize0);
        Equal(8192, h.BlockSize1);
    }

    [TestMethod]
    public void VorbisIdent_WrongVersion_Throws()
    {
        var data = BuildIdentHeader(1, 2, 48000, 0, 0, 0, 8, 11);
        bool threw = false;
        try { _ = VorbisIdentificationHeaderParser.Parse(data); }
        catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void VorbisIdent_ZeroChannels_Throws()
    {
        var data = BuildIdentHeader(0, 0, 48000, 0, 0, 0, 8, 11);
        bool threw = false;
        try { _ = VorbisIdentificationHeaderParser.Parse(data); }
        catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void VorbisIdent_Bs0LargerThanBs1_Throws()
    {
        var data = BuildIdentHeader(0, 2, 48000, 0, 0, 0, 11, 8);
        bool threw = false;
        try { _ = VorbisIdentificationHeaderParser.Parse(data); }
        catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void VorbisIdent_FramingFlagClear_Throws()
    {
        var data = BuildIdentHeader(0, 2, 48000, 0, 0, 0, 8, 11, framingFlag: false);
        bool threw = false;
        try { _ = VorbisIdentificationHeaderParser.Parse(data); }
        catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void VorbisIdent_WrongPacketType_Throws()
    {
        var data = BuildIdentHeader(0, 2, 48000, 0, 0, 0, 8, 11);
        data[0] = 0x03;
        bool threw = false;
        try { _ = VorbisIdentificationHeaderParser.Parse(data); }
        catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void VorbisIdent_BadMagic_Throws()
    {
        var data = BuildIdentHeader(0, 2, 48000, 0, 0, 0, 8, 11);
        data[3] = (byte)'X';
        bool threw = false;
        try { _ = VorbisIdentificationHeaderParser.Parse(data); }
        catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    // -------- Comment header tests --------

    [TestMethod]
    public void VorbisComment_VendorAndThreeComments_Parses()
    {
        string vendor = "Test Encoder 1.0";
        var comments = new[] { "ARTIST=TJ", "TITLE=SpawnDev Codecs", "ALBUM=Phase 1a" };
        var bytes = BuildCommentHeader(vendor, comments, framingFlag: true);
        var h = VorbisCommentHeaderParser.Parse(bytes);
        Equal(vendor, h.Vendor);
        Equal(3, h.UserComments.Count);
        Equal("ARTIST=TJ", h.UserComments[0]);
        Equal("TITLE=SpawnDev Codecs", h.UserComments[1]);
        Equal("ALBUM=Phase 1a", h.UserComments[2]);
    }

    [TestMethod]
    public void VorbisComment_Utf8Content_RoundTrips()
    {
        string vendor = "VendΘr"; // non-ASCII
        var comments = new[] { "TITLE=Μπουζούκι", "ARTIST=漢字" };
        var bytes = BuildCommentHeader(vendor, comments, framingFlag: true);
        var h = VorbisCommentHeaderParser.Parse(bytes);
        Equal(vendor, h.Vendor);
        Equal(comments[0], h.UserComments[0]);
        Equal(comments[1], h.UserComments[1]);
    }

    [TestMethod]
    public void VorbisComment_EmptyCommentList_Parses()
    {
        var bytes = BuildCommentHeader("Vendor", Array.Empty<string>(), framingFlag: true);
        var h = VorbisCommentHeaderParser.Parse(bytes);
        Equal("Vendor", h.Vendor);
        Equal(0, h.UserComments.Count);
    }

    [TestMethod]
    public void VorbisComment_FramingFlagOptional_ForNonVorbisUsers()
    {
        // Opus / FLAC embed the Vorbis comment structure but omit the framing flag.
        var bytes = BuildCommentHeader("Vendor", new[] { "A=1" }, framingFlag: false);
        var h = VorbisCommentHeaderParser.Parse(bytes);
        Equal("Vendor", h.Vendor);
        Equal("A=1", h.UserComments[0]);
    }

    [TestMethod]
    public void VorbisComment_WrongPacketType_Throws()
    {
        var bytes = BuildCommentHeader("V", new[] { "A=1" }, framingFlag: true);
        bytes[0] = 0x01;
        bool threw = false;
        try { _ = VorbisCommentHeaderParser.Parse(bytes); }
        catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void VorbisComment_BadMagic_Throws()
    {
        var bytes = BuildCommentHeader("V", new[] { "A=1" }, framingFlag: true);
        bytes[5] = (byte)'X';
        bool threw = false;
        try { _ = VorbisCommentHeaderParser.Parse(bytes); }
        catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    private static byte[] BuildCommentHeader(string vendor, string[] userComments, bool framingFlag)
    {
        var vendorBytes = System.Text.Encoding.UTF8.GetBytes(vendor);
        var commentBytesArr = userComments
            .Select(System.Text.Encoding.UTF8.GetBytes)
            .ToArray();

        int size = 1 + 6 + 4 + vendorBytes.Length + 4;
        foreach (var b in commentBytesArr) size += 4 + b.Length;
        if (framingFlag) size += 1;
        var bytes = new byte[size];
        int pos = 0;
        bytes[pos++] = 0x03;
        byte[] magic = { (byte)'v', (byte)'o', (byte)'r', (byte)'b', (byte)'i', (byte)'s' };
        Array.Copy(magic, 0, bytes, pos, 6); pos += 6;
        WriteUInt32Le(bytes, pos, (uint)vendorBytes.Length); pos += 4;
        Array.Copy(vendorBytes, 0, bytes, pos, vendorBytes.Length); pos += vendorBytes.Length;
        WriteUInt32Le(bytes, pos, (uint)commentBytesArr.Length); pos += 4;
        foreach (var b in commentBytesArr)
        {
            WriteUInt32Le(bytes, pos, (uint)b.Length); pos += 4;
            Array.Copy(b, 0, bytes, pos, b.Length); pos += b.Length;
        }
        if (framingFlag) bytes[pos] = 0x01;
        return bytes;
    }
}
