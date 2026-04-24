using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="FlacMetadataParser"/>. Stream prelude parsing covers the
/// 4-byte "fLaC" marker, the 4-byte metadata block header (last-flag + type + length),
/// and the 34-byte STREAMINFO payload. Non-STREAMINFO blocks are skipped but their
/// headers are still consumed. All test vectors are hand-built to exact RFC 9639
/// Section 8.1 bit-layout.
/// </summary>
public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Canonical STREAMINFO payload used across multiple tests:
    /// MinBlock=4096, MaxBlock=4096, MinFrame=100, MaxFrame=8192,
    /// SampleRate=44100, Channels=2, BitsPerSample=16, TotalSamples=1000000, MD5=0x01..0x10.
    /// </summary>
    private static byte[] BuildCanonicalStreamInfoPayload() => new byte[]
    {
        // MinBlock=0x1000, MaxBlock=0x1000
        0x10, 0x00, 0x10, 0x00,
        // MinFrame=0x000064, MaxFrame=0x002000
        0x00, 0x00, 0x64, 0x00, 0x20, 0x00,
        // Packed 64-bit field: SampleRate(20)=44100 + Channels-1(3)=1 + BPS-1(5)=15 + TotalSamples(36)=1000000
        0x0A, 0xC4, 0x42, 0xF0, 0x00, 0x0F, 0x42, 0x40,
        // MD5 signature
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10,
    };

    private static byte[] BuildCanonicalPrelude()
    {
        var payload = BuildCanonicalStreamInfoPayload();
        var bytes = new List<byte>();
        // "fLaC"
        bytes.AddRange(new byte[] { (byte)'f', (byte)'L', (byte)'a', (byte)'C' });
        // Header: isLast=true (1 bit), blockType=0 (7 bits), length=34 (24 bits)
        // -> 0x80 0x00 0x00 0x22
        bytes.AddRange(new byte[] { 0x80, 0x00, 0x00, 0x22 });
        bytes.AddRange(payload);
        return bytes.ToArray();
    }

    [TestMethod]
    public void FlacMetadataParser_ReadStreamMarker_Valid()
    {
        byte[] data = { (byte)'f', (byte)'L', (byte)'a', (byte)'C', 0x00 };
        FlacMetadataParser.ReadStreamMarker(data, out int bytesRead);
        Equal(4, bytesRead);
    }

    [TestMethod]
    public void FlacMetadataParser_ReadStreamMarker_Wrong_Throws()
    {
        byte[] data = { (byte)'F', (byte)'L', (byte)'A', (byte)'C' };
        Throws<InvalidDataException>(() =>
            FlacMetadataParser.ReadStreamMarker(data, out _));
    }

    [TestMethod]
    public void FlacMetadataParser_ReadStreamMarker_Short_Throws()
    {
        byte[] data = { (byte)'f', (byte)'L', (byte)'a' };
        Throws<InvalidDataException>(() =>
            FlacMetadataParser.ReadStreamMarker(data, out _));
    }

    [TestMethod]
    public void FlacMetadataParser_ReadBlockHeader_Decomposes()
    {
        // isLast=1, blockType=6 (PICTURE), length=0x123456 -> 0x86 0x12 0x34 0x56
        byte[] data = { 0x86, 0x12, 0x34, 0x56 };
        var hdr = FlacMetadataParser.ReadBlockHeader(data, out int bytesRead);
        Equal(4, bytesRead);
        True(hdr.IsLast, "IsLast should be true.");
        Equal(6, hdr.BlockType);
        Equal(0x123456, hdr.LengthBytes);
    }

    [TestMethod]
    public void FlacMetadataParser_ReadBlockHeader_NotLast()
    {
        // isLast=0, blockType=3 (SEEKTABLE), length=0 -> 0x03 0x00 0x00 0x00
        byte[] data = { 0x03, 0x00, 0x00, 0x00 };
        var hdr = FlacMetadataParser.ReadBlockHeader(data, out _);
        False(hdr.IsLast, "IsLast should be false.");
        Equal(FlacConstants.MetadataSeekTable, hdr.BlockType);
        Equal(0, hdr.LengthBytes);
    }

    [TestMethod]
    public void FlacMetadataParser_ReadStreamInfo_CanonicalFields()
    {
        var payload = BuildCanonicalStreamInfoPayload();
        var info = FlacMetadataParser.ReadStreamInfo(payload);
        Equal(4096, info.MinBlockSize);
        Equal(4096, info.MaxBlockSize);
        Equal(100, info.MinFrameSize);
        Equal(8192, info.MaxFrameSize);
        Equal(44100, info.SampleRateHz);
        Equal(2, info.Channels);
        Equal(16, info.BitsPerSample);
        Equal(1_000_000UL, info.TotalSamples);
        Equal(16, info.Md5Signature.Length);
        for (int i = 0; i < 16; i++)
            Equal((byte)(i + 1), info.Md5Signature[i]);
    }

    [TestMethod]
    public void FlacMetadataParser_ReadStreamInfo_VariableRate_48kHz_Mono_24bit()
    {
        // MinBlock=4608, MaxBlock=4608, MinFrame=0, MaxFrame=0, Rate=48000, Ch=1, BPS=24, Total=0
        // SampleRate=48000=0x0BB80, Ch-1=0, BPS-1=23, TotalSamples=0
        // Packed 64 bits: (0x0BB80 << 44) | (0 << 41) | (23 << 36) | 0
        // = 0x0B B8 0 X Y Z where...
        // bits 63..56 = 0x0B
        // bits 55..48 = 0xB8
        // bits 47..44 = 0x0 (low 4 of rate)
        // bits 43..41 = 0x0 (channels-1)
        // bit 40 = 1 (MSB of BPS-1=23=10111)
        // bits 39..36 = 0x7 (low 4 of BPS-1)
        // bits 35..32 = 0x0 (top of TotalSamples)
        // bits 31..0 = 0x00000000
        // byte 12 = 0000_000_1 = 0x01
        // byte 13 = 0111_0000 = 0x70
        var payload = new byte[]
        {
            0x12, 0x00, 0x12, 0x00,                     // min/max block = 4608
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00,         // min/max frame = 0
            0x0B, 0xB8, 0x01, 0x70, 0x00, 0x00, 0x00, 0x00, // packed
            // MD5 all zero (no decoded data)
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        };
        var info = FlacMetadataParser.ReadStreamInfo(payload);
        Equal(4608, info.MinBlockSize);
        Equal(4608, info.MaxBlockSize);
        Equal(0, info.MinFrameSize);
        Equal(0, info.MaxFrameSize);
        Equal(48000, info.SampleRateHz);
        Equal(1, info.Channels);
        Equal(24, info.BitsPerSample);
        Equal(0UL, info.TotalSamples);
        for (int i = 0; i < 16; i++)
            Equal((byte)0, info.Md5Signature[i]);
    }

    [TestMethod]
    public void FlacMetadataParser_ReadStreamInfo_TooShort_Throws()
    {
        byte[] data = new byte[10];
        Throws<InvalidDataException>(() =>
            FlacMetadataParser.ReadStreamInfo(data));
    }

    [TestMethod]
    public void FlacMetadataParser_ReadStreamInfo_InvalidSampleRate_Throws()
    {
        // SampleRate = 0 is invalid.
        var payload = new byte[34];
        // Bytes 0..9 already zero. Bytes 10..17 all zero = SampleRate=0, ch-1=0, bps-1=0, total=0.
        // But bps=0+1=1 is invalid first, so change bytes to test rate specifically.
        // Set BPS-1 = 7 (bps=8, valid), rate=0.
        // Packed bits: rate(20)=0, ch-1(3)=0, bps-1(5)=7, total(36)=0
        // bits 47..40 = 0000_000_0 = 0x00
        // bits 39..32 = 0111_0000 = 0x70
        // rate bytes 10,11 = 0, 0; byte 12 low nibble part of rate = 0
        payload[12] = 0x00;
        payload[13] = 0x70;
        Throws<InvalidDataException>(() =>
            FlacMetadataParser.ReadStreamInfo(payload));
    }

    [TestMethod]
    public void FlacMetadataParser_ReadStreamInfo_InvalidBitsPerSample_Throws()
    {
        // BPS = 3 (below min 4) is invalid. BPS-1 = 2.
        // Build payload with valid rate (8000 = 0x01F40) and invalid bps.
        // (0x01F40 << 44) | (0 << 41) | (2 << 36) | 0
        // bits 63..56 = 0x01
        // bits 55..48 = 0xF4
        // bits 47..44 = 0x0
        // bits 43..41 = 0
        // bit 40 = 0
        // bits 39..36 = 2 = 0010
        // bits 35..0 = 0
        // byte 12 = 0000_000_0 = 0x00
        // byte 13 = 0010_0000 = 0x20
        var payload = new byte[34];
        payload[10] = 0x01; payload[11] = 0xF4;
        payload[12] = 0x00; payload[13] = 0x20;
        Throws<InvalidDataException>(() =>
            FlacMetadataParser.ReadStreamInfo(payload));
    }

    [TestMethod]
    public void FlacMetadataParser_ReadStreamPrelude_SingleStreamInfoBlock()
    {
        var data = BuildCanonicalPrelude();
        var (info, audioOffset) = FlacMetadataParser.ReadStreamPrelude(data);
        Equal(44100, info.SampleRateHz);
        Equal(2, info.Channels);
        Equal(16, info.BitsPerSample);
        Equal(4 + 4 + 34, audioOffset); // marker + header + STREAMINFO payload
    }

    [TestMethod]
    public void FlacMetadataParser_ReadStreamPrelude_MultipleBlocks_StopsAtLast()
    {
        // "fLaC" + STREAMINFO (not last) + PADDING (last, 10 zero bytes) + fake audio byte.
        var streamInfoPayload = BuildCanonicalStreamInfoPayload();
        var bytes = new List<byte>();
        bytes.AddRange(new byte[] { (byte)'f', (byte)'L', (byte)'a', (byte)'C' });
        // STREAMINFO header: isLast=false, type=0, len=34 -> 0x00 0x00 0x00 0x22
        bytes.AddRange(new byte[] { 0x00, 0x00, 0x00, 0x22 });
        bytes.AddRange(streamInfoPayload);
        // PADDING header: isLast=true, type=1, len=10 -> 0x81 0x00 0x00 0x0A
        bytes.AddRange(new byte[] { 0x81, 0x00, 0x00, 0x0A });
        bytes.AddRange(new byte[10]);
        // Trailing audio byte that should not be consumed.
        bytes.Add(0xFF);

        var arr = bytes.ToArray();
        var (info, audioOffset) = FlacMetadataParser.ReadStreamPrelude(arr);
        Equal(44100, info.SampleRateHz);
        // audioOffset = 4 (marker) + 4 (STREAMINFO header) + 34 (STREAMINFO) + 4 (PADDING header) + 10 (PADDING)
        Equal(56, audioOffset);
        Equal((byte)0xFF, arr[audioOffset]);
    }

    [TestMethod]
    public void FlacMetadataParser_ReadStreamPrelude_BadMarker_Throws()
    {
        var data = new byte[42];
        // First 4 bytes wrong.
        data[0] = 0xDE; data[1] = 0xAD; data[2] = 0xBE; data[3] = 0xEF;
        Throws<InvalidDataException>(() =>
            FlacMetadataParser.ReadStreamPrelude(data));
    }

    [TestMethod]
    public void FlacMetadataParser_ReadStreamPrelude_FirstBlockNotStreamInfo_Throws()
    {
        // "fLaC" + PADDING as first block (invalid - STREAMINFO must be first).
        var bytes = new List<byte>();
        bytes.AddRange(new byte[] { (byte)'f', (byte)'L', (byte)'a', (byte)'C' });
        // PADDING header: isLast=true, type=1, len=0
        bytes.AddRange(new byte[] { 0x81, 0x00, 0x00, 0x00 });
        Throws<InvalidDataException>(() =>
            FlacMetadataParser.ReadStreamPrelude(bytes.ToArray()));
    }

    [TestMethod]
    public void FlacMetadataParser_ReadStreamPrelude_StreamInfoTruncated_Throws()
    {
        var bytes = new List<byte>();
        bytes.AddRange(new byte[] { (byte)'f', (byte)'L', (byte)'a', (byte)'C' });
        bytes.AddRange(new byte[] { 0x80, 0x00, 0x00, 0x22 });
        // Only 20 bytes of STREAMINFO (not 34).
        bytes.AddRange(new byte[20]);
        Throws<InvalidDataException>(() =>
            FlacMetadataParser.ReadStreamPrelude(bytes.ToArray()));
    }
}
