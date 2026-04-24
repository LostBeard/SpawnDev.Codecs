using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="FlacFrameHeaderParser"/>. Header layout per RFC 9639 Section
/// 9.1: 14-bit sync + 1 reserved + 1 blocking-strategy + 4 bsize + 4 srate +
/// 4 chan + 3 bps + 1 reserved + UTF-8 sample/frame number + optional side
/// bytes + CRC-8. Test helpers hand-build bytes to exact spec layout, then
/// append a valid CRC-8 computed via <see cref="FlacCrc"/> (which is verified
/// against the industry-standard check vector in FlacCrc tests).
/// </summary>
public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Build a minimal frame header given the raw codes. Codes must already be
    /// constrained to their field widths. Appends UTF-8 sample/frame number as
    /// a single byte (value 0 for the simplest case).
    /// </summary>
    private static byte[] BuildFrameHeaderBytes(
        int bsizeCode, int srateCode, int chanCode, int bpsCode, int blocking,
        byte utf8SampleOrFrameNumber = 0x00,
        byte[]? sideBytes = null)
    {
        // First 4 bytes from the packed 32-bit field.
        // Bits 31..18: sync 0x3FFE
        // Bit 17: reserved 0
        // Bit 16: blocking
        // Bits 15..12: bsize
        // Bits 11..8:  srate
        // Bits 7..4:   chan
        // Bits 3..1:   bps
        // Bit 0:       reserved 0
        uint packed = 0;
        packed |= (uint)FlacConstants.FrameSyncCode << 18;
        packed |= (uint)(blocking & 1) << 16;
        packed |= (uint)(bsizeCode & 0xF) << 12;
        packed |= (uint)(srateCode & 0xF) << 8;
        packed |= (uint)(chanCode & 0xF) << 4;
        packed |= (uint)(bpsCode & 0x7) << 1;

        var list = new List<byte>
        {
            (byte)(packed >> 24),
            (byte)(packed >> 16),
            (byte)(packed >> 8),
            (byte)packed,
            utf8SampleOrFrameNumber,
        };
        if (sideBytes is not null) list.AddRange(sideBytes);

        byte crc = FlacCrc.Compute8(list.ToArray());
        list.Add(crc);
        return list.ToArray();
    }

    /// <summary>
    /// Canonical STREAMINFO used for tests that fall back to it. 44.1kHz, stereo, 16-bit.
    /// </summary>
    private static FlacStreamInfo TestStreamInfo() => new FlacStreamInfo
    {
        MinBlockSize = 4096,
        MaxBlockSize = 4096,
        MinFrameSize = 0,
        MaxFrameSize = 0,
        SampleRateHz = 44100,
        Channels = 2,
        BitsPerSample = 16,
        TotalSamples = 0,
        Md5Signature = new byte[16],
    };

    [TestMethod]
    public void FlacFrameHeader_Fixed_192_44100_Stereo_16bit_Parses()
    {
        // bsize=0b0001 (192), srate=0b1001 (44100), chan=0b0001 (2 indep), bps=0b100 (16), blocking=0
        var bytes = BuildFrameHeaderBytes(0b0001, 0b1001, 0b0001, 0b100, blocking: 0);
        var header = FlacFrameHeaderParser.Parse(bytes, TestStreamInfo());
        Equal(FlacBlockingStrategy.Fixed, header.BlockingStrategy);
        Equal(192, header.BlockSize);
        Equal(44100, header.SampleRateHz);
        Equal(FlacChannelAssignment.Independent, header.ChannelAssignment);
        Equal(2, header.Channels);
        Equal(16, header.BitsPerSample);
        Equal(0UL, header.SampleOrFrameNumber);
        Equal(bytes.Length, header.HeaderBytesConsumed);
    }

    [TestMethod]
    public void FlacFrameHeader_Variable_48000_MidSide_24bit_Parses()
    {
        // bsize=0b1010 (1024), srate=0b1010 (48000), chan=0b1010 (M/S), bps=0b110 (24), blocking=1
        var bytes = BuildFrameHeaderBytes(0b1010, 0b1010, 0b1010, 0b110, blocking: 1);
        var header = FlacFrameHeaderParser.Parse(bytes, TestStreamInfo());
        Equal(FlacBlockingStrategy.Variable, header.BlockingStrategy);
        Equal(1024, header.BlockSize);
        Equal(48000, header.SampleRateHz);
        Equal(FlacChannelAssignment.MidSide, header.ChannelAssignment);
        Equal(2, header.Channels);
        Equal(24, header.BitsPerSample);
    }

    [TestMethod]
    public void FlacFrameHeader_BlockSizeCode0110_8bitSide_Resolves()
    {
        // bsize=0b0110 -> read 8 bits side => actual block size = side + 1.
        // Side byte = 99 -> block size 100.
        var bytes = BuildFrameHeaderBytes(0b0110, 0b1001, 0b0001, 0b100, blocking: 0,
            sideBytes: new byte[] { 99 });
        var header = FlacFrameHeaderParser.Parse(bytes, TestStreamInfo());
        Equal(100, header.BlockSize);
    }

    [TestMethod]
    public void FlacFrameHeader_BlockSizeCode0111_16bitSide_Resolves()
    {
        // bsize=0b0111 -> read 16 bits side => actual block size = side + 1.
        // Side bytes = 0x1F 0xFF = 8191 -> block size 8192.
        var bytes = BuildFrameHeaderBytes(0b0111, 0b1001, 0b0001, 0b100, blocking: 0,
            sideBytes: new byte[] { 0x1F, 0xFF });
        var header = FlacFrameHeaderParser.Parse(bytes, TestStreamInfo());
        Equal(8192, header.BlockSize);
    }

    [TestMethod]
    public void FlacFrameHeader_SampleRateCode1100_kHzSide_Resolves()
    {
        // srate=0b1100 -> 8-bit side byte in kHz. Side = 48 -> 48000 Hz.
        var bytes = BuildFrameHeaderBytes(0b0001, 0b1100, 0b0001, 0b100, blocking: 0,
            sideBytes: new byte[] { 48 });
        var header = FlacFrameHeaderParser.Parse(bytes, TestStreamInfo());
        Equal(48000, header.SampleRateHz);
    }

    [TestMethod]
    public void FlacFrameHeader_SampleRateCode1101_HzSide_Resolves()
    {
        // srate=0b1101 -> 16-bit side in Hz. Side = 0xAC44 = 44100.
        var bytes = BuildFrameHeaderBytes(0b0001, 0b1101, 0b0001, 0b100, blocking: 0,
            sideBytes: new byte[] { 0xAC, 0x44 });
        var header = FlacFrameHeaderParser.Parse(bytes, TestStreamInfo());
        Equal(44100, header.SampleRateHz);
    }

    [TestMethod]
    public void FlacFrameHeader_SampleRateCode1110_DecaHzSide_Resolves()
    {
        // srate=0b1110 -> 16-bit side in decaHz. Side = 4410 (0x113A) -> 44100 Hz.
        var bytes = BuildFrameHeaderBytes(0b0001, 0b1110, 0b0001, 0b100, blocking: 0,
            sideBytes: new byte[] { 0x11, 0x3A });
        var header = FlacFrameHeaderParser.Parse(bytes, TestStreamInfo());
        Equal(44100, header.SampleRateHz);
    }

    [TestMethod]
    public void FlacFrameHeader_SampleRateCode0000_FallsBackToStreamInfo()
    {
        var bytes = BuildFrameHeaderBytes(0b0001, 0b0000, 0b0001, 0b100, blocking: 0);
        var header = FlacFrameHeaderParser.Parse(bytes, TestStreamInfo());
        Equal(44100, header.SampleRateHz);
    }

    [TestMethod]
    public void FlacFrameHeader_BpsCode000_FallsBackToStreamInfo()
    {
        var bytes = BuildFrameHeaderBytes(0b0001, 0b1001, 0b0001, 0b000, blocking: 0);
        var header = FlacFrameHeaderParser.Parse(bytes, TestStreamInfo());
        Equal(16, header.BitsPerSample); // from streaminfo
    }

    [TestMethod]
    public void FlacFrameHeader_ChannelAssignment_LeftSide()
    {
        var bytes = BuildFrameHeaderBytes(0b0001, 0b1001, 0b1000, 0b100, blocking: 0);
        var header = FlacFrameHeaderParser.Parse(bytes, TestStreamInfo());
        Equal(FlacChannelAssignment.LeftSide, header.ChannelAssignment);
        Equal(2, header.Channels);
    }

    [TestMethod]
    public void FlacFrameHeader_ChannelAssignment_RightSide()
    {
        var bytes = BuildFrameHeaderBytes(0b0001, 0b1001, 0b1001, 0b100, blocking: 0);
        var header = FlacFrameHeaderParser.Parse(bytes, TestStreamInfo());
        Equal(FlacChannelAssignment.RightSide, header.ChannelAssignment);
    }

    [TestMethod]
    public void FlacFrameHeader_ChannelAssignment_EightIndependent()
    {
        var bytes = BuildFrameHeaderBytes(0b0001, 0b1001, 0b0111, 0b100, blocking: 0);
        var header = FlacFrameHeaderParser.Parse(bytes, TestStreamInfo());
        Equal(FlacChannelAssignment.Independent, header.ChannelAssignment);
        Equal(8, header.Channels);
    }

    [TestMethod]
    public void FlacFrameHeader_BadSync_Throws()
    {
        var bytes = BuildFrameHeaderBytes(0b0001, 0b1001, 0b0001, 0b100, blocking: 0);
        // Corrupt the sync.
        bytes[0] = 0x00;
        Throws<InvalidDataException>(() =>
            FlacFrameHeaderParser.Parse(bytes, TestStreamInfo()));
    }

    [TestMethod]
    public void FlacFrameHeader_BadCrc8_Throws()
    {
        var bytes = BuildFrameHeaderBytes(0b0001, 0b1001, 0b0001, 0b100, blocking: 0);
        // Corrupt the final CRC byte.
        bytes[bytes.Length - 1] ^= 0xFF;
        Throws<InvalidDataException>(() =>
            FlacFrameHeaderParser.Parse(bytes, TestStreamInfo()));
    }

    [TestMethod]
    public void FlacFrameHeader_ReservedBpsCode011_Throws()
    {
        var bytes = BuildFrameHeaderBytes(0b0001, 0b1001, 0b0001, 0b011, blocking: 0);
        Throws<InvalidDataException>(() =>
            FlacFrameHeaderParser.Parse(bytes, TestStreamInfo()));
    }

    [TestMethod]
    public void FlacFrameHeader_ReservedBlockSizeCode0000_Throws()
    {
        var bytes = BuildFrameHeaderBytes(0b0000, 0b1001, 0b0001, 0b100, blocking: 0);
        Throws<InvalidDataException>(() =>
            FlacFrameHeaderParser.Parse(bytes, TestStreamInfo()));
    }

    [TestMethod]
    public void FlacFrameHeader_InvalidSampleRateCode1111_Throws()
    {
        var bytes = BuildFrameHeaderBytes(0b0001, 0b1111, 0b0001, 0b100, blocking: 0);
        Throws<InvalidDataException>(() =>
            FlacFrameHeaderParser.Parse(bytes, TestStreamInfo()));
    }

    [TestMethod]
    public void FlacFrameHeader_ReservedChannelCode1011_Throws()
    {
        var bytes = BuildFrameHeaderBytes(0b0001, 0b1001, 0b1011, 0b100, blocking: 0);
        Throws<InvalidDataException>(() =>
            FlacFrameHeaderParser.Parse(bytes, TestStreamInfo()));
    }

    [TestMethod]
    public void FlacFrameHeader_BlockSize_AllPowerOfTwoCodes()
    {
        // Codes 0b1000..0b1111 -> 256, 512, 1024, 2048, 4096, 8192, 16384, 32768.
        int[] expected = { 256, 512, 1024, 2048, 4096, 8192, 16384, 32768 };
        for (int i = 0; i < 8; i++)
        {
            int code = 0b1000 + i;
            var bytes = BuildFrameHeaderBytes(code, 0b1001, 0b0001, 0b100, blocking: 0);
            var header = FlacFrameHeaderParser.Parse(bytes, TestStreamInfo());
            Equal(expected[i], header.BlockSize, $"code=0b{code:B4}");
        }
    }

    [TestMethod]
    public void FlacFrameHeader_SampleRate_AllFixedCodes()
    {
        // Codes 0b0001..0b1011 -> 88200, 176400, 192000, 8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000.
        int[] expected = { 88200, 176400, 192000, 8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000 };
        for (int i = 0; i < expected.Length; i++)
        {
            int code = i + 1;
            var bytes = BuildFrameHeaderBytes(0b0001, code, 0b0001, 0b100, blocking: 0);
            var header = FlacFrameHeaderParser.Parse(bytes, TestStreamInfo());
            Equal(expected[i], header.SampleRateHz, $"srate code=0b{code:B4}");
        }
    }

    [TestMethod]
    public void FlacFrameHeader_BitsPerSample_AllFixedCodes()
    {
        var expected = new Dictionary<int, int>
        {
            [0b001] = 8,
            [0b010] = 12,
            [0b100] = 16,
            [0b101] = 20,
            [0b110] = 24,
            [0b111] = 32,
        };
        foreach (var kv in expected)
        {
            var bytes = BuildFrameHeaderBytes(0b0001, 0b1001, 0b0001, kv.Key, blocking: 0);
            var header = FlacFrameHeaderParser.Parse(bytes, TestStreamInfo());
            Equal(kv.Value, header.BitsPerSample, $"bps code=0b{kv.Key:B3}");
        }
    }

    [TestMethod]
    public void FlacFrameHeader_Utf8FrameNumber_MultiByte_Parses()
    {
        // Frame number = 0xA2 (162) encoded as 2-byte UTF-8: 0xC2 0xA2.
        // BuildFrameHeaderBytes expects a single byte for the UTF-8 number; we bypass it.
        var bytes = BuildFrameHeaderBytes(0b0001, 0b1001, 0b0001, 0b100, blocking: 0);
        // Replace the single 0x00 number byte with a 2-byte 0xC2 0xA2 and recompute CRC.
        var list = new List<byte>(bytes);
        list.RemoveAt(list.Count - 1); // drop CRC
        list.RemoveAt(list.Count - 1); // drop 0x00 frame number
        list.Add(0xC2);
        list.Add(0xA2);
        byte crc = FlacCrc.Compute8(list.ToArray());
        list.Add(crc);
        var bytes2 = list.ToArray();

        var header = FlacFrameHeaderParser.Parse(bytes2, TestStreamInfo());
        Equal(0xA2UL, header.SampleOrFrameNumber);
    }
}
