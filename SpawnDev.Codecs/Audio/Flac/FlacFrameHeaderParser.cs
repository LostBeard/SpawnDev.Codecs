// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// FLAC frame header parser. Matches libFLAC stream_decoder.c::read_frame_header_.
// Every audio frame starts with this header, which carries enough information
// to locate the frame in time and to know how many samples, at what rate,
// at what bit depth, and in what channel layout the rest of the frame contains.

namespace SpawnDev.Codecs.Audio.Flac;

/// <summary>
/// Parses a FLAC frame header and verifies its CRC-8.
/// </summary>
public static class FlacFrameHeaderParser
{
    /// <summary>
    /// Parse a FLAC frame header at the start of <paramref name="data"/>, resolving
    /// "get from STREAMINFO" codes against <paramref name="streamInfo"/>. Validates the
    /// trailing CRC-8 and throws <see cref="InvalidDataException"/> on mismatch.
    /// </summary>
    public static FlacFrameHeader Parse(ReadOnlySpan<byte> data, FlacStreamInfo streamInfo)
    {
        if (data.Length < 5)
            throw new InvalidDataException("Frame header needs at least 5 bytes.");

        var r = new FlacBitReader(data);
        int sync = (int)r.ReadBits(14);
        if (sync != FlacConstants.FrameSyncCode)
            throw new InvalidDataException($"Invalid frame sync code: 0x{sync:X4}, expected 0x{FlacConstants.FrameSyncCode:X4}.");
        int reserved1 = (int)r.ReadBit();
        if (reserved1 != 0)
            throw new InvalidDataException($"Reserved bit after sync must be 0, got {reserved1}.");
        var blocking = (FlacBlockingStrategy)r.ReadBit();

        int bsizeCode = (int)r.ReadBits(4);
        int srateCode = (int)r.ReadBits(4);
        int chanCode = (int)r.ReadBits(4);
        int bpsCode = (int)r.ReadBits(3);
        int reserved2 = (int)r.ReadBit();
        if (reserved2 != 0)
            throw new InvalidDataException($"Reserved bit before sample number must be 0, got {reserved2}.");

        // UTF-8 sample-or-frame number: max 6 bytes (31-bit frame number) for fixed, 7 bytes (36-bit sample number) for variable.
        int utf8MaxBytes = blocking == FlacBlockingStrategy.Variable ? 7 : 6;
        ulong sampleOrFrame = r.ReadUtf8VariableLength(utf8MaxBytes);

        // Optional block-size side bytes.
        int blockSize = ResolveBlockSizeCode(bsizeCode, out bool readBs8, out bool readBs16);
        if (readBs8) blockSize = (int)r.ReadBits(8) + 1;
        else if (readBs16) blockSize = (int)r.ReadBits(16) + 1;

        // Optional sample-rate side bytes.
        int sampleRate = ResolveSampleRateCode(srateCode, streamInfo, out int sideBytes);
        if (sideBytes == 1) sampleRate = (int)r.ReadBits(8) * 1000;
        else if (sideBytes == 2 && srateCode == 0b1101) sampleRate = (int)r.ReadBits(16);
        else if (sideBytes == 2 && srateCode == 0b1110) sampleRate = (int)r.ReadBits(16) * 10;

        // Resolve channel count and bits-per-sample.
        (var channelAssignment, int channels) = ResolveChannelCode(chanCode);
        int bitsPerSample = ResolveBitsPerSampleCode(bpsCode, streamInfo);

        // Bit reader must be byte-aligned here - all side-bytes are 8- or 16-bit,
        // and we began on a byte boundary with the 32-bit packed section.
        if ((r.Position & 7) != 0)
            throw new InvalidDataException("Frame header unaligned before CRC-8.");
        int headerBytes = r.Position / 8;
        if (data.Length < headerBytes + 1)
            throw new InvalidDataException("Frame header truncated before CRC-8.");
        byte expectedCrc = data[headerBytes];
        byte actualCrc = FlacCrc.Compute8(data.Slice(0, headerBytes));
        if (expectedCrc != actualCrc)
            throw new InvalidDataException(
                $"Frame header CRC-8 mismatch: expected 0x{expectedCrc:X2}, computed 0x{actualCrc:X2}.");

        return new FlacFrameHeader
        {
            BlockingStrategy = blocking,
            BlockSize = blockSize,
            SampleRateHz = sampleRate,
            ChannelAssignment = channelAssignment,
            Channels = channels,
            BitsPerSample = bitsPerSample,
            SampleOrFrameNumber = sampleOrFrame,
            HeaderBytesConsumed = headerBytes + 1, // +1 for the CRC-8 byte
        };
    }

    private static int ResolveBlockSizeCode(int code, out bool read8, out bool read16)
    {
        read8 = false;
        read16 = false;
        switch (code)
        {
            case 0b0000:
                throw new InvalidDataException("Frame header block-size code 0b0000 is reserved.");
            case 0b0001: return 192;
            case 0b0010: return 576;
            case 0b0011: return 1152;
            case 0b0100: return 2304;
            case 0b0101: return 4608;
            case 0b0110: read8 = true; return 0;
            case 0b0111: read16 = true; return 0;
            default: return 256 << (code - 8); // 0b1000..0b1111 → 256..32768
        }
    }

    private static int ResolveSampleRateCode(int code, FlacStreamInfo info, out int sideBytes)
    {
        sideBytes = 0;
        switch (code)
        {
            case 0b0000: return info.SampleRateHz;
            case 0b0001: return 88200;
            case 0b0010: return 176400;
            case 0b0011: return 192000;
            case 0b0100: return 8000;
            case 0b0101: return 16000;
            case 0b0110: return 22050;
            case 0b0111: return 24000;
            case 0b1000: return 32000;
            case 0b1001: return 44100;
            case 0b1010: return 48000;
            case 0b1011: return 96000;
            case 0b1100: sideBytes = 1; return 0;
            case 0b1101: sideBytes = 2; return 0;
            case 0b1110: sideBytes = 2; return 0;
            case 0b1111:
                throw new InvalidDataException("Frame header sample-rate code 0b1111 is invalid.");
            default:
                throw new InvalidDataException($"Unreachable: sample-rate code {code}.");
        }
    }

    private static (FlacChannelAssignment, int) ResolveChannelCode(int code) => code switch
    {
        >= 0 and <= 7 => (FlacChannelAssignment.Independent, code + 1),
        8 => (FlacChannelAssignment.LeftSide, 2),
        9 => (FlacChannelAssignment.RightSide, 2),
        10 => (FlacChannelAssignment.MidSide, 2),
        _ => throw new InvalidDataException($"Reserved channel assignment code: {code}."),
    };

    private static int ResolveBitsPerSampleCode(int code, FlacStreamInfo info) => code switch
    {
        0b000 => info.BitsPerSample,
        0b001 => 8,
        0b010 => 12,
        0b011 => throw new InvalidDataException("Frame header bits-per-sample code 0b011 is reserved."),
        0b100 => 16,
        0b101 => 20,
        0b110 => 24,
        0b111 => 32,
        _ => throw new InvalidDataException($"Unreachable: bits-per-sample code {code}."),
    };
}
