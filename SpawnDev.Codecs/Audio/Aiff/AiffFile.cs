// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// AIFF (Audio Interchange File Format) reader + writer. AIFF is the
// Apple/Electronic Arts big-endian counterpart to Microsoft's WAV. Like WAV,
// it's a container for uncompressed PCM (the AIFC variant adds compression
// metadata but we target plain AIFF only).
//
// Key spec differences vs WAV:
//  - Top-level magic is "FORM" + "AIFF" (not "RIFF" + "WAVE").
//  - Chunks are big-endian (not little-endian).
//  - Sample rate is stored as an IEEE 754 80-bit extended-precision float.
//  - Sample data in the SSND chunk is big-endian signed (not little-endian).

namespace SpawnDev.Codecs.Audio.Aiff;

/// <summary>Parsed AIFF audio file: geometry + interleaved integer PCM samples.</summary>
public sealed record AiffFile
{
    /// <summary>Sample rate in Hz (rounded to the nearest integer from the 80-bit float field).</summary>
    public int SampleRateHz { get; init; }

    /// <summary>Channel count (1 = mono, 2 = stereo, etc.).</summary>
    public int Channels { get; init; }

    /// <summary>Bits per sample (8, 16, 24, or 32).</summary>
    public int BitsPerSample { get; init; }

    /// <summary>
    /// Interleaved signed integer samples <c>[ch0[0], ch1[0], ...]</c>.
    /// Length = <c>TotalSamplesPerChannel * Channels</c>.
    /// </summary>
    public required int[] InterleavedSamples { get; init; }

    /// <summary>Sample frames per channel.</summary>
    public int TotalSamplesPerChannel { get; init; }
}

/// <summary>Reader/writer for AIFF PCM files.</summary>
public static class AiffFileCodec
{
    /// <summary>Read an AIFF file from a byte buffer.</summary>
    public static AiffFile Read(ReadOnlySpan<byte> data)
    {
        if (data.Length < 54) throw new InvalidDataException($"AIFF file too short: {data.Length} bytes.");
        if (data[0] != 'F' || data[1] != 'O' || data[2] != 'R' || data[3] != 'M')
            throw new InvalidDataException("AIFF missing 'FORM' magic.");
        // bytes[4..7] = overall size - 8 (big-endian, ignored).
        if (data[8] != 'A' || data[9] != 'I' || data[10] != 'F' || (data[11] != 'F' && data[11] != 'C'))
            throw new InvalidDataException("AIFF missing 'AIFF' or 'AIFC' form type.");

        int pos = 12;
        int channels = 0, bps = 0, sampleFrames = 0;
        double sampleRate = 0;
        ReadOnlySpan<byte> sampleBytes = default;
        bool sawComm = false, sawSsnd = false;

        while (pos + 8 <= data.Length)
        {
            string chunkId = System.Text.Encoding.ASCII.GetString(data.Slice(pos, 4));
            uint chunkSize = ReadUInt32Be(data.Slice(pos + 4, 4));
            int chunkStart = pos + 8;
            if (chunkStart + chunkSize > data.Length)
                throw new InvalidDataException($"AIFF chunk '{chunkId}' extends past end of file.");
            if (chunkId == "COMM")
            {
                if (chunkSize < 18)
                    throw new InvalidDataException($"AIFF 'COMM' chunk too small: {chunkSize} bytes.");
                channels = ReadInt16Be(data.Slice(chunkStart, 2));
                sampleFrames = (int)ReadUInt32Be(data.Slice(chunkStart + 2, 4));
                bps = ReadInt16Be(data.Slice(chunkStart + 6, 2));
                sampleRate = ReadExtended80Be(data.Slice(chunkStart + 8, 10));
                sawComm = true;
            }
            else if (chunkId == "SSND")
            {
                if (chunkSize < 8)
                    throw new InvalidDataException($"AIFF 'SSND' chunk too small: {chunkSize} bytes.");
                uint offset = ReadUInt32Be(data.Slice(chunkStart, 4));
                // block size at chunkStart + 4 ignored.
                int sampleStart = chunkStart + 8 + (int)offset;
                int sampleLen = (int)chunkSize - 8 - (int)offset;
                sampleBytes = data.Slice(sampleStart, sampleLen);
                sawSsnd = true;
            }
            // Other chunks ignored.
            pos = chunkStart + (int)chunkSize;
            if ((chunkSize & 1) != 0 && pos < data.Length) pos++; // chunks pad to even
        }
        if (!sawComm) throw new InvalidDataException("AIFF missing 'COMM' chunk.");
        if (!sawSsnd) throw new InvalidDataException("AIFF missing 'SSND' chunk.");
        if (channels < 1) throw new InvalidDataException("AIFF channel count invalid.");
        if (bps is not (8 or 16 or 24 or 32))
            throw new InvalidDataException($"AIFF bit depth {bps} unsupported (use 8/16/24/32).");

        int bytesPerSample = (bps + 7) / 8;
        int totalInterleavedSamples = sampleFrames * channels;
        int[] samples = new int[totalInterleavedSamples];
        DecodeBigEndianIntegerSamples(sampleBytes, bps, samples);

        return new AiffFile
        {
            SampleRateHz = (int)Math.Round(sampleRate),
            Channels = channels,
            BitsPerSample = bps,
            InterleavedSamples = samples,
            TotalSamplesPerChannel = sampleFrames,
        };
    }

    /// <summary>Write interleaved PCM to an AIFF byte buffer.</summary>
    public static byte[] Write(ReadOnlySpan<int> interleavedSamples, int sampleRateHz, int channels, int bitsPerSample)
    {
        if (channels < 1) throw new ArgumentException("channels >= 1 required.", nameof(channels));
        if (bitsPerSample is not (8 or 16 or 24 or 32))
            throw new ArgumentException("bitsPerSample must be 8, 16, 24, or 32.", nameof(bitsPerSample));
        if (interleavedSamples.Length % channels != 0)
            throw new ArgumentException("interleavedSamples length must be a multiple of channels.");
        if (sampleRateHz < 1) throw new ArgumentException("sampleRateHz must be positive.", nameof(sampleRateHz));

        int bytesPerSample = (bitsPerSample + 7) / 8;
        int sampleFrames = interleavedSamples.Length / channels;
        int ssndPayloadSize = interleavedSamples.Length * bytesPerSample;
        int totalFileSize =
            4                                // "AIFF"
            + 8 + 18                         // "COMM" header + 18-byte body
            + 8 + 8 + ssndPayloadSize;       // "SSND" header + 8-byte offset/block + payload
        int fullSize = 8 + totalFileSize;    // + "FORM" + size field
        var result = new byte[fullSize];
        int pos = 0;
        WriteAscii(result, pos, "FORM"); pos += 4;
        WriteUInt32Be(result, pos, (uint)(fullSize - 8)); pos += 4;
        WriteAscii(result, pos, "AIFF"); pos += 4;
        WriteAscii(result, pos, "COMM"); pos += 4;
        WriteUInt32Be(result, pos, 18); pos += 4;
        WriteInt16Be(result, pos, (short)channels); pos += 2;
        WriteUInt32Be(result, pos, (uint)sampleFrames); pos += 4;
        WriteInt16Be(result, pos, (short)bitsPerSample); pos += 2;
        WriteExtended80Be(result, pos, sampleRateHz); pos += 10;
        WriteAscii(result, pos, "SSND"); pos += 4;
        WriteUInt32Be(result, pos, (uint)(8 + ssndPayloadSize)); pos += 4;
        WriteUInt32Be(result, pos, 0); pos += 4; // offset = 0
        WriteUInt32Be(result, pos, 0); pos += 4; // block size = 0 (not used)
        for (int i = 0; i < interleavedSamples.Length; i++)
        {
            int v = interleavedSamples[i];
            // Sign-extend already in int32; write bytesPerSample bytes big-endian.
            for (int b = bytesPerSample - 1; b >= 0; b--)
                result[pos++] = (byte)(v >> (8 * b));
        }
        return result;
    }

    private static void DecodeBigEndianIntegerSamples(ReadOnlySpan<byte> bytes, int bps, int[] dest)
    {
        int bytesPerSample = (bps + 7) / 8;
        int pos = 0;
        for (int i = 0; i < dest.Length; i++)
        {
            int raw = 0;
            for (int b = 0; b < bytesPerSample; b++)
                raw = (raw << 8) | bytes[pos + b];
            int shift = 32 - bps;
            dest[i] = (raw << shift) >> shift;
            pos += bytesPerSample;
        }
    }

    /// <summary>Decode a 10-byte IEEE 754 80-bit extended-precision float (big-endian).</summary>
    private static double ReadExtended80Be(ReadOnlySpan<byte> bytes)
    {
        int sign = (bytes[0] & 0x80) >> 7;
        int exponent = ((bytes[0] & 0x7F) << 8) | bytes[1];
        ulong mantissa = 0;
        for (int i = 0; i < 8; i++) mantissa = (mantissa << 8) | bytes[2 + i];
        if (exponent == 0 && mantissa == 0) return 0.0;
        double value = mantissa * Math.Pow(2.0, exponent - 16383 - 63);
        return sign != 0 ? -value : value;
    }

    /// <summary>Encode a double as a 10-byte IEEE 754 80-bit extended float (big-endian).</summary>
    private static void WriteExtended80Be(byte[] dest, int offset, double value)
    {
        if (value == 0)
        {
            for (int i = 0; i < 10; i++) dest[offset + i] = 0;
            return;
        }
        bool negative = value < 0;
        if (negative) value = -value;
        int exponent = (int)Math.Floor(Math.Log2(value));
        double mantissaDouble = value / Math.Pow(2.0, exponent);
        // mantissaDouble is in [1, 2). Integer bit at position 63 should be set.
        ulong mantissa = (ulong)Math.Round(mantissaDouble * Math.Pow(2.0, 63));
        int biasedExp = exponent + 16383;
        ushort topWord = (ushort)((negative ? 0x8000 : 0) | (biasedExp & 0x7FFF));
        dest[offset] = (byte)(topWord >> 8);
        dest[offset + 1] = (byte)topWord;
        for (int i = 0; i < 8; i++) dest[offset + 2 + i] = (byte)(mantissa >> (8 * (7 - i)));
    }

    private static short ReadInt16Be(ReadOnlySpan<byte> s) => (short)((s[0] << 8) | s[1]);
    private static uint ReadUInt32Be(ReadOnlySpan<byte> s)
    {
        uint v = 0;
        for (int i = 0; i < 4; i++) v = (v << 8) | s[i];
        return v;
    }

    private static void WriteAscii(byte[] dest, int offset, string s)
    {
        for (int i = 0; i < s.Length; i++) dest[offset + i] = (byte)s[i];
    }

    private static void WriteInt16Be(byte[] dest, int offset, short value)
    {
        dest[offset] = (byte)(value >> 8);
        dest[offset + 1] = (byte)value;
    }

    private static void WriteUInt32Be(byte[] dest, int offset, uint value)
    {
        dest[offset] = (byte)(value >> 24);
        dest[offset + 1] = (byte)(value >> 16);
        dest[offset + 2] = (byte)(value >> 8);
        dest[offset + 3] = (byte)value;
    }
}
