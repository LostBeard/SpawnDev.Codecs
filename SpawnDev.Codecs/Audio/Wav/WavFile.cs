// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// WAV (RIFF/WAVE) container reader + writer for uncompressed PCM. Not a codec
// per se - WAV is just a byte container for raw PCM - but essential for
// interoperating with real audio files from our encoder/decoder pipeline.
//
// Supports:
//  - PCM integer (format tag 1) at 8 / 16 / 24 / 32 bit depths.
//  - IEEE float (format tag 3) at 32 bit depth.
//  - Mono through 8 channels.
//  - "fmt "/"data" chunks; other chunks (LIST, bext, cue, ...) are skipped.

namespace SpawnDev.Codecs.Audio.Wav;

/// <summary>
/// Parsed RIFF/WAVE file: format parameters + interleaved integer PCM samples
/// (float samples are converted to int at the declared bit depth).
/// </summary>
public sealed record WavFile
{
    /// <summary>Sample rate in Hz.</summary>
    public int SampleRateHz { get; init; }

    /// <summary>Channel count (1 = mono, 2 = stereo, ...).</summary>
    public int Channels { get; init; }

    /// <summary>Bits per sample (8, 16, 24, or 32 for integer PCM; 32 for IEEE float).</summary>
    public int BitsPerSample { get; init; }

    /// <summary>True when the source wav stored samples as 32-bit IEEE floats; false when signed integer PCM.</summary>
    public bool IsFloat { get; init; }

    /// <summary>
    /// Interleaved integer PCM samples: <c>[ch0[0], ch1[0], ch0[1], ch1[1], ...]</c>.
    /// Length = <c>TotalSamplesPerChannel</c> × <c>Channels</c>.
    /// </summary>
    public required int[] InterleavedSamples { get; init; }

    /// <summary>Total samples per channel.</summary>
    public int TotalSamplesPerChannel { get; init; }
}

/// <summary>Reader and writer for WAV (RIFF/WAVE) uncompressed PCM files.</summary>
public static class WavFileCodec
{
    /// <summary>Read a WAV file from a byte buffer.</summary>
    public static WavFile Read(ReadOnlySpan<byte> data)
    {
        if (data.Length < 44)
            throw new InvalidDataException($"WAV file must be at least 44 bytes, got {data.Length}.");
        if (data[0] != 'R' || data[1] != 'I' || data[2] != 'F' || data[3] != 'F')
            throw new InvalidDataException("Missing 'RIFF' magic.");
        // bytes[4..7] = file size - 8 (ignored, we use explicit chunk lengths).
        if (data[8] != 'W' || data[9] != 'A' || data[10] != 'V' || data[11] != 'E')
            throw new InvalidDataException("Missing 'WAVE' format identifier.");

        int pos = 12;
        int sampleRate = 0, channels = 0, bps = 0;
        bool isFloat = false;
        ReadOnlySpan<byte> sampleBytes = default;
        bool sawFmt = false, sawData = false;

        while (pos + 8 <= data.Length)
        {
            string chunkId = System.Text.Encoding.ASCII.GetString(data.Slice(pos, 4));
            uint chunkSize = ReadUInt32Le(data.Slice(pos + 4, 4));
            int chunkStart = pos + 8;
            if (chunkStart + chunkSize > data.Length)
                throw new InvalidDataException($"WAV chunk '{chunkId}' extends past end of file.");
            if (chunkId == "fmt ")
            {
                if (chunkSize < 16)
                    throw new InvalidDataException($"WAV 'fmt ' chunk too small: {chunkSize} bytes.");
                ushort formatTag = (ushort)ReadUInt16Le(data.Slice(chunkStart, 2));
                channels = ReadUInt16Le(data.Slice(chunkStart + 2, 2));
                sampleRate = (int)ReadUInt32Le(data.Slice(chunkStart + 4, 4));
                // byte rate (4) + block align (2) ignored - we derive from bps/channels.
                bps = ReadUInt16Le(data.Slice(chunkStart + 14, 2));
                if (formatTag == 1) isFloat = false;
                else if (formatTag == 3) isFloat = true;
                else if (formatTag == 0xFFFE)
                {
                    // WAVE_FORMAT_EXTENSIBLE: first 2 bytes of the subformat GUID carry the real tag.
                    if (chunkSize < 40 || chunkStart + 24 + 2 > data.Length)
                        throw new InvalidDataException("WAV EXTENSIBLE chunk too short for sub-format.");
                    ushort subTag = (ushort)ReadUInt16Le(data.Slice(chunkStart + 24, 2));
                    if (subTag == 1) isFloat = false;
                    else if (subTag == 3) isFloat = true;
                    else throw new InvalidDataException($"WAV EXTENSIBLE sub-format tag 0x{subTag:X4} unsupported.");
                }
                else throw new InvalidDataException($"WAV format tag 0x{formatTag:X4} unsupported (only PCM=1 and IEEE float=3).");
                sawFmt = true;
            }
            else if (chunkId == "data")
            {
                sampleBytes = data.Slice(chunkStart, (int)chunkSize);
                sawData = true;
            }
            // Other chunk ids are skipped.
            pos = chunkStart + (int)chunkSize;
            if ((chunkSize & 1) != 0 && pos < data.Length) pos++; // chunks pad to word boundary
        }
        if (!sawFmt) throw new InvalidDataException("WAV missing 'fmt ' chunk.");
        if (!sawData) throw new InvalidDataException("WAV missing 'data' chunk.");
        if (channels < 1) throw new InvalidDataException("WAV invalid channel count.");
        if (sampleRate < 1) throw new InvalidDataException("WAV invalid sample rate.");
        if (bps is not (8 or 16 or 24 or 32))
            throw new InvalidDataException($"WAV bit depth {bps} unsupported (use 8/16/24/32).");

        int bytesPerSample = bps / 8;
        int totalInterleavedSamples = sampleBytes.Length / bytesPerSample;
        if (totalInterleavedSamples % channels != 0)
            throw new InvalidDataException(
                $"WAV data chunk size not a multiple of channels*bps/8: {sampleBytes.Length} bytes.");
        int[] samples = new int[totalInterleavedSamples];
        if (!isFloat)
        {
            DecodeIntegerSamples(sampleBytes, bps, samples);
        }
        else
        {
            DecodeFloatSamples(sampleBytes, bps, samples);
        }

        return new WavFile
        {
            SampleRateHz = sampleRate,
            Channels = channels,
            BitsPerSample = bps,
            IsFloat = isFloat,
            InterleavedSamples = samples,
            TotalSamplesPerChannel = totalInterleavedSamples / channels,
        };
    }

    /// <summary>Write interleaved integer PCM samples to a WAV byte buffer (format tag PCM = 1).</summary>
    public static byte[] Write(ReadOnlySpan<int> interleavedSamples, int sampleRateHz, int channels, int bitsPerSample)
    {
        if (channels < 1) throw new ArgumentException("channels >= 1 required.", nameof(channels));
        if (bitsPerSample is not (8 or 16 or 24 or 32))
            throw new ArgumentException("bitsPerSample must be 8, 16, 24, or 32.", nameof(bitsPerSample));
        if (interleavedSamples.Length % channels != 0)
            throw new ArgumentException("interleavedSamples length must be a multiple of channels.");
        if (sampleRateHz < 1) throw new ArgumentException("sampleRateHz must be positive.", nameof(sampleRateHz));

        int bytesPerSample = bitsPerSample / 8;
        int dataBytes = interleavedSamples.Length * bytesPerSample;
        int totalSize = 44 + dataBytes;
        var result = new byte[totalSize];
        int pos = 0;
        WriteAscii(result, pos, "RIFF"); pos += 4;
        WriteUInt32Le(result, pos, (uint)(totalSize - 8)); pos += 4;
        WriteAscii(result, pos, "WAVE"); pos += 4;
        WriteAscii(result, pos, "fmt "); pos += 4;
        WriteUInt32Le(result, pos, 16); pos += 4;          // fmt chunk length
        WriteUInt16Le(result, pos, 1); pos += 2;           // format tag PCM
        WriteUInt16Le(result, pos, (ushort)channels); pos += 2;
        WriteUInt32Le(result, pos, (uint)sampleRateHz); pos += 4;
        int byteRate = sampleRateHz * channels * bytesPerSample;
        WriteUInt32Le(result, pos, (uint)byteRate); pos += 4;
        int blockAlign = channels * bytesPerSample;
        WriteUInt16Le(result, pos, (ushort)blockAlign); pos += 2;
        WriteUInt16Le(result, pos, (ushort)bitsPerSample); pos += 2;
        WriteAscii(result, pos, "data"); pos += 4;
        WriteUInt32Le(result, pos, (uint)dataBytes); pos += 4;

        // 8-bit WAV is UNSIGNED (DC offset 128); 16/24/32 are signed little-endian.
        for (int i = 0; i < interleavedSamples.Length; i++)
        {
            int v = interleavedSamples[i];
            if (bitsPerSample == 8)
            {
                result[pos++] = (byte)(v + 128);
            }
            else
            {
                for (int b = 0; b < bytesPerSample; b++)
                    result[pos++] = (byte)(v >> (8 * b));
            }
        }
        return result;
    }

    private static void DecodeIntegerSamples(ReadOnlySpan<byte> bytes, int bps, int[] dest)
    {
        int bytesPerSample = bps / 8;
        int pos = 0;
        for (int i = 0; i < dest.Length; i++)
        {
            int v;
            if (bps == 8)
            {
                // WAV 8-bit PCM is unsigned with offset 128.
                v = bytes[pos] - 128;
            }
            else
            {
                int raw = 0;
                for (int b = 0; b < bytesPerSample; b++)
                    raw |= bytes[pos + b] << (8 * b);
                // Sign-extend from `bps` bits to 32 bits.
                int shift = 32 - bps;
                v = (raw << shift) >> shift;
            }
            dest[i] = v;
            pos += bytesPerSample;
        }
    }

    private static void DecodeFloatSamples(ReadOnlySpan<byte> bytes, int bps, int[] dest)
    {
        if (bps != 32)
            throw new InvalidDataException($"WAV float samples must be 32-bit, got {bps}-bit.");
        int pos = 0;
        for (int i = 0; i < dest.Length; i++)
        {
            uint bits = 0;
            for (int b = 0; b < 4; b++) bits |= (uint)bytes[pos + b] << (8 * b);
            float f = BitConverter.UInt32BitsToSingle(bits);
            // Convert to 32-bit signed PCM; clip to [-1.0, 1.0] then scale.
            float clamped = Math.Clamp(f, -1.0f, 1.0f);
            dest[i] = (int)(clamped * int.MaxValue);
            pos += 4;
        }
    }

    private static void WriteAscii(byte[] dest, int offset, string s)
    {
        for (int i = 0; i < s.Length; i++) dest[offset + i] = (byte)s[i];
    }

    private static void WriteUInt16Le(byte[] dest, int offset, ushort value)
    {
        dest[offset] = (byte)value;
        dest[offset + 1] = (byte)(value >> 8);
    }

    private static void WriteUInt32Le(byte[] dest, int offset, uint value)
    {
        for (int i = 0; i < 4; i++) dest[offset + i] = (byte)(value >> (8 * i));
    }

    private static int ReadUInt16Le(ReadOnlySpan<byte> s) => s[0] | (s[1] << 8);

    private static uint ReadUInt32Le(ReadOnlySpan<byte> s)
    {
        uint v = 0;
        for (int i = 0; i < 4; i++) v |= (uint)s[i] << (8 * i);
        return v;
    }
}
