// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Vorbis identification header (packet 0). Defined in Vorbis I specification
// section 4.2.2. Carries stream geometry - channels, sample rate, bitrate
// hints, and the two blocksizes used by the MDCT.

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// Parsed Vorbis identification header. First header packet of every Vorbis
/// stream (packet type 0x01 followed by the "vorbis" magic).
/// </summary>
public sealed record VorbisIdentificationHeader
{
    /// <summary>Vorbis version - always 0 per the Vorbis I specification.</summary>
    public int VorbisVersion { get; init; }

    /// <summary>Audio channel count (1-255).</summary>
    public int AudioChannels { get; init; }

    /// <summary>Audio sample rate in Hz.</summary>
    public int SampleRateHz { get; init; }

    /// <summary>
    /// Maximum bitrate hint (bits per second). Encoder-supplied; may be 0 or negative
    /// if unknown / not provided. Stored signed on the wire.
    /// </summary>
    public int BitrateMaximum { get; init; }

    /// <summary>Nominal bitrate hint.</summary>
    public int BitrateNominal { get; init; }

    /// <summary>Minimum bitrate hint.</summary>
    public int BitrateMinimum { get; init; }

    /// <summary>Short block size in samples (power of 2, 64 through 8192).</summary>
    public int BlockSize0 { get; init; }

    /// <summary>Long block size in samples (power of 2, 64 through 8192).</summary>
    public int BlockSize1 { get; init; }
}

/// <summary>Parses the Vorbis identification header packet.</summary>
public static class VorbisIdentificationHeaderParser
{
    /// <summary>Parse the identification header bytes from an Ogg packet payload.</summary>
    public static VorbisIdentificationHeader Parse(ReadOnlySpan<byte> packet)
    {
        if (packet.Length < 30)
            throw new InvalidDataException(
                $"Vorbis identification header must be at least 30 bytes, got {packet.Length}.");
        ValidatePacketType(packet[0], expected: 0x01);
        ValidateMagic(packet.Slice(1, 6));

        int version = ReadInt32Le(packet.Slice(7, 4));
        if (version != 0)
            throw new InvalidDataException($"Vorbis version must be 0, got {version}.");
        int channels = packet[11];
        if (channels < 1)
            throw new InvalidDataException("Vorbis channels must be >= 1.");
        int sampleRate = ReadInt32Le(packet.Slice(12, 4));
        if (sampleRate < 1)
            throw new InvalidDataException("Vorbis sample rate must be >= 1.");
        int brMax = ReadInt32Le(packet.Slice(16, 4));
        int brNom = ReadInt32Le(packet.Slice(20, 4));
        int brMin = ReadInt32Le(packet.Slice(24, 4));
        byte bsByte = packet[28];
        int bs0Log = bsByte & 0x0F;
        int bs1Log = (bsByte >> 4) & 0x0F;
        if (bs0Log < 6 || bs0Log > 13 || bs1Log < 6 || bs1Log > 13)
            throw new InvalidDataException(
                $"Vorbis blocksize exponents out of range [6,13]: bs0_log={bs0Log}, bs1_log={bs1Log}.");
        if (bs0Log > bs1Log)
            throw new InvalidDataException("Vorbis blocksize_0 must be <= blocksize_1.");
        byte framing = packet[29];
        if ((framing & 1) == 0)
            throw new InvalidDataException("Vorbis identification framing flag not set.");

        return new VorbisIdentificationHeader
        {
            VorbisVersion = version,
            AudioChannels = channels,
            SampleRateHz = sampleRate,
            BitrateMaximum = brMax,
            BitrateNominal = brNom,
            BitrateMinimum = brMin,
            BlockSize0 = 1 << bs0Log,
            BlockSize1 = 1 << bs1Log,
        };
    }

    internal static void ValidatePacketType(byte actual, byte expected)
    {
        if (actual != expected)
            throw new InvalidDataException(
                $"Vorbis packet-type mismatch: expected 0x{expected:X2}, got 0x{actual:X2}.");
    }

    internal static void ValidateMagic(ReadOnlySpan<byte> bytes)
    {
        ReadOnlySpan<byte> magic = new[] { (byte)'v', (byte)'o', (byte)'r', (byte)'b', (byte)'i', (byte)'s' };
        for (int i = 0; i < 6; i++)
            if (bytes[i] != magic[i])
                throw new InvalidDataException($"Vorbis magic mismatch at byte {i}: got 0x{bytes[i]:X2}.");
    }

    private static int ReadInt32Le(ReadOnlySpan<byte> s)
    {
        uint v = 0;
        for (int i = 0; i < 4; i++) v |= (uint)s[i] << (8 * i);
        return unchecked((int)v);
    }
}
