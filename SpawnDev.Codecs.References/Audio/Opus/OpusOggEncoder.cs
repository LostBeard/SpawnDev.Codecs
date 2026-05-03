// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Package pre-encoded Opus packets into a valid Ogg-Opus byte stream per
// RFC 7845. This is a packaging layer; the caller supplies Opus packets from
// any source (e.g. libopus, Concentus, a network stream, or the future
// SpawnDev.Codecs Opus encoder) and this helper emits OpusHead + OpusTags +
// properly-granuled audio pages.

using SpawnDev.Codecs.Container.Ogg;

namespace SpawnDev.Codecs.Audio.Opus;

/// <summary>Options for the Opus-in-Ogg encoder.</summary>
public sealed record OpusOggEncoderOptions
{
    /// <summary>Output channel count (1-2; multi-stream surround is not yet supported).</summary>
    public int OutputChannels { get; init; } = 2;

    /// <summary>Pre-skip samples at 48 kHz (encoder lookahead; typical value 312).</summary>
    public int PreSkip { get; init; } = 312;

    /// <summary>Input sample rate hint in Hz. Informational only; Opus always decodes at 48 kHz.</summary>
    public uint InputSampleRateHz { get; init; } = 48000;

    /// <summary>Output gain in Q7.8 dB (0 = unity).</summary>
    public int OutputGainQ7_8 { get; init; } = 0;

    /// <summary>Vendor string for the OpusTags packet.</summary>
    public string Vendor { get; init; } = "SpawnDev.Codecs";

    /// <summary>Optional user comments for OpusTags (format "TAG=value").</summary>
    public IReadOnlyList<string>? UserComments { get; init; }

    /// <summary>Bitstream serial number; defaults to a random-ish value if 0.</summary>
    public uint BitstreamSerial { get; init; } = 0;
}

/// <summary>Wraps pre-encoded Opus packets into a valid Opus-in-Ogg byte stream.</summary>
public static class OpusOggEncoder
{
    /// <summary>Encode a sequence of Opus packets to a complete .opus byte stream.</summary>
    /// <param name="opusPackets">Each byte array is a complete Opus packet as produced by an Opus encoder.</param>
    /// <param name="options">Stream parameters (channels, pre-skip, metadata).</param>
    public static byte[] Encode(IReadOnlyList<byte[]> opusPackets, OpusOggEncoderOptions options)
    {
        if (opusPackets is null) throw new ArgumentNullException(nameof(opusPackets));
        if (options is null) throw new ArgumentNullException(nameof(options));
        if (options.OutputChannels < 1 || options.OutputChannels > 2)
            throw new ArgumentException("Only 1 or 2 output channels supported (multi-stream deferred).",
                nameof(options));

        uint serial = options.BitstreamSerial != 0
            ? options.BitstreamSerial
            : (uint)Random.Shared.Next(1, int.MaxValue);

        byte[] headPacket = BuildOpusHeadPacket(options);
        byte[] tagsPacket = BuildOpusTagsPacket(options);

        var outgoing = new List<OggOutgoingPacket>
        {
            new OggOutgoingPacket { Data = headPacket, GranulePosition = 0 },
            new OggOutgoingPacket { Data = tagsPacket, GranulePosition = 0 },
        };
        long runningSamples = 0;
        for (int i = 0; i < opusPackets.Count; i++)
        {
            int packetSamples48k = CountSamplesAt48k(opusPackets[i]);
            runningSamples += packetSamples48k;
            outgoing.Add(new OggOutgoingPacket
            {
                Data = opusPackets[i],
                GranulePosition = runningSamples,
            });
        }
        return OggPageWriter.WriteStream(serial, outgoing);
    }

    /// <summary>Build a minimal OpusHead packet (channel mapping family 0).</summary>
    private static byte[] BuildOpusHeadPacket(OpusOggEncoderOptions options)
    {
        // 19 fixed bytes for family-0 OpusHead.
        var bytes = new byte[19];
        byte[] magic = { (byte)'O', (byte)'p', (byte)'u', (byte)'s', (byte)'H', (byte)'e', (byte)'a', (byte)'d' };
        Array.Copy(magic, bytes, 8);
        bytes[8] = 1;                            // version
        bytes[9] = (byte)options.OutputChannels;
        bytes[10] = (byte)options.PreSkip;
        bytes[11] = (byte)(options.PreSkip >> 8);
        for (int i = 0; i < 4; i++) bytes[12 + i] = (byte)(options.InputSampleRateHz >> (8 * i));
        bytes[16] = (byte)options.OutputGainQ7_8;
        bytes[17] = (byte)(options.OutputGainQ7_8 >> 8);
        bytes[18] = 0;                           // channel mapping family
        return bytes;
    }

    /// <summary>Build an OpusTags packet with vendor + optional user comments.</summary>
    private static byte[] BuildOpusTagsPacket(OpusOggEncoderOptions options)
    {
        byte[] vendorBytes = System.Text.Encoding.UTF8.GetBytes(options.Vendor);
        var comments = options.UserComments ?? Array.Empty<string>();
        byte[][] cmtBytes = comments.Select(c => System.Text.Encoding.UTF8.GetBytes(c)).ToArray();
        int size = 8 + 4 + vendorBytes.Length + 4;
        foreach (var cb in cmtBytes) size += 4 + cb.Length;
        var bytes = new byte[size];
        int pos = 0;
        foreach (var c in "OpusTags") bytes[pos++] = (byte)c;
        WriteUInt32Le(bytes, pos, (uint)vendorBytes.Length); pos += 4;
        Array.Copy(vendorBytes, 0, bytes, pos, vendorBytes.Length); pos += vendorBytes.Length;
        WriteUInt32Le(bytes, pos, (uint)cmtBytes.Length); pos += 4;
        foreach (var cb in cmtBytes)
        {
            WriteUInt32Le(bytes, pos, (uint)cb.Length); pos += 4;
            Array.Copy(cb, 0, bytes, pos, cb.Length); pos += cb.Length;
        }
        return bytes;
    }

    /// <summary>
    /// Derive the 48 kHz sample count this Opus packet produces from its TOC byte.
    /// Used to advance the Ogg granule position correctly.
    /// </summary>
    private static int CountSamplesAt48k(byte[] opusPacket)
    {
        if (opusPacket.Length == 0)
            throw new ArgumentException("Empty Opus packet.");
        var toc = new OpusTocByte(opusPacket[0]);
        int perFrame = toc.GetSamplesPerFrame(48_000);
        int frameCount = toc.FrameCountCode switch
        {
            0 => 1,
            1 => 2,
            2 => 2,
            _ => InferCode3FrameCount(opusPacket),
        };
        return perFrame * frameCount;
    }

    private static int InferCode3FrameCount(byte[] opusPacket)
    {
        if (opusPacket.Length < 2) return 1;
        // Code 3 TOC: byte 1 low 6 bits = frame count (1-48).
        int count = opusPacket[1] & 0x3F;
        return Math.Max(1, count);
    }

    private static void WriteUInt32Le(byte[] dest, int offset, uint value)
    {
        for (int i = 0; i < 4; i++) dest[offset + i] = (byte)(value >> (8 * i));
    }
}
