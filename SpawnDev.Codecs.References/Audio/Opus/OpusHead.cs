// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Opus-in-Ogg identification header per RFC 7845 Section 5.1. The first Ogg
// packet of an Opus stream carries this structure and tells the decoder the
// channel count, pre-skip, input sample rate hint, output gain, and channel
// mapping family.

namespace SpawnDev.Codecs.Audio.Opus;

/// <summary>
/// Parsed Opus-in-Ogg identification header ("OpusHead").
/// </summary>
public sealed record OpusHead
{
    /// <summary>Header version byte. Known valid values are 1.x (accept lower nibble, per RFC 7845).</summary>
    public int Version { get; init; }

    /// <summary>Output channel count (1 through 8 for standard channel mappings).</summary>
    public int OutputChannels { get; init; }

    /// <summary>Pre-skip in samples at 48 kHz (decoded samples to drop from the front of the first frame).</summary>
    public int PreSkip { get; init; }

    /// <summary>
    /// Original input sample rate hint from the encoder, in Hz. Informational only -
    /// Opus itself always decodes to 48 kHz internally.
    /// </summary>
    public uint InputSampleRateHz { get; init; }

    /// <summary>Output gain in Q7.8 dB. Applied as a fixed per-sample scale during playback.</summary>
    public int OutputGainQ7_8 { get; init; }

    /// <summary>
    /// Channel mapping family: 0 for mono/stereo, 1 for Vorbis surround mappings,
    /// 2+ for ambisonics / future extensions.
    /// </summary>
    public int ChannelMappingFamily { get; init; }

    /// <summary>
    /// Channel mapping table (stream count + coupled-stream count + per-channel mapping).
    /// Null when <see cref="ChannelMappingFamily"/> == 0.
    /// </summary>
    public OpusChannelMapping? ChannelMapping { get; init; }
}

/// <summary>
/// Optional channel mapping table for mapping family >= 1.
/// </summary>
public sealed record OpusChannelMapping
{
    /// <summary>Total Opus substreams in this logical stream.</summary>
    public int StreamCount { get; init; }

    /// <summary>Number of Opus substreams that are coupled stereo pairs.</summary>
    public int CoupledCount { get; init; }

    /// <summary>Per-output-channel index into the Opus substream output (length = OutputChannels).</summary>
    public required byte[] Mapping { get; init; }
}

/// <summary>Parses the "OpusHead" identification header from an Ogg packet.</summary>
public static class OpusHeadParser
{
    private static readonly byte[] Magic = { (byte)'O', (byte)'p', (byte)'u', (byte)'s', (byte)'H', (byte)'e', (byte)'a', (byte)'d' };

    /// <summary>Parse an OpusHead packet. Length must be at least 19 bytes for family 0.</summary>
    public static OpusHead Parse(ReadOnlySpan<byte> packet)
    {
        if (packet.Length < 19)
            throw new InvalidDataException($"OpusHead must be at least 19 bytes, got {packet.Length}.");
        for (int i = 0; i < 8; i++)
        {
            if (packet[i] != Magic[i])
                throw new InvalidDataException(
                    $"OpusHead magic mismatch at byte {i}: expected 0x{Magic[i]:X2}, got 0x{packet[i]:X2}.");
        }
        int version = packet[8];
        if ((version & 0xF0) != 0)
            throw new InvalidDataException($"Unsupported OpusHead version major: 0x{version:X2}.");
        int channels = packet[9];
        if (channels < 1)
            throw new InvalidDataException("OpusHead channel count must be >= 1.");
        int preSkip = packet[10] | (packet[11] << 8);
        uint inputRate = (uint)packet[12] | ((uint)packet[13] << 8) | ((uint)packet[14] << 16) | ((uint)packet[15] << 24);
        int gain = (short)(packet[16] | (packet[17] << 8));
        int family = packet[18];

        OpusChannelMapping? mapping = null;
        if (family != 0)
        {
            int mapSize = 2 + channels;
            if (packet.Length < 19 + mapSize)
                throw new InvalidDataException(
                    $"OpusHead channel mapping truncated: need {mapSize} more bytes, have {packet.Length - 19}.");
            int streamCount = packet[19];
            int coupledCount = packet[20];
            if (streamCount < 1 || coupledCount > streamCount)
                throw new InvalidDataException(
                    $"OpusHead invalid stream/coupled counts: stream={streamCount}, coupled={coupledCount}.");
            byte[] map = new byte[channels];
            for (int i = 0; i < channels; i++) map[i] = packet[21 + i];
            mapping = new OpusChannelMapping
            {
                StreamCount = streamCount,
                CoupledCount = coupledCount,
                Mapping = map,
            };
        }

        return new OpusHead
        {
            Version = version,
            OutputChannels = channels,
            PreSkip = preSkip,
            InputSampleRateHz = inputRate,
            OutputGainQ7_8 = gain,
            ChannelMappingFamily = family,
            ChannelMapping = mapping,
        };
    }
}
