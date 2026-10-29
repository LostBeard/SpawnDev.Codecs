// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Vorbis audio packet header per Vorbis I Section 4.3.1. Every audio packet
// begins with:
//   1 bit: packet type (must be 0 for audio)
//   ilog(modes - 1) bits: mode number
// If the selected mode uses the long block size, two additional 1-bit flags
// (previous and next window) describe how this block's window overlaps with
// its neighbours. Short-block modes have no window flags.

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>Parsed Vorbis audio packet header.</summary>
public sealed record VorbisAudioPacketHeader
{
    /// <summary>Which mode (from the setup header mode list) this packet uses.</summary>
    public int ModeNumber { get; init; }

    /// <summary>Effective block size in samples for this packet.</summary>
    public int BlockSize { get; init; }

    /// <summary>True when <see cref="ModeNumber"/> selects a long-block mode.</summary>
    public bool IsLongBlock { get; init; }

    /// <summary>Previous-window flag; only meaningful for long blocks.</summary>
    public bool PreviousWindowLong { get; init; }

    /// <summary>Next-window flag; only meaningful for long blocks.</summary>
    public bool NextWindowLong { get; init; }
}

/// <summary>Parses the per-audio-packet header using a Vorbis setup header + ident header for context.</summary>
public static class VorbisAudioPacketHeaderParser
{
    /// <summary>
    /// Parse the audio packet header from the start of an Ogg-assembled Vorbis
    /// audio packet. <paramref name="setup"/> supplies the mode list;
    /// <paramref name="ident"/> supplies block sizes.
    /// </summary>
    public static VorbisAudioPacketHeader Parse(
        ReadOnlySpan<byte> packetData,
        VorbisSetupHeader setup,
        VorbisIdentificationHeader ident)
    {
        var reader = new VorbisBitReader(packetData);
        return ParseFromReader(ref reader, setup, ident);
    }

    internal static VorbisAudioPacketHeader ParseFromReader(
        ref VorbisBitReader reader,
        VorbisSetupHeader setup,
        VorbisIdentificationHeader ident)
    {
        uint packetType = reader.ReadBit();
        if (packetType != 0)
            throw new InvalidDataException($"Vorbis audio packet type must be 0 (audio), got 1 (header).");

        int modeCount = setup.Modes.Length;
        int modeBits = VorbisMath.Ilog(modeCount - 1);
        int modeNumber = modeBits > 0 ? (int)reader.ReadBits(modeBits) : 0;
        if (modeNumber < 0 || modeNumber >= modeCount)
            throw new InvalidDataException(
                $"Vorbis audio packet mode {modeNumber} out of range [0, {modeCount}).");

        var mode = setup.Modes[modeNumber];
        int blockSize = mode.BlockFlag ? ident.BlockSize1 : ident.BlockSize0;

        bool prevWindowLong = false;
        bool nextWindowLong = false;
        if (mode.BlockFlag)
        {
            prevWindowLong = reader.ReadBit() != 0;
            nextWindowLong = reader.ReadBit() != 0;
        }

        return new VorbisAudioPacketHeader
        {
            ModeNumber = modeNumber,
            BlockSize = blockSize,
            IsLongBlock = mode.BlockFlag,
            PreviousWindowLong = prevWindowLong,
            NextWindowLong = nextWindowLong,
        };
    }
}
