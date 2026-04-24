// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Full Vorbis setup header parser per Vorbis I Section 4.2.4. Consumes the
// third header packet of a Vorbis stream and produces the parsed configuration
// lists (codebooks, floor 1 configs, residue configs, mappings, modes) that
// audio-packet decoders rely on.
//
// Floor type 0 is a rarely-used legacy option; this parser throws
// NotSupportedException for it and defers the implementation to a later slice
// if ever needed.

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>Parsed Vorbis setup header (the third header packet).</summary>
public sealed record VorbisSetupHeader
{
    /// <summary>All parsed codebooks in declaration order.</summary>
    public required VorbisCodebook[] Codebooks { get; init; }

    /// <summary>Floor configurations; each entry is a <see cref="VorbisFloor1Config"/> (floor 0 is not supported).</summary>
    public required VorbisFloor1Config[] Floors { get; init; }

    /// <summary>Residue configurations.</summary>
    public required VorbisResidueConfig[] Residues { get; init; }

    /// <summary>Mapping configurations.</summary>
    public required VorbisMappingConfig[] Mappings { get; init; }

    /// <summary>Modes; the audio packet header selects one by index.</summary>
    public required VorbisModeConfig[] Modes { get; init; }
}

/// <summary>Parses the Vorbis setup header (packet type 5).</summary>
public static class VorbisSetupHeaderParser
{
    private const byte PacketType = 0x05;
    private static readonly byte[] Magic =
        { (byte)'v', (byte)'o', (byte)'r', (byte)'b', (byte)'i', (byte)'s' };

    /// <summary>
    /// Parse a setup header packet. <paramref name="audioChannels"/> is the
    /// channel count from the identification header (needed for mapping
    /// parse).
    /// </summary>
    public static VorbisSetupHeader Parse(ReadOnlySpan<byte> packet, int audioChannels)
    {
        if (packet.Length < 8)
            throw new InvalidDataException($"Vorbis setup header too short: {packet.Length}.");
        if (packet[0] != PacketType)
            throw new InvalidDataException($"Vorbis setup packet type must be 0x{PacketType:X2}, got 0x{packet[0]:X2}.");
        for (int i = 0; i < 6; i++)
            if (packet[1 + i] != Magic[i])
                throw new InvalidDataException($"Vorbis setup magic mismatch at byte {i + 1}.");

        // Skip the 7-byte packet type + magic before reading LSB-first bits.
        var reader = new VorbisBitReader(packet.Slice(7));

        // ----- Codebooks -----
        int codebookCount = (int)reader.ReadBits(8) + 1;
        var codebooks = new VorbisCodebook[codebookCount];
        for (int i = 0; i < codebookCount; i++)
            codebooks[i] = VorbisCodebookParser.Parse(ref reader);

        // ----- Time (legacy placeholders, always 0) -----
        int timeCount = (int)reader.ReadBits(6) + 1;
        for (int i = 0; i < timeCount; i++)
        {
            int t = (int)reader.ReadBits(16);
            if (t != 0)
                throw new InvalidDataException($"Vorbis time_count entry {i} must be 0, got {t}.");
        }

        // ----- Floors -----
        int floorCount = (int)reader.ReadBits(6) + 1;
        var floors = new VorbisFloor1Config[floorCount];
        for (int i = 0; i < floorCount; i++)
        {
            int type = (int)reader.ReadBits(16);
            if (type == 0)
                throw new NotSupportedException("Vorbis floor type 0 is not yet supported (floor type 1 is the universal choice in modern encoders).");
            if (type != 1)
                throw new InvalidDataException($"Reserved floor type {type}.");
            floors[i] = VorbisFloor1ConfigParser.Parse(ref reader);
        }

        // ----- Residues -----
        int residueCount = (int)reader.ReadBits(6) + 1;
        var residues = new VorbisResidueConfig[residueCount];
        for (int i = 0; i < residueCount; i++)
        {
            int type = (int)reader.ReadBits(16);
            residues[i] = VorbisResidueConfigParser.Parse(ref reader, (VorbisResidueType)type);
        }

        // ----- Mappings -----
        int mappingCount = (int)reader.ReadBits(6) + 1;
        var mappings = new VorbisMappingConfig[mappingCount];
        for (int i = 0; i < mappingCount; i++)
        {
            int type = (int)reader.ReadBits(16);
            if (type != 0)
                throw new InvalidDataException($"Vorbis mapping type {type} is not defined (only type 0 exists).");
            mappings[i] = VorbisMappingConfigParser.Parse(ref reader, audioChannels);
        }

        // ----- Modes -----
        int modeCount = (int)reader.ReadBits(6) + 1;
        var modes = VorbisModeConfigParser.Parse(ref reader, modeCount);

        return new VorbisSetupHeader
        {
            Codebooks = codebooks,
            Floors = floors,
            Residues = residues,
            Mappings = mappings,
            Modes = modes,
        };
    }
}
