// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Vorbis I mode configuration (Section 8.8.1). The audio-packet header reads
// a mode index; the selected VorbisModeConfig then selects block size
// (short/long) and the mapping to use.

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>One Vorbis audio mode.</summary>
public sealed record VorbisModeConfig
{
    /// <summary>True if this mode uses the long block size (blocksize_1), false for the short (blocksize_0).</summary>
    public bool BlockFlag { get; init; }

    /// <summary>Window type (always 0 in Vorbis I).</summary>
    public int WindowType { get; init; }

    /// <summary>Transform type (always 0 in Vorbis I).</summary>
    public int TransformType { get; init; }

    /// <summary>Mapping configuration index.</summary>
    public int Mapping { get; init; }
}

/// <summary>
/// Parses the mode configuration list at the end of the Vorbis setup header,
/// including the mandatory framing-flag check.
/// </summary>
internal static class VorbisModeConfigParser
{
    /// <summary>Parse <paramref name="modeCount"/> modes + the framing-flag terminator.</summary>
    internal static VorbisModeConfig[] Parse(ref VorbisBitReader reader, int modeCount)
    {
        var modes = new VorbisModeConfig[modeCount];
        for (int i = 0; i < modeCount; i++)
        {
            bool blockFlag = reader.ReadBit() != 0;
            int windowType = (int)reader.ReadBits(16);
            int transformType = (int)reader.ReadBits(16);
            int mapping = (int)reader.ReadBits(8);
            if (windowType != 0)
                throw new InvalidDataException(
                    $"Vorbis mode {i}: window_type must be 0, got {windowType}.");
            if (transformType != 0)
                throw new InvalidDataException(
                    $"Vorbis mode {i}: transform_type must be 0, got {transformType}.");
            modes[i] = new VorbisModeConfig
            {
                BlockFlag = blockFlag,
                WindowType = windowType,
                TransformType = transformType,
                Mapping = mapping,
            };
        }
        int framing = (int)reader.ReadBit();
        if (framing != 1)
            throw new InvalidDataException("Vorbis setup header framing flag must be 1.");
        return modes;
    }
}
