// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Vorbis I mapping type 0 configuration (Section 8.7.3). Mappings tell the
// audio-packet decoder which floor and residue to apply per channel/submap
// and describe any coupling steps used to decorrelate channels.

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// Parsed Vorbis mapping type 0 configuration.
/// </summary>
public sealed record VorbisMappingConfig
{
    /// <summary>Number of submaps (independent floor/residue pairs) under this mapping.</summary>
    public int Submaps { get; init; }

    /// <summary>Per-coupling-step magnitude channel index.</summary>
    public required int[] CouplingMagnitudeChannels { get; init; }

    /// <summary>Per-coupling-step angle channel index.</summary>
    public required int[] CouplingAngleChannels { get; init; }

    /// <summary>Per-channel index into <see cref="Submaps"/>.</summary>
    public required int[] Mux { get; init; }

    /// <summary>Per-submap floor configuration index.</summary>
    public required int[] SubmapFloor { get; init; }

    /// <summary>Per-submap residue configuration index.</summary>
    public required int[] SubmapResidue { get; init; }
}

/// <summary>Parses Vorbis mapping type 0 from the setup header bitstream.</summary>
internal static class VorbisMappingConfigParser
{
    /// <summary>Parse one mapping type 0. <paramref name="audioChannels"/> is
    /// the stream's channel count from the identification header.</summary>
    internal static VorbisMappingConfig Parse(ref VorbisBitReader reader, int audioChannels)
    {
        // Optional submap count.
        bool submapsFlag = reader.ReadBit() != 0;
        int submaps = submapsFlag ? (int)reader.ReadBits(4) + 1 : 1;

        // Optional coupling table.
        bool couplingFlag = reader.ReadBit() != 0;
        int couplingSteps = couplingFlag ? (int)reader.ReadBits(8) + 1 : 0;
        var couplingMag = new int[couplingSteps];
        var couplingAng = new int[couplingSteps];
        if (couplingSteps > 0)
        {
            int couplingIndexBits = VorbisMath.Ilog(audioChannels - 1);
            for (int i = 0; i < couplingSteps; i++)
            {
                couplingMag[i] = (int)reader.ReadBits(couplingIndexBits);
                couplingAng[i] = (int)reader.ReadBits(couplingIndexBits);
                if (couplingMag[i] == couplingAng[i] || couplingMag[i] >= audioChannels || couplingAng[i] >= audioChannels)
                    throw new InvalidDataException(
                        $"Invalid coupling pair: mag={couplingMag[i]}, ang={couplingAng[i]}, channels={audioChannels}.");
            }
        }

        int reserved = (int)reader.ReadBits(2);
        if (reserved != 0)
            throw new InvalidDataException($"Mapping type 0 reserved bits must be 0, got 0b{reserved:B2}.");

        // Per-channel mux.
        var mux = new int[audioChannels];
        if (submaps > 1)
        {
            for (int j = 0; j < audioChannels; j++)
            {
                mux[j] = (int)reader.ReadBits(4);
                if (mux[j] >= submaps)
                    throw new InvalidDataException(
                        $"Channel {j} mux index {mux[j]} exceeds submap count {submaps}.");
            }
        }

        // Per-submap floor + residue indices.
        var submapFloor = new int[submaps];
        var submapResidue = new int[submaps];
        for (int j = 0; j < submaps; j++)
        {
            _ = reader.ReadBits(8); // reserved / time-domain placeholder
            submapFloor[j] = (int)reader.ReadBits(8);
            submapResidue[j] = (int)reader.ReadBits(8);
        }

        return new VorbisMappingConfig
        {
            Submaps = submaps,
            CouplingMagnitudeChannels = couplingMag,
            CouplingAngleChannels = couplingAng,
            Mux = mux,
            SubmapFloor = submapFloor,
            SubmapResidue = submapResidue,
        };
    }
}
