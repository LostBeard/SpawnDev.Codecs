// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Vorbis I residue configuration (Section 8.6.2). Residue types 0, 1, and 2
// all use the same on-wire configuration layout; only the packet-time decode
// path differs (type 2 interleaves channels). This parser captures the shared
// configuration.

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>Which residue decode algorithm this configuration selects at packet time.</summary>
public enum VorbisResidueType
{
    /// <summary>Residue 0: non-interleaved, partition-first.</summary>
    Type0 = 0,

    /// <summary>Residue 1: non-interleaved, entry-first.</summary>
    Type1 = 1,

    /// <summary>Residue 2: format-interleaved (channels packed into partitions).</summary>
    Type2 = 2,
}

/// <summary>Parsed Vorbis residue configuration.</summary>
public sealed record VorbisResidueConfig
{
    /// <summary>Residue algorithm to use at packet decode time.</summary>
    public VorbisResidueType Type { get; init; }

    /// <summary>Spectrum bin at which residue decoding begins.</summary>
    public int Begin { get; init; }

    /// <summary>Spectrum bin at which residue decoding ends (exclusive).</summary>
    public int End { get; init; }

    /// <summary>Partition size in residue entries.</summary>
    public int PartitionSize { get; init; }

    /// <summary>Number of classification buckets; residue entries are assigned to classifications via the classbook.</summary>
    public int Classifications { get; init; }

    /// <summary>Codebook used to pick the classification for each partition.</summary>
    public int Classbook { get; init; }

    /// <summary>8-bit cascade flag per classification; bit N indicates a codebook is present at pass N.</summary>
    public required int[] Cascade { get; init; }

    /// <summary>
    /// [classification][pass] -> codebook index or -1 when absent. Inner length is
    /// always 8 (max passes per classification).
    /// </summary>
    public required int[][] Books { get; init; }
}

/// <summary>Parses a Vorbis residue configuration from the setup header bitstream.</summary>
internal static class VorbisResidueConfigParser
{
    /// <summary>
    /// Parse one residue configuration of the given <paramref name="type"/>
    /// from <paramref name="reader"/>.
    /// </summary>
    internal static VorbisResidueConfig Parse(ref VorbisBitReader reader, VorbisResidueType type)
    {
        if (type < VorbisResidueType.Type0 || type > VorbisResidueType.Type2)
            throw new InvalidDataException($"Reserved residue type {(int)type}.");

        int begin = (int)reader.ReadBits(24);
        int end = (int)reader.ReadBits(24);
        int partitionSize = (int)reader.ReadBits(24) + 1;
        int classifications = (int)reader.ReadBits(6) + 1;
        int classbook = (int)reader.ReadBits(8);

        if (end < begin)
            throw new InvalidDataException($"Residue end {end} < begin {begin}.");

        var cascade = new int[classifications];
        for (int i = 0; i < classifications; i++)
        {
            int lowBits = (int)reader.ReadBits(3);
            bool bitflag = reader.ReadBit() != 0;
            int highBits = bitflag ? (int)reader.ReadBits(5) : 0;
            cascade[i] = (highBits << 3) | lowBits;
        }

        var books = new int[classifications][];
        for (int i = 0; i < classifications; i++)
        {
            books[i] = new int[8];
            for (int j = 0; j < 8; j++)
            {
                if (((cascade[i] >> j) & 1) != 0)
                    books[i][j] = (int)reader.ReadBits(8);
                else
                    books[i][j] = -1;
            }
        }

        return new VorbisResidueConfig
        {
            Type = type,
            Begin = begin,
            End = end,
            PartitionSize = partitionSize,
            Classifications = classifications,
            Classbook = classbook,
            Cascade = cascade,
            Books = books,
        };
    }
}
