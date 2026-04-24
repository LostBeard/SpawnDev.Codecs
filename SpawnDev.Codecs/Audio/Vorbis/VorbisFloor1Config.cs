// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Vorbis I floor type 1 configuration header (Section 7.2.2). Every modern
// Vorbis encoder uses floor 1 - floor 0 is legacy and rare. This record
// captures the per-floor configuration read from the setup header; actual
// per-packet floor 1 decoding (reading the posteriors and rebuilding the
// piecewise-linear spectral envelope) happens at audio-packet decode time.

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// Parsed configuration for a Vorbis floor type 1.
/// </summary>
public sealed record VorbisFloor1Config
{
    /// <summary>Number of partitions that describe the floor curve.</summary>
    public int Partitions { get; init; }

    /// <summary>Class index for each partition (length = <see cref="Partitions"/>).</summary>
    public required int[] PartitionClassList { get; init; }

    /// <summary>Dimensions per class (posteriors per partition). Length = <c>MaximumClass + 1</c>.</summary>
    public required int[] ClassDimensions { get; init; }

    /// <summary>Log2 subclass count per class; 0 means "single codebook" (no subclass bits).</summary>
    public required int[] ClassSubclasses { get; init; }

    /// <summary>Master codebook index per class (only used when <c>ClassSubclasses &gt; 0</c>).</summary>
    public required int[] ClassMasterbooks { get; init; }

    /// <summary>
    /// Per-class, per-subclass codebook index. Length of outer array = number of
    /// classes. Inner length for class <c>c</c> = <c>1 &lt;&lt; ClassSubclasses[c]</c>.
    /// Value <c>-1</c> means "no codebook - force posterior to 0".
    /// </summary>
    public required int[][] ClassSubclassBooks { get; init; }

    /// <summary>Floor curve quantization multiplier (1..4); picks the output LSP quantum.</summary>
    public int Multiplier { get; init; }

    /// <summary>Number of bits used to encode each X coordinate in the setup.</summary>
    public int RangeBits { get; init; }

    /// <summary>
    /// Floor breakpoint X coordinates. The first two entries are always <c>0</c>
    /// and <c>1 &lt;&lt; RangeBits</c>; subsequent entries come from the setup.
    /// Length = <c>2 + sum(ClassDimensions[PartitionClassList[i]] for i in 0..Partitions-1)</c>.
    /// </summary>
    public required int[] XList { get; init; }
}

/// <summary>Parses Vorbis floor 1 configuration from the setup header bitstream.</summary>
internal static class VorbisFloor1ConfigParser
{
    internal static VorbisFloor1Config Parse(ref VorbisBitReader reader)
    {
        int partitions = (int)reader.ReadBits(5);
        var partitionClassList = new int[partitions];
        int maximumClass = -1;
        for (int i = 0; i < partitions; i++)
        {
            partitionClassList[i] = (int)reader.ReadBits(4);
            if (partitionClassList[i] > maximumClass)
                maximumClass = partitionClassList[i];
        }
        int classCount = maximumClass + 1;

        var classDimensions = new int[classCount];
        var classSubclasses = new int[classCount];
        var classMasterbooks = new int[classCount];
        var classSubclassBooks = new int[classCount][];
        for (int c = 0; c < classCount; c++)
        {
            classDimensions[c] = (int)reader.ReadBits(3) + 1;
            classSubclasses[c] = (int)reader.ReadBits(2);
            if (classSubclasses[c] != 0)
                classMasterbooks[c] = (int)reader.ReadBits(8);
            else
                classMasterbooks[c] = -1;
            int subCount = 1 << classSubclasses[c];
            var books = new int[subCount];
            for (int j = 0; j < subCount; j++)
                books[j] = (int)reader.ReadBits(8) - 1; // -1 => no book
            classSubclassBooks[c] = books;
        }

        int multiplier = (int)reader.ReadBits(2) + 1;
        int rangeBits = (int)reader.ReadBits(4);

        // Count X list entries: sum of class dimensions across all partitions + 2 boundary values.
        int floor1Values = 2;
        for (int i = 0; i < partitions; i++) floor1Values += classDimensions[partitionClassList[i]];
        if (floor1Values > 65)
            throw new InvalidDataException($"Floor 1 X list size {floor1Values} exceeds spec maximum 65.");

        var xList = new int[floor1Values];
        xList[0] = 0;
        xList[1] = 1 << rangeBits;
        int xIndex = 2;
        for (int i = 0; i < partitions; i++)
        {
            int cls = partitionClassList[i];
            int dims = classDimensions[cls];
            for (int j = 0; j < dims; j++)
                xList[xIndex++] = (int)reader.ReadBits(rangeBits);
        }

        return new VorbisFloor1Config
        {
            Partitions = partitions,
            PartitionClassList = partitionClassList,
            ClassDimensions = classDimensions,
            ClassSubclasses = classSubclasses,
            ClassMasterbooks = classMasterbooks,
            ClassSubclassBooks = classSubclassBooks,
            Multiplier = multiplier,
            RangeBits = rangeBits,
            XList = xList,
        };
    }
}
