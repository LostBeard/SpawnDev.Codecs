// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-flat representation of a parsed VorbisSetupHeader, designed to be
// uploaded once per stream and consumed by VorbisPacketDecodeKernels at
// per-packet time. Pairs with VorbisHuffmanCodebookSetGpu (already in the
// library) which packs the codebook tree + multiplicands.
//
// All host-side parsing and flattening happens at construction (allowed
// per CARDINAL rule's "metadata struct setup" allowance). The runtime
// per-packet decode then reads from these flat buffers + the codebook
// set with no host-side codec-data work.
//
// Everything that was a 2D / variable-length array in the CPU config
// records is collapsed into a single backing int array per category
// + a parallel offset/length table:
//
//   Floor 1 configs:
//     - FloorScalars          : int[] - per-floor 5 scalars (Partitions, Multiplier, RangeBits, XListLength, BasePartitionClassListOffset)
//     - PartitionClassList    : int[] - concat across floors
//     - ClassDimensions       : int[] - concat across floors
//     - ClassSubclasses       : int[] - concat across floors
//     - ClassMasterbooks      : int[] - concat across floors
//     - ClassSubclassBooks    : int[] - concat across (floor, class), packed [c0_sub0..c0_subN, c1_sub0..]
//     - ClassDataOffsets      : int[] - per-floor offsets into the 5 class arrays + ClassSubclassBooks
//     - XList                 : int[] - concat across floors
//
//   Residue configs:
//     - ResidueScalars        : int[] - per-residue 6 scalars (Type, Begin, End, PartitionSize, Classifications, Classbook)
//     - ResidueBooksFlat      : int[] - per-residue, vqclass * 8 + pass (length = sum_residues classifications * 8)
//     - ResidueBooksOffsets   : int[] - per-residue offset into ResidueBooksFlat
//
//   Mappings:
//     - MappingScalars        : int[] - per-mapping (Submaps + reserved fields)
//     - MappingMux            : int[] - concat per-mapping per-channel (length = mappings * channels)
//     - MappingMuxOffsets     : int[] - per-mapping offset into MappingMux
//     - MappingFloors         : int[] - per-mapping per-submap floor index
//     - MappingResidues       : int[] - per-mapping per-submap residue index
//     - MappingSubmapOffsets  : int[] - per-mapping offset into MappingFloors+MappingResidues
//
//   Modes:
//     - ModeBlockFlags        : byte[] - 1 byte per mode (BlockFlag bit packed in bit 0)
//     - ModeMappings          : int[] - mapping index per mode
//
// XList lengths per floor are stored in FloorScalars[floorIdx * 5 + 3].
// XList for floor f starts at the matching offset in XListOffsets.

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// GPU-flat representation of a parsed <see cref="VorbisSetupHeader"/>
/// for upload to the accelerator. Pairs with
/// <see cref="VorbisCodebookSetFlat"/> for the codebook content.
/// </summary>
public sealed record VorbisSetupHeaderFlat
{
    // ---- Floor 1 configs ----

    /// <summary>Per-floor 5 scalars: Partitions, Multiplier, RangeBits, XListLength, ClassCount.
    /// Length = floors.Length * 5.</summary>
    public required int[] FloorScalars { get; init; }

    /// <summary>Concat of PartitionClassList across all floors.</summary>
    public required int[] FloorPartitionClassList { get; init; }
    /// <summary>Per-floor offset into FloorPartitionClassList.</summary>
    public required int[] FloorPartitionClassListOffsets { get; init; }

    /// <summary>Concat of ClassDimensions across all floors.</summary>
    public required int[] FloorClassDimensions { get; init; }
    /// <summary>Per-floor offset into FloorClassDimensions.</summary>
    public required int[] FloorClassDimensionsOffsets { get; init; }

    /// <summary>Concat of ClassSubclasses across all floors.</summary>
    public required int[] FloorClassSubclasses { get; init; }
    /// <summary>Per-floor offset into FloorClassSubclasses.</summary>
    public required int[] FloorClassSubclassesOffsets { get; init; }

    /// <summary>Concat of ClassMasterbooks across all floors.</summary>
    public required int[] FloorClassMasterbooks { get; init; }
    /// <summary>Per-floor offset into FloorClassMasterbooks.</summary>
    public required int[] FloorClassMasterbooksOffsets { get; init; }

    /// <summary>Concat of ClassSubclassBooks across (floor, class).
    /// Per class c the count is (1 &lt;&lt; ClassSubclasses[c]).</summary>
    public required int[] FloorClassSubclassBooks { get; init; }
    /// <summary>Per-(floor, class-within-floor) offset into FloorClassSubclassBooks.
    /// Length = sum_floors classCount(floor).</summary>
    public required int[] FloorClassSubclassBooksOffsets { get; init; }

    /// <summary>Concat of XList across all floors.</summary>
    public required int[] FloorXList { get; init; }
    /// <summary>Per-floor offset into FloorXList.</summary>
    public required int[] FloorXListOffsets { get; init; }

    // ---- Residue configs ----

    /// <summary>Per-residue 6 scalars: Type, Begin, End, PartitionSize, Classifications, Classbook.</summary>
    public required int[] ResidueScalars { get; init; }

    /// <summary>Per-residue book table flat: residueBooks[(vqclass * 8) + pass] = bookIdx (-1 if absent).</summary>
    public required int[] ResidueBooks { get; init; }
    /// <summary>Per-residue offset into ResidueBooks.</summary>
    public required int[] ResidueBooksOffsets { get; init; }

    // ---- Mappings ----

    /// <summary>Per-mapping 1 scalar: Submaps. (Coupling not yet supported in v1 mono decoder.)</summary>
    public required int[] MappingScalars { get; init; }

    /// <summary>Concat of per-channel Mux across mappings.</summary>
    public required int[] MappingMux { get; init; }
    /// <summary>Per-mapping offset into MappingMux.</summary>
    public required int[] MappingMuxOffsets { get; init; }

    /// <summary>Concat of per-submap SubmapFloor across mappings.</summary>
    public required int[] MappingFloors { get; init; }
    /// <summary>Concat of per-submap SubmapResidue across mappings.</summary>
    public required int[] MappingResidues { get; init; }
    /// <summary>Per-mapping offset into MappingFloors / MappingResidues (parallel).</summary>
    public required int[] MappingSubmapOffsets { get; init; }

    // ---- Modes ----

    /// <summary>Per-mode BlockFlag byte (bit 0 = BlockFlag).</summary>
    public required byte[] ModeBlockFlags { get; init; }
    /// <summary>Per-mode mapping index.</summary>
    public required int[] ModeMappings { get; init; }
}

/// <summary>
/// Host-side helper that flattens a <see cref="VorbisSetupHeader"/> into
/// a <see cref="VorbisSetupHeaderFlat"/> for upload to GPU. Runs once per
/// stream at decoder construction time (metadata struct setup, allowed
/// under CARDINAL rule).
/// </summary>
public static class VorbisSetupHeaderGpu
{
    /// <summary>Build the flat representation.</summary>
    public static VorbisSetupHeaderFlat Build(VorbisSetupHeader setup)
    {
        ArgumentNullException.ThrowIfNull(setup);
        return new VorbisSetupHeaderFlat
        {
            // Floors
            FloorScalars = BuildFloorScalars(setup.Floors),
            FloorPartitionClassList = ConcatInt(setup.Floors, f => f.PartitionClassList),
            FloorPartitionClassListOffsets = OffsetsInt(setup.Floors, f => f.PartitionClassList.Length),
            FloorClassDimensions = ConcatInt(setup.Floors, f => f.ClassDimensions),
            FloorClassDimensionsOffsets = OffsetsInt(setup.Floors, f => f.ClassDimensions.Length),
            FloorClassSubclasses = ConcatInt(setup.Floors, f => f.ClassSubclasses),
            FloorClassSubclassesOffsets = OffsetsInt(setup.Floors, f => f.ClassSubclasses.Length),
            FloorClassMasterbooks = ConcatInt(setup.Floors, f => f.ClassMasterbooks),
            FloorClassMasterbooksOffsets = OffsetsInt(setup.Floors, f => f.ClassMasterbooks.Length),
            FloorClassSubclassBooks = FlattenSubclassBooks(setup.Floors),
            FloorClassSubclassBooksOffsets = SubclassBooksOffsets(setup.Floors),
            FloorXList = ConcatInt(setup.Floors, f => f.XList),
            FloorXListOffsets = OffsetsInt(setup.Floors, f => f.XList.Length),

            // Residues
            ResidueScalars = BuildResidueScalars(setup.Residues),
            ResidueBooks = FlattenResidueBooks(setup.Residues),
            ResidueBooksOffsets = ResidueBooksOffsets(setup.Residues),

            // Mappings
            MappingScalars = BuildMappingScalars(setup.Mappings),
            MappingMux = ConcatInt(setup.Mappings, m => m.Mux),
            MappingMuxOffsets = OffsetsInt(setup.Mappings, m => m.Mux.Length),
            MappingFloors = ConcatInt(setup.Mappings, m => m.SubmapFloor),
            MappingResidues = ConcatInt(setup.Mappings, m => m.SubmapResidue),
            MappingSubmapOffsets = OffsetsInt(setup.Mappings, m => m.SubmapFloor.Length),

            // Modes
            ModeBlockFlags = BuildModeBlockFlags(setup.Modes),
            ModeMappings = setup.Modes.Select(m => m.Mapping).ToArray(),
        };
    }

    private static int[] BuildFloorScalars(VorbisFloor1Config[] floors)
    {
        var arr = new int[floors.Length * 5];
        for (int i = 0; i < floors.Length; i++)
        {
            var f = floors[i];
            arr[i * 5 + 0] = f.Partitions;
            arr[i * 5 + 1] = f.Multiplier;
            arr[i * 5 + 2] = f.RangeBits;
            arr[i * 5 + 3] = f.XList.Length;
            arr[i * 5 + 4] = f.ClassDimensions.Length; // class count
        }
        return arr;
    }

    private static int[] BuildResidueScalars(VorbisResidueConfig[] residues)
    {
        var arr = new int[residues.Length * 6];
        for (int i = 0; i < residues.Length; i++)
        {
            var r = residues[i];
            arr[i * 6 + 0] = (int)r.Type;
            arr[i * 6 + 1] = r.Begin;
            arr[i * 6 + 2] = r.End;
            arr[i * 6 + 3] = r.PartitionSize;
            arr[i * 6 + 4] = r.Classifications;
            arr[i * 6 + 5] = r.Classbook;
        }
        return arr;
    }

    private static int[] BuildMappingScalars(VorbisMappingConfig[] mappings)
    {
        var arr = new int[mappings.Length];
        for (int i = 0; i < mappings.Length; i++)
        {
            arr[i] = mappings[i].Submaps;
        }
        return arr;
    }

    private static byte[] BuildModeBlockFlags(VorbisModeConfig[] modes)
    {
        var arr = new byte[modes.Length];
        for (int i = 0; i < modes.Length; i++)
        {
            arr[i] = modes[i].BlockFlag ? (byte)1 : (byte)0;
        }
        return arr;
    }

    private static int[] ConcatInt<T>(T[] src, Func<T, int[]> getter)
    {
        int total = 0;
        for (int i = 0; i < src.Length; i++) total += getter(src[i]).Length;
        var arr = new int[Math.Max(1, total)];
        int o = 0;
        for (int i = 0; i < src.Length; i++)
        {
            var s = getter(src[i]);
            Array.Copy(s, 0, arr, o, s.Length);
            o += s.Length;
        }
        return arr;
    }

    private static int[] OffsetsInt<T>(T[] src, Func<T, int> getLen)
    {
        var arr = new int[src.Length + 1];
        for (int i = 0; i < src.Length; i++) arr[i + 1] = arr[i] + getLen(src[i]);
        return arr;
    }

    private static int[] FlattenSubclassBooks(VorbisFloor1Config[] floors)
    {
        int total = 0;
        for (int i = 0; i < floors.Length; i++)
        {
            for (int c = 0; c < floors[i].ClassSubclassBooks.Length; c++)
                total += floors[i].ClassSubclassBooks[c].Length;
        }
        var arr = new int[Math.Max(1, total)];
        int o = 0;
        for (int i = 0; i < floors.Length; i++)
        {
            for (int c = 0; c < floors[i].ClassSubclassBooks.Length; c++)
            {
                var s = floors[i].ClassSubclassBooks[c];
                Array.Copy(s, 0, arr, o, s.Length);
                o += s.Length;
            }
        }
        return arr;
    }

    private static int[] SubclassBooksOffsets(VorbisFloor1Config[] floors)
    {
        // Offset per (floor, class). Length = sum_floors classCount(floor) + 1
        int totalClasses = 0;
        for (int i = 0; i < floors.Length; i++) totalClasses += floors[i].ClassDimensions.Length;
        var arr = new int[totalClasses + 1];
        int classIdx = 0;
        for (int i = 0; i < floors.Length; i++)
        {
            for (int c = 0; c < floors[i].ClassSubclassBooks.Length; c++)
            {
                arr[classIdx + 1] = arr[classIdx] + floors[i].ClassSubclassBooks[c].Length;
                classIdx++;
            }
        }
        return arr;
    }

    private static int[] FlattenResidueBooks(VorbisResidueConfig[] residues)
    {
        // Residue books: per-residue, classifications * 8 ints (vqclass, pass).
        int total = 0;
        for (int i = 0; i < residues.Length; i++) total += residues[i].Classifications * 8;
        var arr = new int[Math.Max(1, total)];
        int o = 0;
        for (int i = 0; i < residues.Length; i++)
        {
            var r = residues[i];
            for (int v = 0; v < r.Classifications; v++)
            {
                for (int p = 0; p < 8; p++)
                {
                    arr[o + v * 8 + p] = r.Books[v][p];
                }
            }
            o += r.Classifications * 8;
        }
        return arr;
    }

    private static int[] ResidueBooksOffsets(VorbisResidueConfig[] residues)
    {
        var arr = new int[residues.Length + 1];
        for (int i = 0; i < residues.Length; i++)
            arr[i + 1] = arr[i] + residues[i].Classifications * 8;
        return arr;
    }
}
