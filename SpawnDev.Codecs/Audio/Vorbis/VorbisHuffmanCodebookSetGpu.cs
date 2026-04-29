// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Host-side helper that flattens an entire set of Vorbis codebooks
// (typically the full set parsed from a VorbisSetupHeader) into a
// single block of flat ArrayView-friendly buffers for upload to GPU.
//
// Each codebook contributes:
//   - children     : flat tree (2 ints per node, packed)
//   - leafToEntry  : codebook entry per leaf node
//   - maxDepth     : per-codebook scalar
//
// Output layout (flat across all codebooks):
//   - allChildren[]      : concat of children arrays
//   - allLeafToEntry[]   : concat of leafToEntry arrays
//   - childrenOffsets[]  : per-codebook offset into allChildren (length = N+1; last = total)
//   - leafOffsets[]      : per-codebook offset into allLeafToEntry
//   - maxDepths[]        : per-codebook max depth
//   - allMultiplicands[] : concat of codebook multiplicand arrays
//   - multOffsets[]      : per-codebook offset into allMultiplicands
//   - multLengths[]      : per-codebook multiplicand count
//
// This is metadata struct setup (per CARDINAL rule) - it runs once per
// stream on host, the GPU side never reconstructs it.

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// Flattened representation of a set of Vorbis codebooks for GPU upload.
/// </summary>
public sealed record VorbisCodebookSetFlat
{
    /// <summary>Concatenated children arrays from every codebook's Huffman tree.</summary>
    public required int[] AllChildren { get; init; }
    /// <summary>Concatenated leafToEntry arrays.</summary>
    public required int[] AllLeafToEntry { get; init; }
    /// <summary>Per-codebook offset into AllChildren (length = codebookCount + 1).</summary>
    public required int[] ChildrenOffsets { get; init; }
    /// <summary>Per-codebook offset into AllLeafToEntry (length = codebookCount + 1).</summary>
    public required int[] LeafOffsets { get; init; }
    /// <summary>Per-codebook Huffman max depth.</summary>
    public required int[] MaxDepths { get; init; }
    /// <summary>Concatenated multiplicand arrays across all codebooks.</summary>
    public required int[] AllMultiplicands { get; init; }
    /// <summary>Per-codebook offset into AllMultiplicands (length = codebookCount + 1).</summary>
    public required int[] MultOffsets { get; init; }
    /// <summary>Per-codebook multiplicand count (== AllMultiplicands slice length).</summary>
    public required int[] MultLengths { get; init; }
    /// <summary>Per-codebook scalar config (Dimensions, Entries, LookupType, Quantvals).</summary>
    public required int[] CodebookDimensions { get; init; }
    /// <summary>Per-codebook entries count.</summary>
    public required int[] CodebookEntries { get; init; }
    /// <summary>Per-codebook lookup type (0, 1, or 2).</summary>
    public required int[] CodebookLookupTypes { get; init; }
    /// <summary>Per-codebook quantvals (lookup1_values for type 1, 0 otherwise).</summary>
    public required int[] CodebookQuantvals { get; init; }
    /// <summary>Per-codebook MinValue (parallel array to MultOffsets).</summary>
    public required double[] CodebookMinValues { get; init; }
    /// <summary>Per-codebook DeltaValue.</summary>
    public required double[] CodebookDeltaValues { get; init; }
    /// <summary>Per-codebook SequenceP flag (0 or 1).</summary>
    public required int[] CodebookSequenceP { get; init; }
}

/// <summary>
/// Host-side helper that flattens a set of <see cref="VorbisCodebook"/>
/// records to GPU-friendly flat buffers.
/// </summary>
public static class VorbisHuffmanCodebookSetGpu
{
    /// <summary>
    /// Flatten <paramref name="codebooks"/> into a single
    /// <see cref="VorbisCodebookSetFlat"/> for upload to GPU.
    /// </summary>
    internal static VorbisCodebookSetFlat Build(VorbisCodebook[] codebooks)
    {
        ArgumentNullException.ThrowIfNull(codebooks);
        int n = codebooks.Length;

        var childrenOffsets = new int[n + 1];
        var leafOffsets = new int[n + 1];
        var maxDepths = new int[n];
        var multOffsets = new int[n + 1];
        var multLengths = new int[n];
        var dims = new int[n];
        var entries = new int[n];
        var lookupTypes = new int[n];
        var quantvals = new int[n];
        var minValues = new double[n];
        var deltaValues = new double[n];
        var sequenceP = new int[n];

        // Build per-codebook Huffman decoders + flatten.
        var perBookChildren = new int[n][];
        var perBookLeafToEntry = new int[n][];
        for (int i = 0; i < n; i++)
        {
            var cb = codebooks[i];
            var table = VorbisHuffman.Build(cb.Lengths);
            var decoder = new VorbisHuffmanDecoder(table);
            var (children, leafToEntry, maxDepth) = decoder.BuildFlatGpu();
            perBookChildren[i] = children;
            perBookLeafToEntry[i] = leafToEntry;
            maxDepths[i] = maxDepth;

            childrenOffsets[i + 1] = childrenOffsets[i] + children.Length;
            leafOffsets[i + 1] = leafOffsets[i] + leafToEntry.Length;
            multOffsets[i + 1] = multOffsets[i] + cb.Multiplicands.Length;
            multLengths[i] = cb.Multiplicands.Length;

            dims[i] = cb.Dimensions;
            entries[i] = cb.Entries;
            lookupTypes[i] = cb.LookupType;
            quantvals[i] = cb.LookupType == 1 ? Lookup1Values(cb.Entries, cb.Dimensions) : 0;
            minValues[i] = cb.MinValue;
            deltaValues[i] = cb.DeltaValue;
            sequenceP[i] = cb.SequenceP ? 1 : 0;
        }

        var allChildren = new int[childrenOffsets[n]];
        var allLeafToEntry = new int[leafOffsets[n]];
        var allMultiplicands = new int[multOffsets[n]];
        for (int i = 0; i < n; i++)
        {
            Buffer.BlockCopy(perBookChildren[i], 0, allChildren, childrenOffsets[i] * sizeof(int), perBookChildren[i].Length * sizeof(int));
            Buffer.BlockCopy(perBookLeafToEntry[i], 0, allLeafToEntry, leafOffsets[i] * sizeof(int), perBookLeafToEntry[i].Length * sizeof(int));
            if (codebooks[i].Multiplicands.Length > 0)
                Buffer.BlockCopy(codebooks[i].Multiplicands, 0, allMultiplicands, multOffsets[i] * sizeof(int), codebooks[i].Multiplicands.Length * sizeof(int));
        }

        return new VorbisCodebookSetFlat
        {
            AllChildren = allChildren,
            AllLeafToEntry = allLeafToEntry,
            ChildrenOffsets = childrenOffsets,
            LeafOffsets = leafOffsets,
            MaxDepths = maxDepths,
            AllMultiplicands = allMultiplicands,
            MultOffsets = multOffsets,
            MultLengths = multLengths,
            CodebookDimensions = dims,
            CodebookEntries = entries,
            CodebookLookupTypes = lookupTypes,
            CodebookQuantvals = quantvals,
            CodebookMinValues = minValues,
            CodebookDeltaValues = deltaValues,
            CodebookSequenceP = sequenceP,
        };
    }

    /// <summary>
    /// libvorbis _book_maptype1_quantvals: smallest q such that q^dimensions >= entries.
    /// </summary>
    private static int Lookup1Values(int entries, int dimensions)
    {
        int q = 1;
        while (true)
        {
            long pow = 1;
            for (int i = 0; i < dimensions; i++) pow *= q;
            if (pow >= entries) return q;
            q++;
        }
    }
}
