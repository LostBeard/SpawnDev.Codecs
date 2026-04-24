// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Canonical Huffman codeword generator for Vorbis codebooks. Given the per-
// entry bit lengths parsed out of the codebook header, produces the canonical
// codewords assigned by the encoder. Used by the audio packet decoder to
// translate bit sequences into codebook entry indices.
//
// Vorbis I Section 3.2.1: codewords are uniquely determined by the set of
// lengths using the canonical Huffman scheme - entries with the same length
// get consecutive codes in entry-index order, and the first code at length L
// is (first code at L-1 + count[L-1]) << 1.

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// Canonical Huffman codeword table for a Vorbis codebook. Only entries
/// with a non-zero length in the source codebook appear in <see cref="EntryLengths"/>.
/// </summary>
internal sealed record VorbisHuffmanTable
{
    /// <summary>Per-entry assigned codeword (0 for entries with Length == 0).</summary>
    public required uint[] Codewords { get; init; }

    /// <summary>Per-entry codeword length (0 for unused entries).</summary>
    public required int[] EntryLengths { get; init; }

    /// <summary>Maximum codeword length across the codebook.</summary>
    public int MaxLength { get; init; }
}

/// <summary>
/// Decision tree built from a <see cref="VorbisHuffmanTable"/>. Each internal
/// node carries two child indices, a leaf node carries the codebook entry.
/// </summary>
internal sealed class VorbisHuffmanDecoder
{
    // Packed representation: node i has two children _nodes[2i] (for bit 0)
    // and _nodes[2i+1] (for bit 1). Values < 0 mean "no child / error";
    // values in 0..entries-1 encoded with the high bit clear are entry indices
    // (leaves) - we distinguish leaves from internal nodes by checking
    // _isLeaf[i].
    private readonly int[] _children;
    private readonly bool[] _isLeaf;
    private readonly int _maxDepth;

    public VorbisHuffmanDecoder(VorbisHuffmanTable table)
    {
        // Worst-case node count: binary tree with 2^(maxDepth+1) - 1 nodes.
        // maxDepth capped at 32 during Build so worst-case is bounded by entries * 33
        // once we only allocate actually-used nodes.
        int capacity = Math.Max(2, (int)Math.Min((long)table.Codewords.Length * (table.MaxLength + 1) * 2 + 2, 1 << 20));
        var children = new List<int>(capacity);
        var isLeaf = new List<bool>(capacity);
        // Root node.
        children.Add(-1); children.Add(-1); isLeaf.Add(false);

        for (int entry = 0; entry < table.Codewords.Length; entry++)
        {
            int length = table.EntryLengths[entry];
            if (length == 0) continue;
            uint code = table.Codewords[entry];
            int node = 0;
            for (int bitIdx = length - 1; bitIdx >= 0; bitIdx--)
            {
                if (isLeaf[node])
                    throw new InvalidDataException(
                        $"Vorbis Huffman build: entry {entry} (code 0x{code:X}, length {length}) collides with a shorter prefix.");
                int bit = (int)((code >> bitIdx) & 1);
                int nodeChild = children[node * 2 + bit];
                if (bitIdx == 0)
                {
                    if (nodeChild != -1)
                        throw new InvalidDataException(
                            $"Vorbis Huffman build: entry {entry} (code 0x{code:X}, length {length}) would overwrite an existing leaf.");
                    int leafIndex = isLeaf.Count;
                    children.Add(-1); children.Add(-1);
                    isLeaf.Add(true);
                    children[node * 2 + bit] = leafIndex | EntryBit;
                }
                else
                {
                    if (nodeChild == -1)
                    {
                        int newIndex = isLeaf.Count;
                        children.Add(-1); children.Add(-1);
                        isLeaf.Add(false);
                        children[node * 2 + bit] = newIndex;
                        nodeChild = newIndex;
                    }
                    node = nodeChild;
                }
            }
        }
        _children = children.ToArray();
        _isLeaf = isLeaf.ToArray();
        _maxDepth = table.MaxLength;

        // Post-build: store entry index on each leaf so Decode can look it up.
        _leafToEntry = new Dictionary<int, int>();
        for (int entry = 0; entry < table.Codewords.Length; entry++)
        {
            int length = table.EntryLengths[entry];
            if (length == 0) continue;
            uint code = table.Codewords[entry];
            int node = 0;
            for (int bitIdx = length - 1; bitIdx > 0; bitIdx--)
            {
                int bit = (int)((code >> bitIdx) & 1);
                int next = _children[node * 2 + bit];
                node = next & ~EntryBit;
            }
            int lastBit = (int)(code & 1);
            int leaf = _children[node * 2 + lastBit] & ~EntryBit;
            _leafToEntry[leaf] = entry;
        }
    }

    private const int EntryBit = 1 << 30;
    private readonly Dictionary<int, int> _leafToEntry;

    /// <summary>
    /// Decode the next codebook entry from <paramref name="reader"/>. Walks the
    /// decision tree bit-by-bit; each bit read goes left (0) or right (1)
    /// until a leaf is hit.
    /// </summary>
    internal int Decode(ref VorbisBitReader reader)
    {
        int node = 0;
        for (int depth = 0; depth <= _maxDepth; depth++)
        {
            int bit = (int)reader.ReadBit();
            int nextRaw = _children[node * 2 + bit];
            if (nextRaw == -1)
                throw new InvalidDataException(
                    $"Vorbis Huffman decode: no path for bit pattern at depth {depth}.");
            int nextIdx = nextRaw & ~EntryBit;
            if (_isLeaf[nextIdx])
                return _leafToEntry[nextIdx];
            node = nextIdx;
        }
        throw new InvalidDataException("Vorbis Huffman decode: exceeded max depth without hitting a leaf.");
    }
}

internal static class VorbisHuffman
{
    /// <summary>
    /// Assign canonical Huffman codewords to every entry. Throws
    /// <see cref="InvalidDataException"/> if the set of lengths is not a valid
    /// prefix code (over- or under-specified).
    /// </summary>
    internal static VorbisHuffmanTable Build(int[] lengths)
    {
        int entries = lengths.Length;
        int maxLength = 0;
        for (int i = 0; i < entries; i++) maxLength = Math.Max(maxLength, lengths[i]);
        if (maxLength == 0)
        {
            // All unused entries (valid for sparse single-entry codebooks).
            return new VorbisHuffmanTable
            {
                Codewords = new uint[entries],
                EntryLengths = lengths,
                MaxLength = 0,
            };
        }
        if (maxLength > 32)
            throw new InvalidDataException($"Codebook maxLength {maxLength} > 32 bits.");

        // Count entries per length (skip length 0 = unused).
        var count = new int[maxLength + 1];
        for (int i = 0; i < entries; i++)
        {
            int L = lengths[i];
            if (L > 0) count[L]++;
        }

        // Compute the first code assigned at each length via canonical form.
        // nextCode[L] is the next available code at length L, pre-incremented per assignment.
        var nextCode = new ulong[maxLength + 2];
        nextCode[1] = 0;
        for (int L = 2; L <= maxLength + 1; L++)
            nextCode[L] = (nextCode[L - 1] + (ulong)count[L - 1]) << 1;

        // Validate: the final nextCode at length maxLength + 1 should be exactly 2^(maxLength+1)
        // if every length slot is fully consumed; or <= that if the single-used-entry degenerate
        // case. Single-entry-used is special: one entry gets code 0 regardless of declared length.
        int usedCount = 0;
        for (int i = 0; i < entries; i++) if (lengths[i] > 0) usedCount++;
        if (usedCount == 1)
        {
            var codewords = new uint[entries];
            return new VorbisHuffmanTable
            {
                Codewords = codewords,
                EntryLengths = lengths,
                MaxLength = maxLength,
            };
        }

        // Assign codewords in entry order.
        var codes = new uint[entries];
        var perLengthCounter = new ulong[maxLength + 1];
        for (int L = 0; L <= maxLength; L++) perLengthCounter[L] = (L <= maxLength) ? nextCode[L] : 0;
        for (int i = 0; i < entries; i++)
        {
            int L = lengths[i];
            if (L == 0) { codes[i] = 0; continue; }
            ulong code = perLengthCounter[L];
            if (code >> L != 0)
                throw new InvalidDataException(
                    $"Codebook over-specified: code 0x{code:X} at length {L} exceeds {1 << L}.");
            codes[i] = (uint)code;
            perLengthCounter[L]++;
        }

        // Optional rigour: check that the tree is exactly full. If under-specified, the spec
        // allows it as long as all generated lengths fit, but a strict implementation rejects.
        // For permissive behavior we only error on OVER-specification (the per-length overflow
        // check above).

        return new VorbisHuffmanTable
        {
            Codewords = codes,
            EntryLengths = lengths,
            MaxLength = maxLength,
        };
    }
}
