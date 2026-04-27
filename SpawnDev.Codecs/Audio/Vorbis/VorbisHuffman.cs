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

    /// <summary>
    /// EOP-aware variant of <see cref="Decode"/>. Returns -1 when the packet
    /// runs out of bits mid-codeword (Vorbis I sec 8.6.5: residue decode
    /// terminates gracefully on end-of-packet). Mirrors libvorbis
    /// <c>vorbis_book_decode</c> convention of returning -1 on EOP.
    /// </summary>
    internal int TryDecode(ref VorbisBitReader reader)
    {
        int node = 0;
        for (int depth = 0; depth <= _maxDepth; depth++)
        {
            if (!reader.TryReadBit(out uint bitVal)) return -1;
            int bit = (int)bitVal;
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
    /// Assign canonical Huffman codewords to every entry per Vorbis I
    /// section 3.2.1 / libvorbis lib/sharedbook.c <c>_make_words</c>. Vorbis's
    /// canonical scheme processes entries in entry-index order (NOT sorted by
    /// length) and uses per-length "marker" counters that get pruned and
    /// rebased as longer codes branch from already-claimed shorter prefixes.
    /// A textbook count-based canonical Huffman assignment only matches when
    /// lengths happen to be sorted ascending - which Vorbis codebooks are
    /// usually not, so we have to mirror the marker algorithm exactly.
    /// Throws <see cref="InvalidDataException"/> if the set of lengths is not
    /// a valid prefix code (over-specified tree).
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

        // Single-entry-used short-circuit: that entry always gets code 0.
        int usedCount = 0;
        for (int i = 0; i < entries; i++) if (lengths[i] > 0) usedCount++;
        if (usedCount == 1)
        {
            return new VorbisHuffmanTable
            {
                Codewords = new uint[entries],
                EntryLengths = lengths,
                MaxLength = maxLength,
            };
        }

        // libvorbis _make_words marker algorithm. marker[L] tracks the next
        // available codeword at length L. When we claim a leaf at length L,
        // we must prune any longer branches that would have dangled from the
        // claimed node, and rebase them so they branch from the new "current"
        // node at length L.
        var marker = new uint[maxLength + 1];
        var codes = new uint[entries];

        for (int i = 0; i < entries; i++)
        {
            int length = lengths[i];
            if (length <= 0) { codes[i] = 0; continue; }

            uint entryCode = marker[length];
            if (length < 32 && (entryCode >> length) != 0)
                throw new InvalidDataException(
                    $"Codebook over-specified: marker[{length}]=0x{entryCode:X} doesn't fit in {length} bits.");
            codes[i] = entryCode;

            // Walk the marker UP from this length to length 1, claiming the
            // path. If a marker at level j is already odd (a 1-bit at the
            // bottom), we have to "jump branches" - rebase from the parent's
            // marker shifted up.
            for (int j = length; j > 0; j--)
            {
                if ((marker[j] & 1) != 0)
                {
                    if (j == 1) marker[1]++;
                    else marker[j] = marker[j - 1] << 1;
                    break;
                }
                marker[j]++;
            }

            // Walk DOWN from length+1 to maxLength, pruning any longer
            // markers that pointed to the node we just claimed. They have to
            // dangle from the new node instead.
            for (int j = length + 1; j <= maxLength; j++)
            {
                if ((marker[j] >> 1) == entryCode)
                {
                    entryCode = marker[j];
                    marker[j] = marker[j - 1] << 1;
                }
                else break;
            }
        }

        // NOTE: libvorbis bit-reverses codes here so its LSB-first oggpack
        // writer emits the canonical-MSB bit first. We don't need to do that
        // ourselves because our Huffman tree is built MSB-first from the
        // canonical codes, and our LSB-first reader will pull the bits in the
        // same canonical-MSB order the encoder emitted.

        return new VorbisHuffmanTable
        {
            Codewords = codes,
            EntryLengths = lengths,
            MaxLength = maxLength,
        };
    }
}
