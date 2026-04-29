// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable Vorbis Huffman decoder. Mirror of
// VorbisHuffmanDecoder.Decode for in-kernel use.
//
// Tree representation (flat array, packed):
//   - children: int[2 * nodeCount], children[2*n + bit] = next node index
//                or (entry | EntryBit) for a leaf, or -1 = error.
//   - leafToEntry: int[leafCapacity], maps leaf node index -> entry value.
//   - maxDepth scalar.
//
// Caller flattens the tree once via VorbisHuffmanDecoder.BuildFlatGpu()
// (host-side helper) and uploads the resulting flat buffers; the
// per-call decode just walks the tree.
//
// Bit reading uses a Vorbis LSB-first state struct (mirror of
// VorbisBitReader) - see VorbisBitReaderGpu.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// GPU-callable Vorbis Huffman decoder. Walks a flat-tree
/// representation of a Vorbis codebook decoder produced by
/// <see cref="VorbisHuffmanDecoder.BuildFlatGpu"/>.
/// </summary>
public static class VorbisHuffmanDecoderGpu
{
    /// <summary>libaom-style EntryBit marker (top bit of int).</summary>
    public const int EntryBit = 1 << 30;

    /// <summary>
    /// Decode the next codebook entry by walking the flat tree, reading
    /// one bit per descent until a leaf is hit. Returns the entry index,
    /// or -1 if the bit reader ran out of bits mid-codeword.
    /// </summary>
    /// <param name="state">Vorbis LSB-first bit reader state (mutated).</param>
    /// <param name="data">Packet bytes.</param>
    /// <param name="children">Flat tree children array.</param>
    /// <param name="childrenBase">Base offset into <paramref name="children"/>.</param>
    /// <param name="leafToEntry">Flat leaf-index -> entry-index lookup.</param>
    /// <param name="leafToEntryBase">Base offset into <paramref name="leafToEntry"/>.</param>
    /// <param name="maxDepth">Maximum codeword length in this codebook.</param>
    public static int TryDecode(
        ref VorbisBitReaderGpuState state,
        ArrayView<byte> data,
        ArrayView<int> children, long childrenBase,
        ArrayView<int> leafToEntry, long leafToEntryBase,
        int maxDepth)
    {
        int node = 0;
        for (int depth = 0; depth <= maxDepth; depth++)
        {
            if (VorbisBitReaderGpu.IsEnd(in state)) return -1;
            int bit = (int)VorbisBitReaderGpu.ReadBits(ref state, data, 1);
            int nextRaw = children[childrenBase + node * 2 + bit];
            if (nextRaw == -1) return -2; // tree-level error sentinel
            int nextIdx = nextRaw & ~EntryBit;
            if ((nextRaw & EntryBit) != 0)
            {
                return leafToEntry[leafToEntryBase + nextIdx];
            }
            node = nextIdx;
        }
        return -2; // exceeded max depth
    }
}
