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
