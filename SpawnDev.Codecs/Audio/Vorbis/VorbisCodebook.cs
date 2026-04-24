// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Parse a Vorbis codebook per Vorbis I Section 3.2. The codebook holds the
// Huffman lengths for each entry (for the bit decoder), plus an optional
// lookup table (types 1 and 2) that maps each entry to a vector of
// floating-point multiplicands used in floor / residue decode.
//
// This parser produces the RAW structural fields. Building the Huffman
// decoding table from the entry lengths is the next slice.

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// Parsed Vorbis codebook. See Vorbis I Section 3.2 for the on-wire fields.
/// </summary>
public sealed record VorbisCodebook
{
    /// <summary>Dimensions of each codeword entry (each entry encodes a vector of this length).</summary>
    public int Dimensions { get; init; }

    /// <summary>Number of entries in the codebook.</summary>
    public int Entries { get; init; }

    /// <summary>True if the codebook was ordered-coded (lengths are monotonically non-decreasing).</summary>
    public bool Ordered { get; init; }

    /// <summary>True if the codebook was sparsely coded (some entries are unused with length 0).</summary>
    public bool Sparse { get; init; }

    /// <summary>
    /// Per-entry codeword bit lengths. Length <c>Entries</c>. Unused entries
    /// (sparse codebooks) have <c>Lengths[i] == 0</c>.
    /// </summary>
    public required int[] Lengths { get; init; }

    /// <summary>Lookup type: 0 (no lookup), 1 (implicitly-populated), or 2 (explicitly-populated).</summary>
    public int LookupType { get; init; }

    /// <summary>Minimum value for the multiplicand table (only when LookupType != 0).</summary>
    public double MinValue { get; init; }

    /// <summary>Per-step delta for the multiplicand table.</summary>
    public double DeltaValue { get; init; }

    /// <summary>Bits per multiplicand entry.</summary>
    public int ValueBits { get; init; }

    /// <summary>Sequential reconstruction flag.</summary>
    public bool SequenceP { get; init; }

    /// <summary>
    /// Raw multiplicand array length: <c>lookup1_values(Entries, Dimensions)</c> for
    /// type 1, or <c>Entries * Dimensions</c> for type 2. Empty for type 0.
    /// </summary>
    public required int[] Multiplicands { get; init; }
}

/// <summary>Parses Vorbis codebooks from the setup header bitstream.</summary>
internal static class VorbisCodebookParser
{
    /// <summary>
    /// Parse one codebook at the current reader position. The 24-bit sync word
    /// <c>0x564342</c> (the ASCII sequence "BCV" in reverse) must be the next bits.
    /// </summary>
    internal static VorbisCodebook Parse(ref VorbisBitReader reader)
    {
        uint sync = reader.ReadBits(24);
        if (sync != 0x564342)
            throw new InvalidDataException($"Vorbis codebook sync mismatch: expected 0x564342, got 0x{sync:X6}.");

        int dimensions = (int)reader.ReadBits(16);
        int entries = (int)reader.ReadBits(24);
        if (dimensions == 0 || entries == 0)
            throw new InvalidDataException(
                $"Vorbis codebook invalid geometry: dimensions={dimensions}, entries={entries}.");

        bool ordered = reader.ReadBit() != 0;
        bool sparse = false;
        var lengths = new int[entries];

        if (ordered)
        {
            int currentLength = (int)reader.ReadBits(5) + 1;
            int entryIndex = 0;
            while (entryIndex < entries)
            {
                int countBits = VorbisMath.Ilog(entries - entryIndex);
                int number = (int)reader.ReadBits(countBits);
                if (number + entryIndex > entries)
                    throw new InvalidDataException("Ordered codebook run past end of entries.");
                for (int i = 0; i < number; i++) lengths[entryIndex + i] = currentLength;
                entryIndex += number;
                currentLength++;
                if (currentLength > 33)
                    throw new InvalidDataException("Ordered codebook length exceeded 32 bits.");
            }
        }
        else
        {
            sparse = reader.ReadBit() != 0;
            for (int i = 0; i < entries; i++)
            {
                if (sparse)
                {
                    bool used = reader.ReadBit() != 0;
                    if (used) lengths[i] = (int)reader.ReadBits(5) + 1;
                    else lengths[i] = 0; // unused
                }
                else
                {
                    lengths[i] = (int)reader.ReadBits(5) + 1;
                }
            }
        }

        int lookupType = (int)reader.ReadBits(4);
        if (lookupType > 2)
            throw new InvalidDataException($"Vorbis codebook reserved lookup type {lookupType}.");

        double minValue = 0, deltaValue = 0;
        int valueBits = 0;
        bool sequenceP = false;
        int[] multiplicands = Array.Empty<int>();

        if (lookupType != 0)
        {
            minValue = VorbisMath.Float32Unpack(reader.ReadBits(32));
            deltaValue = VorbisMath.Float32Unpack(reader.ReadBits(32));
            valueBits = (int)reader.ReadBits(4) + 1;
            sequenceP = reader.ReadBit() != 0;
            int count = lookupType == 1
                ? VorbisMath.Lookup1Values(entries, dimensions)
                : entries * dimensions;
            multiplicands = new int[count];
            for (int i = 0; i < count; i++) multiplicands[i] = (int)reader.ReadBits(valueBits);
        }

        return new VorbisCodebook
        {
            Dimensions = dimensions,
            Entries = entries,
            Ordered = ordered,
            Sparse = sparse,
            Lengths = lengths,
            LookupType = lookupType,
            MinValue = minValue,
            DeltaValue = deltaValue,
            ValueBits = valueBits,
            SequenceP = sequenceP,
            Multiplicands = multiplicands,
        };
    }
}
