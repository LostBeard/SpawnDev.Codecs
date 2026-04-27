// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Encoder-side codebook helpers. The Vorbis I bitstream packs codebooks with
// the canonical-Huffman lengths array; the decoder rebuilds the codewords.
// This file gives the encoder a paired writer for emitting codebook headers
// AND mirrors VorbisHuffman.Build so the encoder can compute the canonical
// codeword for any given entry index.
//
// Vorbis I Section 3.2 / libvorbis lib/codebook.c vorbis_staticbook_pack.

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// Encoder-side helpers for emitting Vorbis codebook headers and mapping
/// entry indices to their canonical codewords. Pairs with the
/// <see cref="VorbisCodebookParser"/> on the decoder side.
/// </summary>
internal static class VorbisCodebookEncoder
{
    /// <summary>
    /// Pack a codebook into <paramref name="writer"/> matching what
    /// <see cref="VorbisCodebookParser.Parse"/> reads. Writes the 24-bit sync
    /// word plus dimensions, entries, length list, and (optionally) the
    /// lookup table.
    /// </summary>
    internal static void Pack(VorbisBitWriter writer, VorbisCodebook book)
    {
        // Sync pattern: 0x564342 (the bytes "BCV" reversed by Vorbis spec).
        writer.WriteBits(0x564342u, 24);
        writer.WriteBits((uint)book.Dimensions, 16);
        writer.WriteBits((uint)book.Entries, 24);

        // Always emit unordered (simpler) and only sparse if any zero-length entries.
        bool anyZero = false;
        for (int i = 0; i < book.Entries; i++)
            if (book.Lengths[i] == 0) { anyZero = true; break; }

        writer.WriteBit(0u); // ordered = 0
        writer.WriteBit(anyZero ? 1u : 0u);

        for (int i = 0; i < book.Entries; i++)
        {
            int len = book.Lengths[i];
            if (anyZero)
            {
                if (len == 0) writer.WriteBit(0u);
                else
                {
                    writer.WriteBit(1u);
                    if (len < 1 || len > 32)
                        throw new InvalidOperationException($"Codebook entry {i} length {len} out of range [1, 32].");
                    writer.WriteBits((uint)(len - 1), 5);
                }
            }
            else
            {
                if (len < 1 || len > 32)
                    throw new InvalidOperationException($"Codebook entry {i} length {len} out of range [1, 32].");
                writer.WriteBits((uint)(len - 1), 5);
            }
        }

        writer.WriteBits((uint)book.LookupType, 4);
        if (book.LookupType != 0)
        {
            writer.WriteBits(EncodeFloat32(book.MinValue), 32);
            writer.WriteBits(EncodeFloat32(book.DeltaValue), 32);
            int valueBitsField = book.ValueBits - 1;
            if (valueBitsField < 0 || valueBitsField > 15)
                throw new InvalidOperationException($"Codebook ValueBits-1 out of [0,15]: {valueBitsField}.");
            writer.WriteBits((uint)valueBitsField, 4);
            writer.WriteBit(book.SequenceP ? 1u : 0u);
            int count = book.LookupType == 1
                ? VorbisMath.Lookup1Values(book.Entries, book.Dimensions)
                : book.Entries * book.Dimensions;
            if (book.Multiplicands.Length != count)
                throw new InvalidOperationException(
                    $"Codebook multiplicand length {book.Multiplicands.Length} != expected {count}.");
            for (int i = 0; i < count; i++)
            {
                int v = book.Multiplicands[i];
                if (v < 0 || v >= (1 << book.ValueBits))
                    throw new InvalidOperationException(
                        $"Multiplicand[{i}]={v} out of range for ValueBits={book.ValueBits}.");
                writer.WriteBits((uint)v, book.ValueBits);
            }
        }
    }

    /// <summary>
    /// Compute the canonical Huffman codeword for every entry of the
    /// codebook. Mirrors the marker algorithm in
    /// <see cref="VorbisHuffman.Build"/> exactly so encoder and decoder see
    /// the same code-to-entry mapping.
    /// </summary>
    internal static (uint code, int length)[] BuildCodewords(int[] lengths)
    {
        int entries = lengths.Length;
        var result = new (uint, int)[entries];
        if (entries == 0) return result;

        int maxLength = 0;
        for (int i = 0; i < entries; i++) maxLength = Math.Max(maxLength, lengths[i]);
        if (maxLength == 0)
        {
            for (int i = 0; i < entries; i++) result[i] = (0u, 0);
            return result;
        }

        int usedCount = 0;
        for (int i = 0; i < entries; i++) if (lengths[i] > 0) usedCount++;
        if (usedCount == 1)
        {
            for (int i = 0; i < entries; i++) result[i] = (0u, lengths[i]);
            return result;
        }

        var marker = new uint[maxLength + 1];
        for (int i = 0; i < entries; i++)
        {
            int length = lengths[i];
            if (length <= 0) { result[i] = (0u, 0); continue; }
            uint entryCode = marker[length];
            if (length < 32 && (entryCode >> length) != 0)
                throw new InvalidOperationException(
                    $"Codebook over-specified at entry {i}: marker[{length}]=0x{entryCode:X}.");
            result[i] = (entryCode, length);

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
        return result;
    }

    /// <summary>
    /// Encode a value into Vorbis's custom 32-bit float format defined by
    /// <see cref="VorbisMath.Float32Unpack"/>. The format packs as
    /// [sign:1][exponent:10][mantissa:21] with value = mantissa * 2^(exp - 788).
    /// </summary>
    internal static uint EncodeFloat32(double value)
    {
        if (value == 0.0) return 0u;
        bool negative = value < 0;
        double mag = Math.Abs(value);
        // Find exponent so that mantissa lies in [2^20, 2^21).
        int exponent = 788;
        // Scale up if too small.
        while (mag < (1 << 20))
        {
            mag *= 2.0;
            exponent--;
            if (exponent < 0) { exponent = 0; break; }
        }
        // Scale down if too large.
        while (mag >= (1 << 21))
        {
            mag /= 2.0;
            exponent++;
            if (exponent >= 1024) { exponent = 1023; break; }
        }
        long mantissa = (long)Math.Round(mag);
        if (mantissa < 0) mantissa = 0;
        if (mantissa > 0x1FFFFF) mantissa = 0x1FFFFF;
        uint result = (uint)mantissa & 0x1FFFFFu;
        result |= ((uint)exponent & 0x3FFu) << 21;
        if (negative) result |= 0x80000000u;
        return result;
    }
}
