// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Vorbis I residue decoder per Section 8.6.5. Shared classification logic
// plus type-0/1/2 vector unpacking. Produces an N-length float residue per
// channel that the audio-packet decoder multiplies into the floor curve.
//
// Residue types:
//   0 - non-interleaved partition-first: each partition's entries are
//       consumed in partition-major order, one classification at a time.
//   1 - non-interleaved entry-first: each partition's entries are consumed
//       one at a time, moving through all classifications.
//   2 - format-interleaved: channels are interleaved sample-by-sample into a
//       single long vector, then the vector is residue-type-1 decoded.

namespace SpawnDev.Codecs.Audio.Vorbis;

internal static class VorbisResidueDecoder
{
    /// <summary>
    /// Decode residue for one mapping pass. Writes floats into each channel's
    /// <paramref name="residueOut"/> (length = n).
    /// </summary>
    /// <param name="reader">Bit reader positioned at the residue section of the audio packet.</param>
    /// <param name="config">Residue configuration from the setup header.</param>
    /// <param name="decoders">Pre-built Huffman decoders indexed by setup codebook index.</param>
    /// <param name="codebooks">Setup codebooks (for multiplicand lookup on coded vectors).</param>
    /// <param name="residueOut">Per-channel output buffers. Each must have length == n. Pre-zeroed by caller.</param>
    /// <param name="doNotDecode">Per-channel flag set by the mapping stage - silent or coupling-absent channels skip the residue entirely.</param>
    /// <param name="n">Samples per channel to decode (i.e. blockSize / 2 for non-interleaved types, blockSize * channels / 2 for type 2).</param>
    internal static void Decode(
        ref VorbisBitReader reader,
        VorbisResidueConfig config,
        VorbisHuffmanDecoder[] decoders,
        VorbisCodebook[] codebooks,
        Span<float[]> residueOut,
        ReadOnlySpan<bool> doNotDecode,
        int n)
    {
        if (config.Type == VorbisResidueType.Type2)
        {
            DecodeType2(ref reader, config, decoders, codebooks, residueOut, doNotDecode, n);
            return;
        }
        DecodeType0Or1(ref reader, config, decoders, codebooks, residueOut, doNotDecode, n);
    }

    private static void DecodeType0Or1(
        ref VorbisBitReader reader,
        VorbisResidueConfig config,
        VorbisHuffmanDecoder[] decoders,
        VorbisCodebook[] codebooks,
        Span<float[]> residueOut,
        ReadOnlySpan<bool> doNotDecode,
        int n)
    {
        int channels = residueOut.Length;
        // Early out: all channels skipped or zero-length range avoid any
        // bit-stream reads and any codebook lookups.
        bool anyChannelDecoding = false;
        for (int ch = 0; ch < channels; ch++) if (!doNotDecode[ch]) { anyChannelDecoding = true; break; }
        if (!anyChannelDecoding) return;

        // Effective decode range per RFC: clamp End to n and begin to end.
        int actualBegin = Math.Min(config.Begin, n);
        int actualEnd = Math.Min(config.End, n);
        int nToRead = actualEnd - actualBegin;
        if (nToRead <= 0) return;

        int partitionSize = config.PartitionSize;
        int partitionsToRead = nToRead / partitionSize;
        if (partitionsToRead * partitionSize != nToRead)
            throw new InvalidDataException(
                $"Residue range {nToRead} not a multiple of partition size {partitionSize}.");

        // Classifications are indexed [channel][partitionIndex].
        var classifications = new int[channels][];
        for (int ch = 0; ch < channels; ch++)
            classifications[ch] = new int[partitionsToRead];

        var classbook = codebooks[config.Classbook];
        int classwordsPerCodeword = classbook.Dimensions;
        int classificationsCount = config.Classifications;

        // Phase 1: classify. Pass 0 also reads the classification codewords.
        // Phase 2: iterate all 8 passes applying the per-classification-per-pass books.
        // EOP handling: Vorbis I sec 8.6.5 specifies that end-of-packet during
        // residue decode terminates decoding gracefully (remaining samples stay
        // at zero). Mirrors libvorbis res0.c `eopbreak` goto pattern: any -1 from
        // the EOP-aware Huffman decode falls out of every loop.
        for (int pass = 0; pass < 8; pass++)
        {
            int partitionCount = 0;
            bool eop = false;
            while (partitionCount < partitionsToRead && !eop)
            {
                if (pass == 0)
                {
                    for (int ch = 0; ch < channels; ch++)
                    {
                        if (doNotDecode[ch]) continue;
                        int temp = decoders[config.Classbook].TryDecode(ref reader);
                        if (temp < 0)
                        {
                            eop = true;
                            break;
                        }
                        for (int i = classwordsPerCodeword - 1; i >= 0; i--)
                        {
                            int pi = partitionCount + i;
                            if (pi < partitionsToRead)
                                classifications[ch][pi] = temp % classificationsCount;
                            temp /= classificationsCount;
                        }
                    }
                    if (eop) break;
                }

                for (int i = 0; i < classwordsPerCodeword && partitionCount < partitionsToRead && !eop; i++)
                {
                    for (int ch = 0; ch < channels; ch++)
                    {
                        if (doNotDecode[ch]) continue;
                        int vqclass = classifications[ch][partitionCount];
                        int bookIndex = config.Books[vqclass][pass];
                        if (bookIndex < 0) continue;
                        int offset = actualBegin + partitionCount * partitionSize;
                        var bookDecoder = decoders[bookIndex];
                        var book = codebooks[bookIndex];
                        bool partitionOk;
                        if (config.Type == VorbisResidueType.Type0)
                        {
                            partitionOk = TryDecodeType0Partition(
                                ref reader, bookDecoder, book,
                                residueOut[ch].AsSpan(offset, partitionSize));
                        }
                        else
                        {
                            partitionOk = TryDecodeType1Partition(
                                ref reader, bookDecoder, book,
                                residueOut[ch].AsSpan(offset, partitionSize));
                        }
                        if (!partitionOk) { eop = true; break; }
                    }
                    partitionCount++;
                }
            }
            if (eop) return;
        }
    }

    private static bool TryDecodeType0Partition(
        ref VorbisBitReader reader,
        VorbisHuffmanDecoder bookDecoder,
        VorbisCodebook book,
        Span<float> partitionOut)
    {
        int step = partitionOut.Length / book.Dimensions;
        Span<float> entryVec = stackalloc float[book.Dimensions];
        for (int i = 0; i < step; i++)
        {
            int entry = bookDecoder.TryDecode(ref reader);
            if (entry < 0) return false;
            VorbisCodebookVector.LookupVector(book, entry, entryVec);
            for (int d = 0; d < book.Dimensions; d++)
                partitionOut[i + d * step] += entryVec[d];
        }
        return true;
    }

    private static bool TryDecodeType1Partition(
        ref VorbisBitReader reader,
        VorbisHuffmanDecoder bookDecoder,
        VorbisCodebook book,
        Span<float> partitionOut)
    {
        int i = 0;
        Span<float> entryVec = stackalloc float[book.Dimensions];
        while (i < partitionOut.Length)
        {
            int entry = bookDecoder.TryDecode(ref reader);
            if (entry < 0) return false;
            VorbisCodebookVector.LookupVector(book, entry, entryVec);
            for (int d = 0; d < book.Dimensions; d++)
                partitionOut[i + d] += entryVec[d];
            i += book.Dimensions;
        }
        return true;
    }

    private static void DecodeType2(
        ref VorbisBitReader reader,
        VorbisResidueConfig config,
        VorbisHuffmanDecoder[] decoders,
        VorbisCodebook[] codebooks,
        Span<float[]> residueOut,
        ReadOnlySpan<bool> doNotDecode,
        int n)
    {
        int channels = residueOut.Length;
        // If all channels are do-not-decode, skip entirely.
        bool allSkip = true;
        for (int ch = 0; ch < channels; ch++) if (!doNotDecode[ch]) { allSkip = false; break; }
        if (allSkip) return;

        // Concatenate channels into a single interleaved vector of length channels * n.
        int interleavedLen = channels * n;
        var interleaved = new float[interleavedLen];
        var interleavedArr = new float[][] { interleaved };
        var interleavedDoNotDecode = new[] { false };
        DecodeType0Or1(
            ref reader,
            config with { Type = VorbisResidueType.Type1 },
            decoders, codebooks,
            interleavedArr,
            interleavedDoNotDecode,
            interleavedLen);
        // De-interleave back into per-channel outputs.
        for (int i = 0; i < n; i++)
        {
            for (int ch = 0; ch < channels; ch++)
            {
                if (!doNotDecode[ch])
                    residueOut[ch][i] += interleaved[i * channels + ch];
            }
        }
    }
}

/// <summary>Look up the multiplicand vector for one entry in a codebook lookup table.</summary>
internal static class VorbisCodebookVector
{
    /// <summary>
    /// Produce the per-dimension multiplicand vector for codebook entry
    /// <paramref name="entry"/> per Vorbis I Section 3.2.2 / libvorbis
    /// _book_unquantize. Output length must equal <c>book.Dimensions</c>.
    /// Mirrors libvorbis lib/sharedbook.c maptype 1 / maptype 2 logic
    /// including <c>fabs()</c> on the multiplicand and the cross-dimension
    /// <c>q_sequencep</c> accumulator (the previous dimension's value is
    /// added into the next dimension's value).
    /// </summary>
    internal static void LookupVector(VorbisCodebook book, int entry, Span<float> outVec)
    {
        int dims = book.Dimensions;
        if (outVec.Length != dims)
            throw new ArgumentException(
                $"outVec length {outVec.Length} != book dimensions {dims}.", nameof(outVec));

        if (book.LookupType == 0 || entry < 0 || entry >= book.Entries)
        {
            for (int d = 0; d < dims; d++) outVec[d] = 0f;
            return;
        }
        double mindel = book.MinValue;
        double delta = book.DeltaValue;
        double last = 0;
        if (book.LookupType == 1)
        {
            // Lookup type 1: each dim picks its own row from the shorter
            // multiplicand table; the per-dim index is (entry / quantvals^d) % quantvals.
            int quantvals = book.Multiplicands.Length;
            int indexDivisor = 1;
            for (int d = 0; d < dims; d++)
            {
                int multiplicandIndex = (entry / indexDivisor) % quantvals;
                double m = book.Multiplicands[multiplicandIndex];
                double val = Math.Abs(m) * delta + mindel + last;
                if (book.SequenceP) last = val;
                outVec[d] = (float)val;
                indexDivisor *= quantvals;
            }
            return;
        }
        // Lookup type 2: flat table indexed by (entry * dimensions + dim).
        int baseIndex = entry * dims;
        for (int d = 0; d < dims; d++)
        {
            int flatIndex = baseIndex + d;
            double m = (flatIndex < 0 || flatIndex >= book.Multiplicands.Length)
                ? 0
                : book.Multiplicands[flatIndex];
            double val = Math.Abs(m) * delta + mindel + last;
            if (book.SequenceP) last = val;
            outVec[d] = (float)val;
        }
    }
}
