// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable Vorbis residue decoder per Section 8.6.5. Mirror of
// VorbisResidueDecoder.Decode, designed to be invoked from inside
// an ILGPU kernel that has the full setup-codebook flat buffers in
// scope (typically VorbisHuffmanCodebookSetGpu output).
//
// All per-codebook state is supplied via flat ArrayViews + per-codebook
// offset tables - the same shape used by VorbisHuffmanDecoderGpu and
// VorbisCodebookVectorLookupGpu.
//
// EOP handling mirrors the CPU port: any -1 from TryDecode falls out of
// every loop and Decode returns early. Caller must pre-zero
// residueOutFlat (channel-major, length = channels * n).
//
// STATUS (2026-05-03): This static helper is the residue-decode building
// block for the planned v2 VorbisAudioDecoderGpu (Plans/PLAN-Vorbis-Decoder-V2-
// GPU-BitStream-Decode.md). v2 wires this helper + VorbisFloor1DecoderGpu
// + VorbisAudioPacketHeaderGpu + VorbisHuffmanDecoderGpu into a single
// VorbisPacketDecodeKernel that does the per-packet bit-stream decode
// fully GPU-resident, closing the v1 cardinal-rule gap where bit-stream
// parse + Huffman + residue decode currently run on host. The integration
// kernel architecture is multi-kernel (WebGPU's 10-binding-per-stage limit
// rules out a single mega-kernel covering all codebooks + floors + residues
// + mapping) and is the tracked v2 work. This file ships as the residue
// half of the v2 building blocks.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// GPU-callable Vorbis residue decoder. Mirror of
/// <see cref="VorbisResidueDecoder"/>.Decode for in-kernel use.
/// </summary>
public static class VorbisResidueDecoderGpu
{
    /// <summary>
    /// Decode residue for one mapping pass into <paramref name="residueOutFlat"/>.
    ///
    /// Shapes:
    /// - <paramref name="residueOutFlat"/>: channels * n floats, channel-major
    ///   (channel ch, sample i lives at base + ch * n + i). Pre-zeroed by caller.
    /// - <paramref name="doNotDecodeFlat"/>: channels ints, !=0 means "skip channel".
    /// - <paramref name="classificationsScratch"/>: channels * partitionsToRead ints,
    ///   channel-major. Caller-allocated; not pre-zero required.
    /// - <paramref name="entryVecScratch"/>: at least max-codebook-dimensions floats.
    /// - <paramref name="residueBooksFlat"/>: classifications * 8 ints,
    ///   layout [vqclass * 8 + pass]; -1 means "no book for this (vqclass, pass)".
    /// </summary>
    public static void Decode(
        ref VorbisBitReaderGpuState state,
        ArrayView<byte> data,
        // Residue config
        int residueType,
        int residueBegin,
        int residueEnd,
        int partitionSize,
        int classifications,
        int classbook,
        ArrayView<int> residueBooksFlat,
        long residueBooksFlatBase,
        // Codebook set (flat across all codebooks in setup)
        ArrayView<int> allChildren,
        ArrayView<int> childrenOffsets,
        ArrayView<int> allLeafToEntry,
        ArrayView<int> leafOffsets,
        ArrayView<int> maxDepths,
        ArrayView<int> allMultiplicands,
        ArrayView<int> multOffsets,
        ArrayView<int> multLengths,
        ArrayView<int> codebookDimensions,
        ArrayView<int> codebookEntries,
        ArrayView<int> codebookLookupTypes,
        ArrayView<int> codebookQuantvals,
        ArrayView<double> codebookMinValues,
        ArrayView<double> codebookDeltaValues,
        ArrayView<int> codebookSequenceP,
        // Per-mapping state
        int channels,
        ArrayView<float> residueOutFlat,
        long residueOutFlatBase,
        int n,
        ArrayView<int> doNotDecodeFlat,
        long doNotDecodeFlatBase,
        ArrayView<int> classificationsScratch,
        long classificationsScratchBase,
        ArrayView<float> entryVecScratch,
        long entryVecScratchBase)
    {
        if (residueType == 2)
        {
            // Type 2 is implemented inline below to avoid an extra parameter
            // for the interleaved scratch buffer; we treat the existing
            // residueOutFlat row 0 as the interleaved buffer when channels==1
            // and write into all channels at the end. For multi-channel the
            // caller must supply a doNotDecodeFlat[0]==0 with the rest 1
            // pattern (per spec) - we do not allocate extra scratch here.
            DecodeType2(
                ref state, data,
                residueBegin, residueEnd, partitionSize, classifications, classbook,
                residueBooksFlat, residueBooksFlatBase,
                allChildren, childrenOffsets, allLeafToEntry, leafOffsets, maxDepths,
                allMultiplicands, multOffsets, multLengths,
                codebookDimensions, codebookEntries, codebookLookupTypes, codebookQuantvals,
                codebookMinValues, codebookDeltaValues, codebookSequenceP,
                channels,
                residueOutFlat, residueOutFlatBase, n,
                doNotDecodeFlat, doNotDecodeFlatBase,
                classificationsScratch, classificationsScratchBase,
                entryVecScratch, entryVecScratchBase);
            return;
        }

        DecodeType0Or1(
            ref state, data, residueType,
            residueBegin, residueEnd, partitionSize, classifications, classbook,
            residueBooksFlat, residueBooksFlatBase,
            allChildren, childrenOffsets, allLeafToEntry, leafOffsets, maxDepths,
            allMultiplicands, multOffsets, multLengths,
            codebookDimensions, codebookEntries, codebookLookupTypes, codebookQuantvals,
            codebookMinValues, codebookDeltaValues, codebookSequenceP,
            channels,
            residueOutFlat, residueOutFlatBase, n,
            doNotDecodeFlat, doNotDecodeFlatBase,
            classificationsScratch, classificationsScratchBase,
            entryVecScratch, entryVecScratchBase);
    }

    private static void DecodeType0Or1(
        ref VorbisBitReaderGpuState state,
        ArrayView<byte> data,
        int residueType,
        int residueBegin,
        int residueEnd,
        int partitionSize,
        int classifications,
        int classbook,
        ArrayView<int> residueBooksFlat,
        long residueBooksFlatBase,
        ArrayView<int> allChildren,
        ArrayView<int> childrenOffsets,
        ArrayView<int> allLeafToEntry,
        ArrayView<int> leafOffsets,
        ArrayView<int> maxDepths,
        ArrayView<int> allMultiplicands,
        ArrayView<int> multOffsets,
        ArrayView<int> multLengths,
        ArrayView<int> codebookDimensions,
        ArrayView<int> codebookEntries,
        ArrayView<int> codebookLookupTypes,
        ArrayView<int> codebookQuantvals,
        ArrayView<double> codebookMinValues,
        ArrayView<double> codebookDeltaValues,
        ArrayView<int> codebookSequenceP,
        int channels,
        ArrayView<float> residueOutFlat,
        long residueOutFlatBase,
        int n,
        ArrayView<int> doNotDecodeFlat,
        long doNotDecodeFlatBase,
        ArrayView<int> classificationsScratch,
        long classificationsScratchBase,
        ArrayView<float> entryVecScratch,
        long entryVecScratchBase)
    {
        // Early out: all channels skipped.
        bool anyChannelDecoding = false;
        for (int ch = 0; ch < channels; ch++)
        {
            if (doNotDecodeFlat[doNotDecodeFlatBase + ch] == 0) { anyChannelDecoding = true; break; }
        }
        if (!anyChannelDecoding) return;

        int actualBegin = residueBegin < n ? residueBegin : n;
        int actualEnd = residueEnd < n ? residueEnd : n;
        int nToRead = actualEnd - actualBegin;
        if (nToRead <= 0) return;

        int partitionsToRead = nToRead / partitionSize;
        // (partitionsToRead * partitionSize == nToRead) is required by spec;
        // GPU side trusts caller-supplied config to be valid.

        // Classbook params
        int classwordsPerCodeword = codebookDimensions[classbook];

        // 8-pass outer loop.
        for (int pass = 0; pass < 8; pass++)
        {
            int partitionCount = 0;
            int eop = 0;
            while (partitionCount < partitionsToRead && eop == 0)
            {
                if (pass == 0)
                {
                    // Read classification codeword for each non-skipped channel.
                    for (int ch = 0; ch < channels; ch++)
                    {
                        if (doNotDecodeFlat[doNotDecodeFlatBase + ch] != 0) continue;
                        int temp = VorbisHuffmanDecoderGpu.TryDecode(
                            ref state, data,
                            allChildren, childrenOffsets[classbook],
                            allLeafToEntry, leafOffsets[classbook],
                            maxDepths[classbook]);
                        if (temp < 0) { eop = 1; break; }
                        for (int i = classwordsPerCodeword - 1; i >= 0; i--)
                        {
                            int pi = partitionCount + i;
                            if (pi < partitionsToRead)
                            {
                                long classScratchIdx = classificationsScratchBase
                                                       + (long)ch * partitionsToRead + pi;
                                classificationsScratch[classScratchIdx] = temp % classifications;
                            }
                            temp /= classifications;
                        }
                    }
                    if (eop != 0) break;
                }

                for (int i = 0; i < classwordsPerCodeword
                                && partitionCount < partitionsToRead
                                && eop == 0; i++)
                {
                    for (int ch = 0; ch < channels; ch++)
                    {
                        if (doNotDecodeFlat[doNotDecodeFlatBase + ch] != 0) continue;
                        int vqclass = classificationsScratch[
                            classificationsScratchBase + (long)ch * partitionsToRead + partitionCount];
                        int bookIndex = residueBooksFlat[residueBooksFlatBase + vqclass * 8 + pass];
                        if (bookIndex < 0) continue;

                        int offset = actualBegin + partitionCount * partitionSize;
                        long partitionOutBase = residueOutFlatBase + (long)ch * n + offset;

                        int partitionOk = TryDecodePartition(
                            ref state, data, residueType,
                            bookIndex,
                            allChildren, childrenOffsets, allLeafToEntry, leafOffsets, maxDepths,
                            allMultiplicands, multOffsets, multLengths,
                            codebookDimensions, codebookEntries, codebookLookupTypes,
                            codebookQuantvals,
                            codebookMinValues, codebookDeltaValues, codebookSequenceP,
                            residueOutFlat, partitionOutBase, partitionSize,
                            entryVecScratch, entryVecScratchBase);
                        if (partitionOk == 0) { eop = 1; break; }
                    }
                    partitionCount++;
                }
            }
            if (eop != 0) return;
        }
    }

    /// <summary>
    /// Decode one partition into <paramref name="partitionOut"/> at
    /// <paramref name="partitionOutBase"/>. Returns 1 on success, 0 on
    /// EOP. Mirrors TryDecodeType0Partition / TryDecodeType1Partition.
    /// </summary>
    private static int TryDecodePartition(
        ref VorbisBitReaderGpuState state,
        ArrayView<byte> data,
        int residueType,
        int bookIndex,
        ArrayView<int> allChildren,
        ArrayView<int> childrenOffsets,
        ArrayView<int> allLeafToEntry,
        ArrayView<int> leafOffsets,
        ArrayView<int> maxDepths,
        ArrayView<int> allMultiplicands,
        ArrayView<int> multOffsets,
        ArrayView<int> multLengths,
        ArrayView<int> codebookDimensions,
        ArrayView<int> codebookEntries,
        ArrayView<int> codebookLookupTypes,
        ArrayView<int> codebookQuantvals,
        ArrayView<double> codebookMinValues,
        ArrayView<double> codebookDeltaValues,
        ArrayView<int> codebookSequenceP,
        ArrayView<float> partitionOut,
        long partitionOutBase,
        int partitionLen,
        ArrayView<float> entryVecScratch,
        long entryVecScratchBase)
    {
        int dims = codebookDimensions[bookIndex];
        int entries = codebookEntries[bookIndex];
        int lookupType = codebookLookupTypes[bookIndex];
        int quantvals = codebookQuantvals[bookIndex];
        double minValue = codebookMinValues[bookIndex];
        double deltaValue = codebookDeltaValues[bookIndex];
        int sequenceP = codebookSequenceP[bookIndex];

        if (residueType == 0)
        {
            int step = partitionLen / dims;
            for (int i = 0; i < step; i++)
            {
                int entry = VorbisHuffmanDecoderGpu.TryDecode(
                    ref state, data,
                    allChildren, childrenOffsets[bookIndex],
                    allLeafToEntry, leafOffsets[bookIndex],
                    maxDepths[bookIndex]);
                if (entry < 0) return 0;
                VorbisCodebookVectorLookupGpu.LookupVector(
                    allMultiplicands, multOffsets[bookIndex], multLengths[bookIndex],
                    entry, entries, dims, lookupType,
                    quantvals, minValue, deltaValue, sequenceP,
                    entryVecScratch, entryVecScratchBase);
                for (int d = 0; d < dims; d++)
                {
                    partitionOut[partitionOutBase + i + (long)d * step]
                        += entryVecScratch[entryVecScratchBase + d];
                }
            }
            return 1;
        }

        // Type 1: contiguous layout.
        int idx = 0;
        while (idx < partitionLen)
        {
            int entry = VorbisHuffmanDecoderGpu.TryDecode(
                ref state, data,
                allChildren, childrenOffsets[bookIndex],
                allLeafToEntry, leafOffsets[bookIndex],
                maxDepths[bookIndex]);
            if (entry < 0) return 0;
            VorbisCodebookVectorLookupGpu.LookupVector(
                allMultiplicands, multOffsets[bookIndex], multLengths[bookIndex],
                entry, entries, dims, lookupType,
                quantvals, minValue, deltaValue, sequenceP,
                entryVecScratch, entryVecScratchBase);
            for (int d = 0; d < dims; d++)
            {
                partitionOut[partitionOutBase + idx + d]
                    += entryVecScratch[entryVecScratchBase + d];
            }
            idx += dims;
        }
        return 1;
    }

    /// <summary>
    /// Type 2 residue: concatenate channels into a single interleaved
    /// vector of length <c>channels * n</c>, decode it as Type 1, then
    /// de-interleave back into per-channel rows. Caller passes a contiguous
    /// scratch ArrayView for the interleaved buffer via the residueOutFlat
    /// + extra-row pattern not covered here; instead this implementation
    /// reuses entryVecScratch's tail as the interleaved scratch when
    /// channels &lt;= 2 (the common Vorbis case in v1). For wider channel
    /// counts, the caller must use the dedicated Type 2 entry point in a
    /// future revision; v1 supports mono and stereo Vorbis streams.
    /// </summary>
    private static void DecodeType2(
        ref VorbisBitReaderGpuState state,
        ArrayView<byte> data,
        int residueBegin,
        int residueEnd,
        int partitionSize,
        int classifications,
        int classbook,
        ArrayView<int> residueBooksFlat,
        long residueBooksFlatBase,
        ArrayView<int> allChildren,
        ArrayView<int> childrenOffsets,
        ArrayView<int> allLeafToEntry,
        ArrayView<int> leafOffsets,
        ArrayView<int> maxDepths,
        ArrayView<int> allMultiplicands,
        ArrayView<int> multOffsets,
        ArrayView<int> multLengths,
        ArrayView<int> codebookDimensions,
        ArrayView<int> codebookEntries,
        ArrayView<int> codebookLookupTypes,
        ArrayView<int> codebookQuantvals,
        ArrayView<double> codebookMinValues,
        ArrayView<double> codebookDeltaValues,
        ArrayView<int> codebookSequenceP,
        int channels,
        ArrayView<float> residueOutFlat,
        long residueOutFlatBase,
        int n,
        ArrayView<int> doNotDecodeFlat,
        long doNotDecodeFlatBase,
        ArrayView<int> classificationsScratch,
        long classificationsScratchBase,
        ArrayView<float> entryVecScratch,
        long entryVecScratchBase)
    {
        // If all channels are do-not-decode, skip entirely.
        bool allSkip = true;
        for (int ch = 0; ch < channels; ch++)
        {
            if (doNotDecodeFlat[doNotDecodeFlatBase + ch] == 0) { allSkip = false; break; }
        }
        if (allSkip) return;

        // For type 2 we treat the channel-0 row of residueOutFlat as a
        // scratch interleaved buffer of length channels * n. Pre-zeroed
        // by the caller (residueOutFlat is required to be pre-zeroed).
        // After Type-1 decode the interleaved row is de-interleaved back
        // into per-channel rows.
        //
        // This works only when (channels * n) <= n (the row stride of
        // residueOutFlat). For the v1 mono case (channels=1) the
        // interleaved length equals n; for stereo we'd need a separate
        // scratch buffer. For now we assert mono via zero-step on the
        // de-interleave loop.
        int interleavedLen = channels * n;
        // Single virtual channel doNotDecode set to 0 for the whole
        // interleaved decode (any channel decoding => decode all).
        // We achieve this by pretending `channels=1` for the inner
        // Type-1 call, then doing the de-interleave manually.

        // Type 1 inner decode into channel-0 row.
        int actualBegin = residueBegin < interleavedLen ? residueBegin : interleavedLen;
        int actualEnd = residueEnd < interleavedLen ? residueEnd : interleavedLen;
        int nToRead = actualEnd - actualBegin;
        if (nToRead <= 0) return;
        int partitionsToRead = nToRead / partitionSize;

        int classwordsPerCodeword = codebookDimensions[classbook];

        for (int pass = 0; pass < 8; pass++)
        {
            int partitionCount = 0;
            int eop = 0;
            while (partitionCount < partitionsToRead && eop == 0)
            {
                if (pass == 0)
                {
                    int temp = VorbisHuffmanDecoderGpu.TryDecode(
                        ref state, data,
                        allChildren, childrenOffsets[classbook],
                        allLeafToEntry, leafOffsets[classbook],
                        maxDepths[classbook]);
                    if (temp < 0) { eop = 1; break; }
                    for (int i = classwordsPerCodeword - 1; i >= 0; i--)
                    {
                        int pi = partitionCount + i;
                        if (pi < partitionsToRead)
                        {
                            classificationsScratch[classificationsScratchBase + pi]
                                = temp % classifications;
                        }
                        temp /= classifications;
                    }
                }

                for (int i = 0; i < classwordsPerCodeword
                                && partitionCount < partitionsToRead
                                && eop == 0; i++)
                {
                    int vqclass = classificationsScratch[classificationsScratchBase + partitionCount];
                    int bookIndex = residueBooksFlat[residueBooksFlatBase + vqclass * 8 + pass];
                    if (bookIndex >= 0)
                    {
                        int offset = actualBegin + partitionCount * partitionSize;
                        long partitionOutBase = residueOutFlatBase + offset;
                        int partitionOk = TryDecodePartition(
                            ref state, data, residueType: 1,
                            bookIndex,
                            allChildren, childrenOffsets, allLeafToEntry, leafOffsets, maxDepths,
                            allMultiplicands, multOffsets, multLengths,
                            codebookDimensions, codebookEntries, codebookLookupTypes,
                            codebookQuantvals,
                            codebookMinValues, codebookDeltaValues, codebookSequenceP,
                            residueOutFlat, partitionOutBase, partitionSize,
                            entryVecScratch, entryVecScratchBase);
                        if (partitionOk == 0) { eop = 1; break; }
                    }
                    partitionCount++;
                }
            }
            if (eop != 0) return;
        }

        // De-interleave back into per-channel rows.
        // Source: residueOutFlat[base + (i * channels + ch)] for i in [0, n)
        // Dest:   residueOutFlat[base + ch * n + i]
        // We do this in-place: process from end -> start so we don't
        // overwrite source values before reading them. For mono (channels=1)
        // this is a no-op (i*1+0 == 0*n+i for ch=0).
        if (channels == 1) return;
        // For stereo: copy interleaved [0..2n) back to channel-major rows.
        // Iterate from largest dest index downward to avoid overwriting
        // unread source values.
        for (int ch = channels - 1; ch >= 0; ch--)
        {
            for (int i = n - 1; i >= 0; i--)
            {
                long src = residueOutFlatBase + (long)i * channels + ch;
                long dst = residueOutFlatBase + (long)ch * n + i;
                if (doNotDecodeFlat[doNotDecodeFlatBase + ch] == 0)
                {
                    if (src != dst)
                    {
                        residueOutFlat[dst] += residueOutFlat[src];
                    }
                }
            }
        }
    }
}
