// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-callable Vorbis Floor 1 posterior decoder. Mirror of
// VorbisFloor1Decoder.Decode (Vorbis I sec 7.2.3).
//
// Reads from a Vorbis bit stream:
//   1 bit    : nonzero flag (returns -1 yLen if 0)
//   N bits   : endpoint Y values (N from EndpointBits[multiplier])
//   for each partition:
//     - if cbits > 0: master codebook entry -> cval
//     - for each cdim: subclass book index from cval -> codebook entry -> y
//
// Caller flattens VorbisFloor1Config + the per-stream codebook set
// (via VorbisHuffmanCodebookSetGpu) once per stream.
//
// Returns the number of Y values written. Caller pre-zeroes yOut so
// silent-floor packets (returns 0) decode to all-zeros.

using ILGPU;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// GPU-callable Vorbis Floor 1 posterior decoder. Mirror of
/// <see cref="VorbisFloor1Decoder"/>.Decode.
/// </summary>
public static class VorbisFloor1DecoderGpu
{
    /// <summary>
    /// Decode the floor 1 Y posteriors for one channel. Returns:
    ///   - the number of Y values written to <paramref name="yOut"/>
    ///     (= xList length when the floor is non-silent)
    ///   - 0 when the packet signalled a silent floor (nonzero flag = 0;
    ///     caller's yOut buffer is left untouched).
    /// </summary>
    /// <param name="state">Vorbis bit reader state (mutated).</param>
    /// <param name="data">Packet bytes.</param>
    /// <param name="partitions">Floor partitions count.</param>
    /// <param name="multiplier">Floor multiplier (1..4).</param>
    /// <param name="xListLen">Number of Y values to decode (= XList length).</param>
    /// <param name="partitionClassList">Per-partition class index.</param>
    /// <param name="partitionClassListBase">Base offset.</param>
    /// <param name="classDimensions">Per-class dimensions (cdim).</param>
    /// <param name="classDimensionsBase">Base offset.</param>
    /// <param name="classSubclasses">Per-class log2 subclass count (cbits).</param>
    /// <param name="classSubclassesBase">Base offset.</param>
    /// <param name="classMasterbooks">Per-class master codebook index (-1 if none).</param>
    /// <param name="classMasterbooksBase">Base offset.</param>
    /// <param name="classSubclassBooksFlat">Concat of per-class subclass book arrays.</param>
    /// <param name="classSubclassBooksFlatBase">Base offset.</param>
    /// <param name="classSubclassBooksOffsets">Per-class slice offsets (length classCount+1).</param>
    /// <param name="classSubclassBooksOffsetsBase">Base offset.</param>
    /// <param name="huffmanChildren">Concat huffman flat-tree children (from VorbisHuffmanCodebookSetGpu).</param>
    /// <param name="huffmanLeafToEntry">Concat huffman leaf -> entry table.</param>
    /// <param name="codebookParams">Per-codebook 3-int params: [childrenOff, leafOff, maxDepth].</param>
    /// <param name="codebookParamsBase">Base offset.</param>
    /// <param name="yOut">Output Y array (length >= xListLen).</param>
    /// <param name="yOutBase">Base offset.</param>
    /// <returns>Number of Y values written, or 0 for silent floor.</returns>
    public static int Decode(
        ref VorbisBitReaderGpuState state, ArrayView<byte> data,
        int partitions, int multiplier, int xListLen,
        ArrayView<int> partitionClassList, long partitionClassListBase,
        ArrayView<int> classDimensions, long classDimensionsBase,
        ArrayView<int> classSubclasses, long classSubclassesBase,
        ArrayView<int> classMasterbooks, long classMasterbooksBase,
        ArrayView<int> classSubclassBooksFlat, long classSubclassBooksFlatBase,
        ArrayView<int> classSubclassBooksOffsets, long classSubclassBooksOffsetsBase,
        ArrayView<int> huffmanChildren,
        ArrayView<int> huffmanLeafToEntry,
        ArrayView<int> codebookParams, long codebookParamsBase,
        ArrayView<int> yOut, long yOutBase)
    {
        // 1. nonzero flag.
        if (VorbisBitReaderGpu.IsEnd(in state)) return 0;
        int nonzero = (int)VorbisBitReaderGpu.ReadBits(ref state, data, 1);
        if (nonzero == 0) return 0;

        // 2. Endpoint Y values. Bits = EndpointBits[multiplier]:
        //    multiplier 1 -> 8, 2 -> 7, 3 -> 7, 4 -> 6.
        int endpointBits = multiplier == 1 ? 8
                         : multiplier == 2 ? 7
                         : multiplier == 3 ? 7
                         : 6;
        yOut[yOutBase + 0] = (int)VorbisBitReaderGpu.ReadBits(ref state, data, endpointBits);
        yOut[yOutBase + 1] = (int)VorbisBitReaderGpu.ReadBits(ref state, data, endpointBits);

        // 3. Per-partition decode.
        int yIndex = 2;
        for (int partition = 0; partition < partitions; partition++)
        {
            int cls = partitionClassList[partitionClassListBase + partition];
            int cdim = classDimensions[classDimensionsBase + cls];
            int cbits = classSubclasses[classSubclassesBase + cls];
            int csub = (1 << cbits) - 1;
            int cval = 0;

            if (cbits > 0)
            {
                int masterbook = classMasterbooks[classMasterbooksBase + cls];
                cval = DecodeBook(ref state, data,
                    huffmanChildren, huffmanLeafToEntry, codebookParams, codebookParamsBase,
                    masterbook);
            }

            // Subclass books slice for this class.
            long classSlice = classSubclassBooksFlatBase
                + classSubclassBooksOffsets[classSubclassBooksOffsetsBase + cls];

            for (int j = 0; j < cdim; j++)
            {
                int book = classSubclassBooksFlat[classSlice + (cval & csub)];
                cval >>= cbits;
                if (book >= 0)
                {
                    yOut[yOutBase + yIndex] = DecodeBook(ref state, data,
                        huffmanChildren, huffmanLeafToEntry, codebookParams, codebookParamsBase,
                        book);
                }
                else
                {
                    yOut[yOutBase + yIndex] = 0;
                }
                yIndex++;
            }
        }

        return xListLen;
    }

    /// <summary>
    /// Decode one entry from codebook <paramref name="bookIndex"/> by
    /// looking up its (childrenOffset, leafOffset, maxDepth) triple in
    /// <paramref name="codebookParams"/> and walking the flat tree.
    /// </summary>
    private static int DecodeBook(
        ref VorbisBitReaderGpuState state, ArrayView<byte> data,
        ArrayView<int> children, ArrayView<int> leafToEntry,
        ArrayView<int> codebookParams, long codebookParamsBase,
        int bookIndex)
    {
        long pBase = codebookParamsBase + bookIndex * 3;
        int childOff = codebookParams[pBase + 0];
        int leafOff = codebookParams[pBase + 1];
        int maxDepth = codebookParams[pBase + 2];
        return VorbisHuffmanDecoderGpu.TryDecode(
            ref state, data,
            children, childOff, leafToEntry, leafOff, maxDepth);
    }
}
