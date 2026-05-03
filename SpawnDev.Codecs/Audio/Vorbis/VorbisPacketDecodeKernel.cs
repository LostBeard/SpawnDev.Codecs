// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VorbisPacketDecodeKernel - GPU integration kernel for Vorbis v2
// audio packet decode. Single-thread kernel that wires:
//
//   1. VorbisAudioPacketHeaderGpu.Parse    (header bits)
//   2. VorbisFloor1DecoderGpu.Decode       (per channel)
//   3. VorbisResidueDecoderGpu.Decode      (per submap)
//
// Replaces the CPU `DecodeSpectrumOnCpu` path inside
// VorbisAudioDecoderGpu - the v2 step in
// Plans/PLAN-Vorbis-Decoder-V2-GPU-BitStream-Decode.md.
//
// Bit-stream decoding is fundamentally sequential (Vorbis uses a
// codeword-aligned reader, not range coding, but the per-bit walk
// remains serial). Run as one GPU thread per packet (workgroup size 1);
// the *post*-spectrum chain (floor render, multiply, IMDCT, etc.) runs
// on the existing batched per-channel kernels.
//
// Caller responsibility (host = pure coordinator):
//   - Allocate + upload the flat-packed setup header + codebook set
//     ONCE per stream (metadata struct setup carve-out per CARDINAL rule).
//   - Allocate per-call scratch + output buffers.
//   - Dispatch this kernel.
//   - Read back the small header struct + floorOk flags (the decision
//     bits the post-spectrum chain needs).
//
// Captain's 1.0.0 architectural directive: this kernel closes the
// remaining cardinal-rule gap in VorbisAudioDecoderGpu (the CPU
// VorbisBitReader instance constructed at line 352 of the existing
// DecodeSpectrumOnCpu path).

using ILGPU;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// Kernel scalars + flat-packed setup header + flat-packed codebook
/// set views grouped for ergonomics. Held by the integration class
/// across packets - allocated once per stream.
/// </summary>
public readonly struct VorbisPacketDecodeStaticInputs
{
    /// <summary>Per-floor 5 scalars: Partitions, Multiplier, RangeBits, XListLength, ClassCount.</summary>
    public required ArrayView<int> FloorScalars { get; init; }
    /// <summary>Concat of PartitionClassList across floors.</summary>
    public required ArrayView<int> FloorPartitionClassList { get; init; }
    /// <summary>Per-floor offset into FloorPartitionClassList.</summary>
    public required ArrayView<int> FloorPartitionClassListOffsets { get; init; }
    /// <summary>Concat of ClassDimensions across floors.</summary>
    public required ArrayView<int> FloorClassDimensions { get; init; }
    /// <summary>Per-floor offset into FloorClassDimensions.</summary>
    public required ArrayView<int> FloorClassDimensionsOffsets { get; init; }
    /// <summary>Concat of ClassSubclasses across floors.</summary>
    public required ArrayView<int> FloorClassSubclasses { get; init; }
    /// <summary>Per-floor offset into FloorClassSubclasses.</summary>
    public required ArrayView<int> FloorClassSubclassesOffsets { get; init; }
    /// <summary>Concat of ClassMasterbooks across floors.</summary>
    public required ArrayView<int> FloorClassMasterbooks { get; init; }
    /// <summary>Per-floor offset into FloorClassMasterbooks.</summary>
    public required ArrayView<int> FloorClassMasterbooksOffsets { get; init; }
    /// <summary>Concat of ClassSubclassBooks across floors+classes.</summary>
    public required ArrayView<int> FloorClassSubclassBooks { get; init; }
    /// <summary>Per-(floor, class) offset into FloorClassSubclassBooks.</summary>
    public required ArrayView<int> FloorClassSubclassBooksOffsets { get; init; }

    /// <summary>Per-residue 6 scalars: Type, Begin, End, PartitionSize, Classifications, Classbook.</summary>
    public required ArrayView<int> ResidueScalars { get; init; }
    /// <summary>Per-residue books table (classifications*8 ints flat).</summary>
    public required ArrayView<int> ResidueBooks { get; init; }
    /// <summary>Per-residue offset into ResidueBooks.</summary>
    public required ArrayView<int> ResidueBooksOffsets { get; init; }

    /// <summary>Per-mapping submap count.</summary>
    public required ArrayView<int> MappingScalars { get; init; }
    /// <summary>Concat of per-channel Mux across mappings.</summary>
    public required ArrayView<int> MappingMux { get; init; }
    /// <summary>Per-mapping offset into MappingMux.</summary>
    public required ArrayView<int> MappingMuxOffsets { get; init; }
    /// <summary>Concat of per-submap SubmapFloor across mappings.</summary>
    public required ArrayView<int> MappingFloors { get; init; }
    /// <summary>Concat of per-submap SubmapResidue across mappings.</summary>
    public required ArrayView<int> MappingResidues { get; init; }
    /// <summary>Per-mapping offset into MappingFloors / MappingResidues.</summary>
    public required ArrayView<int> MappingSubmapOffsets { get; init; }

    /// <summary>Per-mode block flag bit (0 = short, 1 = long).</summary>
    public required ArrayView<int> ModeBlockFlags { get; init; }
    /// <summary>Per-mode mapping index.</summary>
    public required ArrayView<int> ModeMappings { get; init; }

    /// <summary>Concat children flat-tree across all codebooks.</summary>
    public required ArrayView<int> AllChildren { get; init; }
    /// <summary>Concat leaf-to-entry tables across all codebooks.</summary>
    public required ArrayView<int> AllLeafToEntry { get; init; }
    /// <summary>Per-codebook offset into AllChildren (length codebookCount+1).</summary>
    public required ArrayView<int> ChildrenOffsets { get; init; }
    /// <summary>Per-codebook offset into AllLeafToEntry.</summary>
    public required ArrayView<int> LeafOffsets { get; init; }
    /// <summary>Per-codebook Huffman max depth.</summary>
    public required ArrayView<int> MaxDepths { get; init; }
    /// <summary>Per-codebook 3-int params [childrenOff, leafOff, maxDepth] for Floor1.</summary>
    public required ArrayView<int> CodebookParams { get; init; }
    /// <summary>Concat multiplicand tables across codebooks.</summary>
    public required ArrayView<int> AllMultiplicands { get; init; }
    /// <summary>Per-codebook offset into AllMultiplicands.</summary>
    public required ArrayView<int> MultOffsets { get; init; }
    /// <summary>Per-codebook multiplicand count.</summary>
    public required ArrayView<int> MultLengths { get; init; }
    /// <summary>Per-codebook dimensions.</summary>
    public required ArrayView<int> CodebookDimensions { get; init; }
    /// <summary>Per-codebook entries count.</summary>
    public required ArrayView<int> CodebookEntries { get; init; }
    /// <summary>Per-codebook lookup type (0 / 1 / 2).</summary>
    public required ArrayView<int> CodebookLookupTypes { get; init; }
    /// <summary>Per-codebook quantvals (lookup1_values for type 1, 0 otherwise).</summary>
    public required ArrayView<int> CodebookQuantvals { get; init; }
    /// <summary>Per-codebook MinValue (parallel array to MultOffsets).</summary>
    public required ArrayView<double> CodebookMinValues { get; init; }
    /// <summary>Per-codebook DeltaValue.</summary>
    public required ArrayView<double> CodebookDeltaValues { get; init; }
    /// <summary>Per-codebook SequenceP flag (0 / 1).</summary>
    public required ArrayView<int> CodebookSequenceP { get; init; }
}

/// <summary>
/// GPU integration kernel that decodes one Vorbis audio packet's
/// bit-stream payload (header + per-channel floor + per-submap residue)
/// in a single dispatch. Single-thread kernel: workgroup size 1, one
/// thread per packet. Wires the existing GPU primitives
/// (<see cref="VorbisAudioPacketHeaderGpu"/>,
/// <see cref="VorbisFloor1DecoderGpu"/>,
/// <see cref="VorbisResidueDecoderGpu"/>) into the integration order
/// the spec defines (Vorbis I sec 4.3).
/// </summary>
public static class VorbisPacketDecodeKernel
{
    /// <summary>
    /// Decode one audio packet's spectrum on GPU. Output buffers
    /// (packetHeader, floorOk, floorIndexPerChannel, posteriorsFlat,
    /// residuesFlat, err) are written directly. Caller pre-zeroes
    /// posteriorsFlat + residuesFlat (silent-floor channels skip the
    /// floor decode and leave their slot zero) and pre-fills
    /// classificationsScratch + entryVecScratch with caller-allocated
    /// space.
    /// </summary>
    /// <summary>
    /// Layout offsets within the combined int-output buffer (allIntOut).
    /// Section 0 [0..5)         : packet header (ModeNumber, BlockSize, IsLongBlock, PreviousWindowLong, NextWindowLong)
    /// Section 1 [5..5+ch)      : floorOk per channel
    /// Section 2 [5+ch..5+2ch)  : floorIndex per channel
    /// Section 3 [5+2ch..5+2ch+ch*maxXList) : floor posteriors per channel (channel-major)
    /// Section 4 [last 1 int]   : err flag (last position = 5+2ch + ch*maxXList)
    /// </summary>
    public const int PacketHeaderOffset = 0;
    /// <summary>Length of the packet header section in ints.</summary>
    public const int PacketHeaderLength = 5;

    /// <summary>
    /// Compute the required <c>allIntOut</c> length for the given configuration.
    /// </summary>
    public static long ComputeAllIntOutLength(int channels, int maxXListLength)
        => PacketHeaderLength + (long)channels * 2 + (long)channels * maxXListLength + 1;

    /// <summary>
    /// Compute the offset of the err flag within <c>allIntOut</c>.
    /// </summary>
    public static long ErrOutOffset(int channels, int maxXListLength)
        => PacketHeaderLength + (long)channels * 2 + (long)channels * maxXListLength;

    /// <summary>
    /// Decode one audio packet's spectrum on GPU. Output buffers
    /// (allIntOut, residuesFlatOut) are written directly. Caller pre-zeroes
    /// allIntOut + residuesFlatOut (silent-floor channels skip the floor
    /// decode and leave their slot zero).
    /// </summary>
    /// <param name="_">Dispatch index (unused; kernel is single-thread).</param>
    /// <param name="packet">Encoded audio packet bytes.</param>
    /// <param name="modeBits">Precomputed VorbisMath.Ilog(modes - 1); 0 if modes == 1.</param>
    /// <param name="channels">Audio channel count (= ident.AudioChannels).</param>
    /// <param name="blockSize0">Short-block size from ident.</param>
    /// <param name="blockSize1">Long-block size from ident.</param>
    /// <param name="halfBlock">Half-block size derived from the packet's mode (caller pre-resolves once per packet).</param>
    /// <param name="maxXListLength">Maximum XList length across all floors (sized for posterior section).</param>
    /// <param name="setup">Static, per-stream flat-packed setup tables uploaded once by the caller.</param>
    /// <param name="allIntOut">Combined int output buffer; layout per <see cref="PacketHeaderOffset"/> + sections.</param>
    /// <param name="residuesFlatOut">[channels * halfBlock] residue floats, channel-major.</param>
    /// <param name="intScratch">Combined int scratch (classifications + doNotDecode flags). Caller-allocated.</param>
    /// <param name="entryVecScratch">[max codebook dimensions] residue decode entry vector scratch.</param>
    public static void Run(
        Index1D _,
        ArrayView<byte> packet,
        int modeBits,
        int channels,
        int blockSize0,
        int blockSize1,
        int halfBlock,
        int maxXListLength,
        VorbisPacketDecodeStaticInputs setup,
        ArrayView<int> allIntOut,
        ArrayView<float> residuesFlatOut,
        ArrayView<int> intScratch,
        ArrayView<float> entryVecScratch)
    {
        // Section offsets within allIntOut.
        long floorOkBase = PacketHeaderLength;
        long floorIndexBase = floorOkBase + channels;
        long posteriorsBase = floorIndexBase + channels;
        long errBase = posteriorsBase + (long)channels * maxXListLength;

        // intScratch layout: [0..channels*partitionsToReadMax) = classifications; [tail..] = doNotDecode (channels ints)
        // The residue decoder reads partitionsToReadMax from the runtime computed value, so we
        // use the END of the scratch buffer for doNotDecode (channels ints).
        long doNotDecodeBase = intScratch.Length - channels;

        // 0. Initialize bit reader.
        var state = VorbisBitReaderGpu.Init((int)packet.Length);

        // 1. Parse audio packet header.
        var header = VorbisAudioPacketHeaderGpu.Parse(
            ref state, packet,
            modeBits,
            setup.ModeBlockFlags, 0,
            blockSize0, blockSize1);

        allIntOut[PacketHeaderOffset + 0] = header.ModeNumber;
        allIntOut[PacketHeaderOffset + 1] = header.BlockSize;
        allIntOut[PacketHeaderOffset + 2] = header.IsLongBlock;
        allIntOut[PacketHeaderOffset + 3] = header.PreviousWindowLong;
        allIntOut[PacketHeaderOffset + 4] = header.NextWindowLong;

        // 2. Resolve mapping for this mode.
        int mappingIdx = setup.ModeMappings[header.ModeNumber];
        int mappingMuxBase = setup.MappingMuxOffsets[mappingIdx];
        int mappingSubmapBase = setup.MappingSubmapOffsets[mappingIdx];
        int submapCount = setup.MappingScalars[mappingIdx];

        // 3. Per-channel floor decode.
        for (int ch = 0; ch < channels; ch++)
        {
            int submap = setup.MappingMux[mappingMuxBase + ch];
            int floorIdx = setup.MappingFloors[mappingSubmapBase + submap];
            allIntOut[floorIndexBase + ch] = floorIdx;

            int partitions = setup.FloorScalars[floorIdx * 5 + 0];
            int multiplier = setup.FloorScalars[floorIdx * 5 + 1];
            int xListLen = setup.FloorScalars[floorIdx * 5 + 3];

            int yLen = VorbisFloor1DecoderGpu.Decode(
                ref state, packet,
                partitions, multiplier, xListLen,
                setup.FloorPartitionClassList, setup.FloorPartitionClassListOffsets[floorIdx],
                setup.FloorClassDimensions, setup.FloorClassDimensionsOffsets[floorIdx],
                setup.FloorClassSubclasses, setup.FloorClassSubclassesOffsets[floorIdx],
                setup.FloorClassMasterbooks, setup.FloorClassMasterbooksOffsets[floorIdx],
                setup.FloorClassSubclassBooks, setup.FloorClassSubclassBooksOffsets[floorIdx],
                setup.FloorClassSubclassBooksOffsets, 0,
                setup.AllChildren, setup.AllLeafToEntry,
                setup.CodebookParams, 0,
                allIntOut, posteriorsBase + (long)ch * maxXListLength);

            allIntOut[floorOkBase + ch] = yLen > 0 ? 1 : 0;
        }

        // 4. Per-submap residue decode.
        // For each submap: collect channels with mapping.Mux[ch] == submap,
        // build a doNotDecode flag array (1 = skip) for those channels,
        // call the residue decoder writing into residuesFlatOut at the
        // submap's offset.
        for (int s = 0; s < submapCount; s++)
        {
            int residueIdx = setup.MappingResidues[mappingSubmapBase + s];

            // Count + collect channels in this submap, build doNotDecode flags.
            int membersInSubmap = 0;
            for (int ch = 0; ch < channels; ch++)
            {
                if (setup.MappingMux[mappingMuxBase + ch] == s)
                {
                    intScratch[doNotDecodeBase + membersInSubmap] = allIntOut[floorOkBase + ch] != 0 ? 0 : 1;
                    membersInSubmap++;
                }
            }
            if (membersInSubmap == 0) continue;

            int residueType = setup.ResidueScalars[residueIdx * 6 + 0];
            int residueBegin = setup.ResidueScalars[residueIdx * 6 + 1];
            int residueEnd = setup.ResidueScalars[residueIdx * 6 + 2];
            int partitionSize = setup.ResidueScalars[residueIdx * 6 + 3];
            int classifications = setup.ResidueScalars[residueIdx * 6 + 4];
            int classbook = setup.ResidueScalars[residueIdx * 6 + 5];

            // Find the FIRST in-submap channel index in the global channel
            // numbering - that's the offset into residuesFlatOut where
            // this submap's residue rows start. v1 minimum-viable
            // assumption: channels in the same submap are CONTIGUOUS in
            // global channel order (true for our default mapping with
            // Submaps==1 covering all channels).
            int firstSubmapChannel = -1;
            for (int ch = 0; ch < channels; ch++)
            {
                if (setup.MappingMux[mappingMuxBase + ch] == s) { firstSubmapChannel = ch; break; }
            }

            VorbisResidueDecoderGpu.Decode(
                ref state, packet,
                residueType, residueBegin, residueEnd, partitionSize, classifications, classbook,
                setup.ResidueBooks, setup.ResidueBooksOffsets[residueIdx],
                setup.AllChildren, setup.ChildrenOffsets,
                setup.AllLeafToEntry, setup.LeafOffsets,
                setup.MaxDepths,
                setup.AllMultiplicands, setup.MultOffsets, setup.MultLengths,
                setup.CodebookDimensions, setup.CodebookEntries,
                setup.CodebookLookupTypes, setup.CodebookQuantvals,
                setup.CodebookMinValues, setup.CodebookDeltaValues,
                setup.CodebookSequenceP,
                membersInSubmap,
                residuesFlatOut, (long)firstSubmapChannel * halfBlock,
                halfBlock,
                intScratch, doNotDecodeBase,
                intScratch, 0,
                entryVecScratch, 0);
        }

        // 5. Success.
        allIntOut[errBase] = 0;
    }
}
