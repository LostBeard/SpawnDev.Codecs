// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VorbisAudioDecoderGpu - end-to-end Vorbis audio packet decoder on
// GPU. Pairs with VorbisAudioEncoderGpu - closes the v1 Vorbis
// encoder/decoder pair.
//
// v1 scope (matches the encoder's bit-exact silence-path scope):
//   - Mono + stereo audio.
//   - GPU-resident post-spectrum chain: IMDCT -> window apply ->
//     overlap-add -> interleave -> readback as float PCM.
//   - Bit-stream spectrum decode (floor + residue + Huffman) currently
//     delegates to the CPU VorbisAudioDecoder so we have a working
//     end-to-end decode TODAY. The Vorbis-Huffman GPU bit-reader is
//     the keystone primitive that will let the entire decode run
//     in-kernel; it lands as a follow-up integration step (same
//     pattern as Opus entropy decoder).
//
// State maintained per-stream (across packets):
//   - Previous block's right-half samples per channel, for overlap-add
//     into the next packet's left half.
//   - Previous block size (Vorbis can switch between long/short blocks).
//
// Per-packet flow:
//   1. CPU spectrum decode -> channel-major spectrum coefficients
//      (VorbisAudioDecoder internals). For silence packets the
//      coefficients come back all-zero.
//   2. Upload spectrum to GPU.
//   3. GPU IMDCT per channel (ImdctReferenceGpu equivalent path; we
//      currently use the CPU IMDCT as v1 placeholder pending the
//      ImdctReferenceGpu single-block helper - its existing kernel
//      is batched-only).
//   4. GPU PostImdct: window apply + overlap-add against previous
//      right-half + new-right-half-save (one kernel pass per channel).
//   5. GPU Interleave: channel-major -> sample-major.
//   6. Readback PCM, return to caller.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Transforms;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// End-to-end Vorbis audio decoder running the post-spectrum chain on
/// GPU. Pairs with <see cref="VorbisAudioEncoderGpu"/>.
/// </summary>
public sealed class VorbisAudioDecoderGpu : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly VorbisIdentificationHeader _ident;
    private readonly VorbisSetupHeader _setup;
    // Pre-built Huffman decoders, one per codebook in the setup header.
    // Construction-time metadata setup per CLAUDE.md cardinal rule's
    // "metadata struct setup" carve-out. No CPU decoder INSTANCE needed.
    private readonly VorbisHuffmanDecoder[] _huffman;

    // GPU per-channel previous-right-half buffers (lifetime = decoder lifetime).
    // null until the first packet has been decoded; sized to halfBlock at that point.
    private MemoryBuffer1D<float, Stride1D.Dense>? _dPrevRightHalf;
    private int _previousBlockSize = 0;
    private int _previousHalfBlock = 0;

    // Pre-uploaded static lookups (lifetime = decoder lifetime).
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _dInverseDb;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dXListsFlat;
    private readonly int[] _xListOffsets;   // [floorIdx] -> offset into _dXListsFlat
    private readonly int[] _xListLengths;   // [floorIdx] -> XList.Length
    private readonly int _maxXListLength;

    // Vorbis v2 infrastructure (Plans/PLAN-Vorbis-Decoder-V2-...md
    // Step 3a). Per-stream flat-packed setup + codebook tables uploaded
    // once at construction time (CARDINAL rule "metadata struct setup"
    // carve-out). The v2 kernel reads from these directly to do
    // bit-stream decode on GPU - replacing the CPU work in
    // DecodeSpectrumOnCpu. The dispatch itself (Step 3b) is a
    // follow-up commit.
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2FloorScalars;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2FloorPartitionClassList;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2FloorPartitionClassListOffsets;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2FloorClassDimensions;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2FloorClassDimensionsOffsets;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2FloorClassSubclasses;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2FloorClassSubclassesOffsets;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2FloorClassMasterbooks;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2FloorClassMasterbooksOffsets;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2FloorClassSubclassBooks;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2FloorClassSubclassBooksOffsets;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2ResidueScalars;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2ResidueBooks;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2ResidueBooksOffsets;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2MappingScalars;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2MappingMux;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2MappingMuxOffsets;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2MappingFloors;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2MappingResidues;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2MappingSubmapOffsets;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2ModeBlockFlags;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2ModeMappings;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2AllChildren;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2AllLeafToEntry;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2ChildrenOffsets;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2LeafOffsets;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2MaxDepths;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2CodebookParams;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2AllMultiplicands;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2MultOffsets;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2MultLengths;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2CodebookDimensions;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2CodebookEntries;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2CodebookLookupTypes;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2CodebookQuantvals;
    private readonly MemoryBuffer1D<double, Stride1D.Dense> _dV2CodebookMinValues;
    private readonly MemoryBuffer1D<double, Stride1D.Dense> _dV2CodebookDeltaValues;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2CodebookSequenceP;

    // Pre-built struct that bundles all v2 ArrayViews into one kernel parameter.
    private readonly VorbisPacketDecodeStaticInputs _v2StaticInputs;

    // The v2 packet-decode kernel.
    private readonly Action<
        Index1D,
        ArrayView<byte>,
        int, int, int, int, int, int,
        VorbisPacketDecodeStaticInputs,
        ArrayView<int>,
        ArrayView<float>,
        ArrayView<int>,
        ArrayView<float>>
        _v2DecodeKernel;

    // Compiled kernels.
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<float>,
        ArrayView<float>, ArrayView<int>, ArrayView<byte>, int, int, int, int>
        _floorRenderKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>
        _multiplyKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>> _inverseCouplingKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, int> _imdctKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
        ArrayView<float>, ArrayView<float>, int> _postImdctKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, int, int> _interleaveKernel;

    /// <summary>Identification header (sample rate, channels, etc.).</summary>
    public VorbisIdentificationHeader Identification => _ident;

    /// <summary>Setup header (codebooks, floors, residues, mappings, modes).</summary>
    public VorbisSetupHeader Setup => _setup;

    /// <summary>Construct + bind to the accelerator.</summary>
    public VorbisAudioDecoderGpu(
        Accelerator accelerator,
        VorbisIdentificationHeader ident,
        VorbisSetupHeader setup)
    {
        _accelerator = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        _ident = ident ?? throw new ArgumentNullException(nameof(ident));
        _setup = setup ?? throw new ArgumentNullException(nameof(setup));
        // Build Huffman decoders directly from the setup codebooks - same
        // expression VorbisAudioDecoder uses, but here we own them so the
        // GPU decoder needs no CPU decoder instance.
        _huffman = new VorbisHuffmanDecoder[setup.Codebooks.Length];
        for (int i = 0; i < setup.Codebooks.Length; i++)
        {
            var cb = setup.Codebooks[i];
            _huffman[i] = new VorbisHuffmanDecoder(VorbisHuffman.Build(cb.Lengths));
        }

        // Pre-flatten + upload static floor data: xLists per floor + 256-entry inverse-dB lookup.
        int totalXList = 0, maxXList = 0;
        _xListOffsets = new int[setup.Floors.Length];
        _xListLengths = new int[setup.Floors.Length];
        for (int f = 0; f < setup.Floors.Length; f++)
        {
            var floor = setup.Floors[f];
            _xListOffsets[f] = totalXList;
            _xListLengths[f] = floor.XList.Length;
            totalXList += floor.XList.Length;
            if (floor.XList.Length > maxXList) maxXList = floor.XList.Length;
        }
        _maxXListLength = maxXList;
        var xListsFlat = new int[Math.Max(1, totalXList)];
        for (int f = 0; f < setup.Floors.Length; f++)
        {
            Array.Copy(setup.Floors[f].XList, 0, xListsFlat, _xListOffsets[f], _xListLengths[f]);
        }
        _dXListsFlat = accelerator.Allocate1D<int>(xListsFlat.Length);
        _dXListsFlat.View.CopyFromCPU(xListsFlat);
        _dInverseDb = accelerator.Allocate1D<float>(256);
        _dInverseDb.View.CopyFromCPU(VorbisFloor1InverseDbGpu.BuildInverseDbTable());

        // Vorbis v2 infrastructure: build flat-packed setup + codebook
        // tables, upload once, build the static-inputs struct used by
        // VorbisPacketDecodeKernel.Run, compile the kernel.
        var v2Setup = VorbisSetupHeaderGpu.Build(setup);
        var v2Codebooks = VorbisHuffmanCodebookSetGpu.Build(setup.Codebooks);
        // Codebook params for Floor1: 3 ints per codebook
        // [childrenOff, leafOff, maxDepth].
        var codebookParams = new int[v2Codebooks.MaxDepths.Length * 3];
        for (int i = 0; i < v2Codebooks.MaxDepths.Length; i++)
        {
            codebookParams[i * 3 + 0] = v2Codebooks.ChildrenOffsets[i];
            codebookParams[i * 3 + 1] = v2Codebooks.LeafOffsets[i];
            codebookParams[i * 3 + 2] = v2Codebooks.MaxDepths[i];
        }
        // Setup tables.
        _dV2FloorScalars = AllocAndUpload(accelerator, v2Setup.FloorScalars);
        _dV2FloorPartitionClassList = AllocAndUpload(accelerator, v2Setup.FloorPartitionClassList);
        _dV2FloorPartitionClassListOffsets = AllocAndUpload(accelerator, v2Setup.FloorPartitionClassListOffsets);
        _dV2FloorClassDimensions = AllocAndUpload(accelerator, v2Setup.FloorClassDimensions);
        _dV2FloorClassDimensionsOffsets = AllocAndUpload(accelerator, v2Setup.FloorClassDimensionsOffsets);
        _dV2FloorClassSubclasses = AllocAndUpload(accelerator, v2Setup.FloorClassSubclasses);
        _dV2FloorClassSubclassesOffsets = AllocAndUpload(accelerator, v2Setup.FloorClassSubclassesOffsets);
        _dV2FloorClassMasterbooks = AllocAndUpload(accelerator, v2Setup.FloorClassMasterbooks);
        _dV2FloorClassMasterbooksOffsets = AllocAndUpload(accelerator, v2Setup.FloorClassMasterbooksOffsets);
        _dV2FloorClassSubclassBooks = AllocAndUpload(accelerator, v2Setup.FloorClassSubclassBooks);
        _dV2FloorClassSubclassBooksOffsets = AllocAndUpload(accelerator, v2Setup.FloorClassSubclassBooksOffsets);
        _dV2ResidueScalars = AllocAndUpload(accelerator, v2Setup.ResidueScalars);
        _dV2ResidueBooks = AllocAndUpload(accelerator, v2Setup.ResidueBooks);
        _dV2ResidueBooksOffsets = AllocAndUpload(accelerator, v2Setup.ResidueBooksOffsets);
        _dV2MappingScalars = AllocAndUpload(accelerator, v2Setup.MappingScalars);
        _dV2MappingMux = AllocAndUpload(accelerator, v2Setup.MappingMux);
        _dV2MappingMuxOffsets = AllocAndUpload(accelerator, v2Setup.MappingMuxOffsets);
        _dV2MappingFloors = AllocAndUpload(accelerator, v2Setup.MappingFloors);
        _dV2MappingResidues = AllocAndUpload(accelerator, v2Setup.MappingResidues);
        _dV2MappingSubmapOffsets = AllocAndUpload(accelerator, v2Setup.MappingSubmapOffsets);
        // Mode block flags shipped as byte[]; convert to int[] for GPU.
        var modeBlockFlagsAsInt = new int[v2Setup.ModeBlockFlags.Length];
        for (int i = 0; i < modeBlockFlagsAsInt.Length; i++)
            modeBlockFlagsAsInt[i] = v2Setup.ModeBlockFlags[i];
        _dV2ModeBlockFlags = AllocAndUpload(accelerator, modeBlockFlagsAsInt);
        _dV2ModeMappings = AllocAndUpload(accelerator, v2Setup.ModeMappings);
        // Codebook tables.
        _dV2AllChildren = AllocAndUpload(accelerator, v2Codebooks.AllChildren);
        _dV2AllLeafToEntry = AllocAndUpload(accelerator, v2Codebooks.AllLeafToEntry);
        _dV2ChildrenOffsets = AllocAndUpload(accelerator, v2Codebooks.ChildrenOffsets);
        _dV2LeafOffsets = AllocAndUpload(accelerator, v2Codebooks.LeafOffsets);
        _dV2MaxDepths = AllocAndUpload(accelerator, v2Codebooks.MaxDepths);
        _dV2CodebookParams = AllocAndUpload(accelerator, codebookParams);
        _dV2AllMultiplicands = AllocAndUpload(accelerator, v2Codebooks.AllMultiplicands);
        _dV2MultOffsets = AllocAndUpload(accelerator, v2Codebooks.MultOffsets);
        _dV2MultLengths = AllocAndUpload(accelerator, v2Codebooks.MultLengths);
        _dV2CodebookDimensions = AllocAndUpload(accelerator, v2Codebooks.CodebookDimensions);
        _dV2CodebookEntries = AllocAndUpload(accelerator, v2Codebooks.CodebookEntries);
        _dV2CodebookLookupTypes = AllocAndUpload(accelerator, v2Codebooks.CodebookLookupTypes);
        _dV2CodebookQuantvals = AllocAndUpload(accelerator, v2Codebooks.CodebookQuantvals);
        _dV2CodebookMinValues = AllocAndUpload(accelerator, v2Codebooks.CodebookMinValues);
        _dV2CodebookDeltaValues = AllocAndUpload(accelerator, v2Codebooks.CodebookDeltaValues);
        _dV2CodebookSequenceP = AllocAndUpload(accelerator, v2Codebooks.CodebookSequenceP);

        _v2StaticInputs = new VorbisPacketDecodeStaticInputs
        {
            FloorScalars = _dV2FloorScalars.View,
            FloorPartitionClassList = _dV2FloorPartitionClassList.View,
            FloorPartitionClassListOffsets = _dV2FloorPartitionClassListOffsets.View,
            FloorClassDimensions = _dV2FloorClassDimensions.View,
            FloorClassDimensionsOffsets = _dV2FloorClassDimensionsOffsets.View,
            FloorClassSubclasses = _dV2FloorClassSubclasses.View,
            FloorClassSubclassesOffsets = _dV2FloorClassSubclassesOffsets.View,
            FloorClassMasterbooks = _dV2FloorClassMasterbooks.View,
            FloorClassMasterbooksOffsets = _dV2FloorClassMasterbooksOffsets.View,
            FloorClassSubclassBooks = _dV2FloorClassSubclassBooks.View,
            FloorClassSubclassBooksOffsets = _dV2FloorClassSubclassBooksOffsets.View,
            ResidueScalars = _dV2ResidueScalars.View,
            ResidueBooks = _dV2ResidueBooks.View,
            ResidueBooksOffsets = _dV2ResidueBooksOffsets.View,
            MappingScalars = _dV2MappingScalars.View,
            MappingMux = _dV2MappingMux.View,
            MappingMuxOffsets = _dV2MappingMuxOffsets.View,
            MappingFloors = _dV2MappingFloors.View,
            MappingResidues = _dV2MappingResidues.View,
            MappingSubmapOffsets = _dV2MappingSubmapOffsets.View,
            ModeBlockFlags = _dV2ModeBlockFlags.View,
            ModeMappings = _dV2ModeMappings.View,
            AllChildren = _dV2AllChildren.View,
            AllLeafToEntry = _dV2AllLeafToEntry.View,
            ChildrenOffsets = _dV2ChildrenOffsets.View,
            LeafOffsets = _dV2LeafOffsets.View,
            MaxDepths = _dV2MaxDepths.View,
            CodebookParams = _dV2CodebookParams.View,
            AllMultiplicands = _dV2AllMultiplicands.View,
            MultOffsets = _dV2MultOffsets.View,
            MultLengths = _dV2MultLengths.View,
            CodebookDimensions = _dV2CodebookDimensions.View,
            CodebookEntries = _dV2CodebookEntries.View,
            CodebookLookupTypes = _dV2CodebookLookupTypes.View,
            CodebookQuantvals = _dV2CodebookQuantvals.View,
            CodebookMinValues = _dV2CodebookMinValues.View,
            CodebookDeltaValues = _dV2CodebookDeltaValues.View,
            CodebookSequenceP = _dV2CodebookSequenceP.View,
        };

        _v2DecodeKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>,
            int, int, int, int, int, int,
            VorbisPacketDecodeStaticInputs,
            ArrayView<int>,
            ArrayView<float>,
            ArrayView<int>,
            ArrayView<float>>(VorbisPacketDecodeKernel.Run);

        _floorRenderKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, ArrayView<float>,
            ArrayView<float>, ArrayView<int>, ArrayView<byte>, int, int, int, int>(FloorRenderKernel);
        _multiplyKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(MultiplyKernel);
        _inverseCouplingKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>>(InverseCouplingKernel);
        _imdctKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int>(ImdctKernel);
        _postImdctKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, ArrayView<float>, int>(PostImdctKernel);
        _interleaveKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int, int>(InterleaveKernel);
    }

    /// <summary>
    /// Decode one audio packet. Returns the interleaved float PCM samples
    /// emitted by this packet. The first packet returns an empty array
    /// (priming the overlap-add state); subsequent packets return
    /// blockSize/2 sample frames * channels values.
    /// </summary>
    public async Task<float[]> DecodePacketAsync(ReadOnlyMemory<byte> packet)
    {
        if (packet.Length == 0) return Array.Empty<float>();
        int channels = _ident.AudioChannels;

        // Dual-path dispatch: desktop backends (CPU + CUDA + OpenCL) use the
        // v2 GPU bit-stream decode kernel (cardinal-rule compliant). Browser
        // backends (WebGPU + Wasm) fall back to the v1 hybrid CPU bit-stream
        // path because:
        //   - WebGPU rejects the v2 kernel dispatch: 38 ArrayView struct
        //     fields produce 44 storage-buffer bindings, exceeding Chrome's
        //     maxStorageBuffersPerShaderStage = 10.
        //   - Wasm hits a memory OOB at dispatch (overlap with Geordi's open
        //     Bug 2). When Geordi coalesces struct-of-ArrayView bindings on
        //     WebGPU (or we restructure to combine 36 int fields into one
        //     ArrayView<int> with offset table), this branch goes away and v2
        //     ships everywhere.
        bool useV2Path = _accelerator.AcceleratorType is
            AcceleratorType.CPU or AcceleratorType.Cuda or AcceleratorType.OpenCL;

        int blockSize;
        int halfBlock;
        int residueStride;
        bool[] floorOk;
        int[] floorIndex;
        VorbisMappingConfig mapping;

        // Persistent buffers allocated conditionally per path. Disposed in
        // the finally block at function exit.
        MemoryBuffer1D<int, Stride1D.Dense>? dV2AllIntOut = null;
        MemoryBuffer1D<float, Stride1D.Dense>? dV2Residue = null;
        MemoryBuffer1D<int, Stride1D.Dense>? dV1Posteriors = null;
        MemoryBuffer1D<float, Stride1D.Dense>? dV1Residue = null;

        try
        {

        if (useV2Path)
        {
            // v2: dispatch the integration kernel, read back small header
            // info, leave posteriors + residues GPU-resident.
            residueStride = _ident.BlockSize0 / 2;
            int modeBits = VorbisMath.Ilog(_setup.Modes.Length - 1);

            var packetBytes = packet.ToArray();
            using var dPacket = _accelerator.Allocate1D<byte>(packetBytes.Length);
            dPacket.View.CopyFromCPU(packetBytes);

            long allIntOutLen = VorbisPacketDecodeKernel.ComputeAllIntOutLength(channels, _maxXListLength);
            dV2AllIntOut = _accelerator.Allocate1D<int>(allIntOutLen);
            dV2Residue = _accelerator.Allocate1D<float>((long)channels * residueStride);
            long intScratchLen = (long)channels * 256L + channels;
            using var dIntScratch = _accelerator.Allocate1D<int>(intScratchLen);
            using var dEntryVecScratch = _accelerator.Allocate1D<float>(256);

            dV2AllIntOut.View.MemSetToZero();
            dV2Residue.View.MemSetToZero();

            _v2DecodeKernel(
                new Index1D(1),
                dPacket.View,
                modeBits, channels,
                _ident.BlockSize0, _ident.BlockSize1,
                residueStride,
                _maxXListLength,
                _v2StaticInputs,
                dV2AllIntOut.View, dV2Residue.View,
                dIntScratch.View, dEntryVecScratch.View);
            await _accelerator.SynchronizeAsync();

            int[] headerArr = new int[VorbisPacketDecodeKernel.PacketHeaderLength];
            dV2AllIntOut.View.SubView(VorbisPacketDecodeKernel.PacketHeaderOffset, headerArr.Length).CopyToCPU(headerArr);
            blockSize = headerArr[1];
            halfBlock = blockSize / 2;

            int[] floorOkArr = new int[channels];
            dV2AllIntOut.View.SubView(VorbisPacketDecodeKernel.PacketHeaderLength, channels).CopyToCPU(floorOkArr);
            int[] floorIndexArr = new int[channels];
            dV2AllIntOut.View.SubView(VorbisPacketDecodeKernel.PacketHeaderLength + (long)channels, channels).CopyToCPU(floorIndexArr);

            floorOk = new bool[channels];
            for (int i = 0; i < channels; i++) floorOk[i] = floorOkArr[i] != 0;
            floorIndex = floorIndexArr;
            mapping = _setup.Mappings[_setup.Modes[headerArr[0]].Mapping];
        }
        else
        {
            // v1: CPU bit-stream decode + per-channel uploads.
            var bitstream = DecodeSpectrumOnCpu(packet);
            blockSize = bitstream.BlockSize;
            halfBlock = blockSize / 2;
            residueStride = halfBlock;
            floorOk = bitstream.FloorOk;
            floorIndex = bitstream.FloorIndexPerChannel;
            mapping = bitstream.Mapping;

            dV1Posteriors = _accelerator.Allocate1D<int>((long)channels * _maxXListLength);
            dV1Residue = _accelerator.Allocate1D<float>((long)channels * halfBlock);

            dV1Posteriors.View.MemSetToZero();
            for (int ch = 0; ch < channels; ch++)
            {
                if (floorOk[ch] && bitstream.FloorPosteriors[ch] is { } yArr)
                    dV1Posteriors.View.SubView((long)ch * _maxXListLength, yArr.Length).CopyFromCPU(yArr);
                dV1Residue.View.SubView((long)ch * halfBlock, halfBlock).CopyFromCPU(bitstream.Residues[ch]);
            }
        }

        // Common views for post-spectrum chain (uses dPosteriorsView +
        // dResidueView regardless of which path produced them).
        ArrayView<int> dPosteriorsView = useV2Path
            ? dV2AllIntOut!.View.SubView(
                VorbisPacketDecodeKernel.PacketHeaderLength + 2L * channels,
                (long)channels * _maxXListLength)
            : dV1Posteriors!.View;
        ArrayView<float> dResidueView = useV2Path ? dV2Residue!.View : dV1Residue!.View;

        // Allocate per-call GPU buffers.
        long tdBufferLen = (long)channels * blockSize;
        long specBufferLen = (long)channels * halfBlock;
        using var dFloor = _accelerator.Allocate1D<float>(specBufferLen);
        using var dSpec = _accelerator.Allocate1D<float>(specBufferLen);
        using var dTd = _accelerator.Allocate1D<float>(tdBufferLen);
        using var dWindow = _accelerator.Allocate1D<float>(blockSize);
        using var dPcmCm = _accelerator.Allocate1D<float>((long)channels * halfBlock);
        using var dPcmInterleaved = _accelerator.Allocate1D<float>((long)channels * halfBlock);

        // Floor render scratch buffers (sized for the largest possible xList).
        using var dScratchInt = _accelerator.Allocate1D<int>((long)channels * 2 * _maxXListLength);
        using var dScratchByte = _accelerator.Allocate1D<byte>((long)channels * _maxXListLength);

        dFloor.View.MemSetToZero();
        dWindow.View.CopyFromCPU(VorbisWindow.GenerateCanonical(blockSize));

        // Spec must start zeroed - silent-floor channels are left as zero
        // (the GPU multiply only runs for floorOk channels).
        dSpec.View.MemSetToZero();

        // Step 1.5: GPU floor curve render per non-silent channel. Single-
        // thread kernel orchestrating the per-channel render via
        // VorbisFloor1RenderCurveGpu.Render. dFloor was pre-zeroed for the
        // silent channels above so we can skip their dispatches entirely.
        for (int ch = 0; ch < channels; ch++)
        {
            if (!floorOk[ch]) continue;
            int floorIdx = floorIndex[ch];
            int values = _xListLengths[floorIdx];
            int xListBase = _xListOffsets[floorIdx];
            int multiplier = _setup.Floors[floorIdx].Multiplier;

            long posteriorOffs = (long)ch * _maxXListLength;
            long scratchIntOffs = (long)ch * 2 * _maxXListLength;
            long scratchByteOffs = (long)ch * _maxXListLength;
            long floorOffs = (long)ch * halfBlock;

            var posteriorView = dPosteriorsView.SubView(posteriorOffs, values);
            var scratchIntView = dScratchInt.View.SubView(scratchIntOffs, 2 * values);
            var scratchByteView = dScratchByte.View.SubView(scratchByteOffs, values);
            var floorView = dFloor.View.SubView(floorOffs, halfBlock);

            _floorRenderKernel(
                new Index1D(1),
                _dXListsFlat.View, posteriorView, floorView,
                _dInverseDb.View, scratchIntView, scratchByteView,
                xListBase, values, multiplier, halfBlock);
        }

        // Allocate / re-allocate previous-right-half buffer if blockSize changed.
        bool havePrevious = _previousBlockSize == blockSize && _dPrevRightHalf is not null;
        MemoryBuffer1D<float, Stride1D.Dense> dPrev;
        if (havePrevious)
        {
            dPrev = _dPrevRightHalf!;
        }
        else
        {
            _dPrevRightHalf?.Dispose();
            _dPrevRightHalf = _accelerator.Allocate1D<float>((long)channels * halfBlock);
            // First packet (or block-size change): zero-init the previous-right-half.
            var zeros = new float[(long)channels * halfBlock];
            _dPrevRightHalf.View.CopyFromCPU(zeros);
            dPrev = _dPrevRightHalf;
        }

        // Allocate the new-right-half buffer (replaces dPrev after this call).
        var dNewRight = _accelerator.Allocate1D<float>((long)channels * halfBlock);

        // Step 2: GPU floor x residue multiply (per-bin parallel). Silent-
        // floor channels stay zero (we don't dispatch their multiply).
        for (int ch = 0; ch < channels; ch++)
        {
            if (!floorOk[ch]) continue;
            long offs = (long)ch * halfBlock;
            var floorView = dFloor.View.SubView(offs, halfBlock);
            long residueOffs = (long)ch * residueStride;
            var residueView = dResidueView.SubView(residueOffs, halfBlock);
            var specView = dSpec.View.SubView(offs, halfBlock);
            _multiplyKernel(new Index1D(halfBlock), floorView, residueView, specView);
        }

        // Step 3: GPU inverse channel coupling per coupling step in REVERSE
        // order (Vorbis I sec 4.3.8). Mono streams have zero coupling steps.
        var couplingMag = mapping.CouplingMagnitudeChannels;
        var couplingAng = mapping.CouplingAngleChannels;
        for (int step = couplingMag.Length - 1; step >= 0; step--)
        {
            int magCh = couplingMag[step];
            int angCh = couplingAng[step];
            long magOffs = (long)magCh * halfBlock;
            long angOffs = (long)angCh * halfBlock;
            var magView = dSpec.View.SubView(magOffs, halfBlock);
            var angView = dSpec.View.SubView(angOffs, halfBlock);
            _inverseCouplingKernel(new Index1D(halfBlock), magView, angView);
        }

        // Step 4: GPU IMDCT per channel (one thread per output sample).
        // Reads from dSpec (halfBlock floats / channel), writes 2N=blockSize
        // samples / channel to dTd. Bit-near-exact mirror of CPU
        // ImdctReference.Transform via the GPU helper.
        for (int ch = 0; ch < channels; ch++)
        {
            long specBase = (long)ch * halfBlock;
            long tdBase = (long)ch * blockSize;
            var specView = dSpec.View.SubView(specBase, halfBlock);
            var tdView = dTd.View.SubView(tdBase, blockSize);

            _imdctKernel(new Index1D(blockSize), specView, tdView, halfBlock);
        }

        // Step 5: GPU PostImdct (window + overlap-add + new-right-half-save) per channel.
        for (int ch = 0; ch < channels; ch++)
        {
            long tdBase = (long)ch * blockSize;
            long halfBase = (long)ch * halfBlock;
            var tdView = dTd.View.SubView(tdBase, blockSize);
            var prevView = dPrev.View.SubView(halfBase, halfBlock);
            var newRightView = dNewRight.View.SubView(halfBase, halfBlock);
            var pcmView = dPcmCm.View.SubView(halfBase, halfBlock);

            _postImdctKernel(new Index1D(halfBlock),
                tdView, dWindow.View, prevView, newRightView, pcmView, halfBlock);
        }

        // Step 6: GPU interleave channel-major PCM -> sample-major PCM.
        if (havePrevious)
        {
            _interleaveKernel(new Index1D(halfBlock * channels),
                dPcmCm.View, dPcmInterleaved.View, channels, halfBlock);
        }

        await _accelerator.SynchronizeAsync();

        // Persist the new right half for the next call.
        _dPrevRightHalf!.Dispose();
        _dPrevRightHalf = dNewRight;
        _previousBlockSize = blockSize;
        _previousHalfBlock = halfBlock;

        // First packet: no audio output (priming overlap-add).
        if (!havePrevious) return Array.Empty<float>();

        // Read back interleaved PCM.
        var pcm = await dPcmInterleaved.CopyToHostAsync();
        return pcm;

        } // end try
        finally
        {
            // Dispose the conditionally-allocated path-specific buffers.
            dV2AllIntOut?.Dispose();
            dV2Residue?.Dispose();
            dV1Posteriors?.Dispose();
            dV1Residue?.Dispose();
        }
    }

    /// <summary>Result of the v1 CPU bit-stream parse. Floor curves are
    /// rendered on GPU, so we hand back the per-channel posterior Y values
    /// + the floor index for each channel + the residue buffers.</summary>
    private readonly record struct CpuBitstream(
        int[]?[] FloorPosteriors, int[] FloorIndexPerChannel,
        float[][] Residues, bool[] FloorOk,
        VorbisMappingConfig Mapping, int BlockSize);

    /// <summary>
    /// CPU helper for v1: parse packet via the existing CPU decoder + return
    /// per-channel floor curves + residue buffers + floor flags + mapping
    /// (for inverse coupling steps). The multiply + inverse coupling + IMDCT
    /// + post-IMDCT + interleave steps all run on GPU. The hot path moves to
    /// GPU as the Vorbis-Huffman GPU bit-reader keystone primitive lands.
    /// </summary>
    private CpuBitstream DecodeSpectrumOnCpu(ReadOnlyMemory<byte> packet)
    {
        // Parse packet header on CPU to learn blockSize.
        var bitReader = new VorbisBitReader(packet.Span);
        var header = VorbisAudioPacketHeaderParser.ParseFromReader(ref bitReader, _setup, _ident);
        int blockSize = header.BlockSize;
        int halfBlock = blockSize / 2;
        int channels = _ident.AudioChannels;

        // Per-channel floor decode (CPU bit-stream + Huffman). The
        // posterior int[] is handed to the GPU floor render kernel below.
        var mapping = _setup.Mappings[_setup.Modes[header.ModeNumber].Mapping];
        var floorOk = new bool[channels];
        var posteriors = new int[]?[channels];
        var floorIndexPerChannel = new int[channels];
        // Use our own pre-built Huffman decoders (no reflection / no CPU
        // decoder instance dependency).
        var huffman = _huffman;

        for (int ch = 0; ch < channels; ch++)
        {
            int submap = mapping.Mux[ch];
            int floorIdx = mapping.SubmapFloor[submap];
            floorIndexPerChannel[ch] = floorIdx;
            var floorCfg = _setup.Floors[floorIdx];
            int[]? posterior = VorbisFloor1Decoder.Decode(ref bitReader, floorCfg, huffman);
            floorOk[ch] = posterior is not null;
            posteriors[ch] = posterior;
        }

        // Residue decode per submap.
        var doNotDecode = new bool[channels];
        for (int ch = 0; ch < channels; ch++) doNotDecode[ch] = !floorOk[ch];
        var residueBuffers = new float[channels][];
        for (int ch = 0; ch < channels; ch++) residueBuffers[ch] = new float[halfBlock];

        for (int s = 0; s < mapping.Submaps; s++)
        {
            int residueIdx = mapping.SubmapResidue[s];
            var residueCfg = _setup.Residues[residueIdx];
            int membersInSubmap = 0;
            for (int ch = 0; ch < channels; ch++) if (mapping.Mux[ch] == s) membersInSubmap++;
            if (membersInSubmap == 0) continue;
            var subBuffers = new float[membersInSubmap][];
            var subSkip = new bool[membersInSubmap];
            int k = 0;
            for (int ch = 0; ch < channels; ch++)
            {
                if (mapping.Mux[ch] == s)
                {
                    subBuffers[k] = residueBuffers[ch];
                    subSkip[k] = doNotDecode[ch];
                    k++;
                }
            }
            VorbisResidueDecoder.Decode(
                ref bitReader, residueCfg, huffman, _setup.Codebooks,
                subBuffers, subSkip, halfBlock);
        }

        // Floor render, multiply, inverse coupling, IMDCT, and the rest of
        // the chain now all run on GPU - we hand the raw posteriors +
        // residues + flags + mapping to the caller.
        return new CpuBitstream(
            posteriors, floorIndexPerChannel, residueBuffers, floorOk, mapping, blockSize);
    }

    /// <summary>
    /// Allocate a 1D GPU buffer of length <paramref name="data"/>.Length and
    /// upload <paramref name="data"/> into it. Helper used at construction
    /// time for the v2 setup + codebook flat-pack uploads. Length-zero
    /// arrays still get an allocation of length 1 (ILGPU rejects zero-
    /// length allocations on some backends, and the shaders' offset-into-
    /// buffer accesses simply never read length-zero slices).
    /// </summary>
    private static MemoryBuffer1D<T, Stride1D.Dense> AllocAndUpload<T>(
        Accelerator accelerator, T[] data) where T : unmanaged
    {
        int len = Math.Max(1, data.Length);
        var buf = accelerator.Allocate1D<T>(len);
        if (data.Length > 0)
            buf.View.SubView(0, data.Length).CopyFromCPU(data);
        return buf;
    }

    /// <summary>
    /// Single-thread floor 1 curve render kernel orchestrator.
    /// Dispatches to <see cref="VorbisFloor1RenderCurveGpu.Render"/>.
    /// </summary>
    private static void FloorRenderKernel(
        Index1D _,
        ArrayView<int> xList, ArrayView<int> decodedY, ArrayView<float> curveOut,
        ArrayView<float> inverseDb, ArrayView<int> scratchInt, ArrayView<byte> scratchByte,
        int xListBase, int values, int multiplier, int halfBlock)
    {
        VorbisFloor1RenderCurveGpu.Render(
            xList, xListBase, values,
            decodedY, 0,
            multiplier, halfBlock,
            curveOut, 0,
            inverseDb, 0,
            scratchInt, 0,
            scratchByte, 0);
    }

    /// <summary>
    /// Per-bin floor x residue multiply kernel. One thread per spectrum bin.
    /// </summary>
    private static void MultiplyKernel(
        Index1D index, ArrayView<float> floor, ArrayView<float> residue,
        ArrayView<float> spectrum)
    {
        VorbisFloorMultiplyGpu.MultiplyAt(floor, 0, residue, 0, spectrum, 0, index.X);
    }

    /// <summary>
    /// Per-coefficient inverse coupling kernel. One thread per coefficient
    /// of the (magnitude, angle) channel pair.
    /// </summary>
    private static void InverseCouplingKernel(
        Index1D index, ArrayView<float> magBuf, ArrayView<float> angBuf)
    {
        VorbisInverseCouplingGpu.ApplyAtCoefficient(magBuf, 0, angBuf, 0, index.X);
    }

    /// <summary>
    /// Per-sample IMDCT kernel: compute one time-domain output sample
    /// from N=halfBlockSize spectrum coefficients. One thread per
    /// 2N=blockSize output sample.
    /// </summary>
    private static void ImdctKernel(
        Index1D index, ArrayView<float> spectrum, ArrayView<float> td, int halfBlockSize)
    {
        td[index.X] = ImdctReferenceGpu.Sample(spectrum, 0, halfBlockSize, index.X);
    }

    /// <summary>
    /// Per-sample PostImdct kernel: window apply + overlap-add against
    /// previous right half + save new right half. One thread per
    /// half-block sample.
    /// </summary>
    private static void PostImdctKernel(
        Index1D index,
        ArrayView<float> td, ArrayView<float> window,
        ArrayView<float> previousRightHalf, ArrayView<float> newRightHalfOut,
        ArrayView<float> pcmOut, int halfBlockSize)
    {
        VorbisPostImdctGpu.ProcessAt(td, 0, window, 0, previousRightHalf, 0,
            newRightHalfOut, 0, pcmOut, 0, halfBlockSize, index.X);
    }

    /// <summary>
    /// Per-element interleave kernel: channel-major (channel * numFrames + n)
    /// -> sample-major (n * channels + channel).
    /// </summary>
    private static void InterleaveKernel(
        Index1D index,
        ArrayView<float> channelMajor, ArrayView<float> interleavedOut,
        int channels, int numFrames)
    {
        VorbisInterleaveOutputGpu.InterleaveAt(channelMajor, 0, interleavedOut, 0,
            channels, numFrames, index.X);
    }

    /// <summary>Release GPU resources. Does NOT dispose the accelerator.</summary>
    public void Dispose()
    {
        _dPrevRightHalf?.Dispose();
        _dInverseDb.Dispose();
        _dXListsFlat.Dispose();
        // v2 infrastructure - 38 buffers.
        _dV2FloorScalars.Dispose();
        _dV2FloorPartitionClassList.Dispose();
        _dV2FloorPartitionClassListOffsets.Dispose();
        _dV2FloorClassDimensions.Dispose();
        _dV2FloorClassDimensionsOffsets.Dispose();
        _dV2FloorClassSubclasses.Dispose();
        _dV2FloorClassSubclassesOffsets.Dispose();
        _dV2FloorClassMasterbooks.Dispose();
        _dV2FloorClassMasterbooksOffsets.Dispose();
        _dV2FloorClassSubclassBooks.Dispose();
        _dV2FloorClassSubclassBooksOffsets.Dispose();
        _dV2ResidueScalars.Dispose();
        _dV2ResidueBooks.Dispose();
        _dV2ResidueBooksOffsets.Dispose();
        _dV2MappingScalars.Dispose();
        _dV2MappingMux.Dispose();
        _dV2MappingMuxOffsets.Dispose();
        _dV2MappingFloors.Dispose();
        _dV2MappingResidues.Dispose();
        _dV2MappingSubmapOffsets.Dispose();
        _dV2ModeBlockFlags.Dispose();
        _dV2ModeMappings.Dispose();
        _dV2AllChildren.Dispose();
        _dV2AllLeafToEntry.Dispose();
        _dV2ChildrenOffsets.Dispose();
        _dV2LeafOffsets.Dispose();
        _dV2MaxDepths.Dispose();
        _dV2CodebookParams.Dispose();
        _dV2AllMultiplicands.Dispose();
        _dV2MultOffsets.Dispose();
        _dV2MultLengths.Dispose();
        _dV2CodebookDimensions.Dispose();
        _dV2CodebookEntries.Dispose();
        _dV2CodebookLookupTypes.Dispose();
        _dV2CodebookQuantvals.Dispose();
        _dV2CodebookMinValues.Dispose();
        _dV2CodebookDeltaValues.Dispose();
        _dV2CodebookSequenceP.Dispose();
    }
}
