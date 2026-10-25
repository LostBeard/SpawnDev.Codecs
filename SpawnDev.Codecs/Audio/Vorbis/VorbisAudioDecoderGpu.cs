// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// VorbisAudioDecoderGpu - end-to-end Vorbis audio packet decoder on
// GPU. Pairs with VorbisAudioEncoderGpu - closes the v1 Vorbis
// encoder/decoder pair.
//
// Scope: any channel count; the whole packet decode runs on the GPU.
//
// State maintained per-stream (across packets):
//   - Previous block's windowed right half per channel (two device buffers used in turn: one is read
//     while the next packet's right half is written to the other), for the overlap-add.
//   - Previous block size (Vorbis switches between short and long blocks).
//
// Per-packet flow (every launch bounded; per-channel launches paced with LaunchPacer), Vorbis I sec 4.3:
//   1. Bit-stream decode (header + floors + nonzero propagate + residues): VorbisPacketDecodeKernel.Step,
//      a resumable single-thread state machine of at most StepBudget work units per launch, launched in
//      batches with one done-flag readback per batch (the work is content-dependent).
//   2. Floor curve render per channel (one thread: one floor's points over halfBlock bins).
//   3. Inverse channel coupling of the residue vectors (per-coefficient parallel).
//   4. Floor x residue multiply (per-bin parallel); an unused floor leaves the channel zero.
//   5. IMDCT per channel (one thread per output sample, halfBlock terms each).
//   6. Window + block-switching overlap-add + new-right-half save, all channels in one launch (per-sample
//      parallel), written interleaved; readback PCM.
// The window of a block depends on its size and window flags (a long block next to a short one has short
// slopes): the five windows of a stream are a table uploaded once at construction.
//
// The v1 hybrid path (CPU bit-stream decode, DecodeSpectrumOnCpu) is kept as
// the reference shape behind useV2Path.

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

    // Windowed right halves of the previous block, [channels * BlockSize1 / 2] each (row stride BlockSize1 / 2).
    // The packet reads _dRightHalf[_prevRight] and writes the other one.
    private readonly MemoryBuffer1D<float, Stride1D.Dense>[] _dRightHalf;
    private int _prevRight;
    // Size of the previous block; 0 before the first packet (it only primes the overlap-add).
    private int _previousBlockSize = 0;

    // The stream's windows (VorbisWindow.Generate): [0, BlockSize0) the short block's, then four long-block
    // windows of BlockSize1 at BlockSize0 + (prevLong * 2 + nextLong) * BlockSize1.
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _dWindows;

    // Pre-uploaded static lookups (lifetime = decoder lifetime).
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _dInverseDb;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dXListsFlat;
    private readonly int[] _xListOffsets;   // [floorIdx] -> offset into _dXListsFlat
    private readonly int[] _xListLengths;   // [floorIdx] -> XList.Length
    private readonly int _maxXListLength;

    // Vorbis v2 infrastructure (Plans/PLAN-Vorbis-Decoder-V2-...md).
    // Per-stream setup + codebook tables flattened into one int pool and
    // one double pool (VorbisPacketDecodeSetup), uploaded once at
    // construction time (CARDINAL rule "metadata struct setup" carve-out).
    // The resumable v2 kernel reads from them to do the bit-stream decode
    // on GPU - replacing the CPU work in DecodeSpectrumOnCpu.
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2SetupInts;
    private readonly MemoryBuffer1D<double, Stride1D.Dense> _dV2SetupDoubles;
    private readonly int _v2CodebookCount;
    // Resumable decode state (VorbisPacketDecodeKernel.StateInts ints, zeroed per packet).
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2State;

    /// <summary>Step launches between two reads of the done flag (most packets finish in the first).</summary>
    private const int V2StepsPerCheck = 4;

    // The v2 packet-decode step kernel.
    private readonly Action<
        Index1D,
        ArrayView<byte>,
        ArrayView<int>,
        ArrayView<double>,
        ArrayView<int>,
        ArrayView<float>,
        ArrayView<int>,
        ArrayView<float>,
        ArrayView<int>,
        VorbisPacketDecodeScalars>
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
        ArrayView<float>, ArrayView<float>, int, int, int, int> _overlapKernel;

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

        // The stream's five windows (static table, like the inverse-dB lookup).
        int bs0 = ident.BlockSize0, bs1 = ident.BlockSize1;
        var windows = new float[bs0 + 4 * bs1];
        VorbisWindow.Generate(bs0, bs0, bs0).CopyTo(windows, 0);
        for (int v = 0; v < 4; v++)
            VorbisWindow.Generate(bs1, (v & 2) != 0 ? bs1 : bs0, (v & 1) != 0 ? bs1 : bs0).CopyTo(windows, bs0 + v * bs1);
        _dWindows = accelerator.Allocate1D<float>(windows.Length);
        _dWindows.View.CopyFromCPU(windows);

        long rightLen = (long)ident.AudioChannels * (bs1 / 2);
        _dRightHalf = new[] { accelerator.Allocate1D<float>(rightLen), accelerator.Allocate1D<float>(rightLen) };

        // Vorbis v2 infrastructure: flatten the setup + codebook tables into
        // the two pools, upload once, compile the resumable step kernel.
        var v2Setup = VorbisPacketDecodeSetup.Build(setup);
        _v2CodebookCount = v2Setup.CodebookCount;
        _dV2SetupInts = AllocAndUpload(accelerator, v2Setup.Ints);
        _dV2SetupDoubles = AllocAndUpload(accelerator, v2Setup.Doubles);
        _dV2State = accelerator.Allocate1D<int>(VorbisPacketDecodeKernel.StateInts);

        _v2DecodeKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView<byte>,
            ArrayView<int>,
            ArrayView<double>,
            ArrayView<int>,
            ArrayView<float>,
            ArrayView<int>,
            ArrayView<float>,
            ArrayView<int>,
            VorbisPacketDecodeScalars>(VorbisPacketDecodeKernel.Step);

        _floorRenderKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, ArrayView<float>,
            ArrayView<float>, ArrayView<int>, ArrayView<byte>, int, int, int, int>(VorbisDecodeFloorRenderKernel);
        _multiplyKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(VorbisDecodeMultiplyKernel);
        _inverseCouplingKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>>(VorbisDecodeInverseCouplingKernel);
        _imdctKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int>(VorbisDecodeImdctKernel);
        _overlapKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>,
            ArrayView<float>, ArrayView<float>, int, int, int, int>(VorbisDecodeOverlapKernel);
    }

    /// <summary>
    /// Decode one audio packet. Returns the interleaved float PCM samples
    /// emitted by this packet. The first packet returns an empty array
    /// (priming the overlap-add state); every later packet returns
    /// previousBlockSize / 4 + blockSize / 4 sample frames * channels values.
    /// </summary>
    public async Task<float[]> DecodePacketAsync(ReadOnlyMemory<byte> packet)
    {
        if (packet.Length == 0) return Array.Empty<float>();
        int channels = _ident.AudioChannels;

        // v2 (GPU bit-stream decode) ships on every backend. The step kernel
        // binds 8 storage buffers (the setup + codebook tables live in two
        // pools), within WebGPU's per-stage limit; the v1 hybrid CPU path
        // below stays as the reference shape.
        bool useV2Path = true;

        int blockSize;
        int halfBlock;
        int residueStride;
        bool[] floorOk;
        int[] floorIndex;
        int windowOffset;
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
            // v2: run the resumable decode kernel, read back the small header
            // info, leave posteriors + residues GPU-resident. The residue
            // buffer holds the longest block; rows are halfBlock apart.
            int modeBits = VorbisMath.Ilog(_setup.Modes.Length - 1);

            var packetBytes = packet.ToArray();
            using var dPacket = _accelerator.Allocate1D<byte>(packetBytes.Length);
            dPacket.View.CopyFromCPU(packetBytes);

            long allIntOutLen = VorbisPacketDecodeKernel.ComputeAllIntOutLength(channels, _maxXListLength);
            dV2AllIntOut = _accelerator.Allocate1D<int>(allIntOutLen);
            dV2Residue = _accelerator.Allocate1D<float>((long)channels * (_ident.BlockSize1 / 2));
            long intScratchLen = VorbisPacketDecodeKernel.ComputeIntScratchLength(channels, _ident.BlockSize1);
            using var dIntScratch = _accelerator.Allocate1D<int>(intScratchLen);
            using var dEntryVecScratch = _accelerator.Allocate1D<float>(MaxCodebookDimensions(_setup));

            dV2AllIntOut.View.MemSetToZero();
            dV2Residue.View.MemSetToZero();
            _dV2State.View.MemSetToZero();

            var scalars = new VorbisPacketDecodeScalars
            {
                ModeBits = modeBits,
                Channels = channels,
                BlockSize0 = _ident.BlockSize0,
                BlockSize1 = _ident.BlockSize1,
                MaxXListLength = _maxXListLength,
                CodebookCount = _v2CodebookCount,
            };
            // The work is content-dependent: launch the steps in batches,
            // read one done flag per batch (a launch after the end is a no-op).
            var pacer = new LaunchPacer(_accelerator);
            while (true)
            {
                for (int k = 0; k < V2StepsPerCheck; k++)
                {
                    _v2DecodeKernel(
                        new Index1D(1),
                        dPacket.View,
                        _dV2SetupInts.View, _dV2SetupDoubles.View,
                        dV2AllIntOut.View, dV2Residue.View,
                        dIntScratch.View, dEntryVecScratch.View,
                        _dV2State.View, scalars);
                    await pacer.LaunchedAsync();
                }
                await _accelerator.SynchronizeAsync();
                var phase = await _dV2State.View.SubView(VorbisPacketDecodeKernel.SPhase, 1).CopyToHostAsync();
                if (phase[0] == VorbisPacketDecodeKernel.PhaseDone) break;
            }

            // Header + floorOk + floorIndex are contiguous: one async readback
            // (browser backends have no synchronous GPU -> CPU copy).
            var head = await dV2AllIntOut.View.SubView(0, VorbisPacketDecodeKernel.PacketHeaderLength + 2L * channels).CopyToHostAsync();
            blockSize = head[1];
            halfBlock = blockSize / 2;
            residueStride = halfBlock;
            windowOffset = head[2] != 0 ? _ident.BlockSize0 + (head[3] * 2 + head[4]) * _ident.BlockSize1 : 0;

            floorOk = new bool[channels];
            floorIndex = new int[channels];
            for (int i = 0; i < channels; i++)
            {
                floorOk[i] = head[VorbisPacketDecodeKernel.PacketHeaderLength + i] != 0;
                floorIndex[i] = head[VorbisPacketDecodeKernel.PacketHeaderLength + channels + i];
            }
            mapping = _setup.Mappings[_setup.Modes[head[0]].Mapping];
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
            windowOffset = bitstream.WindowOffset;

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

        // Overlap-add geometry (Vorbis I sec 4.3.8): this packet returns prevN / 4 + n / 4 frames, none for the first.
        int prevBlockSize = _previousBlockSize;
        int outFrames = prevBlockSize > 0 ? prevBlockSize / 4 + blockSize / 4 : 0;

        // Allocate per-call GPU buffers.
        long tdBufferLen = (long)channels * blockSize;
        long specBufferLen = (long)channels * halfBlock;
        using var dFloor = _accelerator.Allocate1D<float>(specBufferLen);
        using var dSpec = _accelerator.Allocate1D<float>(specBufferLen);
        using var dTd = _accelerator.Allocate1D<float>(tdBufferLen);
        using var dPcmInterleaved = _accelerator.Allocate1D<float>(Math.Max(1L, (long)channels * outFrames));

        // Floor render scratch buffers (sized for the largest possible xList).
        using var dScratchInt = _accelerator.Allocate1D<int>((long)channels * 2 * _maxXListLength);
        using var dScratchByte = _accelerator.Allocate1D<byte>((long)channels * _maxXListLength);

        dFloor.View.MemSetToZero();
        // Per-channel launches below: pace them (the browser's command encoder
        // must not collect a long queue for many-channel streams).
        var chainPacer = new LaunchPacer(_accelerator);

        // Spec must start zeroed - unused-floor channels are left as zero
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
            await chainPacer.LaunchedAsync();
        }

        // Step 2: GPU inverse channel coupling of the RESIDUE vectors per coupling step in REVERSE
        // order (Vorbis I sec 4.3.6), before the floor multiply. Mono streams have zero coupling steps.
        var couplingMag = mapping.CouplingMagnitudeChannels;
        var couplingAng = mapping.CouplingAngleChannels;
        for (int step = couplingMag.Length - 1; step >= 0; step--)
        {
            var magView = dResidueView.SubView((long)couplingMag[step] * residueStride, halfBlock);
            var angView = dResidueView.SubView((long)couplingAng[step] * residueStride, halfBlock);
            _inverseCouplingKernel(new Index1D(halfBlock), magView, angView);
            await chainPacer.LaunchedAsync();
        }

        // Step 3: GPU floor x residue multiply (per-bin parallel, sec 4.3.7). Unused-
        // floor channels stay zero (we don't dispatch their multiply).
        for (int ch = 0; ch < channels; ch++)
        {
            if (!floorOk[ch]) continue;
            long offs = (long)ch * halfBlock;
            var floorView = dFloor.View.SubView(offs, halfBlock);
            var residueView = dResidueView.SubView((long)ch * residueStride, halfBlock);
            var specView = dSpec.View.SubView(offs, halfBlock);
            _multiplyKernel(new Index1D(halfBlock), floorView, residueView, specView);
            await chainPacer.LaunchedAsync();
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
            await chainPacer.LaunchedAsync();
        }

        // Step 5: window + overlap-add with the previous block + new right half, all channels, interleaved output.
        int rowStride = _ident.BlockSize1 / 2;
        var dPrev = _dRightHalf[_prevRight];
        var dNext = _dRightHalf[1 - _prevRight];
        _overlapKernel(new Index1D(channels * Math.Max(outFrames, halfBlock)),
            dTd.View, _dWindows.View.SubView(windowOffset, blockSize), dPrev.View, dNext.View,
            dPcmInterleaved.View, channels, blockSize, prevBlockSize, rowStride);

        await _accelerator.SynchronizeAsync();

        // The right half just written is the next packet's previous one.
        _prevRight = 1 - _prevRight;
        _previousBlockSize = blockSize;

        // First packet: no audio output (priming overlap-add).
        if (outFrames == 0) return Array.Empty<float>();

        // Read back interleaved PCM.
        return await dPcmInterleaved.View.SubView(0, (long)channels * outFrames).CopyToHostAsync();

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
        VorbisMappingConfig Mapping, int BlockSize, int WindowOffset);

    /// <summary>
    /// CPU helper for v1: parse packet via the existing CPU decoder + return
    /// per-channel floor curves + residue buffers + floor flags + mapping
    /// (for inverse coupling steps). The inverse coupling + multiply + IMDCT
    /// + post-IMDCT steps all run on GPU. The hot path moves to
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
        int windowOffset = header.IsLongBlock
            ? _ident.BlockSize0 + ((header.PreviousWindowLong ? 2 : 0) + (header.NextWindowLong ? 1 : 0)) * _ident.BlockSize1
            : 0;

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

        // Residue decode per submap, after the nonzero vector propagate (sec 4.3.4).
        var doNotDecode = new bool[channels];
        for (int ch = 0; ch < channels; ch++) doNotDecode[ch] = !floorOk[ch];
        for (int i = 0; i < mapping.CouplingMagnitudeChannels.Length; i++)
        {
            int mag = mapping.CouplingMagnitudeChannels[i], ang = mapping.CouplingAngleChannels[i];
            if (!doNotDecode[mag] || !doNotDecode[ang]) { doNotDecode[mag] = false; doNotDecode[ang] = false; }
        }
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

        // Floor render, inverse coupling, multiply, IMDCT, and the rest of
        // the chain now all run on GPU - we hand the raw posteriors +
        // residues + flags + mapping to the caller.
        return new CpuBitstream(
            posteriors, floorIndexPerChannel, residueBuffers, floorOk, mapping, blockSize, windowOffset);
    }

    /// <summary>Largest codebook dimension count (the residue entry-vector scratch size, at least 1).</summary>
    private static int MaxCodebookDimensions(VorbisSetupHeader setup)
    {
        int max = 1;
        foreach (var cb in setup.Codebooks) if (cb.Dimensions > max) max = cb.Dimensions;
        return max;
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
    private static void VorbisDecodeFloorRenderKernel(
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
    private static void VorbisDecodeMultiplyKernel(
        Index1D index, ArrayView<float> floor, ArrayView<float> residue,
        ArrayView<float> spectrum)
    {
        VorbisFloorMultiplyGpu.MultiplyAt(floor, 0, residue, 0, spectrum, 0, index.X);
    }

    /// <summary>
    /// Per-coefficient inverse coupling kernel. One thread per coefficient
    /// of the (magnitude, angle) channel pair.
    /// </summary>
    private static void VorbisDecodeInverseCouplingKernel(
        Index1D index, ArrayView<float> magBuf, ArrayView<float> angBuf)
    {
        VorbisInverseCouplingGpu.ApplyAtCoefficient(magBuf, 0, angBuf, 0, index.X);
    }

    /// <summary>
    /// Per-sample IMDCT kernel: compute one time-domain output sample
    /// from N=halfBlockSize spectrum coefficients. One thread per
    /// 2N=blockSize output sample.
    /// </summary>
    private static void VorbisDecodeImdctKernel(
        Index1D index, ArrayView<float> spectrum, ArrayView<float> td, int halfBlockSize)
    {
        td[index.X] = ImdctReferenceGpu.Sample(spectrum, 0, halfBlockSize, index.X);
    }

    /// <summary>
    /// Window + block-switching overlap-add + right-half save for all channels
    /// (<see cref="VorbisPostImdctGpu.OverlapSwitchingAt"/>). One thread per
    /// (frame or right-half sample, channel).
    /// </summary>
    private static void VorbisDecodeOverlapKernel(
        Index1D index,
        ArrayView<float> td, ArrayView<float> window,
        ArrayView<float> previousRightHalf, ArrayView<float> newRightHalfOut,
        ArrayView<float> pcmOut, int channels, int blockSize, int prevBlockSize, int rowStride)
    {
        VorbisPostImdctGpu.OverlapSwitchingAt(td, window, previousRightHalf, newRightHalfOut, pcmOut,
            channels, blockSize, prevBlockSize, rowStride, index.X);
    }

    /// <summary>Release GPU resources. Does NOT dispose the accelerator.</summary>
    public void Dispose()
    {
        _dRightHalf[0].Dispose();
        _dRightHalf[1].Dispose();
        _dWindows.Dispose();
        _dInverseDb.Dispose();
        _dXListsFlat.Dispose();
        // v2 infrastructure.
        _dV2SetupInts.Dispose();
        _dV2SetupDoubles.Dispose();
        _dV2State.Dispose();
    }
}
