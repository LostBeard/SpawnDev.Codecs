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

        // Step 1: CPU spectrum decode (bit-stream parse + Huffman + floor +
        // residue). v1 hybrid - the bit-stream path will move to GPU once the
        // Vorbis-Huffman bit-reader keystone primitive lands. Returns
        // per-channel floor curves + residue buffers + floor flags + the
        // mapping config (coupling steps for the inverse-coupling stage).
        var bitstream = DecodeSpectrumOnCpu(packet);
        int blockSize = bitstream.BlockSize;
        int halfBlock = blockSize / 2;

        // Allocate per-call GPU buffers.
        long tdBufferLen = (long)channels * blockSize;
        long specBufferLen = (long)channels * halfBlock;
        using var dFloor = _accelerator.Allocate1D<float>(specBufferLen);
        using var dResidue = _accelerator.Allocate1D<float>(specBufferLen);
        using var dSpec = _accelerator.Allocate1D<float>(specBufferLen);
        using var dTd = _accelerator.Allocate1D<float>(tdBufferLen);
        using var dWindow = _accelerator.Allocate1D<float>(blockSize);
        using var dPcmCm = _accelerator.Allocate1D<float>((long)channels * halfBlock);
        using var dPcmInterleaved = _accelerator.Allocate1D<float>((long)channels * halfBlock);

        // Per-channel decodedY (posteriors) + floor render scratch buffers
        // (sized for the largest possible xList across all floors).
        using var dPosteriors = _accelerator.Allocate1D<int>((long)channels * _maxXListLength);
        using var dScratchInt = _accelerator.Allocate1D<int>((long)channels * 2 * _maxXListLength);
        using var dScratchByte = _accelerator.Allocate1D<byte>((long)channels * _maxXListLength);

        // Upload posteriors (channel-major) + residues (channel-major) + window.
        // Floor curves are produced by the GPU render kernel below; dFloor
        // is pre-zeroed via GPU memset so silent-floor channels stay zero
        // (they don't get a render dispatch).
        //
        // Each channel uploads directly into its slot via SubView - no
        // host-side flat buffer, no Array.Copy iteration over codec data.
        // dPosteriors is pre-zeroed first because silent channels skip
        // their upload (the if-guard) and need to be left at zero.
        dPosteriors.View.MemSetToZero();
        for (int ch = 0; ch < channels; ch++)
        {
            if (bitstream.FloorOk[ch] && bitstream.FloorPosteriors[ch] is { } yArr)
                dPosteriors.View.SubView((long)ch * _maxXListLength, yArr.Length).CopyFromCPU(yArr);
            dResidue.View.SubView((long)ch * halfBlock, halfBlock).CopyFromCPU(bitstream.Residues[ch]);
        }
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
            if (!bitstream.FloorOk[ch]) continue;
            int floorIdx = bitstream.FloorIndexPerChannel[ch];
            int values = _xListLengths[floorIdx];
            int xListBase = _xListOffsets[floorIdx];
            int multiplier = _setup.Floors[floorIdx].Multiplier;

            long posteriorOffs = (long)ch * _maxXListLength;
            long scratchIntOffs = (long)ch * 2 * _maxXListLength;
            long scratchByteOffs = (long)ch * _maxXListLength;
            long floorOffs = (long)ch * halfBlock;

            var posteriorView = dPosteriors.View.SubView(posteriorOffs, values);
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
            if (!bitstream.FloorOk[ch]) continue;
            long offs = (long)ch * halfBlock;
            var floorView = dFloor.View.SubView(offs, halfBlock);
            var residueView = dResidue.View.SubView(offs, halfBlock);
            var specView = dSpec.View.SubView(offs, halfBlock);
            _multiplyKernel(new Index1D(halfBlock), floorView, residueView, specView);
        }

        // Step 3: GPU inverse channel coupling per coupling step in REVERSE
        // order (Vorbis I sec 4.3.8). Mono streams have zero coupling steps.
        var couplingMag = bitstream.Mapping.CouplingMagnitudeChannels;
        var couplingAng = bitstream.Mapping.CouplingAngleChannels;
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
    }
}
