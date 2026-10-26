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
// Every per-packet buffer is sized for the longest block at construction (Vorbis bounds a packet by
// BlockSize1 and the channel count), so a packet allocates nothing on the device. The bit-stream decode
// is the GPU step kernel only: the CPU reference decoder lives in SpawnDev.Codecs.References.

using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Transforms;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

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
    // on GPU.
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2SetupInts;
    private readonly MemoryBuffer1D<double, Stride1D.Dense> _dV2SetupDoubles;
    private readonly int _v2CodebookCount;
    // Resumable decode state (VorbisPacketDecodeKernel.StateInts ints, zeroed per packet).
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dV2State;

    // Per-packet work buffers, sized for the longest block (BlockSize1) at construction and reused.
    private readonly StageBuffer<byte> _dPacket;                       // the packet bytes (grow-only)
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dAllIntOut;  // header + floor flags + posteriors
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _dResidue;  // channels rows of BlockSize1 / 2 (row stride = halfBlock)
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dIntScratch;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _dEntryVecScratch;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _dFloor;    // channels * BlockSize1 / 2
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _dSpec;     // channels * BlockSize1 / 2
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _dTd;       // channels * BlockSize1
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _dPcm;      // channels * BlockSize1 / 2 (the most frames a packet returns)
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _dScratchInt;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _dScratchByte;

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
            System.Array.Copy(setup.Floors[f].XList, 0, xListsFlat, _xListOffsets[f], _xListLengths[f]);
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

        int channels = ident.AudioChannels;
        long halfMax = (long)channels * (bs1 / 2);
        _dPacket = new StageBuffer<byte>(accelerator);
        _dAllIntOut = accelerator.Allocate1D<int>(VorbisPacketDecodeKernel.ComputeAllIntOutLength(channels, _maxXListLength));
        _dResidue = accelerator.Allocate1D<float>(halfMax);
        _dIntScratch = accelerator.Allocate1D<int>(VorbisPacketDecodeKernel.ComputeIntScratchLength(channels, bs1));
        _dEntryVecScratch = accelerator.Allocate1D<float>(MaxCodebookDimensions(setup));
        _dFloor = accelerator.Allocate1D<float>(halfMax);
        _dSpec = accelerator.Allocate1D<float>(halfMax);
        _dTd = accelerator.Allocate1D<float>((long)channels * bs1);
        _dPcm = accelerator.Allocate1D<float>(halfMax);
        _dScratchInt = accelerator.Allocate1D<int>(Math.Max(1L, (long)channels * 2 * _maxXListLength));
        _dScratchByte = accelerator.Allocate1D<byte>(Math.Max(1L, (long)channels * _maxXListLength));

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
    /// Most interleaved samples one packet returns (<c>channels * BlockSize1 / 2</c>): the size a caller's PCM view needs for
    /// <see cref="DecodePacketAsync(ArrayView1D{byte, Stride1D.Dense}, int, ArrayView1D{float, Stride1D.Dense})"/>.
    /// </summary>
    public int MaxSamplesPerPacket => _ident.AudioChannels * (_ident.BlockSize1 / 2);

    /// <summary>
    /// Decode one audio packet that is already on the accelerator, writing the interleaved float PCM into a caller-owned
    /// view. Returns the number of sample FRAMES written (samples = frames * channels): 0 for the first packet (it primes the
    /// overlap-add), previousBlockSize / 4 + blockSize / 4 for every later one. Nothing but a few header ints crosses to the host.
    /// </summary>
    /// <param name="packet">The packet bytes (the view may be longer than <paramref name="packetLength"/>).</param>
    /// <param name="packetLength">Bytes of the packet.</param>
    /// <param name="pcmOut">Receives the samples; at least <see cref="MaxSamplesPerPacket"/> floats.</param>
    public async Task<int> DecodePacketAsync(ArrayView1D<byte, Stride1D.Dense> packet, int packetLength, ArrayView1D<float, Stride1D.Dense> pcmOut)
    {
        if (packetLength < 0 || packetLength > packet.Length) throw new ArgumentOutOfRangeException(nameof(packetLength));
        if (pcmOut.Length < MaxSamplesPerPacket)
            throw new ArgumentException($"pcmOut must hold {MaxSamplesPerPacket} floats (MaxSamplesPerPacket).", nameof(pcmOut));
        if (packetLength == 0) return 0;
        return await DecodeCoreAsync(packet.SubView(0, packetLength), pcmOut);
    }

    /// <summary>
    /// Decode one audio packet. Returns the interleaved float PCM samples
    /// emitted by this packet. The first packet returns an empty array
    /// (priming the overlap-add state); every later packet returns
    /// previousBlockSize / 4 + blockSize / 4 sample frames * channels values.
    /// A terminal CPU sink; in a browser prefer the TypedArray overload.
    /// </summary>
    public async Task<float[]> DecodePacketAsync(ReadOnlyMemory<byte> packet)
    {
        if (packet.Length == 0) return System.Array.Empty<float>();
        var dPacket = _dPacket.Ensure(packet.Length);
        DeviceTransfer.Upload(dPacket, packet.Span);
        int frames = await DecodeCoreAsync(dPacket.SubView(0, packet.Length), _dPcm.View);
        if (frames == 0) return System.Array.Empty<float>();
        return await _dPcm.View.SubView(0, (long)_ident.AudioChannels * frames).CopyToHostAsync();
    }

    /// <summary>
    /// Decode one audio packet held in a JS typed array (browser accelerators only); the samples come back as a JS
    /// <see cref="Float32Array"/> (empty for the first packet). Neither the packet nor the samples enter the .NET heap.
    /// The caller disposes the result.
    /// </summary>
    [SupportedOSPlatform("browser")]
    public async Task<Float32Array> DecodePacketAsync(TypedArray packet)
    {
        ArgumentNullException.ThrowIfNull(packet);
        int length = (int)packet.ByteLength;
        if (length == 0) return new Float32Array(0);
        var dPacket = _dPacket.Ensure(length);
        DeviceTransfer.Upload(dPacket, packet);
        int frames = await DecodeCoreAsync(dPacket.SubView(0, length), _dPcm.View);
        return await DeviceTransfer.DownloadFloat32ArrayAsync(_dPcm.View, (long)_ident.AudioChannels * frames);
    }

    /// <summary>
    /// The decode itself: <paramref name="packet"/> (exactly the packet's bytes) on the accelerator, interleaved PCM written to
    /// <paramref name="pcmOut"/>. Returns the sample frames written.
    /// </summary>
    private async Task<int> DecodeCoreAsync(ArrayView1D<byte, Stride1D.Dense> packet, ArrayView1D<float, Stride1D.Dense> pcmOut)
    {
        int channels = _ident.AudioChannels;

        // ---- bit-stream decode on the accelerator: the resumable step kernel ----
        int modeBits = VorbisMath.Ilog(_setup.Modes.Length - 1);

        _dAllIntOut.View.MemSetToZero();
        _dResidue.View.MemSetToZero();
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
                    packet,
                    _dV2SetupInts.View, _dV2SetupDoubles.View,
                    _dAllIntOut.View, _dResidue.View,
                    _dIntScratch.View, _dEntryVecScratch.View,
                    _dV2State.View, scalars);
                await pacer.LaunchedAsync();
            }
            await _accelerator.SynchronizeAsync();
            var phase = await _dV2State.View.SubView(VorbisPacketDecodeKernel.SPhase, 1).CopyToHostAsync();
            if (phase[0] == VorbisPacketDecodeKernel.PhaseDone) break;
        }

        // Header + floorOk + floorIndex are contiguous: one async readback
        // (browser backends have no synchronous GPU -> CPU copy).
        var head = await _dAllIntOut.View.SubView(0, VorbisPacketDecodeKernel.PacketHeaderLength + 2L * channels).CopyToHostAsync();
        int blockSize = head[1];
        int halfBlock = blockSize / 2;
        int residueStride = halfBlock;
        int windowOffset = head[2] != 0 ? _ident.BlockSize0 + (head[3] * 2 + head[4]) * _ident.BlockSize1 : 0;
        var floorOk = new bool[channels];
        var floorIndex = new int[channels];
        for (int i = 0; i < channels; i++)
        {
            floorOk[i] = head[VorbisPacketDecodeKernel.PacketHeaderLength + i] != 0;
            floorIndex[i] = head[VorbisPacketDecodeKernel.PacketHeaderLength + channels + i];
        }
        var mapping = _setup.Mappings[_setup.Modes[head[0]].Mapping];

        ArrayView<int> dPosteriorsView = _dAllIntOut.View.SubView(
            VorbisPacketDecodeKernel.PacketHeaderLength + 2L * channels,
            (long)channels * _maxXListLength);
        ArrayView<float> dResidueView = _dResidue.View;

        // Overlap-add geometry (Vorbis I sec 4.3.8): this packet returns prevN / 4 + n / 4 frames, none for the first.
        int prevBlockSize = _previousBlockSize;
        int outFrames = prevBlockSize > 0 ? prevBlockSize / 4 + blockSize / 4 : 0;

        long specLen = (long)channels * halfBlock;
        var dFloor = _dFloor.View.SubView(0, specLen);
        var dSpec = _dSpec.View.SubView(0, specLen);
        // Unused-floor channels stay zero: their floor render and multiply are not dispatched.
        dFloor.MemSetToZero();
        dSpec.MemSetToZero();
        // Per-channel launches below: pace them (the browser's command encoder
        // must not collect a long queue for many-channel streams).
        var chainPacer = new LaunchPacer(_accelerator);

        // Step 1: GPU floor curve render per non-silent channel (single-thread
        // kernel per channel via VorbisFloor1RenderCurveGpu.Render).
        for (int ch = 0; ch < channels; ch++)
        {
            if (!floorOk[ch]) continue;
            int floorIdx = floorIndex[ch];
            int values = _xListLengths[floorIdx];
            int xListBase = _xListOffsets[floorIdx];
            int multiplier = _setup.Floors[floorIdx].Multiplier;

            var posteriorView = dPosteriorsView.SubView((long)ch * _maxXListLength, values);
            var scratchIntView = _dScratchInt.View.SubView((long)ch * 2 * _maxXListLength, 2 * values);
            var scratchByteView = _dScratchByte.View.SubView((long)ch * _maxXListLength, values);
            var floorView = dFloor.SubView((long)ch * halfBlock, halfBlock);

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

        // Step 3: GPU floor x residue multiply (per-bin parallel, sec 4.3.7).
        for (int ch = 0; ch < channels; ch++)
        {
            if (!floorOk[ch]) continue;
            long offs = (long)ch * halfBlock;
            var floorView = dFloor.SubView(offs, halfBlock);
            var residueView = dResidueView.SubView((long)ch * residueStride, halfBlock);
            var specView = dSpec.SubView(offs, halfBlock);
            _multiplyKernel(new Index1D(halfBlock), floorView, residueView, specView);
            await chainPacer.LaunchedAsync();
        }

        // Step 4: GPU IMDCT per channel (one thread per output sample): halfBlock
        // coefficients -> blockSize samples (ImdctReferenceGpu.Sample).
        for (int ch = 0; ch < channels; ch++)
        {
            var specView = dSpec.SubView((long)ch * halfBlock, halfBlock);
            var tdView = _dTd.View.SubView((long)ch * blockSize, blockSize);
            _imdctKernel(new Index1D(blockSize), specView, tdView, halfBlock);
            await chainPacer.LaunchedAsync();
        }

        // Step 5: window + overlap-add with the previous block + new right half, all channels, interleaved output.
        int rowStride = _ident.BlockSize1 / 2;
        var dPrev = _dRightHalf[_prevRight];
        var dNext = _dRightHalf[1 - _prevRight];
        _overlapKernel(new Index1D(channels * Math.Max(outFrames, halfBlock)),
            _dTd.View.SubView(0, (long)channels * blockSize), _dWindows.View.SubView(windowOffset, blockSize),
            dPrev.View, dNext.View, pcmOut, channels, blockSize, prevBlockSize, rowStride);

        await _accelerator.SynchronizeAsync();

        // The right half just written is the next packet's previous one.
        _prevRight = 1 - _prevRight;
        _previousBlockSize = blockSize;

        // 0 for the first packet (priming the overlap-add)
        return outFrames;
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
        _dPacket.Dispose();
        _dAllIntOut.Dispose();
        _dResidue.Dispose();
        _dIntScratch.Dispose();
        _dEntryVecScratch.Dispose();
        _dFloor.Dispose();
        _dSpec.Dispose();
        _dTd.Dispose();
        _dPcm.Dispose();
        _dScratchInt.Dispose();
        _dScratchByte.Dispose();
    }
}
