// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// V1 GPU Vorbis audio packet encoder. Wraps the existing CPU
// VorbisAudioEncoder for setup metadata + 3-header-packet emission +
// Ogg framing (all metadata-struct-setup work allowed under the
// CARDINAL rule), and dispatches a 6-kernel chain for the per-packet
// codec-data math:
//
//   1. Window apply           (per-sample parallel)
//   2. Forward MDCT + 4/N     (per-bin parallel)
//   3. Floor fit (peak + Y)   (single thread)
//   4. Floor curve render     (single thread)
//   5. Divide + quantize      (per-bin parallel)
//   6. Bitstream emit         (single thread)
//
// Each stage is one of the Vorbis GPU primitives shipped previously.
// The integration class only orchestrates dispatches + buffer mgmt.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Vorbis;

/// <summary>
/// V1 GPU Vorbis audio encoder. Per-packet encode runs entirely on
/// the accelerator; setup + Ogg framing run on host (metadata only).
/// </summary>
public sealed class VorbisAudioEncoderGpu : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly VorbisAudioEncoderOptions _opts;
    private readonly VorbisAudioEncoder _cpuRef;
    private readonly int _endpointBits;
    private readonly int _modeBits;

    // Pre-uploaded constant buffers (lifetime = encoder lifetime).
    private readonly MemoryBuffer1D<float, global::ILGPU.Stride1D.Dense> _dInverseDb;
    private readonly MemoryBuffer1D<uint, global::ILGPU.Stride1D.Dense> _dClassbookCodes;
    private readonly MemoryBuffer1D<int, global::ILGPU.Stride1D.Dense> _dClassbookLengths;
    private readonly MemoryBuffer1D<uint, global::ILGPU.Stride1D.Dense> _dResidueBookCodes;
    private readonly MemoryBuffer1D<int, global::ILGPU.Stride1D.Dense> _dResidueBookLengths;
    private readonly MemoryBuffer1D<int, global::ILGPU.Stride1D.Dense> _dXList;

    // Compiled kernels.
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, int> _windowKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, int> _mdctKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, int, float> _floorFitKernel;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<float>, ArrayView<int>, ArrayView<byte>, int, int, int> _floorRenderKernel;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, int, float, int> _divQuantKernel;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<long>, ArrayView<int>, ArrayView<uint>, ArrayView<int>, ArrayView<uint>, ArrayView<int>, EmitPacketParams> _emitKernel;

    /// <summary>Construct + bind to the accelerator. One-time setup uploads.</summary>
    public VorbisAudioEncoderGpu(Accelerator accelerator, VorbisAudioEncoderOptions options)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        ArgumentNullException.ThrowIfNull(options);
        _accelerator = accelerator;
        _opts = options;
        _cpuRef = new VorbisAudioEncoder(options);

        var floorCfg = (VorbisFloor1Config)_cpuRef.Setup.Floors[0];
        _endpointBits = floorCfg.Multiplier switch { 1 => 8, 2 => 7, 3 => 7, _ => 6 };

        int modeCount = _cpuRef.Setup.Modes.Length;
        _modeBits = VorbisMath.Ilog(modeCount - 1);

        // Build classbook + residue book code/length tables.
        var classCb = VorbisCodebookEncoder.BuildCodewords(_cpuRef.Setup.Codebooks[0].Lengths);
        var residueCb = VorbisCodebookEncoder.BuildCodewords(_cpuRef.Setup.Codebooks[1].Lengths);
        var classCodes = new uint[classCb.Length];
        var classLens = new int[classCb.Length];
        for (int i = 0; i < classCb.Length; i++) { classCodes[i] = classCb[i].code; classLens[i] = classCb[i].length; }
        var residueCodes = new uint[residueCb.Length];
        var residueLens = new int[residueCb.Length];
        for (int i = 0; i < residueCb.Length; i++) { residueCodes[i] = residueCb[i].code; residueLens[i] = residueCb[i].length; }

        // Upload constants.
        _dInverseDb = accelerator.Allocate1D<float>(256);
        _dInverseDb.View.CopyFromCPU(VorbisFloor1InverseDbGpu.BuildInverseDbTable());
        _dClassbookCodes = accelerator.Allocate1D<uint>(classCodes.Length);
        _dClassbookCodes.View.CopyFromCPU(classCodes);
        _dClassbookLengths = accelerator.Allocate1D<int>(classLens.Length);
        _dClassbookLengths.View.CopyFromCPU(classLens);
        _dResidueBookCodes = accelerator.Allocate1D<uint>(residueCodes.Length);
        _dResidueBookCodes.View.CopyFromCPU(residueCodes);
        _dResidueBookLengths = accelerator.Allocate1D<int>(residueLens.Length);
        _dResidueBookLengths.View.CopyFromCPU(residueLens);
        _dXList = accelerator.Allocate1D<int>(floorCfg.XList.Length);
        _dXList.View.CopyFromCPU(floorCfg.XList);

        // Compile kernels.
        _windowKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int>(WindowKernel);
        _mdctKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int>(MdctKernel);
        _floorFitKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, int, float>(FloorFitKernel);
        _floorRenderKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, ArrayView<float>, ArrayView<float>,
            ArrayView<int>, ArrayView<byte>, int, int, int>(FloorRenderKernel);
        _divQuantKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, ArrayView<int>, int, float, int>(DivQuantKernel);
        _emitKernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<long>, ArrayView<int>,
            ArrayView<uint>, ArrayView<int>, ArrayView<uint>, ArrayView<int>,
            EmitPacketParams>(EmitKernel);
    }

    /// <summary>Identification header (delegates to CPU encoder).</summary>
    public VorbisIdentificationHeader Identification => _cpuRef.Identification;

    /// <summary>Setup header (delegates to CPU encoder).</summary>
    public VorbisSetupHeader Setup => _cpuRef.Setup;

    /// <summary>
    /// Encode one audio packet from a single block of mono PCM. Returns
    /// the bitstream bytes for that packet (the same bytes
    /// VorbisAudioEncoder.EncodeAudioPacket would produce).
    /// </summary>
    public async Task<byte[]> EncodeAudioPacketAsync(ReadOnlyMemory<float> block)
    {
        int n = _opts.BlockSize;
        int half = n / 2;
        if (block.Length != n)
            throw new ArgumentException($"block must be {n} samples, got {block.Length}.");

        var floorCfg = (VorbisFloor1Config)_cpuRef.Setup.Floors[0];
        const float headroom = 1.25f;
        const float residueRange = 2.0f;
        const int residueBookEntries = 1024;

        long worstCaseBytes = (long)half * 4 + 256;

        using var dInput = _accelerator.Allocate1D<float>(n);
        using var dWindowed = _accelerator.Allocate1D<float>(n);
        using var dSpectrum = _accelerator.Allocate1D<float>(half);
        using var dPosteriors = _accelerator.Allocate1D<int>(2);
        using var dFloorCurve = _accelerator.Allocate1D<float>(half);
        using var dResidueQ = _accelerator.Allocate1D<int>(half);
        using var dScratchInt = _accelerator.Allocate1D<int>(2 * 2);  // 2 * values
        using var dScratchByte = _accelerator.Allocate1D<byte>(2);
        using var dOutBytes = _accelerator.Allocate1D<byte>(worstCaseBytes);
        using var dOutLen = _accelerator.Allocate1D<long>(1);

        dInput.View.CopyFromCPU(block.ToArray());
        dWindowed.View.CopyFromCPU(new float[n]);
        dSpectrum.View.CopyFromCPU(new float[half]);
        dPosteriors.View.CopyFromCPU(new int[2]);
        dFloorCurve.View.CopyFromCPU(new float[half]);
        dResidueQ.View.CopyFromCPU(new int[half]);
        dScratchInt.View.CopyFromCPU(new int[4]);
        dScratchByte.View.CopyFromCPU(new byte[2]);
        dOutBytes.View.CopyFromCPU(new byte[worstCaseBytes]);
        dOutLen.View.CopyFromCPU(new long[1]);

        // 1. Window apply (parallel over n samples).
        _windowKernel(new Index1D(n), dInput.View, dWindowed.View, n);

        // 2. Forward MDCT + 4/N scale (parallel over half bins).
        _mdctKernel(new Index1D(half), dWindowed.View, dSpectrum.View, half);

        // 3. Floor fit (single thread).
        _floorFitKernel(new Index1D(1), dSpectrum.View, _dInverseDb.View, dPosteriors.View, half, headroom);

        // 4. Floor curve render (single thread; 2-endpoint floor).
        _floorRenderKernel(new Index1D(1),
            _dXList.View, dPosteriors.View,
            dFloorCurve.View, _dInverseDb.View,
            dScratchInt.View, dScratchByte.View,
            /*values*/ 2, floorCfg.Multiplier, half);

        // 5. Divide + quantize (parallel over half bins).
        _divQuantKernel(new Index1D(half),
            dSpectrum.View, dFloorCurve.View, dResidueQ.View,
            half, residueRange, residueBookEntries);

        // 6. Bitstream emit (single thread, reads dPosteriors back to host
        //    and passes as scalars).
        await _accelerator.SynchronizeAsync();
        var posteriors = await dPosteriors.CopyToHostAsync();

        var emitParams = new EmitPacketParams
        {
            Count = half,
            PosteriorY0 = posteriors[0],
            PosteriorY1 = posteriors[1],
            EndpointBits = _endpointBits,
            ModeBits = _modeBits,
        };
        _emitKernel(new Index1D(1),
            dOutBytes.View, dOutLen.View, dResidueQ.View,
            _dClassbookCodes.View, _dClassbookLengths.View,
            _dResidueBookCodes.View, _dResidueBookLengths.View,
            emitParams);
        await _accelerator.SynchronizeAsync();

        long outLen = (await dOutLen.CopyToHostAsync())[0];
        var fullBytes = await dOutBytes.CopyToHostAsync();
        var result = new byte[outLen];
        Array.Copy(fullBytes, result, outLen);
        return result;
    }

    /// <summary>Release every resource owned by this encoder.</summary>
    public void Dispose()
    {
        _dInverseDb.Dispose();
        _dClassbookCodes.Dispose();
        _dClassbookLengths.Dispose();
        _dResidueBookCodes.Dispose();
        _dResidueBookLengths.Dispose();
        _dXList.Dispose();
    }

    // ===========================================================================
    // Kernel entry points (thin wrappers around the existing primitives)
    // ===========================================================================

    private static void WindowKernel(
        Index1D idx, ArrayView<float> input, ArrayView<float> output, int n)
    {
        if (idx >= n) return;
        VorbisWindowGpu.ApplyWindowAt(input, 0, output, 0, idx, n);
    }

    private static void MdctKernel(
        Index1D idx, ArrayView<float> input, ArrayView<float> output, int n)
    {
        if (idx >= n) return;
        VorbisFwdMdctScaledGpu.ForwardScaledAt(input, 0, output, 0, n, idx);
    }

    private static void FloorFitKernel(
        Index1D _, ArrayView<float> spectrum, ArrayView<float> inverseDb,
        ArrayView<int> posteriors, int halfBlock, float headroom)
    {
        VorbisEncoderFloorFitGpu.FitFloorEndpoints(
            spectrum, 0, halfBlock, headroom,
            inverseDb, 0, posteriors, 0);
    }

    private static void FloorRenderKernel(
        Index1D _,
        ArrayView<int> xList, ArrayView<int> decodedY,
        ArrayView<float> curveOut, ArrayView<float> inverseDb,
        ArrayView<int> scratchInt, ArrayView<byte> scratchByte,
        int values, int multiplier, int halfBlock)
    {
        VorbisFloor1RenderCurveGpu.Render(
            xList, 0, values,
            decodedY, 0,
            multiplier, halfBlock,
            curveOut, 0,
            inverseDb, 0,
            scratchInt, 0,
            scratchByte, 0);
    }

    private static void DivQuantKernel(
        Index1D idx,
        ArrayView<float> spectrum, ArrayView<float> floor, ArrayView<int> output,
        int count, float residueRange, int bookEntries)
    {
        if (idx >= count) return;
        VorbisEncoderHelpersGpu.DivideQuantizeAt(
            spectrum, 0, floor, 0, output, 0,
            idx, residueRange, bookEntries);
    }

    private static void EmitKernel(
        Index1D _,
        ArrayView<byte> outBuf, ArrayView<long> outLen, ArrayView<int> residueQ,
        ArrayView<uint> classCodes, ArrayView<int> classLens,
        ArrayView<uint> resCodes, ArrayView<int> resLens,
        EmitPacketParams p)
    {
        VorbisEncoderBitstreamEmitGpu.EmitPacket(
            outBuf, outLen, residueQ, 0, p.Count,
            p.PosteriorY0, p.PosteriorY1,
            p.EndpointBits, p.ModeBits,
            classCodes, 0, classLens, 0,
            resCodes, 0, resLens, 0);
    }

    /// <summary>Packed scalar args for the bitstream emit kernel.</summary>
    public struct EmitPacketParams
    {
        /// <summary>Residue entry count (= halfBlock).</summary>
        public int Count;
        /// <summary>Floor endpoint Y[0] (low band).</summary>
        public int PosteriorY0;
        /// <summary>Floor endpoint Y[1] (high band).</summary>
        public int PosteriorY1;
        /// <summary>Bits per floor endpoint (8/7/7/6).</summary>
        public int EndpointBits;
        /// <summary>ilog(modes - 1); 0 for single-mode.</summary>
        public int ModeBits;
    }
}
