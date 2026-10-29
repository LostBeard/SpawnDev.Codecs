// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// SILK decode of one Opus frame (libopus silk_Decode with FLAG_DECODE_NORMAL) on the accelerator: the VAD / LBRR
// header and the LBRR skip, then per SILK frame the stereo predictors, each channel's indices -> pulses ->
// parameters -> decode_core -> state update, then MS->LR (or the mono buffering) and the resampler to the API rate,
// writing float PCM. Statement for statement with the CPU OpusDecoder.DecodeSilkFrame in SpawnDev.Codecs.References.
//
// Host as coordinator: the host reads only the TOC-derived geometry (internal rate, SILK frame length / count,
// stream channels) and the stream-level transitions that follow from it (rate / channel changes), and dispatches.
// Everything decoded from the bitstream (flags, predictors, the mid-only decision, indices, parameters) and every
// piece of inter-frame state stays in device buffers. Every kernel is one thread over one SILK frame (at most 20 ms
// at 16 kHz), so its serial work is bounded by the format.
//
// Upstream Copyright (c) 2006-2011 Skype Limited. BSD 3-Clause. See NOTICE.md.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Audio.Opus.Silk;

/// <summary>Per-dispatch geometry of a SILK frame decode (host metadata from the TOC).</summary>
public struct SilkFrameGeometry
{
    /// <summary>Packet bytes of this Opus frame in the packet view.</summary>
    public int PacketStart;
    /// <summary>Bytes of this Opus frame.</summary>
    public int PacketLength;
    /// <summary>Internal rate in kHz (8, 12, 16).</summary>
    public int FsKHz;
    /// <summary>Subframes per SILK frame (2 or 4).</summary>
    public int NbSubfr;
    /// <summary>Samples per SILK frame at the internal rate.</summary>
    public int FrameLength;
    /// <summary>Samples per subframe.</summary>
    public int SubfrLength;
    /// <summary>LTP memory length.</summary>
    public int LtpMemLength;
    /// <summary>LPC order (10 or 16).</summary>
    public int LpcOrder;
    /// <summary>NLSF codebook vectors.</summary>
    public int NVectors;
    /// <summary>NLSF quantizer step.</summary>
    public int QuantStepSizeQ16;
    /// <summary>Pitch-contour codebook size.</summary>
    public int ContourCbSize;
    /// <summary>SILK frames in this Opus frame (1..3).</summary>
    public int FrameCount;
    /// <summary>Coded channels (1 or 2).</summary>
    public int StreamChannels;
    /// <summary>SILK frame index within the Opus frame.</summary>
    public int Frame;
    /// <summary>Channel (0 mid, 1 side).</summary>
    public int Channel;
    /// <summary>Output samples per SILK frame at the API rate.</summary>
    public int OutLength;
    /// <summary>API channels (1 or 2).</summary>
    public int ApiChannels;
    /// <summary>1 on the first SILK frame after a stereo stream turned mono (the right resampler runs once more).</summary>
    public int StereoToMonoFirst;
    /// <summary>Flags for the transition kernel (<c>Transition*</c> constants).</summary>
    public int TransitionFlags;
    /// <summary>1 when this SILK frame is lost (concealed), 0 when decoded.</summary>
    public int Lost;
}

/// <summary>Where one channel's parameter state lives (offsets into the shared parameter buffers).</summary>
public struct SilkChannelParamsState
{
    /// <summary>The parameter decoder's state for this channel.</summary>
    public SilkParametersState Params;
}

/// <summary>
/// GPU SILK decoder for Opus frames (libopus <c>silk_Decode</c>, FLAG_DECODE_NORMAL; no PLC / CNG). One instance per
/// Opus stream: it keeps the mid and side channel states, the stereo state and the two output resamplers on the
/// accelerator across calls.
/// </summary>
public sealed class SilkPacketDecoderGpu : IDisposable
{
    // ---- work ints (flags, decoded-per-packet values, per-channel scalar state) ----
    internal const int WVad = 0;                 // [channel * 3 + frame]
    internal const int WLbrrFlags = 6;           // [channel * 3 + frame]
    internal const int WLbrrFlag = 12;           // [channel]
    internal const int WPredQ13 = 14;            // this frame's predictors
    internal const int WPredPrevQ13 = 16;        // stereo state: previous predictors
    internal const int WDecodeOnlyMiddle = 18;
    internal const int WPrevDecodeOnlyMiddle = 19;
    internal const int WChan = 20;               // + channel * WChanStride
    internal const int WChanStride = 8;
    internal const int CPrevLagIndex = 0, CPrevVoiced = 1, CFirstFrameAfterReset = 2, CPrevSignalType = 3,
        CLagPrev = 4, CLossCnt = 5, CConditional = 6, CSkipped = 7;
    internal const int WIndices = 40;            // + channel * SilkDecodedIndicesLayout.TotalSlots
    internal const int WLbrrIndices = WIndices + 2 * SilkDecodedIndicesLayout.TotalSlots;
    internal const int WLbrrPrev = WLbrrIndices + SilkDecodedIndicesLayout.TotalSlots; // prevLag[2], prevVoiced[2]
    internal const int WorkInts = WLbrrPrev + 4;

    // ---- mix shorts: mid / side with the 2-sample prefix, stereo history, resampled output ----
    internal const int MixChannelStride = SilkConstants.MAX_FRAME_LENGTH + 2;
    internal const int MixSMid = 2 * MixChannelStride;
    internal const int MixSSide = MixSMid + 2;
    internal const int MixOutL = MixSSide + 2;
    internal const int MaxOutLength = 960; // 20 ms at 48 kHz
    internal const int MixOutR = MixOutL + MaxOutLength;
    internal const int MixShorts = MixOutR + MaxOutLength;

    /// <summary>Transition flag: full reset of the mid channel (silk_init_decoder).</summary>
    public const int TransitionResetMid = 1;
    /// <summary>Transition flag: full reset of the side channel.</summary>
    public const int TransitionResetSide = 2;
    /// <summary>Transition flag: prediction-memory reset of the mid channel (internal rate changed).</summary>
    public const int TransitionPredMemMid = 4;
    /// <summary>Transition flag: prediction-memory reset of the side channel.</summary>
    public const int TransitionPredMemSide = 8;
    /// <summary>Transition flag: zero both resamplers (re-initialised for a new internal rate).</summary>
    public const int TransitionResetResamplers = 16;
    /// <summary>Transition flag: stereo starts (predictors / side history zeroed, right resampler = left).</summary>
    public const int TransitionStereoStart = 32;

    // libopus BWE_AFTER_LOSS_Q16 (silk/define.h): 0.97 in Q16.
    private const int BweAfterLossQ16 = 63570;

    // libopus silk_LBRR_flags_2_iCDF / silk_LBRR_flags_3_iCDF.
    private static readonly byte[] LbrrFlags2Icdf = { 203, 150, 0 };
    private static readonly byte[] LbrrFlags3Icdf = { 215, 195, 166, 125, 110, 82, 0 };

    private readonly Accelerator _acc;
    private readonly int _apiRateHz;
    private readonly int _apiChannels;

    // Tables + state buffers.
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _indicesBytes;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _ltpGainOffsets;
    private readonly MemoryBuffer1D<short, Stride1D.Dense> _ecIxScratch;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _pulsesBytes;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _pulsesInts;
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _paramsBytes;
    private readonly MemoryBuffer1D<short, Stride1D.Dense> _paramsShorts;
    private readonly MemoryBuffer1D<sbyte, Stride1D.Dense> _paramsSBytes;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _paramsInts;
    private readonly MemoryBuffer1D<short, Stride1D.Dense> _coreShorts;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _coreInts;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _paramsIntOut;
    private readonly MemoryBuffer1D<short, Stride1D.Dense> _paramsShortOut;
    private readonly MemoryBuffer1D<short, Stride1D.Dense> _pulses;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _work;
    private readonly MemoryBuffer1D<short, Stride1D.Dense> _mix;
    private readonly MemoryBuffer1D<OpusRangeDecoderGpuState, Stride1D.Dense> _ec;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _resInts;
    private readonly MemoryBuffer1D<short, Stride1D.Dense> _resShorts;
    private readonly MemoryBuffer1D<short, Stride1D.Dense> _shortTables; // resampler tables + stereo quant table

    // Per (fsKHz, nbSubfr) table sets.
    private readonly Dictionary<(int, int), (SilkIndicesInputs Indices, SilkParametersInputs Params, int ContourCbSize)> _variants = new();
    private readonly SilkPulsesInputs _pulsesInputs;
    private readonly SilkChannelParamsState[] _chanParams = new SilkChannelParamsState[2];
    private readonly SilkDecodeCoreInputs[] _chanCore = new SilkDecodeCoreInputs[2];
    private readonly SilkPlcChannel[] _chanPlc = new SilkPlcChannel[2];
    private readonly int _lbrr2At, _lbrr3At, _jointAt, _uniform3At, _uniform5At, _midOnlyAt, _stereoQuantAt;
    private readonly int[] _resTableOffsets = new int[7];

    // Host mirrors of the TOC-derived configuration (libopus silk_decoder_set_fs bookkeeping).
    private int _internalKHz;          // 0 until the first frame
    private int _prevStreamChannels;
    private int _midFsKHz, _sideFsKHz; // 0 = not yet initialised
    private SilkResamplerGpuConfig _resL, _resR;

    private readonly Action<Index1D, ArrayView<byte>, ArrayView<OpusRangeDecoderGpuState>, ArrayView<int>, SilkIndicesInputs, SilkPulsesInputs, ArrayView<short>, SilkFrameGeometry, SilkLbrrTableOffsets> _headerKernel;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<OpusRangeDecoderGpuState>, ArrayView<int>, ArrayView<byte>, ArrayView<short>, SilkResetTargets, SilkFrameGeometry, SilkStereoTableOffsets> _stereoKernel;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<OpusRangeDecoderGpuState>, ArrayView<int>, SilkIndicesInputs, SilkFrameGeometry> _indicesKernel;
    private readonly Action<Index1D, ArrayView<byte>, ArrayView<OpusRangeDecoderGpuState>, ArrayView<int>, SilkPulsesInputs, ArrayView<short>, SilkFrameGeometry> _pulsesKernel;
    private readonly Action<Index1D, ArrayView<int>, SilkParametersInputs, SilkParametersState, ArrayView<int>, ArrayView<short>, SilkFrameGeometry> _paramsKernel;
    private readonly Action<Index1D, SilkDecodeCoreInputs, ArrayView<int>, SilkFrameGeometry> _coreKernel;
    private readonly Action<Index1D, SilkFrameViews, ArrayView<int>, ArrayView<short>, SilkPlcChannel, SilkFrameGeometry> _finalizeKernel;
    private readonly Action<Index1D, SilkFrameViews, ArrayView<int>, ArrayView<short>, SilkPlcChannel, SilkFrameGeometry> _concealKernel;
    private readonly Action<Index1D, SilkResetTargets, ArrayView<int>, SilkFrameGeometry> _lostPrepKernel;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<int>, ArrayView<int>, ArrayView<short>, ArrayView<short>, ArrayView<float>, SilkFrameGeometry, SilkOutputParams> _outputKernel;
    private readonly Action<Index1D, SilkResetTargets, ArrayView<short>, ArrayView<int>, ArrayView<short>, ArrayView<int>, SilkFrameGeometry, SilkOutputParams> _transitionKernel;

    /// <summary>A SILK decoder producing <paramref name="apiChannels"/> channels at <paramref name="apiRateHz"/>.</summary>
    public SilkPacketDecoderGpu(Accelerator accelerator, int apiRateHz, int apiChannels)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        if (apiRateHz != 8000 && apiRateHz != 12000 && apiRateHz != 16000 && apiRateHz != 24000 && apiRateHz != 48000)
            throw new ArgumentOutOfRangeException(nameof(apiRateHz));
        if (apiChannels != 1 && apiChannels != 2) throw new ArgumentOutOfRangeException(nameof(apiChannels));
        _acc = accelerator;
        _apiRateHz = apiRateHz;
        _apiChannels = apiChannels;

        // ---- indices tables (byte), all variants ----
        var ib = new PackedDeviceBuffer<byte>();
        int typeVad = ib.Add(SilkIcdfTables.TypeOffsetVad), typeNoVad = ib.Add(SilkIcdfTables.TypeOffsetNoVad);
        int uni4 = ib.Add(SilkIcdfTables.Uniform4), gain = ib.Add(SilkIcdfTables.Gain), dgain = ib.Add(SilkIcdfTables.DeltaGain);
        int uni8 = ib.Add(SilkIcdfTables.Uniform8), nlsfExt = ib.Add(SilkIcdfTables.NlsfExt);
        int interp = ib.Add(SilkIcdfTables.NlsfInterpolationFactor), pdelta = ib.Add(SilkIcdfTables.PitchDelta);
        int plag = ib.Add(SilkIcdfTables.PitchLag), ltpPer = ib.Add(SilkIcdfTables.LtpPerIndex), ltpScale = ib.Add(SilkIcdfTables.LtpScale);
        var ltpFlat = SilkIcdfTables.LtpGain0.Concat(SilkIcdfTables.LtpGain1).Concat(SilkIcdfTables.LtpGain2).ToArray();
        int ltpGainFlat = ib.Add(ltpFlat);
        int predQ8Scratch = ib.Reserve(SilkConstants.MAX_LPC_ORDER);
        _lbrr2At = ib.Add(LbrrFlags2Icdf);
        _lbrr3At = ib.Add(LbrrFlags3Icdf);
        _jointAt = ib.Add(SilkStereoTables.StereoPredJointIcdf);
        _uniform3At = ib.Add(SilkIcdfTables.Uniform3);
        _uniform5At = ib.Add(SilkIcdfTables.Uniform5);
        _midOnlyAt = ib.Add(SilkStereoTables.StereoOnlyCodeMidIcdf);
        var cbAt = new Dictionary<int, (int Cb1, int EcIcdf, int EcSel, int PredQ8)>();
        foreach (var (key, cb) in new[] { (10, SilkNlsfCodebookTables.NbMb), (16, SilkNlsfCodebookTables.Wb) })
            cbAt[key] = (ib.Add(cb.Cb1Icdf), ib.Add(cb.EcIcdf), ib.Add(cb.EcSel), ib.Add(cb.PredQ8));
        var lagLowAt = new Dictionary<int, int>();
        foreach (int fs in new[] { 8, 12, 16 }) lagLowAt[fs] = ib.Add(SilkIcdfTables.SelectPitchLagLowBits(fs));
        var contourAt = new Dictionary<(int, int), int>();
        foreach (int fs in new[] { 8, 12, 16 })
            foreach (int nb in new[] { 2, 4 }) contourAt[(fs, nb)] = ib.Add(SilkIcdfTables.SelectPitchContour(fs, nb));
        _indicesBytes = ib.Upload(accelerator);
        _ltpGainOffsets = accelerator.Allocate1D(new[] { 0, SilkIcdfTables.LtpGain0.Length, SilkIcdfTables.LtpGain0.Length + SilkIcdfTables.LtpGain1.Length });
        _ecIxScratch = accelerator.Allocate1D<short>(SilkConstants.MAX_LPC_ORDER);

        // ---- pulses tables ----
        var pb = new PackedDeviceBuffer<byte>();
        var pi = new PackedDeviceBuffer<int>();
        _pulsesInputs = SilkPulsesInputs.AddStandard(pb, pi);
        _pulsesBytes = pb.Upload(accelerator);
        _pulsesInts = pi.Upload(accelerator);
        _pulsesInputs.Bind(_pulsesBytes.View, _pulsesInts.View);

        // ---- parameter tables (all variants) + per-channel parameter state ----
        var qb = new PackedDeviceBuffer<byte>();
        var qs = new PackedDeviceBuffer<short>();
        var qsb = new PackedDeviceBuffer<sbyte>();
        var qi = new PackedDeviceBuffer<int>();
        var cbParamsAt = new Dictionary<int, (int Cb1NlsfQ8, int Cb1WghtQ9, int EcSel, int PredQ8, int DeltaMin)>();
        foreach (var (key, cb) in new[] { (10, SilkNlsfCodebookTables.NbMb), (16, SilkNlsfCodebookTables.Wb) })
            cbParamsAt[key] = (qb.Add(cb.Cb1NlsfQ8), qs.Add(cb.Cb1WghtQ9), qb.Add(cb.EcSel), qb.Add(cb.PredQ8), qs.Add(cb.DeltaMinQ15));
        int lsfCos = qs.Add(SilkLsfCosTab.Q12);
        var ltpGains = new[] { SilkLtpGainTables.Select(0), SilkLtpGainTables.Select(1), SilkLtpGainTables.Select(2) };
        int ltpGainsFlatAt = qsb.Add(ltpGains[0].Concat(ltpGains[1]).Concat(ltpGains[2]).ToArray());
        int ltpGainOffsetsAt = qi.Add(new[] { 0, ltpGains[0].Length, ltpGains[0].Length + ltpGains[1].Length });
        int ltpScalesAt = qs.Add(new short[] { 15565, 12288, 8192 });
        var contourCbAt = new Dictionary<(int, int), (int At, int Size)>
        {
            [(8, 4)] = (qsb.Add(SilkPitchContourTables.Stage2), 11),
            [(8, 2)] = (qsb.Add(SilkPitchContourTables.Stage210Ms), 3),
            [(12, 4)] = (qsb.Add(SilkPitchContourTables.Stage3), 34),
            [(12, 2)] = (qsb.Add(SilkPitchContourTables.Stage310Ms), 12),
        };
        contourCbAt[(16, 4)] = contourCbAt[(12, 4)];
        contourCbAt[(16, 2)] = contourCbAt[(12, 2)];
        int nlsfDecScratch = qs.Reserve(3 * SilkConstants.MAX_LPC_ORDER);
        int nlsfDecPredScratch = qb.Reserve(SilkConstants.MAX_LPC_ORDER);
        int nlsf2aScratch = qi.Reserve(66);
        int nlsfIdxScratch = qsb.Reserve(SilkConstants.MAX_LPC_ORDER + 1);
        int gainIdxScratch = qsb.Reserve(SilkConstants.MAX_NB_SUBFR);
        for (int c = 0; c < 2; c++)
        {
            _chanParams[c].Params = new SilkParametersState
            {
                PrevNlsfQ15At = qs.Reserve(SilkConstants.MAX_LPC_ORDER),
                LastGainIndexAt = qi.Reserve(1),
                NlsfDecodeScratchAt = nlsfDecScratch,
                NlsfDecodePredScratchAt = nlsfDecPredScratch,
                Nlsf2aScratchAt = nlsf2aScratch,
                NlsfIndicesScratchAt = nlsfIdxScratch,
                GainIndicesScratchAt = gainIdxScratch,
            };
        }
        _paramsBytes = qb.Upload(accelerator);
        _paramsShorts = qs.Upload(accelerator);
        _paramsSBytes = qsb.Upload(accelerator);
        _paramsInts = qi.Upload(accelerator);

        foreach (int fs in new[] { 8, 12, 16 })
        {
            int order = fs == 16 ? 16 : 10;
            var cb = order == 16 ? SilkNlsfCodebookTables.Wb : SilkNlsfCodebookTables.NbMb;
            var ca = cbAt[order];
            var pa = cbParamsAt[order];
            foreach (int nb in new[] { 2, 4 })
            {
                var indices = new SilkIndicesInputs
                {
                    Bytes = _indicesBytes.View, LtpGainOffsets = _ltpGainOffsets.View, EcIxScratch = _ecIxScratch.View,
                    TypeOffsetVadIcdfAt = typeVad, TypeOffsetNoVadIcdfAt = typeNoVad, Uniform4IcdfAt = uni4,
                    GainIcdfAt = gain, DeltaGainIcdfAt = dgain, Uniform8IcdfAt = uni8,
                    Cb1IcdfAt = ca.Cb1, EcIcdfAt = ca.EcIcdf, EcSelAt = ca.EcSel, PredQ8SourceAt = ca.PredQ8,
                    NlsfExtIcdfAt = nlsfExt, NlsfInterpolationFactorIcdfAt = interp, PitchDeltaIcdfAt = pdelta,
                    PitchLagIcdfAt = plag, LagLowBitsIcdfAt = lagLowAt[fs], ContourIcdfAt = contourAt[(fs, nb)],
                    LtpPerIndexIcdfAt = ltpPer, LtpGainIcdfFlatAt = ltpGainFlat, LtpScaleIcdfAt = ltpScale,
                    PredQ8ScratchAt = predQ8Scratch,
                };
                var (cAt, cSize) = contourCbAt[(fs, nb)];
                var prm = new SilkParametersInputs
                {
                    Bytes = _paramsBytes.View, Shorts = _paramsShorts.View, SBytes = _paramsSBytes.View, Ints = _paramsInts.View,
                    Cb1NlsfQ8At = pa.Cb1NlsfQ8, Cb1WghtQ9At = pa.Cb1WghtQ9, EcSelAt = pa.EcSel, PredQ8SourceAt = pa.PredQ8,
                    DeltaMinQ15At = pa.DeltaMin, LsfCosTabQ12At = lsfCos, ContourCbAt = cAt,
                    LtpGainTablesFlatAt = ltpGainsFlatAt, LtpGainOffsetsAt = ltpGainOffsetsAt, LtpScaleQ14TableAt = ltpScalesAt,
                };
                _variants[(fs, nb)] = (indices, prm, cSize);
            }
        }

        // ---- decode-core state per channel + shared scratch ----
        var cs = new PackedDeviceBuffer<short>();
        var ci = new PackedDeviceBuffer<int>();
        const int MaxLtp = SilkConstants.MAX_LTP_MEM_LENGTH, MaxFrame = SilkConstants.MAX_FRAME_LENGTH, MaxSubfr = SilkConstants.MAX_FRAME_LENGTH / SilkConstants.MAX_NB_SUBFR;
        int sLpcScratch = ci.Reserve(SilkConstants.MAX_LPC_ORDER + MaxSubfr);
        int sLtpQ15 = ci.Reserve(MaxLtp + MaxFrame), presQ14 = ci.Reserve(MaxSubfr), gainAdj = ci.Reserve(1);
        int sLtp = cs.Reserve(MaxLtp), xq = cs.Reserve(MaxFrame);
        _paramsIntOut = accelerator.Allocate1D<int>(SilkDecodedParametersLayout.IntTotalSlots);
        _paramsShortOut = accelerator.Allocate1D<short>(SilkDecodedParametersLayout.ShortTotalSlots);
        _pulses = accelerator.Allocate1D<short>(MaxFrame + SilkConstants.SHELL_CODEC_FRAME_LENGTH);
        var coreAt = new (int OutBuf, int SLpc, int PrevGain, int Exc, int PlcInts, int PlcShorts)[2];
        for (int c = 0; c < 2; c++)
            coreAt[c] = (cs.Reserve(MaxLtp + MaxFrame), ci.Reserve(SilkConstants.MAX_LPC_ORDER), ci.Reserve(1), ci.Reserve(MaxFrame),
                ci.Reserve(SilkPlcGpu.IntsPerChannel), cs.Reserve(SilkPlcGpu.ShortsPerChannel));
        int plcScratchShorts = cs.Reserve(SilkPlcGpu.ScratchShorts), plcScratchInts = ci.Reserve(SilkPlcGpu.ScratchInts);
        _coreShorts = cs.Upload(accelerator);
        _coreInts = ci.Upload(accelerator);
        for (int c = 0; c < 2; c++)
        {
            _chanCore[c] = new SilkDecodeCoreInputs
            {
                ParamsShort = _paramsShortOut.View, ParamsInt = _paramsIntOut.View, Pulses = _pulses.View,
                Shorts = _coreShorts.View, Ints = _coreInts.View,
                PredCoefQ12At = SilkDecodedParametersLayout.ShortPredCoefQ12Half1Offset,
                GainsQ16At = SilkDecodedParametersLayout.IntGainsQ16Offset,
                PitchLAt = SilkDecodedParametersLayout.IntPitchLOffset,
                LtpCoefQ14At = SilkDecodedParametersLayout.ShortLtpCoefQ14Offset,
                OutBufInOutAt = coreAt[c].OutBuf, SLpcQ14BufInOutAt = coreAt[c].SLpc, PrevGainQ16InOutAt = coreAt[c].PrevGain,
                ExcQ14OutAt = coreAt[c].Exc, SLpcScratchAt = sLpcScratch, SLtpQ15ScratchAt = sLtpQ15, SLtpScratchAt = sLtp,
                PresQ14ScratchAt = presQ14, GainAdjScratchAt = gainAdj, XqOutAt = xq,
            };
        }

        for (int c = 0; c < 2; c++)
        {
            _chanPlc[c] = new SilkPlcChannel
            {
                IntsAt = coreAt[c].PlcInts, ShortsAt = coreAt[c].PlcShorts, OutBufAt = coreAt[c].OutBuf, SLpcAt = coreAt[c].SLpc,
                ExcAt = coreAt[c].Exc, PrevNlsfAt = _chanParams[c].Params.PrevNlsfQ15At, LsfCosAt = lsfCos,
                ScratchShortsAt = plcScratchShorts, ScratchIntsAt = plcScratchInts,
            };
        }

        // ---- short tables: resampler + stereo quantizer ----
        var st = new PackedDeviceBuffer<short>();
        var resTables = SilkResamplerGpu.Tables;
        for (int k = 0; k < resTables.Length; k++) _resTableOffsets[k] = st.Add(resTables[k]);
        _stereoQuantAt = st.Add(SilkStereoTables.StereoPredQuantQ13);
        _shortTables = st.Upload(accelerator);

        _work = accelerator.Allocate1D<int>(WorkInts);
        _mix = accelerator.Allocate1D<short>(MixShorts);
        _ec = accelerator.Allocate1D<OpusRangeDecoderGpuState>(1);
        _resInts = accelerator.Allocate1D<int>(2 * SilkResamplerGpu.IntsPerResampler);
        _resShorts = accelerator.Allocate1D<short>(2 * SilkResamplerGpu.ShortsPerResampler);
        _work.MemSetToZero();
        _mix.MemSetToZero();
        _resInts.MemSetToZero();
        _resShorts.MemSetToZero();

        _headerKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<OpusRangeDecoderGpuState>, ArrayView<int>, SilkIndicesInputs, SilkPulsesInputs, ArrayView<short>, SilkFrameGeometry, SilkLbrrTableOffsets>(HeaderKernel);
        _stereoKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<OpusRangeDecoderGpuState>, ArrayView<int>, ArrayView<byte>, ArrayView<short>, SilkResetTargets, SilkFrameGeometry, SilkStereoTableOffsets>(StereoKernel);
        _indicesKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<OpusRangeDecoderGpuState>, ArrayView<int>, SilkIndicesInputs, SilkFrameGeometry>(IndicesKernel);
        _pulsesKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<OpusRangeDecoderGpuState>, ArrayView<int>, SilkPulsesInputs, ArrayView<short>, SilkFrameGeometry>(PulsesKernel);
        _paramsKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, SilkParametersInputs, SilkParametersState, ArrayView<int>, ArrayView<short>, SilkFrameGeometry>(ParamsKernel);
        _coreKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, SilkDecodeCoreInputs, ArrayView<int>, SilkFrameGeometry>(CoreKernel);
        _finalizeKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, SilkFrameViews, ArrayView<int>, ArrayView<short>, SilkPlcChannel, SilkFrameGeometry>(FinalizeKernel);
        _concealKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, SilkFrameViews, ArrayView<int>, ArrayView<short>, SilkPlcChannel, SilkFrameGeometry>(ConcealKernel);
        _lostPrepKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, SilkResetTargets, ArrayView<int>, SilkFrameGeometry>(LostPrepKernel);
        _outputKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<short>, ArrayView<int>, ArrayView<int>, ArrayView<short>, ArrayView<short>, ArrayView<float>, SilkFrameGeometry, SilkOutputParams>(OutputKernel);
        _transitionKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, SilkResetTargets, ArrayView<short>, ArrayView<int>, ArrayView<short>, ArrayView<int>, SilkFrameGeometry, SilkOutputParams>(TransitionKernel);
    }

    private SilkResetTargets ResetTargets => new SilkResetTargets
    {
        CoreShorts = _coreShorts.View, CoreInts = _coreInts.View, ParamsShorts = _paramsShorts.View, ParamsInts = _paramsInts.View,
        MidOutBufAt = _chanCore[0].OutBufInOutAt, MidSLpcAt = _chanCore[0].SLpcQ14BufInOutAt, MidPrevGainAt = _chanCore[0].PrevGainQ16InOutAt,
        SideOutBufAt = _chanCore[1].OutBufInOutAt, SideSLpcAt = _chanCore[1].SLpcQ14BufInOutAt, SidePrevGainAt = _chanCore[1].PrevGainQ16InOutAt,
        MidPrevNlsfAt = _chanParams[0].Params.PrevNlsfQ15At, MidLastGainAt = _chanParams[0].Params.LastGainIndexAt,
        SidePrevNlsfAt = _chanParams[1].Params.PrevNlsfQ15At, SideLastGainAt = _chanParams[1].Params.LastGainIndexAt,
        MidPlc = _chanPlc[0], SidePlc = _chanPlc[1],
    };

    /// <summary>
    /// Queue the SILK decode of one Opus frame (<paramref name="packetLength"/> bytes at <paramref name="packetStart"/>
    /// of <paramref name="packet"/>; TOC already removed) whose TOC gives <paramref name="internalKHz"/>, the SILK frame
    /// length <paramref name="frameLengthMs"/> (10 or 20), <paramref name="frameCount"/> SILK frames and
    /// <paramref name="streamChannels"/>. Writes <c>frameCount * frameLengthMs * apiRate / 1000</c> samples per API channel,
    /// interleaved, as float in [-1, 1) at <paramref name="pcmOutBase"/>.
    /// </summary>
    public void Decode(ArrayView<byte> packet, int packetStart, int packetLength, int internalKHz, int frameLengthMs,
        int frameCount, int streamChannels, ArrayView<float> pcmOut, long pcmOutBase)
    {
        if (internalKHz != 8 && internalKHz != 12 && internalKHz != 16) throw new ArgumentOutOfRangeException(nameof(internalKHz));
        if (frameLengthMs != 10 && frameLengthMs != 20) throw new ArgumentOutOfRangeException(nameof(frameLengthMs));
        if (frameCount < 1 || frameCount > 3) throw new ArgumentOutOfRangeException(nameof(frameCount));
        if (streamChannels != 1 && streamChannels != 2) throw new ArgumentOutOfRangeException(nameof(streamChannels));
        int nbSubfr = frameLengthMs == 20 ? SilkConstants.MAX_NB_SUBFR : SilkConstants.MAX_NB_SUBFR / 2;
        var (indicesInputs, paramsInputs, contourCbSize) = _variants[(internalKHz, nbSubfr)];
        int order = internalKHz == 16 ? 16 : 10;
        var codebook = order == 16 ? SilkNlsfCodebookTables.Wb : SilkNlsfCodebookTables.NbMb;

        // ---- stream transitions (silk_Decode before the frame header; host bookkeeping, state work on the GPU) ----
        bool monoToStereo = streamChannels > _prevStreamChannels;
        bool stereoToMono = streamChannels == 1 && _prevStreamChannels == 2 && internalKHz == _internalKHz;
        int flags = 0;
        if (_midFsKHz == 0) { flags |= TransitionResetMid; _midFsKHz = internalKHz; }
        if (monoToStereo || _sideFsKHz == 0) { flags |= TransitionResetSide; _sideFsKHz = internalKHz; }
        if (_midFsKHz != internalKHz) { flags |= TransitionPredMemMid; _midFsKHz = internalKHz; }
        if (streamChannels == 2 && _sideFsKHz != internalKHz) { flags |= TransitionPredMemSide; _sideFsKHz = internalKHz; }
        if (_internalKHz != internalKHz)
        {
            flags |= TransitionResetResamplers;
            _resL = SilkResamplerGpu.InitDecoder(internalKHz * 1000, _apiRateHz, _resTableOffsets, 0, 0);
            _resR = SilkResamplerGpu.InitDecoder(internalKHz * 1000, _apiRateHz, _resTableOffsets,
                SilkResamplerGpu.IntsPerResampler, SilkResamplerGpu.ShortsPerResampler);
        }
        if (_apiChannels == 2 && streamChannels == 2 && _prevStreamChannels == 1) flags |= TransitionStereoStart;
        _internalKHz = internalKHz;
        _prevStreamChannels = streamChannels;

        int frameLength = internalKHz * frameLengthMs;
        var g = new SilkFrameGeometry
        {
            PacketStart = packetStart, PacketLength = packetLength,
            FsKHz = internalKHz, NbSubfr = nbSubfr, FrameLength = frameLength,
            SubfrLength = SilkConstants.SUB_FRAME_LENGTH_MS * internalKHz,
            LtpMemLength = SilkConstants.LTP_MEM_LENGTH_MS * internalKHz,
            LpcOrder = order, NVectors = codebook.NVectors, QuantStepSizeQ16 = codebook.QuantStepSizeQ16,
            ContourCbSize = contourCbSize, FrameCount = frameCount, StreamChannels = streamChannels,
            OutLength = _apiRateHz / 1000 * frameLengthMs, ApiChannels = _apiChannels, TransitionFlags = flags,
        };
        var outParams = new SilkOutputParams
        {
            Left = _resL, Right = _resR,
            PcmOutBase = pcmOutBase,
        };
        if (flags != 0)
            _transitionKernel(1, ResetTargets, _mix.View, _work.View, _resShorts.View, _resInts.View, g, outParams);

        var lbrr = new SilkLbrrTableOffsets { Lbrr2At = _lbrr2At, Lbrr3At = _lbrr3At, JointAt = _jointAt, Uniform3At = _uniform3At, Uniform5At = _uniform5At, MidOnlyAt = _midOnlyAt };
        _headerKernel(1, packet, _ec.View, _work.View, indicesInputs, _pulsesInputs, _pulses.View, g, lbrr);

        var stereoOffsets = new SilkStereoTableOffsets { JointAt = _jointAt, Uniform3At = _uniform3At, Uniform5At = _uniform5At, MidOnlyAt = _midOnlyAt, QuantAt = _stereoQuantAt };
        for (int f = 0; f < frameCount; f++)
        {
            g.Frame = f;
            g.StereoToMonoFirst = stereoToMono && f == 0 ? 1 : 0;
            if (streamChannels == 2)
                _stereoKernel(1, packet, _ec.View, _work.View, _indicesBytes.View, _shortTables.View, ResetTargets, g, stereoOffsets);
            for (int c = 0; c < streamChannels; c++)
            {
                g.Channel = c;
                _indicesKernel(1, packet, _ec.View, _work.View, indicesInputs, g);
                _pulsesKernel(1, packet, _ec.View, _work.View, _pulsesInputs, _pulses.View, g);
                _paramsKernel(1, _work.View, paramsInputs, _chanParams[c].Params, _paramsIntOut.View, _paramsShortOut.View, g);
                _coreKernel(1, _chanCore[c], _work.View, g);
                _finalizeKernel(1, FrameViews, _work.View, _mix.View, _chanPlc[c], g);
            }
            _outputKernel(1, _mix.View, _work.View, _resInts.View, _resShorts.View, _shortTables.View, pcmOut, g, outParams);
        }
    }

    private SilkFrameViews FrameViews => new SilkFrameViews
    {
        CoreShorts = _coreShorts.View, CoreInts = _coreInts.View, ParamsShorts = _paramsShorts.View,
        ParamsShortOut = _paramsShortOut.View, ParamsIntOut = _paramsIntOut.View, XqAt = _chanCore[0].XqOutAt,
    };

    /// <summary>
    /// Queue the concealment of one LOST SILK frame of <paramref name="frameLengthMs"/> (10 or 20) ms (libopus silk_Decode
    /// with FLAG_PACKET_LOST): the previous frame's internal rate and channels, silk_PLC + silk_CNG per channel, then the
    /// same MS->LR / buffering and resampling as a decoded frame. Writes <c>frameLengthMs * apiRate / 1000</c> samples per
    /// API channel at <paramref name="pcmOutBase"/>. Requires a decoded frame before (the caller returns silence otherwise).
    /// </summary>
    public void DecodeLost(int frameLengthMs, ArrayView<float> pcmOut, long pcmOutBase)
    {
        if (frameLengthMs != 10 && frameLengthMs != 20) throw new ArgumentOutOfRangeException(nameof(frameLengthMs));
        if (_internalKHz == 0) throw new InvalidOperationException("Nothing to conceal before the first decoded SILK frame.");
        int internalKHz = _internalKHz, streamChannels = _prevStreamChannels;
        int nbSubfr = frameLengthMs == 20 ? SilkConstants.MAX_NB_SUBFR : SilkConstants.MAX_NB_SUBFR / 2;
        int order = internalKHz == 16 ? 16 : 10;
        var g = new SilkFrameGeometry
        {
            FsKHz = internalKHz, NbSubfr = nbSubfr, FrameLength = internalKHz * frameLengthMs,
            SubfrLength = SilkConstants.SUB_FRAME_LENGTH_MS * internalKHz,
            LtpMemLength = SilkConstants.LTP_MEM_LENGTH_MS * internalKHz,
            LpcOrder = order, FrameCount = 1, StreamChannels = streamChannels,
            OutLength = _apiRateHz / 1000 * frameLengthMs, ApiChannels = _apiChannels, Lost = 1,
        };
        var outParams = new SilkOutputParams { Left = _resL, Right = _resR, PcmOutBase = pcmOutBase };
        _lostPrepKernel(1, ResetTargets, _work.View, g);
        for (int c = 0; c < streamChannels; c++)
        {
            g.Channel = c;
            _concealKernel(1, FrameViews, _work.View, _mix.View, _chanPlc[c], g);
        }
        _outputKernel(1, _mix.View, _work.View, _resInts.View, _resShorts.View, _shortTables.View, pcmOut, g, outParams);
    }

    /// <summary>The work ints (flags, per-channel scalar state); tests read the decoded flags.</summary>
    internal ArrayView<int> WorkView => _work.View;

    /// <inheritdoc/>
    public void Dispose()
    {
        foreach (var b in new MemoryBuffer[] { _indicesBytes, _ltpGainOffsets, _ecIxScratch, _pulsesBytes, _pulsesInts, _paramsBytes,
                     _paramsShorts, _paramsSBytes, _paramsInts, _coreShorts, _coreInts, _paramsIntOut, _paramsShortOut, _pulses,
                     _work, _mix, _ec, _resInts, _resShorts, _shortTables })
            b.Dispose();
    }

    // ================================ kernels ================================

    private static int Chan(int c, int field) => WChan + c * WChanStride + field;

    // VAD / LBRR flags (RFC 6716 4.2.3) and the LBRR frames read and thrown away (silk_Decode FLAG_DECODE_NORMAL).
    private static void HeaderKernel(Index1D _, ArrayView<byte> packet, ArrayView<OpusRangeDecoderGpuState> ecBuf, ArrayView<int> work,
        SilkIndicesInputs indices, SilkPulsesInputs pulsesIn, ArrayView<short> pulses, SilkFrameGeometry g, SilkLbrrTableOffsets t)
    {
        uint storage = (uint)g.PacketLength;
        var ec = OpusRangeDecoderGpu.Init(packet, g.PacketStart, storage);
        for (int k = 0; k < 12; k++) work[WVad + k] = 0;
        work[WLbrrFlag] = 0; work[WLbrrFlag + 1] = 0;
        work[WDecodeOnlyMiddle] = 0;
        for (int n = 0; n < g.StreamChannels; n++)
        {
            for (int f = 0; f < g.FrameCount; f++) work[WVad + n * 3 + f] = OpusRangeDecoderGpu.DecodeBitLogP(ref ec, packet, g.PacketStart, storage, 1);
            work[WLbrrFlag + n] = OpusRangeDecoderGpu.DecodeBitLogP(ref ec, packet, g.PacketStart, storage, 1);
        }
        for (int n = 0; n < g.StreamChannels; n++)
        {
            if (work[WLbrrFlag + n] == 0) continue;
            if (g.FrameCount == 1)
            {
                work[WLbrrFlags + n * 3] = 1;
            }
            else
            {
                int sym = OpusRangeDecoderGpu.DecodeIcdf(ref ec, packet, g.PacketStart, storage, indices.Bytes,
                    g.FrameCount == 2 ? t.Lbrr2At : t.Lbrr3At, 8) + 1;
                for (int f = 0; f < g.FrameCount; f++) work[WLbrrFlags + n * 3 + f] = (sym >> f) & 1;
            }
        }
        if (work[WLbrrFlag] != 0 || work[WLbrrFlag + 1] != 0)
        {
            for (int k = 0; k < 4; k++) work[WLbrrPrev + k] = 0;
            for (int f = 0; f < g.FrameCount; f++)
            {
                for (int n = 0; n < g.StreamChannels; n++)
                {
                    if (work[WLbrrFlags + n * 3 + f] == 0) continue;
                    if (g.StreamChannels == 2 && n == 0)
                    {
                        // Predictors (joint, then 2 x (uniform3, uniform5)) read and discarded, then the mid-only flag.
                        OpusRangeDecoderGpu.DecodeIcdf(ref ec, packet, g.PacketStart, storage, indices.Bytes, t.JointAt, 8);
                        OpusRangeDecoderGpu.DecodeIcdf(ref ec, packet, g.PacketStart, storage, indices.Bytes, t.Uniform3At, 8);
                        OpusRangeDecoderGpu.DecodeIcdf(ref ec, packet, g.PacketStart, storage, indices.Bytes, t.Uniform5At, 8);
                        OpusRangeDecoderGpu.DecodeIcdf(ref ec, packet, g.PacketStart, storage, indices.Bytes, t.Uniform3At, 8);
                        OpusRangeDecoderGpu.DecodeIcdf(ref ec, packet, g.PacketStart, storage, indices.Bytes, t.Uniform5At, 8);
                        if (work[WLbrrFlags + 3 + f] == 0)
                            OpusRangeDecoderGpu.DecodeIcdf(ref ec, packet, g.PacketStart, storage, indices.Bytes, t.MidOnlyAt, 8);
                    }
                    int conditional = f > 0 && work[WLbrrFlags + n * 3 + f - 1] != 0 ? 1 : 0;
                    SilkIndicesDecoderGpu.Decode(ref ec, packet, g.PacketStart, storage, indices,
                        g.NVectors, g.LpcOrder, g.NbSubfr, g.FsKHz, 0, 1, conditional,
                        work[WLbrrPrev + n], work[WLbrrPrev + 2 + n], 0, work, WLbrrIndices);
                    int signalType = work[WLbrrIndices + SilkDecodedIndicesLayout.SignalTypeOffset];
                    work[WLbrrPrev + n] = work[WLbrrIndices + SilkDecodedIndicesLayout.LagIndexOffset];
                    work[WLbrrPrev + 2 + n] = signalType == SilkConstants.TYPE_VOICED ? 1 : 0;
                    SilkPulsesDecoderGpu.Decode(ref ec, packet, g.PacketStart, storage, pulsesIn, signalType,
                        work[WLbrrIndices + SilkDecodedIndicesLayout.QuantOffsetTypeOffset], g.FrameLength, pulses, 0);
                }
            }
        }
        ecBuf[0] = ec;
    }

    // silk_stereo_decode_pred + silk_stereo_decode_mid_only for SILK frame g.Frame; resets the side channel's prediction
    // memory when it is coded again after mid-only frames.
    private static void StereoKernel(Index1D _, ArrayView<byte> packet, ArrayView<OpusRangeDecoderGpuState> ecBuf, ArrayView<int> work,
        ArrayView<byte> icdf, ArrayView<short> shortTables, SilkResetTargets r, SilkFrameGeometry g, SilkStereoTableOffsets t)
    {
        uint storage = (uint)g.PacketLength;
        var ec = ecBuf[0];
        int n = OpusRangeDecoderGpu.DecodeIcdf(ref ec, packet, g.PacketStart, storage, icdf, t.JointAt, 8);
        int ix0_2 = n / 5;
        int ix1_2 = n - 5 * ix0_2;
        int ix0_0 = OpusRangeDecoderGpu.DecodeIcdf(ref ec, packet, g.PacketStart, storage, icdf, t.Uniform3At, 8);
        int ix0_1 = OpusRangeDecoderGpu.DecodeIcdf(ref ec, packet, g.PacketStart, storage, icdf, t.Uniform5At, 8);
        int ix1_0 = OpusRangeDecoderGpu.DecodeIcdf(ref ec, packet, g.PacketStart, storage, icdf, t.Uniform3At, 8);
        int ix1_1 = OpusRangeDecoderGpu.DecodeIcdf(ref ec, packet, g.PacketStart, storage, icdf, t.Uniform5At, 8);
        SilkStereoDecodePredGpu.ApplyAt(work, WPredQ13, shortTables, t.QuantAt, ix0_0, ix0_1, ix0_2, ix1_0, ix1_1, ix1_2);
        int decodeOnlyMiddle = 0;
        if (work[WVad + 3 + g.Frame] == 0)
            decodeOnlyMiddle = OpusRangeDecoderGpu.DecodeIcdf(ref ec, packet, g.PacketStart, storage, icdf, t.MidOnlyAt, 8);
        work[WDecodeOnlyMiddle] = decodeOnlyMiddle;
        if (decodeOnlyMiddle == 0 && work[WPrevDecodeOnlyMiddle] != 0)
            ResetPredictionMemory(r, work, 1);
        ecBuf[0] = ec;
    }

    // SilkChannelDecoderState.ResetPredictionMemory.
    private static void ResetPredictionMemory(SilkResetTargets r, ArrayView<int> work, int c)
    {
        int outBuf = c == 0 ? r.MidOutBufAt : r.SideOutBufAt;
        int sLpc = c == 0 ? r.MidSLpcAt : r.SideSLpcAt;
        for (int k = 0; k < SilkConstants.MAX_LTP_MEM_LENGTH + SilkConstants.MAX_FRAME_LENGTH; k++) r.CoreShorts[outBuf + k] = 0;
        for (int k = 0; k < SilkConstants.MAX_LPC_ORDER; k++) r.CoreInts[sLpc + k] = 0;
        work[Chan(c, CLagPrev)] = 100;
        r.ParamsInts[c == 0 ? r.MidLastGainAt : r.SideLastGainAt] = 10;
        work[Chan(c, CPrevSignalType)] = SilkConstants.TYPE_NO_VOICE_ACTIVITY;
        work[Chan(c, CFirstFrameAfterReset)] = 1;
    }

    // SilkChannelDecoderState.Reset (silk_init_decoder).
    private static void ResetChannel(SilkResetTargets r, ArrayView<int> work, int c)
    {
        r.ParamsInts[c == 0 ? r.MidLastGainAt : r.SideLastGainAt] = 0;
        work[Chan(c, CPrevLagIndex)] = 0;
        work[Chan(c, CPrevVoiced)] = 0;
        int nlsf = c == 0 ? r.MidPrevNlsfAt : r.SidePrevNlsfAt;
        for (int k = 0; k < SilkConstants.MAX_LPC_ORDER; k++) r.ParamsShorts[nlsf + k] = 0;
        int outBuf = c == 0 ? r.MidOutBufAt : r.SideOutBufAt;
        int sLpc = c == 0 ? r.MidSLpcAt : r.SideSLpcAt;
        for (int k = 0; k < SilkConstants.MAX_LTP_MEM_LENGTH + SilkConstants.MAX_FRAME_LENGTH; k++) r.CoreShorts[outBuf + k] = 0;
        for (int k = 0; k < SilkConstants.MAX_LPC_ORDER; k++) r.CoreInts[sLpc + k] = 0;
        r.CoreInts[c == 0 ? r.MidPrevGainAt : r.SidePrevGainAt] = 65536;
        work[Chan(c, CLossCnt)] = 0;
        work[Chan(c, CPrevSignalType)] = SilkConstants.TYPE_NO_VOICE_ACTIVITY;
        work[Chan(c, CFirstFrameAfterReset)] = 1;
        work[Chan(c, CLagPrev)] = 100;
        SilkPlcGpu.Reset(c == 0 ? r.MidPlc : r.SidePlc, r.CoreShorts, r.CoreInts);
    }

    // silk_decode_indices for channel g.Channel of SILK frame g.Frame (skipped for a mid-only side).
    private static void IndicesKernel(Index1D _, ArrayView<byte> packet, ArrayView<OpusRangeDecoderGpuState> ecBuf, ArrayView<int> work,
        SilkIndicesInputs indices, SilkFrameGeometry g)
    {
        int c = g.Channel;
        int skipped = c == 1 && work[WDecodeOnlyMiddle] != 0 ? 1 : 0;
        work[Chan(c, CSkipped)] = skipped;
        if (skipped != 0) return;
        int conditional, skipLtpScale = 0;
        if (c == 0)
        {
            conditional = g.Frame > 0 ? 1 : 0;
        }
        else
        {
            // CODE_INDEPENDENTLY for the first frame, CODE_INDEPENDENTLY_NO_LTP_SCALING right after a mid-only frame,
            // CODE_CONDITIONALLY otherwise.
            bool noLtpScaling = g.Frame > 0 && work[WPrevDecodeOnlyMiddle] != 0;
            conditional = g.Frame > 0 && !noLtpScaling ? 1 : 0;
            skipLtpScale = noLtpScaling ? 1 : 0;
        }
        work[Chan(c, CConditional)] = conditional;
        var ec = ecBuf[0];
        SilkIndicesDecoderGpu.Decode(ref ec, packet, g.PacketStart, (uint)g.PacketLength, indices,
            g.NVectors, g.LpcOrder, g.NbSubfr, g.FsKHz, work[WVad + c * 3 + g.Frame], 0, conditional,
            work[Chan(c, CPrevLagIndex)], work[Chan(c, CPrevVoiced)], work[Chan(c, CFirstFrameAfterReset)],
            work, WIndices + c * SilkDecodedIndicesLayout.TotalSlots, skipLtpScale);
        ecBuf[0] = ec;
    }

    // silk_decode_pulses.
    private static void PulsesKernel(Index1D _, ArrayView<byte> packet, ArrayView<OpusRangeDecoderGpuState> ecBuf, ArrayView<int> work,
        SilkPulsesInputs pulsesIn, ArrayView<short> pulses, SilkFrameGeometry g)
    {
        int c = g.Channel;
        if (work[Chan(c, CSkipped)] != 0) return;
        int ix = WIndices + c * SilkDecodedIndicesLayout.TotalSlots;
        var ec = ecBuf[0];
        SilkPulsesDecoderGpu.Decode(ref ec, packet, g.PacketStart, (uint)g.PacketLength, pulsesIn,
            work[ix + SilkDecodedIndicesLayout.SignalTypeOffset], work[ix + SilkDecodedIndicesLayout.QuantOffsetTypeOffset],
            g.FrameLength, pulses, 0);
        ecBuf[0] = ec;
    }

    // silk_decode_parameters.
    private static void ParamsKernel(Index1D _, ArrayView<int> work, SilkParametersInputs inputs, SilkParametersState state,
        ArrayView<int> intOut, ArrayView<short> shortOut, SilkFrameGeometry g)
    {
        int c = g.Channel;
        if (work[Chan(c, CSkipped)] != 0) return;
        SilkParametersDecoderGpu.Decode(work, WIndices + c * SilkDecodedIndicesLayout.TotalSlots, inputs, g.ContourCbSize, state,
            g.QuantStepSizeQ16, g.LpcOrder, g.NbSubfr, g.FsKHz, work[Chan(c, CConditional)], intOut, 0, shortOut, 0);
        // silk_decode_parameters: after a packet loss, bandwidth-expand both LPC coefficient sets.
        if (work[Chan(c, CLossCnt)] != 0)
        {
            SilkBwexpanderGpu.Expand16(shortOut, SilkDecodedParametersLayout.ShortPredCoefQ12Half1Offset, g.LpcOrder, BweAfterLossQ16);
            SilkBwexpanderGpu.Expand16(shortOut, SilkDecodedParametersLayout.ShortPredCoefQ12Half2Offset, g.LpcOrder, BweAfterLossQ16);
        }
    }

    // silk_decode_core.
    private static void CoreKernel(Index1D _, SilkDecodeCoreInputs inputs, ArrayView<int> work, SilkFrameGeometry g)
    {
        int c = g.Channel;
        if (work[Chan(c, CSkipped)] != 0) return;
        int ix = WIndices + c * SilkDecodedIndicesLayout.TotalSlots;
        var s = new SilkDecodeCoreScalars
        {
            SignalType = work[ix + SilkDecodedIndicesLayout.SignalTypeOffset],
            QuantOffsetType = work[ix + SilkDecodedIndicesLayout.QuantOffsetTypeOffset],
            Seed = work[ix + SilkDecodedIndicesLayout.SeedOffset],
            LpcOrder = g.LpcOrder, NbSubfr = g.NbSubfr, SubfrLength = g.SubfrLength, FrameLength = g.FrameLength,
            LtpMemLength = g.LtpMemLength,
            LtpScaleQ14 = inputs.ParamsShort[SilkDecodedParametersLayout.ShortLtpScaleQ14Offset],
            NlsfInterpEnabled = work[ix + SilkDecodedIndicesLayout.NlsfInterpCoefQ2Offset] < 4 ? 1 : 0,
            LossCnt = work[Chan(c, CLossCnt)],
            PrevSignalType = work[Chan(c, CPrevSignalType)],
            LagPrev = work[Chan(c, CLagPrev)],
        };
        SilkDecodeCoreGpu.Decode(inputs, s);
    }

    // silk_decode_frame's tail for a decoded frame: outBuf slide, silk_PLC update, state, the frame into the mix buffer
    // (after the 2-sample prefix), silk_CNG, silk_PLC_glue_frames, lagPrev.
    private static void FinalizeKernel(Index1D _, SilkFrameViews v, ArrayView<int> work, ArrayView<short> mix, SilkPlcChannel plc, SilkFrameGeometry g)
    {
        int c = g.Channel;
        int mixAt = c * MixChannelStride + 2;
        if (work[Chan(c, CSkipped)] != 0)
        {
            for (int i = 0; i < g.FrameLength; i++) mix[mixAt + i] = 0;
            return;
        }
        int mvLen = g.LtpMemLength - g.FrameLength;
        for (int i = 0; i < mvLen; i++) v.CoreShorts[plc.OutBufAt + i] = v.CoreShorts[plc.OutBufAt + g.FrameLength + i];
        for (int i = 0; i < g.FrameLength; i++)
        {
            short x = v.CoreShorts[v.XqAt + i];
            v.CoreShorts[plc.OutBufAt + mvLen + i] = x;
            mix[mixAt + i] = x;
        }
        int ix = WIndices + c * SilkDecodedIndicesLayout.TotalSlots;
        int signalType = work[ix + SilkDecodedIndicesLayout.SignalTypeOffset];
        SilkPlcGpu.Update(plc, v.CoreShorts, v.CoreInts, v.ParamsShortOut, v.ParamsIntOut,
            g.FsKHz, g.FrameLength, g.NbSubfr, g.SubfrLength, g.LpcOrder, signalType);
        work[Chan(c, CLossCnt)] = 0;
        work[Chan(c, CPrevSignalType)] = signalType;
        work[Chan(c, CPrevVoiced)] = signalType == SilkConstants.TYPE_VOICED ? 1 : 0;
        work[Chan(c, CPrevLagIndex)] = work[ix + SilkDecodedIndicesLayout.LagIndexOffset];
        work[Chan(c, CFirstFrameAfterReset)] = 0;
        SilkPlcGpu.Cng(plc, v.CoreShorts, v.CoreInts, v.ParamsShorts, v.ParamsIntOut, mix, mixAt,
            g.FsKHz, g.FrameLength, g.NbSubfr, g.SubfrLength, g.LpcOrder, 0, signalType);
        SilkPlcGpu.Glue(plc, v.CoreInts, mix, mixAt, g.FrameLength, 0);
        work[Chan(c, CLagPrev)] = v.ParamsIntOut[SilkDecodedParametersLayout.IntPitchLOffset + g.NbSubfr - 1];
    }

    // silk_Decode before the frames of a LOST packet: the previous predictors, no mid-only flag (the side channel's
    // prediction memory resets after a mid-only frame), the side decodes only if the last frame coded it, LastGainIndex 10.
    private static void LostPrepKernel(Index1D _, SilkResetTargets r, ArrayView<int> work, SilkFrameGeometry g)
    {
        work[WDecodeOnlyMiddle] = 0;
        work[Chan(0, CSkipped)] = 0;
        if (g.StreamChannels == 2)
        {
            work[WPredQ13] = work[WPredPrevQ13];
            work[WPredQ13 + 1] = work[WPredPrevQ13 + 1];
            if (work[WPrevDecodeOnlyMiddle] == 1) ResetPredictionMemory(r, work, 1);
            work[Chan(1, CSkipped)] = work[WPrevDecodeOnlyMiddle] == 0 ? 0 : 1;
        }
        for (int c = 0; c < g.StreamChannels; c++) r.ParamsInts[c == 0 ? r.MidLastGainAt : r.SideLastGainAt] = 10;
    }

    // silk_decode_frame for a lost frame: silk_PLC conceal, outBuf slide, silk_CNG, silk_PLC_glue_frames, lagPrev.
    private static void ConcealKernel(Index1D _, SilkFrameViews v, ArrayView<int> work, ArrayView<short> mix, SilkPlcChannel plc, SilkFrameGeometry g)
    {
        int c = g.Channel;
        int mixAt = c * MixChannelStride + 2;
        if (work[Chan(c, CSkipped)] != 0)
        {
            for (int i = 0; i < g.FrameLength; i++) mix[mixAt + i] = 0;
            return;
        }
        int lossCnt = work[Chan(c, CLossCnt)];
        int prevSignalType = work[Chan(c, CPrevSignalType)];
        int lag = SilkPlcGpu.Conceal(plc, v.CoreShorts, v.CoreInts, mix, mixAt, g.FsKHz, g.FrameLength, g.NbSubfr, g.SubfrLength,
            g.LtpMemLength, g.LpcOrder, lossCnt, prevSignalType, work[Chan(c, CFirstFrameAfterReset)]);
        lossCnt++;
        work[Chan(c, CLossCnt)] = lossCnt;
        int mvLen = g.LtpMemLength - g.FrameLength;
        for (int i = 0; i < mvLen; i++) v.CoreShorts[plc.OutBufAt + i] = v.CoreShorts[plc.OutBufAt + g.FrameLength + i];
        for (int i = 0; i < g.FrameLength; i++) v.CoreShorts[plc.OutBufAt + mvLen + i] = mix[mixAt + i];
        SilkPlcGpu.Cng(plc, v.CoreShorts, v.CoreInts, v.ParamsShorts, v.ParamsIntOut, mix, mixAt,
            g.FsKHz, g.FrameLength, g.NbSubfr, g.SubfrLength, g.LpcOrder, lossCnt, prevSignalType);
        SilkPlcGpu.Glue(plc, v.CoreInts, mix, mixAt, g.FrameLength, lossCnt);
        work[Chan(c, CLagPrev)] = lag;
    }

    // MS->LR (stereo) or the mono buffering, the resamplers, float PCM out (silk_Decode's tail).
    private static void OutputKernel(Index1D _, ArrayView<short> mix, ArrayView<int> work, ArrayView<int> resInts, ArrayView<short> resShorts,
        ArrayView<short> tables, ArrayView<float> pcmOut, SilkFrameGeometry g, SilkOutputParams p)
    {
        int len = g.FrameLength;
        int mid = 0, side = MixChannelStride;
        if (g.ApiChannels == 2 && g.StreamChannels == 2)
        {
            MsToLr(mix, mid, side, work, g.FsKHz, len);
            SilkResamplerGpu.Apply(p.Left, resInts, resShorts, tables, mix, mid + 1, mix, MixOutL, len);
            SilkResamplerGpu.Apply(p.Right, resInts, resShorts, tables, mix, side + 1, mix, MixOutR, len);
        }
        else
        {
            mix[mid] = mix[MixSMid];
            mix[mid + 1] = mix[MixSMid + 1];
            mix[MixSMid] = mix[mid + len];
            mix[MixSMid + 1] = mix[mid + len + 1];
            SilkResamplerGpu.Apply(p.Left, resInts, resShorts, tables, mix, mid + 1, mix, MixOutL, len);
            if (g.ApiChannels == 2)
            {
                if (g.StereoToMonoFirst != 0)
                    SilkResamplerGpu.Apply(p.Right, resInts, resShorts, tables, mix, mid + 1, mix, MixOutR, len);
                else
                    for (int i = 0; i < g.OutLength; i++) mix[MixOutR + i] = mix[MixOutL + i];
            }
        }
        if (g.Lost == 0) work[WPrevDecodeOnlyMiddle] = work[WDecodeOnlyMiddle];

        long baseAt = p.PcmOutBase + (long)g.Frame * g.OutLength * g.ApiChannels;
        for (int i = 0; i < g.OutLength; i++)
        {
            if (g.ApiChannels == 2)
            {
                pcmOut[baseAt + 2 * i] = mix[MixOutL + i] / 32768.0f;
                pcmOut[baseAt + 2 * i + 1] = mix[MixOutR + i] / 32768.0f;
            }
            else
            {
                pcmOut[baseAt + i] = mix[MixOutL + i] / 32768.0f;
            }
        }
    }

    // silk_stereo_MS_to_LR (SilkStereoMsToLr.Apply).
    private static void MsToLr(ArrayView<short> mix, int x1, int x2, ArrayView<int> work, int fsKHz, int frameLength)
    {
        mix[x1] = mix[MixSMid];
        mix[x1 + 1] = mix[MixSMid + 1];
        mix[x2] = mix[MixSSide];
        mix[x2 + 1] = mix[MixSSide + 1];
        mix[MixSMid] = mix[x1 + frameLength];
        mix[MixSMid + 1] = mix[x1 + frameLength + 1];
        mix[MixSSide] = mix[x2 + frameLength];
        mix[MixSSide + 1] = mix[x2 + frameLength + 1];

        int predPrev0 = work[WPredPrevQ13], predPrev1 = work[WPredPrevQ13 + 1];
        int pred0 = work[WPredQ13], pred1 = work[WPredQ13 + 1];
        int interpLen = 8 * fsKHz;
        int denomQ16 = (1 << 16) / interpLen;
        int delta0 = RShiftRound((short)(pred0 - predPrev0) * (short)denomQ16, 16);
        int delta1 = RShiftRound((short)(pred1 - predPrev1) * (short)denomQ16, 16);
        for (int n = 0; n < frameLength; n++)
            SilkStereoMsToLrGpu.ApplySideAt(mix, x1, mix, x2, predPrev0, delta0, pred0, predPrev1, delta1, pred1, interpLen, n);
        work[WPredPrevQ13] = pred0;
        work[WPredPrevQ13 + 1] = pred1;
        for (int n = 0; n < frameLength; n++) SilkStereoMsToLrGpu.ApplyMixAt(mix, x1, mix, x2, n);
    }

    private static int RShiftRound(int a, int shift) => shift == 1 ? (a >> 1) + (a & 1) : ((a >> (shift - 1)) + 1) >> 1;

    // Stream transitions: channel resets, resampler resets, stereo start (silk_Decode before the frame header).
    private static void TransitionKernel(Index1D _, SilkResetTargets r, ArrayView<short> mix, ArrayView<int> work,
        ArrayView<short> resShorts, ArrayView<int> resInts, SilkFrameGeometry g, SilkOutputParams p)
    {
        int f = g.TransitionFlags;
        if ((f & TransitionResetMid) != 0) ResetChannel(r, work, 0);
        if ((f & TransitionResetSide) != 0) ResetChannel(r, work, 1);
        if ((f & TransitionPredMemMid) != 0) ResetPredictionMemory(r, work, 0);
        if ((f & TransitionPredMemSide) != 0) ResetPredictionMemory(r, work, 1);
        if ((f & TransitionResetResamplers) != 0)
        {
            for (int k = 0; k < 2 * SilkResamplerGpu.IntsPerResampler; k++) resInts[k] = 0;
            for (int k = 0; k < 2 * SilkResamplerGpu.ShortsPerResampler; k++) resShorts[k] = 0;
        }
        if ((f & TransitionStereoStart) != 0)
        {
            work[WPredPrevQ13] = 0;
            work[WPredPrevQ13 + 1] = 0;
            mix[MixSSide] = 0;
            mix[MixSSide + 1] = 0;
            for (int k = 0; k < SilkResamplerGpu.IntsPerResampler; k++) resInts[SilkResamplerGpu.IntsPerResampler + k] = resInts[k];
            for (int k = 0; k < SilkResamplerGpu.ShortsPerResampler; k++) resShorts[SilkResamplerGpu.ShortsPerResampler + k] = resShorts[k];
        }
    }
}

/// <summary>Icdf offsets the header kernel reads (in the indices byte table).</summary>
public struct SilkLbrrTableOffsets
{
    /// <summary>LBRR flags, 2 frames.</summary>
    public int Lbrr2At;
    /// <summary>LBRR flags, 3 frames.</summary>
    public int Lbrr3At;
    /// <summary>Stereo joint predictor.</summary>
    public int JointAt;
    /// <summary>Uniform 3.</summary>
    public int Uniform3At;
    /// <summary>Uniform 5.</summary>
    public int Uniform5At;
    /// <summary>Mid-only flag.</summary>
    public int MidOnlyAt;
}

/// <summary>Table offsets the stereo kernel reads.</summary>
public struct SilkStereoTableOffsets
{
    /// <summary>Stereo joint predictor icdf.</summary>
    public int JointAt;
    /// <summary>Uniform 3 icdf.</summary>
    public int Uniform3At;
    /// <summary>Uniform 5 icdf.</summary>
    public int Uniform5At;
    /// <summary>Mid-only icdf.</summary>
    public int MidOnlyAt;
    /// <summary>Predictor quantizer (shorts).</summary>
    public int QuantAt;
}

/// <summary>Where the channel state the reset paths touch lives.</summary>
public struct SilkResetTargets
{
    /// <summary>Decode-core shorts (outBuf).</summary>
    public ArrayView<short> CoreShorts;
    /// <summary>Decode-core ints (sLPC, prevGain).</summary>
    public ArrayView<int> CoreInts;
    /// <summary>Parameter shorts (prevNLSF).</summary>
    public ArrayView<short> ParamsShorts;
    /// <summary>Parameter ints (lastGainIndex).</summary>
    public ArrayView<int> ParamsInts;
    /// <summary>Mid outBuf.</summary>
    public int MidOutBufAt;
    /// <summary>Mid sLPC_Q14_buf.</summary>
    public int MidSLpcAt;
    /// <summary>Mid prev_gain_Q16.</summary>
    public int MidPrevGainAt;
    /// <summary>Side outBuf.</summary>
    public int SideOutBufAt;
    /// <summary>Side sLPC_Q14_buf.</summary>
    public int SideSLpcAt;
    /// <summary>Side prev_gain_Q16.</summary>
    public int SidePrevGainAt;
    /// <summary>Mid prevNLSF_Q15.</summary>
    public int MidPrevNlsfAt;
    /// <summary>Mid LastGainIndex.</summary>
    public int MidLastGainAt;
    /// <summary>Side prevNLSF_Q15.</summary>
    public int SidePrevNlsfAt;
    /// <summary>Side LastGainIndex.</summary>
    public int SideLastGainAt;
    /// <summary>Mid PLC / CNG state.</summary>
    public SilkPlcChannel MidPlc;
    /// <summary>Side PLC / CNG state.</summary>
    public SilkPlcChannel SidePlc;
}

/// <summary>The two output resamplers and where the PCM goes.</summary>
public struct SilkOutputParams
{
    /// <summary>Left (or mono) resampler.</summary>
    public SilkResamplerGpuConfig Left;
    /// <summary>Right resampler.</summary>
    public SilkResamplerGpuConfig Right;
    /// <summary>First float of this Opus frame's PCM.</summary>
    public long PcmOutBase;
}

/// <summary>The buffers the frame-tail kernels touch: decode-core state + PLC / CNG state (core shorts / ints), the
/// parameter state (prevNLSF, the LSF cosine table) and the parameter decoder's outputs.</summary>
public struct SilkFrameViews
{
    /// <summary>Decode-core shorts.</summary>
    public ArrayView<short> CoreShorts;
    /// <summary>Decode-core ints.</summary>
    public ArrayView<int> CoreInts;
    /// <summary>Parameter state + tables (shorts).</summary>
    public ArrayView<short> ParamsShorts;
    /// <summary>Parameter decoder short outputs.</summary>
    public ArrayView<short> ParamsShortOut;
    /// <summary>Parameter decoder int outputs.</summary>
    public ArrayView<int> ParamsIntOut;
    /// <summary>decode_core's frame output (core shorts).</summary>
    public int XqAt;
}
