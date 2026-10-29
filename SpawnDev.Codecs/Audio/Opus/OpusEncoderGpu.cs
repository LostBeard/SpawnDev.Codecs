// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Opus encoder on the accelerator, CELT-only (RFC 6716; libopus 1.5.2 opus_encode, fixed point, with
// OPUS_SET_FORCE_MODE(MODE_CELT_ONLY)). This is libopus' Opus layer (opus_encode_native + opus_encode_frame_native) as a
// dispatcher around the CELT encoder. The host makes only the decisions libopus makes from the settings and its own
// earlier decisions: the CBR byte budget, the equivalent rate, the mono / stereo stream decision, the bandwidth (end
// band), the 20 ms split of 40..120 ms packets and the TOC byte value. None of these reads a sample. Kernels do all the
// work on the samples and the bytes: the delay-compensation buffer, the DC reject (or VOIP high-pass), the low-rate
// stereo width fade, the CELT frame encode, and the packet assembly (TOC, code 1 / 3 framing, CBR padding).
//
// SILK and hybrid encoding are not here (CELT-only is the 1.0.0 scope); libopus' tonality analysis runs only at
// complexity 10 and is not ported, so complexity is 0..9 (libopus' default is 9). With CELT-only forced, libopus' stereo
// width estimate (compute_stereo_width) only feeds the SILK / CELT mode choice and DTX needs the analysis, so neither
// changes a CELT-only packet.
//
// Upstream: libopus src/opus_encoder.c, src/repacketizer.c, silk/biquad_alt.c. BSD 3-Clause, NOTICE.md.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Opus.Celt;
using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Opus;

/// <summary>Opus encoder application (libopus OPUS_APPLICATION_*). Named apart from Concentus' OpusApplication so code
/// referencing both has no ambiguity.</summary>
public enum OpusEncoderApplication
{
    /// <summary>OPUS_APPLICATION_VOIP: the Opus layer high-passes the input (60 Hz) instead of the 3 Hz DC reject.</summary>
    Voip = 2048,
    /// <summary>OPUS_APPLICATION_AUDIO.</summary>
    Audio = 2049,
    /// <summary>OPUS_APPLICATION_RESTRICTED_LOWDELAY: no 4 ms delay compensation (CELT-only in libopus too).</summary>
    RestrictedLowDelay = 2051,
}

/// <summary>The signal type hint (libopus OPUS_SET_SIGNAL); it moves the bandwidth and mono / stereo thresholds.</summary>
public enum OpusEncoderSignal
{
    /// <summary>OPUS_AUTO.</summary>
    Auto = -1000,
    /// <summary>OPUS_SIGNAL_VOICE.</summary>
    Voice = 3001,
    /// <summary>OPUS_SIGNAL_MUSIC.</summary>
    Music = 3002,
}

/// <summary>
/// Opus encoder (CELT-only) whose encode runs on the accelerator. Byte-identical to a fixed-point libopus 1.5.2 encoder
/// with OPUS_SET_FORCE_MODE(MODE_CELT_ONLY) (or OPUS_APPLICATION_RESTRICTED_LOWDELAY) at the same settings, complexity
/// 0..9. One instance per stream (it keeps the inter-frame state on the accelerator). Input is interleaved int16 PCM at
/// <see cref="SampleRateHz"/> with <see cref="ChannelCount"/> channels, 2.5 .. 120 ms per call.
/// </summary>
public sealed class OpusEncoderGpu : IDisposable
{
    /// <summary>OPUS_AUTO for <see cref="Bitrate"/>.</summary>
    public const int BitrateAuto = -1000;
    /// <summary>OPUS_BITRATE_MAX for <see cref="Bitrate"/>.</summary>
    public const int BitrateMax = -1;
    /// <summary>The largest packet libopus produces (it caps the output at 1276 bytes per 20 ms frame).</summary>
    public const int MaxPacketBytes = 1276;
    /// <summary>The longest Opus packet (120 ms at 48 kHz) in samples per channel.</summary>
    public const int MaxSamplesPerChannel = 5760;

    private const int ModeSilk = 1000, ModeHybrid = 1001, ModeCelt = 1002;     // libopus MODE_*
    private const int BwNarrow = 1101, BwMedium = 1102, BwWide = 1103, BwSuperWide = 1104, BwFull = 1105;
    private const int MaxFrames = 6;                                           // 120 ms = 6 x 20 ms CELT frames
    private const int VariableHpMinCutoffHz = 60;                              // silk/tuning_parameters.h
    private const int StereoVoiceThreshold = 19000, StereoMusicThreshold = 17000;

    // opus_encoder.c bandwidth transition tables: {threshold, hysteresis} for NB<->MB, MB<->WB, WB<->SWB, SWB<->FB.
    private static readonly int[] MonoVoiceBandwidthThresholds = { 9000, 700, 9000, 700, 13500, 1000, 14000, 2000 };
    private static readonly int[] MonoMusicBandwidthThresholds = { 9000, 700, 9000, 700, 11000, 1000, 12000, 2000 };
    private static readonly int[] StereoVoiceBandwidthThresholds = { 9000, 700, 9000, 700, 13500, 1000, 14000, 2000 };
    private static readonly int[] StereoMusicBandwidthThresholds = { 9000, 700, 9000, 700, 11000, 1000, 12000, 2000 };

    private readonly Accelerator _accelerator;
    private readonly int _encoderBuffer;                                       // st->encoder_buffer (10 ms)
    private readonly int _delayCompensation;                                   // st->delay_compensation (4 ms)
    private readonly int _hpB0, _hpB1, _hpB2, _hpA0, _hpA1;                     // hp_cutoff's biquad at the CELT-only cutoff
    private readonly MemoryBuffer1D<short, Stride1D.Dense> _delay;             // st->delay_buffer
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _hpMem;               // st->hp_mem
    private readonly MemoryBuffer1D<short, Stride1D.Dense> _pcmBuf;            // pcm_buf (delay + one 20 ms frame)
    private readonly MemoryBuffer1D<short, Stride1D.Dense> _window;            // the CELT window, Q15
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _frameLengths;        // CELT bytes per frame of the packet
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _packetLength;        // the assembled packet's length
    private readonly StageBuffer<short> _dInput;
    private readonly StageBuffer<byte> _frames;                                // one slot of curr_max bytes per frame
    private readonly StageBuffer<byte> _packet;

    private readonly Action<Index1D, ArrayView<short>, ArrayView<short>, int> _delayHeadKernel;
    private readonly Action<Index1D, ArrayView<short>, int, ArrayView<short>, int, ArrayView<int>, int, int, int> _dcRejectKernel;
    private readonly Action<Index1D, ArrayView<short>, int, ArrayView<short>, int, ArrayView<int>, int, int, HpBiquad> _hpCutoffKernel;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<short>, int, int, int, int> _delayUpdateKernel;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<short>, int, int, int, int> _stereoFadeKernel;
    private readonly Action<Index1D, ArrayView<byte>, int, ArrayView<int>, int, int, int, int, ArrayView<byte>, ArrayView<int>> _assembleKernel;

    // libopus OpusEncoder state the dispatch keeps (st->...).
    private int _streamChannels;
    private int _mode = ModeHybrid;                                            // opus_encoder_init sets MODE_HYBRID
    private int _bandwidth = BwFull, _autoBandwidth;
    private bool _first = true;
    private int _hybridStereoWidthQ14 = 1 << 14;
    // The CELT encoder's CTL state that the Opus layer sets and leaves (it persists between calls, as in libopus).
    private int _celtVbr, _celtConstrainedVbr = 1, _celtBitrate = BitrateMax, _celtEnd = 21;

    /// <summary>An encoder for <paramref name="channelCount"/> channels at <paramref name="sampleRateHz"/> (8000, 12000,
    /// 16000, 24000 or 48000).</summary>
    public OpusEncoderGpu(Accelerator accelerator, int sampleRateHz, int channelCount, OpusEncoderApplication application)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        if (sampleRateHz != 8000 && sampleRateHz != 12000 && sampleRateHz != 16000 && sampleRateHz != 24000 && sampleRateHz != 48000)
            throw new ArgumentOutOfRangeException(nameof(sampleRateHz), "Opus encodes 8, 12, 16, 24 or 48 kHz.");
        if (channelCount != 1 && channelCount != 2) throw new ArgumentOutOfRangeException(nameof(channelCount));
        if (application != OpusEncoderApplication.Voip && application != OpusEncoderApplication.Audio && application != OpusEncoderApplication.RestrictedLowDelay)
            throw new ArgumentOutOfRangeException(nameof(application));
        _accelerator = accelerator;
        SampleRateHz = sampleRateHz;
        ChannelCount = channelCount;
        Application = application;
        _streamChannels = channelCount;
        _encoderBuffer = sampleRateHz / 100;
        _delayCompensation = sampleRateHz / 250;

        // hp_cutoff's coefficients. CELT-only: variable_HP_smth2_Q15 starts at, and is smoothed toward, the same
        // silk_lin2log(60) << 8, so the cutoff never moves.
        int smth2Q15 = SilkLog2Gpu.Lin2LogQ7(VariableHpMinCutoffHz) << 8;
        int cutoffHz = SilkLog2Gpu.Log2LinQ7(smth2Q15 >> 8);
        int fcQ19 = (int)((short)FixConst(1.5 * 3.14159 / 1000, 19) * (short)cutoffHz) / (sampleRateHz / 1000);
        int rQ28 = FixConst(1.0, 28) - FixConst(0.92, 9) * fcQ19;
        _hpB0 = rQ28;
        _hpB1 = -rQ28 << 1;
        _hpB2 = rQ28;
        int rQ22 = rQ28 >> 6;
        _hpA0 = SmulWW(rQ22, SmulWW(fcQ19, fcQ19) - FixConst(2.0, 22));
        _hpA1 = SmulWW(rQ22, rQ22);

        _delay = accelerator.Allocate1D<short>((long)_encoderBuffer * channelCount);
        _delay.MemSetToZero();
        _hpMem = accelerator.Allocate1D<int>(4);
        _hpMem.MemSetToZero();
        _pcmBuf = accelerator.Allocate1D<short>((long)(_delayCompensation + sampleRateHz / 50) * channelCount);
        _window = accelerator.Allocate1D(CeltModeTables.Window.Select(w => (short)w).ToArray());
        _frameLengths = accelerator.Allocate1D<int>(MaxFrames);
        _packetLength = accelerator.Allocate1D<int>(1);
        _dInput = new StageBuffer<short>(accelerator);
        _frames = new StageBuffer<byte>(accelerator);
        _packet = new StageBuffer<byte>(accelerator);

        _delayHeadKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<short>, ArrayView<short>, int>(DelayHeadKernel);
        _dcRejectKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<short>, int, ArrayView<short>, int, ArrayView<int>, int, int, int>(DcRejectKernel);
        _hpCutoffKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<short>, int, ArrayView<short>, int, ArrayView<int>, int, int, HpBiquad>(HpCutoffKernel);
        _delayUpdateKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<short>, ArrayView<short>, int, int, int, int>(DelayUpdateKernel);
        _stereoFadeKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<short>, ArrayView<short>, int, int, int, int>(StereoFadeKernel);
        _assembleKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, int, ArrayView<int>, int, int, int, int, ArrayView<byte>, ArrayView<int>>(AssembleKernel);
    }

    /// <summary>Input sample rate in Hz.</summary>
    public int SampleRateHz { get; }

    /// <summary>Input channels.</summary>
    public int ChannelCount { get; }

    /// <summary>The application the encoder was created for.</summary>
    public OpusEncoderApplication Application { get; }

    /// <summary>Bits per second (OPUS_SET_BITRATE): 500..512000, <see cref="BitrateAuto"/> (the default) or
    /// <see cref="BitrateMax"/>.</summary>
    public int Bitrate
    {
        get => _userBitrate;
        set
        {
            if (value != BitrateAuto && value != BitrateMax && value <= 0) throw new ArgumentOutOfRangeException(nameof(value));
            _userBitrate = value == BitrateAuto || value == BitrateMax ? value : Math.Clamp(value, 500, 750000 * ChannelCount);
        }
    }
    private int _userBitrate = BitrateAuto;

    /// <summary>Variable bitrate (OPUS_SET_VBR); libopus' default is on.</summary>
    public bool Vbr { get; set; } = true;

    /// <summary>Constrained VBR (OPUS_SET_VBR_CONSTRAINT); libopus' default is on.</summary>
    public bool VbrConstraint { get; set; } = true;

    /// <summary>Complexity 0..9 (OPUS_SET_COMPLEXITY; libopus' default 9). Complexity 10 needs libopus' tonality
    /// analysis, which is not ported.</summary>
    public int Complexity
    {
        get => _complexity;
        set => _complexity = value is >= 0 and <= 9 ? value : throw new ArgumentOutOfRangeException(nameof(value), "0..9");
    }
    private int _complexity = 9;

    /// <summary>The coded bandwidth (OPUS_SET_BANDWIDTH), or null for automatic (the default).</summary>
    public OpusBandwidth? Bandwidth { get; set; }

    /// <summary>The widest automatic bandwidth (OPUS_SET_MAX_BANDWIDTH); libopus' default is fullband.</summary>
    public OpusBandwidth MaxBandwidth { get; set; } = OpusBandwidth.Fullband;

    /// <summary>The signal type hint (OPUS_SET_SIGNAL).</summary>
    public OpusEncoderSignal Signal { get; set; } = OpusEncoderSignal.Auto;

    /// <summary>Forced stream channels (OPUS_SET_FORCE_CHANNELS): null for automatic (the default), 1 or 2.</summary>
    public int? ForceChannels
    {
        get => _forceChannels;
        set => _forceChannels = value is null or 1 || (value == 2 && ChannelCount == 2) ? value : throw new ArgumentOutOfRangeException(nameof(value));
    }
    private int? _forceChannels;

    /// <summary>Expected packet loss in percent (OPUS_SET_PACKET_LOSS_PERC), 0..100: CELT codes the energies more
    /// robustly.</summary>
    public int PacketLossPercent
    {
        get => _lossPercent;
        set => _lossPercent = value is >= 0 and <= 100 ? value : throw new ArgumentOutOfRangeException(nameof(value));
    }
    private int _lossPercent;

    /// <summary>OPUS_SET_PREDICTION_DISABLED: frames independent of the previous ones (no pitch prefilter, intra energy).</summary>
    public bool PredictionDisabled { get; set; }

    /// <summary>The input's significant bits (OPUS_SET_LSB_DEPTH), 8..24; int16 input caps it at 16 as libopus does.</summary>
    public int LsbDepth
    {
        get => _lsbDepth;
        set => _lsbDepth = value is >= 8 and <= 24 ? value : throw new ArgumentOutOfRangeException(nameof(value));
    }
    private int _lsbDepth = 24;

    // ---- the CELT seam ----

    /// <summary>One celt_encode_with_ec call of the Opus layer: the CTL values on entry and where its input and output are.</summary>
    internal readonly record struct CeltFrameCall(
        int Call, int FrameSize, int Channels, int NbCompressedBytes, int Bitrate, bool Vbr, bool ConstrainedVbr, int Complexity,
        int StartBand, int EndBand, int StreamChannels, int LsbDepth, bool DisablePrefilter, bool ForceIntra, int LossRate,
        ArrayView<short> Pcm, ArrayView<byte> Out, int OutAt, ArrayView<int> Length, int LengthAt);

    /// <summary>The CELT frame encoder the Opus layer calls. It must encode <see cref="CeltFrameCall.Pcm"/> into
    /// Out[OutAt..] (at most NbCompressedBytes bytes) and store the byte count at Length[LengthAt].</summary>
    internal Func<CeltFrameCall, Task>? CeltEncode { get; set; }

    private int _celtCalls;

    // ---- encode ----

    /// <summary>
    /// Encode one packet of <paramref name="frameSize"/> samples per channel (2.5, 5, 10, 20, 40, 60, 80, 100 or 120 ms)
    /// from interleaved int16 PCM; at most <paramref name="maxPacketBytes"/> bytes (libopus' out_data_bytes).
    /// </summary>
    internal async Task<byte[]> EncodeAsync(ReadOnlyMemory<short> pcm, int frameSize, int maxPacketBytes = 1500)
    {
        if (pcm.Length < (long)frameSize * ChannelCount) throw new ArgumentException("pcm holds fewer than frameSize samples per channel.", nameof(pcm));
        var dPcm = _dInput.Ensure((long)frameSize * ChannelCount);
        dPcm.SubView(0, (long)frameSize * ChannelCount).CopyFromCPU(pcm.Span[..(frameSize * ChannelCount)].ToArray());
        int length = await EncodeDeviceAsync(dPcm, frameSize, maxPacketBytes);
        return await DeviceTransfer.DownloadAsync(_packet.View, length);
    }

    // opus_encode_native, CELT-only. Leaves the packet in _packet; returns its length.
    private async Task<int> EncodeDeviceAsync(ArrayView<short> pcm, int frameSize, int outDataBytes)
    {
        int fs = SampleRateHz, ch = ChannelCount;
        if (!ValidFrameSize(frameSize)) throw new ArgumentOutOfRangeException(nameof(frameSize), "2.5, 5, 10, 20, 40, 60, 80, 100 or 120 ms.");
        if (outDataBytes <= 0) throw new ArgumentOutOfRangeException(nameof(outDataBytes));
        int maxDataBytes = Math.Min(MaxPacketBytes, outDataBytes);
        if (maxDataBytes == 1 && fs == frameSize * 10) throw new ArgumentException("100 ms cannot be encoded in 1 byte.");
        int lsbDepth = Math.Min(16, _lsbDepth);                                   // opus_encode: int16 input = 16 bits

        int bitrate = UserBitrateToBitrate(frameSize, maxDataBytes);
        int frameRate = fs / frameSize;
        int cbrBytes = -1;
        if (!Vbr)
        {
            int frameRate12 = 12 * fs / frameSize;
            cbrBytes = Math.Min((12 * bitrate / 8 + frameRate12 / 2) / frameRate12, maxDataBytes);
            bitrate = cbrBytes * frameRate12 * 8 / 12;
            maxDataBytes = Math.Max(1, cbrBytes);
        }
        if (maxDataBytes < 3 || bitrate < 3 * frameRate * 8 || (frameRate < 50 && (maxDataBytes * frameRate < 300 || bitrate < 2400)))
            return EncodeTooSmall(frameRate, maxDataBytes, outDataBytes);

        int equivRate = ComputeEquivRate(bitrate, ch, frameRate, Vbr, 0);
        int voiceEst = Signal == OpusEncoderSignal.Voice ? 127 : Signal == OpusEncoderSignal.Music ? 0 : Application == OpusEncoderApplication.Voip ? 115 : 48;
        if (_forceChannels is int forced && ch == 2)
            _streamChannels = forced;
        else if (ch == 2)
        {
            int stereoThreshold = StereoMusicThreshold + ((voiceEst * voiceEst * (StereoVoiceThreshold - StereoMusicThreshold)) >> 14);
            stereoThreshold += _streamChannels == 2 ? -1000 : 1000;
            _streamChannels = equivRate > stereoThreshold ? 2 : 1;
        }
        else
            _streamChannels = ch;
        // The mode: CELT-only (RESTRICTED_LOWDELAY, or forced). No SILK, so no redundancy, no to-mono delay, no prefill.
        _mode = ModeCelt;
        equivRate = ComputeEquivRate(bitrate, _streamChannels, frameRate, Vbr, ModeCelt);

        // Automatic (rate-dependent) bandwidth selection; CELT-only selects every frame.
        {
            bool stereoTables = ch == 2 && _forceChannels != 1;
            var voice = stereoTables ? StereoVoiceBandwidthThresholds : MonoVoiceBandwidthThresholds;
            var music = stereoTables ? StereoMusicBandwidthThresholds : MonoMusicBandwidthThresholds;
            int bandwidth = BwFull;
            do
            {
                int i = 2 * (bandwidth - BwMedium);
                int threshold = music[i] + ((voiceEst * voiceEst * (voice[i] - music[i])) >> 14);
                int hysteresis = music[i + 1] + ((voiceEst * voiceEst * (voice[i + 1] - music[i + 1])) >> 14);
                if (!_first)
                    threshold += _autoBandwidth >= bandwidth ? -hysteresis : hysteresis;
                if (equivRate >= threshold) break;
            }
            while (--bandwidth > BwNarrow);
            if (bandwidth == BwMedium) bandwidth = BwWide;
            _bandwidth = _autoBandwidth = bandwidth;
        }
        int maxBandwidth = BandwidthCode(MaxBandwidth);
        if (_bandwidth > maxBandwidth) _bandwidth = maxBandwidth;
        if (Bandwidth is OpusBandwidth user) _bandwidth = BandwidthCode(user);
        if (fs <= 24000 && _bandwidth > BwSuperWide) _bandwidth = BwSuperWide;
        if (fs <= 16000 && _bandwidth > BwWide) _bandwidth = BwWide;
        if (fs <= 12000 && _bandwidth > BwMedium) _bandwidth = BwMedium;
        if (fs <= 8000 && _bandwidth > BwNarrow) _bandwidth = BwNarrow;
        if (_bandwidth == BwMedium) _bandwidth = BwWide;                         // CELT has no mediumband
        int currBandwidth = _bandwidth;

        if (frameSize > fs / 50)
        {
            // 40..120 ms: 20 ms CELT frames, repacketized.
            int encFrameSize = fs / 50;
            int nbFrames = frameSize / encFrameSize;
            int maxHeaderBytes = nbFrames == 2 ? 3 : 2 + (nbFrames - 1) * 2;
            int repacketizeLen = Vbr || _userBitrate == BitrateMax ? outDataBytes : Math.Min(cbrBytes, outDataBytes);
            int maxLenSum = nbFrames + repacketizeLen - maxHeaderBytes;
            // curr_max = IMIN(max_len_sum - tot_size, ...) never binds: every frame is at most the first term.
            int currMax = Math.Min(3 * bitrate / (3 * 8 * fs / encFrameSize), maxLenSum / nbFrames);
            var slots = _frames.Ensure((long)currMax * nbFrames);
            for (int i = 0; i < nbFrames; i++)
                await EncodeFrameAsync(pcm.SubView((long)i * ch * encFrameSize), encFrameSize, currMax, bitrate, equivRate, currBandwidth, lsbDepth, slots, i * currMax, i);
            byte toc = GenToc(ModeCelt, fs / encFrameSize, currBandwidth, _streamChannels);
            var packet = _packet.Ensure(Math.Max(repacketizeLen, 1));
            _assembleKernel(1, slots, currMax, _frameLengths.View, nbFrames, toc, Vbr ? 0 : 1, repacketizeLen, packet, _packetLength.View);
        }
        else
        {
            var slots = _frames.Ensure(maxDataBytes);
            await EncodeFrameAsync(pcm, frameSize, maxDataBytes, bitrate, equivRate, currBandwidth, lsbDepth, slots, 0, 0);
            byte toc = GenToc(ModeCelt, frameRate, currBandwidth, _streamChannels);
            var packet = _packet.Ensure(maxDataBytes);
            _assembleKernel(1, slots, maxDataBytes, _frameLengths.View, 1, toc, Vbr ? 0 : 1, maxDataBytes, packet, _packetLength.View);
        }
        var len = await _packetLength.View.SubView(0, 1).CopyToHostAsync();
        if (len[0] <= 0) throw new InvalidOperationException($"Opus packet assembly failed ({len[0]}).");
        return len[0];
    }

    // opus_encode_frame_native, CELT-only: the CELT frame's bytes go to slots[slotAt..], their count to _frameLengths[frame].
    private async Task EncodeFrameAsync(ArrayView<short> pcm, int frameSize, int maxDataBytes, int bitrate, int equivRate, int currBandwidth,
        int lsbDepth, ArrayView<byte> slots, int slotAt, int frame)
    {
        int ch = ChannelCount, fs = SampleRateHz;
        int totalBuffer = Application == OpusEncoderApplication.RestrictedLowDelay ? 0 : _delayCompensation;

        // pcm_buf = the delay buffer's tail, then this frame high-passed (VOIP) or DC-rejected.
        if (totalBuffer > 0) _delayHeadKernel(totalBuffer * ch, _pcmBuf.View, _delay.View, (_encoderBuffer - totalBuffer) * ch);
        if (Application == OpusEncoderApplication.Voip)
            _hpCutoffKernel(ch, pcm, 0, _pcmBuf.View, totalBuffer * ch, _hpMem.View, frameSize, ch,
                new HpBiquad { B0 = _hpB0, B1 = _hpB1, B2 = _hpB2, A0 = _hpA0, A1 = _hpA1 });
        else
            _dcRejectKernel(ch, pcm, 0, _pcmBuf.View, totalBuffer * ch, _hpMem.View, frameSize, ch, CeltIlog2(fs / (3 * 4)));

        // CELT CTLs the Opus layer sets for this frame.
        _celtEnd = currBandwidth switch { BwNarrow => 13, BwMedium or BwWide => 17, BwSuperWide => 19, _ => 21 };
        _celtBitrate = BitrateMax;
        int celtPred = PredictionDisabled ? 0 : 2;                             // CELT_SET_PREDICTION
        bool disablePf = celtPred <= 1, forceIntra = celtPred == 0;

        // The delay buffer keeps the newest encoder_buffer samples of pcm_buf (before the stereo fade).
        _delayUpdateKernel(ch, _delay.View, _pcmBuf.View, _encoderBuffer, frameSize, totalBuffer, ch);

        // Stereo width reduction at low rates (CELT-only: the SILK width from the equivalent rate).
        int stereoWidthQ14 = equivRate > 32000 ? 16384 : equivRate < 16000 ? 0 : 16384 - 2048 * (32000 - equivRate) / (equivRate - 14000);
        if (ch == 2 && (_hybridStereoWidthQ14 < (1 << 14) || stereoWidthQ14 < (1 << 14)))
        {
            int g1 = _hybridStereoWidthQ14 == 16384 ? 32767 : _hybridStereoWidthQ14 << 1;
            int g2 = stereoWidthQ14 == 16384 ? 32767 : stereoWidthQ14 << 1;
            int inc = 48000 / fs;
            _stereoFadeKernel(frameSize, _pcmBuf.View, _window.View, 32767 - g1, 32767 - g2, CeltModeTables.Window.Length / inc, inc);
            _hybridStereoWidthQ14 = stereoWidthQ14;
        }

        int nbComprBytes = maxDataBytes - 1;
        _celtVbr = Vbr ? 1 : 0;
        if (Vbr)
        {
            _celtConstrainedVbr = VbrConstraint ? 1 : 0;
            _celtBitrate = bitrate;
        }
        // libopus skips CELT when ec_tell (1 bit) exceeds the budget; the too-small-packet path above already took every
        // budget under 3 bytes, so CELT always runs here. CELT never writes past nbCompressedBytes, so libopus' "SILK
        // busted its target" PLC fallback cannot trigger in CELT-only mode.
        var hook = CeltEncode ?? throw new InvalidOperationException("The CELT frame encoder is not attached.");
        await hook(new CeltFrameCall(_celtCalls++, frameSize, ch, nbComprBytes, _celtBitrate, _celtVbr != 0, _celtConstrainedVbr != 0,
            _complexity, 0, _celtEnd, _streamChannels, lsbDepth, disablePf, forceIntra, _lossPercent,
            _pcmBuf.View, slots, slotAt, _frameLengths.View, frame));
        _first = false;
    }

    // The packet too small to code anything (libopus' 'PLC' frame): the TOC, and for CBR padding to the budget.
    private int EncodeTooSmall(int frameRate, int maxDataBytes, int outDataBytes)
    {
        int tocMode = _mode;
        int bw = _bandwidth == 0 ? BwNarrow : _bandwidth;
        int packetCode = 0, numMultiframes = 0;
        if (tocMode == 0) tocMode = ModeSilk;
        if (frameRate > 100) tocMode = ModeCelt;
        if (frameRate == 25 && tocMode != ModeSilk)
        {
            frameRate = 50;
            packetCode = 1;
        }
        if (frameRate <= 16)
        {
            if (outDataBytes == 1 || (tocMode == ModeSilk && frameRate != 10))
            {
                tocMode = ModeSilk;
                packetCode = frameRate <= 12 ? 1 : 0;
                frameRate = frameRate == 12 ? 25 : 16;
            }
            else
            {
                numMultiframes = 50 / frameRate;
                frameRate = 50;
                packetCode = 3;
            }
        }
        if (tocMode == ModeSilk && bw > BwWide) bw = BwWide;
        else if (tocMode == ModeCelt && bw == BwMedium) bw = BwNarrow;
        else if (tocMode == ModeHybrid && bw <= BwSuperWide) bw = BwSuperWide;
        byte toc = GenToc(tocMode, frameRate, bw, _streamChannels);
        // data = TOC | code (+ the frame count) = that many empty frames; opus_packet_pad re-frames it the same way.
        int frames = packetCode == 3 ? numMultiframes : packetCode == 1 ? 2 : 1;
        int ret = packetCode <= 1 ? 1 : 2;
        maxDataBytes = Math.Max(maxDataBytes, ret);
        _frameLengths.View.SubView(0, frames).MemSetToZero();
        var slots = _frames.Ensure(1);
        var packet = _packet.Ensure(maxDataBytes);
        _assembleKernel(1, slots, 0, _frameLengths.View, frames, toc, Vbr ? 0 : 1, Vbr ? ret : maxDataBytes, packet, _packetLength.View);
        return Vbr ? ret : maxDataBytes;
    }

    private int UserBitrateToBitrate(int frameSize, int maxDataBytes)
    {
        if (frameSize == 0) frameSize = SampleRateHz / 400;
        int b = _userBitrate == BitrateAuto ? 60 * SampleRateHz / frameSize + SampleRateHz * ChannelCount
            : _userBitrate == BitrateMax ? maxDataBytes * 8 * SampleRateHz / frameSize
            : _userBitrate;
        return b;
    }

    private int ComputeEquivRate(int bitrate, int channels, int frameRate, bool vbr, int mode)
    {
        int equiv = bitrate;
        if (frameRate > 50) equiv -= (40 * channels + 20) * (frameRate - 50);
        if (!vbr) equiv -= equiv / 12;
        equiv = equiv * (90 + _complexity) / 100;
        if (mode == ModeCelt)
        {
            if (_complexity < 5) equiv = equiv * 9 / 10;
        }
        else
            equiv -= equiv * _lossPercent / (12 * _lossPercent + 20);              // mode not known yet
        return equiv;
    }

    private bool ValidFrameSize(int n)
    {
        int fs = SampleRateHz;
        return 400 * n == fs || 200 * n == fs || 100 * n == fs || 50 * n == fs || 25 * n == fs || 50 * n == 3 * fs
            || 50 * n == 4 * fs || 50 * n == 5 * fs || 50 * n == 6 * fs;
    }

    private static int BandwidthCode(OpusBandwidth b) => b switch
    {
        OpusBandwidth.Narrowband => BwNarrow, OpusBandwidth.Mediumband => BwMedium, OpusBandwidth.Wideband => BwWide,
        OpusBandwidth.Superwideband => BwSuperWide, _ => BwFull,
    };

    // gen_toc
    private static byte GenToc(int mode, int frameRate, int bandwidth, int channels)
    {
        int period = 0;
        while (frameRate < 400)
        {
            frameRate <<= 1;
            period++;
        }
        int toc;
        if (mode == ModeSilk)
            toc = ((bandwidth - BwNarrow) << 5) | ((period - 2) << 3);
        else if (mode == ModeCelt)
            toc = 0x80 | (Math.Max(0, bandwidth - BwMedium) << 5) | (period << 3);
        else
            toc = 0x60 | ((bandwidth - BwSuperWide) << 4) | ((period - 2) << 3);
        return (byte)(toc | (channels == 2 ? 4 : 0));
    }

    private static int FixConst(double c, int q) => (int)(c * (1L << q) + 0.5);

    private static int SmulWW(int a, int b) => (int)((long)a * b >> 16);           // OPUS_FAST_INT64 (64-bit builds)

    private static int CeltIlog2(int x) => 31 - System.Numerics.BitOperations.LeadingZeroCount((uint)x);

    // ---- kernels ----

    /// <summary>hp_cutoff's biquad coefficients (Q28), a kernel argument (public for the kernel loaders).</summary>
    public struct HpBiquad
    {
        /// <summary>MA coefficients.</summary>
        public int B0, B1, B2;
        /// <summary>AR coefficients.</summary>
        public int A0, A1;
    }

    // pcm_buf[i] = delay_buffer[from + i]
    private static void DelayHeadKernel(Index1D i, ArrayView<short> pcmBuf, ArrayView<short> delay, int from)
        => pcmBuf[i] = delay[from + i];

    // dc_reject (fixed point), one thread per channel: out = in - (its 3 Hz lowpass).
    private static void DcRejectKernel(Index1D c, ArrayView<short> input, int inAt, ArrayView<short> output, int outAt,
        ArrayView<int> hpMem, int len, int channels, int shift)
    {
        int m = hpMem[2 * c];
        int round = 1 << (shift - 1);
        for (int i = 0; i < len; i++)
        {
            int x = input[inAt + channels * i + c] << 14;
            int y = x - m;
            m += (x - m + round) >> shift;                                      // PSHR32(x - hp_mem, shift)
            int v = (y + 8192) >> 14;                                           // PSHR32(y, 14)
            output[outAt + channels * i + c] = (short)(v > 32767 ? 32767 : v < -32767 ? -32767 : v);
        }
        hpMem[2 * c] = m;
    }

    // hp_cutoff = silk_biquad_alt_stride1 / stride2 (the same arithmetic per channel), one thread per channel.
    private static void HpCutoffKernel(Index1D c, ArrayView<short> input, int inAt, ArrayView<short> output, int outAt,
        ArrayView<int> hpMem, int len, int channels, HpBiquad q)
    {
        int s0 = hpMem[2 * c], s1 = hpMem[2 * c + 1];
        int a0L = -q.A0 & 0x3FFF, a0U = -q.A0 >> 14;
        int a1L = -q.A1 & 0x3FFF, a1U = -q.A1 >> 14;
        for (int k = 0; k < len; k++)
        {
            int inval = input[inAt + channels * k + c];
            int out32Q14 = SilkMacros.silk_SMLAWB(s0, q.B0, inval) << 2;
            s0 = s1 + RShiftRound14(SilkMacros.silk_SMULWB(out32Q14, a0L));
            s0 = SilkMacros.silk_SMLAWB(s0, out32Q14, a0U);
            s0 = SilkMacros.silk_SMLAWB(s0, q.B1, inval);
            s1 = RShiftRound14(SilkMacros.silk_SMULWB(out32Q14, a1L));
            s1 = SilkMacros.silk_SMLAWB(s1, out32Q14, a1U);
            s1 = SilkMacros.silk_SMLAWB(s1, q.B2, inval);
            int v = (out32Q14 + (1 << 14) - 1) >> 14;
            output[outAt + channels * k + c] = (short)(v > 32767 ? 32767 : v < -32768 ? -32768 : v);
        }
        hpMem[2 * c] = s0;
        hpMem[2 * c + 1] = s1;
    }

    private static int RShiftRound14(int a) => ((a >> 13) + 1) >> 1;

    // The delay buffer update after the high-pass, one thread per channel (the move overlaps, so it runs in order).
    private static void DelayUpdateKernel(Index1D c, ArrayView<short> delay, ArrayView<short> pcmBuf, int encoderBuffer, int frameSize,
        int totalBuffer, int channels)
    {
        int keep = encoderBuffer - frameSize - totalBuffer;
        if (keep > 0)
        {
            for (int i = 0; i < keep; i++)
                delay[i * channels + c] = delay[(i + frameSize) * channels + c];
            for (int i = 0; i < frameSize + totalBuffer; i++)
                delay[(keep + i) * channels + c] = pcmBuf[i * channels + c];
        }
        else
        {
            int from = frameSize + totalBuffer - encoderBuffer;
            for (int i = 0; i < encoderBuffer; i++)
                delay[i * channels + c] = pcmBuf[(from + i) * channels + c];
        }
    }

    // stereo_fade in place on pcm_buf (stereo), g1 / g2 already Q15ONE - g: the side signal scaled down.
    private static void StereoFadeKernel(Index1D i, ArrayView<short> pcm, ArrayView<short> window, int g1, int g2, int overlap, int inc)
    {
        int g = g2;
        if (i < overlap)
        {
            int w0 = window[i * inc];
            int w = (w0 * w0) >> 15;
            g = (w * g2 + (32767 - w) * g1) >> 15;
        }
        int l = pcm[2 * i], r = pcm[2 * i + 1];
        int diff = (short)((l - r) >> 1);                                       // EXTRACT16(HALF32(l - r))
        diff = (g * diff) >> 15;                                                // MULT16_16_Q15
        pcm[2 * i] = (short)(l - diff);
        pcm[2 * i + 1] = (short)(r + diff);
    }

    // opus_repacketizer_out_range_impl (not self-delimited, no extensions) over `count` frames of one TOC: frame f is
    // slots[f * slotSize ..] with lengths[f] bytes. pad != 0 fills the packet to maxlen (code 3 padding). The length (or a
    // negative libopus error) goes to outLen[0].
    private static void AssembleKernel(Index1D _, ArrayView<byte> slots, int slotSize, ArrayView<int> lengths, int count, int toc, int pad,
        int maxlen, ArrayView<byte> data, ArrayView<int> outLen)
    {
        int len0 = lengths[0];
        int totSize = 0;
        int ptr = 0;
        bool code3 = count > 2;
        if (count == 1)
        {
            totSize = len0 + 1;
            if (totSize > maxlen) { outLen[0] = -2; return; }
            data[ptr++] = (byte)(toc & 0xFC);
        }
        else if (count == 2)
        {
            int len1 = lengths[1];
            if (len1 == len0)
            {
                totSize = 2 * len0 + 1;
                if (totSize > maxlen) { outLen[0] = -2; return; }
                data[ptr++] = (byte)((toc & 0xFC) | 1);
            }
            else
            {
                totSize = len0 + len1 + 2 + (len0 >= 252 ? 1 : 0);
                if (totSize > maxlen) { outLen[0] = -2; return; }
                data[ptr++] = (byte)((toc & 0xFC) | 2);
                ptr = EncodeSize(len0, data, ptr);
            }
        }
        if (code3 || (pad != 0 && totSize < maxlen))
        {
            ptr = 0;
            totSize = 0;
            bool vbr = false;
            for (int i = 1; i < count; i++)
                if (lengths[i] != len0) vbr = true;
            if (vbr)
            {
                totSize += 2;
                for (int i = 0; i < count - 1; i++)
                    totSize += 1 + (lengths[i] >= 252 ? 1 : 0) + lengths[i];
                totSize += lengths[count - 1];
                if (totSize > maxlen) { outLen[0] = -2; return; }
                data[ptr++] = (byte)((toc & 0xFC) | 3);
                data[ptr++] = (byte)(count | 0x80);
            }
            else
            {
                totSize += count * len0 + 2;
                if (totSize > maxlen) { outLen[0] = -2; return; }
                data[ptr++] = (byte)((toc & 0xFC) | 3);
                data[ptr++] = (byte)count;
            }
            int padAmount = pad != 0 ? maxlen - totSize : 0;
            if (padAmount != 0)
            {
                data[1] = (byte)(data[1] | 0x40);
                int nb255s = (padAmount - 1) / 255;
                if (totSize + nb255s + 1 > maxlen) { outLen[0] = -2; return; }
                for (int i = 0; i < nb255s; i++)
                    data[ptr++] = 255;
                data[ptr++] = (byte)(padAmount - 255 * nb255s - 1);
                totSize += padAmount;
            }
            if (vbr)
                for (int i = 0; i < count - 1; i++)
                    ptr = EncodeSize(lengths[i], data, ptr);
        }
        for (int f = 0; f < count; f++)
        {
            int n = lengths[f];
            for (int i = 0; i < n; i++)
                data[ptr + i] = slots[f * slotSize + i];
            ptr += n;
        }
        if (pad != 0)
            while (ptr < maxlen)
                data[ptr++] = 0;
        outLen[0] = totSize;
    }

    // encode_size
    private static int EncodeSize(int size, ArrayView<byte> data, int ptr)
    {
        if (size < 252)
        {
            data[ptr] = (byte)size;
            return ptr + 1;
        }
        int b0 = 252 + (size & 3);
        data[ptr] = (byte)b0;
        data[ptr + 1] = (byte)((size - b0) >> 2);
        return ptr + 2;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _delay.Dispose();
        _hpMem.Dispose();
        _pcmBuf.Dispose();
        _window.Dispose();
        _frameLengths.Dispose();
        _packetLength.Dispose();
        _dInput.Dispose();
        _frames.Dispose();
        _packet.Dispose();
    }
}
