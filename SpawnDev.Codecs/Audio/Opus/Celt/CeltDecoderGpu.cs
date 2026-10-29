// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// The CELT decoder on the accelerator (libopus 1.5.2 celt/celt_decoder.c celt_decode_with_ec, fixed-point build, the static
// 48 kHz mode): one Opus frame's CELT layer, CELT-only (start band 0) or the upper bands of a hybrid frame (start band 17, on
// the range decoder SILK leaves behind). Every bit of the frame is decoded on the accelerator and the int16 PCM is written
// there; the host passes the frame geometry (from the TOC) and dispatches.
//
// Four kernels per frame, each one thread over the frame (its serial work is bounded by the format): the header, energy and
// bit allocation; the band shapes; the synthesis (denormalise + inverse MDCT); the post-filter, state update and de-emphasis.
// The decoder state (history, energies, post-filter, de-emphasis memory, the folding seed) and the per-frame work live in two
// int views; the range decoder state is one OpusRangeDecoderGpuState element that each kernel loads and stores.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using static SpawnDev.Codecs.Audio.Opus.Celt.CeltFixed;

namespace SpawnDev.Codecs.Audio.Opus.Celt;

/// <summary>Per-dispatch geometry of a CELT frame decode (host metadata from the TOC and the Opus layer).</summary>
public struct CeltFrameGeometry
{
    /// <summary>Byte offset of the frame in the packet view.</summary>
    public int PacketStart;
    /// <summary>Bytes of the frame (libopus len; for a hybrid frame with redundancy, without the redundant bytes).</summary>
    public int PacketLength;
    /// <summary>1 to start a fresh range decoder over the frame, 0 to continue the one in the decoder-state view (hybrid).</summary>
    public int InitEc;
    /// <summary>First coded band (0, or 17 in a hybrid frame).</summary>
    public int Start;
    /// <summary>One past the last coded band (13, 17, 19 or 21 by bandwidth).</summary>
    public int End;
    /// <summary>Coded channels (libopus stream_channels).</summary>
    public int C;
    /// <summary>Output channels (libopus channels).</summary>
    public int CC;
    /// <summary>Frame size log2 over 120 samples.</summary>
    public int LM;
    /// <summary>48 kHz / the output rate (1, 2, 3, 4 or 6).</summary>
    public int Downsample;
    /// <summary>1 to add the output into the PCM already there (a hybrid frame on top of SILK).</summary>
    public int Accum;
    /// <summary>Offset of the frame's interleaved int16 output in the PCM view.</summary>
    public int PcmAt;
}

/// <summary>
/// The CELT layer of an Opus decoder on the accelerator (libopus 1.5.2 celt_decode_with_ec). One instance per stream: it keeps
/// the inter-frame state on the accelerator.
/// </summary>
public sealed class CeltDecoderGpu : IDisposable
{
    private const int Nb = CeltModeTables.NbEBands;
    internal const int Overlap = 120, ShortMdctSize = 120, DecodeBufferSize = 2048, ChannelStride = DecodeBufferSize + Overlap;
    private const int Preemph0 = 27853;            // QCONST16(0.8500061035f, 15), mode->preemph[0]
    private const int SigSat = 300000000;          // SIG_SAT
    private const int Neg28 = -28672;              // -QCONST16(28.f, DB_SHIFT)

    /// <summary>Decoder-state layout (ints): both channels' decode_mem, the four energy arrays, the PLC LPC, the de-emphasis memory, scalars.</summary>
    public const int DecodeMemAt = 0, OldBandEAt = 2 * ChannelStride, OldLogEAt = OldBandEAt + 2 * Nb, OldLogE2At = OldLogEAt + 2 * Nb,
        BackgroundLogEAt = OldLogE2At + 2 * Nb, LpcAt = BackgroundLogEAt + 2 * Nb, PreemphMemAt = LpcAt + 2 * 24, ScalarsAt = PreemphMemAt + 2,
        FreqAt = ScalarsAt + 16, StateInts = FreqAt + 2 * 960;
    /// <summary>State scalars (offsets from <see cref="ScalarsAt"/>), libopus' names.</summary>
    public const int SRng = 0, SError = 1, SLastPitchIndex = 2, SLossDuration = 3, SSkipPlc = 4, SPostfilterPeriod = 5,
        SPostfilterPeriodOld = 6, SPostfilterGain = 7, SPostfilterGainOld = 8, SPostfilterTapset = 9, SPostfilterTapsetOld = 10,
        SPrefilterAndFold = 11;

    // Per-frame work layout (ints).
    private const int WXAt = 0, WNormAt = 1920, WFreqAt = 3520, WFineQuantAt = 4480, WPulsesAt = 4501, WCapAt = 4522, WOffsetsAt = 4543,
        WFinePriorityAt = 4564, WTfResAt = 4585, WMasksAt = 4606, WScalarsAt = 4648, WAllocScratchAt = 4672, WQuantScratchAt = 4756,
        WDeemphScratchAt = WQuantScratchAt + CeltQuantBandsGpu.ScratchInts, WEtmpAt = WDeemphScratchAt + 960,
        WorkInts = WEtmpAt + Overlap;
    // Work scalars (offsets from WScalarsAt).
    private const int WSilence = 0, WPfPitch = 1, WPfGain = 2, WPfTapset = 3, WIsTransient = 4, WIntra = 5, WSpread = 6, WAllocTrim = 7,
        WCodedBands = 8, WIntensity = 9, WDualStereo = 10, WBalance = 11, WAntiCollapseRsv = 12, WSeed = 13;

    private readonly Accelerator _accelerator;
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _state, _work, _modeBuffer;
    private readonly MemoryBuffer1D<uint, Stride1D.Dense> _pvqU;
    private readonly MemoryBuffer1D<OpusRangeDecoderGpuState, Stride1D.Dense> _ownEc;
    private readonly CeltModeGpu _mode;
    private readonly Action<Index1D, ArrayView<int>> _resetKernel;
    private readonly Action<Index1D, CeltModeGpu, ArrayView<int>, ArrayView<int>, ArrayView<OpusRangeDecoderGpuState>, ArrayView<byte>, CeltFrameGeometry> _headerKernel;
    private readonly Action<Index1D, CeltModeGpu, ArrayView<int>, ArrayView<int>, ArrayView<OpusRangeDecoderGpuState>, ArrayView<byte>, ArrayView<uint>, CeltFrameGeometry> _bandsKernel;
    private readonly Action<Index1D, CeltModeGpu, ArrayView<int>, ArrayView<int>, CeltFrameGeometry> _synthesisKernel;
    private readonly Action<Index1D, CeltModeGpu, ArrayView<int>, ArrayView<int>, ArrayView<OpusRangeDecoderGpuState>, ArrayView<short>, CeltFrameGeometry> _postKernel;
    private readonly Action<Index1D, CeltModeGpu, ArrayView<int>, ArrayView<int>, ArrayView<short>, CeltFrameGeometry> _lostKernel;

    /// <summary>A decoder for <paramref name="channelCount"/> output channels at <paramref name="sampleRateHz"/> (8, 12, 16, 24 or 48 kHz).</summary>
    public CeltDecoderGpu(Accelerator accelerator, int sampleRateHz, int channelCount)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        if (channelCount != 1 && channelCount != 2) throw new ArgumentOutOfRangeException(nameof(channelCount));
        Downsample = sampleRateHz switch { 48000 => 1, 24000 => 2, 16000 => 3, 12000 => 4, 8000 => 6, _ => 0 };
        if (Downsample == 0) throw new ArgumentOutOfRangeException(nameof(sampleRateHz), "CELT decodes to 8, 12, 16, 24 or 48 kHz.");
        _accelerator = accelerator;
        SampleRateHz = sampleRateHz;
        ChannelCount = channelCount;
        (_mode, _modeBuffer) = CeltModeGpu.Upload(accelerator);
        _pvqU = accelerator.Allocate1D(CeltPvqTables.Data);
        _state = accelerator.Allocate1D<int>(StateInts);
        _work = accelerator.Allocate1D<int>(WorkInts);
        _ownEc = accelerator.Allocate1D<OpusRangeDecoderGpuState>(1);
        _resetKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>>(ResetKernel);
        _headerKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, CeltModeGpu, ArrayView<int>, ArrayView<int>,
            ArrayView<OpusRangeDecoderGpuState>, ArrayView<byte>, CeltFrameGeometry>(HeaderKernel);
        _bandsKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, CeltModeGpu, ArrayView<int>, ArrayView<int>,
            ArrayView<OpusRangeDecoderGpuState>, ArrayView<byte>, ArrayView<uint>, CeltFrameGeometry>(BandsKernel);
        _synthesisKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, CeltModeGpu, ArrayView<int>, ArrayView<int>,
            CeltFrameGeometry>(SynthesisKernel);
        _postKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, CeltModeGpu, ArrayView<int>, ArrayView<int>,
            ArrayView<OpusRangeDecoderGpuState>, ArrayView<short>, CeltFrameGeometry>(PostKernel);
        _lostKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, CeltModeGpu, ArrayView<int>, ArrayView<int>, ArrayView<short>,
            CeltFrameGeometry>(LostKernel);
        ResetState();
    }

    /// <summary>Output sample rate in Hz.</summary>
    public int SampleRateHz { get; }
    /// <summary>Output channels.</summary>
    public int ChannelCount { get; }
    /// <summary>48 kHz / <see cref="SampleRateHz"/>.</summary>
    public int Downsample { get; }
    /// <summary>The decoder state view (layout: the <c>*At</c> and <c>S*</c> constants); for tests and the Opus layer.</summary>
    public ArrayView<int> State => _state.View;

    /// <summary>libopus OPUS_RESET_STATE: clear the state, the previous energies to -28 dB, skip the pitch PLC once.</summary>
    public void ResetState() => _resetKernel(StateInts, _state.View);

    /// <summary>
    /// Decode one CELT frame of <paramref name="samplesPerChannel"/> output samples (2.5, 5, 10 or 20 ms) into
    /// <paramref name="pcm"/> at <paramref name="pcmAt"/>, interleaved int16 (added in when <paramref name="accum"/>).
    /// <paramref name="ec"/> null starts a fresh range decoder over the frame's bytes; otherwise the CELT layer continues on it
    /// (hybrid: the decoder SILK left behind) and leaves it advanced. <paramref name="startBand"/> 0 (CELT-only) or 17 (hybrid),
    /// <paramref name="endBand"/> by bandwidth (NB 13, MB / WB 17, SWB 19, FB 21). A frame of 0 or 1 bytes is concealed
    /// (libopus: <see cref="DecodeLost"/>).
    /// </summary>
    public void Decode(ArrayView<byte> packet, int packetStart, int packetLength, ArrayView<OpusRangeDecoderGpuState>? ec,
        int startBand, int endBand, int streamChannels, int samplesPerChannel, bool accum, ArrayView<short> pcm, int pcmAt)
    {
        if (packetLength <= 1)
        {
            DecodeLost(startBand, endBand, samplesPerChannel, accum, pcm, pcmAt);
            return;
        }
        if (packetLength > 1275) throw new ArgumentOutOfRangeException(nameof(packetLength));
        CheckBands(startBand, endBand);
        if (streamChannels != 1 && streamChannels != 2) throw new ArgumentOutOfRangeException(nameof(streamChannels));
        int lm = FrameLm(samplesPerChannel);
        var g = new CeltFrameGeometry
        {
            PacketStart = packetStart, PacketLength = packetLength, InitEc = ec is null ? 1 : 0, Start = startBand, End = endBand,
            C = streamChannels, CC = ChannelCount, LM = lm, Downsample = Downsample, Accum = accum ? 1 : 0, PcmAt = pcmAt,
        };
        var ecView = ec ?? _ownEc.View;
        _headerKernel(1, _mode, _state.View, _work.View, ecView, packet, g);
        _bandsKernel(1, _mode, _state.View, _work.View, ecView, packet, _pvqU.View, g);
        _synthesisKernel(1, _mode, _state.View, _work.View, g);
        _postKernel(1, _mode, _state.View, _work.View, ecView, pcm, g);
    }

    /// <summary>
    /// Conceal one lost CELT frame of <paramref name="samplesPerChannel"/> output samples (libopus celt_decode_with_ec with no
    /// data: celt_decode_lost, then the de-emphasis) into <paramref name="pcm"/> at <paramref name="pcmAt"/>. The bands are the
    /// decoder's current ones (libopus st->start / st->end: the Opus layer's last settings).
    /// </summary>
    public void DecodeLost(int startBand, int endBand, int samplesPerChannel, bool accum, ArrayView<short> pcm, int pcmAt)
    {
        CheckBands(startBand, endBand);
        var g = new CeltFrameGeometry
        {
            Start = startBand, End = endBand, C = ChannelCount, CC = ChannelCount, LM = FrameLm(samplesPerChannel),
            Downsample = Downsample, Accum = accum ? 1 : 0, PcmAt = pcmAt,
        };
        _lostKernel(1, _mode, _state.View, _work.View, pcm, g);
    }

    private static void CheckBands(int startBand, int endBand)
    {
        if (startBand != 0 && startBand != 17) throw new ArgumentOutOfRangeException(nameof(startBand));
        if (endBand <= startBand || endBand > Nb) throw new ArgumentOutOfRangeException(nameof(endBand));
    }

    private int FrameLm(int samplesPerChannel)
    {
        int n = samplesPerChannel * Downsample, lm = 0;
        while (lm <= 3 && ShortMdctSize << lm != n) lm++;
        if (lm > 3) throw new ArgumentOutOfRangeException(nameof(samplesPerChannel), "A CELT frame is 2.5, 5, 10 or 20 ms.");
        return lm;
    }

    // celt_decode_with_ec without data: celt_decode_lost, then the de-emphasis of out_syn.
    private static void LostKernel(Index1D _, CeltModeGpu m, ArrayView<int> st, ArrayView<int> w, ArrayView<short> pcm, CeltFrameGeometry g)
    {
        CeltPlcGpu.DecodeLost(m, st, w, WXAt, g.CC, g.LM, g.Start, g.End, g.Downsample, WEtmpAt);
        int N = ShortMdctSize << g.LM;
        CeltSynthesisGpu.Deemphasis(st, DecodeMemAt + DecodeBufferSize - N, ChannelStride, pcm, g.PcmAt, N, g.CC, g.Downsample, Preemph0,
            st, PreemphMemAt, g.Accum, w, WDeemphScratchAt);
    }

    /// <summary>The range coder's final state after the last frame (libopus OPUS_GET_FINAL_RANGE for the CELT layer).</summary>
    public async Task<uint> GetFinalRangeAsync()
    {
        var s = await _state.View.SubView(ScalarsAt + SRng, 1).CopyToHostAsync();
        return (uint)s[0];
    }

    private static void ResetKernel(Index1D i, ArrayView<int> st)
    {
        int v = 0;
        if (i >= OldLogEAt && i < BackgroundLogEAt) v = Neg28;   // oldLogE and oldLogE2
        if (i == ScalarsAt + SSkipPlc) v = 1;
        st[i] = v;
    }

    // The header, the band energies and the bit allocation: celt_decode_with_ec up to unquant_fine_energy.
    private static void HeaderKernel(Index1D _, CeltModeGpu m, ArrayView<int> st, ArrayView<int> w,
        ArrayView<OpusRangeDecoderGpuState> ecBuf, ArrayView<byte> pkt, CeltFrameGeometry g)
    {
        int at = g.PacketStart, len = g.PacketLength, C = g.C, LM = g.LM, M = 1 << LM, start = g.Start, end = g.End;
        uint storage = (uint)len;
        var dec = g.InitEc != 0 ? OpusRangeDecoderGpu.Init(pkt, at, storage) : ecBuf[0];
        int s0 = ScalarsAt;
        if (st[s0 + SLossDuration] == 0) st[s0 + SSkipPlc] = 0;
        if (C == 1)
            for (int i = 0; i < Nb; i++)
                st[OldBandEAt + i] = MAX16(st[OldBandEAt + i], st[OldBandEAt + Nb + i]);
        int total_bits = len * 8;
        int tell = CeltEntropyGpu.Tell(ref dec);
        int silence;
        if (tell >= total_bits) silence = 1;
        else if (tell == 1) silence = OpusRangeDecoderGpu.DecodeBitLogP(ref dec, pkt, at, storage, 15);
        else silence = 0;
        if (silence != 0)
        {
            // Pretend we've read all the remaining bits
            tell = len * 8;
            dec.NBitsTotal += tell - CeltEntropyGpu.Tell(ref dec);
        }
        int postfilter_gain = 0, postfilter_pitch = 0, postfilter_tapset = 0;
        if (start == 0 && tell + 16 <= total_bits)
        {
            if (OpusRangeDecoderGpu.DecodeBitLogP(ref dec, pkt, at, storage, 1) != 0)
            {
                int octave = (int)OpusRangeDecoderGpu.DecodeUint(ref dec, pkt, at, storage, 6);
                postfilter_pitch = (16 << octave) + (int)OpusRangeDecoderGpu.DecodeBits(ref dec, pkt, at, storage, 4 + octave) - 1;
                int qg = (int)OpusRangeDecoderGpu.DecodeBits(ref dec, pkt, at, storage, 3);
                if (CeltEntropyGpu.Tell(ref dec) + 2 <= total_bits)
                    postfilter_tapset = CeltEntropyGpu.DecodeIcdfMode(ref dec, pkt, at, storage, m, m.TapsetIcdfAt, 2);
                postfilter_gain = 3072 * (qg + 1);   // QCONST16(.09375f, 15) * (qg + 1)
            }
            tell = CeltEntropyGpu.Tell(ref dec);
        }
        int isTransient = 0;
        if (LM > 0 && tell + 3 <= total_bits)
        {
            isTransient = OpusRangeDecoderGpu.DecodeBitLogP(ref dec, pkt, at, storage, 3);
            tell = CeltEntropyGpu.Tell(ref dec);
        }
        int intra_ener = tell + 3 <= total_bits ? OpusRangeDecoderGpu.DecodeBitLogP(ref dec, pkt, at, storage, 3) : 0;
        int lossDuration = st[s0 + SLossDuration];
        if (intra_ener == 0 && lossDuration != 0)
        {
            // Recovering from packet loss: make the energy prediction safe.
            for (int c = 0; c < 2; c++)
            {
                int safety = 0;
                int missing = IMIN(10, lossDuration >> LM);
                if (LM == 0) safety = 1536;        // QCONST16(1.5f, DB_SHIFT)
                else if (LM == 1) safety = 512;    // QCONST16(.5f, DB_SHIFT)
                for (int i = start; i < end; i++)
                {
                    int e = c * Nb + i;
                    int e0 = st[OldBandEAt + e], e1 = st[OldLogEAt + e], e2 = st[OldLogE2At + e];
                    if (e0 < MAX16(e1, e2))
                    {
                        // If energy is going down already, continue the trend.
                        int slope = MAX32(e1 - e0, HALF32(e2 - e0));
                        e0 -= MAX32(0, (1 + missing) * slope);
                        st[OldBandEAt + e] = MAX32(-20480, e0);
                    }
                    else
                    {
                        // Otherwise take the min of the last frames.
                        st[OldBandEAt + e] = MIN16(MIN16(e0, e1), e2);
                    }
                    st[OldBandEAt + e] -= safety;
                }
            }
        }
        CeltEnergyGpu.UnquantCoarseEnergy(m, start, end, st, OldBandEAt, intra_ener, ref dec, pkt, at, storage, C, LM);
        CeltSynthesisGpu.TfDecode(m, start, end, isTransient, w, WTfResAt, LM, ref dec, pkt, at, storage);
        tell = CeltEntropyGpu.Tell(ref dec);
        int spread_decision = 2;   // SPREAD_NORMAL
        if (tell + 4 <= total_bits)
            spread_decision = CeltEntropyGpu.DecodeIcdfMode(ref dec, pkt, at, storage, m, m.SpreadIcdfAt, 5);
        CeltRateGpu.InitCaps(m, w, WCapAt, LM, C);
        int dynalloc_logp = 6;
        total_bits <<= 3;
        tell = (int)CeltEntropyGpu.TellFrac(ref dec);
        for (int i = start; i < end; i++)
        {
            int width = C * (m.EBandsOf(i + 1) - m.EBandsOf(i)) << LM;
            // quanta is 6 bits, but no more than 1 bit/sample and no less than 1/8 bit/sample
            int quanta = IMIN(width << 3, IMAX(6 << 3, width));
            int dynalloc_loop_logp = dynalloc_logp;
            int boost = 0;
            while (tell + (dynalloc_loop_logp << 3) < total_bits && boost < w[WCapAt + i])
            {
                int flag = OpusRangeDecoderGpu.DecodeBitLogP(ref dec, pkt, at, storage, dynalloc_loop_logp);
                tell = (int)CeltEntropyGpu.TellFrac(ref dec);
                if (flag == 0) break;
                boost += quanta;
                total_bits -= quanta;
                dynalloc_loop_logp = 1;
            }
            w[WOffsetsAt + i] = boost;
            // Making dynalloc more likely
            if (boost > 0) dynalloc_logp = IMAX(2, dynalloc_logp - 1);
        }
        int alloc_trim = tell + (6 << 3) <= total_bits ? CeltEntropyGpu.DecodeIcdfMode(ref dec, pkt, at, storage, m, m.TrimIcdfAt, 7) : 5;
        int bits = ((len * 8) << 3) - (int)CeltEntropyGpu.TellFrac(ref dec) - 1;
        int anti_collapse_rsv = isTransient != 0 && LM >= 2 && bits >= ((LM + 2) << 3) ? (1 << 3) : 0;
        bits -= anti_collapse_rsv;
        int intensity = 0, dual_stereo = 0;
        var offsets = w.SubView(WOffsetsAt, Nb);
        var cap = w.SubView(WCapAt, Nb);
        var pulses = w.SubView(WPulsesAt, Nb);
        var fineQuant = w.SubView(WFineQuantAt, Nb);
        var finePriority = w.SubView(WFinePriorityAt, Nb);
        int codedBands = CeltRateGpu.ComputeAllocationDecode(m, start, end, offsets, cap, alloc_trim, ref intensity, ref dual_stereo,
            bits, out int balance, pulses, fineQuant, finePriority, 0, w, WAllocScratchAt, C, LM, ref dec, pkt, at, storage, 0, 0);
        CeltEnergyGpu.UnquantFineEnergy(start, end, st, OldBandEAt, w, WFineQuantAt, ref dec, pkt, at, storage, C);
        int ws = WScalarsAt;
        w[ws + WSilence] = silence;
        w[ws + WPfPitch] = postfilter_pitch;
        w[ws + WPfGain] = postfilter_gain;
        w[ws + WPfTapset] = postfilter_tapset;
        w[ws + WIsTransient] = isTransient;
        w[ws + WIntra] = intra_ener;
        w[ws + WSpread] = spread_decision;
        w[ws + WAllocTrim] = alloc_trim;
        w[ws + WCodedBands] = codedBands;
        w[ws + WIntensity] = intensity;
        w[ws + WDualStereo] = dual_stereo;
        w[ws + WBalance] = balance;
        w[ws + WAntiCollapseRsv] = anti_collapse_rsv;
        ecBuf[0] = dec;
    }

    // The band shapes, the anti-collapse bit, the last fine-energy bits and the anti-collapse.
    private static void BandsKernel(Index1D _, CeltModeGpu m, ArrayView<int> st, ArrayView<int> w,
        ArrayView<OpusRangeDecoderGpuState> ecBuf, ArrayView<byte> pkt, ArrayView<uint> pvqU, CeltFrameGeometry g)
    {
        int at = g.PacketStart, len = g.PacketLength, C = g.C, LM = g.LM, start = g.Start, end = g.End;
        int N = ShortMdctSize << LM;
        uint storage = (uint)len;
        var dec = ecBuf[0];
        int ws = WScalarsAt;
        int isTransient = w[ws + WIsTransient];
        int anti_collapse_rsv = w[ws + WAntiCollapseRsv];
        uint seed = (uint)st[ScalarsAt + SRng];
        CeltQuantBandsGpu.QuantAllBandsDecode(m, start, end, w, WXAt, C == 2 ? WXAt + N : -1, WNormAt, w, WMasksAt, w, WPulsesAt,
            isTransient != 0 ? 1 << LM : 0, w[ws + WSpread], w[ws + WDualStereo], w[ws + WIntensity], w, WTfResAt,
            len * (8 << 3) - anti_collapse_rsv, w[ws + WBalance], ref dec, pkt, at, storage, LM, w[ws + WCodedBands], ref seed,
            w, WQuantScratchAt, pvqU, 0, g.CC == 1 ? 1 : 0);   // libopus disable_inv = (channels == 1)
        int anti_collapse_on = 0;
        if (anti_collapse_rsv > 0)
            anti_collapse_on = (int)OpusRangeDecoderGpu.DecodeBits(ref dec, pkt, at, storage, 1);
        CeltEnergyGpu.UnquantEnergyFinalise(start, end, st, OldBandEAt, w, WFineQuantAt, w, WFinePriorityAt,
            len * 8 - CeltEntropyGpu.Tell(ref dec), ref dec, pkt, at, storage, C);
        if (anti_collapse_on != 0)
            CeltBandsGpu.AntiCollapse(m, w, WXAt, N, w, WMasksAt, LM, C, start, end, st, OldBandEAt, st, OldLogEAt, st, OldLogE2At,
                w, WPulsesAt, seed);
        if (w[ws + WSilence] != 0)
            for (int i = 0; i < C * Nb; i++)
                st[OldBandEAt + i] = Neg28;
        w[ws + WSeed] = (int)seed;
        ecBuf[0] = dec;
    }

    // The decode_mem shift and celt_synthesis: denormalise_bands, the inverse MDCT of each block, the saturation.
    private static void SynthesisKernel(Index1D _, CeltModeGpu m, ArrayView<int> st, ArrayView<int> w, CeltFrameGeometry g)
    {
        int CC = g.CC, LM = g.LM;
        int N = ShortMdctSize << LM;
        for (int c = 0; c < CC; c++)
        {
            int d = DecodeMemAt + c * ChannelStride;
            for (int i = 0; i < DecodeBufferSize - N + Overlap; i++)
                st[d + i] = st[d + N + i];
        }
        if (st[ScalarsAt + SPrefilterAndFold] != 0)
            CeltPlcGpu.PrefilterAndFold(m, st, CC, N, w, WEtmpAt);
        Synthesis(m, st, w, WXAt, g.C, CC, w[WScalarsAt + WIsTransient], LM, g.Start, g.End, g.Downsample, w[WScalarsAt + WSilence]);
    }

    /// <summary>
    /// libopus celt_synthesis: denormalise_bands of the normalised spectrum at <paramref name="xAt"/> in <paramref name="w"/> (C
    /// channels, N apart) with the state's band energies, the inverse MDCT of each block into out_syn, the saturation.
    /// </summary>
    internal static void Synthesis(CeltModeGpu m, ArrayView<int> st, ArrayView<int> w, int xAt, int C, int CC, int isTransient,
        int LM, int start, int effEnd, int downsample, int silence)
    {
        int M = 1 << LM;
        int N = ShortMdctSize << LM;
        int B, NB, shift;
        if (isTransient != 0)
        {
            B = M;
            NB = ShortMdctSize;
            shift = 3;
        }
        else
        {
            B = 1;
            NB = ShortMdctSize << LM;
            shift = 3 - LM;
        }
        int outSyn0 = DecodeMemAt + DecodeBufferSize - N;
        // Each coded channel's MDCT coefficients in its own buffer (FreqAt + c * 960), one denormalise_bands and one inverse
        // MDCT call site (on WebGPU every call site is inlined). libopus borrows out_syn as the second buffer when copying a
        // mono stream to two channels or downmixing a stereo one; every sample it borrows is overwritten before it is read, so
        // the output is the same.
        for (int c = 0; c < C; c++)
            CeltSynthesisGpu.DenormaliseBands(m, w, xAt + c * N, st, FreqAt + c * 960, st, OldBandEAt + c * Nb, start, effEnd, M,
                downsample, silence);
        if (CC == 2 && C == 1)
        {
            // Copying a mono stream to two channels (the IMDCT destroys its input)
            for (int i = 0; i < N; i++) st[FreqAt + 960 + i] = st[FreqAt + i];
        }
        else if (CC == 1 && C == 2)
        {
            // Downmixing a stereo stream to mono
            for (int i = 0; i < N; i++)
                st[FreqAt + i] = ADD32(HALF32(st[FreqAt + i]), HALF32(st[FreqAt + 960 + i]));
        }
        for (int c = 0; c < CC; c++)
            for (int b = 0; b < B; b++)
                CeltMdctGpu.Backward(m, st, FreqAt + c * 960 + b, st, outSyn0 + c * ChannelStride + NB * b, Overlap, shift, B);
        // Saturate the IMDCT output so that the pitch post-filter cannot overflow.
        for (int c = 0; c < CC; c++)
            for (int i = 0; i < N; i++)
            {
                int k = outSyn0 + c * ChannelStride + i;
                int v = st[k];
                st[k] = v > SigSat ? SigSat : v < -SigSat ? -SigSat : v;
            }
    }

    // The pitch post-filter, the inter-frame state update and the de-emphasis into int16 PCM.
    private static void PostKernel(Index1D _, CeltModeGpu m, ArrayView<int> st, ArrayView<int> w,
        ArrayView<OpusRangeDecoderGpuState> ecBuf, ArrayView<short> pcm, CeltFrameGeometry g)
    {
        int C = g.C, CC = g.CC, LM = g.LM, M = 1 << LM, start = g.Start, end = g.End;
        int N = ShortMdctSize << LM;
        int s0 = ScalarsAt, ws = WScalarsAt;
        int postfilter_pitch = w[ws + WPfPitch], postfilter_gain = w[ws + WPfGain], postfilter_tapset = w[ws + WPfTapset];
        int isTransient = w[ws + WIsTransient];
        int outSyn0 = DecodeMemAt + DecodeBufferSize - N;
        for (int c = 0; c < CC; c++)
        {
            st[s0 + SPostfilterPeriod] = IMAX(st[s0 + SPostfilterPeriod], 15);       // COMBFILTER_MINPERIOD
            st[s0 + SPostfilterPeriodOld] = IMAX(st[s0 + SPostfilterPeriodOld], 15);
            int o = outSyn0 + c * ChannelStride;
            CeltCombFilterGpu.CombFilter(m, st, o, st, o, st[s0 + SPostfilterPeriodOld], st[s0 + SPostfilterPeriod], ShortMdctSize,
                st[s0 + SPostfilterGainOld], st[s0 + SPostfilterGain], st[s0 + SPostfilterTapsetOld], st[s0 + SPostfilterTapset], Overlap);
            if (LM != 0)
                CeltCombFilterGpu.CombFilter(m, st, o + ShortMdctSize, st, o + ShortMdctSize, st[s0 + SPostfilterPeriod], postfilter_pitch,
                    N - ShortMdctSize, st[s0 + SPostfilterGain], postfilter_gain, st[s0 + SPostfilterTapset], postfilter_tapset, Overlap);
        }
        st[s0 + SPostfilterPeriodOld] = st[s0 + SPostfilterPeriod];
        st[s0 + SPostfilterGainOld] = st[s0 + SPostfilterGain];
        st[s0 + SPostfilterTapsetOld] = st[s0 + SPostfilterTapset];
        st[s0 + SPostfilterPeriod] = postfilter_pitch;
        st[s0 + SPostfilterGain] = postfilter_gain;
        st[s0 + SPostfilterTapset] = postfilter_tapset;
        if (LM != 0)
        {
            st[s0 + SPostfilterPeriodOld] = st[s0 + SPostfilterPeriod];
            st[s0 + SPostfilterGainOld] = st[s0 + SPostfilterGain];
            st[s0 + SPostfilterTapsetOld] = st[s0 + SPostfilterTapset];
        }
        if (C == 1)
            for (int i = 0; i < Nb; i++)
                st[OldBandEAt + Nb + i] = st[OldBandEAt + i];
        if (isTransient == 0)
        {
            for (int i = 0; i < 2 * Nb; i++)
            {
                st[OldLogE2At + i] = st[OldLogEAt + i];
                st[OldLogEAt + i] = st[OldBandEAt + i];
            }
        }
        else
        {
            for (int i = 0; i < 2 * Nb; i++)
                st[OldLogEAt + i] = MIN16(st[OldLogEAt + i], st[OldBandEAt + i]);
        }
        // The noise floor rises by at most 2.4 dB/second, but in DTX the missing packets' weight goes to the update packet.
        int max_background_increase = IMIN(160, st[s0 + SLossDuration] + M) * 1;   // QCONST16(0.001f, DB_SHIFT) = 1
        for (int i = 0; i < 2 * Nb; i++)
            st[BackgroundLogEAt + i] = MIN16(st[BackgroundLogEAt + i] + max_background_increase, st[OldBandEAt + i]);
        // In case start or end were to change
        for (int c = 0; c < 2; c++)
        {
            for (int i = 0; i < start; i++)
            {
                st[OldBandEAt + c * Nb + i] = 0;
                st[OldLogEAt + c * Nb + i] = st[OldLogE2At + c * Nb + i] = Neg28;
            }
            for (int i = end; i < Nb; i++)
            {
                st[OldBandEAt + c * Nb + i] = 0;
                st[OldLogEAt + c * Nb + i] = st[OldLogE2At + c * Nb + i] = Neg28;
            }
        }
        var dec = ecBuf[0];
        st[s0 + SRng] = (int)dec.Rng;
        CeltSynthesisGpu.Deemphasis(st, outSyn0, ChannelStride, pcm, g.PcmAt, N, CC, g.Downsample, Preemph0, st, PreemphMemAt, g.Accum,
            w, WDeemphScratchAt);
        st[s0 + SLossDuration] = 0;
        st[s0 + SPrefilterAndFold] = 0;
        if (CeltEntropyGpu.Tell(ref dec) > 8 * g.PacketLength) st[s0 + SError] = 1;   // libopus returns OPUS_INTERNAL_ERROR
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _state.Dispose();
        _work.Dispose();
        _modeBuffer.Dispose();
        _pvqU.Dispose();
        _ownEc.Dispose();
    }
}
