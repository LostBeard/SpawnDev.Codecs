// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Opus decoder on the accelerator (RFC 6716; libopus 1.5.2 opus_decode, fixed point). The host parses the packet's TOC
// byte and frame packing (codes 0-3: container framing, the same metadata a demuxer reads) and runs libopus'
// opus_decode_frame as a dispatcher: the SILK decoder (SilkPacketDecoderGpu) and the CELT decoder (CeltDecoderGpu,
// created on the first CELT or hybrid frame) decode every bit of the frames on the accelerator into an int16 PCM
// buffer there, and small kernels here do the rest of the glue (the redundancy flags on the shared range decoder,
// smooth_fade, the SILK + CELT sum) before one conversion to float. The only bitstream values that cross to the host
// are the four redundancy ints after the SILK layer (whether a 5 ms CELT redundancy frame follows, its direction, its
// bytes and the shortened frame length), which the dispatch needs, as the Vorbis decoder reads its packet header ints.
//
// Upstream: libopus src/opus_decoder.c (opus_decode_frame, opus_decode_native, smooth_fade). BSD 3-Clause, NOTICE.md.

using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Opus.Celt;
using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Audio.Opus;

/// <summary>
/// Opus decoder whose decode runs on the accelerator: SILK, hybrid and CELT frames, the mode transitions (redundancy
/// frames, the transition concealment and fades, CELT resets), packet-loss concealment and DTX. One instance per stream
/// (it keeps the inter-frame state on the accelerator). Output is interleaved float PCM at <see cref="SampleRateHz"/>
/// with <see cref="ChannelCount"/> channels, equal to a fixed-point libopus 1.5.2's int16 output / 32768.
/// </summary>
public sealed class OpusDecoderGpu : IDisposable
{
    /// <summary>Most samples per channel one packet can produce (120 ms at 48 kHz).</summary>
    public const int MaxSamplesPerChannel = 5760;

    // Packet bytes read back for the frame layout of a device-resident packet: TOC, count byte, frame sizes, a short
    // padding run. A longer padding run reads the whole packet back.
    private const int HeaderReadback = 2 + 2 * OpusPacketParser.MaxFramesPerPacket + 8;

    private const int ModeNone = 0, ModeSilk = 1000, ModeHybrid = 1001, ModeCelt = 1002; // libopus MODE_* (0 = none yet)

    private readonly Accelerator _accelerator;
    private readonly SilkPacketDecoderGpu _silk;
    private CeltDecoderGpu? _celt;
    private readonly StageBuffer<byte> _dPacket;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _dPcm;
    private readonly MemoryBuffer1D<short, Stride1D.Dense> _pcm16;          // the packet's int16 PCM (libopus pcm)
    private readonly MemoryBuffer1D<short, Stride1D.Dense> _pcmSilk;        // pcm_silk (when CELT does not accumulate)
    private readonly MemoryBuffer1D<short, Stride1D.Dense> _pcmTransition;  // pcm_transition (5 ms)
    private readonly MemoryBuffer1D<short, Stride1D.Dense> _redundantAudio; // redundant_audio (5 ms)
    private readonly MemoryBuffer1D<short, Stride1D.Dense> _window;         // the CELT window, Q15
    private readonly MemoryBuffer1D<byte, Stride1D.Dense> _silence;         // the {0xFF, 0xFF} CELT silence frame
    private readonly MemoryBuffer1D<int, Stride1D.Dense> _info;             // redundancy decisions + the range state

    private readonly Action<Index1D, ArrayView<byte>, ArrayView<OpusRangeDecoderGpuState>, ArrayView<int>, int, int, int> _redundancyKernel;
    private readonly Action<Index1D, ArrayView<short>, int, ArrayView<short>, int, int, int, int, ArrayView<short>, int> _fadeKernel;
    private readonly Action<Index1D, ArrayView<short>, int, ArrayView<short>, int> _addKernel;
    private readonly Action<Index1D, ArrayView<short>, int, ArrayView<short>, int> _copyKernel;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<float>> _toFloatKernel;
    private readonly Action<Index1D, ArrayView<OpusRangeDecoderGpuState>, ArrayView<int>> _rangeKernel;

    // libopus decoder state the dispatch needs (st->...).
    private int _mode, _bandwidth, _frameSize, _streamChannels;   // the current packet's TOC (st->mode etc.)
    private int _prevMode;
    private bool _prevRedundancy;
    private int _celtStart, _celtEnd = 21;                         // CELT st->start / st->end (CELT_SET_*_BAND persist)
    private int _silkKHz;                                          // DecControl.internalSampleRate
    // Final range of the last frame (OPUS_GET_FINAL_RANGE): from the SILK range decoder, CELT's, or none.
    private int _rangeSource;                                      // 0 none, 1 SILK ec ^ redundant, 2 CELT
    private uint _redundantRng;

    /// <summary>A decoder producing <paramref name="channelCount"/> channels at <paramref name="sampleRateHz"/>
    /// (8000, 12000, 16000, 24000 or 48000).</summary>
    public OpusDecoderGpu(Accelerator accelerator, int sampleRateHz, int channelCount)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        if (sampleRateHz != 8000 && sampleRateHz != 12000 && sampleRateHz != 16000 && sampleRateHz != 24000 && sampleRateHz != 48000)
            throw new ArgumentOutOfRangeException(nameof(sampleRateHz), "Opus decodes to 8, 12, 16, 24 or 48 kHz.");
        if (channelCount != 1 && channelCount != 2) throw new ArgumentOutOfRangeException(nameof(channelCount));
        _accelerator = accelerator;
        SampleRateHz = sampleRateHz;
        ChannelCount = channelCount;
        _silk = new SilkPacketDecoderGpu(accelerator, sampleRateHz, channelCount);
        _dPacket = new StageBuffer<byte>(accelerator);
        _dPcm = accelerator.Allocate1D<float>(MaxSamplesPerPacket);
        int f20 = sampleRateHz / 50;
        _pcm16 = accelerator.Allocate1D<short>(MaxSamplesPerPacket);
        _pcmSilk = accelerator.Allocate1D<short>((long)f20 * channelCount);
        _pcmTransition = accelerator.Allocate1D<short>((long)f20 / 4 * channelCount);
        _redundantAudio = accelerator.Allocate1D<short>((long)f20 / 4 * channelCount);
        _window = accelerator.Allocate1D(CeltModeTables.Window.Select(w => (short)w).ToArray());
        _silence = accelerator.Allocate1D(new byte[] { 0xFF, 0xFF });
        _info = accelerator.Allocate1D<int>(8);
        _redundancyKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<OpusRangeDecoderGpuState>, ArrayView<int>, int, int, int>(RedundancyKernel);
        _fadeKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<short>, int, ArrayView<short>, int, int, int, int, ArrayView<short>, int>(FadeKernel);
        _addKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<short>, int, ArrayView<short>, int>(AddKernel);
        _copyKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<short>, int, ArrayView<short>, int>(CopyKernel);
        _toFloatKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<short>, ArrayView<float>>(ToFloatKernel);
        _rangeKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<OpusRangeDecoderGpuState>, ArrayView<int>>(RangeKernel);
    }

    /// <summary>Output sample rate in Hz.</summary>
    public int SampleRateHz { get; }

    /// <summary>Output channels.</summary>
    public int ChannelCount { get; }

    /// <summary>Most floats one packet can produce (<see cref="MaxSamplesPerChannel"/> * channels).</summary>
    public int MaxSamplesPerPacket => MaxSamplesPerChannel * ChannelCount;

    private CeltDecoderGpu Celt => _celt ??= new CeltDecoderGpu(_accelerator, SampleRateHz, ChannelCount);

    /// <summary>
    /// Decode one packet that is already on the accelerator into a caller-owned view (at least
    /// <see cref="MaxSamplesPerPacket"/> floats), interleaved. Returns the samples per channel written. Only the
    /// packet's header bytes (TOC and frame sizes) and the redundancy ints cross to the host.
    /// </summary>
    public async Task<int> DecodePacketAsync(ArrayView1D<byte, Stride1D.Dense> packet, int packetLength, ArrayView1D<float, Stride1D.Dense> pcmOut)
    {
        if (packetLength <= 0 || packetLength > packet.Length) throw new ArgumentOutOfRangeException(nameof(packetLength));
        if (pcmOut.Length < MaxSamplesPerPacket)
            throw new ArgumentException($"pcmOut must hold {MaxSamplesPerPacket} floats (MaxSamplesPerPacket).", nameof(pcmOut));
        int head = Math.Min(packetLength, HeaderReadback);
        var layoutBytes = new byte[packetLength];
        (await packet.SubView(0, head).CopyToHostAsync()).AsSpan(0, head).CopyTo(layoutBytes);
        var parsed = Parse(layoutBytes);
        if (parsed.PayloadOffset > head)
        {
            // A long padding run: the frame layout needs more than the header readback.
            layoutBytes = await packet.SubView(0, packetLength).CopyToHostAsync();
            parsed = Parse(layoutBytes);
        }
        int samples = await DecodeParsedAsync(parsed, layoutBytes, packet);
        _toFloatKernel(samples * ChannelCount, _pcm16.View, pcmOut);
        return samples;
    }

    /// <summary>Decode one packet. Returns the interleaved float PCM. A terminal CPU sink; in a browser prefer the
    /// TypedArray overload.</summary>
    public async Task<float[]> DecodePacketAsync(ReadOnlyMemory<byte> packet)
    {
        if (packet.Length == 0) throw new ArgumentException("Empty Opus packet.", nameof(packet));
        var bytes = packet.ToArray();
        var parsed = Parse(bytes);
        var dPacket = _dPacket.Ensure(bytes.Length);
        DeviceTransfer.Upload(dPacket, bytes);
        int samples = await DecodeParsedAsync(parsed, bytes, dPacket);
        _toFloatKernel(samples * ChannelCount, _pcm16.View, _dPcm.View);
        return await _dPcm.View.SubView(0, (long)samples * ChannelCount).CopyToHostAsync();
    }

    /// <summary>Decode one packet held in a JS typed array (browser accelerators only); the samples come back as a JS
    /// <see cref="Float32Array"/> the caller disposes. Only the packet's header bytes enter the .NET heap.</summary>
    [SupportedOSPlatform("browser")]
    public async Task<Float32Array> DecodePacketAsync(TypedArray packet)
    {
        ArgumentNullException.ThrowIfNull(packet);
        int length = (int)packet.ByteLength;
        if (length == 0) throw new ArgumentException("Empty Opus packet.", nameof(packet));
        var dPacket = _dPacket.Ensure(length);
        DeviceTransfer.Upload(dPacket, packet);
        int samples = await DecodePacketAsync(dPacket, length, _dPcm.View);
        return await DeviceTransfer.DownloadFloat32ArrayAsync(_dPcm.View, (long)samples * ChannelCount);
    }

    /// <summary>
    /// Conceal <paramref name="samplesPerChannel"/> samples (a multiple of 2.5 ms) of a lost packet (libopus opus_decode
    /// with data = NULL): the previous mode extrapolates. Before the first packet the result is silence.
    /// </summary>
    public async Task<float[]> DecodeLostAsync(int samplesPerChannel)
    {
        int n = await DecodeLostAsync(samplesPerChannel, _dPcm.View);
        return await _dPcm.View.SubView(0, (long)n * ChannelCount).CopyToHostAsync();
    }

    /// <summary>Conceal a lost packet into a caller-owned view (at least <see cref="MaxSamplesPerPacket"/> floats).
    /// Returns the samples per channel written.</summary>
    public async Task<int> DecodeLostAsync(int samplesPerChannel, ArrayView1D<float, Stride1D.Dense> pcmOut)
    {
        if (samplesPerChannel <= 0 || samplesPerChannel > MaxSamplesPerChannel || samplesPerChannel % (SampleRateHz / 400) != 0)
            throw new ArgumentOutOfRangeException(nameof(samplesPerChannel), "A multiple of 2.5 ms, at most 120 ms.");
        if (pcmOut.Length < MaxSamplesPerPacket)
            throw new ArgumentException($"pcmOut must hold {MaxSamplesPerPacket} floats (MaxSamplesPerPacket).", nameof(pcmOut));
        // opus_decode_native with data == NULL.
        int count = 0;
        do
        {
            count += await DecodeFrameAsync(false, default, 0, 0, _pcm16.View, count * ChannelCount, samplesPerChannel - count);
        }
        while (count < samplesPerChannel);
        _toFloatKernel(samplesPerChannel * ChannelCount, _pcm16.View, pcmOut);
        return samplesPerChannel;
    }

    /// <summary>The final range of the last decoded frame (libopus OPUS_GET_FINAL_RANGE): 0 after concealment.</summary>
    public async Task<uint> GetFinalRangeAsync()
    {
        if (_rangeSource == 0) return 0;
        if (_rangeSource == 2) return await Celt.GetFinalRangeAsync();
        _rangeKernel(1, _silk.RangeDecoder, _info.View);
        var r = await _info.View.SubView(4, 1).CopyToHostAsync();
        return (uint)r[0] ^ _redundantRng;
    }

    private static OpusPacket Parse(byte[] bytes)
    {
        if (!OpusPacketParser.TryParse(bytes, selfDelimited: false, out var packet, out var error) || packet is null)
            throw new ArgumentException($"Malformed Opus packet ({error}).");
        return packet;
    }

    // opus_decode_native for a packet with data: the TOC sets the state, then every frame decodes.
    private async Task<int> DecodeParsedAsync(OpusPacket packet, byte[] layoutBytes, ArrayView1D<byte, Stride1D.Dense> dPacket)
    {
        var toc = packet.Toc;
        _mode = toc.Mode switch { OpusMode.Silk => ModeSilk, OpusMode.Hybrid => ModeHybrid, _ => ModeCelt };
        _bandwidth = toc.Bandwidth switch
        {
            OpusBandwidth.Narrowband => 1101, OpusBandwidth.Mediumband => 1102, OpusBandwidth.Wideband => 1103,
            OpusBandwidth.Superwideband => 1104, _ => 1105,
        };
        _frameSize = toc.GetSamplesPerFrame(SampleRateHz);
        _streamChannels = toc.IsStereo ? 2 : 1;
        int nbSamples = 0;
        for (int f = 0; f < packet.FrameCount; f++)
        {
            if (!MemoryMarshal.TryGetArray(packet.Frames[f], out var seg) || !ReferenceEquals(seg.Array, layoutBytes))
                throw new InvalidOperationException("Opus frame slice is not a view of the packet bytes.");
            nbSamples += await DecodeFrameAsync(true, dPacket, seg.Offset, seg.Count, _pcm16.View, nbSamples * ChannelCount,
                MaxSamplesPerChannel - nbSamples);
        }
        return nbSamples;
    }

    // libopus opus_decode_frame (fixed point; decode_fec = 0). Writes int16 PCM at pcm[pcmAt]; returns the samples per
    // channel. hasData false = concealment (data == NULL).
    private async Task<int> DecodeFrameAsync(bool hasData, ArrayView<byte> data, int dataStart, int len, ArrayView<short> pcm, int pcmAt, int frameSize)
    {
        int ch = ChannelCount;
        int f20 = SampleRateHz / 50, f10 = f20 >> 1, f5 = f10 >> 1, f2_5 = f5 >> 1;
        if (frameSize < f2_5) throw new ArgumentOutOfRangeException(nameof(frameSize));
        frameSize = Math.Min(frameSize, SampleRateHz / 25 * 3);
        if (len <= 1)
        {
            hasData = false;
            frameSize = Math.Min(frameSize, _frameSize);
        }
        int audiosize, mode, bandwidth;
        if (hasData)
        {
            audiosize = _frameSize;
            mode = _mode;
            bandwidth = _bandwidth;
        }
        else
        {
            audiosize = frameSize;
            mode = _prevRedundancy ? ModeCelt : _prevMode;
            bandwidth = 0;
            if (mode == ModeNone)
            {
                pcm.SubView(pcmAt, (long)audiosize * ch).MemSetToZero();
                return audiosize;
            }
            if (audiosize > f20)
            {
                do
                {
                    int ret = await DecodeFrameAsync(false, default, 0, 0, pcm, pcmAt, Math.Min(audiosize, f20));
                    pcmAt += ret * ch;
                    audiosize -= ret;
                }
                while (audiosize > 0);
                return frameSize;
            }
            else if (audiosize < f20)
            {
                if (audiosize > f10) audiosize = f10;
                else if (mode != ModeSilk && audiosize > f5 && audiosize < f10) audiosize = f5;
            }
        }

        bool celtAccum = mode != ModeCelt && frameSize >= f10;
        bool transition = hasData && _prevMode > 0 && (
            (mode == ModeCelt && _prevMode != ModeCelt && !_prevRedundancy) || (mode != ModeCelt && _prevMode == ModeCelt));
        if (transition && mode == ModeCelt)
            await DecodeFrameAsync(false, default, 0, 0, _pcmTransition.View, 0, Math.Min(f5, audiosize));
        if (audiosize > frameSize) throw new ArgumentException("PCM buffer too small for the frame.");
        frameSize = audiosize;

        // ---- SILK ----
        if (mode != ModeCelt)
        {
            ArrayView<short> silkOut = celtAccum ? pcm : _pcmSilk.View;
            int silkAt = celtAccum ? pcmAt : 0;
            if (_prevMode == ModeCelt) _silk.ResetDecoder();
            int payloadMs = Math.Max(10, 1000 * audiosize / SampleRateHz);
            if (hasData)
            {
                _silkKHz = mode == ModeSilk ? _bandwidth switch { 1101 => 8, 1102 => 12, _ => 16 } : 16;
                (int silkMs, int silkCount) = payloadMs switch { 10 => (10, 1), 20 => (20, 1), 40 => (20, 2), 60 => (20, 3), _ => (0, 0) };
                if (silkCount == 0) throw new ArgumentException($"SILK frame duration {payloadMs} ms is not an Opus SILK duration.");
                _silk.Decode(data, dataStart, len, _silkKHz, silkMs, silkCount, _streamChannels, silkOut, silkAt);
            }
            else
            {
                // silk_Decode with FLAG_PACKET_LOST: one frame of payloadSize_ms (10 or 20 here).
                _silk.DecodeLost(payloadMs, silkOut, silkAt);
            }
        }

        // ---- redundancy (read from the SILK layer's range decoder) ----
        int startBand = 0;
        bool redundancy = false, celtToSilk = false;
        int redundancyBytes = 0;
        if (mode != ModeCelt && hasData)
        {
            _redundancyKernel(1, data, _silk.RangeDecoder, _info.View, dataStart, len, mode == ModeHybrid ? 1 : 0);
            var d = await _info.View.SubView(0, 4).CopyToHostAsync();
            redundancy = d[0] != 0;
            celtToSilk = d[1] != 0;
            redundancyBytes = d[2];
            len = d[3];
        }
        if (mode != ModeCelt) startBand = 17;
        if (redundancy) transition = false;
        if (transition && mode != ModeCelt)
            await DecodeFrameAsync(false, default, 0, 0, _pcmTransition.View, 0, Math.Min(f5, audiosize));
        if (bandwidth != 0)
            _celtEnd = bandwidth switch { 1101 => 13, 1102 or 1103 => 17, 1104 => 19, _ => 21 };
        int celtChannels = _streamChannels; // CELT_SET_CHANNELS(st->stream_channels)

        _redundantRng = 0;
        if (redundancy && celtToSilk)
        {
            Celt.Decode(data, dataStart + len, redundancyBytes, null, 0, _celtEnd, celtChannels, f5, false, _redundantAudio.View, 0);
            _redundantRng = await Celt.GetFinalRangeAsync();
        }
        _celtStart = startBand;

        if (mode != ModeSilk)
        {
            int celtFrameSize = Math.Min(f20, frameSize);
            if (mode != _prevMode && _prevMode > 0 && !_prevRedundancy) Celt.ResetState();
            if (hasData)
                Celt.Decode(data, dataStart, len, mode == ModeHybrid ? _silk.RangeDecoder : null, startBand, _celtEnd,
                    celtChannels, celtFrameSize, celtAccum, pcm, pcmAt);
            else
                Celt.DecodeLost(startBand, _celtEnd, celtFrameSize, celtAccum, pcm, pcmAt);
        }
        else
        {
            if (!celtAccum) pcm.SubView(pcmAt, (long)frameSize * ch).MemSetToZero();
            // Hybrid -> SILK: the CELT MDCT fades out by decoding a silence frame.
            if (_prevMode == ModeHybrid && !(redundancy && celtToSilk && _prevRedundancy))
            {
                _celtStart = 0;
                Celt.Decode(_silence.View, 0, 2, null, 0, _celtEnd, celtChannels, f2_5, celtAccum, pcm, pcmAt);
            }
        }

        if (mode != ModeCelt && !celtAccum)
            _addKernel(frameSize * ch, pcm, pcmAt, _pcmSilk.View, 0);

        int inc = 48000 / SampleRateHz;
        // SILK -> CELT: the redundant 5 ms CELT frame cross-fades into the end of the frame.
        if (redundancy && !celtToSilk)
        {
            Celt.ResetState();
            Celt.Decode(data, dataStart + len, redundancyBytes, null, 0, _celtEnd, celtChannels, f5, false, _redundantAudio.View, 0);
            _redundantRng = await Celt.GetFinalRangeAsync();
            _fadeKernel(f2_5 * ch, pcm, pcmAt + ch * (frameSize - f2_5), _redundantAudio.View, ch * f2_5, 1, f2_5, ch, _window.View, inc);
        }
        // CELT -> SILK: the redundant frame leads (ignored if the previous frame did not use CELT).
        if (redundancy && celtToSilk && (_prevMode != ModeSilk || _prevRedundancy))
        {
            _copyKernel(ch * f2_5, pcm, pcmAt, _redundantAudio.View, 0);
            _fadeKernel(f2_5 * ch, pcm, pcmAt + ch * f2_5, _redundantAudio.View, ch * f2_5, 0, f2_5, ch, _window.View, inc);
        }
        if (transition)
        {
            if (audiosize >= f5)
            {
                _copyKernel(ch * f2_5, pcm, pcmAt, _pcmTransition.View, 0);
                _fadeKernel(f2_5 * ch, pcm, pcmAt + ch * f2_5, _pcmTransition.View, ch * f2_5, 0, f2_5, ch, _window.View, inc);
            }
            else
            {
                _fadeKernel(f2_5 * ch, pcm, pcmAt, _pcmTransition.View, 0, 0, f2_5, ch, _window.View, inc);
            }
        }

        // st->rangeFinal: 0 for concealment, else the frame's range decoder (^ the redundant frame's).
        _rangeSource = len <= 1 || !hasData ? 0 : mode == ModeCelt ? 2 : 1;
        _prevMode = mode;
        _prevRedundancy = redundancy && !celtToSilk;
        return audiosize;
    }

    // ---- kernels ----

    // The redundancy flags after the SILK layer (on its range decoder, storage = the frame's len): info[0] redundancy,
    // [1] celt_to_silk, [2] redundancy_bytes, [3] the frame's new len.
    private static void RedundancyKernel(Index1D _, ArrayView<byte> data, ArrayView<OpusRangeDecoderGpuState> ecBuf, ArrayView<int> info,
        int start, int len, int hybrid)
    {
        var ec = ecBuf[0];
        uint storage = (uint)len;
        int redundancy = 0, celtToSilk = 0, bytes = 0;
        if (Tell(ec) + 17 + 20 * hybrid <= 8 * len)
        {
            redundancy = hybrid != 0 ? OpusRangeDecoderGpu.DecodeBitLogP(ref ec, data, start, storage, 12) : 1;
            if (redundancy != 0)
            {
                celtToSilk = OpusRangeDecoderGpu.DecodeBitLogP(ref ec, data, start, storage, 1);
                bytes = hybrid != 0 ? (int)OpusRangeDecoderGpu.DecodeUint(ref ec, data, start, storage, 256) + 2 : len - ((Tell(ec) + 7) >> 3);
                len -= bytes;
                if (len * 8 < Tell(ec))
                {
                    len = 0;
                    bytes = 0;
                    redundancy = 0;
                }
            }
        }
        info[0] = redundancy;
        info[1] = celtToSilk;
        info[2] = bytes;
        info[3] = len;
        ecBuf[0] = ec;
    }

    // ec_tell: whole bits used.
    private static int Tell(OpusRangeDecoderGpuState ec)
    {
        int l = 0;
        uint r = ec.Rng;
        while (r != 0) { r >>= 1; l++; }
        return ec.NBitsTotal - l;
    }

    // libopus smooth_fade, out = pcm[pcmAt]: the pcm side is in2 (pcmIsIn1 0) or in1 (1); the other input is other[otherAt].
    private static void FadeKernel(Index1D k, ArrayView<short> pcm, int pcmAt, ArrayView<short> other, int otherAt, int pcmIsIn1,
        int overlap, int channels, ArrayView<short> window, int inc)
    {
        int i = k / channels;
        int w0 = window[i * inc];
        int w = (w0 * w0) >> 15;                                   // MULT16_16_Q15(window, window)
        int p = pcm[pcmAt + k], o = other[otherAt + k];
        int in1 = pcmIsIn1 != 0 ? p : o, in2 = pcmIsIn1 != 0 ? o : p;
        pcm[pcmAt + k] = (short)((w * in2 + (32767 - w) * in1) >> 15); // SHR32(MAC16_16(MULT16_16(w, in2), Q15ONE - w, in1), 15)
    }

    // pcm[i] = SAT16(ADD32(pcm[i], pcm_silk[i])).
    private static void AddKernel(Index1D i, ArrayView<short> pcm, int pcmAt, ArrayView<short> silk, int silkAt)
    {
        int v = pcm[pcmAt + i] + silk[silkAt + i];
        pcm[pcmAt + i] = (short)(v > 32767 ? 32767 : v < -32768 ? -32768 : v);
    }

    private static void CopyKernel(Index1D i, ArrayView<short> dst, int dstAt, ArrayView<short> src, int srcAt)
        => dst[dstAt + i] = src[srcAt + i];

    // The fixed-point decoder's float API: int16 / 32768.
    private static void ToFloatKernel(Index1D i, ArrayView<short> pcm, ArrayView<float> output)
        => output[i] = pcm[i] * (1.0f / 32768.0f);

    private static void RangeKernel(Index1D _, ArrayView<OpusRangeDecoderGpuState> ec, ArrayView<int> info)
        => info[4] = (int)ec[0].Rng;

    /// <inheritdoc/>
    public void Dispose()
    {
        _silk.Dispose();
        _celt?.Dispose();
        _dPacket.Dispose();
        _dPcm.Dispose();
        _pcm16.Dispose();
        _pcmSilk.Dispose();
        _pcmTransition.Dispose();
        _redundantAudio.Dispose();
        _window.Dispose();
        _silence.Dispose();
        _info.Dispose();
    }
}
