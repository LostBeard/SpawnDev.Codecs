// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Opus decoder on the accelerator (RFC 6716; libopus opus_decode). The host parses the packet's TOC byte and frame
// packing (codes 0-3: container framing, the same metadata a demuxer reads) and dispatches each frame to the mode's
// GPU decoder; every bit of the frames themselves is decoded on the accelerator and the PCM is written there.
//
// Modes: SILK-only frames run through SilkPacketDecoderGpu. CELT-only and Hybrid frames need the GPU CELT decoder
// (CeltDecoderGpu, in progress) and the hybrid glue (SILK then CELT on one range decoder, the redundancy frames and
// the mode-transition fades); until they land those packets are refused with NotSupportedException, never decoded
// differently.

using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Opus.Silk;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Audio.Opus;

/// <summary>
/// Opus decoder whose decode runs on the accelerator. One instance per stream (it keeps the inter-frame state on the
/// accelerator). Output is interleaved float PCM at <see cref="SampleRateHz"/> with <see cref="ChannelCount"/>
/// channels, the same samples as the CPU <c>OpusDecoder</c> in SpawnDev.Codecs.References.
/// </summary>
public sealed class OpusDecoderGpu : IDisposable
{
    /// <summary>Most samples per channel one packet can produce (120 ms at 48 kHz).</summary>
    public const int MaxSamplesPerChannel = 5760;

    // Packet bytes read back for the frame layout of a device-resident packet: TOC, count byte, frame sizes, a short
    // padding run. A longer padding run reads the whole packet back.
    private const int HeaderReadback = 2 + 2 * OpusPacketParser.MaxFramesPerPacket + 8;

    private readonly Accelerator _accelerator;
    private readonly SilkPacketDecoderGpu _silk;
    private readonly StageBuffer<byte> _dPacket;
    private readonly MemoryBuffer1D<float, Stride1D.Dense> _dPcm;
    private OpusMode? _prevMode; // libopus st->prev_mode (null before the first packet)

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
    }

    /// <summary>Output sample rate in Hz.</summary>
    public int SampleRateHz { get; }

    /// <summary>Output channels.</summary>
    public int ChannelCount { get; }

    /// <summary>Most floats one packet can produce (<see cref="MaxSamplesPerChannel"/> * channels).</summary>
    public int MaxSamplesPerPacket => MaxSamplesPerChannel * ChannelCount;

    /// <summary>
    /// Decode one packet that is already on the accelerator into a caller-owned view (at least
    /// <see cref="MaxSamplesPerPacket"/> floats), interleaved. Returns the samples per channel written. Only the
    /// packet's header bytes (TOC and frame sizes) cross to the host.
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
        return Queue(parsed, layoutBytes, packet, pcmOut);
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
        int samples = Queue(parsed, bytes, dPacket, _dPcm.View);
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
    /// Conceal <paramref name="samplesPerChannel"/> samples of a lost packet (libopus opus_decode with data = NULL):
    /// the previous packet's mode extrapolates. Before the first packet the result is silence. Returns the PCM.
    /// </summary>
    public async Task<float[]> DecodeLostAsync(int samplesPerChannel)
    {
        int n = await DecodeLostAsync(samplesPerChannel, _dPcm.View);
        return await _dPcm.View.SubView(0, (long)n * ChannelCount).CopyToHostAsync();
    }

    /// <summary>Conceal a lost packet into a caller-owned view (at least <see cref="MaxSamplesPerPacket"/> floats).
    /// Returns the samples per channel written.</summary>
    public Task<int> DecodeLostAsync(int samplesPerChannel, ArrayView1D<float, Stride1D.Dense> pcmOut)
    {
        if (samplesPerChannel <= 0 || samplesPerChannel > MaxSamplesPerChannel) throw new ArgumentOutOfRangeException(nameof(samplesPerChannel));
        if (pcmOut.Length < MaxSamplesPerPacket)
            throw new ArgumentException($"pcmOut must hold {MaxSamplesPerPacket} floats (MaxSamplesPerPacket).", nameof(pcmOut));
        QueueLost(samplesPerChannel, pcmOut, 0);
        return Task.FromResult(samplesPerChannel);
    }

    // libopus opus_decode_native / opus_decode_frame with data == NULL: the previous mode conceals in pieces of at most
    // 20 ms (a remainder over 10 ms is cut to 10 ms; SILK conceals whole 10 / 20 ms frames).
    private void QueueLost(int samples, ArrayView1D<float, Stride1D.Dense> pcmOut, long pcmOutBase)
    {
        if (_prevMode is null)
        {
            pcmOut.SubView(pcmOutBase, (long)samples * ChannelCount).MemSetToZero();
            return;
        }
        if (_prevMode != OpusMode.Silk)
            throw new NotSupportedException($"Concealing after an Opus {_prevMode} frame needs the GPU CELT decoder (in progress).");
        int f20 = SampleRateHz / 50, f10 = f20 / 2;
        int done = 0;
        while (done < samples)
        {
            int audiosize = Math.Min(samples - done, f20);
            if (audiosize < f20) audiosize = audiosize > f10 ? f10 : audiosize;
            if (audiosize != f20 && audiosize != f10)
                throw new NotSupportedException($"SILK concealment of {audiosize} samples (not 10 or 20 ms).");
            _silk.DecodeLost(audiosize == f20 ? 20 : 10, pcmOut, pcmOutBase + (long)done * ChannelCount);
            done += audiosize;
        }
    }

    private static OpusPacket Parse(byte[] bytes)
    {
        if (!OpusPacketParser.TryParse(bytes, selfDelimited: false, out var packet, out var error) || packet is null)
            throw new ArgumentException($"Malformed Opus packet ({error}).");
        return packet;
    }

    // Dispatch every frame of the packet; returns the samples per channel.
    private int Queue(OpusPacket packet, byte[] layoutBytes, ArrayView1D<byte, Stride1D.Dense> dPacket, ArrayView1D<float, Stride1D.Dense> pcmOut)
    {
        var toc = packet.Toc;
        int samplesPerFrame = toc.GetSamplesPerFrame(SampleRateHz);
        if (toc.Mode != OpusMode.Silk)
            throw new NotSupportedException($"Opus {toc.Mode} frames are not decoded on the accelerator yet (the GPU CELT decoder is in progress).");
        int internalKHz = toc.Bandwidth switch
        {
            OpusBandwidth.Narrowband => 8,
            OpusBandwidth.Mediumband => 12,
            _ => 16,
        };
        int ms = toc.GetSamplesPerFrame(48_000) / 48;
        (int silkMs, int silkCount) = ms switch
        {
            10 => (10, 1),
            20 => (20, 1),
            40 => (20, 2),
            60 => (20, 3),
            _ => throw new ArgumentException($"SILK frame duration {ms} ms is not an Opus SILK duration."),
        };
        int streamChannels = toc.IsStereo ? 2 : 1;
        for (int f = 0; f < packet.FrameCount; f++)
        {
            if (!MemoryMarshal.TryGetArray(packet.Frames[f], out var seg) || !ReferenceEquals(seg.Array, layoutBytes))
                throw new InvalidOperationException("Opus frame slice is not a view of the packet bytes.");
            if (seg.Count == 0)
            {
                // A frame of 0 bytes (DTX: a TOC-only packet) runs the concealment for the frame's duration.
                QueueLost(samplesPerFrame, pcmOut, (long)f * samplesPerFrame * ChannelCount);
                continue;
            }
            _silk.Decode(dPacket, seg.Offset, seg.Count, internalKHz, silkMs, silkCount, streamChannels,
                pcmOut, (long)f * samplesPerFrame * ChannelCount);
            _prevMode = OpusMode.Silk;
        }
        return samplesPerFrame * packet.FrameCount;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _silk.Dispose();
        _dPacket.Dispose();
        _dPcm.Dispose();
    }
}
