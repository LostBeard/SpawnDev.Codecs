// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// Ogg Opus encoding: PCM -> Opus packets (OpusEncoderGpu, on the accelerator) -> an Ogg Opus file (OggOpusWriter, RFC 7845).
//
// The stream layout matches what the ffmpeg binary (8.1, libopus) writes for the same input, as measured by running it:
//   * application Audio, unconstrained VBR, 20 ms packets; the default bitrate is 64000 b/s mono, 96000 b/s stereo;
//   * pre-skip = the encoder's lookahead (OPUS_GET_LOOKAHEAD) in 48 kHz samples; OpusHead's input rate = the source rate;
//   * ceil((samples + lookahead) / frame) packets (samples, lookahead and frame at the source rate): the input is followed
//     by zeros so the decoder's output covers every input sample after the pre-skip; an empty input gives the two header
//     pages only;
//   * packet k starts at 48 kHz pts -preSkip + 960 k; the last packet ends at the final granule, preSkip + the input length
//     at 48 kHz (RFC 7845 section 4.4).
// The packets themselves are this library's libopus-compatible CELT-only encode, not FFmpeg's (a different libopus build).
//
// Host role (pure coordinator): it uploads the input once, launches the float -> int16 conversion (PcmConvertKernels) and
// the zero padding on the accelerator, dispatches one OpusEncoderGpu packet per 20 ms (each reads its PCM straight from
// the device input, the last one or two from a small zero-padded tail buffer), gathers the packets in a device arena and
// reads them back once per second of audio, then frames them into Ogg pages (a container writer, allowed on the host).

using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Container.Ogg;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnDev.Codecs.Audio.Opus;

/// <summary>Settings of <see cref="OggOpusEncoder"/>.</summary>
public sealed record OggOpusEncodeOptions
{
    /// <summary>Bits per second (OPUS_SET_BITRATE); null = 64000 for mono, 96000 for stereo (FFmpeg's libopus default).</summary>
    public int? Bitrate { get; init; }

    /// <summary>Complexity 0..9 (OPUS_SET_COMPLEXITY).</summary>
    public int Complexity { get; init; } = 9;

    /// <summary>Variable bitrate (OPUS_SET_VBR).</summary>
    public bool Vbr { get; init; } = true;

    /// <summary>Constrained VBR (OPUS_SET_VBR_CONSTRAINT); off, as FFmpeg's libopus wrapper sets it by default.</summary>
    public bool VbrConstraint { get; init; }

    /// <summary>The encoder application (OPUS_SET_APPLICATION).</summary>
    public OpusEncoderApplication Application { get; init; } = OpusEncoderApplication.Audio;

    /// <summary>Expected packet loss in percent, 0..100 (OPUS_SET_PACKET_LOSS_PERC).</summary>
    public int PacketLossPercent { get; init; }

    /// <summary>OPUS_SET_PHASE_INVERSION_DISABLED.</summary>
    public bool PhaseInversionDisabled { get; init; }

    /// <summary>The OpusTags vendor string.</summary>
    public string Vendor { get; init; } = "ffmpeg";

    /// <summary>User comments, written in order as "KEY=value"; null for none.</summary>
    public IReadOnlyList<KeyValuePair<string, string>>? Tags { get; init; }

    /// <summary>The Ogg logical bitstream serial number.</summary>
    public uint Serial { get; init; }

    /// <summary>The input sample rate written to OpusHead; null = the rate of the PCM being encoded.</summary>
    public int? InputSampleRate { get; init; }
}

/// <summary>
/// Encodes PCM to an Ogg Opus file with <see cref="OpusEncoderGpu"/> (on the accelerator) and <see cref="OggOpusWriter"/>.
/// Input is interleaved PCM at 8, 12, 16, 24 or 48 kHz, 1 or 2 channels: int16, or float (full scale +-1.0, converted
/// on the accelerator as <see cref="PcmConvertKernels.FloatToS16Kernel"/> describes). Each input has a managed, a JS
/// typed array (browser accelerators) and an <see cref="ArrayView1D{T, TStride}"/> overload; each returns the file, or
/// writes it to a caller's <see cref="Stream"/>.
/// </summary>
/// <summary>One Opus packet: its bytes, 48 kHz decode time (the first is -PreSkip) and duration.</summary>
public sealed record EncodedOpusPacket(byte[] Data, long Pts, int Duration);

/// <summary>An encoded Opus stream for a container: the OpusHead (RFC 7845), the pre-skip and the packets.</summary>
public sealed record OpusPacketStream(byte[] OpusHead, int PreSkip, int Channels, IReadOnlyList<EncodedOpusPacket> Packets);

public static class OggOpusEncoder
{
    /// <summary>Packets gathered on the accelerator per readback (one second of 20 ms packets).</summary>
    private const int PacketsPerReadback = 50;
    /// <summary>Samples converted per float -> int16 launch (keeps each launch bounded).</summary>
    private const int ConvertChunk = 1 << 20;

    // ---- int16, managed ----

    /// <summary>Encode interleaved int16 PCM; returns the Ogg Opus file.</summary>
    public static async Task<byte[]> EncodeAsync(Accelerator accelerator, ReadOnlyMemory<short> pcm, int sampleRate, int channels,
        OggOpusEncodeOptions? options = null)
    {
        using var ms = new MemoryStream();
        await EncodeAsync(accelerator, pcm, sampleRate, channels, ms, options);
        return ms.ToArray();
    }

    /// <summary>Encode interleaved int16 PCM into <paramref name="output"/> (left open).</summary>
    public static async Task EncodeAsync(Accelerator accelerator, ReadOnlyMemory<short> pcm, int sampleRate, int channels, Stream output,
        OggOpusEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        using var d = accelerator.Allocate1D<short>(Math.Max(1, pcm.Length));
        Upload(d.View, pcm.Span);
        await EncodeAsync(accelerator, d.View.SubView(0, pcm.Length), sampleRate, channels, output, options);
    }

    // ---- float, managed ----

    /// <summary>Encode interleaved float PCM; returns the Ogg Opus file.</summary>
    public static async Task<byte[]> EncodeAsync(Accelerator accelerator, ReadOnlyMemory<float> pcm, int sampleRate, int channels,
        OggOpusEncodeOptions? options = null)
    {
        using var ms = new MemoryStream();
        await EncodeAsync(accelerator, pcm, sampleRate, channels, ms, options);
        return ms.ToArray();
    }

    /// <summary>Encode interleaved float PCM into <paramref name="output"/> (left open).</summary>
    public static async Task EncodeAsync(Accelerator accelerator, ReadOnlyMemory<float> pcm, int sampleRate, int channels, Stream output,
        OggOpusEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        using var d = accelerator.Allocate1D<float>(Math.Max(1, pcm.Length));
        Upload(d.View, pcm.Span);
        await EncodeAsync(accelerator, d.View.SubView(0, pcm.Length), sampleRate, channels, output, options);
    }

    // ---- JS typed arrays (browser accelerators) ----

    /// <summary>Encode interleaved int16 PCM held in a JS <see cref="Int16Array"/> (browser accelerators); the samples go
    /// JS -> GPU without entering the .NET heap. Returns the Ogg Opus file.</summary>
    [SupportedOSPlatform("browser")]
    public static async Task<byte[]> EncodeAsync(Accelerator accelerator, Int16Array pcm, int sampleRate, int channels,
        OggOpusEncodeOptions? options = null)
    {
        using var ms = new MemoryStream();
        await EncodeAsync(accelerator, pcm, sampleRate, channels, ms, options);
        return ms.ToArray();
    }

    /// <summary>Encode interleaved int16 PCM held in a JS <see cref="Int16Array"/> into <paramref name="output"/> (left open).</summary>
    [SupportedOSPlatform("browser")]
    public static async Task EncodeAsync(Accelerator accelerator, Int16Array pcm, int sampleRate, int channels, Stream output,
        OggOpusEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        ArgumentNullException.ThrowIfNull(pcm);
        long n = (long)pcm.ByteLength / sizeof(short);
        using var d = accelerator.Allocate1D<short>(Math.Max(1, n));
        if (n > 0) DeviceTransfer.Upload(d.View.SubView(0, n), (TypedArray)pcm);
        await EncodeAsync(accelerator, d.View.SubView(0, n), sampleRate, channels, output, options);
    }

    /// <summary>Encode interleaved float PCM held in a JS <see cref="Float32Array"/> (browser accelerators); the samples go
    /// JS -> GPU without entering the .NET heap. Returns the Ogg Opus file.</summary>
    [SupportedOSPlatform("browser")]
    public static async Task<byte[]> EncodeAsync(Accelerator accelerator, Float32Array pcm, int sampleRate, int channels,
        OggOpusEncodeOptions? options = null)
    {
        using var ms = new MemoryStream();
        await EncodeAsync(accelerator, pcm, sampleRate, channels, ms, options);
        return ms.ToArray();
    }

    /// <summary>Encode interleaved float PCM held in a JS <see cref="Float32Array"/> into <paramref name="output"/> (left open).</summary>
    [SupportedOSPlatform("browser")]
    public static async Task EncodeAsync(Accelerator accelerator, Float32Array pcm, int sampleRate, int channels, Stream output,
        OggOpusEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        ArgumentNullException.ThrowIfNull(pcm);
        long n = (long)pcm.ByteLength / sizeof(float);
        using var d = accelerator.Allocate1D<float>(Math.Max(1, n));
        if (n > 0) DeviceTransfer.Upload(d.View.SubView(0, n), (TypedArray)pcm);
        await EncodeAsync(accelerator, d.View.SubView(0, n), sampleRate, channels, output, options);
    }

    // ---- on the accelerator ----

    /// <summary>Encode interleaved float PCM already on the accelerator; returns the Ogg Opus file.</summary>
    public static async Task<byte[]> EncodeAsync(Accelerator accelerator, ArrayView1D<float, Stride1D.Dense> pcm, int sampleRate, int channels,
        OggOpusEncodeOptions? options = null)
    {
        using var ms = new MemoryStream();
        await EncodeAsync(accelerator, pcm, sampleRate, channels, ms, options);
        return ms.ToArray();
    }

    /// <summary>Encode interleaved float PCM already on the accelerator into <paramref name="output"/> (left open). The
    /// floats are converted to int16 on the accelerator.</summary>
    public static async Task EncodeAsync(Accelerator accelerator, ArrayView1D<float, Stride1D.Dense> pcm, int sampleRate, int channels, Stream output,
        OggOpusEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        long n = pcm.Length;
        using var s16 = accelerator.Allocate1D<short>(Math.Max(1, n));
        if (n > 0)
        {
            var convert = await OpusEncoderGpu.GetFloatToS16Async(accelerator);   // the accelerator's shared set (kept alive)
            var pacer = new LaunchPacer(accelerator);
            for (long at = 0; at < n; at += ConvertChunk)
            {
                int count = (int)Math.Min(ConvertChunk, n - at);
                convert(count, pcm.SubView(at, count), s16.View.SubView(at, count));
                await pacer.LaunchedAsync();
            }
        }
        await EncodeAsync(accelerator, s16.View.SubView(0, n), sampleRate, channels, output, options);
    }

    /// <summary>Encode interleaved int16 PCM already on the accelerator; returns the Ogg Opus file.</summary>
    public static async Task<byte[]> EncodeAsync(Accelerator accelerator, ArrayView1D<short, Stride1D.Dense> pcm, int sampleRate, int channels,
        OggOpusEncodeOptions? options = null)
    {
        using var ms = new MemoryStream();
        await EncodeAsync(accelerator, pcm, sampleRate, channels, ms, options);
        return ms.ToArray();
    }

    /// <summary>
    /// Encode interleaved int16 PCM already on the accelerator into <paramref name="output"/> (left open). Every other
    /// overload funnels into this one. Only the packets come back from the accelerator.
    /// </summary>
    public static async Task EncodeAsync(Accelerator accelerator, ArrayView1D<short, Stride1D.Dense> pcm, int sampleRate, int channels, Stream output,
        OggOpusEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        ArgumentNullException.ThrowIfNull(output);
        options ??= new OggOpusEncodeOptions();
        OggOpusWriter? writer = null;
        try
        {
            await EncodeCoreAsync(accelerator, pcm, sampleRate, channels, options,
                preSkip => writer = new OggOpusWriter(output, new OggOpusWriterOptions
                {
                    Channels = channels,
                    PreSkip = preSkip,
                    InputSampleRate = options.InputSampleRate ?? sampleRate,
                    Vendor = options.Vendor,
                    Tags = options.Tags,
                    Serial = options.Serial,
                }, leaveOpen: true),
                (packet, pts, duration) => writer!.WritePacket(packet, pts, duration));
            writer?.Finish();
        }
        finally { writer?.Dispose(); }
    }

    /// <summary>
    /// Encode interleaved int16 PCM already on the accelerator into raw Opus packets (20 ms each) plus the OpusHead and
    /// pre-skip a container needs (MP4 / WebM): the same packets <see cref="EncodeAsync(Accelerator, ArrayView1D{short, Stride1D.Dense}, int, int, Stream, OggOpusEncodeOptions?)"/>
    /// puts in its Ogg pages. Packet k decodes at 48 kHz time <c>-PreSkip + 960 k</c>.
    /// </summary>
    public static async Task<OpusPacketStream> EncodePacketsAsync(Accelerator accelerator, ArrayView1D<short, Stride1D.Dense> pcm, int sampleRate, int channels,
        OggOpusEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        options ??= new OggOpusEncodeOptions();
        int skip = 0;
        var packets = new List<EncodedOpusPacket>();
        await EncodeCoreAsync(accelerator, pcm, sampleRate, channels, options, preSkip => skip = preSkip,
            (packet, pts, duration) => packets.Add(new EncodedOpusPacket(packet.ToArray(), pts, duration)));
        var head = OggOpusWriter.BuildOpusHead(new OggOpusWriterOptions { Channels = channels, PreSkip = skip, InputSampleRate = options.InputSampleRate ?? sampleRate });
        return new OpusPacketStream(head, skip, channels, packets);
    }

    /// <summary>As <see cref="EncodePacketsAsync(Accelerator, ArrayView1D{short, Stride1D.Dense}, int, int, OggOpusEncodeOptions?)"/> for interleaved float PCM (managed).</summary>
    public static async Task<OpusPacketStream> EncodePacketsAsync(Accelerator accelerator, ReadOnlyMemory<float> pcm, int sampleRate, int channels,
        OggOpusEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        long n = pcm.Length;
        using var f = accelerator.Allocate1D<float>(Math.Max(1, n));
        Upload(f.View, pcm.Span);
        using var s16 = accelerator.Allocate1D<short>(Math.Max(1, n));
        if (n > 0)
        {
            var convert = await OpusEncoderGpu.GetFloatToS16Async(accelerator);
            var pacer = new LaunchPacer(accelerator);
            for (long at = 0; at < n; at += ConvertChunk)
            {
                int count = (int)Math.Min(ConvertChunk, n - at);
                convert(count, f.View.SubView(at, count), s16.View.SubView(at, count));
                await pacer.LaunchedAsync();
            }
        }
        return await EncodePacketsAsync(accelerator, s16.View.SubView(0, n), sampleRate, channels, options);
    }

    /// <summary>
    /// The encode every overload shares: <paramref name="begin"/> gets the pre-skip (48 kHz samples) before the first
    /// packet, then <paramref name="packet"/> gets each packet with its 48 kHz decode time and duration.
    /// </summary>
    private static async Task EncodeCoreAsync(Accelerator accelerator, ArrayView1D<short, Stride1D.Dense> pcm, int sampleRate, int channels,
        OggOpusEncodeOptions options, Action<int> begin, PacketSink packet)
    {
        if (channels is not (1 or 2)) throw new ArgumentOutOfRangeException(nameof(channels), "1 or 2 channels.");
        if (sampleRate is not (8000 or 12000 or 16000 or 24000 or 48000))
            throw new ArgumentOutOfRangeException(nameof(sampleRate), "Opus encodes 8, 12, 16, 24 or 48 kHz.");
        if (pcm.Length % channels != 0) throw new ArgumentException("The sample count is not a multiple of the channel count.", nameof(pcm));

        using var encoder = new OpusEncoderGpu(accelerator, sampleRate, channels, options.Application)
        {
            Bitrate = options.Bitrate ?? (channels == 1 ? 64000 : 96000),
            Complexity = options.Complexity,
            Vbr = options.Vbr,
            VbrConstraint = options.VbrConstraint,
            PacketLossPercent = options.PacketLossPercent,
            PhaseInversionDisabled = options.PhaseInversionDisabled,
        };
        int scale = 48000 / sampleRate;
        int frame = sampleRate / 50;                                            // 20 ms per channel at the source rate
        int frameTicks = frame * scale;                                         // 960
        int lookahead = encoder.Lookahead;
        int preSkip = lookahead * scale;
        long samples = pcm.Length / channels;
        long packets = samples == 0 ? 0 : (samples + lookahead + frame - 1) / frame;
        long finalGranule = samples * scale + preSkip;

        begin(preSkip);
        if (packets == 0) return;

        // Frames wholly inside the input are encoded from it in place; the rest (the input's last partial frame and the
        // lookahead) from a zeroed tail buffer that starts with the input's leftover samples.
        long whole = samples / frame;
        long frameValues = (long)frame * channels;
        long leftover = (samples - whole * frame) * channels;
        using var tail = accelerator.Allocate1D<short>((packets - whole) * frameValues);
        tail.MemSetToZero();
        if (leftover > 0) tail.View.SubView(0, leftover).CopyFrom(pcm.SubView(whole * frameValues, leftover));

        int slot = OpusEncoderGpu.MaxPacketBytes;
        using var arena = accelerator.Allocate1D<byte>((long)PacketsPerReadback * slot);
        var lengths = new int[PacketsPerReadback];
        long first = 0;                                                         // the first packet in the arena
        int used = 0, bytes = 0;
        for (long k = 0; k < packets; k++)
        {
            var input = k < whole ? pcm.SubView(k * frameValues, frameValues) : tail.View.SubView((k - whole) * frameValues, frameValues);
            lengths[used] = await encoder.EncodePacketAsync(input, frame, arena.View.SubView(bytes, slot));
            bytes += lengths[used++];
            if (used == PacketsPerReadback || k == packets - 1)
            {
                var data = await DeviceTransfer.DownloadAsync(arena.View, bytes);
                int at = 0;
                for (int i = 0; i < used; i++)
                {
                    long index = first + i;
                    long pts = -preSkip + index * frameTicks;
                    int duration = index < packets - 1 ? frameTicks : (int)(finalGranule - preSkip - pts);
                    packet(data.AsSpan(at, lengths[i]), pts, duration);
                    at += lengths[i];
                }
                first += used;
                used = 0;
                bytes = 0;
            }
        }
    }

    private delegate void PacketSink(ReadOnlySpan<byte> packet, long pts, int duration);

    /// <summary>
    /// Compile and load every kernel the encode uses (encodes 50 ms of stereo silence from int16 and from float), so the
    /// first real encode does not wait for them. On browser backends the first shader compile can take a long time; call
    /// this early.
    /// </summary>
    public static async Task WarmUpAsync(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        await EncodeAsync(accelerator, new short[2400 * 2], 48000, 2);
        await EncodeAsync(accelerator, new float[2400 * 2], 48000, 2);
    }

    private static void Upload<T>(ArrayView1D<T, Stride1D.Dense> dest, ReadOnlySpan<T> src) where T : unmanaged
    {
        if (src.Length == 0) return;
        dest.SubView(0, src.Length).CopyFromCPU(ref MemoryMarshal.GetReference(src), src.Length);
    }
}
