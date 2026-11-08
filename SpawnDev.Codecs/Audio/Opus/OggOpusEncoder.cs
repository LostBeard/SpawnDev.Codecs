// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt). See NOTICE.md for upstream attributions.
//
// Ogg Opus file encoding (an .opus file from PCM), driven the way FFmpeg's libopus wrapper (libavcodec/libopusenc.c)
// drives libopus and muxed by OggOpusWriter (a port of FFmpeg's ogg muxer): what PyAV / ffmpeg write, ComfyUI's
// Save Audio (Opus) among them.
//   * application audio, VBR unconstrained, 20 ms frames, mapping family 0 (1 or 2 channels);
//   * the bitrate FFmpeg picks when none is given: 64 kbps per stream + 32 kbps per coupled stream (mono 64k, stereo 96k);
//   * the last frame zero-padded and silent frames sent until the encoder's lookahead has come out, so the file holds
//     ceil((samples + lookahead) / frame) packets; pre-skip = the lookahead at 48 kHz; the final granule = the input
//     length at 48 kHz + pre-skip;
//   * float input is converted to int16 on the accelerator as libswresample converts it (lrintf(x * 32768), clipped):
//     PyAV hands libopus int16 (s16 is the encoder's first sample format).
//
// The container is byte-identical with FFmpeg's for the same packets. The packets are OpusEncoderGpu's: libopus 1.5.2's
// FIXED-point CELT path (complexity up to 9), while FFmpeg builds use libopus' FLOAT build (which also runs the tonality
// analysis from complexity 7): the files decode the same way but their packets are not libopus' bytes.
//
// Host work: one upload of the PCM, one kernel dispatch chain per packet, one readback per packet (the Opus packet).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Container.Ogg;

namespace SpawnDev.Codecs.Audio.Opus;

/// <summary>Settings of <see cref="OggOpusEncoder"/>.</summary>
public sealed record OggOpusEncodeOptions
{
    /// <summary>Bits per second, or null for FFmpeg's default (64000 per channel pair stream: mono 64k, stereo 96k).
    /// ComfyUI sets 64000, 96000, 128000 (its default), 192000 or 320000.</summary>
    public int? Bitrate { get; init; }

    /// <summary>Encoder complexity 0..9 (libopus OPUS_SET_COMPLEXITY; FFmpeg's default is 10, not supported).</summary>
    public int Complexity { get; init; } = 9;

    /// <summary>Variable bitrate (FFmpeg's vbr=on: VBR, unconstrained). False = constant bitrate.</summary>
    public bool Vbr { get; init; } = true;

    /// <summary>Constrained VBR (FFmpeg's vbr=constrained). Default false, as FFmpeg.</summary>
    public bool VbrConstraint { get; init; }

    /// <summary>libopus application (FFmpeg's default: audio).</summary>
    public OpusEncoderApplication Application { get; init; } = OpusEncoderApplication.Audio;

    /// <summary>OPUS_SET_PACKET_LOSS_PERC (FFmpeg's default 0).</summary>
    public int PacketLossPercent { get; init; }

    /// <summary>No stereo phase inversion (FFmpeg's apply_phase_inv=0). Default false, as FFmpeg.</summary>
    public bool PhaseInversionDisabled { get; init; }

    /// <summary>OpusTags vendor (FFmpeg: "ffmpeg" in bit-exact mode, else its libavformat ident, e.g. "Lavf62.12.102").</summary>
    public string Vendor { get; init; } = "ffmpeg";

    /// <summary>OpusTags comment fields in order (ComfyUI: prompt, workflow, then encoder=&lt;vendor&gt;).</summary>
    public IReadOnlyList<KeyValuePair<string, string>>? Tags { get; init; }

    /// <summary>Ogg stream serial number (FFmpeg: 0 in bit-exact mode, random otherwise).</summary>
    public uint Serial { get; init; }

    /// <summary>OpusHead input sample rate, or null for the encoded rate (FFmpeg writes the encoder's input rate).</summary>
    public int? InputSampleRate { get; init; }
}

/// <summary>Encodes PCM to an Ogg Opus file (.opus) with <see cref="OpusEncoderGpu"/> and <see cref="OggOpusWriter"/>.</summary>
public static class OggOpusEncoder
{
    private const int MaxPacketBytesPerStream = 1275 * 3 + 7;   // libopusenc.c's packet allocation

    /// <summary>Encode interleaved int16 PCM (<paramref name="channels"/> 1 or 2 at 8000, 12000, 16000, 24000 or 48000 Hz)
    /// to an Ogg Opus file; returns the file.</summary>
    public static async Task<byte[]> EncodeAsync(Accelerator accelerator, ReadOnlyMemory<short> pcm, int sampleRate, int channels,
        OggOpusEncodeOptions? options = null)
    {
        var ms = new MemoryStream();
        await EncodeAsync(accelerator, pcm, sampleRate, channels, ms, options);
        return ms.ToArray();
    }

    /// <summary>Encode interleaved int16 PCM to an Ogg Opus file written to <paramref name="output"/> (left open).</summary>
    public static Task EncodeAsync(Accelerator accelerator, ReadOnlyMemory<short> pcm, int sampleRate, int channels, Stream output,
        OggOpusEncodeOptions? options = null)
    {
        Validate(pcm.Length, sampleRate, channels);
        return EncodeCoreAsync(accelerator, pcm.Length / channels, sampleRate, channels, output, options,
            dest => dest.SubView(0, pcm.Length).CopyFromCPU(pcm.ToArray()));
    }

    /// <summary>Encode interleaved float PCM (full scale +-1) to an Ogg Opus file; returns the file. The samples become
    /// int16 on the accelerator as libswresample makes them (lrintf(x * 32768) clipped), the conversion PyAV applies
    /// before libopus.</summary>
    public static async Task<byte[]> EncodeAsync(Accelerator accelerator, ReadOnlyMemory<float> pcm, int sampleRate, int channels,
        OggOpusEncodeOptions? options = null)
    {
        var ms = new MemoryStream();
        await EncodeAsync(accelerator, pcm, sampleRate, channels, ms, options);
        return ms.ToArray();
    }

    /// <summary>Encode interleaved float PCM to an Ogg Opus file written to <paramref name="output"/> (left open).</summary>
    public static async Task EncodeAsync(Accelerator accelerator, ReadOnlyMemory<float> pcm, int sampleRate, int channels, Stream output,
        OggOpusEncodeOptions? options = null)
    {
        Validate(pcm.Length, sampleRate, channels);
        using var dFloat = accelerator.Allocate1D<float>(Math.Max(1, pcm.Length));
        if (pcm.Length > 0) dFloat.View.SubView(0, pcm.Length).CopyFromCPU(pcm.ToArray());
        var convert = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<short>>(PcmConvertKernels.FloatToS16Kernel);
        await EncodeCoreAsync(accelerator, pcm.Length / channels, sampleRate, channels, output, options, dest =>
        {
            if (pcm.Length > 0) convert(pcm.Length, dFloat.View, dest);
        });
    }

    /// <summary>
    /// Compile and load every kernel an encode at these settings uses (20 ms of silence, discarded), so the first real
    /// encode does not wait for them. On browser backends (WebGPU / WebGL) the first compile of the CELT stage shaders can
    /// take minutes (Chrome on D3D compiles WGSL through FXC: about 110 s on an RX 590); call this early, for example
    /// when a page shows a node that will save Opus. <paramref name="floatInput"/> also loads the float to int16 kernel.
    /// </summary>
    public static async Task WarmUpAsync(Accelerator accelerator, int sampleRate = 48000, int channels = 2,
        OggOpusEncodeOptions? options = null, bool floatInput = true)
    {
        int n = sampleRate / 50 * channels;
        if (floatInput) await EncodeAsync(accelerator, new float[n], sampleRate, channels, Stream.Null, options);
        else await EncodeAsync(accelerator, new short[n], sampleRate, channels, Stream.Null, options);
    }

    private static void Validate(int length, int sampleRate, int channels)
    {
        if (channels is < 1 or > 2) throw new NotSupportedException("Ogg Opus mapping family 0 carries 1 or 2 channels.");
        if (sampleRate is not (8000 or 12000 or 16000 or 24000 or 48000))
            throw new ArgumentOutOfRangeException(nameof(sampleRate), "Opus encodes 8000, 12000, 16000, 24000 or 48000 Hz.");
        if (length % channels != 0) throw new ArgumentException("The interleaved PCM length is not a multiple of the channel count.");
    }

    private static async Task EncodeCoreAsync(Accelerator accelerator, int samples, int sampleRate, int channels, Stream output,
        OggOpusEncodeOptions? options, Action<ArrayView<short>> fill)
    {
        ArgumentNullException.ThrowIfNull(output);
        var o = options ?? new OggOpusEncodeOptions();
        using var enc = new OpusEncoderGpu(accelerator, sampleRate, channels, o.Application)
        {
            Bitrate = o.Bitrate ?? (channels == 2 ? 96000 : 64000),
            Complexity = o.Complexity,
            Vbr = o.Vbr,
            VbrConstraint = o.VbrConstraint,
            PacketLossPercent = o.PacketLossPercent,
            PhaseInversionDisabled = o.PhaseInversionDisabled,
        };
        int frame = sampleRate / 50;
        int lookahead = enc.Lookahead;
        int scale = 48000 / sampleRate;
        int preSkip = lookahead * scale;
        long packets = (samples + (long)lookahead + frame - 1) / frame;

        // the whole input, zero-padded to the last packet, uploaded once
        using var dPcm = accelerator.Allocate1D<short>(Math.Max(1, packets * frame * channels));
        dPcm.MemSetToZero();
        fill(dPcm.View);

        using var writer = new OggOpusWriter(output, new OggOpusWriterOptions
        {
            Channels = channels, PreSkip = preSkip, InputSampleRate = o.InputSampleRate ?? sampleRate,
            Vendor = o.Vendor, Tags = o.Tags, Serial = o.Serial,
        }, leaveOpen: true);
        long end = (long)samples * scale + preSkip;   // the final granule: every input sample, nothing more
        for (long k = 0; k < packets; k++)
        {
            var packet = await enc.EncodePacketToHostAsync(dPcm.View.SubView(k * frame * channels, (long)frame * channels), frame,
                MaxPacketBytesPerStream);
            long pts = -preSkip + k * 960;
            int duration = (int)Math.Min(960, end - preSkip - pts);
            writer.WritePacket(packet, pts, duration);
        }
        writer.Finish();
    }
}

/// <summary>Sample format conversion kernels.</summary>
public static class PcmConvertKernels
{
    /// <summary>float -> int16 as libswresample converts (x86): lrintf(x * 32768) (round half to even), clipped to
    /// int16; NaN and values beyond the int32 range give -32768 (cvtps2dq's integer indefinite).</summary>
    public static void FloatToS16Kernel(Index1D i, ArrayView<float> src, ArrayView<short> dst)
    {
        float v = src[i] * 32768f;   // exact: a power of two
        short r;
        if (!(v == v) || v >= 2147483648f || v < -2147483648f) r = -32768;
        else if (v >= 32767f) r = 32767;
        else if (v <= -32768f) r = -32768;
        else
        {
            int t = (int)v;
            if (v < t) t--;          // floor
            float frac = v - t;      // exact below 2^23
            if (frac > 0.5f || (frac == 0.5f && (t & 1) != 0)) t++;
            r = (short)t;
        }
        dst[i] = r;
    }
}
