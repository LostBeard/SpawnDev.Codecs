// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// One call from RGB frames (ComfyUI's IMAGE layout: float32 H x W x 3 per frame, values in [0, 1]) + frame rate +
// optional audio to an MP4 or WebM file. Per frame on the accelerator: float RGB -> RGBA8 (ComfyUI's save: x * 255
// clamped and truncated), RGBA8 -> YUV 4:2:0 (WebPRgbaToYuvGpu, BT.601 limited range), then the keyframe encoder (VP9: any
// size; AV1: multiples of 64); the encoded packet is the one readback per frame. Audio: Opus packets (OggOpusEncoder's
// encode; 8, 12, 16, 24 or 48 kHz input). Muxing: Mp4Writer or WebMWriter (host-side framing).
//
// Every frame is a key frame (the encoders are keyframe-only), so files are larger than inter-coded ones.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.Codecs.Container.Mp4;
using SpawnDev.Codecs.Images.Yuv;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video;

/// <summary>The file format <see cref="VideoFileEncoderGpu"/> writes.</summary>
public enum VideoContainer
{
    /// <summary>MP4 (ISO BMFF).</summary>
    Mp4,
    /// <summary>WebM (Matroska).</summary>
    WebM,
}

/// <summary>Options of <see cref="VideoFileEncoderGpu.EncodeAsync"/>.</summary>
public sealed record VideoFileEncodeOptions
{
    /// <summary>The file format.</summary>
    public VideoContainer Container { get; init; } = VideoContainer.Mp4;
    /// <summary>The video codec: VP9 (any size) or AV1 (width and height multiples of 64, the AV1 encoder's current
    /// limit). VP8 is not offered here.</summary>
    public VideoCodec Codec { get; init; } = VideoCodec.Vp9;
    /// <summary>Base quantizer index (1..255; lower is better quality, bigger file). Null: 30 for VP9, 32 for AV1.</summary>
    public int? QIndex { get; init; }
    /// <summary>Opus bitrate in bits per second. Null: 64000 mono, 96000 stereo.</summary>
    public int? AudioBitrate { get; init; }
}

/// <summary>Encodes RGB frames (+ audio) into an MP4 or WebM file on the accelerator.</summary>
public sealed class VideoFileEncoderGpu : IDisposable
{
    private readonly Accelerator _acc;
    private readonly WebPRgbaToYuvGpu _toYuv;
    private Vp9KeyframeEncoderGpu? _vp9;
    private Av1KeyframeEncoderGpu? _av1;
    private Action<Index1D, ArrayView<float>, ArrayView<byte>>? _toRgba;

    /// <summary>An encoder on <paramref name="accelerator"/>. Compiles nothing until the first encode.</summary>
    public VideoFileEncoderGpu(Accelerator accelerator)
    {
        _acc = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        _toYuv = new WebPRgbaToYuvGpu(accelerator);
    }

    /// <summary>
    /// Encode <paramref name="frames"/> (each <c>height * width * 3</c> floats, R, G, B in [0, 1]) at
    /// <paramref name="fpsNum"/> / <paramref name="fpsDen"/> frames per second, with <paramref name="audio"/> (optional,
    /// interleaved float) as an Opus track. Returns the file.
    /// </summary>
    public async Task<byte[]> EncodeAsync(IReadOnlyList<float[]> frames, int width, int height, int fpsNum, int fpsDen,
        LoadedAudio? audio = null, VideoFileEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(frames);
        options ??= new VideoFileEncodeOptions();
        if (frames.Count == 0) throw new ArgumentException("No frames.", nameof(frames));
        if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (fpsNum <= 0 || fpsDen <= 0) throw new ArgumentOutOfRangeException(nameof(fpsNum));
        long values = (long)width * height * 3;
        if (frames.Any(f => f == null || f.LongLength != values)) throw new ArgumentException($"Every frame must hold width * height * 3 = {values} floats.", nameof(frames));
        if (options.Codec is not (VideoCodec.Vp9 or VideoCodec.Av1)) throw new NotSupportedException($"{options.Codec}: VP9 or AV1.");
        bool av1 = options.Codec == VideoCodec.Av1;
        if (av1 && (width % 64 != 0 || height % 64 != 0))
            throw new NotSupportedException($"AV1 encodes sizes that are multiples of 64 so far ({width}x{height}); use VP9 for other sizes.");
        int q = options.QIndex ?? (av1 ? 32 : 30);

        // video packets (every frame a key frame)
        var packets = await EncodeVideoAsync(frames, width, height, av1, q);
        // audio packets
        OpusPacketStream? opus = null;
        if (audio != null && audio.Samples.Length > 0)
            opus = await OggOpusEncoder.EncodePacketsAsync(_acc, audio.Samples, audio.SampleRate, audio.Channels,
                new OggOpusEncodeOptions { Bitrate = options.AudioBitrate });

        using var ms = new MemoryStream();
        if (options.Container == VideoContainer.Mp4) WriteMp4(ms, packets, width, height, fpsNum, fpsDen, av1, opus);
        else WriteWebM(ms, packets, width, height, fpsNum, fpsDen, av1, opus);
        return ms.ToArray();
    }

    private async Task<List<byte[]>> EncodeVideoAsync(IReadOnlyList<float[]> frames, int width, int height, bool av1, int q)
    {
        _toRgba ??= await _acc.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<float>, ArrayView<byte>>(ToRgbaKernel);
        await _toYuv.EnsureKernelsAsync();
        int pixels = width * height;
        long chroma = WebPRgbaToYuvGpu.ChromaPlaneSize(width, height);
        int max = av1 ? Av1KeyframeEncoderGpu.GetMaxEncodedSize(width, height) : Vp9KeyframeEncoderGpu.GetMaxEncodedSize(width, height);
        // separate buffers: WebGPU rejects two read_write bindings of one buffer that share a 4-byte word
        using var rgbF = _acc.Allocate1D<float>((long)pixels * 3);
        using var rgba = _acc.Allocate1D<byte>((long)pixels * 4);
        using var y = _acc.Allocate1D<byte>(pixels);
        using var u = _acc.Allocate1D<byte>(chroma);
        using var v = _acc.Allocate1D<byte>(chroma);
        using var alpha = _acc.Allocate1D<byte>(pixels);
        using var output = _acc.Allocate1D<byte>(max);
        var packets = new List<byte[]>(frames.Count);
        foreach (var frame in frames)
        {
            rgbF.View.CopyFromCPU(frame);
            _toRgba(pixels, rgbF.View, rgba.View);
            _toYuv.Convert(rgba.View, width, height, y.View, u.View, v.View, alpha.View);
            int n;
            if (av1) n = await (_av1 ??= new Av1KeyframeEncoderGpu(_acc)).EncodeKeyFrameAsync(y.View, u.View, v.View, width, height, output.View, q);
            else n = await (_vp9 ??= new Vp9KeyframeEncoderGpu(_acc)).EncodeKeyFrameAsync(y.View, u.View, v.View, width, height, output.View, q);
            var bytes = await output.View.SubView(0, n).CopyToHostAsync();
            packets.Add(av1 ? Av1Isobmff.ToSample(bytes) : bytes);
        }
        return packets;
    }

    /// <summary>ComfyUI's save: each channel x * 255, clamped to [0, 255], truncated to a byte; alpha 255.</summary>
    private static void ToRgbaKernel(Index1D index, ArrayView<float> rgb, ArrayView<byte> rgba)
    {
        int i = index;
        for (int c = 0; c < 3; c++)
        {
            float x = rgb[i * 3 + c] * 255f;
            x = x < 0f ? 0f : x > 255f ? 255f : x;
            rgba[i * 4 + c] = (byte)(int)x;
        }
        rgba[i * 4 + 3] = 255;
    }

    /// <summary>Round half away from zero (ffmpeg's default rescale rounding).</summary>
    private static long RoundDiv(long num, long den) => num >= 0 ? (num + den / 2) / den : -((-num + den / 2) / den);

    private static void WriteMp4(Stream s, List<byte[]> packets, int width, int height, int fpsNum, int fpsDen, bool av1, OpusPacketStream? opus)
    {
        using var w = new Mp4Writer(s, leaveOpen: true);
        int ts = Mp4Writer.VideoTimescale(fpsNum, fpsDen);
        int video = av1
            ? w.AddVideoTrack(new Mp4Av1Track { Av1C = Av1Isobmff.BuildAv1C(packets[0]), Width = width, Height = height, Timescale = ts })
            : w.AddVideoTrack(new Mp4Vp9Track { Width = width, Height = height, Timescale = ts });
        int audio = opus != null ? w.AddAudioTrack(new Mp4OpusTrack { OpusHead = opus.OpusHead }) : 0;
        for (int k = 0; k < packets.Count; k++)
        {
            long dts = (long)k * ts * fpsDen / fpsNum, next = (long)(k + 1) * ts * fpsDen / fpsNum;
            w.WriteFrame(video, packets[k], dts, next - dts, true);
        }
        if (opus != null)
            foreach (var p in opus.Packets) w.WriteFrame(audio, p.Data, p.Pts, p.Duration, true);
        w.Finish();
    }

    private static void WriteWebM(Stream s, List<byte[]> packets, int width, int height, int fpsNum, int fpsDen, bool av1, OpusPacketStream? opus)
    {
        using var w = new WebMWriter(s, leaveOpen: true);
        long frameNs = 1_000_000_000L * fpsDen / fpsNum;
        int video = w.AddVideoTrack(new WebMVideoTrack
        {
            CodecId = av1 ? "V_AV1" : "V_VP9", CodecPrivate = av1 ? Av1Isobmff.BuildAv1C(packets[0]) : null,
            Width = width, Height = height, DefaultDurationNs = frameNs,
        });
        int audio = opus != null ? w.AddAudioTrack(new WebMAudioTrack
        {
            CodecId = "A_OPUS", CodecPrivate = opus.OpusHead, SampleRate = 48000, Channels = opus.Channels, InitialPaddingSamples = opus.PreSkip,
        }) : 0;
        long frameMs = RoundDiv(1000L * fpsDen, fpsNum);
        for (int k = 0; k < packets.Count; k++)
            w.WriteFrame(video, packets[k], RoundDiv(1000L * k * fpsDen, fpsNum), true, frameMs);
        if (opus != null)
            foreach (var p in opus.Packets)
            {
                long discardNs = p.Duration < 960 ? (960 - p.Duration) * 1_000_000_000L / 48000 : 0;
                w.WriteFrame(audio, p.Data, RoundDiv(p.Pts * 1000, 48000), true, RoundDiv(p.Duration * 1000L, 48000), default, discardNs);
            }
        w.Finish();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _vp9?.Dispose();
        _av1?.Dispose();
        _toYuv.Dispose();
    }
}
