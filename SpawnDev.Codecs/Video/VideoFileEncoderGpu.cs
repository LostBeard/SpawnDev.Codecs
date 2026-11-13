// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// One call from RGB frames (ComfyUI's IMAGE layout: float32 H x W x 3 per frame, values in [0, 1]) + frame rate +
// optional audio to an MP4 or WebM file. Per frame on the accelerator: float RGB -> RGBA8 (ComfyUI's save: x * 255
// clamped and truncated), RGBA8 -> YUV 4:2:0 (WebPRgbaToYuvGpu, BT.601 limited range), then the encoder (VP9 or
// AV1, any picture size); the encoded packet is the one readback per frame. Audio: Opus packets (OggOpusEncoder's
// encode; 8, 12, 16, 24 or 48 kHz input). Muxing: Mp4Writer or WebMWriter (host-side framing).
//
// VP9 (Vp9EncoderGpu): a key frame every VideoFileEncodeOptions.KeyFrameInterval frames, the frames between predicted
// from the frame before (motion search); the containers mark only the key frames as sync samples. AV1: every frame
// is a key frame (Av1KeyframeEncoderGpu), so its files are larger than inter-coded ones.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.Codecs.Container.Mp4;
using SpawnDev.Codecs.Images.Yuv;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;
using System.Runtime.Versioning;

namespace SpawnDev.Codecs.Video;

/// <summary>The file format <see cref="VideoFileEncoderGpu"/> writes.</summary>
public enum VideoContainer
{
    /// <summary>MP4 (ISO BMFF).</summary>
    Mp4,
    /// <summary>WebM (Matroska).</summary>
    WebM,
    /// <summary>Matroska (.mkv): the WebM layout with DocType "matroska".</summary>
    Mkv,
}

/// <summary>Options of <see cref="VideoFileEncoderGpu.EncodeAsync"/>.</summary>
public sealed record VideoFileEncodeOptions
{
    /// <summary>The file format.</summary>
    public VideoContainer Container { get; init; } = VideoContainer.Mp4;
    /// <summary>The video codec: VP9 (key frames + inter frames) or AV1 (key frames only so far), any picture size.
    /// VP8 is not offered here.</summary>
    public VideoCodec Codec { get; init; } = VideoCodec.Vp9;
    /// <summary>Base quantizer index (1..255; lower is better quality, bigger file). Null: 30 for VP9, 32 for AV1.</summary>
    public int? QIndex { get; init; }
    /// <summary>Opus bitrate in bits per second. Null: 64000 mono, 96000 stereo.</summary>
    public int? AudioBitrate { get; init; }
    /// <summary>
    /// VP9: a key frame at least every this many frames; the frames between are predicted from the frame before them
    /// (much smaller files for moving pictures that change little from frame to frame). Null: 128. 1: every frame a
    /// key frame. AV1 writes key frames only so far.
    /// </summary>
    public int? KeyFrameInterval { get; init; }
    /// <summary>
    /// AV1: keeps the tile grid inside an AV1 level (a seq_level_idx such as <see cref="Av1.Av1Level.Level5_1"/>;
    /// <see cref="Av1.Av1KeyframeEncoderGpu.LevelLimit"/>): slower to encode, and a file every decoder of that level
    /// must decode. Null: no limit, the fastest grid (one superblock per tile). Either way the file signals the level
    /// it really conforms to, which without a limit is a high one, or "maximum parameters", for most picture sizes.
    /// </summary>
    public int? Av1LevelLimit { get; init; }

    /// <summary>
    /// The base quantizer index for a libvpx / libaom style quantizer (crf / cq-level, 0..63): their quantizer_to_qindex
    /// table (q * 4 for 0..61, 62 -> 249, 63 -> 255), at least 1 (lossless is not implemented). A fixed quantizer for
    /// every frame (no rate control): the same crf does not give the file size libvpx / libaom give.
    /// </summary>
    public static int QIndexFromCrf(int crf)
    {
        if (crf < 0 || crf > 63) throw new ArgumentOutOfRangeException(nameof(crf), "crf is 0..63.");
        int q = crf <= 61 ? crf * 4 : crf == 62 ? 249 : 255;
        return Math.Max(1, q);
    }
}

/// <summary>Encodes RGB frames (+ audio) into an MP4 or WebM file on the accelerator.</summary>
public sealed class VideoFileEncoderGpu : IDisposable
{
    private readonly Accelerator _acc;
    private readonly WebPRgbaToYuvGpu _toYuv;
    private Vp9EncoderGpu? _vp9;
    private Av1KeyframeEncoderGpu? _av1;
    private Action<Index1D, ArrayView<float>, ArrayView<byte>>? _toRgba;

    /// <summary>Measurements (DemoConsole vp9-encode-bench): called with the VP9 encoder before each file's first frame.</summary>
    internal Action<Vp9EncoderGpu>? Vp9Setup { get; set; }

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
    public Task<byte[]> EncodeAsync(IReadOnlyList<float[]> frames, int width, int height, int fpsNum, int fpsDen,
        LoadedAudio? audio = null, VideoFileEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(frames);
        long values = CheckArguments(frames.Count, width, height, fpsNum, fpsDen);
        if (frames.Any(f => f == null || f.LongLength != values)) throw new ArgumentException($"Every frame must hold width * height * 3 = {values} floats.", nameof(frames));
        return EncodeCoreAsync(frames.Count, true, (k, stage) => { stage.CopyFromCPU(frames[k]); return stage; },
            width, height, fpsNum, fpsDen, audio, options);
    }

    /// <summary>
    /// As <see cref="EncodeAsync(IReadOnlyList{float[]}, int, int, int, int, LoadedAudio?, VideoFileEncodeOptions?)"/>
    /// for frames that are already on the accelerator: <paramref name="frames"/> holds
    /// <paramref name="frameCount"/> frames one after the other (ComfyUI's IMAGE batch layout, N x H x W x 3 floats).
    /// Nothing of the picture crosses to the host; each encoded packet is the one readback per frame.
    /// </summary>
    public Task<byte[]> EncodeAsync(ArrayView1D<float, Stride1D.Dense> frames, int frameCount, int width, int height, int fpsNum, int fpsDen,
        LoadedAudio? audio = null, VideoFileEncodeOptions? options = null)
    {
        long values = CheckArguments(frameCount, width, height, fpsNum, fpsDen);
        if (frames.Length < frameCount * values)
            throw new ArgumentException($"frames must hold frameCount * width * height * 3 = {frameCount * values} floats; it holds {frames.Length}.", nameof(frames));
        return EncodeCoreAsync(frameCount, false, (k, _) => frames.SubView(k * values, values), width, height, fpsNum, fpsDen, audio, options);
    }

    /// <summary>
    /// As <see cref="EncodeAsync(IReadOnlyList{float[]}, int, int, int, int, LoadedAudio?, VideoFileEncodeOptions?)"/>
    /// in the browser, for frames held as JS typed arrays (each <c>height * width * 3</c> floats): every frame goes
    /// JS -> GPU, never through the .NET heap. The file (the encoded packets and the container's framing, assembled on
    /// the host) is returned as a <see cref="Uint8Array"/>.
    /// </summary>
    [SupportedOSPlatform("browser")]
    public async Task<Uint8Array> EncodeAsync(IReadOnlyList<Float32Array> frames, int width, int height, int fpsNum, int fpsDen,
        LoadedAudio? audio = null, VideoFileEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(frames);
        long values = CheckArguments(frames.Count, width, height, fpsNum, fpsDen);
        if (frames.Any(f => f == null || f.Length != values)) throw new ArgumentException($"Every frame must hold width * height * 3 = {values} floats.", nameof(frames));
        var file = await EncodeCoreAsync(frames.Count, true, (k, stage) => { DeviceTransfer.Upload(stage, frames[k]); return stage; },
            width, height, fpsNum, fpsDen, audio, options);
        return new Uint8Array(file);
    }

    /// <summary>The arguments every overload shares; returns the floats per frame.</summary>
    private static long CheckArguments(int frameCount, int width, int height, int fpsNum, int fpsDen)
    {
        if (frameCount <= 0) throw new ArgumentException("No frames.", "frames");
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        if (fpsNum <= 0) throw new ArgumentOutOfRangeException(nameof(fpsNum));
        if (fpsDen <= 0) throw new ArgumentOutOfRangeException(nameof(fpsDen));
        return (long)width * height * 3;
    }

    /// <summary>
    /// The one encode behind every overload. <paramref name="frame"/> returns frame k as a view on the accelerator;
    /// with <paramref name="staged"/> it is handed a buffer of one frame to upload into (and return).
    /// </summary>
    private async Task<byte[]> EncodeCoreAsync(int frameCount, bool staged,
        Func<int, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> frame,
        int width, int height, int fpsNum, int fpsDen, LoadedAudio? audio, VideoFileEncodeOptions? options)
    {
        options ??= new VideoFileEncodeOptions();
        if (options.Codec is not (VideoCodec.Vp9 or VideoCodec.Av1)) throw new NotSupportedException($"{options.Codec}: VP9 or AV1.");
        bool av1 = options.Codec == VideoCodec.Av1;
        int q = options.QIndex ?? (av1 ? 32 : 30);
        int keyInterval = options.KeyFrameInterval ?? 128;
        if (keyInterval < 1) throw new ArgumentOutOfRangeException(nameof(options), "KeyFrameInterval must be at least 1.");

        // video packets (AV1: every frame a key frame; VP9: key frames keyInterval apart, inter frames between)
        var packets = await EncodeVideoAsync(frameCount, staged, frame, width, height, fpsNum, fpsDen, av1, q, keyInterval,
            options.Av1LevelLimit ?? Av1Level.MaximumParameters);
        // audio packets
        OpusPacketStream? opus = null;
        if (audio != null && audio.Samples.Length > 0)
            opus = await OggOpusEncoder.EncodePacketsAsync(_acc, audio.Samples, audio.SampleRate, audio.Channels,
                new OggOpusEncodeOptions { Bitrate = options.AudioBitrate });

        using var ms = new MemoryStream();
        if (options.Container == VideoContainer.Mp4) WriteMp4(ms, packets, width, height, fpsNum, fpsDen, av1, opus);
        else WriteWebM(ms, packets, width, height, fpsNum, fpsDen, av1, opus, options.Container == VideoContainer.Mkv ? "matroska" : "webm");
        return ms.ToArray();
    }

    private async Task<List<(byte[] Data, bool Key)>> EncodeVideoAsync(int frameCount, bool staged,
        Func<int, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> frame, int width, int height,
        int fpsNum, int fpsDen, bool av1, int q, int keyInterval, int av1LevelLimit)
    {
        _toRgba ??= await _acc.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<float>, ArrayView<byte>>(ToRgbaKernel);
        await _toYuv.EnsureKernelsAsync();
        int pixels = width * height;
        long chroma = WebPRgbaToYuvGpu.ChromaPlaneSize(width, height);
        int max = av1 ? Av1KeyframeEncoderGpu.GetMaxEncodedSize(width, height) : Vp9EncoderGpu.GetMaxEncodedSize(width, height);
        if (av1)
        {
            // the level each frame signals depends on the rate it is shown at and on the tile grid
            _av1 ??= new Av1KeyframeEncoderGpu(_acc);
            _av1.SetFrameRate(fpsNum, fpsDen);
            _av1.LevelLimit = av1LevelLimit;
        }
        else
        {
            // one file = one sequence: it starts with a key frame
            _vp9 ??= new Vp9EncoderGpu(_acc);
            _vp9.Reset();
            _vp9.KeyFrameInterval = keyInterval;
            Vp9Setup?.Invoke(_vp9);
        }
        // separate buffers: WebGPU rejects two read_write bindings of one buffer that share a 4-byte word
        // one frame of floats to upload into: only for sources that are not on the accelerator
        using var stage = staged ? _acc.Allocate1D<float>((long)pixels * 3) : null;
        using var rgba = _acc.Allocate1D<byte>((long)pixels * 4);
        using var y = _acc.Allocate1D<byte>(pixels);
        using var u = _acc.Allocate1D<byte>(chroma);
        using var v = _acc.Allocate1D<byte>(chroma);
        using var alpha = _acc.Allocate1D<byte>(pixels);
        using var output = _acc.Allocate1D<byte>(max);
        var packets = new List<(byte[] Data, bool Key)>(frameCount);
        for (int k = 0; k < frameCount; k++)
        {
            _toRgba(pixels, frame(k, stage != null ? stage.View : default), rgba.View);
            _toYuv.Convert(rgba.View, width, height, y.View, u.View, v.View, alpha.View);
            int n;
            bool key = true;
            if (av1) n = await _av1!.EncodeKeyFrameAsync(y.View, u.View, v.View, width, height, output.View, q);
            else
            {
                var encoded = await _vp9!.EncodeFrameAsync(y.View, u.View, v.View, width, height, output.View, q);
                n = encoded.Length;
                key = encoded.IsKeyFrame;
            }
            var bytes = await output.View.SubView(0, n).CopyToHostAsync();
            packets.Add((av1 ? Av1Isobmff.ToSample(bytes) : bytes, key));
        }
        if (av1 && packets.Count > 0)
        {
            // one file = one sequence = one sequence header: each frame signalled the level a stream of frames of ITS
            // size conforms to; the file takes the highest of them (the level of its largest frame)
            var units = packets.Select(p => p.Data).ToList();
            Av1Isobmff.UnifySequenceHeaders(units);
            for (int k = 0; k < packets.Count; k++) packets[k] = (units[k], packets[k].Key);
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

    private static void WriteMp4(Stream s, List<(byte[] Data, bool Key)> packets, int width, int height, int fpsNum, int fpsDen, bool av1, OpusPacketStream? opus)
    {
        using var w = new Mp4Writer(s, leaveOpen: true);
        int ts = Mp4Writer.VideoTimescale(fpsNum, fpsDen);
        int video = av1
            ? w.AddVideoTrack(new Mp4Av1Track { Av1C = Av1Isobmff.BuildAv1C(packets[0].Data), Width = width, Height = height, Timescale = ts })
            : w.AddVideoTrack(new Mp4Vp9Track { Width = width, Height = height, Timescale = ts });
        int audio = opus != null ? w.AddAudioTrack(new Mp4OpusTrack { OpusHead = opus.OpusHead }) : 0;
        for (int k = 0; k < packets.Count; k++)
        {
            long dts = (long)k * ts * fpsDen / fpsNum, next = (long)(k + 1) * ts * fpsDen / fpsNum;
            w.WriteFrame(video, packets[k].Data, dts, next - dts, packets[k].Key);
        }
        if (opus != null)
            foreach (var p in opus.Packets) w.WriteFrame(audio, p.Data, p.Pts, p.Duration, true);
        w.Finish();
    }

    private static void WriteWebM(Stream s, List<(byte[] Data, bool Key)> packets, int width, int height, int fpsNum, int fpsDen, bool av1, OpusPacketStream? opus,
        string docType)
    {
        using var w = new WebMWriter(s, new WebMWriterOptions { DocType = docType }, leaveOpen: true);
        long frameNs = 1_000_000_000L * fpsDen / fpsNum;
        int video = w.AddVideoTrack(new WebMVideoTrack
        {
            CodecId = av1 ? "V_AV1" : "V_VP9", CodecPrivate = av1 ? Av1Isobmff.BuildAv1C(packets[0].Data) : null,
            Width = width, Height = height, DefaultDurationNs = frameNs,
        });
        int audio = opus != null ? w.AddAudioTrack(new WebMAudioTrack
        {
            CodecId = "A_OPUS", CodecPrivate = opus.OpusHead, SampleRate = 48000, Channels = opus.Channels, InitialPaddingSamples = opus.PreSkip,
        }) : 0;
        long frameMs = RoundDiv(1000L * fpsDen, fpsNum);
        for (int k = 0; k < packets.Count; k++)
            w.WriteFrame(video, packets[k].Data, RoundDiv(1000L * k * fpsDen, fpsNum), packets[k].Key, frameMs);
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
