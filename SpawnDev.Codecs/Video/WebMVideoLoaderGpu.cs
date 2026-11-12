// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// One call from a WebM file to what ComfyUI's LoadVideo -> GetVideoComponents gives: RGB frames (float32, H x W x 3 per
// frame, bit-exact with ComfyUI for VP9), the frame rate, and the audio. The host demuxes and sequences; the decode and
// the colour conversion run on the accelerator, one readback per frame (a whole clip in one device buffer would exceed
// WebGPU's binding limits).
//
// Measured against ComfyUI as a black box (Plans/PLAN-ComfyUI-Video-IO.md): frames are YUV -> 16-bit RGB with bilinear
// chroma (Yuv420ToRgb16Gpu) times 1/65535, alpha dropped; audio is the Opus track decoded with the pre-skip (CodecDelay)
// and the end padding (DiscardPadding) trimmed. Our Opus decoder is libopus-exact; ComfyUI's ffmpeg uses its own Opus
// decoder, which differs by at most about 1.5e-5.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Opus;
using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.Codecs.Images.Yuv;
using SpawnDev.Codecs.Video.Vp9.Decoder;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video;

/// <summary>Decoded audio: interleaved float PCM.</summary>
/// <param name="SampleRate">Samples per second per channel.</param>
/// <param name="Channels">Channel count.</param>
/// <param name="Samples">Interleaved samples (<c>frames * Channels</c>).</param>
public sealed record LoadedAudio(int SampleRate, int Channels, float[] Samples);

/// <summary>A loaded video.</summary>
/// <param name="Width">Picture width.</param>
/// <param name="Height">Picture height.</param>
/// <param name="FrameRate">Frames per second.</param>
/// <param name="Frames">One float32 array per frame, <c>Height * Width * 3</c> (R, G, B in [0, 1]).</param>
/// <param name="Audio">The first audio track, or null.</param>
public sealed record LoadedVideo(int Width, int Height, double FrameRate, IReadOnlyList<float[]> Frames, LoadedAudio? Audio);

/// <summary>Loads WebM files the way ComfyUI's LoadVideo does (VP9 video, Opus audio).</summary>
public sealed class WebMVideoLoaderGpu
{
    private readonly Accelerator _acc;
    private readonly Yuv420ToRgb16Gpu _rgb;

    /// <summary>A loader on <paramref name="accelerator"/>. Compiles nothing until the first load.</summary>
    public WebMVideoLoaderGpu(Accelerator accelerator)
    {
        _acc = accelerator ?? throw new ArgumentNullException(nameof(accelerator));
        _rgb = new Yuv420ToRgb16Gpu(accelerator);
    }

    /// <summary>Decode <paramref name="file"/> (a WebM / Matroska file with a VP9 video track).</summary>
    public async Task<LoadedVideo> LoadAsync(byte[] file)
    {
        ArgumentNullException.ThrowIfNull(file);
        var mkv = new MatroskaContainer(new MemoryStream(file, writable: false));
        var tracks = mkv.Tracks.ToList();
        var video = tracks.FirstOrDefault(t => t.IsVideo) ?? throw new InvalidDataException("No video track.");
        if (video.CodecId != "V_VP9")
            throw new NotSupportedException($"Video codec {video.CodecId}: only VP9 decodes every frame so far (AV1 / VP8: keyframes only).");
        var audio = tracks.FirstOrDefault(t => t.IsAudio);
        var frames = mkv.Frames.ToList();
        ulong scaleNs = mkv.SegmentInfo.TimestampScaleNs;

        // video
        await _rgb.EnsureKernelsAsync();
        var pictures = new List<float[]>();
        int width = 0, height = 0;
        using (var dec = new Vp9DecoderGpu(_acc))
        {
            MemoryBuffer1D<byte, Stride1D.Dense>? planes = null;
            MemoryBuffer1D<float, Stride1D.Dense>? rgb = null;
            try
            {
                foreach (var f in frames.Where(f => f.TrackNumber == video.TrackNumber))
                {
                    long need = PlanarBound(video, f);
                    if (planes == null || planes.Length < need) { planes?.Dispose(); planes = _acc.Allocate1D<byte>(need); }
                    var format = await dec.DecodePacketAsync(f.Data, planes.View);
                    if (format is not { } fmt) continue;
                    if (fmt.SubsamplingX != 1 || fmt.SubsamplingY != 1) throw new NotSupportedException("Only 4:2:0 VP9 is supported.");
                    width = fmt.Width; height = fmt.Height;
                    long values = (long)fmt.Width * fmt.Height * 3;
                    if (rgb == null || rgb.Length < values) { rgb?.Dispose(); rgb = _acc.Allocate1D<float>(values); }
                    _rgb.ConvertToFloat(planes.View, fmt.Width, fmt.Height, rgb.View);
                    pictures.Add(await rgb.View.SubView(0, values).CopyToHostAsync());
                }
            }
            finally { planes?.Dispose(); rgb?.Dispose(); }
        }

        // frame rate: the track's DefaultDuration, else the mean spacing of the frame timestamps
        double fps = 0;
        if (video.DefaultDurationNs is > 0 and var dd) fps = RateFromDuration(dd);
        else
        {
            var ts = frames.Where(f => f.TrackNumber == video.TrackNumber).Select(f => f.Timestamp).ToList();
            if (ts.Count > 1 && ts[^1] > ts[0]) fps = (ts.Count - 1) * 1e9 / ((ts[^1] - ts[0]) * (double)scaleNs);
        }

        LoadedAudio? sound = null;
        if (audio != null) sound = await DecodeAudioAsync(audio, frames.Where(f => f.TrackNumber == audio.TrackNumber).ToList());
        return new LoadedVideo(width, height, fps, pictures, sound);
    }

    /// <summary>
    /// The frame rate a DefaultDuration (whole nanoseconds, truncated by the muxer) stands for: the fraction p / q with
    /// the smallest q (up to 1001) whose duration truncates to it, so 41666666 ns gives exactly 24 and 33366666 ns
    /// 30000 / 1001, as ffmpeg (ComfyUI's LoadVideo) reports. Checked with ffprobe on 24, 25, 30000/1001, 24000/1001,
    /// 60, 15, 12.5, 50 and 8 fps; 60000/1001 differs by 5e-6 fps (ffmpeg reports 19001/317 there, from its own
    /// timestamp analysis).
    /// </summary>
    internal static double RateFromDuration(ulong durationNs)
    {
        for (long q = 1; q <= 1001; q++)
        {
            long p = (long)Math.Round(1e9 * q / durationNs);
            for (long pp = p - 1; pp <= p + 1; pp++)
                if (pp > 0 && (ulong)(1_000_000_000L * q / pp) == durationNs) return (double)pp / q;
        }
        return 1e9 / durationNs;
    }

    /// <summary>
    /// Planar bytes to hold one picture: the track's PixelWidth x PixelHeight (4:2:0), else 4096 x 4096. A stream that
    /// grows past it fails in the decoder with a clear message rather than silently.
    /// </summary>
    private static long PlanarBound(MatroskaTrack video, MatroskaFrame f)
    {
        long w = (long)(video.PixelWidth ?? 0), h = (long)(video.PixelHeight ?? 0);
        if (w <= 0 || h <= 0) { w = 4096; h = 4096; }
        long cw = (w + 1) / 2, ch = (h + 1) / 2;
        return w * h + 2 * cw * ch;
    }

    private async Task<LoadedAudio?> DecodeAudioAsync(MatroskaTrack track, List<MatroskaFrame> blocks)
    {
        if (track.CodecId != "A_OPUS") throw new NotSupportedException($"Audio codec {track.CodecId}: only Opus so far.");
        var head = track.CodecPrivate;
        if (head == null || head.Length < 19 || System.Text.Encoding.ASCII.GetString(head, 0, 8) != "OpusHead")
            throw new InvalidDataException("Opus track without an OpusHead.");
        int channels = head[9];
        int preSkip = head[10] | head[11] << 8;
        if (head[18] != 0 || channels < 1 || channels > 2) throw new NotSupportedException("Opus with more than 2 channels (mapping family > 0).");
        using var dec = new OpusDecoderGpu(_acc, 48000, channels);
        var pcm = new List<float>();
        long discardEnd = 0;
        foreach (var b in blocks)
        {
            pcm.AddRange(await dec.DecodePacketAsync(b.Data));
            if (b.DiscardPaddingNs is > 0 and var ns) discardEnd = (ns * 48000 + 500_000_000) / 1_000_000_000;
        }
        long total = pcm.Count / channels;
        long keep = Math.Max(0, total - preSkip - discardEnd);
        var samples = pcm.GetRange(preSkip * channels, (int)keep * channels).ToArray();
        return new LoadedAudio(48000, channels, samples);
    }
}
