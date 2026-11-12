// VideoFileEncoderGpu: frames (ComfyUI IMAGE layout) + fps + audio -> MP4 / WebM, on the accelerator.
//
// Round trips (the muxers themselves are byte-identical with ffmpeg in their own tests):
//  - VP9 WebM at an unaligned size (80x50) with stereo audio, read back by WebMVideoLoaderGpu: 6 frames of 80x50, exactly 24
//    fps, every frame's PSNR against the input above a floor (keyframes at q 30), and exactly the input's audio sample
//    count (the Opus pre-skip and end trimming round-trip).
//  - The same frames as VP9 MP4: the MP4's samples (read back from stsz / stco / stsc) are exactly the WebM's blocks.
//  - AV1 MP4 at 64x64: the av1C in the file equals Av1Isobmff.BuildAv1C of the first sample, and the samples carry no
//    temporal delimiter.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.Codecs.Container.Mp4;
using SpawnDev.Codecs.Video;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static List<float[]> TestFrames(int w, int h, int n)
    {
        var frames = new List<float[]>();
        for (int k = 0; k < n; k++)
        {
            var f = new float[w * h * 3];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    int i = (y * w + x) * 3;
                    f[i] = (x + 7f * k) % w / w;
                    f[i + 1] = (float)y / h;
                    f[i + 2] = 0.5f + 0.4f * MathF.Sin((x + y + k) * 0.2f);
                }
            frames.Add(f);
        }
        return frames;
    }

    /// <summary>The MP4's samples of track 1, in decode order (stsz sizes, stco offsets, stsc samples per chunk).</summary>
    private static List<byte[]> Mp4Samples(byte[] mp4, out byte[]? av1c)
    {
        static int U32(byte[] d, int at) => (d[at] << 24) | (d[at + 1] << 16) | (d[at + 2] << 8) | d[at + 3];
        static int Find(byte[] d, string type, int from)
        {
            var t = System.Text.Encoding.ASCII.GetBytes(type);
            for (int i = from; i + 4 <= d.Length; i++)
                if (d[i] == t[0] && d[i + 1] == t[1] && d[i + 2] == t[2] && d[i + 3] == t[3]) return i - 4;   // box start
            return -1;
        }
        int trak = Find(mp4, "trak", 0);
        int stsz = Find(mp4, "stsz", trak), stco = Find(mp4, "stco", trak), stsc = Find(mp4, "stsc", trak);
        int a = Find(mp4, "av1C", trak);
        av1c = a >= 0 && a < stsz ? mp4.AsSpan(a + 8, U32(mp4, a) - 8).ToArray() : null;
        int count = U32(mp4, stsz + 16);
        var sizes = Enumerable.Range(0, count).Select(i => U32(mp4, stsz + 20 + 4 * i)).ToList();
        int chunks = U32(mp4, stco + 12);
        var offsets = Enumerable.Range(0, chunks).Select(i => U32(mp4, stco + 16 + 4 * i)).ToList();
        int entries = U32(mp4, stsc + 12);
        var per = new int[chunks];
        for (int e = 0; e < entries; e++)
        {
            int first = U32(mp4, stsc + 16 + 12 * e) - 1, n = U32(mp4, stsc + 20 + 12 * e);
            int last = e + 1 < entries ? U32(mp4, stsc + 16 + 12 * (e + 1)) - 1 : chunks;
            for (int c = first; c < last; c++) per[c] = n;
        }
        var samples = new List<byte[]>();
        int s = 0;
        for (int c = 0; c < chunks; c++)
        {
            int at = offsets[c];
            for (int k = 0; k < per[c]; k++) { samples.Add(mp4.AsSpan(at, sizes[s]).ToArray()); at += sizes[s++]; }
        }
        return samples;
    }

    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL")]
    public async Task VideoFileEncoderGpu_Vp9WebMAndMp4_RoundTrip()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            const int w = 80, h = 50, n = 6, rate = 48000, channels = 2;
            var frames = TestFrames(w, h, n);
            int audioFrames = 15000;   // 0.3125 s: not a multiple of the 20 ms packet
            var pcm = new float[audioFrames * channels];
            for (int i = 0; i < audioFrames; i++) { pcm[2 * i] = 0.5f * MathF.Sin(i * 0.0576f); pcm[2 * i + 1] = 0.3f * MathF.Sin(i * 0.0864f); }
            var sound = new LoadedAudio(rate, channels, pcm);
            using var enc = new VideoFileEncoderGpu(acc);
            var webm = await enc.EncodeAsync(frames, w, h, 24, 1, sound, new VideoFileEncodeOptions { Container = VideoContainer.WebM });
            var mp4 = await enc.EncodeAsync(frames, w, h, 24, 1, sound, new VideoFileEncodeOptions { Container = VideoContainer.Mp4 });
            if (!OperatingSystem.IsBrowser() && Environment.GetEnvironmentVariable("CODECS_DUMP_DIR") is { Length: > 0 } dump)
            {
                File.WriteAllBytes(Path.Combine(dump, "videofile_vp9.webm"), webm);
                File.WriteAllBytes(Path.Combine(dump, "videofile_vp9.mp4"), mp4);
            }

            var loaded = await new WebMVideoLoaderGpu(acc).LoadAsync(webm);
            Equal(w, loaded.Width);
            Equal(h, loaded.Height);
            True(loaded.FrameRate == 24.0, $"fps {loaded.FrameRate}");
            Equal(n, loaded.Frames.Count);
            double worst = double.MaxValue;
            var perFrame = new List<string>();
            for (int k = 0; k < n; k++)
            {
                double se = 0;
                for (int i = 0; i < frames[k].Length; i++) { double d = (frames[k][i] - loaded.Frames[k][i]) * 255; se += d * d; }
                double psnr = 10 * Math.Log10(255.0 * 255 / Math.Max(se / frames[k].Length, 1e-9));
                worst = Math.Min(worst, psnr);
                perFrame.Add($"{psnr:F1}");
            }
            True(worst > 28, $"worst frame PSNR {worst:F1} dB (frames: {string.Join(", ", perFrame)})");
            var audio = loaded.Audio ?? throw new Exception("no audio");
            Equal(channels, audio.Channels);
            Equal(audioFrames * channels, audio.Samples.Length);

            // MP4 samples == WebM blocks (the same packets)
            var blocks = new MatroskaContainer(new MemoryStream(webm)).Frames.Where(f => f.TrackNumber == 1).Select(f => f.Data).ToList();
            var samples = Mp4Samples(mp4, out _);
            Equal(blocks.Count, samples.Count);
            for (int k = 0; k < blocks.Count; k++) True(blocks[k].AsSpan().SequenceEqual(samples[k]), $"sample {k}: MP4 != WebM");

            // MKV: DocType "matroska", the same blocks, and it loads
            var mkv = await enc.EncodeAsync(frames, w, h, 24, 1, sound, new VideoFileEncodeOptions { Container = VideoContainer.Mkv });
            var mkvDoc = new MatroskaContainer(new MemoryStream(mkv));
            True(mkvDoc.IsMatroska, $"DocType {mkvDoc.DocType}");
            var mkvBlocks = mkvDoc.Frames.Where(f => f.TrackNumber == 1).Select(f => f.Data).ToList();
            Equal(blocks.Count, mkvBlocks.Count);
            for (int k = 0; k < blocks.Count; k++) True(blocks[k].AsSpan().SequenceEqual(mkvBlocks[k]), $"block {k}: MKV != WebM");
            var mkvLoaded = await new WebMVideoLoaderGpu(acc).LoadAsync(mkv);
            Equal(n, mkvLoaded.Frames.Count);
            Equal(audioFrames * channels, mkvLoaded.Audio!.Samples.Length);
            Console.WriteLine($"{acc.AcceleratorType}: VP9 {w}x{h} worst PSNR {worst:F1} dB, audio {audio.Samples.Length / channels} frames, webm {webm.Length} B, mp4 {mp4.Length} B");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>libvpx / libaom quantizer_to_qindex: q * 4 up to 61, 62 -> 249, 63 -> 255; 0 clamps to 1.</summary>
    [TestMethod(Category = "CpuOnly")]
    public void VideoFileEncodeOptions_QIndexFromCrf_MatchesQuantizerToQindex()
    {
        Equal(1, VideoFileEncodeOptions.QIndexFromCrf(0));
        Equal(96, VideoFileEncodeOptions.QIndexFromCrf(24));
        Equal(128, VideoFileEncodeOptions.QIndexFromCrf(32));
        Equal(244, VideoFileEncodeOptions.QIndexFromCrf(61));
        Equal(249, VideoFileEncodeOptions.QIndexFromCrf(62));
        Equal(255, VideoFileEncodeOptions.QIndexFromCrf(63));
        Throws<ArgumentOutOfRangeException>(() => VideoFileEncodeOptions.QIndexFromCrf(64));
    }

    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL")]
    public async Task VideoFileEncoderGpu_Av1Mp4_Av1CAndSamples()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new VideoFileEncoderGpu(acc);
            var mp4 = await enc.EncodeAsync(TestFrames(64, 64, 3), 64, 64, 16, 1, null, new VideoFileEncodeOptions { Codec = VideoCodec.Av1 });
            if (!OperatingSystem.IsBrowser() && Environment.GetEnvironmentVariable("CODECS_DUMP_DIR") is { Length: > 0 } dump)
                File.WriteAllBytes(Path.Combine(dump, "videofile_av1.mp4"), mp4);
            var samples = Mp4Samples(mp4, out var av1c);
            Equal(3, samples.Count);
            True(av1c != null, "no av1C");
            Equal(Convert.ToHexString(Av1Isobmff.BuildAv1C(samples[0])), Convert.ToHexString(av1c!));
            foreach (var s in samples) True(((s[0] >> 3) & 15) != 2, "a sample starts with a temporal delimiter");
            await ThrowsAsync<NotSupportedException>(() => enc.EncodeAsync(TestFrames(70, 50, 1), 70, 50, 24, 1, null, new VideoFileEncodeOptions { Codec = VideoCodec.Av1 }));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
