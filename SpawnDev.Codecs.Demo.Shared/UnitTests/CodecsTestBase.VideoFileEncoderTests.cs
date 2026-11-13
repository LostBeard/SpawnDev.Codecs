// VideoFileEncoderGpu: frames (ComfyUI IMAGE layout) + fps + audio -> MP4 / WebM, on the accelerator.
//
// Round trips (the muxers themselves are byte-identical with ffmpeg in their own tests):
//  - VP9 WebM at an unaligned size (80x50) with stereo audio, read back by WebMVideoLoaderGpu: 6 frames of 80x50, exactly 24
//    fps, every frame's PSNR against the input above a floor (a key frame + 5 inter frames at q 30), and exactly the
//    input's audio sample count (the Opus pre-skip and end trimming round-trip).
//  - The same frames as VP9 MP4: the MP4's samples (read back from stsz / stco / stsc) are exactly the WebM's blocks.
//  - Key frames: the container's flags (WebM SimpleBlock keyframe bit, MP4 stss) equal the bitstream's frame types
//    at the default interval, at KeyFrameInterval 3 (key frames 0 and 3) and at 1 (all key frames, no stss); the
//    video with inter frames is smaller than the key-frame-only video.
//  - AV1 MP4 at 64x64: the av1C in the file equals Av1Isobmff.BuildAv1C of the first sample, and the samples carry no
//    temporal delimiter.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.Codecs.Container.Mp4;
using SpawnDev.Codecs.Video;
using Float32Array = SpawnDev.SpawnJS.JSObjects.Float32Array;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Test frames: red slides 7 pixels per frame, green is still, blue drifts diagonally. Red is a ramp that wraps (a
    /// full-scale edge that moves: under 4:2:0 that edge alone holds a frame near 30 dB), or with
    /// <paramref name="smooth"/> one period of a sine (no edge: the PSNR then measures the coding).
    /// </summary>
    private static List<float[]> TestFrames(int w, int h, int n, bool smooth = false)
    {
        var frames = new List<float[]>();
        for (int k = 0; k < n; k++)
        {
            var f = new float[w * h * 3];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    int i = (y * w + x) * 3;
                    f[i] = smooth ? 0.5f + 0.45f * MathF.Sin(2 * MathF.PI * (x + 7f * k) / w) : (x + 7f * k) % w / w;
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

    /// <summary>
    /// A shown VP9 profile 0 frame's first byte: frame_marker (2 bits) = 2, profile 0, show_existing_frame 0, then
    /// frame_type (bit 2): 0 = key frame.
    /// </summary>
    private static bool Vp9PacketIsKey(byte[] packet)
    {
        if ((packet[0] & 0xF8) != 0x80) throw new Exception($"not a profile 0 VP9 frame header: 0x{packet[0]:X2}");
        return (packet[0] & 0x04) == 0;
    }

    /// <summary>The 1-based sync sample numbers of the MP4's first track (stss), or null when it has no stss box.</summary>
    private static int[]? Mp4SyncSamples(byte[] mp4)
    {
        static int U32(byte[] d, int at) => (d[at] << 24) | (d[at + 1] << 16) | (d[at + 2] << 8) | d[at + 3];
        static int Find(byte[] d, string type, int from, int to)
        {
            for (int i = from; i + 4 <= to; i++)
                if (d[i] == type[0] && d[i + 1] == type[1] && d[i + 2] == type[2] && d[i + 3] == type[3]) return i - 4;   // box start
            return -1;
        }
        int trak = Find(mp4, "trak", 0, mp4.Length);
        int stss = Find(mp4, "stss", trak, trak + U32(mp4, trak));   // inside the first trak only
        if (stss < 0) return null;
        return Enumerable.Range(0, U32(mp4, stss + 12)).Select(i => U32(mp4, stss + 16 + 4 * i)).ToArray();
    }

    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL")]
    public async Task VideoFileEncoderGpu_Vp9WebMAndMp4_RoundTrip()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            const int w = 80, h = 50, n = 6, rate = 48000, channels = 2;
            var frames = TestFrames(w, h, n, smooth: true);
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
            // q 30 on a smooth clip: round-to-nearest quantizer noise alone is about 39.8 dB in RGB (AC step 37 / 8 per
            // pixel, uniform, through the BT.601 matrix); measured 39.0 to 40.5 dB (CPU accelerator, 2026-11-12; the inter
            // frames' dead zone puts them a little under that)
            True(worst > 38, $"worst frame PSNR {worst:F1} dB (frames: {string.Join(", ", perFrame)})");
            var audio = loaded.Audio ?? throw new Exception("no audio");
            Equal(channels, audio.Channels);
            Equal(audioFrames * channels, audio.Samples.Length);

            // MP4 samples == WebM blocks (the same packets: the second file starts its own sequence with a key frame)
            var webmVideo = new MatroskaContainer(new MemoryStream(webm)).Frames.Where(f => f.TrackNumber == 1).ToList();
            var blocks = webmVideo.Select(f => f.Data).ToList();
            var samples = Mp4Samples(mp4, out _);
            Equal(blocks.Count, samples.Count);
            for (int k = 0; k < blocks.Count; k++) True(blocks[k].AsSpan().SequenceEqual(samples[k]), $"sample {k}: MP4 != WebM");

            // Key frames (default interval 128 > 6 frames): the first frame only, the rest predicted from the frame
            // before. The container's flag must be the bitstream's own frame type: a player seeks to sync samples.
            Equal(n, blocks.Count);
            for (int k = 0; k < n; k++)
            {
                True(Vp9PacketIsKey(blocks[k]) == (k == 0), $"frame {k}: bitstream key frame = {Vp9PacketIsKey(blocks[k])}");
                True(webmVideo[k].IsKeyframe == (k == 0), $"frame {k}: WebM keyframe flag = {webmVideo[k].IsKeyframe}");
            }
            Equal("1", string.Join(",", Mp4SyncSamples(mp4) ?? throw new Exception("MP4 with inter frames has no stss")));

            // KeyFrameInterval 3: key frames 0 and 3 in the bitstream, in the WebM flags and in stss; it loads, and the
            // frames after the second key frame are as good as the rest.
            var opt3 = new VideoFileEncodeOptions { Container = VideoContainer.WebM, KeyFrameInterval = 3 };
            var webm3 = await enc.EncodeAsync(frames, w, h, 24, 1, null, opt3);
            var video3 = new MatroskaContainer(new MemoryStream(webm3)).Frames.Where(f => f.TrackNumber == 1).ToList();
            Equal(n, video3.Count);
            for (int k = 0; k < n; k++)
            {
                True(Vp9PacketIsKey(video3[k].Data) == (k % 3 == 0), $"interval 3 frame {k}: bitstream key frame = {Vp9PacketIsKey(video3[k].Data)}");
                True(video3[k].IsKeyframe == (k % 3 == 0), $"interval 3 frame {k}: WebM keyframe flag = {video3[k].IsKeyframe}");
            }
            var loaded3 = await new WebMVideoLoaderGpu(acc).LoadAsync(webm3);
            Equal(n, loaded3.Frames.Count);
            for (int k = 0; k < n; k++)
            {
                double se = 0;
                for (int i = 0; i < frames[k].Length; i++) { double d = (frames[k][i] - loaded3.Frames[k][i]) * 255; se += d * d; }
                double psnr = 10 * Math.Log10(255.0 * 255 / Math.Max(se / frames[k].Length, 1e-9));
                True(psnr > 38, $"interval 3 frame {k}: PSNR {psnr:F1} dB");
            }
            var mp43 = await enc.EncodeAsync(frames, w, h, 24, 1, null, opt3 with { Container = VideoContainer.Mp4 });
            Equal("1,4", string.Join(",", Mp4SyncSamples(mp43) ?? throw new Exception("interval 3 MP4 has no stss")));

            // KeyFrameInterval 1: every frame a key frame, so no stss box (all samples are sync samples); and the
            // reason inter frames exist: the default file's video is smaller than the key-frame-only file's.
            var mp4Keys = await enc.EncodeAsync(frames, w, h, 24, 1, null, new VideoFileEncodeOptions { Container = VideoContainer.Mp4, KeyFrameInterval = 1 });
            var keySamples = Mp4Samples(mp4Keys, out _);
            Equal(n, keySamples.Count);
            for (int k = 0; k < n; k++) True(Vp9PacketIsKey(keySamples[k]), $"interval 1 frame {k} is not a key frame");
            True(Mp4SyncSamples(mp4Keys) == null, "interval 1 MP4 has an stss box");
            int interVideo = blocks.Sum(b => b.Length), keyVideo = keySamples.Sum(s => s.Length);
            True(interVideo < keyVideo, $"video with inter frames {interVideo} B, key frames only {keyVideo} B");
            // An inter frame against the same frame coded as a key frame at the same quantizer: within 2.5 dB. Inter
            // frames drop coefficients under 100/128 of a step (Vp9EncoderGpu.InterZbinQ7: a third fewer bytes at equal
            // PSNR on real clips), so at the same quantizer they sit a little under a key frame: measured here at most
            // 1.6 dB (frame 5: 39.0 vs 40.6 dB), with round to nearest 1.0 dB. The floor above (38 dB) is the tight guard.
            var loadedKeys = await new WebMVideoLoaderGpu(acc).LoadAsync(
                await enc.EncodeAsync(frames, w, h, 24, 1, null, new VideoFileEncodeOptions { Container = VideoContainer.WebM, KeyFrameInterval = 1 }));
            Equal(n, loadedKeys.Frames.Count);
            var keyPsnr = new List<string>();
            for (int k = 0; k < n; k++)
            {
                double se = 0, seKey = 0;
                for (int i = 0; i < frames[k].Length; i++)
                {
                    double d = (frames[k][i] - loaded.Frames[k][i]) * 255, dk = (frames[k][i] - loadedKeys.Frames[k][i]) * 255;
                    se += d * d;
                    seKey += dk * dk;
                }
                double psnr = 10 * Math.Log10(255.0 * 255 / Math.Max(se / frames[k].Length, 1e-9));
                double psnrKey = 10 * Math.Log10(255.0 * 255 / Math.Max(seKey / frames[k].Length, 1e-9));
                keyPsnr.Add($"{psnrKey:F1}");
                True(psnr > psnrKey - 2.5, $"frame {k}: {psnr:F1} dB with inter frames, {psnrKey:F1} dB as a key frame");
            }
            await ThrowsAsync<ArgumentOutOfRangeException>(() => enc.EncodeAsync(frames, w, h, 24, 1, null, new VideoFileEncodeOptions { KeyFrameInterval = 0 }));

            // Frames already on the accelerator (N x H x W x 3 in one buffer): the same file, byte for byte
            using (var onDevice = acc.Allocate1D<float>((long)n * w * h * 3))
            {
                for (int k = 0; k < n; k++) onDevice.View.SubView((long)k * w * h * 3, w * h * 3).CopyFromCPU(frames[k]);
                var fromDevice = await enc.EncodeAsync(onDevice.View, n, w, h, 24, 1, sound, new VideoFileEncodeOptions { Container = VideoContainer.WebM });
                True(fromDevice.AsSpan().SequenceEqual(webm), $"the ArrayView overload's file ({fromDevice.Length} B) differs from the managed overload's ({webm.Length} B)");
                await ThrowsAsync<ArgumentException>(() => enc.EncodeAsync(onDevice.View, n + 1, w, h, 24, 1));
            }

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
            Console.WriteLine($"{acc.AcceleratorType}: VP9 {w}x{h} worst PSNR {worst:F1} dB (frames: {string.Join(", ", perFrame)}; as key frames: {string.Join(", ", keyPsnr)}), audio {audio.Samples.Length / channels} frames, " +
                $"webm {webm.Length} B, mp4 {mp4.Length} B, video {interVideo} B with inter frames / {keyVideo} B key frames only ({100.0 * interVideo / keyVideo:F0}%)");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>Browser: frames as JS typed arrays (JS -> GPU) give the managed overload's file, byte for byte.</summary>
    [TestMethod(Timeout = 1800000, Category = "BrowserOnly,NotOnWebGL")]
    public async Task VideoFileEncoderGpu_TypedArrayOverload_MatchesManaged()
    {
        if (!OperatingSystem.IsBrowser())
            throw new UnsupportedTestException("TypedArray overloads need a browser accelerator.");
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new VideoFileEncoderGpu(acc);
            await RunVideoFileTypedArrayMatchesManaged(enc);
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("browser")]
    private static async Task RunVideoFileTypedArrayMatchesManaged(VideoFileEncoderGpu enc)
    {
        const int w = 80, h = 50, n = 4;
        var frames = TestFrames(w, h, n, smooth: true);
        var options = new VideoFileEncodeOptions { Container = VideoContainer.WebM };
        byte[] managed = await enc.EncodeAsync(frames, w, h, 24, 1, null, options);
        var typed = new List<Float32Array>();
        try
        {
            foreach (var f in frames)
            {
                var j = new Float32Array(f.Length);
                j.Set(f);
                typed.Add(j);
            }
            using var file = await enc.EncodeAsync(typed, w, h, 24, 1, null, options);
            // Reading back to .NET is the TEST's terminal compare, not the product path.
            EqualBytes(managed, file.ReadBytes(), "VideoFileEncoderGpu TypedArray file vs managed");
        }
        finally { foreach (var j in typed) j.Dispose(); }
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
