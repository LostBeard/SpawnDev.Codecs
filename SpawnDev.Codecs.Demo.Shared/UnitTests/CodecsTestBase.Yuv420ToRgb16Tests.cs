// Yuv420ToRgb16Gpu: YUV 4:2:0 -> RGB exactly as ComfyUI's LoadVideo (Docs/comfyui-loadvideo-yuv-to-rgb.md).
//
// Oracles (scripts/video/make_loadvideo_rgb_fixtures.py):
//  - OddSizes: random pictures at the sizes where the rules differ (n < 4 chroma samples on an axis, odd sizes, even and
//    odd 16.16 steps, positions that snap to the next sample); truth = MD5 of `ffmpeg -sws_flags bilinear -pix_fmt
//    gbrp16le` output re-interleaved R,G,B uint16 (ffmpeg as a black box; it equals ComfyUI's output on every frame).
//  - ComfyUiLoadVideo: the three ComfyUI-written VP9 WebMs, decoded by Vp9DecoderGpu and converted; truth = MD5 per
//    frame of ComfyUI's own LoadVideo -> GetVideoComponents IMAGE tensor (float32, dumped by Lore with ComfyUI).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Images.Yuv;
using SpawnDev.Codecs.Video.Vp9.Decoder;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private const string Yuv420Rgb16Prefix = "SpawnDev.Codecs.Demo.Shared.TestData.Yuv420Rgb16.";

    [TestMethod(Timeout = 600000)]
    public async Task Yuv420ToRgb16Gpu_OddSizes_MatchFfmpegBilinear()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var conv = new Yuv420ToRgb16Gpu(acc);
            await conv.EnsureKernelsAsync();
            var rows = System.Text.Encoding.ASCII.GetString(ReadEmbedded(Yuv420Rgb16Prefix + "manifest.tsv"))
                .Split('\n').Select(l => l.Trim()).Where(l => l.Length > 0).Select(l => l.Split('\t')).ToList();
            var failures = new List<string>();
            foreach (var r in rows)
            {
                int w = int.Parse(r[1]), h = int.Parse(r[2]);
                var yuv = ReadEmbedded(Yuv420Rgb16Prefix + r[0] + ".yuv");
                using var dYuv = acc.Allocate1D(yuv);
                using var dRgb = acc.Allocate1D<int>((long)w * h * 3);
                conv.ConvertToUInt16(dYuv.View, w, h, dRgb.View);
                var rgb = await dRgb.View.CopyToHostAsync();
                var bytes = new byte[rgb.Length * 2];
                for (int i = 0; i < rgb.Length; i++) { bytes[2 * i] = (byte)rgb[i]; bytes[2 * i + 1] = (byte)(rgb[i] >> 8); }
                if (Md5Hex(bytes) != r[3]) failures.Add($"{r[0]}: MD5 differs");
            }
            True(rows.Count >= 15, $"only {rows.Count} pictures");
            True(failures.Count == 0, $"{failures.Count} of {rows.Count} differ:\n" + string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL,GiantShader")]
    public async Task Yuv420ToRgb16Gpu_ComfyUiLoadVideo_BitExact()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            var conv = new Yuv420ToRgb16Gpu(acc);
            await conv.EnsureKernelsAsync();
            var failures = new List<string>();
            int frames = 0;
            foreach (var name in new[] { "comfyui_vp9_512x512_crf32", "comfyui_vp9_1024x576_crf32", "comfyui_vp9_rgba_512x320_crf32" })
            {
                var want = Md5Lines(ReadEmbedded(Vp9ComfyPrefix + name + ".webm.loadvideo-f32.md5"));
                using var dec = new Vp9DecoderGpu(acc);
                int n = 0;
                MemoryBuffer1D<float, Stride1D.Dense>? dRgb = null;
                try
                {
                    await foreach (var frame in dec.DecodeFileAsync(ReadEmbedded(Vp9ComfyPrefix + name + ".webm")))
                    {
                        var f = frame.Format;
                        int luma = f.Width * f.Height;
                        using var dPlanes = acc.Allocate1D(frame.Planes);
                        dRgb ??= acc.Allocate1D<float>((long)luma * 3);
                        conv.ConvertToFloat(dPlanes.View, f.Width, f.Height, dRgb.View);
                        var rgb = await dRgb.View.CopyToHostAsync();
                        var bytes = new byte[rgb.Length * 4];
                        Buffer.BlockCopy(rgb, 0, bytes, 0, bytes.Length);
                        if (n >= want.Length) failures.Add($"{name}: more frames than ComfyUI");
                        else if (Md5Hex(bytes) != want[n]) failures.Add($"{name} frame {n}: differs from ComfyUI");
                        n++;
                        frames++;
                    }
                }
                finally { dRgb?.Dispose(); }
                if (n != want.Length) failures.Add($"{name}: {n} frames, ComfyUI {want.Length}");
            }
            Console.WriteLine($"{acc.AcceleratorType}: {frames} frames == ComfyUI LoadVideo");
            True(frames == 108, $"{frames} frames");
            True(failures.Count == 0, $"{failures.Count} mismatches:\n" + string.Join("\n", failures.Take(40)));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>
    /// Frame rate from a WebM DefaultDuration equals what ffprobe reports (avg_frame_rate, which ComfyUI's LoadVideo
    /// returns as a float) for libvpx-vp9 WebMs written by ffmpeg at these rates (measured 2026-11-11).
    /// </summary>
    [TestMethod(Category = "CpuOnly")]
    public void WebMVideoLoaderGpu_FrameRateFromDefaultDuration_MatchesFfprobe()
    {
        var cases = new (ulong Ns, double Fps)[]
        {
            (41666666, 24.0), (40000000, 25.0), (33366666, 30000.0 / 1001), (41708333, 24000.0 / 1001), (16666666, 60.0),
            (66666666, 15.0), (80000000, 12.5), (20000000, 50.0), (125000000, 8.0),
        };
        foreach (var (ns, fps) in cases)
            True(Video.WebMVideoLoaderGpu.RateFromDuration(ns) == fps, $"{ns} ns: {Video.WebMVideoLoaderGpu.RateFromDuration(ns)}, want {fps}");
    }

    /// <summary>
    /// <see cref="Video.WebMVideoLoaderGpu"/> on a ComfyUI VP9 video with ComfyUI's Opus packets (TestData/LoadVideo, see
    /// scripts/video/make_loadvideo_rgb_fixtures.py): frames bit-exact with ComfyUI's LoadVideo, fps 24, audio with
    /// ComfyUI's sample count (pre-skip and end trimming) and within the fixed-point vs float Opus decoder gap of its
    /// values.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL,GiantShader")]
    public async Task WebMVideoLoaderGpu_MatchesComfyUiLoadVideo()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            const string lv = "SpawnDev.Codecs.Demo.Shared.TestData.LoadVideo.";
            var loaded = await new Video.WebMVideoLoaderGpu(acc).LoadAsync(ReadEmbedded(lv + "vp9_with_comfyui_opus.webm"));
            var want = Md5Lines(ReadEmbedded(Vp9ComfyPrefix + "comfyui_vp9_512x512_crf32.webm.loadvideo-f32.md5"));
            var failures = new List<string>();
            Equal(512, loaded.Width);
            Equal(512, loaded.Height);
            True(loaded.FrameRate == 24.0, $"fps {loaded.FrameRate}");
            Equal(want.Length, loaded.Frames.Count);
            for (int n = 0; n < loaded.Frames.Count; n++)
            {
                var bytes = new byte[loaded.Frames[n].Length * 4];
                Buffer.BlockCopy(loaded.Frames[n], 0, bytes, 0, bytes.Length);
                if (Md5Hex(bytes) != want[n]) failures.Add($"frame {n}: differs from ComfyUI");
            }
            var audio = loaded.Audio ?? throw new Exception("no audio");
            Equal(48000, audio.SampleRate);
            Equal(2, audio.Channels);
            var comfyBytes = ReadEmbedded(lv + "comfyui_opus_audio_f32_2x96000.raw");
            var comfy = new float[comfyBytes.Length / 4];
            Buffer.BlockCopy(comfyBytes, 0, comfy, 0, comfyBytes.Length);
            int per = comfy.Length / 2;
            Equal(per * 2, audio.Samples.Length);
            // Same samples as ComfyUI (count = trimming), values within the measured fixed-point vs float decoder gap:
            // OpusDecoderGpu is libopus 1.5.2 FIXED POINT (bit-exact, its own tests); ComfyUI's ffmpeg decodes Opus in
            // float. Measured 2026-11-11 on this stream: max |diff| 3.283e-4 (about -70 dB) on every backend.
            double maxDiff = 0;
            for (int s = 0; s < per && s * 2 + 1 < audio.Samples.Length; s++)
                for (int c = 0; c < 2; c++) maxDiff = Math.Max(maxDiff, Math.Abs(audio.Samples[s * 2 + c] - comfy[c * per + s]));
            if (maxDiff > 4e-4) failures.Add($"audio vs ComfyUI: max |diff| {maxDiff:E3} > 4e-4 (the fixed vs float decoder gap)");
            Console.WriteLine($"{acc.AcceleratorType}: {loaded.Frames.Count} frames == ComfyUI, audio vs ComfyUI max |diff| {maxDiff:E3}");
            True(failures.Count == 0, $"{failures.Count} mismatches:\n" + string.Join("\n", failures.Take(40)));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
