// Tests for WebPLosslessEncoderGpu.EncodeAnimationAsync (animated lossless WebP).
//
// Sources (TestData/WebP/animenc_*, scripts/images/make_webp_anim_encode_fixtures.py): per animation the pixels of every frame,
// the frame table (rectangle, duration, dispose, blend) and the TRUTH: the canvas as it must look while each frame is shown,
// computed by an independent numpy compositor that the generator cross-checks against libwebp. The encoder is given the
// frames and must produce a file that
//   (a) libwebp's animation decoder (through Pillow, scripts/images/verify_webp_files.py) shows frame by frame as exactly those
//       canvases, with the durations, loop count and background colour that were asked for, and metadata libwebp's demuxer sees,
//   (b) WebPDecoderGpu decodes (DecodeFrameAsync) to the same canvases, and ReadFramesAsync / ReadFileInfoAsync report the
//       frame table, loop count and background that were asked for,
//   (c) an independent pure-Python VP8L parser (webp_ref.py) accepts every frame stream of.
// Needs Pillow for (a) and (c): set CODECS_PILLOW_PYTHON (see CodecsTestBase.WebPLosslessEncoderGpuTests.cs).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Images;
using SpawnDev.Codecs.Images.WebP;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private sealed record AnimEncodeSource(string Name, int CanvasW, int CanvasH, ImageFrameInfo[] Frames, byte[][] FramePixels, byte[][] Canvases);

    private static List<AnimEncodeSource> WebPAnimEncodeSources()
    {
        var text = System.Text.Encoding.UTF8.GetString(ReadEmbedded(WebPFixturePrefix + "animenc_manifest.tsv"));
        var list = new List<AnimEncodeSource>();
        foreach (var raw in text.Split('\n'))
        {
            var line = raw.TrimEnd('\r');
            if (line.Length == 0) continue;
            var c = line.Split('\t');
            string name = c[0];
            int cw = int.Parse(c[1]), ch = int.Parse(c[2]), n = int.Parse(c[3]);
            var rows = System.Text.Encoding.UTF8.GetString(ReadEmbedded(WebPFixturePrefix + "animenc_" + name + ".frames.tsv"))
                .Split('\n').Select(l => l.TrimEnd('\r')).Where(l => l.Length > 0).Select(l => l.Split('\t').Select(int.Parse).ToArray()).ToArray();
            if (rows.Length != n) throw new InvalidOperationException($"{name}: manifest says {n} frames, frames.tsv has {rows.Length}");
            var frames = new ImageFrameInfo[n];
            var pixels = new byte[n][];
            var canvases = new byte[n][];
            for (int k = 0; k < n; k++)
            {
                var r = rows[k];
                frames[k] = new ImageFrameInfo(k, r[0], r[1], r[2], r[3], r[4],
                    r[5] == 1 ? FrameDispose.Background : FrameDispose.None, r[6] == 1 ? FrameBlend.Over : FrameBlend.Source);
                pixels[k] = Gunzip(ReadEmbedded(WebPFixturePrefix + "animenc_" + name + ".f" + k + ".rgba8.gz"));
                canvases[k] = Gunzip(ReadEmbedded(WebPFixturePrefix + "animenc_" + name + ".canvas" + k + ".rgba8.gz"));
            }
            list.Add(new AnimEncodeSource(name, cw, ch, frames, pixels, canvases));
        }
        if (list.Count == 0) throw new InvalidOperationException("The animation encoder manifest is empty.");
        return list;
    }

    /// <summary>Writes the sidecars verify_webp_files.py needs to check an animation (.size, .anim, .canvases, metadata).</summary>
    private static async Task WriteAnimationForVerifyAsync(
        string stem, byte[] file, AnimEncodeSource src, WebPAnimationEncodeOptions options, byte[][] canvases)
    {
        await File.WriteAllBytesAsync(stem + ".webp", file);
        await File.WriteAllTextAsync(stem + ".size", $"{src.CanvasW} {src.CanvasH}");
        uint bg = options.BackgroundColor;
        await File.WriteAllTextAsync(stem + ".anim",
            $"loop {options.LoopCount}\nbg {(bg >> 16) & 0xFF} {(bg >> 8) & 0xFF} {bg & 0xFF} {bg >> 24}\ndur {string.Join(" ", src.Frames.Select(f => f.DurationMs))}\n");
        await File.WriteAllBytesAsync(stem + ".canvases", canvases.SelectMany(c => c).ToArray());
        if (options.IccProfile != null) await File.WriteAllBytesAsync(stem + ".icc", options.IccProfile);
        if (options.Exif != null) await File.WriteAllBytesAsync(stem + ".exif", options.Exif);
        if (options.Xmp != null) await File.WriteAllBytesAsync(stem + ".xmp", options.Xmp);
    }

    private static bool AnyAlphaBelow255(byte[][] framePixels)
    {
        foreach (var px in framePixels)
            for (int i = 3; i < px.Length; i += 4)
                if (px[i] != 255) return true;
        return false;
    }

    /// <summary>The expected canvases when the encoder is told it may drop RGB under alpha 0.</summary>
    private static byte[][] ClearRgbUnderAlpha0(byte[][] canvases) => canvases.Select(ZeroRgbUnderAlpha0).ToArray();

    private async Task<List<string>> CheckAnimationAsync(
        WebPLosslessEncoderGpu enc, WebPDecoderGpu dec, AnimEncodeSource src, WebPAnimationEncodeOptions options, string tag, string? keepStem)
    {
        var failures = new List<string>();
        byte[] file = await enc.EncodeAnimationAsync(
            src.FramePixels.Select(p => (ReadOnlyMemory<byte>)p).ToArray(), src.Frames, src.CanvasW, src.CanvasH, options);
        int max = WebPLosslessEncoderGpu.GetMaxAnimationEncodedSize(src.Frames, options);
        if (file.Length > max) failures.Add($"{tag}: {file.Length} bytes exceeds GetMaxAnimationEncodedSize {max}");

        var expectedCanvases = options.Exact ? src.Canvases : ClearRgbUnderAlpha0(src.Canvases);
        var info = await dec.ReadInfoAsync(file);
        if (info.Width != src.CanvasW || info.Height != src.CanvasH) failures.Add($"{tag}: info size {info.Width}x{info.Height}");
        if (info.FrameCount != src.Frames.Length) failures.Add($"{tag}: FrameCount {info.FrameCount}");
        if (info.HasAlpha != AnyAlphaBelow255(src.FramePixels)) failures.Add($"{tag}: HasAlpha {info.HasAlpha}");
        var fi = await dec.ReadFileInfoAsync(file);
        if (!fi.IsLossless || !fi.IsExtended) failures.Add($"{tag}: IsLossless {fi.IsLossless} IsExtended {fi.IsExtended}");
        if (fi.LoopCount != options.LoopCount) failures.Add($"{tag}: LoopCount {fi.LoopCount}, expected {options.LoopCount}");
        if (fi.BackgroundColor != options.BackgroundColor) failures.Add($"{tag}: BackgroundColor 0x{fi.BackgroundColor:X8}, expected 0x{options.BackgroundColor:X8}");
        if (fi.HasIccProfile != (options.IccProfile != null) || fi.HasExif != (options.Exif != null) || fi.HasXmp != (options.Xmp != null))
            failures.Add($"{tag}: metadata flags ICC {fi.HasIccProfile} EXIF {fi.HasExif} XMP {fi.HasXmp}");

        var frames = await dec.ReadFramesAsync(file);
        if (frames.Length != src.Frames.Length) failures.Add($"{tag}: ReadFrames gave {frames.Length} frames");
        else
            for (int k = 0; k < frames.Length; k++)
            {
                var a = src.Frames[k];
                var b = frames[k];
                if (a.X != b.X || a.Y != b.Y || a.Width != b.Width || a.Height != b.Height || a.DurationMs != b.DurationMs || a.Dispose != b.Dispose || a.Blend != b.Blend)
                    failures.Add($"{tag}: frame {k} table {b} differs from the request {a}");
            }
        for (int k = 0; k < src.Frames.Length; k++)
        {
            var canvas = await dec.DecodeFrameAsync(file, k);
            var diff = ComparePixels(expectedCanvases[k], canvas.Pixels, 4, src.CanvasW);
            if (diff != null) failures.Add($"{tag} [our decoder] canvas {k}: {diff}");
        }
        if (keepStem != null) await WriteAnimationForVerifyAsync(keepStem, file, src, options, expectedCanvases);
        return failures;
    }

    [TestMethod(Timeout = 600000)]
    public async Task WebPLosslessEncoderGpu_Animation_AllSources_MatchLibwebpAndOurDecoder_AtSeveralQualities()
    {
        string dir = WebPEncoderOutputDir("animation");
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        var failures = new List<string>();
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            using var dec = new WebPDecoderGpu(acc);
            int variant = 0;
            foreach (var src in WebPAnimEncodeSources())
            {
                foreach (int q in new[] { 0, 40, 75, 100 })
                {
                    var options = new WebPAnimationEncodeOptions
                    {
                        Quality = q,
                        Exact = true,
                        LoopCount = variant % 4 == 0 ? 0 : variant * 41 + 1, // > 255 for most: both bytes of the 16-bit field
                        BackgroundColor = variant % 2 == 0 ? 0u : 0xFF3366CCu - (uint)variant * 0x01010101u,
                    };
                    variant++;
                    string tag = $"{src.Name}_q{q}";
                    try { failures.AddRange(await CheckAnimationAsync(enc, dec, src, options, tag, Path.Combine(dir, tag))); }
                    catch (Exception ex) { failures.Add($"{tag}: {ex.GetType().Name}: {ex.Message}"); }
                }
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
        True(failures.Count == 0, $"{failures.Count} animation encoder failures:\n" + string.Join("\n", failures));
        var verify = await RunPillowVerifyAsync(dir);
        True(verify.Contains("SUMMARY"), "verification script did not finish");
    }

    [TestMethod]
    public async Task WebPLosslessEncoderGpu_Animation_ExactFalse_ClearsRgbUnderAlpha0_Only()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            using var dec = new WebPDecoderGpu(acc);
            var src = WebPAnimEncodeSources().Single(s => s.Name == "rects_64x48");
            int junk = 0;
            foreach (var c in src.Canvases)
                for (int i = 3; i < c.Length; i += 4)
                    if (c[i] == 0 && (c[i - 1] | c[i - 2] | c[i - 3]) != 0) junk++;
            True(junk > 20, "the source canvases must have transparent pixels with non-zero RGB, else this test proves nothing");

            var exact = await CheckAnimationAsync(enc, dec, src, new WebPAnimationEncodeOptions { Exact = true }, "exact", null);
            True(exact.Count == 0, string.Join("\n", exact));
            var loose = await CheckAnimationAsync(enc, dec, src, new WebPAnimationEncodeOptions { Exact = false }, "loose", null);
            True(loose.Count == 0, string.Join("\n", loose));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task WebPLosslessEncoderGpu_Animation_Metadata_IsWrittenAndOpensInLibwebp()
    {
        byte[] icc = new byte[301];
        for (int i = 0; i < icc.Length; i++) icc[i] = (byte)(i * 7 + 1);
        byte[] exif = new byte[] { (byte)'I', (byte)'I', 42, 0, 8, 0, 0, 0, 1, 0, 0x12, 0x01, 3, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0 };
        byte[] xmp = System.Text.Encoding.UTF8.GetBytes("<x:xmpmeta xmlns:x='adobe:ns:meta/'><rdf/></x:xmpmeta>"); // odd length: padded
        string dir = WebPEncoderOutputDir("animation_meta");
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        var failures = new List<string>();
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            using var dec = new WebPDecoderGpu(acc);
            var combos = new (string Name, byte[]? Icc, byte[]? Exif, byte[]? Xmp)[]
            {
                ("all", icc, exif, xmp), ("icc", icc, null, null), ("exif", null, exif, null), ("xmp", null, null, xmp),
                ("exif_xmp", null, exif, xmp), ("icc_xmp", icc, null, xmp),
            };
            foreach (var src in WebPAnimEncodeSources().Where(s => s.Name is "rects_64x48" or "thin_7x9" or "single_32x24"))
            {
                foreach (var (cname, i, e, x) in combos)
                {
                    var options = new WebPAnimationEncodeOptions { Quality = 50, LoopCount = 2, IccProfile = i, Exif = e, Xmp = x };
                    string tag = $"{src.Name}_{cname}";
                    try
                    {
                        failures.AddRange(await CheckAnimationAsync(enc, dec, src, options, tag, Path.Combine(dir, tag)));
                        var file = await File.ReadAllBytesAsync(Path.Combine(dir, tag + ".webp"));
                        var md = await dec.ReadMetadataAsync(file);
                        if (i != null) EqualBytes(i, md.IccProfile!, tag + " ICC"); else True(md.IccProfile is null, tag + " ICC absent");
                        if (e != null) EqualBytes(e, md.Exif!, tag + " EXIF"); else True(md.Exif is null, tag + " EXIF absent");
                        if (x != null) EqualBytes(x, md.Xmp!, tag + " XMP"); else True(md.Xmp is null, tag + " XMP absent");
                        // the chunk ORDER matters to some readers: VP8X, ICCP, ANIM, ANMF..., EXIF, XMP
                        var order = WebPChunksOf(file).Select(c => c.Fcc).ToList();
                        var expectedOrder = new List<string> { "VP8X" };
                        if (i != null) expectedOrder.Add("ICCP");
                        expectedOrder.Add("ANIM");
                        expectedOrder.AddRange(Enumerable.Repeat("ANMF", src.Frames.Length));
                        if (e != null) expectedOrder.Add("EXIF");
                        if (x != null) expectedOrder.Add("XMP ");
                        if (!order.SequenceEqual(expectedOrder)) failures.Add($"{tag}: chunk order {string.Join(",", order)}, expected {string.Join(",", expectedOrder)}");
                    }
                    catch (Exception ex) { failures.Add($"{tag}: {ex.GetType().Name}: {ex.Message}"); }
                }
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
        True(failures.Count == 0, $"{failures.Count} animation metadata failures:\n" + string.Join("\n", failures));
        await RunPillowVerifyAsync(dir);
    }

    [TestMethod]
    public async Task WebPLosslessEncoderGpu_Animation_ArrayViewOverload_WritesOnlyTheFile()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            using var dec = new WebPDecoderGpu(acc);
            var src = WebPAnimEncodeSources().Single(s => s.Name == "rects_64x48");
            var inputs = new List<MemoryBuffer1D<byte, Stride1D.Dense>>();
            try
            {
                foreach (var px in src.FramePixels)
                {
                    var b = acc.Allocate1D<byte>(px.Length + 3);
                    DeviceTransfer.Upload(b.View, px);
                    inputs.Add(b);
                }
                int max = WebPLosslessEncoderGpu.GetMaxAnimationEncodedSize(src.Frames);
                using var dOut = acc.Allocate1D<byte>(max + 64);
                DeviceTransfer.Upload(dOut.View, Poison(max + 64));
                int length = await enc.EncodeAnimationAsync(inputs.Select(b => b.View).ToArray(), src.Frames, src.CanvasW, src.CanvasH, dOut.View);
                True(length > 100 && length <= max, $"length {length} within (100, {max}]");
                var written = await DeviceTransfer.DownloadAsync(dOut.View, length);
                var tail = await dOut.View.SubView(length, 64).CopyToHostAsync();
                for (int i = 0; i < tail.Length; i++)
                    if (tail[i] != 0xAB) throw new Exception($"byte {length + i} past the end of the file was written");
                for (int k = 0; k < src.Frames.Length; k++)
                {
                    var canvas = await dec.DecodeFrameAsync(written, k);
                    Equal(null, ComparePixels(src.Canvases[k], canvas.Pixels, 4, src.CanvasW), $"canvas {k}");
                }
            }
            finally { foreach (var b in inputs) b.Dispose(); }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task WebPLosslessEncoderGpu_Animation_BadArguments_AreRejected()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            async Task<bool> Throws<T>(Func<Task> action) where T : Exception
            {
                try { await action(); }
                catch (T) { return true; }
                return false;
            }
            var px = new ReadOnlyMemory<byte>[] { new byte[8 * 8 * 4] };
            ImageFrameInfo F(int x = 0, int y = 0, int w = 8, int h = 8, int dur = 10, FrameDispose d = FrameDispose.None, FrameBlend b = FrameBlend.Source)
                => new(0, x, y, w, h, dur, d, b);
            var ok = new[] { F() };
            True(await Throws<ArgumentException>(() => enc.EncodeAnimationAsync(System.Array.Empty<ReadOnlyMemory<byte>>(), System.Array.Empty<ImageFrameInfo>(), 8, 8)), "no frames");
            True(await Throws<ArgumentException>(() => enc.EncodeAnimationAsync(px, new[] { F(), F() }, 8, 8)), "pixel buffer count differs from frame count");
            True(await Throws<ArgumentOutOfRangeException>(() => enc.EncodeAnimationAsync(px, ok, 0, 8)), "canvas width 0");
            True(await Throws<ArgumentOutOfRangeException>(() => enc.EncodeAnimationAsync(px, ok, 8, (1 << 24) + 1)), "canvas height too large");
            True(await Throws<ArgumentException>(() => enc.EncodeAnimationAsync(px, new[] { F(x: 1) }, 16, 16)), "odd x offset");
            True(await Throws<ArgumentException>(() => enc.EncodeAnimationAsync(px, new[] { F(y: 3) }, 16, 16)), "odd y offset");
            True(await Throws<ArgumentException>(() => enc.EncodeAnimationAsync(px, new[] { F(x: -2) }, 16, 16)), "negative offset");
            True(await Throws<ArgumentException>(() => enc.EncodeAnimationAsync(px, new[] { F(x: 10) }, 16, 16)), "frame sticks out of the canvas");
            True(await Throws<ArgumentException>(() => enc.EncodeAnimationAsync(px, new[] { F(dur: 1 << 24) }, 16, 16)), "duration 2^24");
            True(await Throws<ArgumentException>(() => enc.EncodeAnimationAsync(px, new[] { F(dur: -1) }, 16, 16)), "negative duration");
            True(await Throws<ArgumentException>(() => enc.EncodeAnimationAsync(px, new[] { F(d: FrameDispose.Previous) }, 16, 16)), "FrameDispose.Previous does not exist in WebP");
            True(await Throws<ArgumentException>(() => enc.EncodeAnimationAsync(px, new[] { F(w: 9) }, 16, 16)), "pixel buffer shorter than the frame");
            True(await Throws<ArgumentOutOfRangeException>(() => enc.EncodeAnimationAsync(px, ok, 8, 8, new WebPAnimationEncodeOptions { LoopCount = 65536 })), "loop count 65536");
            True(await Throws<ArgumentOutOfRangeException>(() => enc.EncodeAnimationAsync(px, ok, 8, 8, new WebPAnimationEncodeOptions { Quality = 101 })), "quality 101");

            using var dIn = acc.Allocate1D<byte>(8 * 8 * 4);
            using var dOut = acc.Allocate1D<byte>(100);
            True(await Throws<ArgumentException>(() => enc.EncodeAnimationAsync(new[] { dIn.View }, ok, 8, 8, dOut.View)), "output smaller than GetMaxAnimationEncodedSize");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
