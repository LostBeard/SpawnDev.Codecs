// Tests for WebPLossyEncoderGpu (Images/WebP/Lossy): the accelerator port of libwebp's lossy encoder must write the SAME BYTES
// as libwebp for the same YUV planes and settings.
//
// Fixtures: TestData/WebPLossyEnc (scripts/images/make_webp_lossy_encode_fixtures.py), the ones of methods 3 and 4 (the
// methods ported to the accelerator): libwebp's own imported planes in, Pillow's file as the expected bytes, qualities
// 0..100, 1x1 .. 640x480. The CPU reference (Vp8LossyEncoderReference) is byte-identical on the same set, so a failure here
// is in the kernels.
//
// Transparency (TestData/WebPLossyAlpha, scripts/images/make_webp_lossy_alpha_fixtures.py): RGBA sources (alpha ramps,
// cut-out masks, scattered transparent pixels; 1x1 .. 640x480) and Pillow's lossy file of each. Our file must be RIFF + VP8X
// + ALPH + VP8 with the VP8 chunk byte-identical to libwebp's and an ALPH that decodes to exactly the source alpha (in
// WebPDecoderGpu and, with SPAWNDEV_IMG_PYTHON set, in libwebp through Pillow).

using SpawnDev.Codecs.Images.WebP;
using SpawnDev.Codecs.Images.WebP.Lossy;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Timeout = 600000)]
    public async Task WebPLossyEncoderGpu_Methods3And4_ByteIdenticalWithLibwebp()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new WebPLossyEncoderGpu(acc);
            var text = System.Text.Encoding.UTF8.GetString(ReadEmbedded(WebPLossyEncPrefix + "manifest.tsv"));
            var yuvCache = new Dictionary<string, byte[]>();
            var failures = new List<string>();
            int identical = 0, total = 0;
            foreach (var line in text.Split('\n'))
            {
                var row = line.TrimEnd('\r');
                if (row.Length == 0) continue;
                var c = row.Split('\t');
                string name = c[0], src = c[1];
                int w = int.Parse(c[2]), h = int.Parse(c[3]), q = int.Parse(c[4]), m = int.Parse(c[5]);
                if (m < 3) continue;
                total++;
                try
                {
                    if (!yuvCache.TryGetValue(src, out var yuv))
                        yuvCache[src] = yuv = Gunzip(ReadEmbedded(WebPLossyEncPrefix + src + ".yuv.gz"));
                    var expected = ReadEmbedded(WebPLossyEncPrefix + name + ".webp");
                    var got = await enc.EncodeYuvAsync(yuv, w, h, new WebPLossyEncodeOptions { Quality = q, Method = m });
                    if (got.AsSpan().SequenceEqual(expected)) { identical++; continue; }
                    int first = 0, lim = Math.Min(got.Length, expected.Length);
                    while (first < lim && got[first] == expected[first]) first++;
                    failures.Add($"{name}: {got.Length} bytes vs libwebp {expected.Length}, first difference at byte {first}");
                }
                catch (Exception ex)
                {
                    failures.Add($"{name}: {ex.GetType().Name}: {ex.Message}");
                }
            }
            True(failures.Count == 0, $"{identical} of {total} byte-identical; {failures.Count} differ:\n" + string.Join("\n", failures.Take(25)));
            True(total >= 200, $"only {total} fixtures");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task WebPLossyEncoderGpu_UnportedMethods_AreRefused()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new WebPLossyEncoderGpu(acc);
            var rgba = new byte[16 * 16 * 4];
            foreach (int m in new[] { 0, 1, 2, 5, 6 })
            {
                bool threw = false;
                try { await enc.EncodeAsync(rgba, 16, 16, new WebPLossyEncodeOptions { Method = m }); }
                catch (NotSupportedException) { threw = true; }
                True(threw, $"method {m} must be refused, not encoded differently");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    private const string WebPLossyAlphaPrefix = "SpawnDev.Codecs.Demo.Shared.TestData.WebPLossyAlpha.";

    private sealed record WebPLossyAlphaCase(string Name, string Source, int Width, int Height, int Quality, int Method);

    private static List<WebPLossyAlphaCase> WebPLossyAlphaCases()
    {
        var list = new List<WebPLossyAlphaCase>();
        foreach (var line in System.Text.Encoding.UTF8.GetString(ReadEmbedded(WebPLossyAlphaPrefix + "manifest.tsv")).Split('\n'))
        {
            var row = line.TrimEnd('\r');
            if (row.Length == 0) continue;
            var c = row.Split('\t');
            list.Add(new WebPLossyAlphaCase(c[0], c[1], int.Parse(c[2]), int.Parse(c[3]), int.Parse(c[4]), int.Parse(c[5])));
        }
        return list;
    }

    /// <summary>The chunks of a RIFF WebP file (tag, payload offset, payload size); null with a reason when the container is wrong.</summary>
    private static List<(string Tag, int Offset, int Size)> WebPChunks(byte[] f, out string? error)
    {
        error = null;
        var list = new List<(string, int, int)>();
        if (f.Length < 12 || f[0] != 'R' || f[1] != 'I' || f[2] != 'F' || f[3] != 'F' || f[8] != 'W' || f[9] != 'E' || f[10] != 'B' || f[11] != 'P')
        {
            error = "no RIFF/WEBP header";
            return list;
        }
        if (BitConverter.ToInt32(f, 4) != f.Length - 8) { error = $"RIFF size {BitConverter.ToInt32(f, 4)} != {f.Length - 8}"; return list; }
        int p = 12;
        while (p + 8 <= f.Length)
        {
            string tag = System.Text.Encoding.ASCII.GetString(f, p, 4);
            int size = BitConverter.ToInt32(f, p + 4);
            if (size < 0 || p + 8 + size > f.Length) { error = $"chunk {tag} size {size} runs past the file"; return list; }
            list.Add((tag, p + 8, size));
            p += 8 + size + (size & 1);
        }
        if (p != f.Length) error = $"{f.Length - p} trailing bytes";
        return list;
    }

    /// <summary>
    /// Our file for a transparent picture next to libwebp's: RIFF + VP8X (alpha flag, canvas) + ALPH + VP8, the VP8 chunk
    /// byte-identical to libwebp's, and the alpha decoded (by WebPDecoderGpu) exactly the source's. Null when it all holds.
    /// </summary>
    private static async Task<string?> CheckWebPLossyAlphaFileAsync(WebPDecoderGpu dec, WebPLossyAlphaCase c, byte[] rgba, byte[] ours, byte[] libwebp)
    {
        var oc = WebPChunks(ours, out var oerr);
        if (oerr != null) return "container: " + oerr;
        var lc = WebPChunks(libwebp, out _);
        string tags = string.Join(",", oc.Select(x => x.Tag));
        if (tags != "VP8X,ALPH,VP8 ") return $"chunks {tags} (expected VP8X,ALPH,VP8 )";
        var x = oc[0];
        if (x.Size != 10 || ours[x.Offset] != 0x10 || ours[x.Offset + 1] != 0 || ours[x.Offset + 2] != 0 || ours[x.Offset + 3] != 0)
            return $"VP8X size {x.Size} flags {ours[x.Offset]:X2}";
        int cw = ours[x.Offset + 4] | ours[x.Offset + 5] << 8 | ours[x.Offset + 6] << 16;
        int ch = ours[x.Offset + 7] | ours[x.Offset + 8] << 8 | ours[x.Offset + 9] << 16;
        if (cw + 1 != c.Width || ch + 1 != c.Height) return $"VP8X canvas {cw + 1}x{ch + 1}";
        var ov = oc[2];
        var lv = lc.Single(k => k.Tag == "VP8 ");
        if (!ours.AsSpan(ov.Offset, ov.Size).SequenceEqual(libwebp.AsSpan(lv.Offset, lv.Size)))
            return $"VP8 chunk differs from libwebp's ({ov.Size} vs {lv.Size} bytes)";
        var decoded = await dec.DecodeAsync(ours);
        if (decoded.Info.Width != c.Width || decoded.Info.Height != c.Height) return $"decodes as {decoded.Info.Width}x{decoded.Info.Height}";
        for (int i = 0; i < c.Width * c.Height; i++)
            if (decoded.Pixels[4 * i + 3] != rgba[4 * i + 3])
                return $"alpha at pixel {i} ({i % c.Width},{i / c.Width}): {decoded.Pixels[4 * i + 3]} != source {rgba[4 * i + 3]}";
        return null;
    }

    [TestMethod(Timeout = 600000)]
    public async Task WebPLossyEncoderGpu_Transparency_Vp8ChunkIdenticalWithLibwebp_AlphaExact()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new WebPLossyEncoderGpu(acc);
            using var dec = new WebPDecoderGpu(acc);
            var cache = new Dictionary<string, byte[]>();
            var failures = new List<string>();
            var cases = WebPLossyAlphaCases();
            foreach (var c in cases)
            {
                try
                {
                    if (!cache.TryGetValue(c.Source, out var rgba))
                        cache[c.Source] = rgba = Gunzip(ReadEmbedded(WebPLossyAlphaPrefix + c.Source + ".rgba.gz"));
                    var ours = await enc.EncodeAsync(rgba, c.Width, c.Height, new WebPLossyEncodeOptions { Quality = c.Quality, Method = c.Method });
                    var diff = await CheckWebPLossyAlphaFileAsync(dec, c, rgba, ours, ReadEmbedded(WebPLossyAlphaPrefix + c.Name + ".webp"));
                    if (diff != null) failures.Add($"{c.Name}: {diff}");
                }
                catch (Exception ex)
                {
                    failures.Add($"{c.Name}: {ex.GetType().Name}: {ex.Message}");
                }
            }
            True(cases.Count >= 140, $"only {cases.Count} fixtures");
            True(failures.Count == 0, $"{failures.Count} of {cases.Count} differ:\n" + string.Join("\n", failures.Take(25)));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000)]
    public async Task WebPLossyEncoderGpu_Transparency_ArrayViewOverload_WritesOnlyTheFile()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new WebPLossyEncoderGpu(acc);
            using var dec = new WebPDecoderGpu(acc);
            var cache = new Dictionary<string, byte[]>();
            var failures = new List<string>();
            // one case per source picture, at libwebp's default quality
            var cases = WebPLossyAlphaCases().Where(k => k.Quality == 75 && k.Method == 4).ToList();
            foreach (var c in cases)
            {
                if (!cache.TryGetValue(c.Source, out var rgba))
                    cache[c.Source] = rgba = Gunzip(ReadEmbedded(WebPLossyAlphaPrefix + c.Source + ".rgba.gz"));
                using var dIn = DeviceVerify.Upload(acc, rgba, slack: 7);
                var libwebp = ReadEmbedded(WebPLossyAlphaPrefix + c.Name + ".webp");
                int room = libwebp.Length * 2 + 4096;
                using var dOut = DeviceVerify.Poisoned(acc, room + 64);
                int length = await enc.EncodeAsync(dIn.View, c.Width, c.Height, dOut.View.SubView(0, room), new WebPLossyEncodeOptions { Quality = 75, Method = 4 });
                Equal(null, await DeviceVerify.AllEqualAsync(acc, dOut.View.SubView(length, 64), 64, 0xAB), $"{c.Name}: bytes past the end of the file");
                var ours = await DeviceTransfer.DownloadAsync(dOut.View, length);
                var diff = await CheckWebPLossyAlphaFileAsync(dec, c, rgba, ours, libwebp);
                if (diff != null) failures.Add($"{c.Name}: {diff}");
                // the managed overload writes the same file
                var managed = await enc.EncodeAsync(rgba, c.Width, c.Height, new WebPLossyEncodeOptions { Quality = 75, Method = 4 });
                if (!managed.AsSpan().SequenceEqual(ours)) failures.Add($"{c.Name}: managed overload wrote a different file");
            }
            True(cases.Count >= 19, $"only {cases.Count} sources");
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000)]
    public async Task WebPLossyEncoderGpu_Transparency_PillowDecodesOurFiles_LikeLibwebpsOwn()
    {
        PillowOracle.RequireAvailable();
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new WebPLossyEncoderGpu(acc);
            var cache = new Dictionary<string, byte[]>();
            var failures = new List<string>();
            var cases = WebPLossyAlphaCases().Where(k => k.Method == 4).ToList();
            foreach (var c in cases)
            {
                if (!cache.TryGetValue(c.Source, out var rgba))
                    cache[c.Source] = rgba = Gunzip(ReadEmbedded(WebPLossyAlphaPrefix + c.Source + ".rgba.gz"));
                var ours = await enc.EncodeAsync(rgba, c.Width, c.Height, new WebPLossyEncodeOptions { Quality = c.Quality, Method = 4 });
                var p = PillowOracle.Decode(ours, ".webp");      // Unsupported (not passed) without SPAWNDEV_IMG_PYTHON
                var r = PillowOracle.Decode(ReadEmbedded(WebPLossyAlphaPrefix + c.Name + ".webp"), ".webp");
                if (p.Width != c.Width || p.Height != c.Height) { failures.Add($"{c.Name}: Pillow sees {p.Width}x{p.Height}"); continue; }
                // same colour as libwebp's own file (the VP8 chunks are identical), alpha exactly the source's
                var diff = ComparePixels(r.FramesRgba[0], p.FramesRgba[0], 4, c.Width);
                if (diff != null) failures.Add($"{c.Name}: vs libwebp's file decoded by libwebp: {diff}");
                for (int i = 0; i < c.Width * c.Height; i++)
                    if (p.FramesRgba[0][4 * i + 3] != rgba[4 * i + 3]) { failures.Add($"{c.Name}: libwebp decodes alpha {p.FramesRgba[0][4 * i + 3]} at pixel {i}, source {rgba[4 * i + 3]}"); break; }
            }
            True(failures.Count == 0, $"{failures.Count} of {cases.Count}:\n" + string.Join("\n", failures.Take(25)));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
