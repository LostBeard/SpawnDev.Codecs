// Tests for the animated lossy WebP encoder (WebPLossyEncoderGpu.EncodeAnimationAsync, a port of libwebp 1.6.0's
// WebPAnimEncoder) and the FilterStrength option of the still lossy encoder that it uses for blended frames.
//
// Fixtures (scripts/images/make_webp_anim_lossy_fixtures.py):
//   TestData/WebPAnimLossy       source frames + Pillow's animated lossy save of them (what ComfyUI's Save Animated WEBP
//                                writes): moving patches, sprites over a photo (flattened blocks), repeated frames
//                                (skipped, durations merged), all-equal frames (a still file), transparent sprites
//                                (dispose to background), a lone sub-rectangle frame (OptimizeSingleFrame), odd sizes,
//                                1x1, scene cuts, 24-bit duration overflow (libwebp's 1x1 split frame), EXIF, loop count
//                                and background, kmin / kmax variants.
//   TestData/WebPLossyEncFilter  libwebp's imported planes and its file for filter_strength 0 / 25 / 60 / 100.
//
// What must hold:
//   * every animation whose libwebp file has no ALPH chunk: OUR FILE IS BYTE-IDENTICAL;
//   * an animation with ALPH chunks (transparent pixels in a frame): our ALPH is coded by our own VP8L coder, so the bytes
//     differ there, and only there: same chunk list, same VP8X / ANIM / EXIF, same ANMF parameters (offset, size, duration,
//     disposal, blending), byte-identical VP8 chunks, and ALPH chunks that decode to the same plane;
//   * libwebp (Pillow) decodes our files to the same frames as its own (desktop, SPAWNDEV_IMG_PYTHON).

using SpawnDev.Codecs.Images.WebP;
using SpawnDev.Codecs.Images.WebP.Lossy;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private const string WebPAnimLossyPrefix = "SpawnDev.Codecs.Demo.Shared.TestData.WebPAnimLossy.";
    private const string WebPLossyFilterPrefix = "SpawnDev.Codecs.Demo.Shared.TestData.WebPLossyEncFilter.";

    private sealed record WebPAnimLossyCase(string Name, string Source, int Width, int Height, int Frames, int[] Durations,
        int Quality, int Method, int? Kmin, int? Kmax, bool Exif, int Loop, uint Background)
    {
        public WebPLossyEncodeOptions Options => new() { Quality = Quality, Method = Method };

        public WebPLossyAnimationOptions Animation(byte[]? exif) => new()
        {
            LoopCount = Loop, BackgroundColor = Background, Exif = exif,
            KeyframeMin = Kmin ?? 3, KeyframeMax = Kmax ?? 5,   // Pillow's lossy defaults when not given
        };
    }

    private static List<WebPAnimLossyCase> WebPAnimLossyCases()
    {
        var list = new List<WebPAnimLossyCase>();
        foreach (var line in System.Text.Encoding.UTF8.GetString(ReadEmbedded(WebPAnimLossyPrefix + "manifest.tsv")).Split('\n'))
        {
            var row = line.TrimEnd('\r');
            if (row.Length == 0) continue;
            var c = row.Split('\t');
            int kmin = int.Parse(c[8]), kmax = int.Parse(c[9]);
            list.Add(new WebPAnimLossyCase(c[0], c[1], int.Parse(c[2]), int.Parse(c[3]), int.Parse(c[4]),
                c[5].Split(',').Select(int.Parse).ToArray(), int.Parse(c[6]), int.Parse(c[7]),
                kmin == -999 ? null : kmin, kmax == -999 ? null : kmax, c[10] == "1", int.Parse(c[11]),
                Convert.ToUInt32(c[12], 16)));
        }
        return list;
    }

    private static List<ReadOnlyMemory<byte>> WebPAnimFrames(WebPAnimLossyCase c, Dictionary<string, byte[]> cache)
    {
        if (!cache.TryGetValue(c.Source, out var raw))
            cache[c.Source] = raw = Gunzip(ReadEmbedded(WebPAnimLossyPrefix + c.Source + ".frames.gz"));
        int size = c.Width * c.Height * 4;
        True(raw.Length == size * c.Frames, $"{c.Name}: {raw.Length} source bytes for {c.Frames} frames of {size}");
        return Enumerable.Range(0, c.Frames).Select(k => new ReadOnlyMemory<byte>(raw, k * size, size)).ToList();
    }

    /// <summary>The chunks inside f[start, end) (tag, payload offset, size).</summary>
    private static List<(string Tag, int Offset, int Size)> WebPInnerChunks(byte[] f, int start, int end)
    {
        var list = new List<(string, int, int)>();
        int p = start;
        while (p + 8 <= end)
        {
            int size = BitConverter.ToInt32(f, p + 4);
            list.Add((System.Text.Encoding.ASCII.GetString(f, p, 4), p + 8, size));
            p += 8 + size + (size & 1);
        }
        return list;
    }

    /// <summary>
    /// Our animated file against libwebp's when only the ALPH coding may differ: null when the container, every frame's
    /// parameters and VP8 chunk are identical and every ALPH decodes to the same plane; otherwise the first difference.
    /// </summary>
    private static async Task<string?> CompareWebPAnimExceptAlphCoding(WebPDecoderGpu dec, byte[] ours, byte[] libwebp)
    {
        var oc = WebPChunks(ours, out var oerr);
        if (oerr != null) return "our container: " + oerr;
        var lc = WebPChunks(libwebp, out _);
        string ot = string.Join(",", oc.Select(x => x.Tag)), lt = string.Join(",", lc.Select(x => x.Tag));
        if (ot != lt) return $"chunks {ot} vs libwebp {lt}";
        int frame = 0;
        for (int i = 0; i < oc.Count; i++)
        {
            var (tag, oo, os) = oc[i];
            var (_, lo, ls) = lc[i];
            if (tag == "ALPH") continue;   // a still's ALPH: compared by plane below
            if (tag != "ANMF")
            {
                if (tag == "VP8 " || tag == "VP8X" || tag == "ANIM" || tag == "EXIF" || tag == "XMP " || tag == "ICCP")
                    if (!ours.AsSpan(oo, os).SequenceEqual(libwebp.AsSpan(lo, ls))) return $"{tag} chunk differs";
                continue;
            }
            if (!ours.AsSpan(oo, 16).SequenceEqual(libwebp.AsSpan(lo, 16)))
                return $"frame {frame}: ANMF parameters {Convert.ToHexString(ours, oo, 16)} vs libwebp {Convert.ToHexString(libwebp, lo, 16)}";
            var oi = WebPInnerChunks(ours, oo + 16, oo + os);
            var li = WebPInnerChunks(libwebp, lo + 16, lo + ls);
            string oit = string.Join(",", oi.Select(x => x.Tag)), lit = string.Join(",", li.Select(x => x.Tag));
            if (oit != lit) return $"frame {frame}: chunks {oit} vs libwebp {lit}";
            for (int j = 0; j < oi.Count; j++)
                if (oi[j].Tag == "VP8 " && !ours.AsSpan(oi[j].Offset, oi[j].Size).SequenceEqual(libwebp.AsSpan(li[j].Offset, li[j].Size)))
                    return $"frame {frame}: VP8 chunk ({oi[j].Size} vs libwebp {li[j].Size} bytes)";
            if (oi.Any(x => x.Tag == "ALPH"))
            {
                var pa = await dec.DecodeAlphaAsync(ours, frame);
                var pb = await dec.DecodeAlphaAsync(libwebp, frame);
                if (pa.Width != pb.Width || pa.Height != pb.Height || !pa.Alpha.AsSpan().SequenceEqual(pb.Alpha))
                    return $"frame {frame}: ALPH decodes to another plane";
            }
            frame++;
        }
        if (oc.Any(x => x.Tag == "ALPH"))
        {
            var pa = await dec.DecodeAlphaAsync(ours, 0);
            var pb = await dec.DecodeAlphaAsync(libwebp, 0);
            if (!pa.Alpha.AsSpan().SequenceEqual(pb.Alpha)) return "ALPH decodes to another plane";
        }
        return null;
    }

    private static bool WebPHasAlph(byte[] f)
    {
        for (int i = 0; i + 4 <= f.Length; i++)
            if (f[i] == 'A' && f[i + 1] == 'L' && f[i + 2] == 'P' && f[i + 3] == 'H') return true;
        return false;
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task WebPLossyEncoderGpu_FilterStrength_ByteIdenticalWithLibwebp()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new WebPLossyEncoderGpu(acc);
            var yuvCache = new Dictionary<string, byte[]>();
            var failures = new List<string>();
            int total = 0;
            var strengths = new HashSet<int>();
            foreach (var line in System.Text.Encoding.UTF8.GetString(ReadEmbedded(WebPLossyFilterPrefix + "manifest.tsv")).Split('\n'))
            {
                var row = line.TrimEnd('\r');
                if (row.Length == 0) continue;
                var c = row.Split('\t');
                string name = c[0], src = c[1];
                int w = int.Parse(c[2]), h = int.Parse(c[3]), q = int.Parse(c[4]), m = int.Parse(c[5]), fs = int.Parse(c[6]);
                total++;
                strengths.Add(fs);
                if (!yuvCache.TryGetValue(src, out var yuv))
                    yuvCache[src] = yuv = Gunzip(ReadEmbedded(WebPLossyFilterPrefix + src + ".yuv.gz"));
                var expected = ReadEmbedded(WebPLossyFilterPrefix + name + ".webp");
                var got = await enc.EncodeYuvAsync(yuv, w, h, new WebPLossyEncodeOptions { Quality = q, Method = m, FilterStrength = fs });
                if (got.AsSpan().SequenceEqual(expected)) continue;
                int first = 0, lim = Math.Min(got.Length, expected.Length);
                while (first < lim && got[first] == expected[first]) first++;
                failures.Add($"{name}: {got.Length} bytes vs libwebp {expected.Length}, first difference at byte {first}");
            }
            True(total >= 72 && strengths.SetEquals(new[] { 0, 25, 60, 100 }), $"{total} fixtures, strengths {string.Join(",", strengths)}");
            True(failures.Count == 0, $"{failures.Count} of {total} differ:\n" + string.Join("\n", failures.Take(25)));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task WebPLossyEncoderGpu_Animation_MatchesLibwebpAnimEncoder()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new WebPLossyEncoderGpu(acc);
            using var dec = new WebPDecoderGpu(acc);
            var cache = new Dictionary<string, byte[]>();
            var failures = new List<string>();
            var cases = WebPAnimLossyCases();
            int identical = 0, alphOnly = 0;
            foreach (var c in cases)
            {
                try
                {
                    var frames = WebPAnimFrames(c, cache);
                    byte[]? exif = c.Exif ? ReadEmbedded(WebPAnimLossyPrefix + c.Name + ".exif") : null;
                    var ours = await enc.EncodeAnimationAsync(frames, c.Width, c.Height, c.Durations, c.Options, c.Animation(exif));
                    var libwebp = ReadEmbedded(WebPAnimLossyPrefix + c.Name + ".webp");
                    if (ours.AsSpan().SequenceEqual(libwebp)) { identical++; continue; }
                    int first = 0, lim = Math.Min(ours.Length, libwebp.Length);
                    while (first < lim && ours[first] == libwebp[first]) first++;
                    string what = $"{ours.Length} bytes vs libwebp {libwebp.Length}, first difference at byte {first}";
                    if (!WebPHasAlph(libwebp)) { failures.Add($"{c.Name} (no ALPH: must be byte-identical): {what}"); continue; }
                    var diff = await CompareWebPAnimExceptAlphCoding(dec, ours, libwebp);
                    if (diff != null) failures.Add($"{c.Name}: {diff} ({what})");
                    else alphOnly++;
                }
                catch (Exception ex)
                {
                    failures.Add($"{c.Name}: {ex.GetType().Name}: {ex.Message}");
                }
            }
            Console.WriteLine($"WebP animation: {identical} of {cases.Count} byte-identical with libwebp, {alphOnly} differ only in ALPH coding");
            True(cases.Count >= 18, $"only {cases.Count} fixtures");
            True(failures.Count == 0, $"{identical} byte-identical, {alphOnly} differ only in ALPH coding, {failures.Count} wrong:\n" + string.Join("\n", failures));
            True(identical >= 13, $"only {identical} byte-identical");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task WebPLossyEncoderGpu_Animation_ArrayViewOverload_WritesOnlyTheFile()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new WebPLossyEncoderGpu(acc);
            var cache = new Dictionary<string, byte[]>();
            foreach (var name in new[] { "patch_q80_m4", "still_exif_q80_m4", "comfy_gradient_exif_q80_m4" })
            {
                var c = WebPAnimLossyCases().Single(k => k.Name == name);
                var frames = WebPAnimFrames(c, cache);
                byte[]? exif = c.Exif ? ReadEmbedded(WebPAnimLossyPrefix + c.Name + ".exif") : null;
                var libwebp = ReadEmbedded(WebPAnimLossyPrefix + c.Name + ".webp");
                var devFrames = frames.Select(f => DeviceVerify.Upload(acc, f.ToArray(), slack: 5)).ToList();
                try
                {
                    var views = devFrames.Select(d => d.View.SubView(0, (long)c.Width * c.Height * 4)).ToList();
                    int room = libwebp.Length;
                    using var dOut = DeviceVerify.Poisoned(acc, room + 64);
                    int length = await enc.EncodeAnimationAsync(views, c.Width, c.Height, c.Durations, dOut.View.SubView(0, room), c.Options, c.Animation(exif));
                    Equal(libwebp.Length, length, $"{name}: length");
                    Equal(null, await DeviceVerify.AllEqualAsync(acc, dOut.View.SubView(length, 64), 64, 0xAB), $"{name}: bytes past the end of the file");
                    var ours = await DeviceTransfer.DownloadAsync(dOut.View, length);
                    True(ours.AsSpan().SequenceEqual(libwebp), $"{name}: ArrayView overload file differs from libwebp's");

                    // one byte short: refused, naming the size the file needs
                    using var small = DeviceVerify.Poisoned(acc, room - 1);
                    string? message = null;
                    try { await enc.EncodeAnimationAsync(views, c.Width, c.Height, c.Durations, small.View, c.Options, c.Animation(exif)); }
                    catch (ArgumentException ex) { message = ex.Message; }
                    True(message != null && message.Contains(libwebp.Length.ToString()), $"{name}: too-small output must be refused naming {libwebp.Length}; got {message ?? "no exception"}");
                }
                finally
                {
                    foreach (var d in devFrames) d.Dispose();
                }
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000, Category = "DesktopOnly,NotOnWebGL")]
    public async Task WebPLossyEncoderGpu_Animation_PillowDecodesOurFiles_LikeLibwebpsOwn()
    {
        PillowOracle.RequireAvailable();
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new WebPLossyEncoderGpu(acc);
            var cache = new Dictionary<string, byte[]>();
            var failures = new List<string>();
            var cases = WebPAnimLossyCases();
            foreach (var c in cases)
            {
                var frames = WebPAnimFrames(c, cache);
                byte[]? exif = c.Exif ? ReadEmbedded(WebPAnimLossyPrefix + c.Name + ".exif") : null;
                var ours = await enc.EncodeAnimationAsync(frames, c.Width, c.Height, c.Durations, c.Options, c.Animation(exif));
                var p = PillowOracle.Decode(ours, ".webp");
                var r = PillowOracle.Decode(ReadEmbedded(WebPAnimLossyPrefix + c.Name + ".webp"), ".webp");
                if (p.FramesRgba.Length != r.FramesRgba.Length) { failures.Add($"{c.Name}: Pillow sees {p.FramesRgba.Length} frames, libwebp's file {r.FramesRgba.Length}"); continue; }
                for (int k = 0; k < p.FramesRgba.Length; k++)
                {
                    var diff = ComparePixels(r.FramesRgba[k], p.FramesRgba[k], 4, c.Width);
                    if (diff != null) { failures.Add($"{c.Name} frame {k}: {diff}"); break; }
                }
            }
            True(failures.Count == 0, $"{failures.Count} of {cases.Count}:\n" + string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
