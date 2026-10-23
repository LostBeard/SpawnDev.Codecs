// Tests for WebPLosslessEncoderGpu.EncodeAlphaAsync (the ALPH chunk of a lossy WebP).
//
// The truth is the SOURCE plane (TestData/WebP/alph_*.alpha8.gz, the planes the decoder fixtures were built from). The
// encoder is asked for an ALPH payload for each plane x filter x quality; the payload is put into a VP8X file next to a real
// libwebp-made VP8 chunk of the same size (the colour half; an ALPH chunk does not depend on it) and the file must
//   (a) decode to EXACTLY the plane in libwebp (through Pillow: alpha channel of the RGBA decode),
//   (b) decode to EXACTLY the plane in an independent pure-Python ALPH decoder (scripts/images/webp_ref.py),
//   (c) decode to EXACTLY the plane in WebPDecoderGpu.DecodeAlphaAsync.
// The header byte is checked (requested filter, pre-processing 0, reserved 0, method 0 when Compress = false).
// Part (a)/(b) need Pillow: set CODECS_PILLOW_PYTHON (see CodecsTestBase.WebPLosslessEncoderGpuTests.cs).

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
    private static readonly string[] WebPAlphaPlaneNames =
    {
        "smooth_64x48", "mask_61x45", "noise_33x29", "steps_50x40", "flat255_32x24", "flat0_17x11",
        "tall_1x37", "wide_41x1", "one_1x1", "big_300x200",
    };

    /// <summary>A source plane (width, height, bytes) and a lossy VP8X file of the same size to host an ALPH chunk.</summary>
    private static (int W, int H, byte[] Plane, byte[] HostFile) WebPAlphaSource(string name)
    {
        string stem = "alph_" + name + (name.StartsWith("big") ? "_f1_vp8l" : "_f0_raw");
        var row = WebPAlphaManifest().Single(r => r.Name == stem);
        return (row.Width, row.Height, WebPAlphaTruth(stem), ReadEmbedded(WebPFixturePrefix + stem + ".webp"));
    }

    private static string AlphaFilterName(WebPAlphaFilter f) => f.ToString().ToLowerInvariant();

    [TestMethod]
    public async Task WebPLosslessEncoderGpu_Alpha_EveryPlaneAndFilter_DecodeBitExact_InLibwebpAndBothDecoders()
    {
        string dir = WebPEncoderOutputDir("alpha");
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        var failures = new List<string>();
        var report = new List<string>();
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            using var dec = new WebPDecoderGpu(acc);
            foreach (var name in WebPAlphaPlaneNames)
            {
                var (w, h, plane, host) = WebPAlphaSource(name);
                bool big = name.StartsWith("big");
                var sizes = new Dictionary<(WebPAlphaFilter, int), int>();
                foreach (var filter in new[] { WebPAlphaFilter.None, WebPAlphaFilter.Horizontal, WebPAlphaFilter.Vertical, WebPAlphaFilter.Gradient, WebPAlphaFilter.Auto })
                {
                    foreach (int quality in big ? new[] { 50 } : new[] { 0, 30, 60, 100 })
                    {
                        string tag = $"{name}_{AlphaFilterName(filter)}_q{quality}";
                        try
                        {
                            var payload = await enc.EncodeAlphaAsync(plane, w, h, new WebPAlphaEncodeOptions { Quality = quality, Filter = filter });
                            if (payload.Length > WebPLosslessEncoderGpu.GetMaxAlphaEncodedSize(w, h))
                                failures.Add($"{tag}: {payload.Length} bytes exceeds GetMaxAlphaEncodedSize");
                            int method = payload[0] & 3, chosen = (payload[0] >> 2) & 3, pre = (payload[0] >> 4) & 3, rsv = payload[0] >> 6;
                            if (pre != 0 || rsv != 0 || method > 1) failures.Add($"{tag}: header byte 0x{payload[0]:X2} has pre-processing {pre}, reserved {rsv}, method {method}");
                            if (filter != WebPAlphaFilter.Auto && chosen != (int)filter)
                                failures.Add($"{tag}: asked for filter {(int)filter}, header says {chosen}");
                            sizes[(filter, quality)] = payload.Length;
                            if (filter == WebPAlphaFilter.Auto) report.Add($"{name}\tq{quality}\tauto picked filter {chosen}\tmethod {method}\tbytes {payload.Length}");

                            var file = WebPWithChunk(host, "ALPH", payload);
                            var back = await dec.DecodeAlphaAsync(file);
                            var diff = ComparePixels(plane, back.Alpha, 1, w);
                            if (diff != null) failures.Add($"{tag} [our decoder]: {diff}");

                            string stem = Path.Combine(dir, tag);
                            await File.WriteAllBytesAsync(stem + ".webp", file);
                            await File.WriteAllBytesAsync(stem + ".alpha8", plane);
                            await File.WriteAllTextAsync(stem + ".size", $"{w} {h}");
                        }
                        catch (Exception ex)
                        {
                            failures.Add($"{tag}: {ex.GetType().Name}: {ex.Message}");
                        }
                    }
                }
                // From effort 50 on, Auto encodes with every filter and keeps the smallest: it must equal the best explicit filter
                // EXACTLY (a pick by a rougher estimate, or a wrong tie-break, would show up as a larger file somewhere).
                foreach (int quality in big ? new[] { 50 } : new[] { 60, 100 })
                {
                    int best = new[] { WebPAlphaFilter.None, WebPAlphaFilter.Horizontal, WebPAlphaFilter.Vertical, WebPAlphaFilter.Gradient }
                        .Min(f => sizes[(f, quality)]);
                    int auto = sizes[(WebPAlphaFilter.Auto, quality)];
                    report.Add($"{name}\tq{quality} sizes none/h/v/g/auto = {sizes[(WebPAlphaFilter.None, quality)]}/{sizes[(WebPAlphaFilter.Horizontal, quality)]}/{sizes[(WebPAlphaFilter.Vertical, quality)]}/{sizes[(WebPAlphaFilter.Gradient, quality)]}/{auto}\tbest {best}");
                    if (auto != best) failures.Add($"{name} q{quality}: Auto gave {auto} bytes, the best explicit filter {best}");
                }

                // Compress = false: always method 0, exactly 1 + w*h bytes
                foreach (var filter in new[] { WebPAlphaFilter.None, WebPAlphaFilter.Gradient, WebPAlphaFilter.Auto })
                {
                    string tag = $"{name}_{AlphaFilterName(filter)}_raw";
                    try
                    {
                        var payload = await enc.EncodeAlphaAsync(plane, w, h, new WebPAlphaEncodeOptions { Filter = filter, Compress = false });
                        if ((payload[0] & 3) != 0 || payload.Length != 1 + w * h) failures.Add($"{tag}: Compress = false must give method 0 and {1 + w * h} bytes, got header 0x{payload[0]:X2}, {payload.Length} bytes");
                        var back = await dec.DecodeAlphaAsync(WebPWithChunk(host, "ALPH", payload));
                        var diff = ComparePixels(plane, back.Alpha, 1, w);
                        if (diff != null) failures.Add($"{tag} [our decoder]: {diff}");
                        string stem = Path.Combine(dir, tag);
                        await File.WriteAllBytesAsync(stem + ".webp", WebPWithChunk(host, "ALPH", payload));
                        await File.WriteAllBytesAsync(stem + ".alpha8", plane);
                        await File.WriteAllTextAsync(stem + ".size", $"{w} {h}");
                    }
                    catch (Exception ex)
                    {
                        failures.Add($"{tag}: {ex.GetType().Name}: {ex.Message}");
                    }
                }
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
        await File.WriteAllLinesAsync(Path.Combine(dir, "autopick.txt"), report);
        True(failures.Count == 0, $"{failures.Count} ALPH encoder failures:\n" + string.Join("\n", failures));
        var verify = await RunPillowVerifyAsync(dir);
        True(verify.Contains("SUMMARY"), "verification script did not finish");
    }

    [TestMethod(Timeout = 600000)]
    public async Task WebPLosslessEncoderGpu_Alpha_IncompressiblePlane_FallsBackToRaw_WithinTheBound()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            var rng = new Random(99);
            int w = 301, h = 251;
            byte[] noise = new byte[w * h];
            rng.NextBytes(noise);
            foreach (var filter in new[] { WebPAlphaFilter.None, WebPAlphaFilter.Auto })
            {
                foreach (int q in new[] { 0, 50, 100 })
                {
                    var payload = await enc.EncodeAlphaAsync(noise, w, h, new WebPAlphaEncodeOptions { Quality = q, Filter = filter });
                    string tag = $"noise {filter} q{q}";
                    // random bytes cannot be compressed: the payload must be the raw plane (1 header byte + w*h), never larger
                    Equal(1 + w * h, payload.Length, tag + " length");
                    Equal(0, payload[0] & 3, tag + " method");
                    // stored raw, None and Auto both mean "no filter": the data after the header is the plane itself
                    Equal(0, (payload[0] >> 2) & 3, tag + " filter");
                    EqualBytes(noise, payload.AsSpan(1).ToArray(), tag + " raw body");
                }
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task WebPLosslessEncoderGpu_Alpha_ArrayViewOverload_WritesOnlyThePayload()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            using var dec = new WebPDecoderGpu(acc);
            foreach (var name in new[] { "smooth_64x48", "mask_61x45", "one_1x1" })
            {
                var (w, h, plane, host) = WebPAlphaSource(name);
                using var dIn = acc.Allocate1D<byte>(plane.Length + 5);
                DeviceTransfer.Upload(dIn.View, plane);
                int max = WebPLosslessEncoderGpu.GetMaxAlphaEncodedSize(w, h);
                using var dOut = acc.Allocate1D<byte>(max + 64);
                DeviceTransfer.Upload(dOut.View, Poison(max + 64));
                int length = await enc.EncodeAlphaAsync(dIn.View, w, h, dOut.View);
                True(length >= 2 && length <= max, $"{name}: length {length} within [2, {max}]");
                var written = await DeviceTransfer.DownloadAsync(dOut.View, length);
                var tail = await dOut.View.SubView(length, 64).CopyToHostAsync();
                for (int i = 0; i < tail.Length; i++)
                    if (tail[i] != 0xAB) throw new Exception($"{name}: byte {length + i} past the end of the payload was written");
                var back = await dec.DecodeAlphaAsync(WebPWithChunk(host, "ALPH", written));
                Equal(null, ComparePixels(plane, back.Alpha, 1, w), name);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task WebPLosslessEncoderGpu_Alpha_BadArguments_AreRejected()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            var px = new byte[16 * 16];
            async Task<bool> Throws<T>(Func<Task> action) where T : Exception
            {
                try { await action(); }
                catch (T) { return true; }
                return false;
            }
            True(await Throws<ArgumentOutOfRangeException>(() => enc.EncodeAlphaAsync(px, 0, 16)), "width 0");
            True(await Throws<ArgumentOutOfRangeException>(() => enc.EncodeAlphaAsync(px, 16, 16385)), "height too large");
            True(await Throws<ArgumentException>(() => enc.EncodeAlphaAsync(px, 17, 16)), "buffer shorter than width*height");
            True(await Throws<ArgumentOutOfRangeException>(() => enc.EncodeAlphaAsync(px, 16, 16, new WebPAlphaEncodeOptions { Quality = -1 })), "quality -1");
            True(await Throws<ArgumentOutOfRangeException>(() => enc.EncodeAlphaAsync(px, 16, 16, new WebPAlphaEncodeOptions { Filter = (WebPAlphaFilter)9 })), "filter 9");
            using var dIn = acc.Allocate1D<byte>(px.Length);
            using var dOut = acc.Allocate1D<byte>(100);
            True(await Throws<ArgumentException>(() => enc.EncodeAlphaAsync(dIn.View, 16, 16, dOut.View)), "output smaller than GetMaxAlphaEncodedSize");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
