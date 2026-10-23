// Tests for WebPLosslessEncoderGpu (Images/WebP).
//
// The encoder is asked to compress the SOURCE images of TestData/WebP/sources (flat, 2/4/16/256-colour palettes with
// bundling, photographic gradient + noise, incompressible noise, horizontal gradients, a 14-mode mosaic, alpha with
// arbitrary RGB under alpha 0, colour-cache-heavy images, 1x1 / 1xN / Nx1 / odd sizes, a 320x240 image) at several
// quality settings. Every result must:
//   (a) open in libwebp (through Pillow, scripts/images/verify_webp_files.py) with the right size and decode to
//       EXACTLY the source pixels,
//   (b) decode to the source in our own WebPDecoderGpu,
//   (c) have a sane size next to libwebp's own file for the same image (the ratio is reported, only a catastrophic
//       ratio fails).
// Part (a) needs Pillow: set CODECS_PILLOW_PYTHON to the python.exe of a venv with `pip install pillow numpy`
// (the test is skipped without it and in browser lanes). CODECS_WEBP_ENC_DIR keeps the encoded files and the
// Pillow report (report.txt) for inspection.

using System.Diagnostics;
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
    private static byte[] WebPSource(string name) =>
        Gunzip(ReadEmbedded(WebPFixturePrefix + "sources." + name + ".rgba8.gz"));

    private static List<WebPFixtureRow> WebPSources() => WebPManifest().Where(r => r.Kind == "source").ToList();

    private static string? FindRepoFile(string relative)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            var candidate = Path.Combine(dir.FullName, relative);
            if (File.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }
        return null;
    }

    /// <summary>A fresh, empty directory for one test's encoded files (under CODECS_WEBP_ENC_DIR when set).</summary>
    private static string WebPEncoderOutputDir(string test)
    {
        var root = Environment.GetEnvironmentVariable("CODECS_WEBP_ENC_DIR");
        if (string.IsNullOrEmpty(root)) root = Path.Combine(Path.GetTempPath(), "codecs-webp-enc-" + Guid.NewGuid().ToString("N"));
        var dir = Path.Combine(root, test);
        if (Directory.Exists(dir)) Directory.Delete(dir, true);
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static byte[] ZeroRgbUnderAlpha0(byte[] rgba)
    {
        var copy = (byte[])rgba.Clone();
        for (int i = 0; i < copy.Length; i += 4)
            if (copy[i + 3] == 0) { copy[i] = 0; copy[i + 1] = 0; copy[i + 2] = 0; }
        return copy;
    }

    /// <summary>Device buffers for one lossless round trip: the encoded file and the decoded canvas.</summary>
    private sealed class WebPRoundTripBuffers : IDisposable
    {
        public required MemoryBuffer1D<byte, Stride1D.Dense> File;
        public required MemoryBuffer1D<byte, Stride1D.Dense> Back;
        public void Dispose() { File.Dispose(); Back.Dispose(); }

        public static WebPRoundTripBuffers For(Accelerator acc, int w, int h, WebPEncodeOptions? options) => new()
        {
            File = acc.Allocate1D<byte>(WebPLosslessEncoderGpu.GetMaxEncodedSize(w, h, options)),
            Back = acc.Allocate1D<byte>(4L * w * h),
        };
    }

    /// <summary>
    /// Encode the device picture <paramref name="dSrc"/> into <paramref name="b"/>.File, decode that into <paramref name="b"/>.Back
    /// and compare it on the accelerator with <paramref name="dExpected"/>. Adds a failure line on a mismatch; returns the file
    /// length (the file stays on the device for a caller that keeps it for Pillow).
    /// </summary>
    private static async Task<int> WebPRoundTripOnDeviceAsync(Accelerator acc, WebPLosslessEncoderGpu enc, WebPDecoderGpu dec,
        ArrayView1D<byte, Stride1D.Dense> dSrc, int w, int h, WebPEncodeOptions options, ArrayView1D<byte, Stride1D.Dense> dExpected,
        WebPRoundTripBuffers b, string tag, List<string> failures)
    {
        int length = await enc.EncodeAsync(dSrc, w, h, b.File.View, options);
        if (length > WebPLosslessEncoderGpu.GetMaxEncodedSize(w, h, options)) failures.Add($"{tag}: {length} bytes exceeds GetMaxEncodedSize");
        var info = await dec.DecodeAsync(b.File.View, length, b.Back.View);
        if (info.Width != w || info.Height != h) failures.Add($"{tag}: decoded size {info.Width}x{info.Height}");
        var diff = await DeviceVerify.CompareAsync(acc, b.Back.View, dExpected, 4L * w * h, 4, w);
        if (diff != null) failures.Add($"{tag} [our decoder]: {diff}");
        return length;
    }

    /// <summary>
    /// Encodes every source at the given qualities and checks our own decoder reproduces it, all on the accelerator; keeps the
    /// files in <paramref name="keepDir"/> (downloaded only then) for the Pillow check.
    /// </summary>
    private async Task<List<string>> EncodeAllSourcesAsync(
        int[] qualities, string? keepDir, bool exact, Func<WebPFixtureRow, WebPEncodeOptions, WebPEncodeOptions>? tweak = null)
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        var failures = new List<string>();
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            using var dec = new WebPDecoderGpu(acc);
            foreach (var row in WebPSources())
            {
                byte[] source = WebPSource(row.Name);
                byte[] expected = exact ? source : ZeroRgbUnderAlpha0(source);
                using var dSrc = DeviceVerify.Upload(acc, source);
                using var dExpected = exact ? null : DeviceVerify.Upload(acc, expected);
                foreach (var q in qualities)
                {
                    string tag = $"{row.Name}_q{q}";
                    try
                    {
                        var options = new WebPEncodeOptions { Quality = q, Exact = exact };
                        if (tweak != null) options = tweak(row, options);
                        using var b = WebPRoundTripBuffers.For(acc, row.Width, row.Height, options);
                        int length = await WebPRoundTripOnDeviceAsync(acc, enc, dec, dSrc.View, row.Width, row.Height, options,
                            (dExpected ?? dSrc).View, b, tag, failures);
                        if (keepDir != null)
                        {
                            string stem = Path.Combine(keepDir, tag);
                            await File.WriteAllBytesAsync(stem + ".webp", await DeviceTransfer.DownloadAsync(b.File.View, length));
                            await File.WriteAllBytesAsync(stem + ".rgba8", expected);
                            await File.WriteAllTextAsync(stem + ".size", $"{row.Width} {row.Height}");
                            if (options.IccProfile != null) await File.WriteAllBytesAsync(stem + ".icc", options.IccProfile);
                            if (options.Exif != null) await File.WriteAllBytesAsync(stem + ".exif", options.Exif);
                            if (options.Xmp != null) await File.WriteAllBytesAsync(stem + ".xmp", options.Xmp);
                        }
                    }
                    catch (Exception ex)
                    {
                        failures.Add($"{tag}: {ex.GetType().Name}: {ex.Message}");
                    }
                }
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
        return failures;
    }

    /// <summary>First statement of a test whose verdict is the libwebp (Pillow) check: skip before encoding anything when it cannot run.</summary>
    private static void RequirePillowVerify()
    {
        if (OperatingSystem.IsBrowser())
            throw new UnsupportedTestException("Pillow verification needs a desktop process.");
        if (!PillowAvailable())
            throw new UnsupportedTestException("Set CODECS_PILLOW_PYTHON to the python.exe of a venv with pillow + numpy.");
    }

    private static async Task<string> RunPillowVerifyAsync(string dir)
    {
        RequirePillowVerify();
        var python = Environment.GetEnvironmentVariable("CODECS_PILLOW_PYTHON")!;
        var script = FindRepoFile(Path.Combine("scripts", "images", "verify_webp_files.py"))
                     ?? throw new InvalidOperationException("scripts/images/verify_webp_files.py not found above " + AppContext.BaseDirectory);
        var psi = new ProcessStartInfo(python)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        psi.ArgumentList.Add("-I");
        psi.ArgumentList.Add(script);
        psi.ArgumentList.Add(dir);
        using var p = Process.Start(psi)!;
        var stdout = await p.StandardOutput.ReadToEndAsync();
        var stderr = await p.StandardError.ReadToEndAsync();
        await p.WaitForExitAsync();
        await File.WriteAllTextAsync(Path.Combine(dir, "report.txt"), stdout + stderr);
        if (p.ExitCode != 0)
            throw new Exception("Pillow rejected encoder output:\n" + string.Join("\n", stdout.Split('\n').Where(l => l.StartsWith("FAIL"))) + "\n" + stderr);
        return stdout;
    }

    [TestMethod(Timeout = 600000)]
    public async Task WebPLosslessEncoderGpu_AllSources_DecodeBitExact_InOurDecoder_AtEveryQuality()
    {
        var failures = await EncodeAllSourcesAsync(new[] { 0, 30, 55, 75, 95 }, null, exact: true);
        True(failures.Count == 0, $"{failures.Count} encoder round-trip failures:\n" + string.Join("\n", failures));
    }

    [TestMethod(Timeout = 600000)]
    public async Task WebPLosslessEncoderGpu_AllSources_OpenInLibwebp_AndDecodeBitExact()
    {
        RequirePillowVerify(); // our decoder alone is AllSources_DecodeBitExact_InOurDecoder
        string dir = WebPEncoderOutputDir("pillow");
        var failures = await EncodeAllSourcesAsync(new[] { 0, 40, 75, 100 }, dir, exact: true);
        True(failures.Count == 0, $"{failures.Count} encoder round-trip failures:\n" + string.Join("\n", failures));
        var report = await RunPillowVerifyAsync(dir);
        True(report.Contains("SUMMARY"), "verification script did not finish");
    }

    [TestMethod]
    public async Task WebPLosslessEncoderGpu_ExactFalse_ClearsRgbUnderTransparentAlpha_Only()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            using var dec = new WebPDecoderGpu(acc);
            byte[] source = WebPSource("enc_alpha_64x40");
            int transparent = 0;
            for (int i = 3; i < source.Length; i += 4) if (source[i] == 0 && (source[i - 1] | source[i - 2] | source[i - 3]) != 0) transparent++;
            True(transparent > 20, "the source must have transparent pixels with non-zero RGB, else this test proves nothing");

            var failures = new List<string>();
            using var dSrc = DeviceVerify.Upload(acc, source);
            using var dCleared = DeviceVerify.Upload(acc, ZeroRgbUnderAlpha0(source));
            using var b = WebPRoundTripBuffers.For(acc, 64, 40, null);
            await WebPRoundTripOnDeviceAsync(acc, enc, dec, dSrc.View, 64, 40, new WebPEncodeOptions { Exact = true }, dSrc.View, b,
                "Exact = true keeps every RGB value", failures);
            await WebPRoundTripOnDeviceAsync(acc, enc, dec, dSrc.View, 64, 40, new WebPEncodeOptions { Exact = false }, dCleared.View, b,
                "Exact = false clears RGB under alpha 0 and changes nothing else", failures);
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task WebPLosslessEncoderGpu_Metadata_IsWrittenAndOpensInLibwebp()
    {
        byte[] icc = new byte[301];
        for (int i = 0; i < icc.Length; i++) icc[i] = (byte)(i * 7 + 1);
        byte[] exif = new byte[] { (byte)'I', (byte)'I', 42, 0, 8, 0, 0, 0, 1, 0, 0x12, 0x01, 3, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0 };
        byte[] xmp = System.Text.Encoding.UTF8.GetBytes("<x:xmpmeta xmlns:x='adobe:ns:meta/'><rdf/></x:xmpmeta>");
        string? dir = PillowAvailable() ? WebPEncoderOutputDir("metadata") : null;
        WebPEncodeOptions WithMetadata(WebPFixtureRow row, WebPEncodeOptions o)
        {
            o.IccProfile = icc;
            o.Exif = row.Width % 2 == 0 ? exif : System.Text.Encoding.ASCII.GetBytes("Exif\0\0").Concat(exif).ToArray();
            o.Xmp = xmp;
            return o;
        }
        var failures = await EncodeAllSourcesAsync(new[] { 75 }, dir, exact: true, WithMetadata);
        True(failures.Count == 0, $"{failures.Count} failures:\n" + string.Join("\n", failures));

        // Our own decoder reads them back and reports the orientation (one small file; only the metadata blobs come back).
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            using var dec = new WebPDecoderGpu(acc);
            var row = WebPSources().Single(r => r.Name == "enc_photo_80x60");
            var one = await enc.EncodeAsync(WebPSource(row.Name), row.Width, row.Height,
                WithMetadata(row, new WebPEncodeOptions { Quality = 75, Exact = true }));
            var md = await dec.ReadMetadataAsync(one);
            EqualBytes(icc, md.IccProfile!, "ICC");
            EqualBytes(exif, md.Exif!, "EXIF");
            EqualBytes(xmp, md.Xmp!, "XMP");
            var fi = await dec.ReadFileInfoAsync(one);
            True(fi.IsExtended && fi.HasIccProfile && fi.HasExif && fi.HasXmp && fi.IsLossless, "VP8X facts");
            Equal(6, fi.Info.Orientation, "orientation from the EXIF we wrote");
        }
        finally { ReleaseAccelerator(ctx, acc); }

        if (dir == null) return; // libwebp's check of the files needs a desktop with Pillow
        // The "Exif\0\0"-prefixed variant must have been stored WITHOUT the prefix: the sidecar holds what the caller passed,
        // so normalise it for libwebp's comparison.
        foreach (var f in Directory.GetFiles(dir, "*.exif"))
            await File.WriteAllBytesAsync(f, exif);
        await RunPillowVerifyAsync(dir);
    }

    [TestMethod]
    public async Task WebPLosslessEncoderGpu_ArrayViewOverload_WritesOnlyTheFile()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            using var dec = new WebPDecoderGpu(acc);
            foreach (var name in new[] { "enc_photo_80x60", "enc_pal4_64x40", "enc_one_pixel_1x1" })
            {
                var row = WebPSources().Single(r => r.Name == name);
                byte[] source = WebPSource(name);
                using var dIn = DeviceVerify.Upload(acc, source, slack: 5);
                int max = WebPLosslessEncoderGpu.GetMaxEncodedSize(row.Width, row.Height);
                using var dOut = DeviceVerify.Poisoned(acc, max + 64);
                int length = await enc.EncodeAsync(dIn.View, row.Width, row.Height, dOut.View);
                True(length > 20 && length <= max, $"{name}: length {length} within (20, {max}]");
                Equal(null, await DeviceVerify.AllEqualAsync(acc, dOut.View.SubView(length, 64), 64, 0xAB), $"{name}: bytes past the end of the file");
                using var dBack = acc.Allocate1D<byte>(source.Length);
                await dec.DecodeAsync(dOut.View, length, dBack.View);
                Equal(null, await DeviceVerify.CompareAsync(acc, dBack.View, dIn.View, source.Length, 4, row.Width), name);
                // the managed overload (a few KB here) must write the same file
                var managed = await enc.EncodeAsync(source, row.Width, row.Height);
                Equal(length, managed.Length, $"{name}: managed overload length");
                Equal(null, await DeviceVerify.CompareAsync(acc, dOut.View, managed), $"{name}: managed overload bytes");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000)]
    public async Task WebPLosslessEncoderGpu_IncompressibleNoise_StaysWithinGetMaxEncodedSize()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            using var dec = new WebPDecoderGpu(acc);
            var rng = new Random(1234);
            // big enough that the fixed slack in GetMaxEncodedSize cannot hide a too-small per-pixel bound
            int w = 301, h = 251;
            byte[] noise = new byte[w * h * 4];
            rng.NextBytes(noise);
            using var dNoise = DeviceVerify.Upload(acc, noise);
            using var b = WebPRoundTripBuffers.For(acc, w, h, null);
            var failures = new List<string>();
            foreach (int q in new[] { 0, 50, 100 })
                await WebPRoundTripOnDeviceAsync(acc, enc, dec, dNoise.View, w, h, new WebPEncodeOptions { Quality = q }, dNoise.View, b, $"q{q} noise", failures);
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task WebPLosslessEncoderGpu_FeaturesAreActuallyUsed_AsSeenByAnIndependentBitstreamParser()
    {
        // A degenerate encoder that wrote only literals would pass every round-trip test. scripts/images/webp_ref.py, an
        // independent pure-Python VP8L decoder, reports what each file really contains.
        var cases = new (string Source, int Quality, Func<string, bool> Check, string Expect)[]
        {
            ("enc_two_color_70x45", 75, s => s.Contains("xf[I3]"), "palette with 8 pixels per word"),
            ("enc_pal4_64x40", 75, s => s.Contains("xf[I2]"), "palette with 4 pixels per word"),
            ("enc_pal16_61x33", 75, s => s.Contains("xf[I1]"), "palette with 2 pixels per word"),
            ("enc_pal256_90x70", 75, s => s.Contains("xf[I0]") && s.Contains("pal=256"), "256-colour palette"),
            ("enc_photo_80x60", 75, s => s.Contains("G,P") && s.Contains(",C"), "subtract green + predictor + cross colour"),
            ("enc_mosaic_96x72", 100, s => PredictorModeCount(s) >= 5, "at least five predictor modes"),
            ("enc_cache_heavy_80x60", 100, s => !s.Contains("cache=0 ") && !s.Contains("copies=0 "), "colour cache and LZ77 copies"),
            ("enc_large_320x240", 75, s => CopyCount(s) > 100, "LZ77 copies on a compressible image"),
            // every tile of this image has red and blue following green: all tiles must carry NON-ZERO multipliers, else the
            // sign of the forward transform is never exercised (an all-zero cross transform decodes the same either way)
            ("enc_correlated_72x56", 75, s => s.Contains(",C") && CrossNonZeroTiles(s) == CrossTiles(s) && CrossTiles(s) > 0,
                "cross colour with non-zero multipliers in every tile (red and blue follow green)"),
            ("enc_meta_groups_128x128", 100, s => GroupCount(s) >= 2, "several prefix groups (meta prefix codes)"),
        };
        RequirePillowVerify(); // the verdict comes from the Python bitstream parser
        string dir = WebPEncoderOutputDir("features");
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            foreach (var (source, quality, _, _) in cases)
            {
                var row = WebPSources().Single(r => r.Name == source);
                byte[] px = WebPSource(source);
                var file = await enc.EncodeAsync(px, row.Width, row.Height, new WebPEncodeOptions { Quality = quality });
                string stem = Path.Combine(dir, source);
                await File.WriteAllBytesAsync(stem + ".webp", file);
                await File.WriteAllBytesAsync(stem + ".rgba8", px);
                await File.WriteAllTextAsync(stem + ".size", $"{row.Width} {row.Height}");
                await File.WriteAllTextAsync(stem + ".stats", "");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }

        var report = await RunPillowVerifyAsync(dir);
        var stats = report.Split('\n').Where(l => l.StartsWith("STATS ")).ToDictionary(l => l.Split(' ')[1], l => l);
        var failures = new List<string>();
        foreach (var (source, quality, check, expect) in cases)
        {
            if (!stats.TryGetValue(source, out var line)) { failures.Add($"{source}: no feature report"); continue; }
            if (!check(line)) failures.Add($"{source} (q{quality}) should use {expect}, the parser saw: {line}");
        }
        True(failures.Count == 0, string.Join("\n", failures));
    }

    private static int PredictorModeCount(string stats)
    {
        var m = System.Text.RegularExpressions.Regex.Match(stats, "pred_modes=([0-9a-f]*)");
        return m.Success ? m.Groups[1].Value.Length : 0;
    }

    private static int CrossNonZeroTiles(string stats)
    {
        var m = System.Text.RegularExpressions.Regex.Match(stats, "cc_nz=([0-9]+)/([0-9]+)");
        return m.Success ? int.Parse(m.Groups[1].Value) : -1;
    }

    private static int CrossTiles(string stats)
    {
        var m = System.Text.RegularExpressions.Regex.Match(stats, "cc_nz=([0-9]+)/([0-9]+)");
        return m.Success ? int.Parse(m.Groups[2].Value) : -1;
    }

    private static int CopyCount(string stats)
    {
        var m = System.Text.RegularExpressions.Regex.Match(stats, "copies=([0-9]+)");
        return m.Success ? int.Parse(m.Groups[1].Value) : 0;
    }

    private static int GroupCount(string stats)
    {
        var m = System.Text.RegularExpressions.Regex.Match(stats, "groups=([0-9]+)/");
        return m.Success ? int.Parse(m.Groups[1].Value) : 0;
    }

    [TestMethod]
    public async Task WebPLosslessEncoderGpu_ExtremeShapes_RoundTrip_AndOpenInLibwebp()
    {
        // The VP8L header has 14-bit sizes: 16384 is the largest. Also sizes around the transform block boundaries.
        var shapes = new (int W, int H)[] { (16384, 1), (1, 16384), (16384, 2), (3, 16384), (257, 255), (16, 16), (17, 15), (15, 17), (64, 1), (1, 64) };
        bool pillow = PillowAvailable();
        string? dir = pillow ? WebPEncoderOutputDir("extreme") : null;
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        var failures = new List<string>();
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            using var dec = new WebPDecoderGpu(acc);
            var rng = new Random(77);
            foreach (var (w, h) in shapes)
            {
                byte[] px = new byte[w * h * 4];
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        int o = (y * w + x) * 4;
                        px[o] = (byte)(x * 3 + rng.Next(4));
                        px[o + 1] = (byte)(y * 5 + (x >> 4));
                        px[o + 2] = (byte)((x ^ y) + rng.Next(3));
                        px[o + 3] = 255;
                    }
                using var dPx = DeviceVerify.Upload(acc, px);
                using var b = WebPRoundTripBuffers.For(acc, w, h, null);
                foreach (int q in new[] { 10, 30, 75 })
                {
                    string tag = $"extreme_{w}x{h}_q{q}";
                    try
                    {
                        int length = await WebPRoundTripOnDeviceAsync(acc, enc, dec, dPx.View, w, h, new WebPEncodeOptions { Quality = q }, dPx.View, b, tag, failures);
                        if (dir != null)
                        {
                            await File.WriteAllBytesAsync(Path.Combine(dir, tag + ".webp"), await DeviceTransfer.DownloadAsync(b.File.View, length));
                            await File.WriteAllBytesAsync(Path.Combine(dir, tag + ".rgba8"), px);
                            await File.WriteAllTextAsync(Path.Combine(dir, tag + ".size"), $"{w} {h}");
                        }
                    }
                    catch (Exception ex)
                    {
                        failures.Add($"{tag}: {ex.GetType().Name}: {ex.Message}");
                    }
                }
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
        True(failures.Count == 0, $"{failures.Count} failures:\n" + string.Join("\n", failures));
        if (dir != null) await RunPillowVerifyAsync(dir);
    }

    [TestMethod]
    public async Task WebPLosslessEncoderGpu_BadArguments_AreRejected()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            var px = new byte[16 * 16 * 4];
            async Task<bool> Throws<T>(Func<Task> action) where T : Exception
            {
                try { await action(); }
                catch (T) { return true; }
                return false;
            }
            True(await Throws<ArgumentOutOfRangeException>(() => enc.EncodeAsync(px, 0, 16)), "width 0");
            True(await Throws<ArgumentOutOfRangeException>(() => enc.EncodeAsync(px, 16, 16385)), "height too large");
            True(await Throws<ArgumentException>(() => enc.EncodeAsync(px, 17, 16)), "buffer shorter than width*height*4");
            True(await Throws<ArgumentOutOfRangeException>(() => enc.EncodeAsync(px, 16, 16, new WebPEncodeOptions { Quality = 101 })), "quality 101");

            using var dIn = acc.Allocate1D<byte>(px.Length);
            using var dOut = acc.Allocate1D<byte>(100);
            True(await Throws<ArgumentException>(() => enc.EncodeAsync(dIn.View, 16, 16, dOut.View)), "output smaller than GetMaxEncodedSize");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
