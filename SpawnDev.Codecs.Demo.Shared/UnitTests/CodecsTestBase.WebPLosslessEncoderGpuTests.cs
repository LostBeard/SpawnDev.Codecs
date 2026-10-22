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

    /// <summary>Encodes every source at the given qualities, checks our own decoder reproduces it, and optionally keeps the files.</summary>
    private async Task<List<string>> EncodeAllSourcesAsync(
        int[] qualities, string? keepDir, bool exact, Func<WebPFixtureRow, WebPEncodeOptions, WebPEncodeOptions>? tweak = null)
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        var failures = new List<string>();
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            using var dec = new WebPDecoderGpu(acc);
            foreach (var q in qualities)
            {
                foreach (var row in WebPSources())
                {
                    string tag = $"{row.Name}_q{q}";
                    try
                    {
                        byte[] source = WebPSource(row.Name);
                        var options = new WebPEncodeOptions { Quality = q, Exact = exact };
                        if (tweak != null) options = tweak(row, options);
                        byte[] file = await enc.EncodeAsync(source, row.Width, row.Height, options);
                        if (file.Length > WebPLosslessEncoderGpu.GetMaxEncodedSize(row.Width, row.Height, options))
                            failures.Add($"{tag}: {file.Length} bytes exceeds GetMaxEncodedSize");

                        byte[] expected = exact ? source : ZeroRgbUnderAlpha0(source);
                        var decoded = await dec.DecodeAsync(file);
                        var diff = ComparePixels(expected, decoded.Pixels, 4, row.Width);
                        if (diff != null) failures.Add($"{tag} [our decoder]: {diff}");
                        if (decoded.Info.Width != row.Width || decoded.Info.Height != row.Height)
                            failures.Add($"{tag}: decoded size {decoded.Info.Width}x{decoded.Info.Height}");

                        if (keepDir != null)
                        {
                            string stem = Path.Combine(keepDir, tag);
                            await File.WriteAllBytesAsync(stem + ".webp", file);
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
        finally { acc.Dispose(); ctx.Dispose(); }
        return failures;
    }

    private static async Task<string> RunPillowVerifyAsync(string dir)
    {
        if (OperatingSystem.IsBrowser())
            throw new UnsupportedTestException("Pillow verification needs a desktop process.");
        var python = Environment.GetEnvironmentVariable("CODECS_PILLOW_PYTHON");
        if (string.IsNullOrEmpty(python) || !File.Exists(python))
            throw new UnsupportedTestException("Set CODECS_PILLOW_PYTHON to the python.exe of a venv with pillow + numpy.");
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

    [TestMethod]
    public async Task WebPLosslessEncoderGpu_AllSources_DecodeBitExact_InOurDecoder_AtEveryQuality()
    {
        var failures = await EncodeAllSourcesAsync(new[] { 0, 30, 55, 75, 95 }, null, exact: true);
        True(failures.Count == 0, $"{failures.Count} encoder round-trip failures:\n" + string.Join("\n", failures));
    }

    [TestMethod]
    public async Task WebPLosslessEncoderGpu_AllSources_OpenInLibwebp_AndDecodeBitExact()
    {
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

            var exactFile = await enc.EncodeAsync(source, 64, 40, new WebPEncodeOptions { Exact = true });
            var exactBack = await dec.DecodeAsync(exactFile);
            Equal(null, ComparePixels(source, exactBack.Pixels, 4, 64), "Exact = true keeps every RGB value");

            var looseFile = await enc.EncodeAsync(source, 64, 40, new WebPEncodeOptions { Exact = false });
            var looseBack = await dec.DecodeAsync(looseFile);
            Equal(null, ComparePixels(ZeroRgbUnderAlpha0(source), looseBack.Pixels, 4, 64), "Exact = false clears RGB under alpha 0 and changes nothing else");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task WebPLosslessEncoderGpu_Metadata_IsWrittenAndOpensInLibwebp()
    {
        byte[] icc = new byte[301];
        for (int i = 0; i < icc.Length; i++) icc[i] = (byte)(i * 7 + 1);
        byte[] exif = new byte[] { (byte)'I', (byte)'I', 42, 0, 8, 0, 0, 0, 1, 0, 0x12, 0x01, 3, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0 };
        byte[] xmp = System.Text.Encoding.UTF8.GetBytes("<x:xmpmeta xmlns:x='adobe:ns:meta/'><rdf/></x:xmpmeta>");
        string dir = WebPEncoderOutputDir("metadata");
        var failures = await EncodeAllSourcesAsync(new[] { 75 }, dir, exact: true, (row, o) =>
        {
            o.IccProfile = icc;
            o.Exif = row.Width % 2 == 0 ? exif : System.Text.Encoding.ASCII.GetBytes("Exif\0\0").Concat(exif).ToArray();
            o.Xmp = xmp;
            return o;
        });
        True(failures.Count == 0, $"{failures.Count} failures:\n" + string.Join("\n", failures));

        // The "Exif\0\0"-prefixed variant must have been stored WITHOUT the prefix: the sidecar holds what the caller passed,
        // so normalise it for libwebp's comparison.
        foreach (var f in Directory.GetFiles(dir, "*.exif"))
            await File.WriteAllBytesAsync(f, exif);

        // Our own decoder reads them back and reports the orientation.
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var dec = new WebPDecoderGpu(acc);
            var one = await File.ReadAllBytesAsync(Directory.GetFiles(dir, "enc_photo_80x60_q75.webp").Single());
            var md = await dec.ReadMetadataAsync(one);
            EqualBytes(icc, md.IccProfile!, "ICC");
            EqualBytes(exif, md.Exif!, "EXIF");
            EqualBytes(xmp, md.Xmp!, "XMP");
            var fi = await dec.ReadFileInfoAsync(one);
            True(fi.IsExtended && fi.HasIccProfile && fi.HasExif && fi.HasXmp && fi.IsLossless, "VP8X facts");
            Equal(6, fi.Info.Orientation, "orientation from the EXIF we wrote");
        }
        finally { acc.Dispose(); ctx.Dispose(); }

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
                using var dIn = acc.Allocate1D<byte>(source.Length + 5);
                DeviceTransfer.Upload(dIn.View, source);
                int max = WebPLosslessEncoderGpu.GetMaxEncodedSize(row.Width, row.Height);
                using var dOut = acc.Allocate1D<byte>(max + 64);
                DeviceTransfer.Upload(dOut.View, Poison(max + 64));
                int length = await enc.EncodeAsync(dIn.View, row.Width, row.Height, dOut.View);
                True(length > 20 && length <= max, $"{name}: length {length} within (20, {max}]");

                var written = await DeviceTransfer.DownloadAsync(dOut.View, length);
                var tail = await dOut.View.SubView(length, 64).CopyToHostAsync();
                for (int i = 0; i < tail.Length; i++)
                    if (tail[i] != 0xAB) throw new Exception($"{name}: byte {length + i} past the end of the file was written");
                var decoded = await dec.DecodeAsync(written);
                Equal(null, ComparePixels(source, decoded.Pixels, 4, row.Width), name);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task WebPLosslessEncoderGpu_IncompressibleNoise_StaysWithinGetMaxEncodedSize()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            using var dec = new WebPDecoderGpu(acc);
            var rng = new Random(1234);
            int w = 61, h = 47;
            byte[] noise = new byte[w * h * 4];
            rng.NextBytes(noise);
            foreach (int q in new[] { 0, 50, 100 })
            {
                var file = await enc.EncodeAsync(noise, w, h, new WebPEncodeOptions { Quality = q });
                True(file.Length <= WebPLosslessEncoderGpu.GetMaxEncodedSize(w, h), $"q{q}: {file.Length} bytes");
                var back = await dec.DecodeAsync(file);
                Equal(null, ComparePixels(noise, back.Pixels, 4, w), $"q{q} noise");
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
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
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
