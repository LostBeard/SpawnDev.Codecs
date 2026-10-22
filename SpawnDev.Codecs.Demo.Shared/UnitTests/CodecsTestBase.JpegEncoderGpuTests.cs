// Tests for JpegEncoderGpu (Images/Jpeg).
//
// Truth: files written by Pillow (libjpeg-turbo) from the SAME RGBA pixels (TestData/Jpeg/encsrc_*.rgba8, noisy alpha
// to prove alpha is ignored), same quality / subsampling / restart interval / EXIF. The GPU encoder is built to
// match libjpeg-turbo's arithmetic exactly (jccolor.c, jcsample.c, jfdctint.c, jcdctmgr.c quantization, jchuff.c with
// the Annex K tables), so the ENTIRE FILE must be byte-identical. Further checks: our own decoder round trip (PSNR
// against the source per quality), the ArrayView1D overload into a poisoned caller buffer, EXIF orientation read back,
// and option validation.

using System.Text.RegularExpressions;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Images;
using SpawnDev.Codecs.Images.Jpeg;
using SpawnDev.ILGPU;
using SpawnDev.SpawnJS.JSObjects;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static string[] JpegEncodeFixtureNames()
    {
        var names = typeof(CodecsTestBase).Assembly.GetManifestResourceNames()
            .Where(n => n.StartsWith(JpegFixturePrefix + "enc_", StringComparison.Ordinal) && n.EndsWith(".jpg", StringComparison.Ordinal))
            .Select(n => n.Substring(JpegFixturePrefix.Length, n.Length - JpegFixturePrefix.Length - 4))
            .OrderBy(n => n, StringComparer.Ordinal)
            .ToArray();
        if (names.Length == 0) throw new InvalidOperationException("No JPEG encoder fixtures are embedded.");
        return names;
    }

    /// <summary>Decode a fixture name written by make_jpeg_fixtures.py into (source name, width, height, options).</summary>
    private static (string Source, int W, int H, JpegEncodeOptions Options) ParseJpegEncodeFixture(string name)
    {
        var m = Regex.Match(name, @"^enc_c(?<sub>444|422|420)_q(?<q>\d+)_(?<w>\d+)x(?<h>\d+)$");
        if (m.Success)
            return ($"{m.Groups["w"].Value}x{m.Groups["h"].Value}", int.Parse(m.Groups["w"].Value), int.Parse(m.Groups["h"].Value),
                new JpegEncodeOptions { Quality = int.Parse(m.Groups["q"].Value), Subsampling = SubsamplingOf(m.Groups["sub"].Value) });
        m = Regex.Match(name, @"^enc_gray_q(?<q>\d+)_(?<w>\d+)x(?<h>\d+)$");
        if (m.Success)
            return ($"gray_{m.Groups["w"].Value}x{m.Groups["h"].Value}", int.Parse(m.Groups["w"].Value), int.Parse(m.Groups["h"].Value),
                new JpegEncodeOptions { Quality = int.Parse(m.Groups["q"].Value), Subsampling = JpegSubsampling.Gray });
        m = Regex.Match(name, @"^enc_rst_n(?<n>\d+)_c(?<sub>444|422|420)_100x75$");
        if (m.Success)
            return ("rst100x75", 100, 75, new JpegEncodeOptions
            {
                Quality = 85, Subsampling = SubsamplingOf(m.Groups["sub"].Value), RestartInterval = int.Parse(m.Groups["n"].Value),
            });
        m = Regex.Match(name, @"^enc_rst_n(?<n>\d+)_gray_100x75$");
        if (m.Success)
            return ("rstgray100x75", 100, 75, new JpegEncodeOptions
            {
                Quality = 85, Subsampling = JpegSubsampling.Gray, RestartInterval = int.Parse(m.Groups["n"].Value),
            });
        m = Regex.Match(name, @"^enc_orient(?<o>\d)_c420_24x16$");
        if (m.Success)
            return ("orient24x16", 24, 16, new JpegEncodeOptions
            {
                Quality = 90, Subsampling = JpegSubsampling.Yuv420, ExifOrientation = int.Parse(m.Groups["o"].Value),
            });
        throw new InvalidOperationException($"Unrecognised encoder fixture name {name}");
    }

    private static JpegSubsampling SubsamplingOf(string s) => s switch
    {
        "444" => JpegSubsampling.Yuv444,
        "422" => JpegSubsampling.Yuv422,
        _ => JpegSubsampling.Yuv420,
    };

    private static string? CompareBytes(byte[] expected, byte[] actual)
    {
        if (expected.Length != actual.Length)
        {
            int n = Math.Min(expected.Length, actual.Length);
            for (int i = 0; i < n; i++)
                if (expected[i] != actual[i]) return $"length {actual.Length} != expected {expected.Length}; first differing byte {i} (got 0x{actual[i]:X2}, expected 0x{expected[i]:X2})";
            return $"length {actual.Length} != expected {expected.Length} (common prefix identical)";
        }
        for (int i = 0; i < expected.Length; i++)
            if (expected[i] != actual[i]) return $"first differing byte {i} (got 0x{actual[i]:X2}, expected 0x{expected[i]:X2}) of {expected.Length}";
        return null;
    }

    [TestMethod]
    public async Task JpegEncoderGpu_AllFixtures_ByteIdenticalToPillow_ManagedAndArrayView()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new JpegEncoderGpu(acc);
            var failures = new List<string>();
            int checkedCount = 0;
            foreach (var name in JpegEncodeFixtureNames())
            {
                var (source, w, h, options) = ParseJpegEncodeFixture(name);
                byte[] rgba = ReadEmbedded(JpegFixturePrefix + "encsrc_" + source + ".rgba8");
                byte[] expected = ReadEmbedded(JpegFixturePrefix + name + ".jpg");
                try
                {
                    // Managed overload.
                    var managed = await enc.EncodeAsync(rgba, w, h, options);
                    var diff = CompareBytes(expected, managed);
                    if (diff != null) failures.Add($"{name} [managed]: {diff}");

                    // ArrayView1D overload into a poisoned caller buffer of exactly GetMaxEncodedSize bytes.
                    int max = JpegEncoderGpu.GetMaxEncodedSize(w, h, options);
                    using var dIn = acc.Allocate1D<byte>(rgba.Length + 13);
                    DeviceTransfer.Upload(dIn.View, rgba);
                    using var dOut = acc.Allocate1D<byte>(max);
                    var fill = new byte[max];
                    System.Array.Fill(fill, (byte)0xAB);
                    DeviceTransfer.Upload(dOut.View, fill);
                    int len = await enc.EncodeAsync(dIn.View, w, h, dOut.View, options);
                    if (len > max) failures.Add($"{name} [ArrayView1D]: length {len} exceeds GetMaxEncodedSize {max}");
                    else
                    {
                        var viewBytes = await DeviceTransfer.DownloadAsync(dOut.View, len);
                        diff = CompareBytes(expected, viewBytes);
                        if (diff != null) failures.Add($"{name} [ArrayView1D]: {diff}");
                    }
                    checkedCount++;
                }
                catch (Exception ex)
                {
                    failures.Add($"{name}: {ex.GetType().Name}: {ex.Message}");
                }
            }
            True(checkedCount > 40, $"only {checkedCount} fixtures were exercised");
            True(failures.Count == 0,
                $"{failures.Count} of {checkedCount + failures.Count} encoder fixtures differ from Pillow:\n" + string.Join("\n", failures));
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task JpegEncoderGpu_ProductionScale_1024x768_ByteIdenticalToPillowSha256()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new JpegEncoderGpu(acc);
            using var dec = new JpegDecoderGpu(acc);
            // Source pixels: Pillow's decode of big_c420, which our decoder reproduces bit-exact (tested above).
            var src = await dec.DecodeAsync(ReadEmbedded(JpegFixturePrefix + "big_c420_1024x768.jpg"));
            var failures = new List<string>();
            var cases = new (string Name, JpegEncodeOptions Options)[]
            {
                ("bigenc_c420_q85", new JpegEncodeOptions { Quality = 85, Subsampling = JpegSubsampling.Yuv420 }),
                ("bigenc_c444_q95", new JpegEncodeOptions { Quality = 95, Subsampling = JpegSubsampling.Yuv444 }),
                ("bigenc_rst64_c420_q85", new JpegEncodeOptions { Quality = 85, Subsampling = JpegSubsampling.Yuv420, RestartInterval = 64 }),
            };
            foreach (var (name, options) in cases)
            {
                var file = await enc.EncodeAsync(src.Pixels, 1024, 768, options);
                string got = Sha256Hex(file);
                string want = ReadEmbeddedText(JpegFixturePrefix + name + ".sha256");
                if (got != want) failures.Add($"{name}: {file.Length} bytes, sha256 {got} != Pillow's {want}");
            }
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task JpegEncoderGpu_TypedArrayOverload_ByteIdenticalToPillow()
    {
        if (!OperatingSystem.IsBrowser())
            throw new UnsupportedTestException("TypedArray overloads need a browser accelerator.");
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new JpegEncoderGpu(acc);
            await RunJpegTypedArrayEncode(enc);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [System.Runtime.Versioning.SupportedOSPlatform("browser")]
    private static async Task RunJpegTypedArrayEncode(JpegEncoderGpu enc)
    {
        foreach (var name in new[] { "enc_c420_q090_100x75", "enc_rst_n3_c420_100x75", "enc_gray_q075_17x33" })
        {
            var (source, w, h, options) = ParseJpegEncodeFixture(name);
            byte[] rgba = ReadEmbedded(JpegFixturePrefix + "encsrc_" + source + ".rgba8");
            byte[] expected = ReadEmbedded(JpegFixturePrefix + name + ".jpg");
            using var jsRgba = new Uint8Array(rgba.Length); jsRgba.Set(rgba);
            using var result = await enc.EncodeAsync(jsRgba, w, h, options);
            // Reading back to .NET is the TEST's terminal compare, not the product path.
            var diff = CompareBytes(expected, result.ReadBytes());
            True(diff == null, $"{name} [TypedArray]: {diff}");
        }
    }

    private static double Psnr(byte[] rgbaA, byte[] rgbaB)
    {
        double se = 0;
        long n = 0;
        for (int i = 0; i < rgbaA.Length; i += 4)
            for (int c = 0; c < 3; c++)
            {
                double d = rgbaA[i + c] - rgbaB[i + c];
                se += d * d;
                n++;
            }
        if (se == 0) return 99;
        return 10 * Math.Log10(255.0 * 255.0 * n / se);
    }

    [TestMethod]
    public async Task JpegEncoderGpu_RoundTrip_PsnrPerQuality_AndDecoderMatchesPillowDecodeOfSameBytes()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new JpegEncoderGpu(acc);
            using var dec = new JpegDecoderGpu(acc);
            var failures = new List<string>();
            byte[] rgba = ReadEmbedded(JpegFixturePrefix + "encsrc_64x48.rgba8");
            // Lower bounds on PSNR (dB) of decode(encode(photo)) vs the source, per quality, a little below the
            // measured values (4:2:0: 21.3 / 28.3 / 31.1 / 33.4 dB; 4:4:4: 22.1 / 30.0 / 32.1 / 50.4 dB at q 1 / 25 / 75 / 100).
            var minPsnr = new Dictionary<int, (double Sub420, double Sub444)>
            {
                [1] = (20, 21), [25] = (27, 29), [75] = (30, 31), [100] = (32, 48),
            };
            foreach (var (q, floors) in minPsnr)
            {
                foreach (var sub in new[] { JpegSubsampling.Yuv420, JpegSubsampling.Yuv444 })
                {
                    double floor = sub == JpegSubsampling.Yuv420 ? floors.Sub420 : floors.Sub444;
                    var options = new JpegEncodeOptions { Quality = q, Subsampling = sub };
                    var file = await enc.EncodeAsync(rgba, 64, 48, options);
                    var img = await dec.DecodeAsync(file);
                    if (img.Info.Width != 64 || img.Info.Height != 48) { failures.Add($"q{q} {sub}: size {img.Info.Width}x{img.Info.Height}"); continue; }
                    double psnr = Psnr(rgba, img.Pixels);
                    if (psnr < floor) failures.Add($"q{q} {sub}: PSNR {psnr:F2} dB < {floor} dB");
                }
            }
            // Pillow's decode of Pillow's file is the truth; our encoder reproduces that file, so our decoder must
            // give exactly Pillow's pixels for it. Check on a fixture that has both (the quality-75 gray 100x75 file
            // is a Pillow encode; decode it and compare with a re-encode decode).
            var pillowFile = ReadEmbedded(JpegFixturePrefix + "enc_c420_q090_100x75.jpg");
            var src = ReadEmbedded(JpegFixturePrefix + "encsrc_100x75.rgba8");
            var ours = await enc.EncodeAsync(src, 100, 75, new JpegEncodeOptions { Quality = 90, Subsampling = JpegSubsampling.Yuv420 });
            var d1 = await dec.DecodeAsync(pillowFile);
            var d2 = await dec.DecodeAsync(ours);
            var diff = ComparePixels(d1.Pixels, d2.Pixels, 4, 100);
            if (diff != null) failures.Add($"decode(ours) != decode(Pillow file): {diff}");
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task JpegEncoderGpu_ExifOrientation_IsReadBackByTheDecoder()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new JpegEncoderGpu(acc);
            using var dec = new JpegDecoderGpu(acc);
            byte[] rgba = ReadEmbedded(JpegFixturePrefix + "encsrc_orient24x16.rgba8");
            var failures = new List<string>();
            for (int o = 1; o <= 8; o++)
            {
                var file = await enc.EncodeAsync(rgba, 24, 16, new JpegEncodeOptions { ExifOrientation = o });
                var info = await dec.ReadInfoAsync(file);
                if (info.Orientation != o) failures.Add($"orientation {o}: decoder reports {info.Orientation}");
                if (info.Width != 24 || info.Height != 16) failures.Add($"orientation {o}: size {info.Width}x{info.Height}");
                bool hasExif = file.AsSpan().IndexOf(new byte[] { (byte)'E', (byte)'x', (byte)'i', (byte)'f' }) >= 0;
                if (hasExif != (o != 1)) failures.Add($"orientation {o}: EXIF segment present = {hasExif}");
            }
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task JpegEncoderGpu_InvalidArguments_AreRejected()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new JpegEncoderGpu(acc);
            var rgba = new byte[16 * 16 * 4];
            var failures = new List<string>();
            async Task Expect<T>(string what, Func<Task> act) where T : Exception
            {
                try { await act(); failures.Add($"{what}: no exception"); }
                catch (T) { }
                catch (Exception ex) { failures.Add($"{what}: {ex.GetType().Name} instead of {typeof(T).Name}"); }
            }
            await Expect<ArgumentOutOfRangeException>("quality 0", () => enc.EncodeAsync(rgba, 16, 16, new JpegEncodeOptions { Quality = 0 }));
            await Expect<ArgumentOutOfRangeException>("quality 101", () => enc.EncodeAsync(rgba, 16, 16, new JpegEncodeOptions { Quality = 101 }));
            await Expect<ArgumentOutOfRangeException>("width 0", () => enc.EncodeAsync(rgba, 0, 16));
            await Expect<ArgumentOutOfRangeException>("exif 9", () => enc.EncodeAsync(rgba, 16, 16, new JpegEncodeOptions { ExifOrientation = 9 }));
            await Expect<ArgumentException>("short input", () => enc.EncodeAsync(new byte[10], 16, 16));
            await Expect<NotSupportedException>("progressive", () => enc.EncodeAsync(rgba, 16, 16, new JpegEncodeOptions { Progressive = true }));
            using var tiny = acc.Allocate1D<byte>(10);
            using var dIn = acc.Allocate1D<byte>(rgba.Length);
            await Expect<ArgumentException>("output too small", () => enc.EncodeAsync(dIn.View, 16, 16, tiny.View));
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
