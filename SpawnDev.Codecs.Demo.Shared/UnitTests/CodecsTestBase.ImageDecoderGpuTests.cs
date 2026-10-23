// Tests for the ImageDecoderGpu facade (Images/ImageDecoderGpu.cs): it must sniff the format and forward to the
// right decoder, so the oracle is the SAME independent truth the per-format decoder tests use (Pillow's decode
// stored beside each fixture as .rgba8), reached through the facade only.
//
// What can fail: wrong dispatch (a PNG sent to the JPEG decoder or the reverse reports another ImageInfo.Format
// or throws), a facade overload that forgets to forward (managed vs ArrayView1D vs ReadInfo), a wrong pixel
// result, an unknown file that is not rejected with ImageDecodeException(Unknown).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Images;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Timeout = 600000)]
    public async Task ImageDecoderGpu_PngAndJpegFixtures_DispatchAndMatchPillow()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var dec = new ImageDecoderGpu(acc);
            var failures = new List<string>();
            int checkedPng = 0, checkedJpeg = 0;

            var files = new List<(string path, string truth, ImageFormat format)>();
            foreach (var n in PngFixtureNames())
                if (HasEmbedded(PngFixturePrefix + n + ".rgba8"))
                    files.Add((PngFixturePrefix + n + ".png", PngFixturePrefix + n + ".rgba8", ImageFormat.Png));
            foreach (var n in JpegTruthFixtureNames())
                files.Add((JpegFixturePrefix + n + ".jpg", JpegFixturePrefix + n + ".rgba8", ImageFormat.Jpeg));

            foreach (var (path, truth, format) in files)
            {
                byte[] file = ReadEmbedded(path);
                byte[] expected = ReadEmbedded(truth);
                try
                {
                    var sniffed = ImageDecoderGpu.Detect(file.AsSpan(0, Math.Min(file.Length, ImageFormatDetector.SniffLength)));
                    if (sniffed != format) failures.Add($"{path}: Detect says {sniffed}, expected {format}");

                    var infoOnly = await dec.ReadInfoAsync(file);
                    if (infoOnly.Format != format) failures.Add($"{path}: ReadInfoAsync format {infoOnly.Format}, expected {format}");

                    var managed = await dec.DecodeAsync(file);
                    if (managed.Info.Format != format) failures.Add($"{path}: DecodeAsync format {managed.Info.Format}, expected {format}");
                    var diff = ComparePixels(expected, managed.Pixels, 4, managed.Info.Width);
                    if (diff != null) failures.Add($"{path} [managed]: {diff}");

                    using var dFile = acc.Allocate1D<byte>(file.Length + 5);
                    DeviceTransfer.Upload(dFile.View, file);
                    using var dOut = acc.Allocate1D<byte>(expected.Length + 3);
                    var fill = new byte[dOut.Length];
                    System.Array.Fill(fill, (byte)0xAB);
                    DeviceTransfer.Upload(dOut.View, fill);
                    var viewInfo = await dec.DecodeAsync(dFile.View, file.Length, dOut.View, PixelFormat.Rgba8);
                    var viewPixels = await DeviceTransfer.DownloadAsync(dOut.View, expected.Length);
                    diff = ComparePixels(expected, viewPixels, 4, viewInfo.Width);
                    if (diff != null) failures.Add($"{path} [ArrayView1D]: {diff}");

                    if (format == ImageFormat.Png) checkedPng++; else checkedJpeg++;
                }
                catch (Exception ex)
                {
                    failures.Add($"{path}: threw {ex.GetType().Name}: {ex.Message}");
                }
            }

            True(checkedPng >= 10, $"too few PNG fixtures went through the facade ({checkedPng})");
            True(checkedJpeg >= 10, $"too few JPEG fixtures went through the facade ({checkedJpeg})");
            True(failures.Count == 0, $"{failures.Count} facade failure(s):\n" + string.Join("\n", failures.Take(15)));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000)]
    public async Task ImageDecoderGpu_GifBmpWebPFixtures_DispatchAndMatchTruth()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var dec = new ImageDecoderGpu(acc);   // one facade instance: all six decoders live side by side
            var failures = new List<string>();
            var counts = new Dictionary<ImageFormat, int>();

            // (label, file, expected RGBA8 of the first frame, format)
            var cases = new List<(string label, byte[] file, byte[] expected, ImageFormat format)>();
            foreach (var n in GifBmpFixtureNames(BmpFixturePrefix, ".bmp"))
                cases.Add(("bmp/" + n, ReadEmbedded(BmpFixturePrefix + n + ".bmp"), ReadEmbedded(BmpFixturePrefix + n + ".rgba8"), ImageFormat.Bmp));
            foreach (var n in GifValidFixtureNames())
            {
                var truth = GifTruth(n, "pil", GifMetaJson(n));
                if (truth != null) cases.Add(("gif/" + n, GifFixture(n), truth[0], ImageFormat.Gif));
            }
            foreach (var row in WebPManifest().Where(r => r.Kind == "still"))
                cases.Add(("webp-lossless/" + row.Name, WebPFile(row.Name), WebPTruth(row.Truth), ImageFormat.WebP));
            foreach (var row in WebPLossyManifest().Where(r => r.HasTruth && r.Frames == 1 && r.Folder == "WebPLossy"))
                cases.Add(("webp-lossy/" + row.Name, WebPLossyFile(row), WebPLossyTruth(row, 0), ImageFormat.WebP));

            foreach (var (label, file, expected, format) in cases)
            {
                try
                {
                    var sniffed = ImageDecoderGpu.Detect(file.AsSpan(0, Math.Min(file.Length, ImageFormatDetector.SniffLength)));
                    if (sniffed != format) failures.Add($"{label}: Detect says {sniffed}, expected {format}");
                    var info = await dec.ReadInfoAsync(file);
                    if (info.Format != format) failures.Add($"{label}: ReadInfoAsync format {info.Format}, expected {format}");

                    var managed = await dec.DecodeAsync(file);
                    if (managed.Info.Format != format) failures.Add($"{label}: DecodeAsync format {managed.Info.Format}, expected {format}");
                    var diff = ComparePixels(expected, managed.Pixels, 4, managed.Info.Width);
                    if (diff != null) failures.Add($"{label} [managed]: {diff}");

                    using var dFile = acc.Allocate1D<byte>(file.Length + 5);
                    DeviceTransfer.Upload(dFile.View, file);
                    using var dOut = acc.Allocate1D<byte>(expected.Length + 3);
                    var fill = new byte[dOut.Length];
                    System.Array.Fill(fill, (byte)0xAB);
                    DeviceTransfer.Upload(dOut.View, fill);
                    var viewInfo = await dec.DecodeAsync(dFile.View, file.Length, dOut.View, PixelFormat.Rgba8);
                    var viewPixels = await DeviceTransfer.DownloadAsync(dOut.View, expected.Length);
                    diff = ComparePixels(expected, viewPixels, 4, viewInfo.Width);
                    if (diff != null) failures.Add($"{label} [ArrayView1D]: {diff}");

                    counts[format] = counts.GetValueOrDefault(format) + 1;
                }
                catch (Exception ex)
                {
                    failures.Add($"{label}: threw {ex.GetType().Name}: {ex.Message}");
                }
            }

            True(failures.Count == 0, $"{failures.Count} facade failure(s):\n" + string.Join("\n", failures.Take(15)));
            True(counts.GetValueOrDefault(ImageFormat.Bmp) >= 5, $"too few BMP fixtures ({counts.GetValueOrDefault(ImageFormat.Bmp)})");
            True(counts.GetValueOrDefault(ImageFormat.Gif) >= 5, $"too few GIF fixtures ({counts.GetValueOrDefault(ImageFormat.Gif)})");
            True(counts.GetValueOrDefault(ImageFormat.WebP) >= 60, $"too few WebP fixtures ({counts.GetValueOrDefault(ImageFormat.WebP)})");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task ImageDecoderGpu_UnrecognisedBytes_AreRejectedAsUnknown()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var dec = new ImageDecoderGpu(acc);
            var junk = new byte[64];
            new Random(0x1AB6).NextBytes(junk);
            junk[0] = 0x00; junk[1] = 0x01; // neither a PNG, JPEG, GIF, BMP, WebP signature
            ImageDecodeException? thrown = null;
            try { await dec.DecodeAsync(junk); }
            catch (ImageDecodeException ex) { thrown = ex; }
            True(thrown != null, "random bytes must throw ImageDecodeException");
            Equal(ImageFormat.Unknown, thrown!.Format, "exception format");

            ImageDecodeException? infoThrown = null;
            try { await dec.ReadInfoAsync(junk); }
            catch (ImageDecodeException ex) { infoThrown = ex; }
            True(infoThrown != null, "ReadInfoAsync on random bytes must throw ImageDecodeException");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
