// PNG encoding of real photographs. The synthetic pictures of the other PNG encoder tests never reached the case SpawnGen's
// live site hit on Chrome WebGPU (RX 590, 2026-11-20, Lore): a 512x512 RGB render whose first deflate block came out with a
// broken dynamic Huffman header (the code-length sequence did not match the block's own code-length code). Photographic
// statistics drive the Huffman construction into branches flat or synthetic data do not (long codes, the length limits).
// The photos here: SpawnGen's broken render itself (TestData/PngPhoto/sdturbo_apple_512_rgb.png, the valid host-written copy of
// the same pixels, with the run's two tEXt chunks), and the three 1024x768 JPEG fixtures, decoded on the accelerator,
// encoded whole and as 512x512 crops; all in RGB as SpawnGen writes them, at every compression level. The host reader
// checks every chunk CRC, the zlib stream (Adler-32 included) and every pixel; a failure also reports whether the serial
// parse / serial emit path (no lanes) writes a valid file on the same accelerator.

using SpawnDev.Codecs.Images;
using SpawnDev.Codecs.Images.Jpeg;
using SpawnDev.Codecs.Images.Png;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task PngEncoderGpu_RealPhotos_512AndWhole_RoundTripOnHostReader()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var dec = new JpegDecoderGpu(acc);
            using var enc = new PngEncoderGpu(acc);
            var failures = new List<string>();
            var text = new Dictionary<string, string> { ["prompt"] = "a photo", ["workflow"] = new string('w', 3000) };
            // SpawnGen's broken render itself (512x512 RGB, its own two tEXt chunks): the pixels of the valid host-written copy
            {
                var apple = ReadPngOnHost(ReadEmbedded(PngPhotoPrefix + "sdturbo_apple_512_rgb.png")).Rgba8;
                var appleText = new Dictionary<string, string>
                {
                    ["prompt"] = ReadEmbeddedText(PngPhotoPrefix + "sdturbo_apple_512_rgb.prompt.txt"),
                    ["workflow"] = ReadEmbeddedText(PngPhotoPrefix + "sdturbo_apple_512_rgb.workflow.txt"),
                };
                foreach (int level in new[] { 1, 2, 3 })
                    await CheckPhotoPngAsync(enc, failures, $"SpawnGen sd-turbo apple 512", apple, 512, 512,
                        new PngEncodeOptions { ColorMode = PngColorMode.Rgb, CompressionLevel = level, TextChunks = appleText });
                // SpawnGen's call: the pixels already on the device, the file into a caller buffer (EncodeAsync(view, ..., view))
                {
                    var o = new PngEncodeOptions { ColorMode = PngColorMode.Rgb, TextChunks = appleText };
                    using var dIn = acc.Allocate1D<byte>(apple.Length);
                    DeviceTransfer.Upload(dIn.View, apple);
                    using var dOut = acc.Allocate1D<byte>(PngEncoderGpu.GetMaxEncodedSize(512, 512, o));
                    int n = await enc.EncodeAsync(dIn.View, 512, 512, dOut.View, o);
                    var file = await DeviceTransfer.DownloadAsync(dOut.View, n);
                    try { AssertPixelsEqual(apple, ReadPngOnHost(file).Rgba8, "SpawnGen sd-turbo apple 512, device in / caller buffer out"); }
                    catch (Exception e) { failures.Add($"device overload ({n} B): {e.Message}"); }
                }
            }
            foreach (var name in new[] { "big_c420_1024x768", "big_prog_c444_1024x768", "big_rst_c422_1024x768" })
            {
                var img = await dec.DecodeAsync(ReadEmbedded(JpegFixturePrefix + name + ".jpg"));
                int W = img.Info.Width, H = img.Info.Height;
                var cases = new List<(string What, byte[] Rgba, int W, int H)> { ($"{name} whole", img.Pixels, W, H) };
                foreach (var (x0, y0) in new[] { (0, 0), (512, 256), (256, 128) })
                    cases.Add(($"{name} 512 at {x0},{y0}", Crop(img.Pixels, W, x0, y0, 512, 512), 512, 512));
                foreach (var (what, rgba, w, h) in cases)
                    foreach (int level in new[] { 1, 2, 3 })
                        await CheckPhotoPngAsync(enc, failures, what, rgba, w, h,
                            new PngEncodeOptions { ColorMode = PngColorMode.Rgb, CompressionLevel = level, TextChunks = text });
            }
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    private const string PngPhotoPrefix = "SpawnDev.Codecs.Demo.Shared.TestData.PngPhoto.";

    private async Task CheckPhotoPngAsync(PngEncoderGpu enc, List<string> failures, string what, byte[] rgba, int w, int h, PngEncodeOptions o)
    {
        int level = o.CompressionLevel;
        byte[] file = await enc.EncodeAsync(rgba, w, h, o);
        string? dumpDir = OperatingSystem.IsBrowser() ? null : Environment.GetEnvironmentVariable("CODECS_DUMP_DIR");
        if (!string.IsNullOrEmpty(dumpDir))
        {
            Directory.CreateDirectory(dumpDir);
            File.WriteAllBytes(Path.Combine(dumpDir, $"photo_{what.Replace(' ', '_').Replace(',', '_')}_L{level}.png"), file);
        }
                        try
                        {
                            var png = ReadPngOnHost(file);
                            AssertPixelsEqual(rgba, png.Rgba8, $"{what} level {level}");
                        }
                        catch (Exception e)
                        {
                            // narrow it: the serial parse + serial emit (no lanes) on the same accelerator
                            string serial;
                            try
                            {
                                enc.SerialParse = true;
                                byte[] f2 = await enc.EncodeAsync(rgba, w, h, o);
                                try { AssertPixelsEqual(rgba, ReadPngOnHost(f2).Rgba8, "serial"); serial = $"serial path OK ({f2.Length} B, first difference from the default at byte {FirstDifference(f2, file)})"; }
                                catch (Exception e2) { serial = $"serial path ALSO bad: {e2.Message}"; }
                            }
                            finally { enc.SerialParse = false; }
                            failures.Add($"{what} level {level} ({file.Length} B): {e.Message}; {serial}");
                        }
    }

    private static byte[] Crop(byte[] rgba, int w, int x0, int y0, int cw, int ch)
    {
        var o = new byte[cw * ch * 4];
        for (int y = 0; y < ch; y++)
            Buffer.BlockCopy(rgba, ((y0 + y) * w + x0) * 4, o, y * cw * 4, cw * 4);
        return o;
    }
}
