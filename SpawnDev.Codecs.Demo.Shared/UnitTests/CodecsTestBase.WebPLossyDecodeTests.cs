// Tests for the lossy half of WebPDecoderGpu: VP8 keyframe decode (Vp8FullKeyframeDecoderGpu) -> libwebp-exact fancy-upsampled
// RGBA (WebPYuvToRgbaGpu) -> ALPH plane merged into channel A, stills and animations (lossy and mixed frames).
//
// Fixtures (TestData/WebPLossy/manifest.tsv, scripts/images/make_webp_lossy_decode_fixtures.py): stills with alpha at
// 1x1 .. 255x129 and 1920x1080, encoder methods 0..6, qualities 20..100, alpha kinds ramp / disc / noise / opaque;
// animations of lossy frames and of mixed lossy + lossless frames from libwebp's animation encoder; plus every lossy
// file already in TestData/WebP (the crafted-ALPH family, the metadata files) and TestData/WebPYuv. The expected pixels
// are what LIBWEBP shows (Pillow decode, default config). Pictures up to 70000 pixels carry the full RGBA truth (a
// failure names the first wrong pixel); bigger ones are compared by SHA-256 of the canvas.
//
// What can fail: any step of the VP8 decode, the fancy upsampler or matrix, the alpha merge, the frame compositing of a
// lossy sub-rectangle, a buffer that is too small when a decoder instance moves between sizes and kinds (one decoder
// decodes every fixture in turn), a write past the end of the caller's buffer (poisoned with 0xAB), and the error paths
// (corrupt VP8 data must surface as ImageDecodeException, not as a VP8 or ILGPU exception).

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
    private const string WebPLossyPrefix = "SpawnDev.Codecs.Demo.Shared.TestData.WebPLossy.";

    private sealed record WebPLossyRow(string Name, string Folder, int Frames, int Width, int Height, string[] Hashes, bool HasTruth);

    private static List<WebPLossyRow> WebPLossyManifest()
    {
        var text = System.Text.Encoding.UTF8.GetString(ReadEmbedded(WebPLossyPrefix + "manifest.tsv"));
        var rows = new List<WebPLossyRow>();
        foreach (var raw in text.Split('\n'))
        {
            var line = raw.TrimEnd('\r');
            if (line.Length == 0) continue;
            var c = line.Split('\t');
            rows.Add(new WebPLossyRow(c[0], c[1], int.Parse(c[2]), int.Parse(c[3]), int.Parse(c[4]), c[5].Split(','), c[6] == "1"));
        }
        if (rows.Count == 0) throw new InvalidOperationException("The lossy WebP manifest is empty.");
        return rows;
    }

    private static string WebPLossyFolderPrefix(string folder) => "SpawnDev.Codecs.Demo.Shared.TestData." + folder + ".";

    private static byte[] WebPLossyFile(WebPLossyRow r) => ReadEmbedded(WebPLossyFolderPrefix(r.Folder) + r.Name + ".webp");

    /// <summary>Full truth of frame <paramref name="k"/> (the .gz lives in TestData/WebPLossy for every folder).</summary>
    private static byte[] WebPLossyTruth(WebPLossyRow r, int k)
        => Gunzip(ReadEmbedded(WebPLossyPrefix + r.Name + (r.Frames == 1 ? ".rgba8.gz" : ".f" + k + ".rgba8.gz")));

    private static string? CheckLossyFrame(WebPLossyRow r, int k, byte[] pixels, int width)
    {
        if (r.HasTruth) return ComparePixels(WebPLossyTruth(r, k), pixels, 4, width);
        return Sha256Hex(pixels) == r.Hashes[k] ? null : $"canvas SHA-256 differs from libwebp's ({pixels.Length} bytes)";
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task WebPDecoderGpu_Lossy_AllFixtures_MatchLibwebp()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var dec = new WebPDecoderGpu(acc);   // one instance decodes everything: buffers must grow and be reused correctly
            var failures = new List<string>();
            int checkedFrames = 0, animated = 0, withAlpha = 0;
            foreach (var row in WebPLossyManifest())
            {
                byte[] file = WebPLossyFile(row);
                try
                {
                    // Frame table facts first: a wrong size here would make every pixel check below meaningless.
                    var info = await dec.ReadInfoAsync(file);
                    if (info.Width != row.Width || info.Height != row.Height || info.FrameCount != row.Frames)
                        failures.Add($"{row.Name}: info {info.Width}x{info.Height} x{info.FrameCount}, libwebp says {row.Width}x{row.Height} x{row.Frames}");
                    if (info.HasAlpha) withAlpha++;
                    if (row.Frames > 1) animated++;

                    for (int k = 0; k < row.Frames; k++)
                    {
                        var managed = await dec.DecodeFrameAsync(file, k);
                        var diff = CheckLossyFrame(row, k, managed.Pixels, row.Width);
                        if (diff != null) failures.Add($"{row.Name} frame {k} [managed]: {diff}");
                        checkedFrames++;
                    }

                    // The ArrayView1D overload into a longer, poisoned caller buffer (first and last frame).
                    int bytes = row.Width * row.Height * 4;
                    var (vInfo, viewPixels) = await WebPDecodeIntoViewAsync(acc, dec, file, bytes, row.Frames > 1 ? row.Frames - 1 : null);
                    var vDiff = CheckLossyFrame(row, row.Frames - 1, viewPixels, vInfo.Width);
                    if (vDiff != null) failures.Add($"{row.Name} [ArrayView1D last frame]: {vDiff}");
                }
                catch (Exception ex)
                {
                    failures.Add($"{row.Name}: {ex.GetType().Name}: {ex.Message}");
                }
            }
            True(failures.Count == 0, $"{failures.Count} failure(s):\n" + string.Join("\n", failures.Take(20)));
            True(checkedFrames >= 150, $"only {checkedFrames} frames were checked");
            True(animated >= 6, $"only {animated} animated fixtures");
            True(withAlpha >= 80, $"only {withAlpha} fixtures with alpha");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>Offset of the first VP8 payload byte in a simple lossy file (RIFF header 12 + "VP8 " + size 8).</summary>
    private const int SimpleLossyPayload = 20;

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task WebPDecoderGpu_Lossy_CorruptVp8Data_IsRejectedAsImageDecodeException()
    {
        var good = ReadEmbedded(WebPLossyFolderPrefix("WebPYuv") + "yuv_64x64_q90.webp");
        Equal((byte)'V', good[12], "fixture is a simple lossy file");
        Equal((byte)' ', good[15], "chunk is 'VP8 '");

        var badStart = (byte[])good.Clone();
        badStart[SimpleLossyPayload + 3] ^= 0xFF;   // 9D 01 2A start code
        await AssertWebPRejectedAsync(badStart, "invalid VP8", "bad VP8 start code");

        var interFrame = (byte[])good.Clone();
        interFrame[SimpleLossyPayload] |= 1;   // frame tag bit 0 = 1: not a key frame
        await AssertWebPRejectedAsync(interFrame, "invalid VP8", "VP8 inter frame");

        var hugePartition = (byte[])good.Clone();
        hugePartition[SimpleLossyPayload + 1] = 0xFF;   // first-partition size (bits 5..23 of the tag) far past the frame
        hugePartition[SimpleLossyPayload + 2] = 0xFF;
        await AssertWebPRejectedAsync(hugePartition, "invalid VP8", "first partition larger than the frame");
    }

    [TestMethod(Category = "NotOnWebGL")]
    public async Task WebPDecoderGpu_Lossy_DecoderStaysUsableAcrossKindsAndSizes()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var dec = new WebPDecoderGpu(acc);
            var rows = WebPLossyManifest();
            var small = rows.First(r => r.Name == "ls_7x5_q75_aq40_disc");
            var big = rows.First(r => r.Name == "ls_big_640x480_q80");
            var anim = rows.First(r => r.Name == "la_mixed_64x48");

            // small lossy -> lossless -> big lossy -> animation -> small again: each must match, none may poison the next
            void Matches(WebPLossyRow row, int k, byte[] pixels, string what)
            {
                var diff = CheckLossyFrame(row, k, pixels, row.Width);
                True(diff == null, $"{what}: {diff}");
            }
            var a = await dec.DecodeAsync(WebPLossyFile(small));
            Matches(small, 0, a.Pixels, "small lossy first");
            var l = await dec.DecodeAsync(WebPFile("flat_32x24"));
            Equal(32, l.Info.Width, "lossless in between");
            var b = await dec.DecodeAsync(WebPLossyFile(big));
            Matches(big, 0, b.Pixels, "big lossy");
            var c = await dec.DecodeFrameAsync(WebPLossyFile(anim), anim.Frames - 1);
            Matches(anim, anim.Frames - 1, c.Pixels, "last frame of the mixed animation");
            var d = await dec.DecodeAsync(WebPLossyFile(small));
            Matches(small, 0, d.Pixels, "small lossy again");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
