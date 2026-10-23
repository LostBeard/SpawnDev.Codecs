// Tests for WebPYuvToRgbaGpu (Images/Yuv): the libwebp-exact 4:2:0 -> RGBA8 step that follows the VP8 keyframe decoder
// in a lossy WebP decode.
//
// Fixtures (TestData/WebPYuv, scripts/images/make_webp_yuv_fixtures.py): lossy WebPs from Pillow/libwebp at 1x1 ..
// 320x200 including odd sizes. The YUV input is the planes FFMPEG's native VP8 decoder produces for the file, the
// expected RGBA8 is what LIBWEBP produces (Pillow decode, default fancy upsampling). Neither side comes from our code, so
// matching proves both the upsampler and the colour matrix.
//
// What can fail: any change to the 9/3/3/1 weights, the rounding, the row pairing at the top / bottom edge, the
// trailing column of an even width, the matrix constants, the clip, or the output layout. Sizes 1x1, 2x2, 3x3, 4x1,
// 1x4 hit the edge rules; the buffers are poisoned with 0xAB so a short write shows up.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Images.Yuv;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private const string WebPYuvPrefix = "SpawnDev.Codecs.Demo.Shared.TestData.WebPYuv.";

    private static string[] WebPYuvFixtureNames()
    {
        var names = typeof(CodecsTestBase).Assembly.GetManifestResourceNames()
            .Where(n => n.StartsWith(WebPYuvPrefix, StringComparison.Ordinal) && n.EndsWith(".yuv", StringComparison.Ordinal))
            .Select(n => n.Substring(WebPYuvPrefix.Length, n.Length - WebPYuvPrefix.Length - 4))
            .OrderBy(n => n, StringComparer.Ordinal)
            .ToArray();
        if (names.Length == 0) throw new InvalidOperationException("No WebPYuv fixtures are embedded.");
        return names;
    }

    private static (int w, int h) ParseWebPYuvSize(string name)
    {
        // yuv_<w>x<h>_q<q>
        var size = name.Split('_')[1].Split('x');
        return (int.Parse(size[0]), int.Parse(size[1]));
    }

    [TestMethod(Timeout = 600000)]
    public async Task WebPYuvToRgbaGpu_AllFixtures_MatchLibwebpDecode()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var conv = new WebPYuvToRgbaGpu(acc);
            var failures = new List<string>();
            int checkedCount = 0;
            foreach (var name in WebPYuvFixtureNames())
            {
                var (w, h) = ParseWebPYuvSize(name);
                byte[] yuv = ReadEmbedded(WebPYuvPrefix + name + ".yuv");
                byte[] expected = ReadEmbedded(WebPYuvPrefix + name + ".rgba8");
                int cw = (w + 1) / 2, ch = (h + 1) / 2;
                int yLen = w * h, cLen = cw * ch;
                if (yuv.Length != yLen + 2 * cLen) { failures.Add($"{name}: yuv fixture is {yuv.Length} bytes, expected {yLen + 2 * cLen}"); continue; }
                try
                {
                    using var dY = acc.Allocate1D<byte>(yLen + 3);
                    using var dU = acc.Allocate1D<byte>(cLen + 3);
                    using var dV = acc.Allocate1D<byte>(cLen + 3);
                    using var dOut = acc.Allocate1D<byte>(expected.Length + 9);
                    DeviceTransfer.Upload(dY.View.SubView(0, yLen), yuv.AsSpan(0, yLen));
                    DeviceTransfer.Upload(dU.View.SubView(0, cLen), yuv.AsSpan(yLen, cLen));
                    DeviceTransfer.Upload(dV.View.SubView(0, cLen), yuv.AsSpan(yLen + cLen, cLen));
                    var fill = new byte[dOut.Length];
                    System.Array.Fill(fill, (byte)0xAB);
                    DeviceTransfer.Upload(dOut.View, fill);

                    conv.Convert(dY.View, dU.View, dV.View, w, h, dOut.View);
                    await acc.SynchronizeAsync();
                    var all = await DeviceTransfer.DownloadAsync(dOut.View, dOut.Length);

                    var diff = ComparePixels(expected, all.AsSpan(0, expected.Length).ToArray(), 4, w);
                    if (diff != null) failures.Add($"{name}: {diff}");
                    for (int i = expected.Length; i < all.Length; i++)
                        if (all[i] != 0xAB) { failures.Add($"{name}: wrote past the end of the image at byte {i}"); break; }
                    checkedCount++;
                }
                catch (Exception ex)
                {
                    failures.Add($"{name}: threw {ex.GetType().Name}: {ex.Message}");
                }
            }
            True(checkedCount >= 20, $"only {checkedCount} fixtures were checked");
            True(failures.Count == 0, $"{failures.Count} failure(s):\n" + string.Join("\n", failures.Take(15)));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod]
    public async Task WebPYuvToRgbaGpu_RejectsShortBuffersAndBadSizes()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var conv = new WebPYuvToRgbaGpu(acc);
            using var y = acc.Allocate1D<byte>(16);
            using var u = acc.Allocate1D<byte>(4);
            using var v = acc.Allocate1D<byte>(4);
            using var o = acc.Allocate1D<byte>(64);
            bool shortU = false, zeroW = false, shortOut = false;
            try { conv.Convert(y.View, u.View.SubView(0, 3), v.View, 4, 4, o.View); } catch (ArgumentException) { shortU = true; }
            try { conv.Convert(y.View, u.View, v.View, 0, 4, o.View); } catch (ArgumentOutOfRangeException) { zeroW = true; }
            try { conv.Convert(y.View, u.View, v.View, 4, 4, o.View.SubView(0, 63)); } catch (ArgumentException) { shortOut = true; }
            True(shortU, "a short U plane must be rejected");
            True(zeroW, "width 0 must be rejected");
            True(shortOut, "a short output must be rejected");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
