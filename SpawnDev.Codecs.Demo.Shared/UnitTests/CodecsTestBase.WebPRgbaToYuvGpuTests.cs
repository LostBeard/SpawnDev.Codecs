// Tests for WebPRgbaToYuvGpu (Images/Yuv): RGBA8 -> YUV 4:2:0 + alpha as libwebp's lossy encoder imports a picture, with and
// without WebPCleanupTransparentArea.
//
// Fixtures (TestData/WebPRgbToYuv, scripts/images/make_webp_rgb2yuv_fixtures.py): sizes 1x1 .. 255x129 (odd widths and
// heights hit the duplicated-sample edge rules) x alpha kinds opaque / ramp / disc / holes / noise / clear, plus 640x480 and
// 1920x1080 procedural pictures (integer formula rebuilt here; truth = SHA-256). The plain truth is LIBWEBP's own
// WebPPictureImportRGBA output (read through cffi); the cleaned truth applies a line-by-line port of
// WebPCleanupTransparentArea to it. Neither side is our code.
//
// What can fail: the luma formula, the gamma tables or their interpolation, the alpha-weighted average and the kInvAlpha
// reciprocal, the 0 / 4*255 unweighted cases, the odd edge duplication, the U/V rounding and clip, the alpha plane, the
// block smoothing (mean of visible luma) and the run flattening (value carried along a strip), and writes past the planes
// (all outputs are poisoned with 0xAB and longer than needed).

using System.Security.Cryptography;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Images.Yuv;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private const string WebPRgbToYuvPrefix = "SpawnDev.Codecs.Demo.Shared.TestData.WebPRgbToYuv.";

    /// <summary>The integer picture of make_webp_rgb2yuv_fixtures.py procedural() (same bytes).</summary>
    private static byte[] ProceduralRgba(int w, int h)
    {
        var p = new byte[(long)w * h * 4];
        long r2 = (long)Math.Min(w, h) * Math.Min(w, h);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                long i = ((long)y * w + x) * 4;
                p[i] = (byte)((x * 5 + y * 3) & 255);
                p[i + 1] = (byte)(((x ^ y) * 7) & 255);
                p[i + 2] = (byte)(((long)x * y >> 3) & 255);
                long dx = 2L * x - w, dy = 2L * y - h, d2 = dx * dx + dy * dy;
                int a = 9 * d2 < 4 * r2 ? 255 : (121 * d2 < 100 * r2 ? 97 : 0);
                bool hole = ((x >> 3) % 5 < 3) && ((y >> 3) % 3 == 1) && 9 * d2 >= 4 * r2;
                p[i + 3] = (byte)(hole ? 0 : a);
            }
        return p;
    }

    private static async Task<byte[]> RgbaToYuvOnDeviceAsync(Accelerator acc, WebPRgbaToYuvGpu conv, byte[] rgba, int w, int h, bool cleanup)
    {
        long n = (long)w * h, c = WebPRgbaToYuvGpu.ChromaPlaneSize(w, h);
        using var dIn = acc.Allocate1D<byte>(rgba.Length + 5);
        DeviceTransfer.Upload(dIn.View, rgba);
        using var dY = acc.Allocate1D<byte>(n + 7);
        using var dU = acc.Allocate1D<byte>(c + 7);
        using var dV = acc.Allocate1D<byte>(c + 7);
        using var dA = acc.Allocate1D<byte>(n + 7);
        foreach (var b in new[] { dY, dU, dV, dA }) DeviceTransfer.Upload(b.View, Poison((int)b.Length));
        conv.Convert(dIn.View, w, h, dY.View, dU.View, dV.View, dA.View, cleanup);
        await acc.SynchronizeAsync();
        var y = await DeviceTransfer.DownloadAsync(dY.View, dY.Length);
        var u = await DeviceTransfer.DownloadAsync(dU.View, dU.Length);
        var v = await DeviceTransfer.DownloadAsync(dV.View, dV.Length);
        var a = await DeviceTransfer.DownloadAsync(dA.View, dA.Length);
        foreach (var (plane, len, what) in new[] { (y, n, "Y"), (u, c, "U"), (v, c, "V"), (a, n, "A") })
            for (long i = len; i < plane.Length; i++)
                if (plane[i] != 0xAB) throw new Exception($"wrote past the end of the {what} plane");
        var all = new byte[2 * n + 2 * c];
        Buffer.BlockCopy(y, 0, all, 0, (int)n);
        Buffer.BlockCopy(u, 0, all, (int)n, (int)c);
        Buffer.BlockCopy(v, 0, all, (int)(n + c), (int)c);
        Buffer.BlockCopy(a, 0, all, (int)(n + 2 * c), (int)n);
        return all;
    }

    private static string? CompareYuva(byte[] expected, byte[] got, int w, int h)
    {
        if (expected.Length != got.Length) return $"length {got.Length} != {expected.Length}";
        long n = (long)w * h, c = WebPRgbaToYuvGpu.ChromaPlaneSize(w, h);
        int cw = (w + 1) / 2;
        for (long i = 0; i < expected.Length; i++)
        {
            if (expected[i] == got[i]) continue;
            string plane; long off; int pw;
            if (i < n) { plane = "Y"; off = i; pw = w; }
            else if (i < n + c) { plane = "U"; off = i - n; pw = cw; }
            else if (i < n + 2 * c) { plane = "V"; off = i - n - c; pw = cw; }
            else { plane = "A"; off = i - n - 2 * c; pw = w; }
            return $"first mismatch in {plane} at x={off % pw} y={off / pw}: got {got[i]}, expected {expected[i]}";
        }
        return null;
    }

    [TestMethod(Timeout = 600000)]
    public async Task WebPRgbaToYuvGpu_AllFixtures_MatchLibwebpImport_WithAndWithoutCleanup()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var conv = new WebPRgbaToYuvGpu(acc);
            var text = System.Text.Encoding.UTF8.GetString(ReadEmbedded(WebPRgbToYuvPrefix + "manifest.tsv"));
            var failures = new List<string>();
            int checkedCount = 0, hashed = 0;
            foreach (var line in text.Split('\n'))
            {
                var row = line.TrimEnd('\r');
                if (row.Length == 0) continue;
                var c = row.Split('\t');
                string name = c[0];
                int w = int.Parse(c[1]), h = int.Parse(c[2]);
                try
                {
                    bool big = HasEmbedded(WebPRgbToYuvPrefix + name + ".sha256");
                    byte[] rgba;
                    string[]? hashes = null;
                    if (big)
                    {
                        hashes = System.Text.Encoding.ASCII.GetString(ReadEmbedded(WebPRgbToYuvPrefix + name + ".sha256"))
                            .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                        rgba = ProceduralRgba(w, h);
                        if (Sha256Hex(rgba) != hashes[0]) { failures.Add($"{name}: procedural input differs from the generator's"); continue; }
                    }
                    else rgba = Gunzip(ReadEmbedded(WebPRgbToYuvPrefix + name + ".rgba.gz"));

                    foreach (bool cleanup in new[] { false, true })
                    {
                        var got = await RgbaToYuvOnDeviceAsync(acc, conv, rgba, w, h, cleanup);
                        string? diff;
                        if (big)
                            diff = Sha256Hex(got) == hashes![cleanup ? 2 : 1] ? null : "SHA-256 of the planes differs from libwebp's";
                        else
                            diff = CompareYuva(Gunzip(ReadEmbedded(WebPRgbToYuvPrefix + name + (cleanup ? ".clean.yuva.gz" : ".yuva.gz"))), got, w, h);
                        if (diff != null) failures.Add($"{name} [{(cleanup ? "cleanup" : "plain")}]: {diff}");
                    }
                    checkedCount++;
                    if (big) hashed++;
                }
                catch (Exception ex)
                {
                    failures.Add($"{name}: {ex.GetType().Name}: {ex.Message}");
                }
            }
            True(failures.Count == 0, $"{failures.Count} failure(s):\n" + string.Join("\n", failures.Take(20)));
            True(checkedCount >= 70, $"only {checkedCount} fixtures were checked");
            True(hashed == 2, $"expected the 2 production-scale cases, checked {hashed}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
