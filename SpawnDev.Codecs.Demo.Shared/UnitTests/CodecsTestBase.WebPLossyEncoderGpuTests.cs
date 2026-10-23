// Tests for WebPLossyEncoderGpu (Images/WebP/Lossy): the accelerator port of libwebp's lossy encoder must write the SAME BYTES
// as libwebp for the same YUV planes and settings.
//
// Fixtures: TestData/WebPLossyEnc (scripts/images/make_webp_lossy_encode_fixtures.py), the ones of methods 3 and 4 (the
// methods ported to the accelerator): libwebp's own imported planes in, Pillow's file as the expected bytes, qualities
// 0..100, 1x1 .. 640x480. The CPU reference (Vp8LossyEncoderReference) is byte-identical on the same set, so a failure here
// is in the kernels.

using SpawnDev.Codecs.Images.WebP.Lossy;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Timeout = 600000)]
    public async Task WebPLossyEncoderGpu_Methods3And4_ByteIdenticalWithLibwebp()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new WebPLossyEncoderGpu(acc);
            var text = System.Text.Encoding.UTF8.GetString(ReadEmbedded(WebPLossyEncPrefix + "manifest.tsv"));
            var yuvCache = new Dictionary<string, byte[]>();
            var failures = new List<string>();
            int identical = 0, total = 0;
            foreach (var line in text.Split('\n'))
            {
                var row = line.TrimEnd('\r');
                if (row.Length == 0) continue;
                var c = row.Split('\t');
                string name = c[0], src = c[1];
                int w = int.Parse(c[2]), h = int.Parse(c[3]), q = int.Parse(c[4]), m = int.Parse(c[5]);
                if (m < 3) continue;
                total++;
                try
                {
                    if (!yuvCache.TryGetValue(src, out var yuv))
                        yuvCache[src] = yuv = Gunzip(ReadEmbedded(WebPLossyEncPrefix + src + ".yuv.gz"));
                    var expected = ReadEmbedded(WebPLossyEncPrefix + name + ".webp");
                    var got = await enc.EncodeYuvAsync(yuv, w, h, new WebPLossyEncodeOptions { Quality = q, Method = m });
                    if (got.AsSpan().SequenceEqual(expected)) { identical++; continue; }
                    int first = 0, lim = Math.Min(got.Length, expected.Length);
                    while (first < lim && got[first] == expected[first]) first++;
                    failures.Add($"{name}: {got.Length} bytes vs libwebp {expected.Length}, first difference at byte {first}");
                }
                catch (Exception ex)
                {
                    failures.Add($"{name}: {ex.GetType().Name}: {ex.Message}");
                }
            }
            True(failures.Count == 0, $"{identical} of {total} byte-identical; {failures.Count} differ:\n" + string.Join("\n", failures.Take(25)));
            True(total >= 200, $"only {total} fixtures");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task WebPLossyEncoderGpu_UnportedMethods_AreRefused()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var enc = new WebPLossyEncoderGpu(acc);
            var rgba = new byte[16 * 16 * 4];
            foreach (int m in new[] { 0, 1, 2, 5, 6 })
            {
                bool threw = false;
                try { await enc.EncodeAsync(rgba, 16, 16, new WebPLossyEncodeOptions { Method = m }); }
                catch (NotSupportedException) { threw = true; }
                True(threw, $"method {m} must be refused, not encoded differently");
            }
            // transparency is refused too until the ALPH chunk is ported
            bool alphaThrew = false;
            try { await enc.EncodeAsync(rgba, 16, 16); }   // all-zero RGBA = fully transparent
            catch (NotSupportedException) { alphaThrew = true; }
            True(alphaThrew, "a transparent picture must be refused");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
