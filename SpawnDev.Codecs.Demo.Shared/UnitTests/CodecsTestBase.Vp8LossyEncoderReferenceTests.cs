// Tests for Vp8LossyEncoderReference (SpawnDev.Codecs.References): the CPU port of libwebp's lossy encoder must produce the
// SAME BYTES as libwebp for the same YUV input and config. It is the oracle the accelerator encoder will be checked against.
//
// Fixtures (TestData/WebPLossyEnc, scripts/images/make_webp_lossy_encode_fixtures.py): 12 opaque source pictures (photo-like,
// graphics with flat areas and hard edges, noise, flat, 1x1 .. 640x480, odd sizes), the YUV libwebp itself imports for each,
// and Pillow's file for qualities 0..100 and methods 0..4 (WebPConfigInit defaults otherwise, exactly what Pillow passes).
//
// What can fail: anything in the port (analysis / segmentation, quantizer and lambda setup, every prediction, transform and
// quantization step, the RD scoring and mode decisions, the error diffusion, the token statistics and the probability
// refreshes, the bool encoder, the headers and the RIFF container). A failure names the first differing byte and how many
// fixtures already match, which localises the stage (headers vs partition 0 vs tokens).

using SpawnDev.Codecs.References.Images.WebP.Lossy;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private const string WebPLossyEncPrefix = "SpawnDev.Codecs.Demo.Shared.TestData.WebPLossyEnc.";

    [TestMethod(Category = "CpuOnly")]
    public void Vp8LossyEncoderReference_AllFixtures_ByteIdenticalWithLibwebp()
    {
        CpuOnlyTest();
        var text = System.Text.Encoding.UTF8.GetString(ReadEmbedded(WebPLossyEncPrefix + "manifest.tsv"));
        var yuvCache = new Dictionary<string, byte[]>();
        var failures = new List<string>();
        int identical = 0, total = 0;
        var byMethod = new Dictionary<int, (int ok, int all)>();
        foreach (var line in text.Split('\n'))
        {
            var row = line.TrimEnd('\r');
            if (row.Length == 0) continue;
            var c = row.Split('\t');
            string name = c[0], src = c[1];
            int w = int.Parse(c[2]), h = int.Parse(c[3]), q = int.Parse(c[4]), m = int.Parse(c[5]);
            total++;
            try
            {
                if (!yuvCache.TryGetValue(src, out var yuv))
                    yuvCache[src] = yuv = Gunzip(ReadEmbedded(WebPLossyEncPrefix + src + ".yuv.gz"));
                int n = w * h, cw = (w + 1) / 2, ch = (h + 1) / 2;
                var y = yuv.AsSpan(0, n).ToArray();
                var u = yuv.AsSpan(n, cw * ch).ToArray();
                var v = yuv.AsSpan(n + cw * ch, cw * ch).ToArray();
                var expected = ReadEmbedded(WebPLossyEncPrefix + name + ".webp");
                var got = Vp8LossyEncoderReference.EncodeWebP(y, u, v, w, h, new Vp8LossyEncoderConfig { Quality = q, Method = m });
                bool same = got.AsSpan().SequenceEqual(expected);
                // Debugging aid: VP8ENC_DUMP_DIR=<dir> writes our file of every differing fixture next to libwebp's.
                var dumpDir = Environment.GetEnvironmentVariable("VP8ENC_DUMP_DIR");
                if (!same && !string.IsNullOrEmpty(dumpDir))
                {
                    File.WriteAllBytes(Path.Combine(dumpDir, name + ".ours.webp"), got);
                    File.WriteAllBytes(Path.Combine(dumpDir, name + ".libwebp.webp"), expected);
                }
                var (ok, all) = byMethod.GetValueOrDefault(m);
                byMethod[m] = (ok + (same ? 1 : 0), all + 1);
                if (same) { identical++; continue; }
                int first = 0;
                int lim = Math.Min(got.Length, expected.Length);
                while (first < lim && got[first] == expected[first]) first++;
                failures.Add($"{name}: {got.Length} bytes vs libwebp {expected.Length}, first difference at byte {first}" +
                    (first < lim ? $" (got 0x{got[first]:x2}, expected 0x{expected[first]:x2})" : " (one is a prefix of the other)"));
            }
            catch (Exception ex)
            {
                failures.Add($"{name}: {ex.GetType().Name}: {ex.Message}");
            }
        }
        var summary = string.Join(", ", byMethod.OrderBy(k => k.Key).Select(k => $"m{k.Key} {k.Value.ok}/{k.Value.all}"));
        var names = string.Join(" ", failures.Select(f => f.Split(':')[0]));
        True(failures.Count == 0, $"{identical} of {total} byte-identical ({summary}); {failures.Count} differ:\n" + string.Join("\n", failures.Take(25)) + "\nall: " + names);
        True(total >= 500, $"only {total} fixtures");
    }
}
