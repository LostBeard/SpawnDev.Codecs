// Tests for the WebPAnimEncoder-driven LOSSLESS animation (WebPLosslessEncoderGpu.EncodeAnimationAsync with full-canvas
// frames + durations): what Pillow's `save(save_all=True, lossless=True, quality=80, method=4)` does - ComfyUI's Save
// Animated WEBP default.
//
// Fixtures: TestData/WebPAnimLossless (scripts/images/make_webp_anim_lossy_fixtures.py), Pillow's lossless files of the
// lossy fixtures' sources (moving patches, sprites, repeats, lone frames, transparent sprites, odd sizes, 1x1, scene cuts,
// duration overflow, EXIF, loop / background, kmin / kmax variants, jitter, semi-transparent layers).
//
// Our VP8L coder is not libwebp's, so the bytes differ. What must hold:
//   * every displayed frame of our file is EXACTLY its source frame (RGB under fully transparent pixels zeroed, as libwebp
//     without `exact`), at the source frame's display time (dropped repeats extend the previous frame);
//   * where libwebp's decisions do not depend on encoded sizes (key frames off, or every frame a key frame): the same
//     container as libwebp's file - chunk list, VP8X, ANIM, EXIF, every ANMF's offset / size / duration / disposal /
//     blending, the same chunk kinds inside.

using SpawnDev.Codecs.Images;
using SpawnDev.Codecs.Images.WebP;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private const string WebPAnimLosslessPrefix = "SpawnDev.Codecs.Demo.Shared.TestData.WebPAnimLossless.";

    private static List<WebPAnimLossyCase> WebPAnimLosslessCases()
    {
        var list = new List<WebPAnimLossyCase>();
        foreach (var line in System.Text.Encoding.UTF8.GetString(ReadEmbedded(WebPAnimLosslessPrefix + "manifest.tsv")).Split('\n'))
        {
            var row = line.TrimEnd('\r');
            if (row.Length == 0) continue;
            var c = row.Split('\t');
            int kmin = int.Parse(c[8]), kmax = int.Parse(c[9]);
            list.Add(new WebPAnimLossyCase(c[0], c[1], int.Parse(c[2]), int.Parse(c[3]), int.Parse(c[4]),
                c[5].Split(',').Select(int.Parse).ToArray(), int.Parse(c[6]), int.Parse(c[7]),
                kmin == -999 ? null : kmin, kmax == -999 ? null : kmax, c[10] == "1", int.Parse(c[11]),
                Convert.ToUInt32(c[12], 16)));
        }
        return list;
    }

    /// <summary>
    /// Every source frame against the frame our file shows at its time: null when all are exact (alpha-0 pixels compared as
    /// transparent black), else the first difference.
    /// </summary>
    private static async Task<string?> CheckLosslessAnimShowsSources(WebPDecoderGpu dec, byte[] file, WebPAnimLossyCase c, List<ReadOnlyMemory<byte>> sources)
    {
        var shown = await dec.ReadFramesAsync(file);
        var starts = new long[shown.Length];
        long t = 0;
        for (int j = 0; j < shown.Length; j++) { starts[j] = t; t += shown[j].DurationMs; }
        long time = 0;
        var cache = new Dictionary<int, byte[]>();
        for (int k = 0; k < sources.Count; k++)
        {
            int j = shown.Length - 1;
            while (j > 0 && starts[j] > time) j--;
            if (!cache.TryGetValue(j, out var canvas))
                cache[j] = canvas = (await dec.DecodeFrameAsync(file, j)).Pixels;
            var src = sources[k].Span;
            for (int i = 0; i < c.Width * c.Height; i++)
            {
                int p = 4 * i;
                bool clear = src[p + 3] == 0;
                for (int ch = 0; ch < 4; ch++)
                {
                    int want = clear ? 0 : src[p + ch];
                    if (canvas[p + ch] != want)
                        return $"source frame {k} (shown as file frame {j}): pixel ({i % c.Width},{i / c.Width}) channel {ch} = {canvas[p + ch]}, source {want}";
                }
            }
            time += c.Durations[k];
        }
        return null;
    }

    /// <summary>One ANMF frame's image chunks as a standalone still file (RIFF + its chunks), to decode the frame alone.</summary>
    private static byte[] WebPFrameAsStill(byte[] f, int anmfPayload, int anmfSize)
    {
        int n = anmfSize - 16;
        var still = new byte[12 + n];
        "RIFF"u8.CopyTo(still);
        BitConverter.GetBytes(4 + n).CopyTo(still, 4);
        "WEBP"u8.CopyTo(still.AsSpan(8));
        Array.Copy(f, anmfPayload + 16, still, 12, n);
        return still;
    }

    /// <summary>
    /// Our animated file against libwebp's where the decisions are size-independent: null when the containers match and
    /// every frame's own pixels (decoded alone, before compositing) are the same.
    /// </summary>
    private static async Task<string?> CompareWebPAnimContainers(WebPDecoderGpu dec, byte[] ours, byte[] libwebp)
    {
        var oc = WebPChunks(ours, out var oerr);
        if (oerr != null) return "our container: " + oerr;
        var lc = WebPChunks(libwebp, out _);
        string ot = string.Join(",", oc.Select(x => x.Tag)), lt = string.Join(",", lc.Select(x => x.Tag));
        if (ot != lt) return $"chunks {ot} vs libwebp {lt}";
        int frame = 0;
        for (int i = 0; i < oc.Count; i++)
        {
            var (tag, oo, os) = oc[i];
            var (_, lo, ls) = lc[i];
            if (tag is "VP8X" or "ANIM" or "EXIF" or "XMP " or "ICCP")
            {
                if (!ours.AsSpan(oo, os).SequenceEqual(libwebp.AsSpan(lo, ls))) return $"{tag} chunk differs";
            }
            else if (tag == "ANMF")
            {
                if (!ours.AsSpan(oo, 16).SequenceEqual(libwebp.AsSpan(lo, 16)))
                    return $"frame {frame}: ANMF parameters {Convert.ToHexString(ours, oo, 16)} vs libwebp {Convert.ToHexString(libwebp, lo, 16)}";
                string oit = string.Join(",", WebPInnerChunks(ours, oo + 16, oo + os).Select(x => x.Tag));
                string lit = string.Join(",", WebPInnerChunks(libwebp, lo + 16, lo + ls).Select(x => x.Tag));
                if (oit != lit) return $"frame {frame}: chunks {oit} vs libwebp {lit}";
                var op = (await dec.DecodeAsync(WebPFrameAsStill(ours, oo, os))).Pixels;
                var lp = (await dec.DecodeAsync(WebPFrameAsStill(libwebp, lo, ls))).Pixels;
                if (op.Length != lp.Length) return $"frame {frame}: {op.Length} vs libwebp {lp.Length} pixel bytes";
                // alpha everywhere, colour where alpha > 0: without `exact` libwebp's VP8L predictor leaves the colour of
                // fully transparent pixels to whatever it predicts (our coder writes 0), and they show nothing either way
                int transparentColourDiffs = 0;
                for (int at = 0; at < op.Length; at += 4)
                {
                    if (op[at + 3] != lp[at + 3])
                        return $"frame {frame}: alpha at pixel {at / 4} = {op[at + 3]}, libwebp {lp[at + 3]}";
                    bool sameColour = op[at] == lp[at] && op[at + 1] == lp[at + 1] && op[at + 2] == lp[at + 2];
                    if (sameColour) continue;
                    if (op[at + 3] != 0)
                        return $"frame {frame}: colour at pixel {at / 4} (alpha {op[at + 3]}) = {op[at]},{op[at + 1]},{op[at + 2]}, libwebp {lp[at]},{lp[at + 1]},{lp[at + 2]}";
                    transparentColourDiffs++;
                }
                _ = transparentColourDiffs;
                frame++;
            }
        }
        return null;
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task WebPLosslessEncoderGpu_AnimEncoder_ShowsSourcesExactly_AndMatchesLibwebpDecisions()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            using var dec = new WebPDecoderGpu(acc);
            var cache = new Dictionary<string, byte[]>();
            var failures = new List<string>();
            var cases = WebPAnimLosslessCases();
            int structural = 0;
            foreach (var c in cases)
            {
                try
                {
                    var frames = WebPAnimFrames(c, cache);
                    byte[]? exif = c.Exif ? ReadEmbedded(WebPAnimLossyPrefix + c.Name + ".exif") : null;
                    var ours = await enc.EncodeAnimationAsync(frames, c.Width, c.Height, c.Durations,
                        new WebPEncodeOptions { Quality = c.Quality, Exact = false }, c.Animation(exif));
                    var shows = await CheckLosslessAnimShowsSources(dec, ours, c, frames);
                    if (shows != null) { failures.Add($"{c.Name}: {shows}"); continue; }
                    // libwebp decides by encoded size only for key frames (after kmin frames; Pillow's lossless default
                    // 9) and for a lone sub-rectangle frame (OptimizeSingleFrame); everything else is size-independent
                    var libwebp = ReadEmbedded(WebPAnimLosslessPrefix + c.Name + ".webp");
                    bool keyFramesSizeFree = (c.Kmax is int km && km <= 1) || c.Frames <= (c.Kmin ?? 9) + 1;
                    bool lone = WebPChunks(libwebp, out _).Count(x => x.Tag == "ANMF") == 1;
                    if (keyFramesSizeFree && !lone)
                    {
                        structural++;
                        var diff = await CompareWebPAnimContainers(dec, ours, libwebp);
                        if (diff != null) failures.Add($"{c.Name}: {diff}");
                    }
                }
                catch (Exception ex)
                {
                    failures.Add($"{c.Name}: {ex.GetType().Name}: {ex.Message}");
                }
            }
            Console.WriteLine($"WebP lossless animation: {cases.Count} fixtures show their sources exactly, {structural} also match libwebp's container and frames");
            True(cases.Count >= 20 && structural >= 12, $"{cases.Count} fixtures, {structural} size-independent");
            True(failures.Count == 0, $"{failures.Count} of {cases.Count}:\n" + string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
