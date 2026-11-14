// The parallel VP8L tokeniser (Vp8lTokenizeParallel) must produce what the serial walk (Vp8lEncodeGpu.TokenizeStep)
// produces: the same tokens, histograms and extra bits, so the same file, byte for byte. Every WebP source fixture at five
// efforts (no LZ77 chain, short / long chains, colour cache sizes, meta prefix groups), plus pictures built to stress it:
// one flat colour (every copy is the longest), noise (no copies, every literal), a 7-pixel period (copies only the hash
// chain finds), a gradient with transparent pixels, and a production size (1920x1080 texture).

using SpawnDev.Codecs.Images.WebP;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] WebPTokenizeStressPicture(int kind, int w, int h)
    {
        var rgba = new byte[w * h * 4];
        var rng = new Random(kind * 977 + w);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int i = (y * w + x) * 4;
                byte r, g, b, a = 255;
                if (kind == 0) { r = 40; g = 90; b = 200; }
                else if (kind == 1) { r = (byte)rng.Next(256); g = (byte)rng.Next(256); b = (byte)rng.Next(256); }
                else if (kind == 2) { int k = (x + y * 3) % 7; r = (byte)(k * 30); g = (byte)(200 - k * 20); b = (byte)(k * 11); }
                else if (kind == 3) { r = (byte)x; g = (byte)(y * 2); b = (byte)(x ^ y); a = (byte)((x / 9 + y / 5) % 3 == 0 ? 0 : 255); }
                else { r = Vp9TestTexture(x, y); g = Vp9TestTexture(x + 300, y + 77); b = Vp9TestTexture(x + 600, y + 400); }
                rgba[i] = r; rgba[i + 1] = g; rgba[i + 2] = b; rgba[i + 3] = a;
            }
        return rgba;
    }

    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL")]
    public async Task WebPLosslessEncoderGpu_ParallelTokenizer_MatchesSerialWalk()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, AtomicKernels);
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            var failures = new List<string>();
            var cases = new List<(string Name, byte[] Rgba, int W, int H, int[] Qualities)>();
            foreach (var row in WebPSources())
                cases.Add((row.Name, WebPSource(row.Name), row.Width, row.Height, new[] { 0, 30, 60, 80, 95 }));
            cases.Add(("flat 300x200", WebPTokenizeStressPicture(0, 300, 200), 300, 200, new[] { 80, 95 }));
            cases.Add(("noise 257x129", WebPTokenizeStressPicture(1, 257, 129), 257, 129, new[] { 30, 80 }));
            cases.Add(("period 7 640x48", WebPTokenizeStressPicture(2, 640, 48), 640, 48, new[] { 30, 80, 95 }));
            cases.Add(("gradient + transparency 512x352", WebPTokenizeStressPicture(3, 512, 352), 512, 352, new[] { 60, 80 }));
            if (!OperatingSystem.IsBrowser()) cases.Add(("texture 1920x1080", WebPTokenizeStressPicture(4, 1920, 1080), 1920, 1080, new[] { 80 }));
            foreach (var (name, rgba, w, h, qs) in cases)
                foreach (int q in qs)
                    foreach (bool exact in new[] { true, false })
                    {
                        var o = new WebPEncodeOptions { Quality = q, Exact = exact };
                        WebPEncodeKernels.SerialTokenize = true;
                        byte[] serial = await enc.EncodeAsync(rgba, w, h, o);
                        WebPEncodeKernels.SerialTokenize = false;
                        byte[] parallel = await enc.EncodeAsync(rgba, w, h, o);
                        string tag = $"{name} q{q} exact {exact}";
                        if (serial.Length != parallel.Length) { failures.Add($"{tag}: parallel {parallel.Length} B, serial {serial.Length} B"); continue; }
                        int first = -1;
                        for (int i = 0; i < serial.Length && first < 0; i++) if (serial[i] != parallel[i]) first = i;
                        if (first >= 0) failures.Add($"{tag}: byte {first} of {serial.Length} differs");
                    }
            True(failures.Count == 0, $"{failures.Count} cases differ:\n" + string.Join("\n", failures.Take(20)));
        }
        finally
        {
            WebPEncodeKernels.SerialTokenize = false;
            ReleaseAccelerator(ctx, acc);
        }
    }
}
