// Fidelity of Vp9KeyframeEncoderGpu against its SOURCE picture: encode, decode with Vp9DecoderGpu (bit-exact with
// libvpx), compare with what went in. The encoder's other tests compare it with its CPU reference, which shares its
// design: a picture region the walker does not encode correctly (2026-11-12: the right-edge partition, task #23) passed
// them all. Near-lossless quantizer (q 1): every plane far above any lossy level; moderate q: still high.
//
// Also: widths whose 8-aligned value is not a multiple of 16 are rejected (until task #23) rather than encoded wrong.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.Codecs.Video.Vp9.Decoder;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL")]
    public async Task Vp9KeyframeEncoderGpu_DecodedPictureMatchesSource()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            var failures = new List<string>();
            var report = new List<string>();
            foreach (var (name, w, h) in new[] { ("smooth", 64, 64), ("noise", 64, 64), ("smooth", 80, 50), ("noise", 144, 72) })
            {
                int cw = (w + 1) / 2, ch = (h + 1) / 2;
                var y = new byte[w * h]; var u = new byte[cw * ch]; var v = new byte[cw * ch];
                var rng = new Random(3);
                bool noise = name == "noise";
                for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) y[r * w + c] = noise ? (byte)rng.Next(16, 236) : (byte)(40 + c * 150 / w + r * 30 / h);
                for (int i = 0; i < u.Length; i++) { u[i] = noise ? (byte)rng.Next(16, 241) : (byte)(100 + i % 40); v[i] = noise ? (byte)rng.Next(16, 241) : (byte)(150 - i % 30); }
                foreach (var (q, floor) in new[] { (1, 50.0), (30, 40.0) })
                {
                    using var enc = new Vp9KeyframeEncoderGpu(acc);
                    var bytes = await enc.EncodeKeyFrameAsync(y, u, v, w, h, q);
                    using var dec = new Vp9DecoderGpu(acc);
                    var f = await dec.DecodePacketAsync(bytes) ?? throw new Exception("no picture");
                    double Psnr(byte[] src, int at, int n) { double se = 0; for (int i = 0; i < n; i++) { double d = src[i] - f.Planes[at + i]; se += d * d; } return 10 * Math.Log10(65025 / Math.Max(se / n, 1e-9)); }
                    double py = Psnr(y, 0, w * h), pu = Psnr(u, w * h, cw * ch), pv = Psnr(v, w * h + cw * ch, cw * ch);
                    report.Add($"{name} {w}x{h} q{q}: Y {py:F1} U {pu:F1} V {pv:F1}");
                    if (Math.Min(py, Math.Min(pu, pv)) < floor) failures.Add($"{name} {w}x{h} q{q}: Y {py:F1} U {pu:F1} V {pv:F1} dB < {floor}");
                }
            }
            using (var enc = new Vp9KeyframeEncoderGpu(acc))
                await ThrowsAsync<NotSupportedException>(() => enc.EncodeKeyFrameAsync(new byte[70 * 50], new byte[35 * 25], new byte[35 * 25], 70, 50, 30));
            Console.WriteLine($"{acc.AcceleratorType}: " + string.Join(" | ", report));
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
