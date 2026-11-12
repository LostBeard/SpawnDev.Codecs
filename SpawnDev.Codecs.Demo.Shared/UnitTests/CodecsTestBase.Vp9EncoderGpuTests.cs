// Vp9EncoderGpu (key frame + inter frames, Plans/PLAN-VP9-Encoder-Inter-Frames.md): every packet of a clip is decoded
// by Vp9DecoderGpu (bit-exact with libvpx on the conformance vectors) and the decoder's picture must equal the
// encoder's own reconstruction of that frame, byte for byte: the encoder predicts the next frame from it, so any
// difference is drift. Then the fidelity against the source, per frame (no decay over the clip).
//
// Sizes cover every edge leaf (bottom 16x8: 80x50; right 8x16: 72x64; both and the corner 8x8: 70x50, 200x9).
// Motion fields (until the motion search exists the test supplies them): zero; one MV for every block (sub-pixel);
// a field that mixes zero, repeated, opposite and far MVs (so NEARESTMV, NEARMV, ZEROMV and NEWMV are all coded, and
// MVs point outside the picture); zero motion over a clip whose left half is flat mid-gray in every frame: the key frame
// reconstructs it exactly (DC prediction of 128), so the inter frames' blocks there have no coefficients and are coded
// as skipped (the red-check "skip flag never set" needs them: the first clip had none and that mutation passed).
// The same with the flat area as a checkerboard of 16x16 blocks: skipped blocks between coded ones, above and to the
// left (the red-check "skipped block leaves its entropy contexts" needs a coded block after a skipped one in the same
// column: with the flat half it passed).
// Last, a clip that is flat mid-gray everywhere: every block of its inter frames is skipped, so they are a few bytes.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.Codecs.Video.Vp9.Decoder;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>libvpx int_mv: row in the low 16 bits, col in the high (1/8 pixel).</summary>
    private static int PackIntMv(int row, int col) => (row & 0xFFFF) | (col << 16);

    /// <summary>Frame <paramref name="n"/> of a test clip: a gradient that drifts, a square that moves, a still noisy band.</summary>
    private static (byte[] Y, byte[] U, byte[] V) Vp9TestClipFrame(int w, int h, int n, int flat = 0)
    {
        int cw = (w + 1) / 2, ch = (h + 1) / 2;
        var y = new byte[w * h]; var u = new byte[cw * ch]; var v = new byte[cw * ch];
        var still = new Random(11);
        for (int r = 0; r < h; r++)
            for (int c = 0; c < w; c++)
            {
                int value = 40 + ((c + n * 2) * 150 / w + r * 40 / h) % 190;
                int noise = still.Next(0, 24);
                if (r >= h / 2 && c < w / 2) value = 90 + noise;                                  // still, noisy
                if (c >= 8 + n * 3 && c < 24 + n * 3 && r >= 4 + n && r < 20 + n) value = 225;   // the moving square
                if (flat == 1 && c < w / 2) value = 128;
                if (flat == 2 && (((c >> 4) + (r >> 4)) & 1) == 0) value = 128;
                y[r * w + c] = (byte)value;
            }
        for (int r = 0; r < ch; r++)
            for (int c = 0; c < cw; c++)
            {
                u[r * cw + c] = (byte)(100 + (c + n) % 40);
                v[r * cw + c] = (byte)(150 - (r + n) % 30);
                if ((flat == 1 && c < (w / 2 + 1) / 2) || (flat == 2 && (((c >> 3) + (r >> 3)) & 1) == 0)) { u[r * cw + c] = 128; v[r * cw + c] = 128; }
            }
        return (y, u, v);
    }

    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL,GiantShader")]
    public async Task Vp9EncoderGpu_InterFrames_DecoderMatchesEncoderReconstruction()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            var failures = new List<string>();
            var report = new List<string>();
            const int frames = 5;
            foreach (var (w, h) in new[] { (64, 64), (144, 72), (80, 50), (72, 64), (70, 50), (200, 9) })
                foreach (int field in new[] { 0, 1, 2, 3, 4 })
                    foreach (var (q, floor) in new[] { (1, 50.0), (30, 38.0) })
                    {
                        string tag = $"{w}x{h} field {field} q{q}";
                        int cols = (w + 15) / 16, rows = (h + 15) / 16;
                        int[]? mvs = null;
                        if (field == 1 || field == 2)
                        {
                            mvs = new int[cols * rows];
                            // field 1: every block (-0.5, +0.75) pixel; field 2: a mix by position
                            int[] mix = { 0, PackIntMv(-4, 6), PackIntMv(-4, 6), PackIntMv(4, -6), PackIntMv(250, -380), 0, PackIntMv(-130, 2), PackIntMv(-4, 6) };
                            for (int i = 0; i < mvs.Length; i++) mvs[i] = field == 1 ? PackIntMv(-4, 6) : mix[(i * 5 + i / cols * 3) % mix.Length];
                        }
                        using var enc = new Vp9EncoderGpu(acc) { DebugMvs = mvs };
                        using var dec = new Vp9DecoderGpu(acc);
                        int keyBytes = 0, interBytes = 0;
                        double worst = double.MaxValue;
                        for (int n = 0; n < frames; n++)
                        {
                            var (y, u, v) = Vp9TestClipFrame(w, h, n, flat: field == 3 ? 1 : field == 4 ? 2 : 0);
                            var (bytes, isKey) = await enc.EncodeFrameAsync(y, u, v, w, h, q);
                            if (isKey != (n == 0)) { failures.Add($"{tag} frame {n}: key = {isKey}"); break; }
                            if (isKey) keyBytes = bytes.Length; else interBytes += bytes.Length;
                            var recon = await enc.ReadReconAsync();
                            Vp9VideoFrame? shown;
                            try { shown = await dec.DecodePacketAsync(bytes); }
                            catch (Exception ex) { failures.Add($"{tag} frame {n}: the decoder threw {ex.GetType().Name}: {ex.Message}"); break; }
                            if (shown == null) { failures.Add($"{tag} frame {n}: the decoder shows no picture"); break; }
                            int diff = -1;
                            if (shown.Planes.Length != recon.Length) diff = Math.Min(shown.Planes.Length, recon.Length);
                            else for (int i = 0; i < recon.Length && diff < 0; i++) if (recon[i] != shown.Planes[i]) diff = i;
                            if (diff >= 0)
                            {
                                int count = 0;
                                for (int i = 0; i < Math.Min(shown.Planes.Length, recon.Length); i++) if (recon[i] != shown.Planes[i]) count++;
                                failures.Add($"{tag} frame {n}: decoder != encoder reconstruction, first at byte {diff} of {recon.Length} " +
                                    $"(lengths {shown.Planes.Length} / {recon.Length}, {count} bytes differ)");
                                break;
                            }
                            int cw = (w + 1) / 2, ch = (h + 1) / 2;
                            double Psnr(byte[] src, int at, int len) { double se = 0; for (int i = 0; i < len; i++) { double d = src[i] - recon[at + i]; se += d * d; } return 10 * Math.Log10(65025 / Math.Max(se / len, 1e-9)); }
                            double p = Math.Min(Psnr(y, 0, w * h), Math.Min(Psnr(u, w * h, cw * ch), Psnr(v, w * h + cw * ch, cw * ch)));
                            worst = Math.Min(worst, p);
                            if (p < floor) { failures.Add($"{tag} frame {n}: {p:F1} dB < {floor}"); break; }
                        }
                        report.Add($"{tag}: key {keyBytes} B, inter avg {interBytes / (frames - 1)} B, worst {worst:F1} dB");
                    }
            // all flat mid-gray: nothing but skipped blocks in the inter frames
            {
                const int w = 144, h = 72;
                var y = new byte[w * h]; var u = new byte[(w / 2) * (h / 2)]; var v = new byte[u.Length];
                Array.Fill(y, (byte)128); Array.Fill(u, (byte)128); Array.Fill(v, (byte)128);
                using var enc = new Vp9EncoderGpu(acc);
                using var dec = new Vp9DecoderGpu(acc);
                for (int n = 0; n < 3; n++)
                {
                    var (bytes, isKey) = await enc.EncodeFrameAsync(y, u, v, w, h, 30);
                    var shown = await dec.DecodePacketAsync(bytes);
                    var recon = await enc.ReadReconAsync();
                    if (shown == null || !shown.Planes.AsSpan().SequenceEqual(recon)) failures.Add($"flat {w}x{h} frame {n}: decoder != encoder reconstruction");
                    if (!isKey && bytes.Length > 40) failures.Add($"flat {w}x{h} frame {n}: {bytes.Length} bytes for an all-skipped inter frame (expected at most 40)");
                    report.Add($"flat frame {n}: {bytes.Length} B");
                }
            }
            Console.WriteLine($"{acc.AcceleratorType}: " + string.Join(" | ", report));
            True(failures.Count == 0, $"{failures.Count} failures:\n" + string.Join("\n", failures.Take(12)));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
