// Vp9EncoderGpu (key frame + inter frames, Plans/PLAN-VP9-Encoder-Inter-Frames.md): every packet of a clip is decoded
// by Vp9DecoderGpu (bit-exact with libvpx on the conformance vectors) and the decoder's picture must equal the
// encoder's own reconstruction of that frame, byte for byte: the encoder predicts the next frame from it, so any
// difference is drift. Then the fidelity against the source, per frame (no decay over the clip).
//
// Sizes cover every edge leaf (bottom 16x8: 80x50; right 8x16: 72x64; both and the corner 8x8: 70x50, 200x9).
// Motion fields: fields 0-4 are supplied by the test (they pin the syntax whatever the search finds), field 5 is the
// encoder's motion search. Zero; one MV for every block (sub-pixel);
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
                foreach (int field in new[] { 0, 1, 2, 3, 4, 5 })
                    foreach (var (q, floor) in new[] { (1, 50.0), (30, 38.0) })
                    {
                        string tag = $"{w}x{h} field {field} q{q}";
                        int cols = (w + 15) / 16, rows = (h + 15) / 16;
                        int[]? mvs = field == 5 ? null : new int[cols * rows];   // null: the motion search
                        if (field == 1 || field == 2)
                        {
                            // field 1: every block (-0.5, +0.75) pixel; field 2: a mix by position
                            int[] mix = { 0, PackIntMv(-4, 6), PackIntMv(-4, 6), PackIntMv(4, -6), PackIntMv(250, -380), 0, PackIntMv(-130, 2), PackIntMv(-4, 6) };
                            for (int i = 0; i < mvs!.Length; i++) mvs[i] = field == 1 ? PackIntMv(-4, 6) : mix[(i * 5 + i / cols * 3) % mix.Length];
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
                        // A picture that is not a multiple of 8 has leaves reaching into padding. Nothing may be coded
                        // there: measured inter / key bytes 36% (80x50), 33% (70x50), 23% (200x9); with the zero
                        // padded source coded against the prediction 66%, 72% and 101% (every frame, the same step).
                        if (field == 5 && q == 30 && (w % 8 != 0 || h % 8 != 0) && interBytes / (frames - 1) * 2 > keyBytes)
                            failures.Add($"{tag}: inter frames average {interBytes / (frames - 1)} B, the key frame {keyBytes} B: more than half (is the padding coded?)");
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

    /// <summary>A textured picture (smooth waves + coarse grain) sampled at (x, y): the same picture at any offset.</summary>
    private static byte Vp9TestTexture(int x, int y)
    {
        double v = 128 + 55 * Math.Sin(x * 0.21) * Math.Cos(y * 0.17) + 30 * Math.Sin((x + y) * 0.05);
        uint g = (uint)((x >> 2) * 73856093) ^ (uint)((y >> 2) * 19349663);
        g ^= g >> 13; g *= 0x5bd1e995; g ^= g >> 15;
        return (byte)Math.Clamp(v + (g & 31) - 16, 0, 255);
    }

    /// <summary>
    /// The motion search earns its keep: a picture that moves 3 pixels right and 2 down per frame. With the search the
    /// inter frames must be much smaller than with zero motion, at the same quantizer and no worse a picture; the
    /// decoder must still match the encoder's reconstruction on every frame.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL,GiantShader")]
    public async Task Vp9EncoderGpu_MotionSearch_ShrinksMovingPicture()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            var failures = new List<string>();
            var report = new List<string>();
            const int frames = 6, q = 40;
            foreach (var (w, h) in new[] { (192, 128), (176, 100) })
            {
                int cw = (w + 1) / 2, ch = (h + 1) / 2;
                var bytesPer = new int[2];
                var worstPer = new double[2];
                for (int pass = 0; pass < 2; pass++)
                {
                    using var enc = new Vp9EncoderGpu(acc) { MotionSearch = pass == 1 };
                    using var dec = new Vp9DecoderGpu(acc);
                    worstPer[pass] = double.MaxValue;
                    for (int n = 0; n < frames; n++)
                    {
                        var y = new byte[w * h]; var u = new byte[cw * ch]; var v = new byte[cw * ch];
                        for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) y[r * w + c] = Vp9TestTexture(c - 3 * n + 64, r - 2 * n + 64);
                        for (int r = 0; r < ch; r++)
                            for (int c = 0; c < cw; c++)
                            {
                                u[r * cw + c] = (byte)(64 + Vp9TestTexture(2 * c - 3 * n + 300, 2 * r - 2 * n + 300) / 2);
                                v[r * cw + c] = (byte)(64 + Vp9TestTexture(2 * c - 3 * n + 600, 2 * r - 2 * n + 600) / 2);
                            }
                        var (bytes, isKey) = await enc.EncodeFrameAsync(y, u, v, w, h, q);
                        var recon = await enc.ReadReconAsync();
                        var shown = await dec.DecodePacketAsync(bytes);
                        if (shown == null || !shown.Planes.AsSpan().SequenceEqual(recon))
                        {
                            failures.Add($"{w}x{h} search {pass} frame {n}: decoder != encoder reconstruction");
                            break;
                        }
                        if (!isKey) bytesPer[pass] += bytes.Length;
                        double se = 0;
                        for (int i = 0; i < y.Length; i++) { double d = y[i] - recon[i]; se += d * d; }
                        worstPer[pass] = Math.Min(worstPer[pass], 10 * Math.Log10(65025 / Math.Max(se / y.Length, 1e-9)));
                    }
                }
                report.Add($"{w}x{h} q{q}: inter bytes zero motion {bytesPer[0]}, search {bytesPer[1]} ({100.0 * bytesPer[1] / Math.Max(bytesPer[0], 1):F0}%), " +
                    $"worst Y {worstPer[0]:F1} / {worstPer[1]:F1} dB");
                if (bytesPer[1] * 2 > bytesPer[0]) failures.Add($"{w}x{h}: the search's inter frames are {bytesPer[1]} bytes, zero motion's {bytesPer[0]}: expected under half");
                if (worstPer[1] < worstPer[0] - 0.5) failures.Add($"{w}x{h}: the search's picture is worse: {worstPer[1]:F1} dB vs {worstPer[0]:F1} dB");
            }
            Console.WriteLine($"{acc.AcceleratorType}: " + string.Join(" | ", report));
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    /// <summary>
    /// The inter frames' dead zone (Vp9EncoderGpu.InterZbinQ7 / InterRoundQ7, default 100 / 48) against round to nearest
    /// (64 / 64) at the same quantizer: a moving picture with fresh noise in every frame, which round to nearest
    /// re-codes frame after frame. The default must be clearly smaller at nearly the same picture. (The setting itself
    /// was chosen on real clips at equal PSNR: DemoConsole vp9-encode-bench.)
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL,GiantShader")]
    public async Task Vp9EncoderGpu_InterDeadZone_SmallerThanRoundToNearest()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            const int w = 192, h = 128, frames = 5, q = 40;
            int cw = (w + 1) / 2, ch = (h + 1) / 2;
            var bytesPer = new int[2];
            var worstPer = new double[2];
            for (int pass = 0; pass < 2; pass++)
            {
                using var enc = new Vp9EncoderGpu(acc);
                if (pass == 1) { enc.InterZbinQ7 = 64; enc.InterRoundQ7 = 64; }   // round to nearest
                using var dec = new Vp9DecoderGpu(acc);
                worstPer[pass] = double.MaxValue;
                for (int n = 0; n < frames; n++)
                {
                    var y = new byte[w * h]; var u = new byte[cw * ch]; var v = new byte[cw * ch];
                    var grain = new Random(2000 + n);
                    for (int r = 0; r < h; r++)
                        for (int c = 0; c < w; c++)
                            y[r * w + c] = (byte)Math.Clamp(Math.Round(Vp9SmoothTexture(c - 3 * n, r - 2 * n, 1)) + grain.Next(-4, 5), 0, 255);
                    for (int r = 0; r < ch; r++)
                        for (int c = 0; c < cw; c++)
                        {
                            u[r * cw + c] = (byte)Math.Clamp(Math.Round(64 + Vp9SmoothTexture(2 * c - 3 * n, 2 * r - 2 * n, 11) / 2) + grain.Next(-2, 3), 0, 255);
                            v[r * cw + c] = (byte)Math.Clamp(Math.Round(64 + Vp9SmoothTexture(2 * c - 3 * n, 2 * r - 2 * n, 21) / 2) + grain.Next(-2, 3), 0, 255);
                        }
                    var (bytes, isKey) = await enc.EncodeFrameAsync(y, u, v, w, h, q);
                    var recon = await enc.ReadReconAsync();
                    var shown = await dec.DecodePacketAsync(bytes);
                    True(shown != null && shown.Planes.AsSpan().SequenceEqual(recon), $"pass {pass} frame {n}: decoder != encoder reconstruction");
                    if (isKey) continue;
                    bytesPer[pass] += bytes.Length;
                    double se = 0;
                    for (int i = 0; i < y.Length; i++) { double d = y[i] - recon[i]; se += d * d; }
                    worstPer[pass] = Math.Min(worstPer[pass], 10 * Math.Log10(65025 / Math.Max(se / y.Length, 1e-9)));
                }
            }
            Console.WriteLine($"{acc.AcceleratorType}: {w}x{h} q{q} inter bytes dead zone {bytesPer[0]}, round to nearest {bytesPer[1]} " +
                $"({100.0 * bytesPer[0] / Math.Max(bytesPer[1], 1):F0}%), worst Y {worstPer[0]:F2} / {worstPer[1]:F2} dB");
            True(bytesPer[0] * 100 <= bytesPer[1] * DeadZoneMaxPercent, $"dead zone {bytesPer[0]} B, round to nearest {bytesPer[1]} B: expected at most {DeadZoneMaxPercent}%");
            True(worstPer[0] >= worstPer[1] - DeadZoneMaxLossDb, $"dead zone worst Y {worstPer[0]:F2} dB, round to nearest {worstPer[1]:F2} dB: more than {DeadZoneMaxLossDb} dB lower");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    // Measured on the CPU accelerator (2026-11-12): 18140 B against 29332 B (62%), worst Y 41.28 against 43.41 dB (the
    // noise it declines to code counts as error here: the source is the noisy picture).
    private const int DeadZoneMaxPercent = 75;
    private const double DeadZoneMaxLossDb = 3.0;

    /// <summary>
    /// Backward adaptation (Vp9EncoderGpu.BackwardAdaptation, the default) against error resilient frames (every
    /// frame coded with the default probabilities) on a clip long enough for the probabilities to settle: a moving,
    /// noisy picture, 12 frames. Both streams decode to the encoder's reconstruction on every frame (a wrong count or
    /// probability anywhere desynchronizes the decoder within a frame); the inter frames' first byte says which
    /// kind they are (error_resilient_mode is its lowest bit); the two reconstructions are the same pictures (the
    /// entropy coding does not change them, only the MV candidates can change which mode codes an MV); and the
    /// adapting stream is smaller.
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL,GiantShader")]
    public async Task Vp9EncoderGpu_BackwardAdaptation_SmallerAndDecodes()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            var failures = new List<string>();
            var report = new List<string>();
            const int frames = 12, q = 40;
            foreach (var (w, h) in new[] { (192, 128), (70, 50) })
            {
                int cw = (w + 1) / 2, ch = (h + 1) / 2;
                var bytesPer = new int[2];
                var recons = new List<byte[]>[] { new(), new() };
                for (int pass = 0; pass < 2; pass++)
                {
                    using var enc = new Vp9EncoderGpu(acc) { BackwardAdaptation = pass == 0 };
                    using var dec = new Vp9DecoderGpu(acc);
                    for (int n = 0; n < frames; n++)
                    {
                        var y = new byte[w * h]; var u = new byte[cw * ch]; var v = new byte[cw * ch];
                        var grain = new Random(3000 + n);
                        for (int r = 0; r < h; r++)
                            for (int c = 0; c < w; c++)
                                y[r * w + c] = (byte)Math.Clamp(Math.Round(Vp9SmoothTexture(c - 2.5 * n, r + 1.25 * n, 1)) + grain.Next(-4, 5), 0, 255);
                        for (int r = 0; r < ch; r++)
                            for (int c = 0; c < cw; c++)
                            {
                                u[r * cw + c] = (byte)Math.Clamp(Math.Round(64 + Vp9SmoothTexture(2 * c - 2.5 * n, 2 * r + 1.25 * n, 11) / 2) + grain.Next(-2, 3), 0, 255);
                                v[r * cw + c] = (byte)Math.Clamp(Math.Round(64 + Vp9SmoothTexture(2 * c - 2.5 * n, 2 * r + 1.25 * n, 21) / 2) + grain.Next(-2, 3), 0, 255);
                            }
                        var (bytes, isKey) = await enc.EncodeFrameAsync(y, u, v, w, h, q);
                        var recon = await enc.ReadReconAsync();
                        recons[pass].Add(recon);
                        Vp9VideoFrame? shown;
                        try { shown = await dec.DecodePacketAsync(bytes); }
                        catch (Exception ex) { failures.Add($"{w}x{h} pass {pass} frame {n}: the decoder threw {ex.GetType().Name}: {ex.Message}"); break; }
                        if (shown == null || !shown.Planes.AsSpan().SequenceEqual(recon))
                        {
                            failures.Add($"{w}x{h} pass {pass} frame {n}: decoder != encoder reconstruction");
                            break;
                        }
                        // The decoder's own state against the encoder's, after every frame: the four saved contexts
                        // (the fifth slot is the frame's working copy: after a key frame a decoder has adapted it and,
                        // refresh_frame_context being 0, thrown it away) and, for inter frames, every symbol count. A
                        // count nobody's probabilities depend on yet (the corner block's implied SPLIT) shows only here.
                        var encCtx = await enc.ReadContextsAsync();
                        var decCtx = await dec.Pipeline.Contexts.ReadContextsAsync();
                        int ctxAt = -1;
                        for (int i = 0; i < 4 * Vp9Ctx.Bytes && ctxAt < 0; i++) if (encCtx[i] != decCtx[i]) ctxAt = i;
                        if (encCtx.Length != decCtx.Length || ctxAt >= 0)
                        {
                            failures.Add($"{w}x{h} pass {pass} frame {n}: frame contexts differ at slot {ctxAt / Vp9Ctx.Bytes} offset {ctxAt % Vp9Ctx.Bytes}: " +
                                $"encoder {(ctxAt >= 0 ? encCtx[ctxAt] : -1)}, decoder {(ctxAt >= 0 ? decCtx[ctxAt] : -1)}");
                            break;
                        }
                        if (isKey) continue;
                        var encCounts = await enc.ReadCountsAsync();
                        var decCounts = await dec.Pipeline.Parser.ReadCountsAsync();
                        int cntAt = -1;
                        for (int i = 0; i < encCounts.Length && cntAt < 0; i++) if (encCounts[i] != decCounts[i]) cntAt = i;
                        if (cntAt >= 0)
                        {
                            failures.Add($"{w}x{h} pass {pass} frame {n}: symbol counts differ at {cntAt}: encoder {encCounts[cntAt]}, decoder {decCounts[cntAt]} " +
                                $"(Vp9Cnt: Partition {Vp9Cnt.Partition}, Coef {Vp9Cnt.Coef}, EobBranch {Vp9Cnt.EobBranch}, InterMode {Vp9Cnt.InterMode}, Skip {Vp9Cnt.Skip}, MvJoints {Vp9Cnt.MvJoints})");
                            break;
                        }
                        bytesPer[pass] += bytes.Length;
                        bool errorResilient = (bytes[0] & 1) != 0;
                        if (errorResilient != (pass == 1)) failures.Add($"{w}x{h} pass {pass} frame {n}: error_resilient_mode = {errorResilient}");
                    }
                }
                report.Add($"{w}x{h} q{q}: inter bytes adapting {bytesPer[0]}, error resilient {bytesPer[1]} ({100.0 * bytesPer[0] / Math.Max(bytesPer[1], 1):F1}%)");
                int same = 0;
                for (int n = 0; n < Math.Min(recons[0].Count, recons[1].Count); n++) if (recons[0][n].AsSpan().SequenceEqual(recons[1][n])) same++;
                report.Add($"{same}/{frames} reconstructions identical");
                if (same != frames) failures.Add($"{w}x{h}: only {same} of {frames} reconstructions are the same with and without adaptation");
                if (bytesPer[0] * 100 > bytesPer[1] * AdaptationMaxPercent)
                    failures.Add($"{w}x{h}: adapting {bytesPer[0]} B, error resilient {bytesPer[1]} B: expected at most {AdaptationMaxPercent}%");
            }
            Console.WriteLine($"{acc.AcceleratorType}: " + string.Join(" | ", report));
            True(failures.Count == 0, $"{failures.Count} failures:\n" + string.Join("\n", failures.Take(12)));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    // Measured on the CPU accelerator (2026-11-12): 88.7% (192x128) and 90.2% (70x50) of the error resilient bytes.
    private const int AdaptationMaxPercent = 94;

    private static double Vp9Lattice(int x, int y, uint seed)
    {
        uint g = (uint)(x * 73856093) ^ (uint)(y * 19349663) ^ (seed * 83492791u);
        g ^= g >> 13; g *= 0x5bd1e995; g ^= g >> 15; g *= 0x27d4eb2d; g ^= g >> 16;
        return (g & 0xFFFF) / 65535.0;
    }

    /// <summary>Smoothly interpolated random lattice with <paramref name="cell"/> pixels between its points: not periodic.</summary>
    private static double Vp9ValueNoise(double x, double y, double cell, uint seed)
    {
        double fx = x / cell, fy = y / cell;
        int x0 = (int)Math.Floor(fx), y0 = (int)Math.Floor(fy);
        double tx = fx - x0, ty = fy - y0;
        tx = tx * tx * (3 - 2 * tx);
        ty = ty * ty * (3 - 2 * ty);
        double a = Vp9Lattice(x0, y0, seed), b = Vp9Lattice(x0 + 1, y0, seed), c = Vp9Lattice(x0, y0 + 1, seed), d = Vp9Lattice(x0 + 1, y0 + 1, seed);
        return a + (b - a) * tx + (c - a) * ty + (a - b - c + d) * tx * ty;
    }

    /// <summary>A picture defined at every real position (three scales of detail), so it can move by a fraction of a pixel.</summary>
    private static double Vp9SmoothTexture(double x, double y, uint seed)
        => 128 + 90 * (Vp9ValueNoise(x, y, 24, seed) - 0.5) + 70 * (Vp9ValueNoise(x, y, 9, seed + 1) - 0.5) + 40 * (Vp9ValueNoise(x, y, 4, seed + 2) - 0.5);

    /// <summary>
    /// The motion search finds the motion: a picture that moves by a known amount per frame, and the MV the search
    /// leaves for every leaf whose block and reference block are inside the picture must be exactly that motion.
    /// One motion per part of the search: a few whole pixels (full size level), far motions in both directions (only
    /// the 1/4 size level reaches them), half-pel, quarter-pel; and a still picture with different noise in every
    /// frame, where every MV must stay zero (the merge pass: a noise fit is not worth an MV in the stream).
    /// </summary>
    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL,GiantShader")]
    public async Task Vp9EncoderGpu_MotionSearch_FindsTheMotion()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            var failures = new List<string>();
            var report = new List<string>();
            const int frames = 3, q = 20;
            var cases = new (double Mx, double My, int Noise)[] { (3, 2, 0), (-29, 22, 0), (26, -30, 0), (2.5, -1.5, 0), (-0.75, 4.25, 0), (0, 0, 3) };
            foreach (var (w, h) in new[] { (192, 128), (176, 100) })
                foreach (var (mx, my, noise) in cases)
                {
                    string tag = $"{w}x{h} motion ({mx}, {my}){(noise > 0 ? $" noise {noise}" : "")}";
                    int cw = (w + 1) / 2, ch = (h + 1) / 2, cols = (w + 15) / 16, rows = (h + 15) / 16;
                    int expected = PackIntMv((int)Math.Round(-my * 8), (int)Math.Round(-mx * 8));
                    using var enc = new Vp9EncoderGpu(acc);
                    using var still = new Vp9EncoderGpu(acc) { MotionSearch = false };   // the noise case's yardstick: zero motion
                    using var dec = new Vp9DecoderGpu(acc);
                    int interior = 0, exact = 0, interBytes = 0, stillBytes = 0;
                    string firstWrong = "";
                    for (int n = 0; n < frames; n++)
                    {
                        var y = new byte[w * h]; var u = new byte[cw * ch]; var v = new byte[cw * ch];
                        var grain = new Random(1000 + n);
                        for (int r = 0; r < h; r++)
                            for (int c = 0; c < w; c++)
                                y[r * w + c] = (byte)Math.Clamp(Math.Round(Vp9SmoothTexture(c - mx * n, r - my * n, 1)) + grain.Next(-noise, noise + 1), 0, 255);
                        for (int r = 0; r < ch; r++)
                            for (int c = 0; c < cw; c++)
                            {
                                u[r * cw + c] = (byte)Math.Clamp(Math.Round(64 + Vp9SmoothTexture(2 * c - mx * n, 2 * r - my * n, 11) / 2), 0, 255);
                                v[r * cw + c] = (byte)Math.Clamp(Math.Round(64 + Vp9SmoothTexture(2 * c - mx * n, 2 * r - my * n, 21) / 2), 0, 255);
                            }
                        var (bytes, isKey) = await enc.EncodeFrameAsync(y, u, v, w, h, q);
                        var recon = await enc.ReadReconAsync();
                        var shown = await dec.DecodePacketAsync(bytes);
                        if (shown == null || !shown.Planes.AsSpan().SequenceEqual(recon))
                        {
                            failures.Add($"{tag} frame {n}: decoder != encoder reconstruction");
                            break;
                        }
                        if (noise > 0)
                        {
                            var (stillFrame, _) = await still.EncodeFrameAsync(y, u, v, w, h, q);
                            if (!isKey) stillBytes += stillFrame.Length;
                        }
                        if (isKey) continue;
                        interBytes += bytes.Length;
                        var mvs = await enc.ReadMvsAsync();
                        for (int r16 = 0; r16 < rows; r16++)
                            for (int c16 = 0; c16 < cols; c16++)
                            {
                                // the leaf's 16x16 block inside the picture, and the block it is predicted from, with
                                // the interpolation filter's reach (3 pixels before, 4 after), inside the reference
                                double x0 = c16 * 16, y0 = r16 * 16, rx = x0 - mx, ry = y0 - my;
                                if (x0 + 15 > w - 1 || y0 + 15 > h - 1 || rx - 3 < 0 || ry - 3 < 0 || rx + 15 + 4 > w - 1 || ry + 15 + 4 > h - 1) continue;
                                interior++;
                                int mv = mvs[r16 * cols + c16];
                                if (mv == expected) exact++;
                                else if (firstWrong.Length == 0)
                                    firstWrong = $", first wrong: leaf ({c16}, {r16}) frame {n} MV (row {(short)(mv & 0xFFFF)}, col {mv >> 16}) / 8";
                            }
                    }
                    report.Add($"{tag}: {exact}/{interior} leaves exact, inter {interBytes} B{(noise > 0 ? $" (zero motion {stillBytes} B)" : "")}");
                    if (interior < 16) failures.Add($"{tag}: only {interior} leaves inside the picture: the case checks too little");
                    if (noise > 0)
                    {
                        // Sub-pixel prediction smooths the reference's noise, so a noise fit can beat zero motion by
                        // a little: the merge pass charges an MV of the leaf's own (NewMvCost) and all but a few
                        // leaves stay at zero. Measured on the CPU accelerator (2026-11-12): 117/120 and 88/90 leaves
                        // zero, inter frames 0.4% / 0.3% over zero motion's; with the charge removed 89/120 and
                        // 73/90, 2.5% / 1.6% over.
                        if (exact * 100 < interior * 95) failures.Add($"{tag}: {exact} of {interior} leaves stay at zero motion (expected at least 95%){firstWrong}");
                        if (interBytes * 100 > stillBytes * 101) failures.Add($"{tag}: the search's inter frames are {interBytes} B, zero motion's {stillBytes} B: more than 1% over");
                        continue;
                    }
                    if (exact != interior)
                        failures.Add($"{tag}: {exact} of {interior} leaves have the motion (row {(short)(expected & 0xFFFF)}, col {expected >> 16}) / 8{firstWrong}");
                }
            Console.WriteLine($"{acc.AcceleratorType}: " + string.Join(" | ", report));
            True(failures.Count == 0, $"{failures.Count} failures:\n" + string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
