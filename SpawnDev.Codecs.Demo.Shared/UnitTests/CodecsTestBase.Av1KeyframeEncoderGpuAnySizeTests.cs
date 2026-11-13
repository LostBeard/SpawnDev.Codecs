// Av1KeyframeEncoderGpu at picture sizes that are not multiples of 64 (16, 8, or neither).
//
// What the encoder does at the bottom / right edge: the AV1 edge partition syntax (split_or_horz, split_or_vert,
// implied split), 8x8 leaves (luma TX_8X8, chroma TX_4X4) where a 16x16 crosses the edge, source pixels past the
// picture repeating its last column / row, reconstruction planes of the coded size (rounded up to 8).
//
// Oracles, and what each can fail on:
//  - MatchesCpuReference (every backend that runs serial stream kernels): the temporal unit must equal the CPU
//    Av1KeyframeEncoder's, byte for byte. Catches a kernel that differs from the reference on a backend (codegen),
//    but not a reading of the AV1 specification that both share.
//  - FfmpegDecodesWhatTheEncoderReconstructs (desktop): FFmpeg (libdav1d), run as a black-box executable, decodes
//    the temporal unit; its picture must equal the encoder's reconstruction cropped to the picture, byte for byte.
//    This is the check an independent decoder gives; it fails on any syntax or context error at the edge (a wrong
//    partition symbol desynchronizes the range decoder for the rest of the tile).
//  - Batch: frames of one batch at an odd size equal their single-frame encodes (per-frame recon slots have the
//    coded size, not the source size).
// Sizes: 80x48 and 16x16 (multiples of 16: edge syntax at 64 and 32 only), 72x40 and 8x8 (multiples of 8: 8x8
// leaves), 70x50, 200x9, 17x33, 130x66, 1x1 (neither: source repetition; 200x9 has one row of 8x8 leaves plus a
// forced split at every level; 1x1 is a single 8x8 leaf under three implied splits).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static readonly (int w, int h)[] Av1EdgeSizes =
    {
        (80, 48), (16, 16), (72, 40), (8, 8), (70, 50), (200, 9), (17, 33), (130, 66), (1, 1),
    };

    /// <summary>A picture with gradients, an oscillation and fine detail, so every transform size codes real coefficients.</summary>
    private static (byte[] y, byte[] u, byte[] v) Av1EdgeTestPicture(int w, int h, int seed)
    {
        int cw = (w + 1) / 2, ch = (h + 1) / 2;
        var y = new byte[w * h]; var u = new byte[cw * ch]; var v = new byte[cw * ch];
        for (int r = 0; r < h; r++)
            for (int c = 0; c < w; c++)
                y[r * w + c] = Av1EdgeTexel(c, r, seed);
        for (int r = 0; r < ch; r++)
            for (int c = 0; c < cw; c++)
            {
                u[r * cw + c] = Av1EdgeTexel(2 * c + 301, 2 * r + 17, seed + 1);
                v[r * cw + c] = Av1EdgeTexel(2 * c + 77, 2 * r + 409, seed + 2);
            }
        return (y, u, v);
    }

    private static byte Av1EdgeTexel(int x, int y, int seed)
    {
        int wave = (int)(52 * Math.Sin(x * 0.19 + seed) + 44 * Math.Cos(y * 0.27 - seed * 0.5));
        int detail = (((x * 7) ^ (y * 13) ^ (seed * 31)) & 31) - 16;
        int v = 128 + wave + detail + ((x + 2 * y) & 63) - 32;
        return (byte)(v < 0 ? 0 : v > 255 ? 255 : v);
    }

    /// <summary>
    /// The CPU reference encoder itself against FFmpeg (no accelerator): every size decodes without an error and the
    /// decoded picture is the source to within the quantizer's error. The reference keeps its reconstruction private,
    /// so the byte-exact decoder comparison is made on the GPU encoder, whose temporal unit equals this one.
    /// </summary>
    [TestMethod(Timeout = 600000, Category = "CpuOnly,DesktopOnly")]
    public Task Av1KeyframeEncoder_AnySize_FfmpegDecodesThePicture()
    {
        CpuOnlyTest();
        FfmpegOracle.RequireAvailable();
        var failures = new List<string>();
        var report = new List<string>();
        int seed = 300;
        var sizes = new List<(int w, int h)> { (64, 64) };
        sizes.AddRange(Av1EdgeSizes);
        foreach (var (w, h) in sizes)
        {
            seed++;
            const int q = 24;
            var (y, u, v) = Av1EdgeTestPicture(w, h, seed);
            byte[] unit = Av1KeyframeEncoder.EncodeKeyFrame(y, w, u, (w + 1) / 2, v, w, h, q);
            List<byte[]> decoded;
            try { decoded = FfmpegOracle.DecodeVideoI420(unit, ".obu", w, h); }
            catch (Exception e) { failures.Add($"{w}x{h}: ffmpeg could not decode the {unit.Length}-byte temporal unit: {e.Message}"); continue; }
            if (decoded.Count != 1) { failures.Add($"{w}x{h}: ffmpeg decoded {decoded.Count} frames, expected 1"); continue; }
            long sad = 0;
            for (int i = 0; i < w * h; i++) sad += Math.Abs(decoded[0][i] - y[i]);
            double mae = (double)sad / (w * h);
            int cw = (w + 1) / 2, ch = (h + 1) / 2;
            long sadC = 0;
            for (int i = 0; i < cw * ch; i++) sadC += Math.Abs(decoded[0][w * h + i] - u[i]) + Math.Abs(decoded[0][w * h + cw * ch + i] - v[i]);
            double maeC = (double)sadC / (2 * cw * ch);
            if (mae > 6.0 || maeC > 6.0) failures.Add($"{w}x{h}: decoded picture is {mae:F2} (luma) / {maeC:F2} (chroma) grey levels from the source on average");
            report.Add($"{w}x{h}: {unit.Length} B, MAE {mae:F2} / {maeC:F2}");
        }
        Console.WriteLine("AV1 CPU reference keyframes decoded by ffmpeg at q24: " + string.Join(" | ", report));
        True(failures.Count == 0, string.Join("\n", failures));
        return Task.CompletedTask;
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task Av1KeyframeEncoderGpu_AnySize_MatchesCpuReference()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new Av1KeyframeEncoderGpu(acc);
            const int baseQIndex = 48;
            var failures = new List<string>();
            int seed = 0;
            foreach (var (w, h) in Av1EdgeSizes)
            {
                seed++;
                var (y, u, v) = Av1EdgeTestPicture(w, h, seed);
                byte[] cpu = Av1KeyframeEncoder.EncodeKeyFrame(y, w, u, (w + 1) / 2, v, w, h, baseQIndex);
                byte[] gpu = await enc.EncodeKeyFrameAsync(y, u, v, w, h, baseQIndex);
                if (gpu.AsSpan().SequenceEqual(cpu)) continue;
                int first = 0, lim = Math.Min(gpu.Length, cpu.Length);
                while (first < lim && gpu[first] == cpu[first]) first++;
                failures.Add($"{w}x{h}: {gpu.Length} bytes vs CPU reference {cpu.Length}, first difference at byte {first}");
            }
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL,DesktopOnly")]
    public async Task Av1KeyframeEncoderGpu_AnySize_FfmpegDecodesWhatTheEncoderReconstructs()
    {
        FfmpegOracle.RequireAvailable();
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new Av1KeyframeEncoderGpu(acc);
            var failures = new List<string>();
            var report = new List<string>();
            int seed = 100;
            // 64x64 and 128x64: the sizes the encoder always took (the decoder check itself must hold there first);
            // 832x480: a real video size whose height is not a multiple of 64 (7.5 superblock rows).
            var sizes = new List<(int w, int h)> { (64, 64), (128, 64) };
            sizes.AddRange(Av1EdgeSizes);
            sizes.Add((832, 480));
            foreach (var (w, h) in sizes)
                foreach (int q in new[] { 24, 120 })
                {
                    if (w * h > 100000 && q != 24) continue;   // the large picture once
                    seed++;
                    string tag = $"{w}x{h} q{q}";
                    var (y, u, v) = Av1EdgeTestPicture(w, h, seed);
                    byte[] unit = await enc.EncodeKeyFrameAsync(y, u, v, w, h, q);
                    byte[] recon = await enc.ReadReconAsync();
                    List<byte[]> decoded;
                    try { decoded = FfmpegOracle.DecodeVideoI420(unit, ".obu", w, h); }
                    catch (Exception e) { failures.Add($"{tag}: ffmpeg could not decode the {unit.Length}-byte temporal unit: {e.Message}"); continue; }
                    if (decoded.Count != 1) { failures.Add($"{tag}: ffmpeg decoded {decoded.Count} frames, expected 1"); continue; }
                    int firstDiff = -1, count = 0;
                    for (int i = 0; i < Math.Min(recon.Length, decoded[0].Length); i++)
                        if (recon[i] != decoded[0][i]) { if (firstDiff < 0) firstDiff = i; count++; }
                    if (firstDiff >= 0 || recon.Length != decoded[0].Length)
                        failures.Add($"{tag}: ffmpeg != encoder reconstruction, first differing byte {firstDiff} of {recon.Length}, " +
                            $"{count} bytes differ (lengths {decoded[0].Length} / {recon.Length})");
                    // The reconstruction must also be a picture of the source, not only self-consistent: mean absolute
                    // luma error against the source (a reconstruction of the wrong pixels would be far off).
                    long sad = 0;
                    for (int i = 0; i < w * h; i++) sad += Math.Abs(recon[i] - y[i]);
                    double mae = (double)sad / (w * h);
                    if (mae > (q == 24 ? 6.0 : 24.0)) failures.Add($"{tag}: reconstruction is {mae:F2} grey levels from the source on average");
                    report.Add($"{tag}: {unit.Length} B, MAE {mae:F2}");
                }
            Console.WriteLine($"{acc.AcceleratorType}: AV1 keyframes decoded by ffmpeg: " + string.Join(" | ", report));
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task Av1KeyframeEncoderGpu_AnySize_BatchEqualsSingleFrameEncodes()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new Av1KeyframeEncoderGpu(acc);
            const int frameCount = 3, width = 70, height = 50, baseQIndex = 40;
            var ys = new byte[frameCount][]; var us = new byte[frameCount][]; var vs = new byte[frameCount][];
            var single = new byte[frameCount][];
            for (int f = 0; f < frameCount; f++)
            {
                (ys[f], us[f], vs[f]) = Av1EdgeTestPicture(width, height, 200 + f);
                single[f] = await enc.EncodeKeyFrameAsync(ys[f], us[f], vs[f], width, height, baseQIndex);
                byte[] cpu = Av1KeyframeEncoder.EncodeKeyFrame(ys[f], width, us[f], (width + 1) / 2, vs[f], width, height, baseQIndex);
                EqualBytes(cpu, single[f], $"frame {f}: single-frame GPU encode vs CPU reference");
            }
            var batch = await enc.EncodeKeyFramesBatchAsync(
                System.Array.ConvertAll(ys, p => (ReadOnlyMemory<byte>)p),
                System.Array.ConvertAll(us, p => (ReadOnlyMemory<byte>)p),
                System.Array.ConvertAll(vs, p => (ReadOnlyMemory<byte>)p),
                width, height, baseQIndex);
            Equal(frameCount, batch.Length, "batch frame count");
            for (int f = 0; f < frameCount; f++)
                EqualBytes(single[f], batch[f], $"frame {f}: batch vs single-frame encode at {width}x{height}");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
