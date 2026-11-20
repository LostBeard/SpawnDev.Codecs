// The lossy encoder's RD loop runs as a wavefront inside each probability window (WebPLossyEncodeRowKernel.Stages.cs):
// every stage launches once per diagonal d = x + 2y, all of the diagonal's macroblocks at once. These tests pin that the
// concurrency changes nothing: the same bytes as launching the macroblocks one at a time in the same dependency order,
// at sizes whose probability windows start inside a row (640x480: windows of 150 / 151 macroblocks over rows of 40 -
// the case that skipped macroblocks when the diagonal loop started at the window's first macroblock), and with row token
// buffers far too small at first (the grow-and-retry path), and with every probability's window statistics replayed in
// raster order (the path of a counter that saturates inside a window, rare in real pictures). Byte identity with libwebp itself is
// WebPLossyEncoderGpu_Methods3And4_ByteIdenticalWithLibwebp's (its fixtures include 640x480).

using SpawnDev.Codecs.Images.WebP.Lossy;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task WebPLossyWavefront_ParallelDiagonals_SameBytesAsOneAtATime()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new WebPLossyEncoderGpu(acc);
            foreach (var (w, h, q) in new[] { (640, 480, 90f), (333, 211, 75f), (16, 400, 80f) })
            {
                var rgba = WebPTokenizeStressPicture(4, w, h);
                for (int i = 3; i < rgba.Length; i += 4) rgba[i] = 255;
                var o = new WebPLossyEncodeOptions { Quality = q, Method = 4 };
                byte[] parallel, serial, smallRows, replayed;
                try
                {
                    parallel = await enc.EncodeAsync(rgba, w, h, o);
                    WebPLossyEncoderGpu.SerialWavefront = true;
                    serial = await enc.EncodeAsync(rgba, w, h, o);
                }
                finally { WebPLossyEncoderGpu.SerialWavefront = false; }
                try
                {
                    WebPLossyEncoderGpu.ReplayAllStatistics = true;
                    replayed = await enc.EncodeAsync(rgba, w, h, o);
                }
                finally { WebPLossyEncoderGpu.ReplayAllStatistics = false; }
                using (var fresh = new WebPLossyEncoderGpu(acc))
                {
                    try
                    {
                        WebPLossyEncoderGpu.RowTokensPerMacroblock = 1;
                        smallRows = await fresh.EncodeAsync(rgba, w, h, o);
                    }
                    finally { WebPLossyEncoderGpu.RowTokensPerMacroblock = 512; }
                }
                Equal(serial.Length + ":" + serial.Length, parallel.Length + ":" + FirstDifference(serial, parallel), $"{w}x{h} q{q}: parallel diagonals vs one macroblock at a time ({parallel.Length} vs {serial.Length} B)");
                Equal(serial.Length + ":" + serial.Length, smallRows.Length + ":" + FirstDifference(serial, smallRows), $"{w}x{h} q{q}: grown row token buffers ({smallRows.Length} vs {serial.Length} B)");
                Equal(serial.Length + ":" + serial.Length, replayed.Length + ":" + FirstDifference(serial, replayed), $"{w}x{h} q{q}: statistics replayed in raster order");
                Console.WriteLine($"[report] {w}x{h} q{q}: {parallel.Length} B, same bytes parallel / serial / grown buffers / replayed statistics");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }


    // The token partition from the parallel big-number kernels (WebPLossyTokenPartitionKernels) against the one-thread
    // VP8EmitTokens walk: the same file, from tiny to token-heavy pictures (q100 noise: long runs, many carries).
    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task WebPLossyTokenPartition_Parallel_SameBytesAsTheWalk()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new WebPLossyEncoderGpu(acc);
            foreach (var (w, h, q, pic) in new[] { (1, 1, 75f, 4), (17, 9, 50f, 4), (333, 211, 75f, 4), (256, 256, 100f, 1), (640, 480, 90f, 4), (64, 64, 0f, 0) })
            {
                var rgba = WebPTokenizeStressPicture(pic, w, h);
                for (int i = 3; i < rgba.Length; i += 4) rgba[i] = 255;
                var o = new WebPLossyEncodeOptions { Quality = q, Method = 4 };
                byte[] parallel = await enc.EncodeAsync(rgba, w, h, o), walk;
                try
                {
                    WebPLossyEncoderGpu.SerialTokenPartition = true;
                    walk = await enc.EncodeAsync(rgba, w, h, o);
                }
                finally { WebPLossyEncoderGpu.SerialTokenPartition = false; }
                Equal(walk.Length + ":" + walk.Length, parallel.Length + ":" + FirstDifference(walk, parallel), $"{w}x{h} q{q} picture {pic}: parallel token partition vs the walk");
                Console.WriteLine($"[report] {w}x{h} q{q} picture {pic}: {parallel.Length} B, same bytes");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
