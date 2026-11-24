// Probe (2026-11-24): do two lossy WebP encodes started together finish sooner than one after the other? An animation frame
// past kmin is encoded twice (sub-frame and key-frame candidate; WebPAnimEncoder SetFrameAsync), one after the other, and
// each encode is latency-bound serial chains (a few macroblocks at a time, the GPU mostly idle: the RX 590 spends 8-10x the
// 4070's time there). If two encodes overlap on the GPU, encoding a frame's two candidates side by side would cut that
// frame's time; if the GPU runs one queue strictly in order, it would not. Two encoder instances (own buffers), the same
// 512x352 picture: the wall time of two encodes in sequence, then of the same two started together; bytes checked.

using System.Diagnostics;
using SpawnDev.Codecs.Images.WebP.Lossy;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task Probe_WebPLossy_TwoEncodesTogether()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, AtomicKernels);
        try
        {
            const int w = 512, h = 352;
            var rgba = WebPTokenizeStressPicture(4, w, h);
            for (int i = 3; i < rgba.Length; i += 4) rgba[i] = 255;
            var options = new WebPLossyEncodeOptions { Quality = 90, Method = 4 };
            using var a = new WebPLossyEncoderGpu(acc);
            using var b = new WebPLossyEncoderGpu(acc);
            byte[] fileA = await a.EncodeAsync(rgba, w, h, options);   // warm
            byte[] fileB = await b.EncodeAsync(rgba, w, h, options);
            True(fileA.AsSpan().SequenceEqual(fileB), "the two encoder instances wrote different files");
            const int runs = 4;
            var seq = Stopwatch.StartNew();
            for (int r = 0; r < runs; r++)
            {
                byte[] x = await a.EncodeAsync(rgba, w, h, options);
                byte[] y = await b.EncodeAsync(rgba, w, h, options);
                True(x.AsSpan().SequenceEqual(fileA) && y.AsSpan().SequenceEqual(fileA), "sequential encode differs");
            }
            double seqMs = seq.Elapsed.TotalMilliseconds / runs;
            var tog = Stopwatch.StartNew();
            for (int r = 0; r < runs; r++)
            {
                var ta = a.EncodeAsync(rgba, w, h, options);
                var tb = b.EncodeAsync(rgba, w, h, options);
                byte[] x = await ta, y = await tb;
                True(x.AsSpan().SequenceEqual(fileA) && y.AsSpan().SequenceEqual(fileA), "concurrent encode differs");
            }
            double togMs = tog.Elapsed.TotalMilliseconds / runs;
            Console.WriteLine($"[report] {acc.AcceleratorType} ({acc.Name}) two 512x352 lossy encodes: one after the other {seqMs:F0} ms, together {togMs:F0} ms ({seqMs / togMs:F2}x)");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
