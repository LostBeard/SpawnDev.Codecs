// Probe (2026-11-22): where an exact-filter ALPH encode spends its time per lane. On Wasm one 512x352 plane took
// 72-92 s (WebPAlphaLockstep_SameBytesAsOneTrialAtATime), on WebGPU about 1 s. Times the steps WebPLosslessEncoderGpu
// marks with Trace (analyze, each strategy trial, each colour-cache trial, tokenize, emit, file), summed over the four
// filter trials, for one warm encode of the same plane the lockstep test uses first (a blended animation frame's mask).

using System.Diagnostics;
using SpawnDev.Codecs.Images.WebP;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Timeout = 900000, Category = "NotOnWebGL")]
    public async Task Probe_WebPAlphaExact_StageTimes()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            var (name, w, h, plane) = AlphaLockstepPlanes().First();
            using var enc = new WebPLosslessEncoderGpu(acc);
            var o = new WebPAlphaEncodeOptions { Quality = 100, Filter = WebPAlphaFilter.Auto, ExactFilterChoice = true };
            WebPLosslessEncoderGpu.LockstepAlphaTrialsOverride = false;   // one trial at a time: steps do not interleave
            try
            {
                await enc.EncodeAlphaAsync(plane, w, h, o);   // warm
                var spent = new Dictionary<string, double>();
                var counts = new Dictionary<string, int>();
                var step = Stopwatch.StartNew();
                var total = Stopwatch.StartNew();
                WebPLosslessEncoderGpu.Trace = mark =>
                {
                    spent[mark] = spent.GetValueOrDefault(mark) + step.Elapsed.TotalMilliseconds;
                    counts[mark] = counts.GetValueOrDefault(mark) + 1;
                    step.Restart();
                };
                byte[] file;
                try { file = await enc.EncodeAlphaAsync(plane, w, h, o); }
                finally { WebPLosslessEncoderGpu.Trace = null; }
                double totalMs = total.Elapsed.TotalMilliseconds;
                Console.WriteLine($"[report] {acc.AcceleratorType} exact ALPH {name} {w}x{h}: {file.Length} B, total {totalMs:F0} ms; " +
                    "time up to each mark (ms / count): " +
                    string.Join(", ", spent.OrderByDescending(kv => kv.Value).Select(kv => $"{kv.Key} {kv.Value:F0}/{counts[kv.Key]}")));
            }
            finally { WebPLosslessEncoderGpu.LockstepAlphaTrialsOverride = null; }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
