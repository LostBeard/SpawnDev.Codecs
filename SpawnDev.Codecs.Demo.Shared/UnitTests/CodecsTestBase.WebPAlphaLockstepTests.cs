// The ALPH plane's exact filter trials run in lockstep on WebGPU in a browser (WebPLosslessEncoderGpu.LockstepAlphaTrials:
// four trials on four encoders, started together, so their host round trips overlap). Each trial makes its own choices, so
// the chosen filter and the payload must be those of the one-trial-after-another loop. The planes: a blended animation
// frame's mask (flat blocks made transparent), a soft gradient with a hard edge, noise, and a mostly opaque plane.
// The test runs both paths (LockstepAlphaTrialsOverride) and checks the default: lockstep on WebGPU only (2026-11-22:
// WebGPU 28-59% faster with it, Wasm within 1.5%: no host round trip to overlap there).

using ILGPU.Runtime;
using SpawnDev.Codecs.Images.WebP;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task WebPAlphaLockstep_SameBytesAsOneTrialAtATime()
    {
        // the lockstep is the browser path (one thread runs every continuation); the desktop keeps the sequential loop
        if (!OperatingSystem.IsBrowser()) throw new UnsupportedTestException("the ALPH trial lockstep runs in a browser only");
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            using var enc = new WebPLosslessEncoderGpu(acc);
            Equal(acc.AcceleratorType == AcceleratorType.WebGPU, enc.LockstepAlphaTrials, $"lockstep by default on {acc.AcceleratorType}");
            // Timed (a warm encode before each timed one) only on WebGPU, where the timing decides the default. Elsewhere the
            // bytes are checked without the warm encodes: Wasm's timing is settled (within 1.5%, twice, 11-22) and the warm
            // encodes doubled a test that ran 649 s there, past its limit (a 512x352 exact alpha encode is ~92 s on Wasm).
            bool timed = acc.AcceleratorType == AcceleratorType.WebGPU;
            var failures = new List<string>();
            foreach (var (name, w, h, plane) in AlphaLockstepPlanes())
            {
                var o = new WebPAlphaEncodeOptions { Quality = 100, Filter = WebPAlphaFilter.Auto, ExactFilterChoice = true };
                byte[] lockstep, serial;
                double lockMs, serialMs;
                try
                {
                    WebPLosslessEncoderGpu.LockstepAlphaTrialsOverride = true;
                    if (timed) await enc.EncodeAlphaAsync(plane, w, h, o);   // warm (the trial encoders and their kernels)
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    lockstep = await enc.EncodeAlphaAsync(plane, w, h, o);
                    lockMs = sw.Elapsed.TotalMilliseconds;
                    WebPLosslessEncoderGpu.LockstepAlphaTrialsOverride = false;
                    if (timed) await enc.EncodeAlphaAsync(plane, w, h, o);
                    sw.Restart();
                    serial = await enc.EncodeAlphaAsync(plane, w, h, o);
                    serialMs = sw.Elapsed.TotalMilliseconds;
                }
                finally { WebPLosslessEncoderGpu.LockstepAlphaTrialsOverride = null; }   // back to the default (WebGPU in a browser)
                if (!lockstep.AsSpan().SequenceEqual(serial))
                    failures.Add($"{name}: lockstep {lockstep.Length} B (filter {(lockstep[0] >> 2) & 3}) vs one at a time {serial.Length} B " +
                        $"(filter {(serial[0] >> 2) & 3}), first difference at byte {FirstDifference(lockstep, serial)}");
                else Console.WriteLine($"[report] {acc.AcceleratorType} {name}: {serial.Length} B, filter {(serial[0] >> 2) & 3}, same bytes; " +
                    $"lockstep {lockMs:F0} ms, one trial at a time {serialMs:F0} ms{(timed ? "" : " (untimed lane: the first includes the trial encoders' creation)")}");
            }
            True(failures.Count == 0, string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    private static IEnumerable<(string Name, int W, int H, byte[] Plane)> AlphaLockstepPlanes()
    {
        const int w = 512, h = 352;
        var mask = new byte[w * h];
        var rng = new Random(0xA1FA);
        for (int by = 0; by < h / 8; by++)
            for (int bx = 0; bx < w / 8; bx++)
            {
                byte v = rng.Next(4) == 0 ? (byte)255 : (byte)0;   // mostly transparent blocks, like a blended frame
                for (int y = 0; y < 8; y++) for (int x = 0; x < 8; x++) mask[(by * 8 + y) * w + bx * 8 + x] = v;
            }
        yield return ("blended-frame mask 512x352", w, h, mask);
        var grad = new byte[300 * 200];
        for (int y = 0; y < 200; y++) for (int x = 0; x < 300; x++) grad[y * 300 + x] = x < 150 ? (byte)(x + y / 3) : (byte)255;
        yield return ("gradient + hard edge 300x200", 300, 200, grad);
        var noise = new byte[97 * 61];
        rng.NextBytes(noise);
        yield return ("noise 97x61", 97, 61, noise);
        var opaque = new byte[640 * 64];
        Array.Fill(opaque, (byte)255);
        for (int i = 0; i < 40; i++) opaque[rng.Next(opaque.Length)] = (byte)rng.Next(256);
        yield return ("mostly opaque 640x64", 640, 64, opaque);
    }
}
