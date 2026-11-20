// A measurement, not a check: where a real-sized PNG decode spends its time on this accelerator. SpawnGen's Load Image
// (Lore, 2026-11-20) did not finish a 512x512 PNG within 2 minutes on Chrome WebGPU with an RX 590 and twice tripped the
// driver's watchdog; the desktop decodes the same file in 56 ms (CPU accelerator) / 463 ms (OpenCL) warm. The fixture is a
// real 512x512 RGB render (TestData/PngPhoto, 707,671 bytes). It passes whenever the decode is right; the stage times go
// to the report line.

using System.Diagnostics;
using SpawnDev.Codecs.Images.Png;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task Probe_PngDecode512_StageTimes()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, SerialStreamKernels);
        try
        {
            byte[] file = ReadEmbedded(PngPhotoPrefix + "sdturbo_apple_512_rgb.png");
            var want = ReadPngOnHost(file).Rgba8;
            using var dec = new PngDecoderGpu(acc);
            await dec.WarmUpAsync();
            var sw = Stopwatch.StartNew();
            var first = await dec.DecodeAsync(file);
            double firstMs = sw.Elapsed.TotalMilliseconds;
            AssertPixelsEqual(want, first.Pixels, "first decode");
            sw.Restart();
            await dec.DecodeAsync(file);
            double warmMs = sw.Elapsed.TotalMilliseconds;
            var spent = new List<string>();
            var step = Stopwatch.StartNew();
            try
            {
                PngDecoderGpu.Trace = name => { spent.Add($"{name} {step.Elapsed.TotalMilliseconds:F0}"); step.Restart(); };
                step.Restart();
                AssertPixelsEqual(want, (await dec.DecodeAsync(file)).Pixels, "traced decode");
            }
            finally { PngDecoderGpu.Trace = null; }
            Console.WriteLine($"[report] {acc.AcceleratorType}: PNG 512x512 RGB ({file.Length} B) decode {firstMs:F0} ms first, {warmMs:F0} ms warm; " +
                "traced stages (ms): " + string.Join(", ", spent));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
