// Probe (2026-11-24): where a lossy WebP encode first parts ways between two GPUs. On the RX 590 in Chrome WebGPU the
// encoder writes files that are not libwebp's (Methods3And4: 36 of 222 identical, since at least 1.1.3; Lore), a 1x1
// picture included, while NVIDIA, CUDA, OpenCL and the CPU write libwebp's. With every working buffer zeroed first
// (PoisonScratch 0) and the macroblocks of a diagonal launched one at a time (SerialWavefront), this prints a hash of the
// encoder's buffers (meta, macroblock records, state, work) after every step: analysis + setup, each RD stage launch,
// refresh, window, token partition. Run it on a GPU that is right and on the one that is not: the first line that
// differs names the step that went wrong (the encode is the same input, the same launches, in the same order).

using SpawnDev.Codecs.Images.WebP.Lossy;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task Probe_WebPLossy_StageHashes()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, AtomicKernels);
        try
        {
            using var enc = new WebPLossyEncoderGpu(acc);
            foreach (var (w, h) in new[] { (1, 1), (16, 16), (40, 24) })
            {
                var rgba = WebPTokenizeStressPicture(4, w, h);
                for (int i = 3; i < rgba.Length; i += 4) rgba[i] = 255;
                var options = new WebPLossyEncodeOptions { Quality = 50, Method = 4 };
                var lines = new List<string>();
                int step = 0;
                WebPLossyEncoderGpu.PoisonScratch = 0;
                WebPLossyEncoderGpu.SerialWavefront = true;
                WebPLossyEncoderGpu.StageDump = async (e, name) => lines.Add($"{step++:D3} {name}: {await e.HashWorkingBuffersAsync()}");
                byte[] file;
                try { file = await enc.EncodeAsync(rgba, w, h, options); }
                finally
                {
                    WebPLossyEncoderGpu.StageDump = null;
                    WebPLossyEncoderGpu.SerialWavefront = false;
                    WebPLossyEncoderGpu.PoisonScratch = null;
                }
                ulong x = 14695981039346656037UL;
                foreach (byte b in file) { x ^= b; x *= 1099511628211UL; }
                Console.WriteLine($"[report] {w}x{h} q50 m4: {file.Length} B, file {x:x16}, {lines.Count} steps");
                foreach (var line in lines) Console.WriteLine($"[report] {w}x{h} {line}");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
