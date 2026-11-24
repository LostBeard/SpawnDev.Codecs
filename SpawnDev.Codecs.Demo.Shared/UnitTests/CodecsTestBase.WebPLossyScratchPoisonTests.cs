// The lossy WebP encoder must not read device memory it did not write in the same encode. On the RX 590 in Chrome WebGPU
// (Lore, 2026-11-23) every encode of the same picture was a different file, a 1x1 picture included (one macroblock, one
// work-group), while NVIDIA wrote the same file every time: a read of leftover data is one way to get that, because
// leftovers can differ between GPUs and runs. This test fills every working buffer with a different byte before each
// encode (WebPLossyEncoderGpu.PoisonScratch) and requires the same file each time, on every lane.

using SpawnDev.Codecs.Images.WebP.Lossy;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Timeout = 600000, Category = "NotOnWebGL")]
    public async Task WebPLossyEncoderGpu_GarbageInWorkingBuffers_SameFile()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, AtomicKernels);
        try
        {
            using var enc = new WebPLossyEncoderGpu(acc);
            foreach (var (w, h, transparent) in new[] { (1, 1, false), (40, 24, false), (128, 96, true) })
            {
                var rgba = WebPTokenizeStressPicture(4, w, h);
                for (int i = 3; i < rgba.Length; i += 4) rgba[i] = transparent ? (byte)(i * 7) : (byte)255;
                var options = new WebPLossyEncodeOptions { Quality = 90, Method = 4 };
                byte[]? first = null;
                string firstName = "";
                foreach (byte poison in new byte[] { 0x00, 0xFF, 0x5A, 0xA5 })
                {
                    WebPLossyEncoderGpu.PoisonScratch = poison;
                    byte[] file;
                    try { file = await enc.EncodeAsync(rgba, w, h, options); }
                    finally { WebPLossyEncoderGpu.PoisonScratch = null; }
                    if (first == null) { first = file; firstName = $"0x{poison:X2}"; continue; }
                    True(file.AsSpan().SequenceEqual(first),
                        $"{w}x{h}{(transparent ? " with alpha" : "")}: the file over garbage 0x{poison:X2} differs from the one over {firstName} ({file.Length} vs {first.Length} B)");
                }
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
