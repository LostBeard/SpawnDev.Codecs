// Vp9KeyframeEncoderGpu's parallel entropy (Vp9KeyframeSymbolsKernel + Vp8BoolEncoderParallelGpu) must code the tile
// the serial walker (Vp9FrameEntropyKernel) codes, byte for byte: every frame-edge shape (bottom 16x8, right 8x16, the
// corner 8x8, partial superblocks), fine quantizers (long Cat6 tokens, the most symbols per leaf), coarse ones (long zero
// runs, empty transforms) and production sizes (1920x1080). The pictures are made on the accelerator and the frames are
// compared there.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    // A textured plane: smooth waves + fine detail + a hash (hard to predict, many coefficients).
    // checker = 1: 8x8 squares alternate between that texture and flat mid-gray, so neighbouring 8x8 transforms
    // alternate between coded and empty (the bottom / right edge leaves' two halves then differ).
    private static void Vp9ParallelEntropyPictureKernel(Index1D i, ArrayView<byte> plane, int width, int seed, int checker)
    {
        int x = i % width, y = i / width;
        if (checker != 0 & (((x >> 3) + (y >> 3)) & 1) == 0)
        {
            plane[i] = 128;
            return;
        }
        uint h = (uint)i * 0x9E3779B1u ^ (uint)seed * 0x85EBCA77u;
        h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12;
        int wave = ((x * 5 + seed * 13) & 63) - 32 + (((y * 3 + x) >> 2) & 31);
        int detail = (((x * 7) ^ (y * 13) ^ (seed * 31)) & 31) - 16;
        int noise = (int)(h & 15u) - 8;
        int v = 128 + wave + detail + noise;
        plane[i] = (byte)(v < 0 ? 0 : v > 255 ? 255 : v);
    }

    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL,GiantShader")]
    public async Task Vp9KeyframeEncoderGpu_ParallelEntropy_MatchesSerialWalker()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, AtomicKernels);
        try
        {
            var picture = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, int, int, int>(Vp9ParallelEntropyPictureKernel);
            using var enc = new Vp9KeyframeEncoderGpu(acc);
            var failures = new List<string>();
            // (width, height, quantizer, checkerboard picture)
            var cases = new (int W, int H, int Q, int Checker)[]
            {
                (64, 64, 40, 0), (70, 50, 1, 0), (70, 50, 40, 0), (72, 64, 120, 0), (200, 9, 40, 0), (8, 8, 1, 0), (130, 130, 255, 0),
                (70, 50, 120, 1), (72, 64, 120, 1), (130, 130, 200, 1), (200, 9, 160, 1), (1912, 1080, 120, 1),
                (512, 512, 1, 0), (512, 512, 40, 0), (1920, 1080, 30, 0),
            };
            int seed = 0;
            foreach (var (w, h, q, checker) in cases)
            {
                seed++;
                int cw = (w + 1) / 2, ch = (h + 1) / 2;
                using var y = acc.Allocate1D<byte>((long)w * h);
                using var u = acc.Allocate1D<byte>((long)cw * ch);
                using var v = acc.Allocate1D<byte>((long)cw * ch);
                picture(w * h, y.View, w, seed, checker);
                picture(cw * ch, u.View, cw, seed + 1000, checker);
                picture(cw * ch, v.View, cw, seed + 2000, checker);
                int cap = Vp9KeyframeEncoderGpu.GetMaxEncodedSize(w, h);
                using var serialOut = acc.Allocate1D<byte>(cap);
                using var parallelOut = acc.Allocate1D<byte>(cap);
                enc.SerialEntropy = true;
                int serialLen = await enc.EncodeKeyFrameAsync(y.View, u.View, v.View, w, h, serialOut.View, q);
                enc.SerialEntropy = false;
                int parallelLen = await enc.EncodeKeyFrameAsync(y.View, u.View, v.View, w, h, parallelOut.View, q);
                string name = $"{w}x{h} q{q}{(checker != 0 ? " checkerboard" : "")}";
                Console.WriteLine($"[report] VP9 key frame {name}: serial {serialLen} bytes, parallel {parallelLen} bytes");
                if (serialLen != parallelLen) { failures.Add($"{name}: parallel {parallelLen} bytes, serial {serialLen}"); continue; }
                string? diff = await DeviceVerify.CompareAsync(acc, parallelOut.View, serialOut.View, serialLen);
                if (diff != null) failures.Add($"{name}: {diff}");
            }
            if (failures.Count > 0) throw new Exception(string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
