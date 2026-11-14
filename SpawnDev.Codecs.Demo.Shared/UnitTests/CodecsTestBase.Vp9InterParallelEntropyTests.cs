// Vp9EncoderGpu's parallel entropy for inter frames (Vp9InterSymbols mode pass + symbol pass + Vp8BoolEncoderParallelGpu)
// must code what the serial walker (Vp9InterEntropy.StepKernel) codes: the same bytes for every frame and the same
// symbol counts (backward adaptation reads them, so a wrong count changes the NEXT frame's probabilities: the frames
// after the first inter frame check that too). Every motion field of Vp9EncoderGpu_InterFrames_* (zero, one MV, the
// NEAREST / NEAR / ZERO / NEW mix with MVs outside the picture, skipped halves, a skipped checkerboard, the motion
// search), every edge-leaf size, fine and coarse quantizers, and a production size.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL,GiantShader")]
    public async Task Vp9EncoderGpu_ParallelEntropy_MatchesSerialWalker()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, AtomicKernels);
        try
        {
            var failures = new List<string>();
            const int frames = 5;
            var cases = new List<(int W, int H, int Field, int Q)>();
            foreach (var (w, h) in new[] { (64, 64), (144, 72), (80, 50), (72, 64), (70, 50), (200, 9) })
                foreach (int field in new[] { 0, 1, 2, 3, 4, 5 })
                    foreach (int q in new[] { 1, 30, 120 })
                        cases.Add((w, h, field, q));
            cases.Add((640, 360, 5, 30));
            cases.Add((1280, 720, 5, 20));
            foreach (var (w, h, field, q) in cases)
            {
                string tag = $"{w}x{h} field {field} q{q}";
                int cols = (w + 15) / 16, rows = (h + 15) / 16;
                int[]? mvs = field == 5 ? null : new int[cols * rows];
                if (field == 1 || field == 2)
                {
                    int[] mix = { 0, PackIntMv(-4, 6), PackIntMv(-4, 6), PackIntMv(4, -6), PackIntMv(250, -380), 0, PackIntMv(-130, 2), PackIntMv(-4, 6) };
                    for (int i = 0; i < mvs!.Length; i++) mvs[i] = field == 1 ? PackIntMv(-4, 6) : mix[(i * 5 + i / cols * 3) % mix.Length];
                }
                using var serial = new Vp9EncoderGpu(acc) { DebugMvs = mvs, SerialEntropy = true };
                using var parallel = new Vp9EncoderGpu(acc) { DebugMvs = mvs, SerialEntropy = false };
                long total = 0;
                for (int n = 0; n < frames; n++)
                {
                    var (y, u, v) = Vp9TestClipFrame(w, h, n, flat: field == 3 ? 1 : field == 4 ? 2 : 0);
                    var (a, aKey) = await serial.EncodeFrameAsync(y, u, v, w, h, q);
                    var (b, bKey) = await parallel.EncodeFrameAsync(y, u, v, w, h, q);
                    total += b.Length;
                    if (aKey != bKey) { failures.Add($"{tag} frame {n}: key {bKey}, serial {aKey}"); break; }
                    if (a.Length != b.Length) { failures.Add($"{tag} frame {n} ({(aKey ? "key" : "inter")}): {b.Length} bytes, serial {a.Length}"); break; }
                    int first = -1;
                    for (int i = 0; i < a.Length && first < 0; i++) if (a[i] != b[i]) first = i;
                    if (first >= 0) { failures.Add($"{tag} frame {n} ({(aKey ? "key" : "inter")}): byte {first} of {a.Length} differs"); break; }
                    if (!aKey)
                    {
                        int[] ca = await serial.ReadCountsAsync(), cb = await parallel.ReadCountsAsync();
                        int at = -1;
                        for (int i = 0; i < ca.Length && at < 0; i++) if (ca[i] != cb[i]) at = i;
                        if (at >= 0) { failures.Add($"{tag} frame {n}: symbol count {at} = {cb[at]}, serial {ca[at]}"); break; }
                    }
                }
                if (w >= 640) Console.WriteLine($"[report] VP9 parallel entropy {tag}: {frames} frames, {total} bytes, identical to the serial walker: {failures.Count == 0}");
            }
            if (failures.Count > 0) throw new Exception($"{failures.Count} of {cases.Count} cases differ:\n" + string.Join("\n", failures.Take(20)));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
