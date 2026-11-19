// The lossless encoder must write the same file on every device: the same pixels and options give the same bytes on
// CUDA, OpenCL (NVIDIA and AMD), Velocity, the CPU accelerator, WebGPU and Wasm. Its only decision made from real
// numbers is the predictor-mode choice (Vp8lPredictorChoose), which since 2026-11-18 scores in integers; before, float
// costs rounded differently per device (the same 25 frames gave other files on an RTX 4070 and an RX 590 in 10 of 25
// frames). The expected SHA-256 values below were produced by this test; every accelerator must reproduce them.
// Pictures with transparency at Exact = false (the choice runs there), at efforts that use it (75, 95), plus an opaque
// picture (the choice does not run: guards the rest of the encoder) and a short animation (WebPAnimEncoder decisions).
// Also checks the integer cost table on the host against double-precision log2.

using System.Security.Cryptography;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.Codecs.Images.WebP;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    // name -> SHA-256 (hex, first 16 chars) of the encoded file; 2026-11-18, identical on CUDA, OpenCL, Velocity and the
    // CPU accelerator (RTX 4070) when first recorded
    private static readonly Dictionary<string, string> WebPDeviceIndependentHashes = new()
    {
        ["gradient + transparency 512x352 q75"] = "9969B57F349DBE71",
        ["gradient + transparency 512x352 q95"] = "3FAC8816072C6545",
        ["cost-oracle pattern 400x352 q75"] = "98434147BA257E86",
        ["cost-oracle pattern 400x352 q95"] = "E83F24534A6E988C",
        ["texture opaque 320x240 q80"] = "9D61787F306B6CDC",
        ["animation 4 frames 256x176 q80"] = "1880A08C092F91DC",
    };

    private static void BinCostTestKernel(Index1D i, ArrayView<long> table, ArrayView<int> input, ArrayView<long> output)
        => output[i] = Vp8lPredictorChoose.BinCost(table, input[3 * i], input[3 * i + 1], input[3 * i + 2]);

    /// <summary>
    /// The integer cost on the device against the same formula in doubles: every term (tile count x up to 1024, accumulated
    /// count y up to 2^28 so the interpolated log2 is used, every bias bucket of v) within 0.1 bit. The file-hash test only
    /// sees terms that flip a choice; this pins each term (red-checked: interpolation removed, bias / 9).
    /// </summary>
    [TestMethod(Timeout = 300000, Category = "NotOnWebGL")]
    public async Task Vp8lPredictorChoose_IntegerCost_MatchesDoubles()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var cases = new List<(int X, int Y, int V)>();
            int[] xs = { 1, 2, 3, 7, 64, 255, 256, 1000, 1024 };
            int[] ys = { 0, 1, 5, 100, 65535, 65536, 65537, 70000, 123457, 1 << 20, 3_000_001, 1 << 28 };
            int[] vs = { 0, 1, 2, 9, 15, 16, 100, 240, 241, 250, 255 };
            foreach (int x in xs) foreach (int y in ys) foreach (int v in vs) cases.Add((x, y, v));
            var input = new int[cases.Count * 3];
            for (int k = 0; k < cases.Count; k++) { input[3 * k] = cases[k].X; input[3 * k + 1] = cases[k].Y; input[3 * k + 2] = cases[k].V; }
            var kernel = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<long>, ArrayView<int>, ArrayView<long>>(BinCostTestKernel);
            using var dTable = acc.Allocate1D(Vp8lPredictorChoose.CostTable);
            using var dIn = acc.Allocate1D(input);
            using var dOut = acc.Allocate1D<long>(cases.Count);
            kernel(cases.Count, dTable.View, dIn.View, dOut.View);
            await acc.SynchronizeAsync();
            long[] got = await dOut.View.CopyToHostAsync();
            var failures = new List<string>();
            double worst = 0;
            for (int k = 0; k < cases.Count; k++)
            {
                var (x, y, v) = cases[k];
                double lx = Math.Log2(x);
                double step = y > 0 ? x * Math.Log2((double)x + y) + y * (Math.Log2((double)x + y) - Math.Log2(y)) : x * lx;
                int wi = v == 0 ? 0 : v < 16 ? v : v > 240 ? 256 - v : -1;
                double w = wi < 0 ? 0 : wi == 0 ? 1.0 : 0.94 * Math.Pow(0.6, wi - 1);
                double want = -(x * lx + step) - x * w / 10.0;
                double err = Math.Abs(got[k] / 4294967296.0 - want);
                worst = Math.Max(worst, err);
                if (err > 0.1 && failures.Count < 10) failures.Add($"x {x} y {y} v {v}: device {got[k] / 4294967296.0:F6}, double {want:F6}");
            }
            Console.WriteLine($"[report] {cases.Count} cases, worst error {worst:G3} bits");
            True(failures.Count == 0, $"worst error {worst:G3} bits:\n" + string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    private static byte[] ArgbToRgba(uint[] px)
    {
        var rgba = new byte[px.Length * 4];
        for (int i = 0; i < px.Length; i++)
        {
            rgba[i * 4] = (byte)(px[i] >> 16); rgba[i * 4 + 1] = (byte)(px[i] >> 8); rgba[i * 4 + 2] = (byte)px[i]; rgba[i * 4 + 3] = (byte)(px[i] >> 24);
        }
        return rgba;
    }

    [TestMethod(Timeout = 900000, Category = "NotOnWebGL")]
    public async Task WebPLosslessEncoderGpu_SameFileOnEveryDevice()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, AtomicKernels);
        try
        {
            var failures = new List<string>();
            var report = new List<string>();

            // the integer cost table against double precision (host only: it is computed with integers on the host)
            var t = Vp8lPredictorChoose.CostTable;
            for (int v = 1; v <= Vp8lPredictorChoose.LogTableMax; v++)
            {
                double want = Math.Log2(v) * 4294967296.0;
                if (Math.Abs(t[v] - want) > 2.0) { failures.Add($"log table [{v}] = {t[v]}, double {want:F1}"); break; }
                if (v > 1 && t[v] <= t[v - 1]) { failures.Add($"log table not increasing at {v}"); break; }
            }
            for (int i = 0; i < 16; i++)
            {
                double w = i == 0 ? 1.0 : 0.94 * Math.Pow(0.6, i - 1);
                if (Math.Abs(t[Vp8lPredictorChoose.BiasOff + i] - w * 4294967296.0) > 2.0) failures.Add($"bias weight {i} = {t[Vp8lPredictorChoose.BiasOff + i]}, double {w * 4294967296.0:F1}");
            }

            using var enc = new WebPLosslessEncoderGpu(acc);
            async Task Check(string name, Func<Task<byte[]>> encode)
            {
                byte[] file = await encode();
                string hash = Convert.ToHexString(SHA256.HashData(file))[..16];
                report.Add($"{name}: {file.Length} B, {hash}");
                string want = WebPDeviceIndependentHashes[name];
                if (want != hash) failures.Add($"{name}: {hash} ({file.Length} B), expected {(want.Length == 0 ? "(not set)" : want)}");
            }
            var grad = WebPTokenizeStressPicture(3, 512, 352);
            var pattern = ArgbToRgba(ChooseTestPicture(1, 400, 352));
            foreach (int q in new[] { 75, 95 })
            {
                await Check($"gradient + transparency 512x352 q{q}", () => enc.EncodeAsync(grad, 512, 352, new WebPEncodeOptions { Quality = q, Exact = false }));
                await Check($"cost-oracle pattern 400x352 q{q}", () => enc.EncodeAsync(pattern, 400, 352, new WebPEncodeOptions { Quality = q, Exact = false }));
            }
            await Check("texture opaque 320x240 q80", () => enc.EncodeAsync(WebPTokenizeStressPicture(4, 320, 240), 320, 240, new WebPEncodeOptions { Quality = 80 }));
            // an animation whose frames change in places, so WebPAnimEncoder makes sub-frames with transparency
            var frames = new List<ReadOnlyMemory<byte>>();
            for (int f = 0; f < 4; f++)
            {
                var px = ChooseTestPicture(0, 256, 176);
                for (int y = 0; y < 176; y++)
                    for (int x = 0; x < 256; x++)
                        if (((x / 32) + (y / 32) + f) % 3 == 0) px[y * 256 + x] ^= (uint)(f * 0x00101010);
                frames.Add(ArgbToRgba(px));
            }
            await Check("animation 4 frames 256x176 q80", () => enc.EncodeAnimationAsync(frames, 256, 176, new[] { 41, 41, 41, 41 }, new WebPEncodeOptions { Quality = 80, Exact = false }));

            Console.WriteLine("[report] " + string.Join(" | ", report));
            True(failures.Count == 0, $"{failures.Count} failures:\n" + string.Join("\n", failures) + "\n" + string.Join("\n", report));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
