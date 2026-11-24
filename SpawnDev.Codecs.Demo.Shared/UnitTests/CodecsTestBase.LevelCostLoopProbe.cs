// Probe (2026-11-24): the smallest kernel with the shape that broke lossy WebP on an AMD RX 590 in Chrome WebGPU (D3D12).
// VP8CalculateLevelCosts as libwebp writes it - one thread, a loop over the 96 (type, band, ctx) tables, and inside it, per
// level, a data-dependent `while (pattern != 0)` walking the level's bit pattern - left 31 of the 32 (type, band) groups of
// the table unwritten on that GPU (Probe_WebPLossy_StageHashes, Lore). This runs that shape alone (the default
// probabilities, libwebp's tables) and the fixed 9-step walk the encoder now uses, and reports how many of the 6,528
// values each one gets right against the host. It asserts nothing about the while shape (it documents a driver/compiler
// fault, for the upstream report); it requires the fixed shape to be exact.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Images.WebP.Lossy;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;
using S = SpawnDev.Codecs.Images.WebP.Lossy.WebPLossySetupKernel;
using T = SpawnDev.Codecs.Images.WebP.Lossy.WebPLossyEncTables;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private const int LcTables = S.NumTypes * S.NumBands * S.NumCtx;

    private static int LcBitCost(ArrayView<int> t, int bit, int proba) => t[T.EntropyCost + (bit != 0 ? 255 - proba : proba)];

    // libwebp's shape: a while loop that runs as long as the pattern has bits left
    private static void LevelCostWhileKernel(Index1D _, ArrayView<int> coeffs, ArrayView<int> levelCost, ArrayView<int> t)
    {
        for (int task = 0; task < LcTables; ++task)
        {
            int ctx = task % S.NumCtx, band = (task / S.NumCtx) % S.NumBands, ctype = task / (S.NumCtx * S.NumBands);
            int p = S.NumProbas * (ctx + S.NumCtx * (band + S.NumBands * ctype));
            int table = S.LevelCostStride * (ctx + S.NumCtx * (band + S.NumBands * ctype));
            int cost0 = ctx > 0 ? LcBitCost(t, 1, coeffs[p + 0]) : 0;
            int costBase = LcBitCost(t, 1, coeffs[p + 1]) + cost0;
            levelCost[table + 0] = LcBitCost(t, 0, coeffs[p + 1]) + cost0;
            for (int v = 1; v <= S.MaxVariableLevel; ++v)
            {
                int pattern = t[T.LevelCodes + (v - 1) * 2 + 0];
                int bits = t[T.LevelCodes + (v - 1) * 2 + 1];
                int cost = 0;
                int i = 2;
                bool go = pattern != 0;
                while (go)
                {
                    if ((pattern & 1) != 0) cost += LcBitCost(t, bits & 1, coeffs[p + i]);
                    bits >>= 1;
                    pattern >>= 1;
                    ++i;
                    go = pattern != 0;
                }
                levelCost[table + v] = costBase + cost;
            }
        }
    }

    // the encoder's shape since 2026-11-24: a fixed walk over the 9 pattern bits
    private static void LevelCostFixedKernel(Index1D _, ArrayView<int> coeffs, ArrayView<int> levelCost, ArrayView<int> t)
    {
        for (int task = 0; task < LcTables; ++task)
        {
            int ctx = task % S.NumCtx, band = (task / S.NumCtx) % S.NumBands, ctype = task / (S.NumCtx * S.NumBands);
            int p = S.NumProbas * (ctx + S.NumCtx * (band + S.NumBands * ctype));
            int table = S.LevelCostStride * (ctx + S.NumCtx * (band + S.NumBands * ctype));
            int cost0 = ctx > 0 ? LcBitCost(t, 1, coeffs[p + 0]) : 0;
            int costBase = LcBitCost(t, 1, coeffs[p + 1]) + cost0;
            levelCost[table + 0] = LcBitCost(t, 0, coeffs[p + 1]) + cost0;
            for (int v = 1; v <= S.MaxVariableLevel; ++v)
            {
                int pattern = t[T.LevelCodes + (v - 1) * 2 + 0];
                int bits = t[T.LevelCodes + (v - 1) * 2 + 1];
                int cost = 0;
                for (int k = 0; k < 9; ++k)
                    if (((pattern >> k) & 1) != 0) cost += LcBitCost(t, (bits >> k) & 1, coeffs[p + 2 + k]);
                levelCost[table + v] = costBase + cost;
            }
        }
    }

    [TestMethod(Timeout = 300000, Category = "NotOnWebGL")]
    public async Task Probe_LevelCostLoopShapes()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            int[] t = T.Build();
            var coeffs = new int[S.ProbaCount];
            Array.Copy(t, T.CoeffsProba0, coeffs, 0, S.ProbaCount);
            // the host's table (the C# of the while shape, run on the CPU)
            var expect = new int[S.LevelCostCount];
            for (int task = 0; task < LcTables; ++task)
            {
                int c = task % S.NumCtx, band = (task / S.NumCtx) % S.NumBands, ctype = task / (S.NumCtx * S.NumBands);
                int p = S.NumProbas * (c + S.NumCtx * (band + S.NumBands * ctype));
                int table = S.LevelCostStride * (c + S.NumCtx * (band + S.NumBands * ctype));
                int Bc(int bit, int pr) => t[T.EntropyCost + (bit != 0 ? 255 - pr : pr)];
                int cost0 = c > 0 ? Bc(1, coeffs[p]) : 0;
                expect[table] = Bc(0, coeffs[p + 1]) + cost0;
                for (int v = 1; v <= S.MaxVariableLevel; ++v)
                {
                    int pattern = t[T.LevelCodes + (v - 1) * 2], bits = t[T.LevelCodes + (v - 1) * 2 + 1], cost = 0;
                    for (int i = 2; pattern != 0; ++i, bits >>= 1, pattern >>= 1) if ((pattern & 1) != 0) cost += Bc(bits & 1, coeffs[p + i]);
                    expect[table + v] = Bc(1, coeffs[p + 1]) + cost0 + cost;
                }
            }
            using var dT = acc.Allocate1D(t);
            using var dC = acc.Allocate1D(coeffs);
            using var dL = acc.Allocate1D<int>(S.LevelCostCount);
            var results = new List<string>();
            int fixedRight = 0;
            foreach (var (name, k) in new (string, Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>)[]
                { ("while", LevelCostWhileKernel), ("fixed 9", LevelCostFixedKernel) })
            {
                dL.View.MemSetToZero();
                var kernel = await acc.LoadAutoGroupedStreamKernelAsync<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(k);
                kernel(1, dC.View, dL.View, dT.View);
                await acc.SynchronizeAsync();
                var got = await dL.View.CopyToHostAsync();
                int right = 0, firstWrong = -1;
                for (int i = 0; i < got.Length; i++) { if (got[i] == expect[i]) right++; else if (firstWrong < 0) firstWrong = i; }
                if (name == "fixed 9") fixedRight = right;
                results.Add($"{name}: {right} of {got.Length} right" + (firstWrong < 0 ? "" : $", first wrong at {firstWrong} (table {firstWrong / S.LevelCostStride}, level {firstWrong % S.LevelCostStride}: {got[firstWrong]} vs {expect[firstWrong]})"));
            }
            Console.WriteLine($"[report] {acc.AcceleratorType} ({acc.Name}) level-cost table, one thread: " + string.Join("; ", results));
            // the WGSL of the while shape, for a report that does not need SpawnDev (SpawnDev.ILGPU's WGSL is deterministic)
            if (acc is SpawnDev.ILGPU.WebGPU.WebGPUAccelerator)
                foreach (var entry in SpawnDev.ILGPU.WebGPU.Backend.WebGPUBackend.WGSLRegistry.Values)
                    if (entry.KernelName.Contains(nameof(LevelCostWhileKernel)))
                    {
                        // in numbered chunks: a report line is cut at a few thousand characters
                        string b64 = Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes(entry.Source));
                        for (int o = 0, n = 0; o < b64.Length; o += 1000, n++)
                            Console.WriteLine($"[report] WGSL {n:D2} {b64.Substring(o, Math.Min(1000, b64.Length - o))}");
                    }
            Equal(S.LevelCostCount, fixedRight, "the fixed 9-step walk's values right");
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
