// Canary for an ILGPU codegen bug found through the VP8 loop filter (hev threshold): a nested conditional with constant arms and two
// threshold tests on one variable, `level >= 40 ? 2 : (level >= 15 ? 1 : 0)`, returned 1 for level 40..63 on CUDA and OpenCL
// (CPU correct, it runs the IL). Root cause (Data): the ILGPU frontend typed the merge phi as Int1 from its first predecessor, so
// the constant 2 became `true`; fixed in SpawnDev.ILGPU 5.4.3-local.2 (SSABuilder.GetPhiType). Vp8FullLoopFilterKernel uses the
// nested form again. The if-chain and the sum-of-ternaries forms are checked in the same kernel so a failure names exactly which
// spelling is wrong.
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static void TernaryReproKernel(Index1D i, ArrayView<int> levels, ArrayView<int> outNested, ArrayView<int> outIf, ArrayView<int> outSum)
    {
        int level = levels[i];
        int nested = level >= 40 ? 2 : (level >= 15 ? 1 : 0);
        int viaIf = 0;
        if (level >= 15) viaIf = 1;
        if (level >= 40) viaIf = 2;
        int sum = (level >= 40 ? 1 : 0) + (level >= 15 ? 1 : 0);
        outNested[i] = nested;
        outIf[i] = viaIf;
        outSum[i] = sum;
    }

    [TestMethod]
    public async Task KernelDialect_NestedThresholdTernary_MatchesIfChain()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            var k = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>>(TernaryReproKernel);
            var levels = Enumerable.Range(0, 64).ToArray();
            using var dl = acc.Allocate1D<int>(64);
            using var dn = acc.Allocate1D<int>(64);
            using var di = acc.Allocate1D<int>(64);
            using var ds = acc.Allocate1D<int>(64);
            dl.View.CopyFromCPU(levels);
            k(64, dl.View, dn.View, di.View, ds.View);
            await acc.SynchronizeAsync();
            var n = await dn.View.CopyToHostAsync();
            var f = await di.View.CopyToHostAsync();
            var s = await ds.View.CopyToHostAsync();
            var bad = new List<string>();
            for (int l = 0; l < 64; l++)
            {
                int want = l >= 40 ? 2 : (l >= 15 ? 1 : 0);
                if (n[l] != want) bad.Add($"nested level {l}: got {n[l]}, want {want}");
                if (f[l] != want) bad.Add($"if-chain level {l}: got {f[l]}, want {want}");
                if (s[l] != want) bad.Add($"sum-of-ternaries level {l}: got {s[l]}, want {want}");
            }
            True(bad.Count == 0, $"{bad.Count} wrong:\n" + string.Join("\n", bad.Take(12)));
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
