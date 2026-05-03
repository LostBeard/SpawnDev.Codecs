// Minimal ILGPU repro for the suspected `int / 2` codegen issue.
//
// Hypothesis: at least one ILGPU backend lowers signed `int / 2` as
// arithmetic-shift-right (which floors toward -infinity for odd
// negative values) instead of C#-style truncate-toward-zero.
//
// Discovered while wiring Vp9ForwardDct8x8Gpu's saturation regression
// test on 2026-04-28 - the post-pass `output[i] /= 2` produced
// off-by-one results on CUDA for odd negative inputs (-32639 / 2 ->
// CPU = -16319, GPU = -16320).
//
// This file isolates the divide-by-2 in the simplest possible kernel
// (one input, one output, `output[i] = input[i] / 2`) so the failure
// can be confirmed independent of the FDCT butterfly math. If the
// repro fails on CUDA but passes on CPU, the codegen path is at
// fault and the fix belongs in SpawnDev.ILGPU per Rule 2 (Fix
// Libraries First).
//
// Test inputs span:
//   - odd negatives: -1, -3, -5, ..., -32639
//   - odd positives: 1, 3, 5, ..., 32767
//   - even values (positive + negative): both methods agree, sanity check
//   - boundary cases: int.MinValue, int.MaxValue
// CPU oracle is the .NET runtime's own `int / 2` (the canonical
// truncate-toward-zero semantic).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public async Task IlgpuIntDivByTwoRepro_OddNegatives_ShouldTruncateTowardZero()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // Build an input array of every odd negative int from -1 to -32767.
            int n = 16384;
            var input = new int[n];
            for (int i = 0; i < n; i++) input[i] = -(2 * i + 1); // -1, -3, -5, ...

            using var dIn = acc.Allocate1D<int>(n);
            using var dOut = acc.Allocate1D<int>(n);
            dIn.View.CopyFromCPU(input);

            var kernel = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(DivByTwoKernel);
            kernel((Index1D)n, dIn.View, dOut.View);
            await acc.SynchronizeAsync();
            var gpu = await dOut.CopyToHostAsync();

            int firstFailure = -1;
            for (int i = 0; i < n; i++)
            {
                int expected = input[i] / 2; // .NET signed int division - truncate toward zero.
                if (gpu[i] != expected)
                {
                    firstFailure = i;
                    throw new Exception(
                        $"int / 2 codegen mismatch at i={i}: input={input[i]} " +
                        $"expected (truncate) = {expected}, gpu = {gpu[i]} " +
                        $"(diff = {gpu[i] - expected}, looks like floor/shr behavior)");
                }
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task IlgpuIntDivByTwoRepro_OddPositives_AgreeAllBackends()
    {
        // Positive odd values - shr 1 and / 2 always agree, this is the
        // sanity-check half. If even this fails, something more
        // fundamental is broken than just the negative path.
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            int n = 16384;
            var input = new int[n];
            for (int i = 0; i < n; i++) input[i] = 2 * i + 1; // 1, 3, 5, ...

            using var dIn = acc.Allocate1D<int>(n);
            using var dOut = acc.Allocate1D<int>(n);
            dIn.View.CopyFromCPU(input);

            var kernel = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(DivByTwoKernel);
            kernel((Index1D)n, dIn.View, dOut.View);
            await acc.SynchronizeAsync();
            var gpu = await dOut.CopyToHostAsync();

            for (int i = 0; i < n; i++)
            {
                int expected = input[i] / 2;
                if (gpu[i] != expected)
                    throw new Exception(
                        $"int / 2 codegen mismatch (positive!) at i={i}: input={input[i]} " +
                        $"expected = {expected}, gpu = {gpu[i]}");
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task IlgpuIntDivByTwoRepro_BoundaryAndEvenValues_AllBackends()
    {
        // Int.MinValue / 2, Int.MaxValue / 2, even positives / negatives
        // (where shr and / agree). All-or-nothing pin on the canonical
        // semantics.
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            int[] input =
            {
                int.MaxValue, int.MinValue,
                int.MaxValue - 1, int.MinValue + 1,
                0, 1, -1, 2, -2, 4, -4,
                100, -100, 1024, -1024,
                -16384, -32768,
                32767, -32767,
            };
            int n = input.Length;

            using var dIn = acc.Allocate1D<int>(n);
            using var dOut = acc.Allocate1D<int>(n);
            dIn.View.CopyFromCPU(input);

            var kernel = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(DivByTwoKernel);
            kernel((Index1D)n, dIn.View, dOut.View);
            await acc.SynchronizeAsync();
            var gpu = await dOut.CopyToHostAsync();

            for (int i = 0; i < n; i++)
            {
                int expected = input[i] / 2;
                if (gpu[i] != expected)
                    throw new Exception(
                        $"int / 2 codegen mismatch at boundary i={i}: input={input[i]} " +
                        $"expected = {expected}, gpu = {gpu[i]} (diff = {gpu[i] - expected})");
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    private static void DivByTwoKernel(Index1D idx, ArrayView<int> input, ArrayView<int> output)
    {
        int i = idx;
        if (i >= input.Length) return;
        // The canonical C# `int / 2`. ILGPU codegen MUST preserve
        // truncate-toward-zero semantics here (matches .NET CLR / IL
        // div instruction / C99 signed integer division). Lowering
        // this to arithmetic-shift-right is a bug for negative
        // operands.
        output[i] = input[i] / 2;
    }
}
