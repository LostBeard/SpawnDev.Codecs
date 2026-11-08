// Minimal ILGPU repro: nested loops whose inner do-while has a short-circuit condition with side effects.
//
// Found 2026-11-08 in Mp3DecodeKernels.Huffman (minimp3's big_values loop): on CUDA the outer counter `ireg` ended
// equal to the number of bands walked (8) instead of the number of regions (2), as if the inner do-while's back
// edge went to the top of the outer loop body (re-running `ireg++` and the region reads). CPU and OpenCL are right.
// The kernel below keeps the loop shape and nothing else; the expected values come from running the same method
// on the host (.NET semantics).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    // MPEG-1 44.1 kHz long-block band widths (minimp3 g_scf_long[0]), 0-terminated.
    private static readonly int[] NestedLoopBands = { 4, 4, 4, 4, 4, 4, 6, 6, 8, 8, 10, 12, 16, 20, 24, 28, 34, 42, 50, 54, 76, 158, 0 };

    // (big_values, region0_count, region1_count) per case; region 2 takes the rest (255 = "to the end").
    private static readonly int[] NestedLoopCases =
    {
        70, 7, 255,   // cbr128_st44 frame 0 granule 0 ch 0 (CPU: ireg 2, sfb 16, bigValCnt -11)
        13, 7, 255,
        76, 8, 255,
        115, 8, 255,
        88, 7, 6,
        69, 9, 4,
        288, 15, 3,
        0, 7, 255,
        1, 0, 0,
        200, 2, 2,
    };

    [TestMethod]
    public async Task IlgpuNestedLoopRepro_InnerDoWhileShortCircuit_MatchesHost()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            int n = NestedLoopCases.Length / 3;
            using var dBands = acc.Allocate1D(NestedLoopBands);
            using var dCases = acc.Allocate1D(NestedLoopCases);
            using var dOut = acc.Allocate1D<int>(n * 4);
            var kernel = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>>(NestedLoopKernel);
            kernel((Index1D)n, dBands.View, dCases.View, dOut.View);
            var gpu = await dOut.CopyToHostAsync();

            var fails = new List<string>();
            for (int c = 0; c < n; c++)
            {
                var (ireg, sfb, left, pairs) = NestedLoopWalk(NestedLoopBands, NestedLoopCases[3 * c], NestedLoopCases[3 * c + 1], NestedLoopCases[3 * c + 2]);
                int[] want = { ireg, sfb, left, pairs };
                for (int k = 0; k < 4; k++)
                    if (gpu[4 * c + k] != want[k])
                    {
                        fails.Add($"case {c} (big {NestedLoopCases[3 * c]}, r0 {NestedLoopCases[3 * c + 1]}, r1 {NestedLoopCases[3 * c + 2]}): " +
                            $"want ireg/sfb/left/pairs = {string.Join(",", want)}, got {gpu[4 * c]},{gpu[4 * c + 1]},{gpu[4 * c + 2]},{gpu[4 * c + 3]}");
                        break;
                    }
            }
            if (fails.Count > 0) throw new Exception($"{acc.AcceleratorType}: {fails.Count}/{n} cases wrong:\n" + string.Join("\n", fails));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }

    private static void NestedLoopKernel(Index1D i, ArrayView<int> bands, ArrayView<int> cases, ArrayView<int> output)
    {
        var (ireg, sfb, left, pairs) = NestedLoopWalk(bands, cases[3 * i], cases[3 * i + 1], cases[3 * i + 2]);
        output[4 * i] = ireg;
        output[4 * i + 1] = sfb;
        output[4 * i + 2] = left;
        output[4 * i + 3] = pairs;
    }

    // Host oracle (.NET semantics): the same loop over a managed array.
    private static (int Ireg, int Sfb, int Left, int Pairs) NestedLoopWalk(int[] bands, int bigValCnt, int region0, int region1)
    {
        int ireg = 0, sfb = 0, total = 0;
        int np, pairs;
        while (bigValCnt > 0 && ireg < 3)
        {
            int sfbCnt = ireg == 0 ? region0 : ireg == 1 ? region1 : 255;
            ireg++;
            do
            {
                np = bands[sfb++] / 2;
                if (np == 0) { bigValCnt = 0; break; }
                pairs = bigValCnt < np ? bigValCnt : np;
                do { total++; } while (--pairs > 0);
            } while ((bigValCnt -= np) > 0 && --sfbCnt >= 0);
        }
        return (ireg, sfb, bigValCnt, total);
    }

    // The shape of Mp3DecodeKernels.Huffman's big_values loop (regions -> bands -> pairs), on the accelerator.
    private static (int Ireg, int Sfb, int Left, int Pairs) NestedLoopWalk(ArrayView<int> bands, int bigValCnt, int region0, int region1)
    {
        int ireg = 0, sfb = 0, total = 0;
        int np, pairs;
        while (bigValCnt > 0 && ireg < 3)
        {
            int sfbCnt = ireg == 0 ? region0 : ireg == 1 ? region1 : 255;
            ireg++;
            do
            {
                np = bands[sfb++] / 2;
                if (np == 0) { bigValCnt = 0; break; }
                pairs = bigValCnt < np ? bigValCnt : np;
                do { total++; } while (--pairs > 0);
            } while ((bigValCnt -= np) > 0 && --sfbCnt >= 0);
        }
        return (ireg, sfb, bigValCnt, total);
    }
}
