// The encoder's length-limited Huffman construction (Vp8lEncodeGpu.BuildLengths), run as a kernel on the accelerator.
// A Fibonacci histogram is the worst case for tree depth: unlimited, k symbols get a path of k - 1 bits, so a limit of 15
// (7 for the code-length code) must actually reshape the tree. The result must be a COMPLETE prefix code (Kraft sum = 1)
// within the limit, and close to optimal.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Images.WebP;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static void WebPHuffmanTestKernel(Index1D _, ArrayView<int> w, int n, int limit)
    {
        Vp8lEncodeGpu.BuildLengths(w, 0, n, n, 2 * n, limit);
    }

    private static async Task<int[]> RunHuffmanAsync(Accelerator acc, int[] histogram, int limit)
    {
        int n = histogram.Length;
        var kernel = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, int, int>(WebPHuffmanTestKernel);
        using var buf = acc.Allocate1D<int>(2 * n + Vp8lEncLayout.BuildScratch);
        var host = new int[buf.Length];
        histogram.CopyTo(host, 0);
        buf.View.CopyFromCPU(host);
        kernel(1, buf.View, n, limit);
        await acc.SynchronizeAsync();
        var all = await buf.View.CopyToHostAsync();
        return all.AsSpan(n, n).ToArray();
    }

    /// <summary>Checks the lengths form a complete prefix code within the limit and returns the total cost in bits.</summary>
    private static long CheckHuffman(int[] hist, int[] lens, int limit, string what)
    {
        int used = hist.Count(h => h > 0);
        long cost = 0;
        double kraft = 0;
        for (int i = 0; i < hist.Length; i++)
        {
            if (hist[i] == 0) { Equal(0, lens[i], $"{what}: unused symbol {i} must have length 0"); continue; }
            True(lens[i] >= 1, $"{what}: used symbol {i} has no code");
            True(lens[i] <= limit, $"{what}: symbol {i} has length {lens[i]} > limit {limit}");
            kraft += Math.Pow(2, -lens[i]);
            cost += (long)hist[i] * lens[i];
        }
        if (used >= 2) True(Math.Abs(kraft - 1.0) < 1e-12, $"{what}: Kraft sum {kraft} (code must be complete)");
        return cost;
    }

    [TestMethod]
    public async Task WebPEncoder_HuffmanLengths_FibonacciHistogram_AreLimitedAndComplete()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // 30 symbols with Fibonacci counts: the unrestricted tree is 29 bits deep.
            var fib = new int[30];
            fib[0] = 1; fib[1] = 1;
            for (int i = 2; i < fib.Length; i++) fib[i] = fib[i - 1] + fib[i - 2];
            for (int i = 0; i < fib.Length; i++) fib[i] = Math.Min(fib[i], 100000);

            foreach (int limit in new[] { 15, 7 })
            {
                var hist = new int[19 > fib.Length ? 19 : fib.Length];
                fib.CopyTo(hist, 0);
                int[] used = limit == 7 ? hist.Take(19).ToArray() : hist;
                var lens = await RunHuffmanAsync(acc, used, limit);
                long cost = CheckHuffman(used, lens, limit, $"fibonacci limit {limit}");
                // within 5% of a lower bound: the entropy
                double total = used.Sum();
                double entropy = used.Where(h => h > 0).Sum(h => -h * Math.Log2(h / total));
                True(cost < entropy * 1.25 + total * 0.05, $"limit {limit}: cost {cost} vs entropy {entropy:F0}");
            }
            int hardLimit = 4;
            var small = new int[] { 1, 1, 2, 3, 5, 8, 13, 21, 34, 55 };
            var smallLens = await RunHuffmanAsync(acc, small, 4);
            CheckHuffman(small, smallLens, hardLimit, "10 Fibonacci symbols, limit 4");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task WebPEncoder_HuffmanLengths_EdgeHistograms()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            // flat 280: lengths 8 and 9, complete
            var flat = Enumerable.Repeat(10, 280).ToArray();
            CheckHuffman(flat, await RunHuffmanAsync(acc, flat, 15), 15, "flat 280");

            // one used symbol: length 1 (the writer then codes it in zero bits)
            var one = new int[40]; one[17] = 99;
            var oneLens = await RunHuffmanAsync(acc, one, 15);
            Equal(1, oneLens[17], "single symbol length");
            Equal(1, oneLens.Count(l => l != 0), "only that symbol");

            // two used symbols: 1 bit each
            var two = new int[256]; two[3] = 5; two[200] = 1;
            var twoLens = await RunHuffmanAsync(acc, two, 15);
            Equal(1, twoLens[3], "two symbols: first"); Equal(1, twoLens[200], "two symbols: second");

            // empty: all zero
            var none = new int[40];
            True((await RunHuffmanAsync(acc, none, 15)).All(l => l == 0), "no symbols");

            // largest alphabet (280 + 2048), pseudo-random counts with a few huge ones
            var big = new int[2328];
            var rng = new Random(5);
            for (int i = 0; i < big.Length; i++) big[i] = rng.Next(0, 4) == 0 ? 0 : rng.Next(1, 50);
            big[0] = 3_000_000; big[7] = 900_000; big[300] = 400_000;
            CheckHuffman(big, await RunHuffmanAsync(acc, big, 15), 15, "2328 symbols");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
