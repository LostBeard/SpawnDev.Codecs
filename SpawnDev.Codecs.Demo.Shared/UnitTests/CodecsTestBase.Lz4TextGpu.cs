// The LZ4 / Snappy tests' "text" input (Lz4GenText: words from an LCG, a period + newline every ~9 words) generated ON THE
// ACCELERATOR, for the large cases (40 MB). Built as a managed array, the input sat in the browser's single-threaded
// .NET heap, and late in a full Wasm lane the heap could not find 40 MB: Lz4FrameCompressorGpu_40MB_TwoWaves failed with
// OutOfMemoryException in Lz4GenText in the 1.1.1 release sweep (2026-11-19), after its output hash had already moved to
// the device (2026-11-17). Bulk test data stays off the .NET heap in the browser, as bulk codec data does.
// One thread walks the same LCG with the same integer arithmetic, in bounded launches of TextChunk bytes that continue
// where the previous one stopped (state: the LCG value and the length written), so the bytes are Lz4GenText's exactly
// (Lz4GenTextGpu_MatchesManagedGenerator).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Bytes generated per launch (bounded launches: a long single-thread launch stalls the desktop / trips the watchdog).
    /// 1 MB was not bounded enough: ~100 ms per launch on CUDA (RTX 4070, nsys 2026-11-23), slower on WebGPU (the LCG is
    /// 64-bit, emulated there), and the 40 launches of a 40 MB input went out unpaced: late in a full WebGPU lane they
    /// reset the GPU (Windows TDR, 2 s) in Lz4FrameCompressorGpu_40MB_TwoWaves, on every Codecs/ILGPU version back to
    /// 1.1.3. 64 KB is ~6 ms on CUDA, and the launches go through LaunchPacer like the codecs' own.
    /// </summary>
    private const int Lz4TextChunk = 1 << 16;

    /// <summary>Inputs larger than this are generated on the accelerator.</summary>
    private const int Lz4LargeInput = 4_000_000;

    // state[0] = the LCG value, state[1] = bytes written; words = the 20 words back to back, info = (offset, length) per word
    private static void Lz4TextKernel(Index1D _, ArrayView<byte> outp, ArrayView<byte> words, ArrayView<int> info, ArrayView<long> state,
        int n, int chunkEnd)
    {
        long x = state[0];
        int len = (int)state[1];
        int end = chunkEnd < n ? chunkEnd : n;
        while (len < end)
        {
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF;
            int wi = (int)(x % 20);
            int off = info[2 * wi];
            int wl = info[2 * wi + 1];
            for (int i = 0; i < wl; i++)
            {
                if (len < n) outp[len] = words[off + i];
                if (len < n) len++;
            }
            if ((x >> 8) % 9 == 0)
            {
                if (len < n) { outp[len] = (byte)'.'; len++; }
                if (len < n) { outp[len] = (byte)'\n'; len++; }
            }
            else if (len < n)
            {
                outp[len] = (byte)' ';
                len++;
            }
        }
        state[0] = x;
        state[1] = len;
    }

    /// <summary>
    /// Lz4GenText(<paramref name="n"/>, <paramref name="seed"/>) generated on the accelerator into a new buffer of n bytes,
    /// followed by <paramref name="slack"/> bytes of <paramref name="poison"/> (as DeviceVerify.Upload lays it out).
    /// </summary>
    private static async Task<MemoryBuffer1D<byte, Stride1D.Dense>> Lz4GenTextGpuAsync(Accelerator acc, int n, int seed, int slack = 0, byte poison = 0xAB)
    {
        var buf = acc.Allocate1D<byte>(Math.Max(1, (long)n + slack));
        if (slack > 0) buf.View.MemSet(poison);
        var wordBytes = new List<byte>();
        var info = new int[2 * Lz4Words.Length];
        for (int w = 0; w < Lz4Words.Length; w++)
        {
            info[2 * w] = wordBytes.Count;
            info[2 * w + 1] = Lz4Words[w].Length;
            foreach (char ch in Lz4Words[w]) wordBytes.Add((byte)ch);
        }
        var kernel = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<long>, int, int>(Lz4TextKernel);
        using var dWords = acc.Allocate1D(wordBytes.ToArray());
        using var dInfo = acc.Allocate1D(info);
        using var dState = acc.Allocate1D(new long[] { seed & 0x7FFFFFFF, 0 });
        var pacer = new LaunchPacer(acc);
        for (long end = Lz4TextChunk; ; end += Lz4TextChunk)
        {
            kernel(1, buf.View, dWords.View, dInfo.View, dState.View, n, (int)Math.Min(end, n));
            await pacer.LaunchedAsync();
            if (end >= n) break;
        }
        // the small buffers are disposed on return: the launches that read them must be done first (async: the browser
        // backends refuse a blocking Synchronize)
        await acc.SynchronizeAsync();
        return buf;
    }

    /// <summary>The device generator writes Lz4GenText's bytes exactly: sizes inside one launch, across launches, odd seeds.</summary>
    [TestMethod(Timeout = 300000, Category = "NotOnWebGL")]
    public async Task Lz4GenTextGpu_MatchesManagedGenerator()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, Lz4Requirements);
        try
        {
            foreach (var (n, seed) in new[] { (1, 3), (13, 31), (1000, 77), (Lz4TextChunk - 3, 5), (Lz4TextChunk + 7, 31), (3 * Lz4TextChunk + 12345, -42) })
            {
                byte[] want = Lz4GenText(n, seed);
                using var got = await Lz4GenTextGpuAsync(acc, n, seed, 16);
                string? diff = await DeviceVerify.CompareAsync(acc, got.View, want);
                True(diff == null, $"n {n} seed {seed}: {diff}");
                True(await DeviceVerify.AllEqualAsync(acc, got.View.SubView(n, 16), 16, 0xAB) == null, $"n {n}: wrote past the end");
            }
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
