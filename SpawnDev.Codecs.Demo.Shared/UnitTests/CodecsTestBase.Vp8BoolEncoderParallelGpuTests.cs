// Vp8BoolEncoderParallelGpu: the parallel bool coder must write the bytes the serial coder (Vp8BoolEncoderGpu, libvpx
// vpx_write) writes for the same symbols, for every stream: empty, tiny, carry-heavy (long runs of 0xFF bytes), streams
// that span several windows (waves), production-size streams (3 million symbols, a large key frame), and streams whose
// carries survive the two carry passes (a carry through a run of 0xFF bytes: found by searching this generator's seeds
// with the CPU model, about one stream in 300) coded with 1-byte carry-lookahead blocks, so every such carry crosses
// a block boundary (random streams with 1024-byte blocks never exercised the lookahead: its red-check passed). The symbols
// are made on the accelerator from a hash of their index, the serial coder runs on the accelerator in bounded steps
// (its state saved between launches), and the bytes are compared on the accelerator: no stream crosses to the host.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp8;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>Symbol <paramref name="i"/> of test stream (seed, mode): (prob &lt;&lt; 1) | bit.</summary>
    private static int ParallelBoolTestSymbol(int i, int seed, int mode)
    {
        uint h = (uint)i * 0x9E3779B1u ^ (uint)seed * 0x85EBCA77u;
        h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
        int prob;
        if (mode == 0) prob = 1 + (int)(h % 255u);
        else if (mode == 1) prob = (h & 1u) != 0 ? 1 : 255;     // the cheapest / dearest symbols: long carry chains
        else if (mode == 2) prob = 128;
        else prob = 200 + (int)(h % 56u);
        int draw = (int)((h >> 8) & 255u);
        int bit = mode == 1 ? (prob == 1 ? 1 : 0) : (draw >= prob ? 1 : 0);
        if (((h >> 20) % 50u) == 0u) bit ^= 1;
        return (prob << 1) | bit;
    }

    // Thread t writes stream symbol baseIdx + t to the coder's window (the caller's symbol pass in production).
    private static void ParallelBoolTestWindowKernel(Index1D t, ArrayView<ushort> window, int baseIdx, int total, int seed, int mode)
    {
        int i = baseIdx + t;
        if (i < total) window[t] = (ushort)ParallelBoolTestSymbol(i, seed, mode);
    }

    // The serial oracle: one thread codes symbols [from, to) from the state in st (Vp8BoolEncoderGpu.Save layout), and
    // the stop bits when `stop` is set (outLen gets the length).
    private static void ParallelBoolTestSerialKernel(Index1D t, ArrayView<byte> output, ArrayView<int> st, ArrayView<long> outLen,
        int from, int to, int seed, int mode, int stop, int vp9)
    {
        var state = from == 0 ? Vp8BoolEncoderGpu.Init() : Vp8BoolEncoderGpu.Load(st, 0);
        for (int i = from; i < to; i++)
        {
            int sym = ParallelBoolTestSymbol(i, seed, mode);
            Vp8BoolEncoderGpu.EncodeBool(ref state, output, sym & 1, sym >> 1);
        }
        if (stop != 0)
        {
            if (vp9 != 0) Vp8BoolEncoderGpu.StopVp9(ref state, output);
            else Vp8BoolEncoderGpu.Stop(ref state, output);
            outLen[0] = state.OutLen;
        }
        Vp8BoolEncoderGpu.Save(st, 0, ref state);
    }

    [TestMethod(Timeout = 1800000, Category = "NotOnWebGL")]
    public async Task Vp8BoolEncoderParallelGpu_MatchesSerialCoder()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        RequireFeatures(acc, AtomicKernels);
        try
        {
            var window = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<ushort>, int, int, int, int>(ParallelBoolTestWindowKernel);
            var serial = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<int>, ArrayView<long>, int, int, int, int, int, int>(ParallelBoolTestSerialKernel);
            using var small = new Vp8BoolEncoderParallelGpu(acc, windowSymbols: Vp8BoolEncoderParallelGpu.ChunkSymbols * Vp8BoolEncoderParallelGpu.GroupChunks);
            using var large = new Vp8BoolEncoderParallelGpu(acc);
            using var tinyBlocks = new Vp8BoolEncoderParallelGpu(acc, windowSymbols: 4 * Vp8BoolEncoderParallelGpu.ChunkSymbols * Vp8BoolEncoderParallelGpu.GroupChunks, blockBytes: 1);
            using var st = acc.Allocate1D<int>(Vp8BoolEncoderGpu.StateInts);
            using var lens = acc.Allocate1D<long>(2);
            var failures = new List<string>();
            // (symbols, mode, coder, seed): coder 0 small windows = several waves per stream, 1 the default coder,
            // 2 1-byte carry blocks; 3 million = a large key frame's stream; seed 0 = the case's own number
            var cases = new List<(int N, int Mode, int Coder, int Seed)>
            {
                (0, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (17, 2, 0, 0), (1000, 0, 0, 0), (1000, 1, 0, 0),
                (16384 - 32, 3, 0, 0), (16384 - 31, 0, 0, 0), (70000, 0, 0, 0), (70000, 1, 0, 0), (70000, 3, 0, 0),
                (1, 0, 2, 0), (1000, 1, 2, 0), (70000, 1, 2, 0), (300_000, 3, 2, 0),
                // a carry survives the two carry passes in each of these (CPU model, 2026-11-13)
                (20000, 0, 2, 13), (20000, 0, 2, 97), (20000, 0, 2, 110), (20000, 2, 2, 199), (20000, 2, 2, 221),
                (20000, 2, 2, 232), (20000, 3, 2, 44), (20000, 3, 2, 102), (20000, 3, 2, 170),
                (3_000_000, 0, 1, 0), (3_000_000, 1, 1, 0), (3_000_000, 3, 0, 0),
            };
            int caseNumber = 0;
            foreach (var (n, mode, which, caseSeed) in cases)
                foreach (int vp9 in new[] { 1, 0 })
                {
                    caseNumber++;
                    int seed = caseSeed != 0 ? caseSeed : 100000 + caseNumber;
                    var coder = which == 0 ? small : which == 1 ? large : tinyBlocks;
                    int capacity = n + 64;   // a symbol costs at most 8 bits
                    using var expected = acc.Allocate1D<byte>(capacity);
                    using var actual = acc.Allocate1D<byte>(capacity);
                    expected.View.MemSetToZero();
                    actual.View.MemSet(0xAB);
                    // segments: the stream cut into pieces of varied size (empty ones included) for the offsets scan
                    var counts = new List<int>();
                    int left = n, k = 0;
                    while (left > 0 || counts.Count == 0)
                    {
                        int c = Math.Min(left, (k * 7919 + seed * 31) % 9 == 0 ? 0 : 1 + (k * 104729 + seed) % 3000);
                        counts.Add(c);
                        left -= c;
                        k++;
                    }
                    using var dCounts = acc.Allocate1D(counts.ToArray());

                    // serial oracle, bounded launches
                    const int step = 65536;
                    for (int from = 0; ; from += step)
                    {
                        int to = Math.Min(n, from + step);
                        bool last = to >= n;
                        serial(1, expected.View, st.View, lens.View.SubView(0, 1), from, to, seed, mode, last ? 1 : 0, vp9);
                        await acc.SynchronizeAsync();
                        if (last) break;
                    }

                    // parallel coder
                    coder.Begin(dCounts.View, counts.Count, capacity);
                    int waves = coder.WaveCount(n);
                    for (int w = 0; w < waves; w++)
                    {
                        window(coder.WindowSymbols, coder.Window, w * coder.WindowSymbols, n, seed, mode);
                        coder.CodeWave(w);
                        await acc.SynchronizeAsync();
                    }
                    coder.Finish(actual.View, lens.View.SubView(1, 1), vp9 != 0);
                    await acc.SynchronizeAsync();

                    long[] l = await lens.View.CopyToHostAsync();
                    string name = $"n={n} mode={mode} seed={seed} vp9={vp9} coder={which} waves={waves}";
                    if (l[0] != l[1]) { failures.Add($"{name}: length {l[1]}, serial {l[0]}"); continue; }
                    if (l[0] > capacity) { failures.Add($"{name}: length {l[0]} over the capacity {capacity}"); continue; }
                    string? diff = await DeviceVerify.CompareAsync(acc, actual.View, expected.View, l[0]);
                    if (diff != null) failures.Add($"{name}: {diff}");
                    // nothing written past the length
                    if (l[0] < capacity && await DeviceVerify.ByteAtAsync(actual.View, l[0]) != 0xAB)
                        failures.Add($"{name}: byte {l[0]} (past the end) was written");
                    // the offsets scan
                    var offsets = await coder.Offsets.SubView(0, counts.Count).CopyToHostAsync();
                    int run = 0;
                    for (int i = 0; i < counts.Count; i++)
                    {
                        if (offsets[i] != run) { failures.Add($"{name}: segment {i} offset {offsets[i]}, expected {run}"); break; }
                        run += counts[i];
                    }
                    Console.WriteLine($"[report] parallel bool coder {name}: {l[0]} bytes, {(diff == null ? "identical" : "DIFFERENT")}");
                }
            if (failures.Count > 0) throw new Exception(string.Join("\n", failures));
        }
        finally { ReleaseAccelerator(ctx, acc); }
    }
}
