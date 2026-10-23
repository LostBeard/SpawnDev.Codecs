// Test-side verification on the accelerator. A test compares what a codec wrote INTO A DEVICE BUFFER with the truth
// uploaded next to it; the comparison runs in kernels and only a handful of ints come back. In the browser the host is
// single-threaded .NET Wasm: a test must never pull megabytes back to walk them byte by byte there.
//
// No atomics (WebGL has none): pass 1 gives every thread a chunk of elements and writes the chunk's (first mismatch,
// mismatch count); further passes fold those pairs the same way until one pair is left. Each thread stores once per output
// buffer at its own slot, every loop only reads and accumulates.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>Accelerator-side compare of device buffers for tests; only the summary crosses to the host.</summary>
internal static class DeviceVerify
{
    private const int Chunk = 512;

    private static void DiffBytesKernel(Index1D c, ArrayView<byte> actual, ArrayView<byte> expected,
        ArrayView<int> first, ArrayView<int> count, int n, int tolerance)
    {
        int start = c * Chunk;
        int end = start + Chunk;
        if (end > n) end = n;
        int f = -1;
        int k = 0;
        for (int i = start; i < end; i++)
        {
            int d = actual[i] - expected[i];
            if (d < 0) d = -d;
            if (d > tolerance)
            {
                if (f < 0) f = i;
                k++;
            }
        }
        first[c] = f;
        count[c] = k;
    }

    private static void FoldKernel(Index1D c, ArrayView<int> firstIn, ArrayView<int> countIn,
        ArrayView<int> firstOut, ArrayView<int> countOut, int n)
    {
        int start = c * Chunk;
        int end = start + Chunk;
        if (end > n) end = n;
        int f = -1;
        int k = 0;
        for (int i = start; i < end; i++)
        {
            int fi = firstIn[i];
            if (f < 0) f = fi;
            k += countIn[i];
        }
        firstOut[c] = f;
        countOut[c] = k;
    }

    private sealed class Kernels
    {
        public required Action<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int, int> Diff;
        public required Action<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int> Fold;
    }

    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<Accelerator, Kernels> Cache = new();

    private static Kernels For(Accelerator acc) => Cache.GetValue(acc, a => new Kernels
    {
        Diff = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, ArrayView<int>, int, int>(DiffBytesKernel),
        Fold = a.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>, ArrayView<int>, ArrayView<int>, int>(FoldKernel),
    });

    /// <summary>Number of elements that differ by more than <paramref name="tolerance"/>, and the index of the first one (-1 if none).</summary>
    public static async Task<(long Mismatches, long First)> DiffAsync(Accelerator acc,
        ArrayView1D<byte, Stride1D.Dense> actual, ArrayView1D<byte, Stride1D.Dense> expected, long length, int tolerance = 0)
    {
        if (length < 0 || length > actual.Length || length > expected.Length) throw new ArgumentOutOfRangeException(nameof(length));
        if (length == 0) return (0, -1);
        if (length > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(length), "DeviceVerify compares up to 2^31-1 elements.");
        var k = For(acc);
        int n = (int)length;
        int chunks = (n + Chunk - 1) / Chunk;
        var first = acc.Allocate1D<int>(chunks);
        var count = acc.Allocate1D<int>(chunks);
        try
        {
            k.Diff(chunks, actual.SubView(0, n), expected.SubView(0, n), first.View, count.View, n, tolerance);
            while (chunks > 1)
            {
                int next = (chunks + Chunk - 1) / Chunk;
                var f2 = acc.Allocate1D<int>(next);
                var c2 = acc.Allocate1D<int>(next);
                k.Fold(next, first.View, count.View, f2.View, c2.View, chunks);
                await acc.SynchronizeAsync();
                first.Dispose();
                count.Dispose();
                first = f2;
                count = c2;
                chunks = next;
            }
            await acc.SynchronizeAsync();
            var f = await first.View.CopyToHostAsync();
            var c = await count.View.CopyToHostAsync();
            return (c[0], f[0]);
        }
        finally
        {
            first.Dispose();
            count.Dispose();
        }
    }

    /// <summary>
    /// Compare <paramref name="length"/> bytes of two device buffers. Returns null when they agree, else a message with the
    /// mismatch count and the first differing byte (as a pixel position when <paramref name="width"/> is given).
    /// </summary>
    public static async Task<string?> CompareAsync(Accelerator acc,
        ArrayView1D<byte, Stride1D.Dense> actual, ArrayView1D<byte, Stride1D.Dense> expected, long length,
        int bytesPerPixel = 1, int width = 0, int tolerance = 0)
    {
        var (bad, firstIndex) = await DiffAsync(acc, actual, expected, length, tolerance);
        if (bad == 0) return null;
        var a = await actual.SubView(firstIndex, 1).CopyToHostAsync();
        var e = await expected.SubView(firstIndex, 1).CopyToHostAsync();
        string where = width > 0
            ? $"byte {firstIndex} (pixel x={firstIndex / bytesPerPixel % width} y={firstIndex / bytesPerPixel / width} channel byte {firstIndex % bytesPerPixel})"
            : $"byte {firstIndex}";
        string tol = tolerance > 0 ? $" by more than {tolerance}" : "";
        return $"{bad} of {length} bytes differ{tol}; first at {where}: got {a[0]}, expected {e[0]}";
    }

    /// <summary>
    /// Compare a result that came back to the host (a managed overload's return value) with device truth: the result is
    /// uploaded and compared on the accelerator. Also checks the length.
    /// </summary>
    public static async Task<string?> CompareHostResultAsync(Accelerator acc, ReadOnlyMemory<byte> actual,
        ArrayView1D<byte, Stride1D.Dense> expected, long expectedLength, int bytesPerPixel = 1, int width = 0, int tolerance = 0)
    {
        if (actual.Length != expectedLength) return $"length {actual.Length} != expected {expectedLength}";
        using var dActual = acc.Allocate1D<byte>(Math.Max(1, actual.Length));
        DeviceTransfer.Upload(dActual.View, actual.Span);
        return await CompareAsync(acc, dActual.View, expected, expectedLength, bytesPerPixel, width, tolerance);
    }

    /// <summary>Null when every one of <paramref name="length"/> bytes of <paramref name="view"/> is <paramref name="value"/>, else a message.</summary>
    public static async Task<string?> AllEqualAsync(Accelerator acc, ArrayView1D<byte, Stride1D.Dense> view, long length, byte value)
    {
        if (length == 0) return null;
        using var fill = acc.Allocate1D<byte>(length);
        fill.View.MemSet(value);
        return await CompareAsync(acc, view, fill.View, length);
    }

    /// <summary>A device buffer holding <paramref name="data"/> followed by <paramref name="slack"/> bytes of <paramref name="poison"/>.</summary>
    public static MemoryBuffer1D<byte, Stride1D.Dense> Upload(Accelerator acc, ReadOnlySpan<byte> data, int slack = 0, byte poison = 0xAB)
    {
        var buf = acc.Allocate1D<byte>(Math.Max(1, data.Length + slack));
        if (slack > 0) buf.View.MemSet(poison);
        DeviceTransfer.Upload(buf.View, data);
        return buf;
    }

    /// <summary>A device buffer of <paramref name="length"/> bytes, every byte <paramref name="poison"/>.</summary>
    public static MemoryBuffer1D<byte, Stride1D.Dense> Poisoned(Accelerator acc, long length, byte poison = 0xAB)
    {
        var buf = acc.Allocate1D<byte>(Math.Max(1, length));
        buf.View.MemSet(poison);
        return buf;
    }

    /// <summary>One byte of a device buffer (header fields, never bulk data).</summary>
    public static async Task<byte> ByteAtAsync(ArrayView1D<byte, Stride1D.Dense> view, long index) =>
        (await view.SubView(index, 1).CopyToHostAsync())[0];

    /// <summary>
    /// Upload <paramref name="expected"/> (the truth: necessary I/O) and compare it on the accelerator with the first
    /// expected.Length bytes of <paramref name="actual"/>.
    /// </summary>
    public static async Task<string?> CompareAsync(Accelerator acc,
        ArrayView1D<byte, Stride1D.Dense> actual, ReadOnlyMemory<byte> expected,
        int bytesPerPixel = 1, int width = 0, int tolerance = 0)
    {
        if (actual.Length < expected.Length) return $"actual holds {actual.Length} bytes, expected {expected.Length}";
        using var dExpected = acc.Allocate1D<byte>(Math.Max(1, expected.Length));
        DeviceTransfer.Upload(dExpected.View, expected.Span);
        return await CompareAsync(acc, actual, dExpected.View, expected.Length, bytesPerPixel, width, tolerance);
    }
}
