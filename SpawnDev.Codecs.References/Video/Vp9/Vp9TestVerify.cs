// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// GPU-side verification helpers for VP9 kernel unit tests. Following
// the SpawnDev.ILGPU GpuTestVerify pattern (libraries CLAUDE.md):
// "NEVER download large buffers to CPU with CopyToHostAsync when a
// GPU kernel can verify the result and return a single violation
// count."
//
// CountByteMismatches dispatches an atomic-counting kernel against a
// pair of pre-uploaded ArrayView<byte> buffers and returns a single
// int (the number of differing bytes). For a 32x32 transform block
// (1024 bytes) batched at 16, this replaces a 16 KiB GPU->CPU
// readback + a 16,384-iteration CPU compare loop with one int read.
// The browser-side speedup is dominant: WebGPU and Wasm pay both a
// JS<->Wasm boundary cost and a per-Equal call cost that the GPU-
// side path skips.
//
// WebGL note: Atomic.Add requires the atomics capability. Every VP9
// kernel test that consumes this helper already guards WebGL out
// (sub-word writes, varying-count limits, etc.), so the absence of
// atomics on WebGL is not a usable-path concern. If a future kernel
// is genuinely WebGL-compatible and wants this verifier, it should
// route through CPU comparison instead - GPU atomic counting is the
// wrong tool there anyway.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// GPU-side verification helpers for VP9 kernel unit tests. Internal
/// surface - not part of the production decoder API.
/// </summary>
internal static class Vp9TestVerify
{
    /// <summary>
    /// Count the number of byte positions where <paramref name="actual"/>
    /// differs from <paramref name="expected"/> over the first
    /// <paramref name="n"/> elements. Pure GPU work; only one int
    /// crosses the GPU/CPU boundary.
    /// </summary>
    public static async Task<int> CountByteMismatches(
        Accelerator accelerator,
        ArrayView<byte> actual,
        ArrayView<byte> expected,
        int n)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        if (n < 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (actual.Length < n)
            throw new ArgumentException("actual too short for n elements", nameof(actual));
        if (expected.Length < n)
            throw new ArgumentException("expected too short for n elements", nameof(expected));

        using var counter = accelerator.Allocate1D(new int[] { 0 });
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, int>(CountKernel);
        kernel((Index1D)n, actual, expected, counter.View, n);
        await accelerator.SynchronizeAsync();
        var result = await counter.CopyToHostAsync();
        return result[0];
    }

    /// <summary>
    /// Per-element compare-byte kernel. One thread per byte; on a
    /// mismatch it atomic-increments the counter. The atomic is
    /// contended only for actual mismatches, which is the cold path
    /// in any green test run.
    /// </summary>
    private static void CountKernel(
        Index1D index,
        ArrayView<byte> actual,
        ArrayView<byte> expected,
        ArrayView<int> count,
        int n)
    {
        int i = index;
        if (i >= n) return;
        if (actual[i] != expected[i])
            Atomic.Add(ref count[0], 1);
    }
}
