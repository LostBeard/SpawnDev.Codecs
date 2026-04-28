// GPU-side verification helpers for codec tests. Compare actual vs
// expected buffers on the device and read back only a single int
// (violation count) - never download the full result buffer to CPU.
//
// Pattern: caller computes the CPU reference, uploads it to a temporary
// device buffer, then runs a comparison kernel. Same total bandwidth as
// downloading the GPU result, but matches Rule 5a's "GPU-side
// verification" guidance and keeps the comparison loop on the device.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

internal static class GpuTestVerifyCodecs
{
    private static void CompareByteKernel(
        Index1D index, ArrayView<byte> actual, ArrayView<byte> expected,
        ArrayView<int> violations, int n)
    {
        if (index < n && actual[index] != expected[index])
            Atomic.Add(ref violations[0], 1);
    }

    private static void CompareShortKernel(
        Index1D index, ArrayView<short> actual, ArrayView<short> expected,
        ArrayView<int> violations, int n)
    {
        if (index < n && actual[index] != expected[index])
            Atomic.Add(ref violations[0], 1);
    }

    /// <summary>
    /// Count byte mismatches between <paramref name="actual"/> and a
    /// host-supplied <paramref name="expected"/> array. Uploads expected
    /// to the GPU and runs the comparison kernel there. Returns the
    /// mismatch count (single int read back).
    /// </summary>
    public static async Task<int> CountByteMismatches(
        Accelerator accelerator,
        ArrayView<byte> actual,
        byte[] expected,
        int n)
    {
        if (n < 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (actual.Length < n) throw new ArgumentException("actual is too short.", nameof(actual));
        if (expected.Length < n) throw new ArgumentException("expected is too short.", nameof(expected));

        using var dExpected = accelerator.Allocate1D<byte>(n);
        dExpected.View.CopyFromCPU(expected.AsSpan(0, n).ToArray());
        using var violations = accelerator.Allocate1D(new int[] { 0 });

        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<byte>, ArrayView<byte>, ArrayView<int>, int>(CompareByteKernel);
        kernel((Index1D)n, actual.SubView(0, n), dExpected.View, violations.View, n);
        await accelerator.SynchronizeAsync();
        var result = await SpawnDevContextExtensions.CopyToHostAsync(violations);
        return result[0];
    }

    /// <summary>
    /// Count short mismatches between <paramref name="actual"/> and a
    /// host-supplied <paramref name="expected"/> array. Uploads expected
    /// to the GPU and runs the comparison kernel there. Returns the
    /// mismatch count (single int read back).
    /// </summary>
    public static async Task<int> CountShortMismatches(
        Accelerator accelerator,
        ArrayView<short> actual,
        short[] expected,
        int n)
    {
        if (n < 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (actual.Length < n) throw new ArgumentException("actual is too short.", nameof(actual));
        if (expected.Length < n) throw new ArgumentException("expected is too short.", nameof(expected));

        using var dExpected = accelerator.Allocate1D<short>(n);
        dExpected.View.CopyFromCPU(expected.AsSpan(0, n).ToArray());
        using var violations = accelerator.Allocate1D(new int[] { 0 });

        var kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<short>, ArrayView<int>, int>(CompareShortKernel);
        kernel((Index1D)n, actual.SubView(0, n), dExpected.View, violations.View, n);
        await accelerator.SynchronizeAsync();
        var result = await SpawnDevContextExtensions.CopyToHostAsync(violations);
        return result[0];
    }
}
