// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP9 inverse DCT 8x8. Batched: one thread per 8x8
// block.
//
// WebGPU-safe design
//   Earlier draft used LocalMemory.Allocate<short>(64) for row-pass
//   intermediates. On WebGPU that packs into array<atomic<u32>> (per
//   SpawnDev.ILGPU CLAUDE.md - sub-word types use atomic RMW on GPU
//   backends). The packing + re-unpacking added load ordering that
//   didn't replicate the reference bit-exactly.
//
//   This version stores intermediates as int (4 bytes each, no packing)
//   which generates straight scalar loads/stores on every backend. 64
//   ints = 256 bytes per thread, well within WebGPU's private-memory
//   budget. Narrowing to int16 happens at the butterfly boundary via
//   the (short)(...) cast, exactly as the CPU reference does.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>Batched ILGPU kernel for the VP9 iDCT 8x8.</summary>
public sealed class Vp9Idct8x8Kernel : IDisposable
{
    private const int CosPi16_64 = 11585;
    private const int CosPi8_64 = 15137;
    private const int CosPi24_64 = 6270;
    private const int CosPi4_64 = 16069;
    private const int CosPi12_64 = 13623;
    private const int CosPi20_64 = 9102;
    private const int CosPi28_64 = 3196;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<byte>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9Idct8x8Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<byte>, int, int>(IdctKernel);
    }

    /// <summary>
    /// Run the iDCT across <paramref name="blockCount"/> 8x8 blocks.
    /// Coefficient buffer is block-major (64 shorts per block);
    /// dest is block-major (64 bytes per block).
    /// </summary>
    public async Task RunAsync(
        ReadOnlyMemory<short> coeffs, Memory<byte> predAndDest, int blockCount,
        int blockStrideBytes = 64)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (coeffs.Length < blockCount * 64L)
            throw new ArgumentException(
                $"coeffs must hold at least blockCount*64 shorts (got {coeffs.Length}).",
                nameof(coeffs));
        if (predAndDest.Length < blockCount * (long)blockStrideBytes)
            throw new ArgumentException(
                $"predAndDest too small for blockCount*blockStrideBytes.",
                nameof(predAndDest));

        using var dCoeffs = _accelerator.Allocate1D<short>(blockCount * 64);
        using var dDest = _accelerator.Allocate1D<byte>(blockCount * (long)blockStrideBytes);
        dCoeffs.View.CopyFromCPU(coeffs.Span.ToArray());
        dDest.View.CopyFromCPU(predAndDest.Span.ToArray());
        _kernel(blockCount, dCoeffs.View, dDest.View, blockCount, blockStrideBytes);
        await _accelerator.SynchronizeAsync();
        var readBack = await dDest.CopyToHostAsync();
        readBack.AsSpan(0, predAndDest.Length).CopyTo(predAndDest.Span);
    }

    private static void IdctKernel(
        Index1D blockIdx,
        ArrayView<short> coeffs,
        ArrayView<byte> dest,
        int blockCount,
        int blockStrideBytes)
    {
        int idx = blockIdx;
        if (idx >= blockCount) return;

        long cBase = (long)idx * 64;
        long dBase = (long)idx * blockStrideBytes;

        // Per-thread intermediate buffer: 64 ints. int storage dodges
        // the WebGPU packed-sub-word path that broke the int16-typed
        // local buffer in the first kernel attempt.
        var tmp = LocalMemory.Allocate<int>(64);

        // Row pass. Read 8 shorts from coefficient buffer, run 1D iDCT,
        // store 8 int16 results (sign-extended into int) into the row
        // slice of tmp.
        for (int row = 0; row < 8; row++)
        {
            long rBase = cBase + row * 8;
            Idct8Row(
                coeffs[rBase + 0], coeffs[rBase + 1], coeffs[rBase + 2], coeffs[rBase + 3],
                coeffs[rBase + 4], coeffs[rBase + 5], coeffs[rBase + 6], coeffs[rBase + 7],
                out int o0, out int o1, out int o2, out int o3,
                out int o4, out int o5, out int o6, out int o7);
            tmp[row * 8 + 0] = o0;
            tmp[row * 8 + 1] = o1;
            tmp[row * 8 + 2] = o2;
            tmp[row * 8 + 3] = o3;
            tmp[row * 8 + 4] = o4;
            tmp[row * 8 + 5] = o5;
            tmp[row * 8 + 6] = o6;
            tmp[row * 8 + 7] = o7;
        }

        // Column pass.
        for (int col = 0; col < 8; col++)
        {
            Idct8Row(
                (short)tmp[0 * 8 + col], (short)tmp[1 * 8 + col],
                (short)tmp[2 * 8 + col], (short)tmp[3 * 8 + col],
                (short)tmp[4 * 8 + col], (short)tmp[5 * 8 + col],
                (short)tmp[6 * 8 + col], (short)tmp[7 * 8 + col],
                out int co0, out int co1, out int co2, out int co3,
                out int co4, out int co5, out int co6, out int co7);
            ApplyResidualAndClip(dest, dBase + 0 * 8 + col, co0);
            ApplyResidualAndClip(dest, dBase + 1 * 8 + col, co1);
            ApplyResidualAndClip(dest, dBase + 2 * 8 + col, co2);
            ApplyResidualAndClip(dest, dBase + 3 * 8 + col, co3);
            ApplyResidualAndClip(dest, dBase + 4 * 8 + col, co4);
            ApplyResidualAndClip(dest, dBase + 5 * 8 + col, co5);
            ApplyResidualAndClip(dest, dBase + 6 * 8 + col, co6);
            ApplyResidualAndClip(dest, dBase + 7 * 8 + col, co7);
        }
    }

    /// <summary>
    /// 8-point 1D iDCT butterfly. Mirrors Vp9Idct8x8Reference.Idct8_1d
    /// bit-exactly; int16 narrowing at each butterfly sub-step (cast)
    /// reproduces libvpx WRAPLOW() semantics.
    /// </summary>
    private static void Idct8Row(
        short i0, short i1, short i2, short i3, short i4, short i5, short i6, short i7,
        out int o0, out int o1, out int o2, out int o3,
        out int o4, out int o5, out int o6, out int o7)
    {
        short s1_0 = i0;
        short s1_1 = i2;
        short s1_2 = i4;
        short s1_3 = i6;

        int t_a = i1 * CosPi28_64 - i7 * CosPi4_64;
        int t_b = i1 * CosPi4_64 + i7 * CosPi28_64;
        short s1_4 = (short)((t_a + (1 << 13)) >> 14);
        short s1_7 = (short)((t_b + (1 << 13)) >> 14);
        int t_c = i5 * CosPi12_64 - i3 * CosPi20_64;
        int t_d = i5 * CosPi20_64 + i3 * CosPi12_64;
        short s1_5 = (short)((t_c + (1 << 13)) >> 14);
        short s1_6 = (short)((t_d + (1 << 13)) >> 14);

        int t_e = (s1_0 + s1_2) * CosPi16_64;
        int t_f = (s1_0 - s1_2) * CosPi16_64;
        short s2_0 = (short)((t_e + (1 << 13)) >> 14);
        short s2_1 = (short)((t_f + (1 << 13)) >> 14);
        int t_g = s1_1 * CosPi24_64 - s1_3 * CosPi8_64;
        int t_h = s1_1 * CosPi8_64 + s1_3 * CosPi24_64;
        short s2_2 = (short)((t_g + (1 << 13)) >> 14);
        short s2_3 = (short)((t_h + (1 << 13)) >> 14);
        short s2_4 = (short)(s1_4 + s1_5);
        short s2_5 = (short)(s1_4 - s1_5);
        short s2_6 = (short)(-s1_6 + s1_7);
        short s2_7 = (short)(s1_6 + s1_7);

        short e1_0 = (short)(s2_0 + s2_3);
        short e1_1 = (short)(s2_1 + s2_2);
        short e1_2 = (short)(s2_1 - s2_2);
        short e1_3 = (short)(s2_0 - s2_3);
        short e1_4 = s2_4;
        int t_i = (s2_6 - s2_5) * CosPi16_64;
        int t_j = (s2_5 + s2_6) * CosPi16_64;
        short e1_5 = (short)((t_i + (1 << 13)) >> 14);
        short e1_6 = (short)((t_j + (1 << 13)) >> 14);
        short e1_7 = s2_7;

        // Return as int (caller stores into int buffer) - sign-extended
        // automatically. Final narrowing happens on the column-pass read.
        o0 = (short)(e1_0 + e1_7);
        o1 = (short)(e1_1 + e1_6);
        o2 = (short)(e1_2 + e1_5);
        o3 = (short)(e1_3 + e1_4);
        o4 = (short)(e1_3 - e1_4);
        o5 = (short)(e1_2 - e1_5);
        o6 = (short)(e1_1 - e1_6);
        o7 = (short)(e1_0 - e1_7);
    }

    private static void ApplyResidualAndClip(ArrayView<byte> dest, long offset, int colOut)
    {
        int residual = (colOut + 16) >> 5;
        int sum = dest[offset] + residual;
        if (sum < 0) sum = 0;
        else if (sum > 255) sum = 255;
        dest[offset] = (byte)sum;
    }

    /// <summary>Release kernel resources. Does not dispose the accelerator.</summary>
    public void Dispose() { }
}
