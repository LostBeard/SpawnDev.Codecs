// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP9 inverse ADST 8x8. Same shape as the iDCT
// 8x8 kernel (slice 120) that now runs 5/6 green after rc.10:
// one thread per block, 64-element LocalMemory<int> row-pass scratch,
// 3-stage iADST butterfly, (x + 16) >> 5 residual round + clip.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>Batched ILGPU kernel for VP9 iADST 8x8.</summary>
public sealed class Vp9Iadst8x8Kernel : IDisposable
{
    private const int CosPi2_64 = 16305;
    private const int CosPi6_64 = 15679;
    private const int CosPi8_64 = 15137;
    private const int CosPi10_64 = 14449;
    private const int CosPi14_64 = 12665;
    private const int CosPi16_64 = 11585;
    private const int CosPi18_64 = 10394;
    private const int CosPi22_64 = 7723;
    private const int CosPi24_64 = 6270;
    private const int CosPi26_64 = 4756;
    private const int CosPi30_64 = 1606;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<byte>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9Iadst8x8Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<byte>, int, int>(IadstKernel);
    }

    /// <summary>
    /// Run the iADST across <paramref name="blockCount"/> 8x8 blocks.
    /// </summary>
    public async Task RunAsync(
        ReadOnlyMemory<short> coeffs, Memory<byte> predAndDest, int blockCount,
        int blockStrideBytes = 64)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (coeffs.Length < blockCount * 64L)
            throw new ArgumentException("coeffs too small", nameof(coeffs));
        if (predAndDest.Length < blockCount * (long)blockStrideBytes)
            throw new ArgumentException("predAndDest too small", nameof(predAndDest));

        using var dCoeffs = _accelerator.Allocate1D<short>(blockCount * 64);
        using var dDest = _accelerator.Allocate1D<byte>(blockCount * (long)blockStrideBytes);
        dCoeffs.View.CopyFromCPU(coeffs.Span.ToArray());
        dDest.View.CopyFromCPU(predAndDest.Span.ToArray());
        _kernel(blockCount, dCoeffs.View, dDest.View, blockCount, blockStrideBytes);
        await _accelerator.SynchronizeAsync();
        var readBack = await dDest.CopyToHostAsync();
        readBack.AsSpan(0, predAndDest.Length).CopyTo(predAndDest.Span);
    }

    private static void IadstKernel(
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

        var tmp = LocalMemory.Allocate<int>(64);

        // Row pass.
        for (int row = 0; row < 8; row++)
        {
            long rBase = cBase + row * 8;
            Iadst8Row(
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
            Iadst8Row(
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
    /// 8-point 1D iADST butterfly. Bit-exact against
    /// <see cref="Vp9Iadst8x8Reference"/>.Iadst8_1d - the input
    /// reordering (x0..x7 from input[7,0,5,2,3,4,1,6]) happens at the
    /// CALL SITE: callers pass in the REORDERED inputs.
    ///
    /// Wait: actually the CALL SITE passes input[0..7] in natural
    /// order, because the kernel reads coeffs[row*8 + 0..7]. The
    /// reordering in Vp9Iadst8x8Reference.Iadst8_1d is internal. We
    /// must match that internal reordering here too. See the unshifted
    /// reads below.
    /// </summary>
    private static void Iadst8Row(
        short i0, short i1, short i2, short i3, short i4, short i5, short i6, short i7,
        out int o0, out int o1, out int o2, out int o3,
        out int o4, out int o5, out int o6, out int o7)
    {
        // libvpx reordering: x0 = input[7], x1 = input[0], x2 = input[5],
        // x3 = input[2], x4 = input[3], x5 = input[4], x6 = input[1],
        // x7 = input[6].
        int x0 = i7;
        int x1 = i0;
        int x2 = i5;
        int x3 = i2;
        int x4 = i3;
        int x5 = i4;
        int x6 = i1;
        int x7 = i6;

        // Stage 1.
        int s0 = CosPi2_64 * x0 + CosPi30_64 * x1;
        int s1 = CosPi30_64 * x0 - CosPi2_64 * x1;
        int s2 = CosPi10_64 * x2 + CosPi22_64 * x3;
        int s3 = CosPi22_64 * x2 - CosPi10_64 * x3;
        int s4 = CosPi18_64 * x4 + CosPi14_64 * x5;
        int s5 = CosPi14_64 * x4 - CosPi18_64 * x5;
        int s6 = CosPi26_64 * x6 + CosPi6_64 * x7;
        int s7 = CosPi6_64 * x6 - CosPi26_64 * x7;

        x0 = (short)((s0 + s4 + (1 << 13)) >> 14);
        x1 = (short)((s1 + s5 + (1 << 13)) >> 14);
        x2 = (short)((s2 + s6 + (1 << 13)) >> 14);
        x3 = (short)((s3 + s7 + (1 << 13)) >> 14);
        x4 = (short)((s0 - s4 + (1 << 13)) >> 14);
        x5 = (short)((s1 - s5 + (1 << 13)) >> 14);
        x6 = (short)((s2 - s6 + (1 << 13)) >> 14);
        x7 = (short)((s3 - s7 + (1 << 13)) >> 14);

        // Stage 2.
        s0 = x0;
        s1 = x1;
        s2 = x2;
        s3 = x3;
        s4 = CosPi8_64 * x4 + CosPi24_64 * x5;
        s5 = CosPi24_64 * x4 - CosPi8_64 * x5;
        s6 = -CosPi24_64 * x6 + CosPi8_64 * x7;
        s7 = CosPi8_64 * x6 + CosPi24_64 * x7;

        x0 = (short)(s0 + s2);
        x1 = (short)(s1 + s3);
        x2 = (short)(s0 - s2);
        x3 = (short)(s1 - s3);
        x4 = (short)((s4 + s6 + (1 << 13)) >> 14);
        x5 = (short)((s5 + s7 + (1 << 13)) >> 14);
        x6 = (short)((s4 - s6 + (1 << 13)) >> 14);
        x7 = (short)((s5 - s7 + (1 << 13)) >> 14);

        // Stage 3.
        s2 = CosPi16_64 * (x2 + x3);
        s3 = CosPi16_64 * (x2 - x3);
        s6 = CosPi16_64 * (x6 + x7);
        s7 = CosPi16_64 * (x6 - x7);

        x2 = (short)((s2 + (1 << 13)) >> 14);
        x3 = (short)((s3 + (1 << 13)) >> 14);
        x6 = (short)((s6 + (1 << 13)) >> 14);
        x7 = (short)((s7 + (1 << 13)) >> 14);

        // Output with sign inversions, cast widened to int for the storage path.
        o0 = (short)x0;
        o1 = (short)-x4;
        o2 = (short)x6;
        o3 = (short)-x2;
        o4 = (short)x3;
        o5 = (short)-x7;
        o6 = (short)x5;
        o7 = (short)-x1;
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
