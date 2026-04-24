// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP9 inverse ADST 4x4. Mirrors the slice-117
// Vp9Idct4x4Kernel structure: one thread per 4x4 block, 16 coefficients
// read inline into locals (no LocalMemory - the 4x4 size fits comfortably
// in per-thread registers on every backend), 2-stage iADST butterfly,
// final residual-add + [0,255] clip into the destination pixels.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>Batched ILGPU kernel for VP9 iADST 4x4.</summary>
public sealed class Vp9Iadst4x4Kernel : IDisposable
{
    private const int SinPi1_9 = 5283;
    private const int SinPi2_9 = 9929;
    private const int SinPi3_9 = 13377;
    private const int SinPi4_9 = 15212;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<byte>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9Iadst4x4Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<byte>, int, int>(IadstKernel);
    }

    /// <summary>
    /// Run the iADST across <paramref name="blockCount"/> 4x4 blocks.
    /// Coefficient buffer is block-major (16 shorts per block); predictor /
    /// dest is block-major (16 bytes per block).
    /// </summary>
    public async Task RunAsync(
        ReadOnlyMemory<short> coeffs, Memory<byte> predAndDest, int blockCount,
        int blockStrideBytes = 16)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (coeffs.Length < blockCount * 16L)
            throw new ArgumentException("coeffs too small", nameof(coeffs));
        if (predAndDest.Length < blockCount * (long)blockStrideBytes)
            throw new ArgumentException("predAndDest too small", nameof(predAndDest));

        using var dCoeffs = _accelerator.Allocate1D<short>(blockCount * 16);
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

        long cBase = (long)idx * 16;
        long dBase = (long)idx * blockStrideBytes;

        // Read the 16 coefficients into named locals.
        short c00 = coeffs[cBase + 0],  c01 = coeffs[cBase + 1],
              c02 = coeffs[cBase + 2],  c03 = coeffs[cBase + 3];
        short c10 = coeffs[cBase + 4],  c11 = coeffs[cBase + 5],
              c12 = coeffs[cBase + 6],  c13 = coeffs[cBase + 7];
        short c20 = coeffs[cBase + 8],  c21 = coeffs[cBase + 9],
              c22 = coeffs[cBase + 10], c23 = coeffs[cBase + 11];
        short c30 = coeffs[cBase + 12], c31 = coeffs[cBase + 13],
              c32 = coeffs[cBase + 14], c33 = coeffs[cBase + 15];

        // Row pass: 4 iADST 1D transforms, results into t_ij intermediates.
        Iadst4Row(c00, c01, c02, c03, out short t00, out short t01, out short t02, out short t03);
        Iadst4Row(c10, c11, c12, c13, out short t10, out short t11, out short t12, out short t13);
        Iadst4Row(c20, c21, c22, c23, out short t20, out short t21, out short t22, out short t23);
        Iadst4Row(c30, c31, c32, c33, out short t30, out short t31, out short t32, out short t33);

        // Column pass: iADST on each column, residual-add + clip on pixel output.
        Iadst4Row(t00, t10, t20, t30, out short co00, out short co10, out short co20, out short co30);
        ApplyResidualAndClip(dest, dBase + 0 * 4 + 0, co00);
        ApplyResidualAndClip(dest, dBase + 1 * 4 + 0, co10);
        ApplyResidualAndClip(dest, dBase + 2 * 4 + 0, co20);
        ApplyResidualAndClip(dest, dBase + 3 * 4 + 0, co30);

        Iadst4Row(t01, t11, t21, t31, out short co01, out short co11, out short co21, out short co31);
        ApplyResidualAndClip(dest, dBase + 0 * 4 + 1, co01);
        ApplyResidualAndClip(dest, dBase + 1 * 4 + 1, co11);
        ApplyResidualAndClip(dest, dBase + 2 * 4 + 1, co21);
        ApplyResidualAndClip(dest, dBase + 3 * 4 + 1, co31);

        Iadst4Row(t02, t12, t22, t32, out short co02, out short co12, out short co22, out short co32);
        ApplyResidualAndClip(dest, dBase + 0 * 4 + 2, co02);
        ApplyResidualAndClip(dest, dBase + 1 * 4 + 2, co12);
        ApplyResidualAndClip(dest, dBase + 2 * 4 + 2, co22);
        ApplyResidualAndClip(dest, dBase + 3 * 4 + 2, co32);

        Iadst4Row(t03, t13, t23, t33, out short co03, out short co13, out short co23, out short co33);
        ApplyResidualAndClip(dest, dBase + 0 * 4 + 3, co03);
        ApplyResidualAndClip(dest, dBase + 1 * 4 + 3, co13);
        ApplyResidualAndClip(dest, dBase + 2 * 4 + 3, co23);
        ApplyResidualAndClip(dest, dBase + 3 * 4 + 3, co33);
    }

    /// <summary>
    /// 4-point 1D iADST butterfly. Bit-exact against
    /// <see cref="Vp9Iadst4x4Reference"/>'s Iadst4_1d.
    /// </summary>
    private static void Iadst4Row(
        short i0, short i1, short i2, short i3,
        out short o0, out short o1, out short o2, out short o3)
    {
        int x0 = i0;
        int x1 = i1;
        int x2 = i2;
        int x3 = i3;

        // Fast-path is handled as math-identity: zero inputs produce zero
        // sinpi*x intermediates and zero outputs, so the unconditional path
        // works without a branch.
        int s0 = SinPi1_9 * x0;
        int s1 = SinPi2_9 * x0;
        int s2 = SinPi3_9 * x1;
        int s3 = SinPi4_9 * x2;
        int s4 = SinPi1_9 * x2;
        int s5 = SinPi2_9 * x3;
        int s6 = SinPi4_9 * x3;
        int s7 = x0 - x2 + x3;

        int c0 = s0 + s3 + s5;
        int c1 = s1 - s4 - s6;
        int c3 = s2;
        int c2 = SinPi3_9 * s7;

        o0 = (short)((c0 + c3 + (1 << 13)) >> 14);
        o1 = (short)((c1 + c3 + (1 << 13)) >> 14);
        o2 = (short)((c2 + (1 << 13)) >> 14);
        o3 = (short)((c0 + c1 - c3 + (1 << 13)) >> 14);
    }

    private static void ApplyResidualAndClip(ArrayView<byte> dest, long offset, short colOut)
    {
        // Final round (x + 8) >> 4 - same as iDCT 4x4.
        int residual = (colOut + 8) >> 4;
        int sum = dest[offset] + residual;
        if (sum < 0) sum = 0;
        else if (sum > 255) sum = 255;
        dest[offset] = (byte)sum;
    }

    /// <summary>Release kernel resources. Does not dispose the accelerator.</summary>
    public void Dispose() { }
}
