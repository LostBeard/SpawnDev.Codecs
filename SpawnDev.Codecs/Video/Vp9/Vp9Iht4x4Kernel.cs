// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP9 inverse 2D 4x4 hybrid transform. This is the
// kernel side of slice 122's Vp9Iht4x4Reference: a per-block tx_type
// selects one of four row+column transform pairs - iDCT or iADST in
// either direction. VP9 intra blocks carry tx_type as per-block
// metadata (spec sec 8.7.1.6), so the kernel reads it from a parallel
// ArrayView keyed by block index.
//
// Inline locals only - 16 shorts per block fit comfortably in thread
// registers on every backend, same shape as slices 117 (iDCT 4x4) and
// 130 (iADST 4x4). Expected backend coverage: 5/6 green. WebGL drops
// out on the same atomics constraint the other 4x4 kernels hit.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel for VP9 iHT 4x4. Each block carries its own
/// tx_type byte (0..3) which selects the row + column 1D transforms.
/// </summary>
public sealed class Vp9Iht4x4Kernel : IDisposable
{
    // Q14 iDCT cosines (VP9 spec sec 8.7.1.2).
    private const int CosPi8_64 = 15137;
    private const int CosPi16_64 = 11585;
    private const int CosPi24_64 = 6270;

    // Q14 iADST sinpi constants (VP9 spec sec 8.7.1.5).
    private const int SinPi1_9 = 5283;
    private const int SinPi2_9 = 9929;
    private const int SinPi3_9 = 13377;
    private const int SinPi4_9 = 15212;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<byte>, ArrayView<byte>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9Iht4x4Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<byte>, ArrayView<byte>, int, int>(IhtKernel);
    }

    /// <summary>
    /// Run the iHT across <paramref name="blockCount"/> 4x4 blocks.
    /// <paramref name="txTypes"/> holds one byte per block (0..3);
    /// the low bit selects row transform (0=iDCT, 1=iADST) and the
    /// high bit selects column transform.
    /// </summary>
    public async Task RunAsync(
        ReadOnlyMemory<short> coeffs,
        ReadOnlyMemory<byte> txTypes,
        Memory<byte> predAndDest,
        int blockCount,
        int blockStrideBytes = 16)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (coeffs.Length < blockCount * 16L)
            throw new ArgumentException("coeffs too small", nameof(coeffs));
        if (txTypes.Length < blockCount)
            throw new ArgumentException("txTypes too small", nameof(txTypes));
        if (predAndDest.Length < blockCount * (long)blockStrideBytes)
            throw new ArgumentException("predAndDest too small", nameof(predAndDest));

        using var dCoeffs = _accelerator.Allocate1D<short>(blockCount * 16);
        using var dTxTypes = _accelerator.Allocate1D<byte>(blockCount);
        using var dDest = _accelerator.Allocate1D<byte>(blockCount * (long)blockStrideBytes);
        dCoeffs.View.CopyFromCPU(coeffs.Span.ToArray());
        dTxTypes.View.CopyFromCPU(txTypes.Span.ToArray());
        dDest.View.CopyFromCPU(predAndDest.Span.ToArray());
        _kernel(blockCount, dCoeffs.View, dTxTypes.View, dDest.View, blockCount, blockStrideBytes);
        await _accelerator.SynchronizeAsync();
        var readBack = await dDest.CopyToHostAsync();
        readBack.AsSpan(0, predAndDest.Length).CopyTo(predAndDest.Span);
    }

    private static void IhtKernel(
        Index1D blockIdx,
        ArrayView<short> coeffs,
        ArrayView<byte> txTypes,
        ArrayView<byte> dest,
        int blockCount,
        int blockStrideBytes)
    {
        int idx = blockIdx;
        if (idx >= blockCount) return;

        long cBase = (long)idx * 16;
        long dBase = (long)idx * blockStrideBytes;

        int tx = txTypes[idx];
        bool rowIsAdst = (tx & 1) != 0;
        bool colIsAdst = (tx & 2) != 0;

        // Read 16 coefficients into registers.
        short c00 = coeffs[cBase + 0],  c01 = coeffs[cBase + 1],
              c02 = coeffs[cBase + 2],  c03 = coeffs[cBase + 3];
        short c10 = coeffs[cBase + 4],  c11 = coeffs[cBase + 5],
              c12 = coeffs[cBase + 6],  c13 = coeffs[cBase + 7];
        short c20 = coeffs[cBase + 8],  c21 = coeffs[cBase + 9],
              c22 = coeffs[cBase + 10], c23 = coeffs[cBase + 11];
        short c30 = coeffs[cBase + 12], c31 = coeffs[cBase + 13],
              c32 = coeffs[cBase + 14], c33 = coeffs[cBase + 15];

        short t00, t01, t02, t03;
        short t10, t11, t12, t13;
        short t20, t21, t22, t23;
        short t30, t31, t32, t33;

        // Row pass - single branch for all 4 rows since they share tx_type.
        if (rowIsAdst)
        {
            Iadst4Row(c00, c01, c02, c03, out t00, out t01, out t02, out t03);
            Iadst4Row(c10, c11, c12, c13, out t10, out t11, out t12, out t13);
            Iadst4Row(c20, c21, c22, c23, out t20, out t21, out t22, out t23);
            Iadst4Row(c30, c31, c32, c33, out t30, out t31, out t32, out t33);
        }
        else
        {
            Idct4Row(c00, c01, c02, c03, out t00, out t01, out t02, out t03);
            Idct4Row(c10, c11, c12, c13, out t10, out t11, out t12, out t13);
            Idct4Row(c20, c21, c22, c23, out t20, out t21, out t22, out t23);
            Idct4Row(c30, c31, c32, c33, out t30, out t31, out t32, out t33);
        }

        // Column pass - 4 columns, each branching on colIsAdst. Inline
        // the residual-add + clip per column so the 4 per-column outputs
        // can live purely in registers.
        short co00, co10, co20, co30;
        if (colIsAdst)
            Iadst4Row(t00, t10, t20, t30, out co00, out co10, out co20, out co30);
        else
            Idct4Row(t00, t10, t20, t30, out co00, out co10, out co20, out co30);
        ApplyResidualAndClip(dest, dBase + 0 * 4 + 0, co00);
        ApplyResidualAndClip(dest, dBase + 1 * 4 + 0, co10);
        ApplyResidualAndClip(dest, dBase + 2 * 4 + 0, co20);
        ApplyResidualAndClip(dest, dBase + 3 * 4 + 0, co30);

        short co01, co11, co21, co31;
        if (colIsAdst)
            Iadst4Row(t01, t11, t21, t31, out co01, out co11, out co21, out co31);
        else
            Idct4Row(t01, t11, t21, t31, out co01, out co11, out co21, out co31);
        ApplyResidualAndClip(dest, dBase + 0 * 4 + 1, co01);
        ApplyResidualAndClip(dest, dBase + 1 * 4 + 1, co11);
        ApplyResidualAndClip(dest, dBase + 2 * 4 + 1, co21);
        ApplyResidualAndClip(dest, dBase + 3 * 4 + 1, co31);

        short co02, co12, co22, co32;
        if (colIsAdst)
            Iadst4Row(t02, t12, t22, t32, out co02, out co12, out co22, out co32);
        else
            Idct4Row(t02, t12, t22, t32, out co02, out co12, out co22, out co32);
        ApplyResidualAndClip(dest, dBase + 0 * 4 + 2, co02);
        ApplyResidualAndClip(dest, dBase + 1 * 4 + 2, co12);
        ApplyResidualAndClip(dest, dBase + 2 * 4 + 2, co22);
        ApplyResidualAndClip(dest, dBase + 3 * 4 + 2, co32);

        short co03, co13, co23, co33;
        if (colIsAdst)
            Iadst4Row(t03, t13, t23, t33, out co03, out co13, out co23, out co33);
        else
            Idct4Row(t03, t13, t23, t33, out co03, out co13, out co23, out co33);
        ApplyResidualAndClip(dest, dBase + 0 * 4 + 3, co03);
        ApplyResidualAndClip(dest, dBase + 1 * 4 + 3, co13);
        ApplyResidualAndClip(dest, dBase + 2 * 4 + 3, co23);
        ApplyResidualAndClip(dest, dBase + 3 * 4 + 3, co33);
    }

    /// <summary>
    /// 4-point 1D iDCT butterfly. Bit-exact against Vp9Idct4x4Reference.
    /// </summary>
    private static void Idct4Row(
        short i0, short i1, short i2, short i3,
        out short o0, out short o1, out short o2, out short o3)
    {
        int t1 = (i0 + i2) * CosPi16_64;
        int t2 = (i0 - i2) * CosPi16_64;
        short step0 = (short)((t1 + (1 << 13)) >> 14);
        short step1 = (short)((t2 + (1 << 13)) >> 14);
        int t3 = i1 * CosPi24_64 - i3 * CosPi8_64;
        int t4 = i1 * CosPi8_64 + i3 * CosPi24_64;
        short step2 = (short)((t3 + (1 << 13)) >> 14);
        short step3 = (short)((t4 + (1 << 13)) >> 14);
        o0 = (short)(step0 + step3);
        o1 = (short)(step1 + step2);
        o2 = (short)(step1 - step2);
        o3 = (short)(step0 - step3);
    }

    /// <summary>
    /// 4-point 1D iADST butterfly. Bit-exact against Vp9Iadst4x4Reference.
    /// </summary>
    private static void Iadst4Row(
        short i0, short i1, short i2, short i3,
        out short o0, out short o1, out short o2, out short o3)
    {
        int x0 = i0;
        int x1 = i1;
        int x2 = i2;
        int x3 = i3;

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
        int residual = (colOut + 8) >> 4;
        int sum = dest[offset] + residual;
        if (sum < 0) sum = 0;
        else if (sum > 255) sum = 255;
        dest[offset] = (byte)sum;
    }

    /// <summary>Release kernel resources. Does not dispose the accelerator.</summary>
    public void Dispose() { }
}
