// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP9 inverse DCT 4x4. Runs the same normative
// integer butterfly as Vp9Idct4x4Reference on any ILGPU backend -
// CPU (emulator), CUDA, OpenCL, WebGPU, WebGL, Wasm. Batched: one
// thread per 4x4 block, N blocks in parallel.
//
// Every ILGPU backend executes the IDENTICAL integer math. VP9 is a
// normative bitstream - the reference requires bit-for-bit output
// equality across all implementations. The test suite asserts this
// directly: for each random coefficient input, the kernel output on
// the active accelerator must match Vp9Idct4x4Reference byte-for-byte.
// A one-byte divergence fails the test.

using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel that runs the VP9 iDCT 4x4 across N independent
/// 4x4 blocks in parallel. The kernel is stateless - create once, dispatch
/// many times against different coefficient + pixel buffers.
/// </summary>
public sealed class Vp9Idct4x4Kernel : IDisposable
{
    // Q14 cosine constants per VP9 spec sec 8.7.1.2. Must match Reference.
    private const int CosPi16_64 = 11585;
    private const int CosPi8_64 = 15137;
    private const int CosPi24_64 = 6270;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<byte>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9Idct4x4Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<byte>, int, int>(IdctKernel);
    }

    /// <summary>
    /// Run the iDCT on <paramref name="blockCount"/> blocks.
    /// <paramref name="coeffs"/> is block-major flat: block i's 16
    /// coefficients live at offset i*16. <paramref name="predAndDest"/>
    /// is block-major flat with per-block stride = 16; each block's 4x4
    /// predictor occupies the first 16 bytes (4 rows of 4 pixels).
    /// After the call the same buffer holds the clipped residual-added
    /// pixels.
    /// </summary>
    public void Run(
        ArrayView<short> coeffs,
        ArrayView<byte> predAndDest,
        int blockCount,
        int blockStrideBytes = 16)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (coeffs.Length < blockCount * 16L)
            throw new ArgumentException(
                $"coeffs buffer must hold at least blockCount*16 shorts (got {coeffs.Length}).",
                nameof(coeffs));
        if (predAndDest.Length < blockCount * (long)blockStrideBytes)
            throw new ArgumentException(
                $"predAndDest buffer must hold at least blockCount*blockStrideBytes bytes.",
                nameof(predAndDest));
        _kernel(blockCount, coeffs, predAndDest, blockCount, blockStrideBytes);
    }

    /// <summary>
    /// Convenience: allocate temporary GPU buffers, run the kernel,
    /// read the result back. For tests and one-shot work where you don't
    /// have long-lived GPU buffers handy.
    /// </summary>
    public void Run(
        ReadOnlySpan<short> coeffs, Span<byte> predAndDest, int blockCount,
        int blockStrideBytes = 16)
    {
        if (blockCount <= 0) return;
        using var dCoeffs = _accelerator.Allocate1D<short>(blockCount * 16);
        using var dDest = _accelerator.Allocate1D<byte>(blockCount * (long)blockStrideBytes);
        dCoeffs.View.CopyFromCPU(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(coeffs),
            blockCount * 16);
        dDest.View.CopyFromCPU(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(predAndDest),
            blockCount * (long)blockStrideBytes);
        _kernel(blockCount, dCoeffs.View, dDest.View, blockCount, blockStrideBytes);
        _accelerator.Synchronize();
        dDest.View.CopyToCPU(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(predAndDest),
            blockCount * (long)blockStrideBytes);
    }

    /// <summary>Kernel body. One thread per block.</summary>
    private static void IdctKernel(
        Index1D blockIdx,
        ArrayView<short> coeffs,
        ArrayView<byte> dest,
        int blockCount,
        int blockStrideBytes)
    {
        int idx = blockIdx;
        if (idx >= blockCount) return;

        // Pull the 16 coefficients for this block into registers.
        long cBase = (long)idx * 16;
        short c00 = coeffs[cBase + 0],  c01 = coeffs[cBase + 1],
              c02 = coeffs[cBase + 2],  c03 = coeffs[cBase + 3];
        short c10 = coeffs[cBase + 4],  c11 = coeffs[cBase + 5],
              c12 = coeffs[cBase + 6],  c13 = coeffs[cBase + 7];
        short c20 = coeffs[cBase + 8],  c21 = coeffs[cBase + 9],
              c22 = coeffs[cBase + 10], c23 = coeffs[cBase + 11];
        short c30 = coeffs[cBase + 12], c31 = coeffs[cBase + 13],
              c32 = coeffs[cBase + 14], c33 = coeffs[cBase + 15];

        // Row transform -> 4x4 int16 intermediates (16 vars).
        // Row 0
        Idct4Row(c00, c01, c02, c03, out short t00, out short t01, out short t02, out short t03);
        // Row 1
        Idct4Row(c10, c11, c12, c13, out short t10, out short t11, out short t12, out short t13);
        // Row 2
        Idct4Row(c20, c21, c22, c23, out short t20, out short t21, out short t22, out short t23);
        // Row 3
        Idct4Row(c30, c31, c32, c33, out short t30, out short t31, out short t32, out short t33);

        // Column transform - produce 4x4 int16 "colOut" values then apply
        // to dest pixels with final rounding and clipping.
        long dBase = (long)idx * blockStrideBytes;

        // Column 0
        Idct4Row(t00, t10, t20, t30, out short co00, out short co10, out short co20, out short co30);
        ApplyResidualAndClip(dest, dBase + 0 * 4 + 0, co00);
        ApplyResidualAndClip(dest, dBase + 1 * 4 + 0, co10);
        ApplyResidualAndClip(dest, dBase + 2 * 4 + 0, co20);
        ApplyResidualAndClip(dest, dBase + 3 * 4 + 0, co30);

        // Column 1
        Idct4Row(t01, t11, t21, t31, out short co01, out short co11, out short co21, out short co31);
        ApplyResidualAndClip(dest, dBase + 0 * 4 + 1, co01);
        ApplyResidualAndClip(dest, dBase + 1 * 4 + 1, co11);
        ApplyResidualAndClip(dest, dBase + 2 * 4 + 1, co21);
        ApplyResidualAndClip(dest, dBase + 3 * 4 + 1, co31);

        // Column 2
        Idct4Row(t02, t12, t22, t32, out short co02, out short co12, out short co22, out short co32);
        ApplyResidualAndClip(dest, dBase + 0 * 4 + 2, co02);
        ApplyResidualAndClip(dest, dBase + 1 * 4 + 2, co12);
        ApplyResidualAndClip(dest, dBase + 2 * 4 + 2, co22);
        ApplyResidualAndClip(dest, dBase + 3 * 4 + 2, co32);

        // Column 3
        Idct4Row(t03, t13, t23, t33, out short co03, out short co13, out short co23, out short co33);
        ApplyResidualAndClip(dest, dBase + 0 * 4 + 3, co03);
        ApplyResidualAndClip(dest, dBase + 1 * 4 + 3, co13);
        ApplyResidualAndClip(dest, dBase + 2 * 4 + 3, co23);
        ApplyResidualAndClip(dest, dBase + 3 * 4 + 3, co33);
    }

    /// <summary>4-point 1D iDCT butterfly. Kernel-safe (no stackalloc).</summary>
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
    /// Add the rounded residual to the predictor byte at
    /// <paramref name="offset"/>, clip to [0, 255], and write it back.
    /// </summary>
    private static void ApplyResidualAndClip(ArrayView<byte> dest, long offset, short colOut)
    {
        int residual = (colOut + 8) >> 4;
        int sum = dest[offset] + residual;
        if (sum < 0) sum = 0;
        else if (sum > 255) sum = 255;
        dest[offset] = (byte)sum;
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped kernels don't need explicit disposal */ }
}
