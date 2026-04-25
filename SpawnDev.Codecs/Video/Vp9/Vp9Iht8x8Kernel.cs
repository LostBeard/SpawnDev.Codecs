// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP9 inverse 2D 8x8 hybrid transform. Kernel-side
// companion to slice 126's Vp9Iht8x8Reference. tx_type is a SCALAR
// kernel parameter - every block in a single dispatch shares it. The
// decoder calls RunAsync up to 4 times per frame, once per tx_type
// group (the CPU reference is identically per-call: one tx_type per
// invocation).
//
// Why scalar (not per-block ArrayView): an earlier draft passed
// tx_type per block via a parallel ArrayView<byte>. On WebGPU and
// Wasm that produced bit-exact divergence at n>=2 batched dispatches
// when blocks within the workgroup carried different tx_types, even
// though the equivalent inline-locals 4x4 kernel handled mixed
// tx_types cleanly at n=64. The divergence reproduced only with the
// LocalMemory<int>(64) row-pass scratch buffer; uniform tx_type and
// uniform CF across the workgroup restores bit-exact output. Real
// VP9 decode groups blocks by tx_type before dispatch anyway.
//
// Same kernel shape as slice 120 (iDCT 8x8) and slice 131 (iADST 8x8):
// one thread per 8x8 block, LocalMemory<int>(64) for row-pass
// intermediates - the int storage avoids the WebGPU sub-word atomic
// path that broke a short-typed buffer in earlier iterations. Final
// residual round (x + 16) >> 5 is shared by both 8-point transforms.
//
// Expected coverage: 5/6 backends (CPU / CUDA / OpenCL / WebGPU /
// Wasm). WebGL is gated out at the runner level - 64 `flat out`
// varyings per thread exceed GL_MAX_VARYING_VECTORS, same architectural
// constraint that blocks slice 120 and 131 today.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>Batched ILGPU kernel for VP9 iHT 8x8.</summary>
public sealed class Vp9Iht8x8Kernel : IDisposable
{
    // Q14 iDCT cosines (VP9 spec sec 8.7.1.2).
    private const int CosPi4_64 = 16069;
    private const int CosPi8_64 = 15137;
    private const int CosPi12_64 = 13623;
    private const int CosPi16_64 = 11585;
    private const int CosPi20_64 = 9102;
    private const int CosPi24_64 = 6270;
    private const int CosPi28_64 = 3196;

    // Q14 iADST cosines (VP9 spec sec 8.7.1.5).
    private const int CosPi2_64 = 16305;
    private const int CosPi6_64 = 15679;
    private const int CosPi10_64 = 14449;
    private const int CosPi14_64 = 12665;
    private const int CosPi18_64 = 10394;
    private const int CosPi22_64 = 7723;
    private const int CosPi26_64 = 4756;
    private const int CosPi30_64 = 1606;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<byte>, int, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9Iht8x8Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<byte>, int, int, int>(IhtKernel);
    }

    /// <summary>
    /// Run the iHT across <paramref name="blockCount"/> 8x8 blocks. The
    /// <paramref name="txType"/> applies uniformly to every block in
    /// this dispatch. Group blocks by tx_type at the call site if
    /// the frame mixes types.
    /// </summary>
    public async Task RunAsync(
        Vp9TxType8x8 txType,
        ReadOnlyMemory<short> coeffs,
        Memory<byte> predAndDest,
        int blockCount,
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
        _kernel(blockCount, dCoeffs.View, dDest.View, (int)txType, blockCount, blockStrideBytes);
        await _accelerator.SynchronizeAsync();
        var readBack = await dDest.CopyToHostAsync();
        readBack.AsSpan(0, predAndDest.Length).CopyTo(predAndDest.Span);
    }

    private static void IhtKernel(
        Index1D blockIdx,
        ArrayView<short> coeffs,
        ArrayView<byte> dest,
        int txType,
        int blockCount,
        int blockStrideBytes)
    {
        int idx = blockIdx;
        if (idx >= blockCount) return;

        long cBase = (long)idx * 64;
        long dBase = (long)idx * blockStrideBytes;

        // Scalar txType -> uniform branches across the workgroup.
        bool rowIsAdst = (txType & 1) != 0;
        bool colIsAdst = (txType & 2) != 0;

        var tmp = LocalMemory.Allocate<int>(64);

        // Row pass.
        if (rowIsAdst)
        {
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
        }
        else
        {
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
        }

        // Column pass.
        if (colIsAdst)
        {
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
        else
        {
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
    }

    /// <summary>
    /// 8-point 1D iDCT butterfly. Bit-exact against Vp9Idct8x8Reference.Idct8_1d.
    /// Mirrors slice 120 verbatim - WRAPLOW narrowing via short cast at
    /// each butterfly sub-step.
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

        o0 = (short)(e1_0 + e1_7);
        o1 = (short)(e1_1 + e1_6);
        o2 = (short)(e1_2 + e1_5);
        o3 = (short)(e1_3 + e1_4);
        o4 = (short)(e1_3 - e1_4);
        o5 = (short)(e1_2 - e1_5);
        o6 = (short)(e1_1 - e1_6);
        o7 = (short)(e1_0 - e1_7);
    }

    /// <summary>
    /// 8-point 1D iADST butterfly. Bit-exact against Vp9Iadst8x8Reference.Iadst8_1d.
    /// Internal libvpx reordering [7,0,5,2,3,4,1,6] is applied before
    /// stage 1.
    /// </summary>
    private static void Iadst8Row(
        short i0, short i1, short i2, short i3, short i4, short i5, short i6, short i7,
        out int o0, out int o1, out int o2, out int o3,
        out int o4, out int o5, out int o6, out int o7)
    {
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

        // Output with sign inversions per libvpx iadst8_c.
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
