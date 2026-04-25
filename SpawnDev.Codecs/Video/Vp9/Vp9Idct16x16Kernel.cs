// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP9 inverse DCT 16x16. Companion to slice 123's
// Vp9Idct16x16Reference. Same shape as slice 120 (iDCT 8x8): one thread
// per 16x16 block, LocalMemory<int>(256) for row-pass scratch, 7-stage
// butterfly per row + per column, final round (x + 32) >> 6.
//
// Unblocked by SpawnDev.ILGPU rc.12 (Geordi's LoopUnrolling body-cost
// cap = 320). Pre-rc.12 the 256-entry LocalMemory loop fully unrolled
// in WGSL and timed out Chrome's validator + V8 compile path. Rc.12
// caps the body-cost product so the loop emits a tight loop instead.
//
// Expected coverage: 5/6 backends (CPU / CUDA / OpenCL / WebGPU /
// Wasm). WebGL is gated out at the runner level for the same
// architectural reason as slices 120 and 131 - 256 `flat out`
// varyings per thread blow past GL_MAX_VARYING_VECTORS.

using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>Batched ILGPU kernel for the VP9 iDCT 16x16.</summary>
public sealed class Vp9Idct16x16Kernel : IDisposable
{
    // Q14 cosine constants per VP9 spec sec 8.7.1.4.
    private const int CosPi16_64 = 11585;
    private const int CosPi8_64 = 15137;
    private const int CosPi24_64 = 6270;
    private const int CosPi4_64 = 16069;
    private const int CosPi12_64 = 13623;
    private const int CosPi20_64 = 9102;
    private const int CosPi28_64 = 3196;
    private const int CosPi2_64 = 16305;
    private const int CosPi6_64 = 15679;
    private const int CosPi10_64 = 14449;
    private const int CosPi14_64 = 12665;
    private const int CosPi18_64 = 10394;
    private const int CosPi22_64 = 7723;
    private const int CosPi26_64 = 4756;
    private const int CosPi30_64 = 1606;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<byte>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9Idct16x16Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<byte>, int, int>(IdctKernel);
    }

    /// <summary>
    /// Run the iDCT across <paramref name="blockCount"/> 16x16 blocks
    /// using already-uploaded GPU buffers. The fast path: no allocate,
    /// no upload, no readback. The caller manages buffer lifetime and
    /// synchronisation - typical pattern is to issue this dispatch
    /// alongside other kernel work on the same accelerator stream and
    /// <c>await accelerator.SynchronizeAsync()</c> at the batch
    /// boundary.
    /// </summary>
    public void RunOnGpu(
        ArrayView<short> coeffs, ArrayView<byte> dest, int blockCount,
        int blockStrideBytes = 256)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (coeffs.Length < blockCount * 256L)
            throw new ArgumentException("coeffs too small", nameof(coeffs));
        if (dest.Length < blockCount * (long)blockStrideBytes)
            throw new ArgumentException("dest too small", nameof(dest));
        _kernel(blockCount, coeffs, dest, blockCount, blockStrideBytes);
    }

    /// <summary>
    /// Convenience wrapper that allocates GPU buffers, uploads
    /// <paramref name="coeffs"/> + <paramref name="predAndDest"/>,
    /// dispatches the kernel via <see cref="RunOnGpu"/>, synchronises,
    /// and reads the result back into <paramref name="predAndDest"/>.
    /// Production decode paths should hold their own GPU buffers and
    /// call <see cref="RunOnGpu"/> directly to avoid the per-block
    /// upload/readback cost; this overload is for one-shot work and
    /// unit tests that need a CPU result.
    /// </summary>
    public async Task RunAsync(
        ReadOnlyMemory<short> coeffs, Memory<byte> predAndDest, int blockCount,
        int blockStrideBytes = 256)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (coeffs.Length < blockCount * 256L)
            throw new ArgumentException("coeffs too small", nameof(coeffs));
        if (predAndDest.Length < blockCount * (long)blockStrideBytes)
            throw new ArgumentException("predAndDest too small", nameof(predAndDest));

        using var dCoeffs = _accelerator.Allocate1D<short>(blockCount * 256);
        using var dDest = _accelerator.Allocate1D<byte>(blockCount * (long)blockStrideBytes);
        dCoeffs.View.CopyFromCPU(coeffs.Span.ToArray());
        dDest.View.CopyFromCPU(predAndDest.Span.ToArray());
        RunOnGpu(dCoeffs.View, dDest.View, blockCount, blockStrideBytes);
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

        long cBase = (long)idx * 256;
        long dBase = (long)idx * blockStrideBytes;

        // Row pass scratch. int storage avoids the WGSL packed sub-word
        // path; same reasoning as the iDCT 8x8 kernel.
        var tmp = LocalMemory.Allocate<int>(256);

        // Row pass: 16 rows of 16-point iDCT.
        for (int row = 0; row < 16; row++)
        {
            long rBase = cBase + row * 16;
            Idct16Row(
                coeffs[rBase + 0],  coeffs[rBase + 1],  coeffs[rBase + 2],  coeffs[rBase + 3],
                coeffs[rBase + 4],  coeffs[rBase + 5],  coeffs[rBase + 6],  coeffs[rBase + 7],
                coeffs[rBase + 8],  coeffs[rBase + 9],  coeffs[rBase + 10], coeffs[rBase + 11],
                coeffs[rBase + 12], coeffs[rBase + 13], coeffs[rBase + 14], coeffs[rBase + 15],
                out int o0,  out int o1,  out int o2,  out int o3,
                out int o4,  out int o5,  out int o6,  out int o7,
                out int o8,  out int o9,  out int o10, out int o11,
                out int o12, out int o13, out int o14, out int o15);
            int rTmp = row * 16;
            tmp[rTmp + 0] = o0;   tmp[rTmp + 1] = o1;   tmp[rTmp + 2] = o2;   tmp[rTmp + 3] = o3;
            tmp[rTmp + 4] = o4;   tmp[rTmp + 5] = o5;   tmp[rTmp + 6] = o6;   tmp[rTmp + 7] = o7;
            tmp[rTmp + 8] = o8;   tmp[rTmp + 9] = o9;   tmp[rTmp + 10] = o10; tmp[rTmp + 11] = o11;
            tmp[rTmp + 12] = o12; tmp[rTmp + 13] = o13; tmp[rTmp + 14] = o14; tmp[rTmp + 15] = o15;
        }

        // Column pass: 16 columns of 16-point iDCT, with residual + clip on output.
        for (int col = 0; col < 16; col++)
        {
            Idct16Row(
                (short)tmp[ 0 * 16 + col], (short)tmp[ 1 * 16 + col],
                (short)tmp[ 2 * 16 + col], (short)tmp[ 3 * 16 + col],
                (short)tmp[ 4 * 16 + col], (short)tmp[ 5 * 16 + col],
                (short)tmp[ 6 * 16 + col], (short)tmp[ 7 * 16 + col],
                (short)tmp[ 8 * 16 + col], (short)tmp[ 9 * 16 + col],
                (short)tmp[10 * 16 + col], (short)tmp[11 * 16 + col],
                (short)tmp[12 * 16 + col], (short)tmp[13 * 16 + col],
                (short)tmp[14 * 16 + col], (short)tmp[15 * 16 + col],
                out int co0,  out int co1,  out int co2,  out int co3,
                out int co4,  out int co5,  out int co6,  out int co7,
                out int co8,  out int co9,  out int co10, out int co11,
                out int co12, out int co13, out int co14, out int co15);

            ApplyResidualAndClip(dest, dBase +  0 * 16 + col, co0);
            ApplyResidualAndClip(dest, dBase +  1 * 16 + col, co1);
            ApplyResidualAndClip(dest, dBase +  2 * 16 + col, co2);
            ApplyResidualAndClip(dest, dBase +  3 * 16 + col, co3);
            ApplyResidualAndClip(dest, dBase +  4 * 16 + col, co4);
            ApplyResidualAndClip(dest, dBase +  5 * 16 + col, co5);
            ApplyResidualAndClip(dest, dBase +  6 * 16 + col, co6);
            ApplyResidualAndClip(dest, dBase +  7 * 16 + col, co7);
            ApplyResidualAndClip(dest, dBase +  8 * 16 + col, co8);
            ApplyResidualAndClip(dest, dBase +  9 * 16 + col, co9);
            ApplyResidualAndClip(dest, dBase + 10 * 16 + col, co10);
            ApplyResidualAndClip(dest, dBase + 11 * 16 + col, co11);
            ApplyResidualAndClip(dest, dBase + 12 * 16 + col, co12);
            ApplyResidualAndClip(dest, dBase + 13 * 16 + col, co13);
            ApplyResidualAndClip(dest, dBase + 14 * 16 + col, co14);
            ApplyResidualAndClip(dest, dBase + 15 * 16 + col, co15);
        }
    }

    /// <summary>
    /// 16-point 1D iDCT butterfly, bit-exact against
    /// Vp9Idct16x16Reference.Idct16_1d. 7 stages.
    /// </summary>
    /// <remarks>
    /// <see cref="MethodImplOptions.NoInlining"/> tells the ILGPU IR
    /// Inliner to skip this method, which routes the WGSL codegen
    /// through the function-definition path Geordi added in
    /// SpawnDev.ILGPU rc.14 commit <c>1cb4f6c</c>. Without this
    /// attribute the WebGPU shader inlines the 7-stage 16-point
    /// butterfly at all 32 call sites in <see cref="IdctKernel"/>,
    /// producing ~3800 lines of straight-line WGSL that hits Chrome's
    /// validator compile cliff (~30s+ per kernel instance, every
    /// test method timing out). With the attribute, WGSL emits one
    /// <c>fn Idct16Row_NN(...)</c> definition + 32 function calls -
    /// the validator chews through it in milliseconds. Other backends
    /// (CPU / CUDA / OpenCL / Wasm) handle non-inlined methods
    /// natively and are unaffected by the attribute.
    /// </remarks>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void Idct16Row(
        short i0,  short i1,  short i2,  short i3,
        short i4,  short i5,  short i6,  short i7,
        short i8,  short i9,  short i10, short i11,
        short i12, short i13, short i14, short i15,
        out int o0,  out int o1,  out int o2,  out int o3,
        out int o4,  out int o5,  out int o6,  out int o7,
        out int o8,  out int o9,  out int o10, out int o11,
        out int o12, out int o13, out int o14, out int o15)
    {
        // Stage 1: bit-reversal-style input reordering.
        short s1_0 = i0;
        short s1_1 = i8;
        short s1_2 = i4;
        short s1_3 = i12;
        short s1_4 = i2;
        short s1_5 = i10;
        short s1_6 = i6;
        short s1_7 = i14;
        short s1_8 = i1;
        short s1_9 = i9;
        short s1_10 = i5;
        short s1_11 = i13;
        short s1_12 = i3;
        short s1_13 = i11;
        short s1_14 = i7;
        short s1_15 = i15;

        // Stage 2: pass-through 0..7, 4 rotations on 8..15.
        short s2_0 = s1_0;
        short s2_1 = s1_1;
        short s2_2 = s1_2;
        short s2_3 = s1_3;
        short s2_4 = s1_4;
        short s2_5 = s1_5;
        short s2_6 = s1_6;
        short s2_7 = s1_7;

        int t8a  = s1_8 * CosPi30_64 - s1_15 * CosPi2_64;
        int t8b  = s1_8 * CosPi2_64  + s1_15 * CosPi30_64;
        short s2_8  = (short)((t8a + (1 << 13)) >> 14);
        short s2_15 = (short)((t8b + (1 << 13)) >> 14);

        int t9a  = s1_9 * CosPi14_64 - s1_14 * CosPi18_64;
        int t9b  = s1_9 * CosPi18_64 + s1_14 * CosPi14_64;
        short s2_9  = (short)((t9a + (1 << 13)) >> 14);
        short s2_14 = (short)((t9b + (1 << 13)) >> 14);

        int t10a = s1_10 * CosPi22_64 - s1_13 * CosPi10_64;
        int t10b = s1_10 * CosPi10_64 + s1_13 * CosPi22_64;
        short s2_10 = (short)((t10a + (1 << 13)) >> 14);
        short s2_13 = (short)((t10b + (1 << 13)) >> 14);

        int t11a = s1_11 * CosPi6_64  - s1_12 * CosPi26_64;
        int t11b = s1_11 * CosPi26_64 + s1_12 * CosPi6_64;
        short s2_11 = (short)((t11a + (1 << 13)) >> 14);
        short s2_12 = (short)((t11b + (1 << 13)) >> 14);

        // Stage 3.
        short s3_0 = s2_0;
        short s3_1 = s2_1;
        short s3_2 = s2_2;
        short s3_3 = s2_3;

        int t4a = s2_4 * CosPi28_64 - s2_7 * CosPi4_64;
        int t4b = s2_4 * CosPi4_64  + s2_7 * CosPi28_64;
        short s3_4 = (short)((t4a + (1 << 13)) >> 14);
        short s3_7 = (short)((t4b + (1 << 13)) >> 14);

        int t5a = s2_5 * CosPi12_64 - s2_6 * CosPi20_64;
        int t5b = s2_5 * CosPi20_64 + s2_6 * CosPi12_64;
        short s3_5 = (short)((t5a + (1 << 13)) >> 14);
        short s3_6 = (short)((t5b + (1 << 13)) >> 14);

        short s3_8  = (short)( s2_8  + s2_9);
        short s3_9  = (short)( s2_8  - s2_9);
        short s3_10 = (short)(-s2_10 + s2_11);
        short s3_11 = (short)( s2_10 + s2_11);
        short s3_12 = (short)( s2_12 + s2_13);
        short s3_13 = (short)( s2_12 - s2_13);
        short s3_14 = (short)(-s2_14 + s2_15);
        short s3_15 = (short)( s2_14 + s2_15);

        // Stage 4.
        int t01a = (s3_0 + s3_1) * CosPi16_64;
        int t01b = (s3_0 - s3_1) * CosPi16_64;
        short s4_0 = (short)((t01a + (1 << 13)) >> 14);
        short s4_1 = (short)((t01b + (1 << 13)) >> 14);

        int t23a = s3_2 * CosPi24_64 - s3_3 * CosPi8_64;
        int t23b = s3_2 * CosPi8_64  + s3_3 * CosPi24_64;
        short s4_2 = (short)((t23a + (1 << 13)) >> 14);
        short s4_3 = (short)((t23b + (1 << 13)) >> 14);

        short s4_4 = (short)( s3_4 + s3_5);
        short s4_5 = (short)( s3_4 - s3_5);
        short s4_6 = (short)(-s3_6 + s3_7);
        short s4_7 = (short)( s3_6 + s3_7);

        short s4_8 = s3_8;
        short s4_15 = s3_15;

        int t9c  = -s3_9  * CosPi8_64  + s3_14 * CosPi24_64;
        int t9d  =  s3_9  * CosPi24_64 + s3_14 * CosPi8_64;
        short s4_9  = (short)((t9c + (1 << 13)) >> 14);
        short s4_14 = (short)((t9d + (1 << 13)) >> 14);

        int t10c = -s3_10 * CosPi24_64 - s3_13 * CosPi8_64;
        int t10d = -s3_10 * CosPi8_64  + s3_13 * CosPi24_64;
        short s4_10 = (short)((t10c + (1 << 13)) >> 14);
        short s4_13 = (short)((t10d + (1 << 13)) >> 14);

        short s4_11 = s3_11;
        short s4_12 = s3_12;

        // Stage 5.
        short s5_0 = (short)(s4_0 + s4_3);
        short s5_1 = (short)(s4_1 + s4_2);
        short s5_2 = (short)(s4_1 - s4_2);
        short s5_3 = (short)(s4_0 - s4_3);
        short s5_4 = s4_4;

        int t56a = (s4_6 - s4_5) * CosPi16_64;
        int t56b = (s4_5 + s4_6) * CosPi16_64;
        short s5_5 = (short)((t56a + (1 << 13)) >> 14);
        short s5_6 = (short)((t56b + (1 << 13)) >> 14);
        short s5_7 = s4_7;

        short s5_8  = (short)( s4_8  + s4_11);
        short s5_9  = (short)( s4_9  + s4_10);
        short s5_10 = (short)( s4_9  - s4_10);
        short s5_11 = (short)( s4_8  - s4_11);
        short s5_12 = (short)(-s4_12 + s4_15);
        short s5_13 = (short)(-s4_13 + s4_14);
        short s5_14 = (short)( s4_13 + s4_14);
        short s5_15 = (short)( s4_12 + s4_15);

        // Stage 6.
        short s6_0 = (short)(s5_0 + s5_7);
        short s6_1 = (short)(s5_1 + s5_6);
        short s6_2 = (short)(s5_2 + s5_5);
        short s6_3 = (short)(s5_3 + s5_4);
        short s6_4 = (short)(s5_3 - s5_4);
        short s6_5 = (short)(s5_2 - s5_5);
        short s6_6 = (short)(s5_1 - s5_6);
        short s6_7 = (short)(s5_0 - s5_7);
        short s6_8  = s5_8;
        short s6_9  = s5_9;

        int t1013a = (-s5_10 + s5_13) * CosPi16_64;
        int t1013b = ( s5_10 + s5_13) * CosPi16_64;
        short s6_10 = (short)((t1013a + (1 << 13)) >> 14);
        short s6_13 = (short)((t1013b + (1 << 13)) >> 14);

        int t1112a = (-s5_11 + s5_12) * CosPi16_64;
        int t1112b = ( s5_11 + s5_12) * CosPi16_64;
        short s6_11 = (short)((t1112a + (1 << 13)) >> 14);
        short s6_12 = (short)((t1112b + (1 << 13)) >> 14);

        short s6_14 = s5_14;
        short s6_15 = s5_15;

        // Stage 7: final combining butterfly.
        o0  = (short)(s6_0  + s6_15);
        o1  = (short)(s6_1  + s6_14);
        o2  = (short)(s6_2  + s6_13);
        o3  = (short)(s6_3  + s6_12);
        o4  = (short)(s6_4  + s6_11);
        o5  = (short)(s6_5  + s6_10);
        o6  = (short)(s6_6  + s6_9);
        o7  = (short)(s6_7  + s6_8);
        o8  = (short)(s6_7  - s6_8);
        o9  = (short)(s6_6  - s6_9);
        o10 = (short)(s6_5  - s6_10);
        o11 = (short)(s6_4  - s6_11);
        o12 = (short)(s6_3  - s6_12);
        o13 = (short)(s6_2  - s6_13);
        o14 = (short)(s6_1  - s6_14);
        o15 = (short)(s6_0  - s6_15);
    }

    private static void ApplyResidualAndClip(ArrayView<byte> dest, long offset, int colOut)
    {
        int residual = (colOut + 32) >> 6;
        int sum = dest[offset] + residual;
        if (sum < 0) sum = 0;
        else if (sum > 255) sum = 255;
        dest[offset] = (byte)sum;
    }

    /// <summary>Release kernel resources. Does not dispose the accelerator.</summary>
    public void Dispose() { }
}
