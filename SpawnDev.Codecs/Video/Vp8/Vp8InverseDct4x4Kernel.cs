// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for VP8 inverse DCT 4x4 with predict-and-add. Bit-exact
// mirror of Vp8InverseTransform.ShortIdct4x4Llm. One thread per 4x4
// block. Operates on packed buffers (predStride=4, dstStride=4 per
// block) - caller scatters into the frame buffer after the kernel
// completes.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp8;

/// <summary>
/// Batched ILGPU kernel for VP8 inverse DCT 4x4 (libvpx
/// vp8_short_idct4x4llm_c). One thread per 4x4 block. Each block:
/// reads 16 coefs, reads 4x4 predictor bytes, runs the 1D IDCT
/// columns then rows, adds prediction, clips to [0,255], writes 4x4
/// destination bytes.
/// </summary>
public sealed class Vp8InverseDct4x4Kernel : IDisposable
{
    // libvpx Q16 constants (vp8_short_idct4x4llm_c).
    private const int CospiSqrt2Minus1 = 20091;
    private const int SinpiSqrt2 = 35468;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<byte>, ArrayView<byte>, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp8InverseDct4x4Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<byte>, ArrayView<byte>, int>(IdctKernel);
    }

    /// <summary>
    /// Run the IDCT+predict+add on <paramref name="blockCount"/> blocks.
    /// Coefs: 16 shorts/block. Pred: 16 bytes/block (4x4 packed). Dst:
    /// 16 bytes/block.
    /// </summary>
    public void Run(ArrayView<short> coefs, ArrayView<byte> pred, ArrayView<byte> dst, int blockCount)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (coefs.Length < blockCount * 16L)
            throw new ArgumentException("coefs must hold blockCount*16 shorts.", nameof(coefs));
        if (pred.Length < blockCount * 16L)
            throw new ArgumentException("pred must hold blockCount*16 bytes.", nameof(pred));
        if (dst.Length < blockCount * 16L)
            throw new ArgumentException("dst must hold blockCount*16 bytes.", nameof(dst));
        _kernel(blockCount, coefs, pred, dst, blockCount);
    }

    private static void IdctKernel(
        Index1D blockIdx,
        ArrayView<short> coefs,
        ArrayView<byte> pred,
        ArrayView<byte> dst,
        int blockCount)
    {
        int idx = blockIdx;
        if (idx >= blockCount) return;
        long cBase = (long)idx * 16;
        long pBase = (long)idx * 16;
        long dBase = (long)idx * 16;

        // Pull coefs into 16 short registers.
        short c00 = coefs[cBase + 0],  c01 = coefs[cBase + 1],
              c02 = coefs[cBase + 2],  c03 = coefs[cBase + 3];
        short c10 = coefs[cBase + 4],  c11 = coefs[cBase + 5],
              c12 = coefs[cBase + 6],  c13 = coefs[cBase + 7];
        short c20 = coefs[cBase + 8],  c21 = coefs[cBase + 9],
              c22 = coefs[cBase + 10], c23 = coefs[cBase + 11];
        short c30 = coefs[cBase + 12], c31 = coefs[cBase + 13],
              c32 = coefs[cBase + 14], c33 = coefs[cBase + 15];

        // Column pass: process columns (i = 0,1,2,3). ip[i+0], ip[i+4],
        // ip[i+8], ip[i+12] are one column.
        Idct4Col(c00, c10, c20, c30, out short s00, out short s10, out short s20, out short s30);
        Idct4Col(c01, c11, c21, c31, out short s01, out short s11, out short s21, out short s31);
        Idct4Col(c02, c12, c22, c32, out short s02, out short s12, out short s22, out short s32);
        Idct4Col(c03, c13, c23, c33, out short s03, out short s13, out short s23, out short s33);

        // Row pass with +4 round, >>3 shift, then predict + add + clip.
        // libvpx writes into stage2 then a separate predict-add loop;
        // we fuse for register efficiency.
        Idct4RowAdd(s00, s01, s02, s03, pred, pBase + 0, dst, dBase + 0);
        Idct4RowAdd(s10, s11, s12, s13, pred, pBase + 4, dst, dBase + 4);
        Idct4RowAdd(s20, s21, s22, s23, pred, pBase + 8, dst, dBase + 8);
        Idct4RowAdd(s30, s31, s32, s33, pred, pBase + 12, dst, dBase + 12);
    }

    /// <summary>1D IDCT column pass (no shift).</summary>
    private static void Idct4Col(
        short i0, short i1, short i2, short i3,
        out short o0, out short o1, out short o2, out short o3)
    {
        int a1 = i0 + i2;
        int b1 = i0 - i2;
        int temp1 = (i1 * SinpiSqrt2) >> 16;
        int temp2 = i3 + ((i3 * CospiSqrt2Minus1) >> 16);
        int c1 = temp1 - temp2;
        temp1 = i1 + ((i1 * CospiSqrt2Minus1) >> 16);
        temp2 = (i3 * SinpiSqrt2) >> 16;
        int d1 = temp1 + temp2;
        // libvpx writes (a+d, b+c, b-c, a-d) into output[0,4,8,12] in
        // column form. Map to (o0,o1,o2,o3) for our row major.
        o0 = (short)(a1 + d1);
        o3 = (short)(a1 - d1);
        o1 = (short)(b1 + c1);
        o2 = (short)(b1 - c1);
    }

    /// <summary>1D IDCT row pass with +4/&gt;&gt;3 + predict-add-clip.</summary>
    private static void Idct4RowAdd(
        short i0, short i1, short i2, short i3,
        ArrayView<byte> pred, long pBase,
        ArrayView<byte> dst, long dBase)
    {
        int a1 = i0 + i2;
        int b1 = i0 - i2;
        int temp1 = (i1 * SinpiSqrt2) >> 16;
        int temp2 = i3 + ((i3 * CospiSqrt2Minus1) >> 16);
        int c1 = temp1 - temp2;
        temp1 = i1 + ((i1 * CospiSqrt2Minus1) >> 16);
        temp2 = (i3 * SinpiSqrt2) >> 16;
        int d1 = temp1 + temp2;
        int r0 = (a1 + d1 + 4) >> 3;
        int r3 = (a1 - d1 + 4) >> 3;
        int r1 = (b1 + c1 + 4) >> 3;
        int r2 = (b1 - c1 + 4) >> 3;
        dst[dBase + 0] = ClipAdd(pred[pBase + 0], r0);
        dst[dBase + 1] = ClipAdd(pred[pBase + 1], r1);
        dst[dBase + 2] = ClipAdd(pred[pBase + 2], r2);
        dst[dBase + 3] = ClipAdd(pred[pBase + 3], r3);
    }

    private static byte ClipAdd(byte p, int r)
    {
        int a = p + r;
        if (a < 0) return 0;
        if (a > 255) return 255;
        return (byte)a;
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped */ }
}
