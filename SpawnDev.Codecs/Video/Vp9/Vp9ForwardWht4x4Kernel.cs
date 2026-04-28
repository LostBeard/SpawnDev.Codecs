// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the VP9 4x4 forward Walsh-Hadamard (encoder side,
// lossless mode). Bit-exact mirror of Vp9ForwardWht4x4.Transform
// (libvpx vpx_fwht4x4_c port). One thread per 4x4 block.
//
// Pass 1 (rows): integer Hadamard butterfly.
// Pass 2 (cols): same butterfly, output multiplied by UNIT_QUANT_FACTOR
// (= 4) so the lossless inverse vpx_iwht4x4_16_add can recover the
// input via simple right shifts.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Vp9;

/// <summary>
/// Batched ILGPU kernel for the VP9 4x4 forward Walsh-Hadamard (lossless
/// mode). Bit-exact mirror of <see cref="Vp9ForwardWht4x4.Transform"/>.
/// </summary>
public sealed class Vp9ForwardWht4x4Kernel : IDisposable
{
    private const int UnitQuantFactor = Vp9ForwardWht4x4.UnitQuantFactor;

    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<short>, ArrayView<int>, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Vp9ForwardWht4x4Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<short>, ArrayView<int>, int>(WhtKernel);
    }

    /// <summary>Run on N blocks. 16 shorts in, 16 ints out per block.</summary>
    public void Run(ArrayView<short> input, ArrayView<int> output, int blockCount)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (blockCount == 0) return;
        if (input.Length < blockCount * 16L)
            throw new ArgumentException("input must hold blockCount*16 shorts.", nameof(input));
        if (output.Length < blockCount * 16L)
            throw new ArgumentException("output must hold blockCount*16 ints.", nameof(output));
        _kernel(blockCount, input, output, blockCount);
    }

    private static void WhtKernel(
        Index1D blockIdx,
        ArrayView<short> input,
        ArrayView<int> output,
        int blockCount)
    {
        int idx = blockIdx;
        if (idx >= blockCount) return;
        long inBase = (long)idx * 16;
        long outBase = (long)idx * 16;

        // Pass 1: rows -> column-major intermediate.
        // Read 4 elements per row at stride 4. Compute butterfly in long.
        // Write to register variables; then reload column-major in pass 2.
        Pass1(input[inBase + 0],  input[inBase + 4],  input[inBase + 8],  input[inBase + 12],
              out int p00, out int p04, out int p08, out int p12);
        Pass1(input[inBase + 1],  input[inBase + 5],  input[inBase + 9],  input[inBase + 13],
              out int p01, out int p05, out int p09, out int p13);
        Pass1(input[inBase + 2],  input[inBase + 6],  input[inBase + 10], input[inBase + 14],
              out int p02, out int p06, out int p10, out int p14);
        Pass1(input[inBase + 3],  input[inBase + 7],  input[inBase + 11], input[inBase + 15],
              out int p03, out int p07, out int p11, out int p15);

        // Pass 2 reads output[i*4 + 0..3] - rows of the intermediate.
        // After pass 1, intermediate[c + r*4] (0 <= c, r < 4) was built as:
        //   c=0,r=0..3: p00, p04, p08, p12
        //   c=1,r=0..3: p01, p05, p09, p13
        //   c=2,r=0..3: p02, p06, p10, p14
        //   c=3,r=0..3: p03, p07, p11, p15
        // Pass 2 row i reads (intermediate[i*4 + 0..3]):
        //   i=0: p00, p01, p02, p03
        //   i=1: p04, p05, p06, p07
        //   i=2: p08, p09, p10, p11
        //   i=3: p12, p13, p14, p15
        Pass2(p00, p01, p02, p03,
              out int q00, out int q01, out int q02, out int q03);
        Pass2(p04, p05, p06, p07,
              out int q04, out int q05, out int q06, out int q07);
        Pass2(p08, p09, p10, p11,
              out int q08, out int q09, out int q10, out int q11);
        Pass2(p12, p13, p14, p15,
              out int q12, out int q13, out int q14, out int q15);

        output[outBase + 0]  = q00; output[outBase + 1]  = q01;
        output[outBase + 2]  = q02; output[outBase + 3]  = q03;
        output[outBase + 4]  = q04; output[outBase + 5]  = q05;
        output[outBase + 6]  = q06; output[outBase + 7]  = q07;
        output[outBase + 8]  = q08; output[outBase + 9]  = q09;
        output[outBase + 10] = q10; output[outBase + 11] = q11;
        output[outBase + 12] = q12; output[outBase + 13] = q13;
        output[outBase + 14] = q14; output[outBase + 15] = q15;
    }

    /// <summary>libvpx pass-1 butterfly: writes 4 ints into intermediate[c+0/4/8/12].</summary>
    private static void Pass1(short s0, short s1, short s2, short s3,
        out int o0, out int o1, out int o2, out int o3)
    {
        long a1 = s0;
        long b1 = s1;
        long c1 = s2;
        long d1 = s3;

        a1 += b1;
        d1 = d1 - c1;
        long e1 = (a1 - d1) >> 1;
        b1 = e1 - b1;
        c1 = e1 - c1;
        a1 -= c1;
        d1 += b1;
        o0 = (int)a1;
        o1 = (int)c1;
        o2 = (int)d1;
        o3 = (int)b1;
    }

    /// <summary>libvpx pass-2 butterfly with UNIT_QUANT_FACTOR multiply.</summary>
    private static void Pass2(int s0, int s1, int s2, int s3,
        out int o0, out int o1, out int o2, out int o3)
    {
        long a1 = s0;
        long b1 = s1;
        long c1 = s2;
        long d1 = s3;

        a1 += b1;
        d1 -= c1;
        long e1 = (a1 - d1) >> 1;
        b1 = e1 - b1;
        c1 = e1 - c1;
        a1 -= c1;
        d1 += b1;
        o0 = (int)(a1 * UnitQuantFactor);
        o1 = (int)(c1 * UnitQuantFactor);
        o2 = (int)(d1 * UnitQuantFactor);
        o3 = (int)(b1 * UnitQuantFactor);
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped */ }
}
