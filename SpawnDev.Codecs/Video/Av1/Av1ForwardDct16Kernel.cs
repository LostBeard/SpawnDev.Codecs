// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the AV1 16-point forward DCT (1D). Bit-exact mirror
// of Av1ForwardDct16.Transform - one thread per 16-element 1D
// transform. Runs on every ILGPU backend.
//
// 7 stages with cospi-driven half_btf rotations + final bit-reversed
// scatter to output. Implementation uses LocalMemory<int>(16) for the
// stage scratch buffers since 16 elements per stage is too many for a
// readable scalar-local form. LocalMemory<int>(N) is safe across all
// backends per SpawnDev.ILGPU rc.10+.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// Batched ILGPU kernel for the AV1 16-point forward DCT (1D). Bit-exact
/// mirror of <see cref="Av1ForwardDct16.Transform"/>.
/// </summary>
public sealed class Av1ForwardDct16Kernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Av1ForwardDct16Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, int, int>(FdctKernel);
    }

    /// <summary>
    /// Run the FDCT on <paramref name="transformCount"/> independent
    /// 16-element transforms.
    /// </summary>
    public void Run(ArrayView<int> input, ArrayView<int> output, int transformCount, int cosBit = Av1ForwardDct16.DefaultCosBit)
    {
        if (transformCount < 0) throw new ArgumentOutOfRangeException(nameof(transformCount));
        if (transformCount == 0) return;
        if (cosBit < 10 || cosBit > 13) throw new ArgumentOutOfRangeException(nameof(cosBit), "must be in [10, 13]");
        if (input.Length < transformCount * 16L)
            throw new ArgumentException($"input must hold at least transformCount*16 ints (got {input.Length}).", nameof(input));
        if (output.Length < transformCount * 16L)
            throw new ArgumentException($"output must hold at least transformCount*16 ints (got {output.Length}).", nameof(output));
        _kernel(transformCount, input, output, transformCount, cosBit);
    }

    /// <summary>Kernel body. One thread per 16-element transform.</summary>
    private static void FdctKernel(
        Index1D transformIdx,
        ArrayView<int> input,
        ArrayView<int> output,
        int transformCount,
        int cosBit)
    {
        int idx = transformIdx;
        if (idx >= transformCount) return;
        long inBase = (long)idx * 16;
        long outBase = (long)idx * 16;

        // Resolve cospi indices used by fdct16: 4, 8, 12, 16, 20, 24,
        // 28, 32, 36, 40, 44, 48, 52, 56, 60.
        int c4, c8, c12, c16, c20, c24, c28, c32, c36, c40, c44, c48, c52, c56, c60;
        ResolveCospi(cosBit, out c4, out c8, out c12, out c16, out c20, out c24,
            out c28, out c32, out c36, out c40, out c44, out c48, out c52, out c56, out c60);

        // Per-thread scratch buffers a / b.
        var a = LocalMemory.Allocate<int>(16);
        var b = LocalMemory.Allocate<int>(16);

        // Stage 1
        a[0]  = input[inBase + 0]  + input[inBase + 15];
        a[1]  = input[inBase + 1]  + input[inBase + 14];
        a[2]  = input[inBase + 2]  + input[inBase + 13];
        a[3]  = input[inBase + 3]  + input[inBase + 12];
        a[4]  = input[inBase + 4]  + input[inBase + 11];
        a[5]  = input[inBase + 5]  + input[inBase + 10];
        a[6]  = input[inBase + 6]  + input[inBase + 9];
        a[7]  = input[inBase + 7]  + input[inBase + 8];
        a[8]  = -input[inBase + 8]  + input[inBase + 7];
        a[9]  = -input[inBase + 9]  + input[inBase + 6];
        a[10] = -input[inBase + 10] + input[inBase + 5];
        a[11] = -input[inBase + 11] + input[inBase + 4];
        a[12] = -input[inBase + 12] + input[inBase + 3];
        a[13] = -input[inBase + 13] + input[inBase + 2];
        a[14] = -input[inBase + 14] + input[inBase + 1];
        a[15] = -input[inBase + 15] + input[inBase + 0];

        // Stage 2
        b[0]  = a[0] + a[7];
        b[1]  = a[1] + a[6];
        b[2]  = a[2] + a[5];
        b[3]  = a[3] + a[4];
        b[4]  = -a[4] + a[3];
        b[5]  = -a[5] + a[2];
        b[6]  = -a[6] + a[1];
        b[7]  = -a[7] + a[0];
        b[8]  = a[8];
        b[9]  = a[9];
        b[10] = HalfBtf(-c32, a[10],  c32, a[13], cosBit);
        b[11] = HalfBtf(-c32, a[11],  c32, a[12], cosBit);
        b[12] = HalfBtf( c32, a[12],  c32, a[11], cosBit);
        b[13] = HalfBtf( c32, a[13],  c32, a[10], cosBit);
        b[14] = a[14];
        b[15] = a[15];

        // Stage 3
        a[0]  = b[0] + b[3];
        a[1]  = b[1] + b[2];
        a[2]  = -b[2] + b[1];
        a[3]  = -b[3] + b[0];
        a[4]  = b[4];
        a[5]  = HalfBtf(-c32, b[5],  c32, b[6], cosBit);
        a[6]  = HalfBtf( c32, b[6],  c32, b[5], cosBit);
        a[7]  = b[7];
        a[8]  = b[8] + b[11];
        a[9]  = b[9] + b[10];
        a[10] = -b[10] + b[9];
        a[11] = -b[11] + b[8];
        a[12] = -b[12] + b[15];
        a[13] = -b[13] + b[14];
        a[14] = b[14] + b[13];
        a[15] = b[15] + b[12];

        // Stage 4
        b[0]  = HalfBtf( c32, a[0],  c32, a[1], cosBit);
        b[1]  = HalfBtf(-c32, a[1],  c32, a[0], cosBit);
        b[2]  = HalfBtf( c48, a[2],  c16, a[3], cosBit);
        b[3]  = HalfBtf( c48, a[3], -c16, a[2], cosBit);
        b[4]  = a[4] + a[5];
        b[5]  = -a[5] + a[4];
        b[6]  = -a[6] + a[7];
        b[7]  = a[7] + a[6];
        b[8]  = a[8];
        b[9]  = HalfBtf(-c16, a[9],   c48, a[14], cosBit);
        b[10] = HalfBtf(-c48, a[10], -c16, a[13], cosBit);
        b[11] = a[11];
        b[12] = a[12];
        b[13] = HalfBtf( c48, a[13], -c16, a[10], cosBit);
        b[14] = HalfBtf( c16, a[14],  c48, a[9],  cosBit);
        b[15] = a[15];

        // Stage 5
        a[0]  = b[0];
        a[1]  = b[1];
        a[2]  = b[2];
        a[3]  = b[3];
        a[4]  = HalfBtf( c56, b[4],  c8,  b[7], cosBit);
        a[5]  = HalfBtf( c24, b[5],  c40, b[6], cosBit);
        a[6]  = HalfBtf( c24, b[6], -c40, b[5], cosBit);
        a[7]  = HalfBtf( c56, b[7], -c8,  b[4], cosBit);
        a[8]  = b[8] + b[9];
        a[9]  = -b[9] + b[8];
        a[10] = -b[10] + b[11];
        a[11] = b[11] + b[10];
        a[12] = b[12] + b[13];
        a[13] = -b[13] + b[12];
        a[14] = -b[14] + b[15];
        a[15] = b[15] + b[14];

        // Stage 6
        b[0]  = a[0]; b[1]  = a[1]; b[2]  = a[2]; b[3]  = a[3];
        b[4]  = a[4]; b[5]  = a[5]; b[6]  = a[6]; b[7]  = a[7];
        b[8]  = HalfBtf( c60, a[8],   c4,  a[15], cosBit);
        b[9]  = HalfBtf( c28, a[9],   c36, a[14], cosBit);
        b[10] = HalfBtf( c44, a[10],  c20, a[13], cosBit);
        b[11] = HalfBtf( c12, a[11],  c52, a[12], cosBit);
        b[12] = HalfBtf( c12, a[12], -c52, a[11], cosBit);
        b[13] = HalfBtf( c44, a[13], -c20, a[10], cosBit);
        b[14] = HalfBtf( c28, a[14], -c36, a[9],  cosBit);
        b[15] = HalfBtf( c60, a[15], -c4,  a[8],  cosBit);

        // Stage 7 (interleave / scatter)
        output[outBase + 0]  = b[0];
        output[outBase + 1]  = b[8];
        output[outBase + 2]  = b[4];
        output[outBase + 3]  = b[12];
        output[outBase + 4]  = b[2];
        output[outBase + 5]  = b[10];
        output[outBase + 6]  = b[6];
        output[outBase + 7]  = b[14];
        output[outBase + 8]  = b[1];
        output[outBase + 9]  = b[9];
        output[outBase + 10] = b[5];
        output[outBase + 11] = b[13];
        output[outBase + 12] = b[3];
        output[outBase + 13] = b[11];
        output[outBase + 14] = b[7];
        output[outBase + 15] = b[15];
    }

    /// <summary>libaom <c>half_btf</c>: kernel-safe variant.</summary>
    private static int HalfBtf(int w0, int in0, int w1, int in1, int bit)
    {
        long result = (long)w0 * in0 + (long)w1 * in1;
        result += 1L << (bit - 1);
        return (int)(result >> bit);
    }

    /// <summary>
    /// Resolves the 15 cospi entries fdct16 needs (every multiple of 4
    /// from 4..60). Inlined as branches so the kernel does not have to
    /// read a 64-element table buffer.
    /// </summary>
    private static void ResolveCospi(int cosBit,
        out int c4, out int c8, out int c12, out int c16, out int c20,
        out int c24, out int c28, out int c32, out int c36, out int c40,
        out int c44, out int c48, out int c52, out int c56, out int c60)
    {
        if (cosBit == 13)
        {
            c4 = 8153; c8 = 8035; c12 = 7839; c16 = 7568; c20 = 7225;
            c24 = 6811; c28 = 6333; c32 = 5793; c36 = 5197; c40 = 4551;
            c44 = 3862; c48 = 3135; c52 = 2378; c56 = 1598; c60 = 803;
        }
        else if (cosBit == 12)
        {
            c4 = 4076; c8 = 4017; c12 = 3920; c16 = 3784; c20 = 3612;
            c24 = 3406; c28 = 3166; c32 = 2896; c36 = 2598; c40 = 2276;
            c44 = 1931; c48 = 1567; c52 = 1189; c56 = 799;  c60 = 401;
        }
        else if (cosBit == 11)
        {
            c4 = 2038; c8 = 2009; c12 = 1960; c16 = 1892; c20 = 1806;
            c24 = 1703; c28 = 1583; c32 = 1448; c36 = 1299; c40 = 1138;
            c44 = 965;  c48 = 784;  c52 = 595;  c56 = 400;  c60 = 201;
        }
        else
        {
            c4 = 1019; c8 = 1004; c12 = 980;  c16 = 946;  c20 = 903;
            c24 = 851;  c28 = 792; c32 = 724;  c36 = 650;  c40 = 569;
            c44 = 483;  c48 = 392; c52 = 297;  c56 = 200;  c60 = 100;
        }
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped */ }
}
