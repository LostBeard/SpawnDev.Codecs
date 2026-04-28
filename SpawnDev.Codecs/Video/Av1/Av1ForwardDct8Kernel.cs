// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the AV1 8-point forward DCT (1D). Bit-exact mirror
// of Av1ForwardDct8.Transform - one thread per 8-element 1D transform.
// Runs on every ILGPU backend.
//
// 5 stages of butterfly + cospi multiplications + final interleave.
// Implementation uses scalar locals only (no LocalMemory) - 8 elements
// fit comfortably in registers across every backend.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// Batched ILGPU kernel for the AV1 8-point forward DCT (1D). Bit-exact
/// mirror of <see cref="Av1ForwardDct8.Transform"/>.
/// </summary>
public sealed class Av1ForwardDct8Kernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Av1ForwardDct8Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, int, int>(FdctKernel);
    }

    /// <summary>
    /// Run the FDCT on <paramref name="transformCount"/> independent
    /// 8-element transforms.
    /// </summary>
    public void Run(ArrayView<int> input, ArrayView<int> output, int transformCount, int cosBit = Av1ForwardDct8.DefaultCosBit)
    {
        if (transformCount < 0) throw new ArgumentOutOfRangeException(nameof(transformCount));
        if (transformCount == 0) return;
        if (cosBit < 10 || cosBit > 13) throw new ArgumentOutOfRangeException(nameof(cosBit), "must be in [10, 13]");
        if (input.Length < transformCount * 8L)
            throw new ArgumentException($"input must hold at least transformCount*8 ints (got {input.Length}).", nameof(input));
        if (output.Length < transformCount * 8L)
            throw new ArgumentException($"output must hold at least transformCount*8 ints (got {output.Length}).", nameof(output));
        _kernel(transformCount, input, output, transformCount, cosBit);
    }

    /// <summary>Kernel body. One thread per 8-element transform.</summary>
    private static void FdctKernel(
        Index1D transformIdx,
        ArrayView<int> input,
        ArrayView<int> output,
        int transformCount,
        int cosBit)
    {
        int idx = transformIdx;
        if (idx >= transformCount) return;
        long inBase = (long)idx * 8;
        long outBase = (long)idx * 8;

        // Resolve cospi indices used by fdct8: 8, 16, 24, 32, 40, 48, 56.
        int c8, c16, c24, c32, c40, c48, c56;
        ResolveCospi(cosBit, out c8, out c16, out c24, out c32, out c40, out c48, out c56);

        int in0 = input[inBase + 0];
        int in1 = input[inBase + 1];
        int in2 = input[inBase + 2];
        int in3 = input[inBase + 3];
        int in4 = input[inBase + 4];
        int in5 = input[inBase + 5];
        int in6 = input[inBase + 6];
        int in7 = input[inBase + 7];

        // Stage 1
        int s10 =  in0 + in7;
        int s11 =  in1 + in6;
        int s12 =  in2 + in5;
        int s13 =  in3 + in4;
        int s14 = -in4 + in3;
        int s15 = -in5 + in2;
        int s16 = -in6 + in1;
        int s17 = -in7 + in0;

        // Stage 2
        int s20 = s10 + s13;
        int s21 = s11 + s12;
        int s22 = -s12 + s11;
        int s23 = -s13 + s10;
        int s24 = s14;
        int s25 = HalfBtf(-c32, s15,  c32, s16, cosBit);
        int s26 = HalfBtf( c32, s16,  c32, s15, cosBit);
        int s27 = s17;

        // Stage 3
        int s30 = HalfBtf( c32, s20,  c32, s21, cosBit);
        int s31 = HalfBtf(-c32, s21,  c32, s20, cosBit);
        int s32 = HalfBtf( c48, s22,  c16, s23, cosBit);
        int s33 = HalfBtf( c48, s23, -c16, s22, cosBit);
        int s34 = s24 + s25;
        int s35 = -s25 + s24;
        int s36 = -s26 + s27;
        int s37 = s27 + s26;

        // Stage 4
        int s40 = s30;
        int s41 = s31;
        int s42 = s32;
        int s43 = s33;
        int s44 = HalfBtf( c56, s34,  c8,  s37, cosBit);
        int s45 = HalfBtf( c24, s35,  c40, s36, cosBit);
        int s46 = HalfBtf( c24, s36, -c40, s35, cosBit);
        int s47 = HalfBtf( c56, s37, -c8,  s34, cosBit);

        // Stage 5 (interleave)
        output[outBase + 0] = s40;
        output[outBase + 1] = s44;
        output[outBase + 2] = s42;
        output[outBase + 3] = s46;
        output[outBase + 4] = s41;
        output[outBase + 5] = s45;
        output[outBase + 6] = s43;
        output[outBase + 7] = s47;
    }

    /// <summary>libaom <c>half_btf</c>: kernel-safe variant.</summary>
    private static int HalfBtf(int w0, int in0, int w1, int in1, int bit)
    {
        long result = (long)w0 * in0 + (long)w1 * in1;
        result += 1L << (bit - 1);
        return (int)(result >> bit);
    }

    /// <summary>
    /// Resolves the 7 cospi entries fdct8 needs (8, 16, 24, 32, 40, 48,
    /// 56) per cos_bit. Inlined as branches so the kernel does not have
    /// to read a 64-element table buffer.
    /// </summary>
    private static void ResolveCospi(int cosBit,
        out int c8, out int c16, out int c24, out int c32,
        out int c40, out int c48, out int c56)
    {
        if (cosBit == 13)      { c8 = 8035; c16 = 7568; c24 = 6811; c32 = 5793; c40 = 4551; c48 = 3035; c56 = 1598; }
        else if (cosBit == 12) { c8 = 4017; c16 = 3784; c24 = 3406; c32 = 2896; c40 = 2276; c48 = 1567; c56 = 799; }
        else if (cosBit == 11) { c8 = 2009; c16 = 1892; c24 = 1703; c32 = 1448; c40 = 1138; c48 = 784;  c56 = 400; }
        else                   { c8 = 1004; c16 = 946;  c24 = 851;  c32 = 724;  c40 = 569;  c48 = 392;  c56 = 200; }
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped */ }
}
