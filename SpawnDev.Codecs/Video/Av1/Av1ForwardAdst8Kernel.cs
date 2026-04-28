// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the AV1 8-point forward Asymmetric DST (1D).
// Bit-exact mirror of Av1ForwardAdst8.Transform - one thread per
// 8-element 1D ADST. Runs on every ILGPU backend.
//
// 7 stages with cospi-driven half_btf rotations + final scatter.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// Batched ILGPU kernel for the AV1 8-point forward ADST (1D). Bit-exact
/// mirror of <see cref="Av1ForwardAdst8.Transform"/>.
/// </summary>
public sealed class Av1ForwardAdst8Kernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Av1ForwardAdst8Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, int, int>(FadstKernel);
    }

    /// <summary>
    /// Run the FADST on <paramref name="transformCount"/> independent
    /// 8-element transforms.
    /// </summary>
    public void Run(ArrayView<int> input, ArrayView<int> output, int transformCount, int cosBit = Av1ForwardAdst8.DefaultCosBit)
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
    private static void FadstKernel(
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

        // Resolve cospi indices needed: 4, 12, 16, 20, 28, 32, 36, 44, 48, 52, 60.
        int c4, c12, c16, c20, c28, c32, c36, c44, c48, c52, c60;
        ResolveCospi(cosBit, out c4, out c12, out c16, out c20, out c28,
            out c32, out c36, out c44, out c48, out c52, out c60);

        // Stage 1: input remap with sign flips (bf1).
        int b10 =  input[inBase + 0];
        int b11 = -input[inBase + 7];
        int b12 = -input[inBase + 3];
        int b13 =  input[inBase + 4];
        int b14 = -input[inBase + 1];
        int b15 =  input[inBase + 6];
        int b16 =  input[inBase + 2];
        int b17 = -input[inBase + 5];

        // Stage 2: cospi[32] rotations on (2,3) and (6,7) - "step".
        int st0 = b10;
        int st1 = b11;
        int st2 = HalfBtf(c32, b12,  c32, b13, cosBit);
        int st3 = HalfBtf(c32, b12, -c32, b13, cosBit);
        int st4 = b14;
        int st5 = b15;
        int st6 = HalfBtf(c32, b16,  c32, b17, cosBit);
        int st7 = HalfBtf(c32, b16, -c32, b17, cosBit);

        // Stage 3: butterfly into bf1.
        int b20 = st0 + st2;
        int b21 = st1 + st3;
        int b22 = st0 - st2;
        int b23 = st1 - st3;
        int b24 = st4 + st6;
        int b25 = st5 + st7;
        int b26 = st4 - st6;
        int b27 = st5 - st7;

        // Stage 4: cospi[16/48] rotations on (4,5) and (6,7).
        int st20 = b20;
        int st21 = b21;
        int st22 = b22;
        int st23 = b23;
        int st24 = HalfBtf( c16, b24,  c48, b25, cosBit);
        int st25 = HalfBtf( c48, b24, -c16, b25, cosBit);
        int st26 = HalfBtf(-c48, b26,  c16, b27, cosBit);
        int st27 = HalfBtf( c16, b26,  c48, b27, cosBit);

        // Stage 5: butterfly across halves.
        int b30 = st20 + st24;
        int b31 = st21 + st25;
        int b32 = st22 + st26;
        int b33 = st23 + st27;
        int b34 = st20 - st24;
        int b35 = st21 - st25;
        int b36 = st22 - st26;
        int b37 = st23 - st27;

        // Stage 6: cospi[4/60/20/44/36/28/52/12] rotations.
        int sf0 = HalfBtf( c4,  b30,  c60, b31, cosBit);
        int sf1 = HalfBtf( c60, b30, -c4,  b31, cosBit);
        int sf2 = HalfBtf( c20, b32,  c44, b33, cosBit);
        int sf3 = HalfBtf( c44, b32, -c20, b33, cosBit);
        int sf4 = HalfBtf( c36, b34,  c28, b35, cosBit);
        int sf5 = HalfBtf( c28, b34, -c36, b35, cosBit);
        int sf6 = HalfBtf( c52, b36,  c12, b37, cosBit);
        int sf7 = HalfBtf( c12, b36, -c52, b37, cosBit);

        // Stage 7: final scatter to output (libaom permutation).
        output[outBase + 0] = sf1;
        output[outBase + 1] = sf6;
        output[outBase + 2] = sf3;
        output[outBase + 3] = sf4;
        output[outBase + 4] = sf5;
        output[outBase + 5] = sf2;
        output[outBase + 6] = sf7;
        output[outBase + 7] = sf0;
    }

    /// <summary>libaom <c>half_btf</c>: kernel-safe variant.</summary>
    private static int HalfBtf(int w0, int in0, int w1, int in1, int bit)
    {
        long result = (long)w0 * in0 + (long)w1 * in1;
        result += 1L << (bit - 1);
        return (int)(result >> bit);
    }

    /// <summary>Resolves the 11 cospi entries fadst8 needs.</summary>
    private static void ResolveCospi(int cosBit,
        out int c4, out int c12, out int c16, out int c20, out int c28,
        out int c32, out int c36, out int c44, out int c48, out int c52,
        out int c60)
    {
        if (cosBit == 13)
        {
            c4 = 8153; c12 = 7839; c16 = 7568; c20 = 7225; c28 = 6333;
            c32 = 5793; c36 = 5197; c44 = 3862; c48 = 3135; c52 = 2378; c60 = 803;
        }
        else if (cosBit == 12)
        {
            c4 = 4076; c12 = 3920; c16 = 3784; c20 = 3612; c28 = 3166;
            c32 = 2896; c36 = 2598; c44 = 1931; c48 = 1567; c52 = 1189; c60 = 401;
        }
        else if (cosBit == 11)
        {
            c4 = 2038; c12 = 1960; c16 = 1892; c20 = 1806; c28 = 1583;
            c32 = 1448; c36 = 1299; c44 = 965;  c48 = 784;  c52 = 595;  c60 = 201;
        }
        else
        {
            c4 = 1019; c12 = 980;  c16 = 946;  c20 = 903;  c28 = 792;
            c32 = 724;  c36 = 650;  c44 = 483;  c48 = 392;  c52 = 297;  c60 = 100;
        }
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped */ }
}
