// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the AV1 4-point forward Asymmetric DST (1D).
// Bit-exact mirror of Av1ForwardAdst4.Transform - one thread per
// 4-element 1D ADST. Runs on every ILGPU backend.
//
// Uses sinpi constants (5 entries per cos_bit) instead of cospi.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// Batched ILGPU kernel for the AV1 4-point forward ADST (1D). Bit-exact
/// mirror of <see cref="Av1ForwardAdst4.Transform"/>.
/// </summary>
public sealed class Av1ForwardAdst4Kernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Av1ForwardAdst4Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, int, int>(FadstKernel);
    }

    /// <summary>
    /// Run the FADST on <paramref name="transformCount"/> independent
    /// 4-element transforms.
    /// </summary>
    public void Run(ArrayView<int> input, ArrayView<int> output, int transformCount, int cosBit = Av1ForwardAdst4.DefaultCosBit)
    {
        if (transformCount < 0) throw new ArgumentOutOfRangeException(nameof(transformCount));
        if (transformCount == 0) return;
        if (cosBit < 10 || cosBit > 13) throw new ArgumentOutOfRangeException(nameof(cosBit), "must be in [10, 13]");
        if (input.Length < transformCount * 4L)
            throw new ArgumentException($"input must hold at least transformCount*4 ints (got {input.Length}).", nameof(input));
        if (output.Length < transformCount * 4L)
            throw new ArgumentException($"output must hold at least transformCount*4 ints (got {output.Length}).", nameof(output));
        _kernel(transformCount, input, output, transformCount, cosBit);
    }

    /// <summary>Kernel body. One thread per 4-element transform.</summary>
    private static void FadstKernel(
        Index1D transformIdx,
        ArrayView<int> input,
        ArrayView<int> output,
        int transformCount,
        int cosBit)
    {
        int idx = transformIdx;
        if (idx >= transformCount) return;
        long inBase = (long)idx * 4;
        long outBase = (long)idx * 4;

        // Resolve 5 sinpi entries (sinpi[0]=0 always).
        int sp1, sp2, sp3, sp4;
        ResolveSinpi(cosBit, out sp1, out sp2, out sp3, out sp4);

        int x0 = input[inBase + 0];
        int x1 = input[inBase + 1];
        int x2 = input[inBase + 2];
        int x3 = input[inBase + 3];

        // Early-out for the all-zero input case (matches CPU reference).
        if ((x0 | x1 | x2 | x3) == 0)
        {
            output[outBase + 0] = 0;
            output[outBase + 1] = 0;
            output[outBase + 2] = 0;
            output[outBase + 3] = 0;
            return;
        }

        // Stage 1: long-precision multiplications (matches CPU reference).
        long s0 = (long)sp1 * x0;
        long s1 = (long)sp4 * x0;
        long s2 = (long)sp2 * x1;
        long s3 = (long)sp1 * x1;
        long s4 = (long)sp3 * x2;
        long s5 = (long)sp4 * x3;
        long s6 = (long)sp2 * x3;
        long s7 = x0 + x1;

        // Stage 2
        s7 = s7 - x3;

        // Stage 3
        long y0 = s0 + s2;
        long y1 = (long)sp3 * s7;
        long y2 = s1 - s3;
        long y3 = s4;

        // Stage 4
        y0 += s5;
        y2 += s6;

        // Stage 5
        long t0 = y0 + y3;
        long t1 = y1;
        long t2 = y2 - y3;
        long t3 = y2 - y0;

        // Stage 6
        t3 += y3;

        // 1-D ADST scaling: round_shift by cos_bit.
        output[outBase + 0] = RoundShift(t0, cosBit);
        output[outBase + 1] = RoundShift(t1, cosBit);
        output[outBase + 2] = RoundShift(t2, cosBit);
        output[outBase + 3] = RoundShift(t3, cosBit);
    }

    /// <summary>libaom round_shift: (value + (1 << (bit-1))) >> bit.</summary>
    private static int RoundShift(long value, int bit)
    {
        return (int)((value + (1L << (bit - 1))) >> bit);
    }

    /// <summary>
    /// Resolves the 4 non-zero sinpi entries (sp1, sp2, sp3, sp4) per
    /// cos_bit. sinpi[0] is always 0, so it's not stored.
    /// </summary>
    private static void ResolveSinpi(int cosBit,
        out int sp1, out int sp2, out int sp3, out int sp4)
    {
        if (cosBit == 13)      { sp1 = 2642; sp2 = 4964; sp3 = 6688; sp4 = 7606; }
        else if (cosBit == 12) { sp1 = 1321; sp2 = 2482; sp3 = 3344; sp4 = 3803; }
        else if (cosBit == 11) { sp1 = 660;  sp2 = 1241; sp3 = 1672; sp4 = 1901; }
        else                   { sp1 = 330;  sp2 = 621;  sp3 = 836;  sp4 = 951; }
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped */ }
}
