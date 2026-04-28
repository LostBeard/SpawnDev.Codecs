// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the AV1 4-point forward DCT (1D). Bit-exact mirror
// of Av1ForwardDct4.Transform - one thread per 4-element 1D transform.
// Runs on every ILGPU backend (CPU emulator, CUDA, OpenCL, WebGPU,
// WebGL, Wasm). Batched: N transforms in parallel, each thread reads
// its own 4 ints and writes its own 4 ints.
//
// AV1 forward 1D primitives operate on 32-bit ints (libaom precision).
// The 2D transform composes column 1D + row 1D + per-axis shift, but
// this kernel is just the 1D building block. Tests verify bit-exact
// agreement with Av1ForwardDct4.Transform across (a) zero, (b)
// DC-only / structured input, (c) random batches.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Video.Av1;

/// <summary>
/// Batched ILGPU kernel for the AV1 4-point forward DCT (1D). Bit-exact
/// mirror of <see cref="Av1ForwardDct4.Transform"/>.
/// </summary>
public sealed class Av1ForwardDct4Kernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<int>, ArrayView<int>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public Av1ForwardDct4Kernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<int>, ArrayView<int>, int, int>(FdctKernel);
    }

    /// <summary>
    /// Run the FDCT on <paramref name="transformCount"/> independent
    /// 4-element transforms. Each transform occupies 4 contiguous ints
    /// in <paramref name="input"/> and <paramref name="output"/>.
    /// </summary>
    /// <param name="input">Input transforms, layout: transform-major (4 ints per transform).</param>
    /// <param name="output">Output coefficients, same layout as input.</param>
    /// <param name="transformCount">Number of 4-element transforms to run.</param>
    /// <param name="cosBit">Cosine precision bits (10..13). Defaults to AV1 default 13.</param>
    public void Run(ArrayView<int> input, ArrayView<int> output, int transformCount, int cosBit = Av1ForwardDct4.DefaultCosBit)
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
    private static void FdctKernel(
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

        // Pull cospi[16], cospi[32], cospi[48] into thread-local ints.
        // These are the only cospi indices the 4-point fdct needs.
        // Inline the values per cos_bit so the kernel doesn't read a
        // global 64-element table on every dispatch.
        int c16, c32, c48;
        ResolveCospi(cosBit, out c16, out c32, out c48);

        int in0 = input[inBase + 0];
        int in1 = input[inBase + 1];
        int in2 = input[inBase + 2];
        int in3 = input[inBase + 3];

        // Stage 1
        int s0 = in0 + in3;
        int s1 = in1 + in2;
        int s2 = -in2 + in1;
        int s3 = -in3 + in0;

        // Stage 2
        int t0 = HalfBtf( c32, s0,  c32, s1, cosBit);
        int t1 = HalfBtf(-c32, s1,  c32, s0, cosBit);
        int t2 = HalfBtf( c48, s2,  c16, s3, cosBit);
        int t3 = HalfBtf( c48, s3, -c16, s2, cosBit);

        // Stage 3 (interleave)
        output[outBase + 0] = t0;
        output[outBase + 1] = t2;
        output[outBase + 2] = t1;
        output[outBase + 3] = t3;
    }

    /// <summary>
    /// libaom <c>half_btf(w0, in0, w1, in1, bit) = round_shift(w0*in0 + w1*in1, bit)</c>.
    /// Kernel-safe variant of <see cref="Av1ForwardDct4.HalfBtf"/>.
    /// </summary>
    private static int HalfBtf(int w0, int in0, int w1, int in1, int bit)
    {
        long result = (long)w0 * in0 + (long)w1 * in1;
        result += 1L << (bit - 1);
        return (int)(result >> bit);
    }

    /// <summary>
    /// Resolves the three cospi entries (16, 32, 48) the 4-point fdct
    /// needs, per cos_bit. Inlined as branches so the kernel doesn't
    /// have to allocate or read a 64-element table buffer.
    /// </summary>
    private static void ResolveCospi(int cosBit, out int c16, out int c32, out int c48)
    {
        if (cosBit == 13)      { c16 =  7568; c32 =  5793; c48 =  3035; }
        else if (cosBit == 12) { c16 =  3784; c32 =  2896; c48 =  1567; }
        else if (cosBit == 11) { c16 =  1892; c32 =  1448; c48 =   784; }
        else                   { c16 =   946; c32 =   724; c48 =   392; }
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped */ }
}
