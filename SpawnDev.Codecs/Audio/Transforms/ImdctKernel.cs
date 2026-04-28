// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the audio Inverse Modified Discrete Cosine
// Transform. Embarrassingly parallel: one thread per output sample,
// per block. Runs the same direct O(N^2) formula as
// ImdctReference.Transform on every ILGPU backend - CPU, CUDA,
// OpenCL, WebGPU, WebGL, Wasm.
//
// Layout (block-major):
//   input  : blockCount * N   floats. Block b at offset b*N.
//   output : blockCount * 2N  floats. Block b at offset b*2N.
//
// One macroframe of an audio decoder runs MANY IMDCTs in parallel:
//   * Vorbis short blocks at N=128 (256 samples), long at N=1024 (2048).
//   * Opus CELT 20ms-48k frames at N=480 (960 samples).
//   * AAC long blocks at N=1024.
// At Vorbis-long N=1024, 1000 blocks/frame produces ~2.05M output
// threads in one dispatch - GPU regime.
//
// Math (matches ImdctReference exactly, modulo float vs double accum):
//   factor = pi / N
//   halfN  = N / 2
//   y[idx] = sum_{k=0..N-1} X[k] * cos(factor * (idx + 0.5 + halfN) * (k + 0.5))
//
// PRECISION STRATEGY:
//
// Naive float math fails the 1e-3 tolerance at large N because the
// raw angle grows without bound. At N=960 the angle reaches ~4500
// radians; cosine of a value that large in float32 has an absolute
// error of ~5e-4 per term, accumulating past 1e-3 over the sum.
//
// The fix is exact integer argument reduction, then float cosine on
// a reduced angle in [0, 2*pi):
//
//   theta = pi/N * (idx + 0.5 + N/2) * (k + 0.5)
//         = pi/(4N) * (2*idx + 1 + N) * (2*k + 1)
//         = pi/(4N) * m * q                       (m, q exact integers)
//
// Cosine has period 2*pi = pi/(4N) * 8N, so cos(theta) is invariant
// under m*q -> m*q mod 8N. We do that reduction in exact long
// arithmetic, then convert to float once per term. Each cosine call
// then sees an angle in [0, 2*pi) where float32 gives ~1e-7 relative
// accuracy on every backend (GPU cos() is fine for small arguments).

using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Transforms;

/// <summary>
/// Batched ILGPU kernel that runs the inverse Modified DCT across N
/// independent blocks in parallel. Bit-exact-equivalent (within
/// float-precision tolerance) to <see cref="ImdctReference"/>. Thread
/// granularity is per-output-sample: kernel launch is
/// <c>blockCount * 2N</c> threads.
/// </summary>
public sealed class ImdctKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public ImdctKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int, int>(ImdctKernelBody);
    }

    /// <summary>
    /// Run the IMDCT on <paramref name="blockCount"/> blocks of size <paramref name="n"/>.
    /// <paramref name="input"/> holds <c>blockCount * N</c> floats (block-major);
    /// <paramref name="output"/> receives <c>blockCount * 2N</c> floats.
    /// </summary>
    public void Run(ArrayView<float> input, ArrayView<float> output, int blockCount, int n)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Block size N must be positive.");
        if (blockCount == 0) return;
        long inLen = (long)blockCount * n;
        long outLen = (long)blockCount * 2L * n;
        if (input.Length < inLen)
            throw new ArgumentException(
                $"input must hold at least blockCount*N floats (got {input.Length}, need {inLen}).",
                nameof(input));
        if (output.Length < outLen)
            throw new ArgumentException(
                $"output must hold at least blockCount*2N floats (got {output.Length}, need {outLen}).",
                nameof(output));
        // One thread per output sample. Total = blockCount * 2N.
        int totalThreads = checked(blockCount * 2 * n);
        _kernel(totalThreads, input, output, blockCount, n);
    }

    /// <summary>
    /// Convenience: allocate temporary GPU buffers, run, copy back.
    /// Async because WebGPU forbids synchronous GPU-to-CPU copies.
    /// </summary>
    public async Task RunAsync(
        ReadOnlyMemory<float> input, Memory<float> output, int blockCount, int n)
    {
        if (blockCount <= 0) return;
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        long inLen = (long)blockCount * n;
        long outLen = (long)blockCount * 2L * n;
        using var dIn = _accelerator.Allocate1D<float>(inLen);
        using var dOut = _accelerator.Allocate1D<float>(outLen);
        dIn.View.CopyFromCPU(input.Span.ToArray());
        Run(dIn.View, dOut.View, blockCount, n);
        await _accelerator.SynchronizeAsync();
        var readBack = await dOut.CopyToHostAsync();
        readBack.AsSpan(0, output.Length).CopyTo(output.Span);
    }

    /// <summary>
    /// Kernel body. One thread per output sample.
    /// Thread t = blockIdx * 2N + idx, where idx is the time-domain
    /// sample index inside its block.
    /// </summary>
    private static void ImdctKernelBody(
        Index1D threadIdx,
        ArrayView<float> input,
        ArrayView<float> output,
        int blockCount,
        int n)
    {
        int t = threadIdx;
        int twoN = 2 * n;
        int total = blockCount * twoN;
        if (t >= total) return;

        int blockIdx = t / twoN;
        int idx = t - blockIdx * twoN;

        long inBase = (long)blockIdx * n;
        long outBase = (long)blockIdx * twoN;

        // Argument reduction (see file header). Express the angle as
        //   theta = pi/(4N) * m * q
        // with m = 2*idx + 1 + N and q = 2*k + 1 exact integers, then
        // reduce m*q modulo 8N before converting to float. We
        // additionally fold residue into [-4N, 4N) so the float angle
        // passed to cos() is in [-pi, pi); WGSL's cos() is defined to
        // be 4-ULP accurate there and UNDEFINED outside it (WebGPU
        // spec). All other backends' cos() are tightest in this range
        // too.
        long modulus = 8L * n;
        long halfModulus = 4L * n;                  // 4N (half cosine period in residue units)
        long m = 2L * idx + 1L + n;
        // step in q between successive k values is 2; precompute
        //   delta = 2*m mod 8N
        // For our valid input ranges m <= 4N-1 so 2m < 8N and the
        // modulo is a no-op; we keep the form for generality.
        long deltaResidue = (2L * m) % modulus;
        long q0 = 1L;                              // q at k=0 = 1
        long residue = (m * q0) % modulus;          // (m * q0) mod 8N, in [0, 8N)
        float invFourN = XMath.PI / (4.0f * n);     // pi/(4N)
        float acc = 0.0f;
        for (int k = 0; k < n; k++)
        {
            // Fold residue from [0, 8N) into [-4N, 4N) so the float
            // angle is in [-pi, pi). Explicit (float) cast on the
            // long->float conversion for unambiguous codegen across
            // all 6 backends.
            long folded = residue >= halfModulus ? residue - modulus : residue;
            float thetaReduced = invFourN * (float)folded;
            acc += input[inBase + k] * XMath.Cos(thetaReduced);
            residue += deltaResidue;
            if (residue >= modulus) residue -= modulus;
        }
        output[outBase + idx] = acc;
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped kernels don't need explicit disposal */ }
}
