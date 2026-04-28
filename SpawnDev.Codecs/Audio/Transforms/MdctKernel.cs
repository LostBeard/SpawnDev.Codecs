// SpawnDev.Codecs is licensed under MIT (see LICENSE.txt).
//
// ILGPU kernel for the audio Modified Discrete Cosine Transform.
// Embarrassingly parallel: one thread per output coefficient,
// per block. Runs the same direct O(N^2) formula as
// MdctReference.Transform on every ILGPU backend - CPU, CUDA,
// OpenCL, WebGPU, WebGL, Wasm.
//
// Layout (block-major):
//   input  : blockCount * 2N floats. Block b at offset b*2N.
//   output : blockCount * N   floats. Block b at offset b*N.
//
// One macroframe of an audio codec runs MANY MDCTs in parallel:
//   * Vorbis short blocks at N=128 (256 samples), long at N=1024 (2048).
//   * Opus CELT 20ms-48k frames at N=480 (960 samples).
//   * AAC long blocks at N=1024.
// At Vorbis-long N=1024 the GPU runs N=1024 threads per block; with
// 1000 blocks/frame that's 1.024M threads in one dispatch - well into
// the regime where the GPU outperforms even SIMD CPU.
//
// Math (matches MdctReference exactly, modulo float vs double accum):
//   factor = pi / N
//   halfN  = N / 2
//   X[k]   = sum_{idx=0..2N-1} x[idx] * cos(factor * (idx + 0.5 + halfN) * (k + 0.5))
//
// PRECISION STRATEGY:
//
// Naive float math fails the 1e-3 tolerance at large N because the
// raw angle (idx + 0.5 + halfN) * (k + 0.5) * pi/N grows without
// bound (~pi*N at idx,k near N). At N=960 the angle reaches ~4500
// radians; cosine of a value that large in float32 has an absolute
// error of ~5e-4 per term, accumulating to ~0.02 over 1920 terms.
//
// The fix is exact integer argument reduction, then float cosine on
// a reduced angle in [0, 2*pi):
//
//   theta = pi/N * (idx + 0.5 + N/2) * (k + 0.5)
//         = pi/(4N) * (2*idx + 1 + N) * (2*k + 1)
//         = pi/(4N) * m * q                       (m, q exact integers)
//
// Cosine has period 2*pi, so cos(theta) = cos(theta - 2*pi*floor(theta/(2*pi))).
// In our parameterisation, theta - 2*pi*j corresponds to subtracting
// 8*N from m*q (because 2*pi = pi/(4N) * 8N). So we can reduce m*q
// modulo 8*N as exact long arithmetic, then convert to float:
//
//   reducedMQ = (long)m * q % (8L * N)            // exact, >= 0
//   thetaReduced = (pi/(4N)) * reducedMQ          // float; in [0, 2*pi)
//   cos(thetaReduced)                              // float; ~1ulp accurate
//
// This keeps each cosine evaluation accurate to ~1e-7 relative on
// every ILGPU backend (GPU cos() is fine for small arguments). The
// resulting float-accumulator sum stays well under 1e-3 absolute
// across all tested sizes through Opus CELT 20ms-48k (N=960).

using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnDev.Codecs.Audio.Transforms;

/// <summary>
/// Batched ILGPU kernel that runs the forward Modified DCT across N
/// independent blocks in parallel. Bit-exact-equivalent (within
/// float-precision tolerance) to <see cref="MdctReference"/>. Thread
/// granularity is per-output-coefficient: kernel launch is
/// <c>blockCount * N</c> threads.
/// </summary>
public sealed class MdctKernel : IDisposable
{
    private readonly Accelerator _accelerator;
    private readonly Action<Index1D, ArrayView<float>, ArrayView<float>, int, int> _kernel;

    /// <summary>Compile the kernel onto <paramref name="accelerator"/>.</summary>
    public MdctKernel(Accelerator accelerator)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        _accelerator = accelerator;
        _kernel = accelerator.LoadAutoGroupedStreamKernel<
            Index1D, ArrayView<float>, ArrayView<float>, int, int>(MdctKernelBody);
    }

    /// <summary>
    /// Run the MDCT on <paramref name="blockCount"/> blocks of size <paramref name="n"/>.
    /// <paramref name="input"/> holds <c>blockCount * 2N</c> floats (block-major);
    /// <paramref name="output"/> receives <c>blockCount * N</c> floats.
    /// </summary>
    public void Run(ArrayView<float> input, ArrayView<float> output, int blockCount, int n)
    {
        if (blockCount < 0) throw new ArgumentOutOfRangeException(nameof(blockCount));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Block size N must be positive.");
        if (blockCount == 0) return;
        long inLen = (long)blockCount * 2L * n;
        long outLen = (long)blockCount * n;
        if (input.Length < inLen)
            throw new ArgumentException(
                $"input must hold at least blockCount*2N floats (got {input.Length}, need {inLen}).",
                nameof(input));
        if (output.Length < outLen)
            throw new ArgumentException(
                $"output must hold at least blockCount*N floats (got {output.Length}, need {outLen}).",
                nameof(output));
        // One thread per output coefficient.
        int totalThreads = checked(blockCount * n);
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
        long inLen = (long)blockCount * 2L * n;
        long outLen = (long)blockCount * n;
        using var dIn = _accelerator.Allocate1D<float>(inLen);
        using var dOut = _accelerator.Allocate1D<float>(outLen);
        dIn.View.CopyFromCPU(input.Span.ToArray());
        Run(dIn.View, dOut.View, blockCount, n);
        await _accelerator.SynchronizeAsync();
        var readBack = await dOut.CopyToHostAsync();
        readBack.AsSpan(0, output.Length).CopyTo(output.Span);
    }

    /// <summary>
    /// Kernel body. One thread per output coefficient.
    /// Thread t = blockIdx * N + k, where k is the coefficient index
    /// inside its block.
    /// </summary>
    private static void MdctKernelBody(
        Index1D threadIdx,
        ArrayView<float> input,
        ArrayView<float> output,
        int blockCount,
        int n)
    {
        int t = threadIdx;
        int total = blockCount * n;
        if (t >= total) return;

        int blockIdx = t / n;
        int k = t - blockIdx * n;

        long inBase = (long)blockIdx * 2L * n;
        long outBase = (long)blockIdx * n;

        // Argument reduction (see file header). Express the angle as
        //   theta = pi/(4N) * m * q
        // with m = 2*idx + 1 + N and q = 2*k + 1 exact integers, then
        // reduce m*q modulo 8N (one full period of cosine) before
        // converting to float. We additionally fold residue into the
        // half-period [-4N, 4N) so the float angle passed to cos()
        // sits in [-pi, pi); WGSL's cos() is well-defined there
        // (UNDEFINED behavior outside that range per the WebGPU spec)
        // and every other backend's cos() is also tightest in this
        // range. Without this reduction the float angle accumulates
        // ~5e-4 absolute error at N=960 which blows past the 1e-3 test
        // tolerance after summing 1920 terms.
        long modulus = 8L * n;
        long halfModulus = 4L * n;                  // 4N (half cosine period in residue units)
        long q = 2L * k + 1L;
        // step in m between successive idx values is 2; precompute the
        // residue increment (2*q mod modulus). For our valid inputs
        // 2q < 8N always, so the modulo is a no-op, but the form
        // generalises to any future larger-N use.
        long deltaResidue = (2L * q) % modulus;
        long m0 = 1L + n;                          // m at idx=0 = 2*0 + 1 + N
        long residue = (m0 * q) % modulus;          // (m0 * q) mod 8N, in [0, 8N)
        float invFourN = XMath.PI / (4.0f * n);     // pi/(4N)
        float acc = 0.0f;
        for (int idx = 0; idx < 2 * n; idx++)
        {
            // Fold residue from [0, 8N) into [-4N, 4N) so the float
            // angle is in [-pi, pi). Explicit (float) cast on the
            // long->float conversion for unambiguous codegen across
            // all 6 backends.
            long folded = residue >= halfModulus ? residue - modulus : residue;
            float thetaReduced = invFourN * (float)folded;
            acc += input[inBase + idx] * XMath.Cos(thetaReduced);
            residue += deltaResidue;
            if (residue >= modulus) residue -= modulus;
        }
        output[outBase + k] = acc;
    }

    /// <summary>Release kernel resources. Does NOT dispose the accelerator.</summary>
    public void Dispose() { /* auto-grouped kernels don't need explicit disposal */ }
}
