// Tests for MdctKernel. Validates the ILGPU kernel's per-block output
// matches MdctReference within float-precision tolerance across every
// backend (CPU/CUDA/OpenCL/WebGPU/WebGL/Wasm).
//
// Tolerance: 1e-3f absolute. The CPU reference accumulates in double;
// the GPU kernel accumulates in float because most ILGPU backends
// emulate f64 (Dekker / native f64 only on CUDA/OpenCL/CPU). At
// N=960 (Opus CELT 20ms-48k) the kernel sums up to 1920 terms in
// float, putting expected absolute error well under 1e-3 even with
// inputs in [-1, 1]. Audio codecs operate at quantization granularity
// orders of magnitude looser than 1e-3 - this tolerance is a hard
// engineering ceiling, not a perceptual one.
//
// Sizes covered: N = 64, 128, 256, 960. 960 is Opus CELT 20ms-48k,
// the codec's primary frame size. 64..256 covers smaller blocks
// (CELT short windows / Vorbis short blocks).

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Transforms;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Maximum absolute delta tolerated between GPU kernel output and the
    /// double-precision CPU reference. Documented in the file header.
    /// </summary>
    private const float MdctTolerance = 1e-3f;

    [TestMethod]
    public async Task MdctKernel_ZeroInput_ProducesAllZeroOutput()
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new MdctKernel(acc);
            const int n = 64;
            var input = new float[2 * n];
            var output = new float[n];
            await kernel.RunAsync(input, output, blockCount: 1, n);
            for (int i = 0; i < n; i++)
                True(Math.Abs(output[i]) <= MdctTolerance,
                    $"zero-input MDCT must produce zero (within tolerance); idx {i} = {output[i]}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task MdctKernel_RandomInput_N64_MatchesCpuReference()
    {
        await RunMdctMatchesReferenceAsync(n: 64, blockCount: 4, seed: 0xC0DE);
    }

    [TestMethod]
    public async Task MdctKernel_RandomInput_N128_MatchesCpuReference()
    {
        await RunMdctMatchesReferenceAsync(n: 128, blockCount: 4, seed: 0xCAFE);
    }

    [TestMethod]
    public async Task MdctKernel_RandomInput_N256_MatchesCpuReference()
    {
        await RunMdctMatchesReferenceAsync(n: 256, blockCount: 4, seed: 0xBEEF);
    }

    [TestMethod]
    public async Task MdctKernel_RandomInput_N960_OpusCelt20ms_MatchesCpuReference()
    {
        // Opus CELT 20ms-48k frame size: N=480 -> 2N=960 input. Per the
        // task spec we use N=960 here (matching "Opus CELT 20ms-48k frame"
        // in the task's tested-N list); the reference + kernel are size-
        // agnostic so the test just exercises a large-N path.
        await RunMdctMatchesReferenceAsync(n: 960, blockCount: 2, seed: 0xFEED);
    }

    [TestMethod]
    public async Task MdctKernel_BatchedDispatch_AllBlocksMatchReference()
    {
        // Batched: 8 blocks at N=128 in one dispatch. Verifies block
        // routing inside the kernel - thread t = blockIdx*N + k must
        // pick the right input window per block.
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new MdctKernel(acc);
            const int n = 128;
            const int blockCount = 8;
            var rng = new Random(0xBA77ED);
            var input = new float[blockCount * 2 * n];
            for (int i = 0; i < input.Length; i++)
                input[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

            // CPU reference: process each block independently.
            var cpuOut = new float[blockCount * n];
            for (int b = 0; b < blockCount; b++)
            {
                MdctReference.Transform(
                    input.AsSpan(b * 2 * n, 2 * n),
                    cpuOut.AsSpan(b * n, n));
            }

            // GPU batched: one dispatch.
            var gpuOut = new float[blockCount * n];
            await kernel.RunAsync(input, gpuOut, blockCount, n);

            float maxAbsDelta = 0f;
            for (int i = 0; i < gpuOut.Length; i++)
            {
                float d = Math.Abs(gpuOut[i] - cpuOut[i]);
                if (d > maxAbsDelta) maxAbsDelta = d;
            }
            True(maxAbsDelta <= MdctTolerance,
                $"batched MDCT max abs delta = {maxAbsDelta}, tolerance = {MdctTolerance}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    /// <summary>
    /// Run the kernel for a single random block of size <paramref name="n"/>
    /// and assert each output coefficient agrees with the CPU reference
    /// within <see cref="MdctTolerance"/>.
    /// </summary>
    private async Task RunMdctMatchesReferenceAsync(int n, int blockCount, int seed)
    {
        var (ctx, acc) = await AcquireAcceleratorOrSkipAsync();
        try
        {
            using var kernel = new MdctKernel(acc);
            var rng = new Random(seed);
            var input = new float[blockCount * 2 * n];
            for (int i = 0; i < input.Length; i++)
                input[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

            var cpuOut = new float[blockCount * n];
            for (int b = 0; b < blockCount; b++)
            {
                MdctReference.Transform(
                    input.AsSpan(b * 2 * n, 2 * n),
                    cpuOut.AsSpan(b * n, n));
            }

            var gpuOut = new float[blockCount * n];
            await kernel.RunAsync(input, gpuOut, blockCount, n);

            float maxAbsDelta = 0f;
            int worstIdx = -1;
            for (int i = 0; i < gpuOut.Length; i++)
            {
                float d = Math.Abs(gpuOut[i] - cpuOut[i]);
                if (d > maxAbsDelta) { maxAbsDelta = d; worstIdx = i; }
            }
            True(maxAbsDelta <= MdctTolerance,
                $"MDCT N={n} block-count={blockCount}: max abs delta = {maxAbsDelta} at index {worstIdx} " +
                $"(cpu={cpuOut[Math.Max(worstIdx, 0)]}, gpu={gpuOut[Math.Max(worstIdx, 0)]}); tolerance = {MdctTolerance}");
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
