// Cross-backend tests for Vp9Idct16x16Kernel. GPU-side verification
// via Vp9TestVerify.CountByteMismatches - no full-buffer CPU readback,
// no per-byte CPU compare loop.
//
// Backend coverage status (2026-04-25, ILGPU 4.9.2-rc.12):
//   - CPU / CUDA / OpenCL:  PASS bit-exact
//   - WebGL:                runner-level NotSupportedException (sub-word
//                           writes lower to atomic RMW, no atomics on WebGL)
//   - WebGPU:               TIMEOUT (~30s+ shader compile time on Chrome's
//                           WGSL validator for the inline 7-stage 16-point
//                           butterfly; investigation filed to Geordi)
//   - Wasm:                 BIT-EXACT DIVERGENCE (~24 of 256 bytes differ vs
//                           CPU reference; CPU/CUDA/OpenCL agree so the
//                           butterfly logic is correct; Wasm codegen has a
//                           specific edge case; investigation filed to Geordi)
//
// The WebGPU + Wasm failures are intentionally left red as regression
// oracles - they will go green automatically when Geordi ships the ILGPU
// fixes. Same pattern as ILGPU's own LocalMemoryRepro_Int256/1024 tests
// that stayed red until rc.12.

using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>
    /// 16x16 transform topology has 256 `flat out` varyings per thread,
    /// which exceeds GL_MAX_VARYING_VECTORS on most WebGL implementations
    /// (same architectural ceiling as the 8x8 kernels). WebGPU + Wasm +
    /// CPU + CUDA + OpenCL all run the kernel cleanly post-rc.12.
    /// </summary>
    private static bool IsIdct16x16KernelSupported(Accelerator acc)
    {
        var name = acc.AcceleratorType.ToString();
        return !name.Equals("WebGL", StringComparison.OrdinalIgnoreCase);
    }

    [TestMethod]
    public async Task Vp9Idct16x16Kernel_ZeroCoefficients_LeavesPredictorUnchanged()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            if (!IsIdct16x16KernelSupported(acc)) return;
            using var kernel = new Vp9Idct16x16Kernel(acc);
            var coeffs = new short[256];
            var dest = new byte[256];
            for (int i = 0; i < 256; i++) dest[i] = 128;

            using var dCoeffs = acc.Allocate1D<short>(256);
            using var dActual = acc.Allocate1D<byte>(256);
            using var dExpected = acc.Allocate1D<byte>(256);
            dCoeffs.View.CopyFromCPU(coeffs);
            dActual.View.CopyFromCPU(dest);
            dExpected.View.CopyFromCPU(dest); // expected == initial predictor (no residual)

            kernel.RunOnGpu(dCoeffs.View, dActual.View, blockCount: 1);
            int mismatches = await Vp9TestVerify.CountByteMismatches(
                acc, dActual.View, dExpected.View, 256);
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Idct16x16Kernel_DcOnly_MatchesReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            if (!IsIdct16x16KernelSupported(acc)) return;
            using var kernel = new Vp9Idct16x16Kernel(acc);
            var coeffs = new short[256];
            coeffs[0] = 1024;
            var pred = new byte[256];
            for (int i = 0; i < 256; i++) pred[i] = 100;

            // CPU reference path computes the expected output once, on host.
            var cpuExpected = (byte[])pred.Clone();
            Vp9Idct16x16Reference.Idct16x16_256_Add(coeffs, cpuExpected, 16);

            // Upload everything to GPU; run kernel into dActual; compare
            // dActual vs dExpected on GPU; read one int back.
            using var dCoeffs = acc.Allocate1D<short>(256);
            using var dActual = acc.Allocate1D<byte>(256);
            using var dExpected = acc.Allocate1D<byte>(256);
            dCoeffs.View.CopyFromCPU(coeffs);
            dActual.View.CopyFromCPU(pred);
            dExpected.View.CopyFromCPU(cpuExpected);

            kernel.RunOnGpu(dCoeffs.View, dActual.View, blockCount: 1);
            int mismatches = await Vp9TestVerify.CountByteMismatches(
                acc, dActual.View, dExpected.View, 256);
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Idct16x16Kernel_RandomInputs_BitExactMatchReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            if (!IsIdct16x16KernelSupported(acc)) return;
            using var kernel = new Vp9Idct16x16Kernel(acc);
            var rng = new Random(unchecked((int)0xADA51610u));

            // Buffers reused across all trials - one allocate, many dispatches.
            using var dCoeffs = acc.Allocate1D<short>(256);
            using var dActual = acc.Allocate1D<byte>(256);
            using var dExpected = acc.Allocate1D<byte>(256);

            for (int trial = 0; trial < 5; trial++)
            {
                var coeffs = new short[256];
                for (int i = 0; i < 256; i++)
                    coeffs[i] = (short)rng.Next(-4096, 4096);
                var pred = new byte[256];
                for (int i = 0; i < 256; i++) pred[i] = (byte)rng.Next(0, 256);
                var cpuExpected = (byte[])pred.Clone();
                Vp9Idct16x16Reference.Idct16x16_256_Add(coeffs, cpuExpected, 16);

                dCoeffs.View.CopyFromCPU(coeffs);
                dActual.View.CopyFromCPU(pred);
                dExpected.View.CopyFromCPU(cpuExpected);

                kernel.RunOnGpu(dCoeffs.View, dActual.View, blockCount: 1);
                int mismatches = await Vp9TestVerify.CountByteMismatches(
                    acc, dActual.View, dExpected.View, 256);
                Equal(0, mismatches);
            }
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }

    [TestMethod]
    public async Task Vp9Idct16x16Kernel_BatchedDispatch_AllBlocksMatchReference()
    {
        var (ctx, acc) = await CreateKernelAcceleratorAsync();
        try
        {
            if (!IsIdct16x16KernelSupported(acc)) return;
            using var kernel = new Vp9Idct16x16Kernel(acc);
            const int n = 4;
            int total = n * 256;
            var rng = new Random(unchecked((int)0xADA51631u));
            var coeffsFlat = new short[total];
            var predFlat = new byte[total];
            for (int b = 0; b < n; b++)
            {
                for (int i = 0; i < 256; i++)
                    coeffsFlat[b * 256 + i] = (short)rng.Next(-4096, 4096);
                for (int i = 0; i < 256; i++)
                    predFlat[b * 256 + i] = (byte)rng.Next(0, 256);
            }

            var cpuExpected = (byte[])predFlat.Clone();
            for (int b = 0; b < n; b++)
            {
                Vp9Idct16x16Reference.Idct16x16_256_Add(
                    coeffsFlat.AsSpan(b * 256, 256),
                    cpuExpected.AsSpan(b * 256, 256),
                    16);
            }

            using var dCoeffs = acc.Allocate1D<short>(total);
            using var dActual = acc.Allocate1D<byte>(total);
            using var dExpected = acc.Allocate1D<byte>(total);
            dCoeffs.View.CopyFromCPU(coeffsFlat);
            dActual.View.CopyFromCPU(predFlat);
            dExpected.View.CopyFromCPU(cpuExpected);

            kernel.RunOnGpu(dCoeffs.View, dActual.View, blockCount: n);
            int mismatches = await Vp9TestVerify.CountByteMismatches(
                acc, dActual.View, dExpected.View, total);
            Equal(0, mismatches);
        }
        finally { acc.Dispose(); ctx.Dispose(); }
    }
}
