using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Demo.Shared.UnitTests;
using SpawnDev.ILGPU.WebGL;

namespace SpawnDev.Codecs.Demo.UnitTests;

/// <summary>
/// Browser-side WebGL-backend entry point for SpawnDev.Codecs tests. Inherits
/// every cross-platform test from <see cref="CodecsTestBase"/>. ILGPU kernel
/// tests that require atomics (including sub-word <c>ArrayView&lt;byte&gt;</c>
/// writes, which ILGPU lowers to atomic RMW on GPU backends) fail on WebGL
/// because WebGL has no atomics support.
/// </summary>
/// <remarks>
/// See <c>tuvok-to-geordi-ilgpu-capability-gating-2026-04-24.md</c> for a
/// proposed library-level capability-gating fix. Until that lands, we
/// throw a clear NotSupportedException here rather than letting the
/// kernel silently produce wrong bytes.
/// </remarks>
public class WebGLCodecsTests : CodecsTestBase
{
    protected override ValueTask<(Context, Accelerator)> CreateKernelAcceleratorAsync()
    {
        // TODO(slice 119+): once SpawnDev.ILGPU gates backends on required
        // capabilities (or the kernel is refactored to ArrayView<uint> with
        // explicit 4-byte packing), remove this throw and use the standard
        // builder.WebGL() / ToContext() / devices[0].CreateAcceleratorAsync
        // pattern like the other browser runners.
        throw new NotSupportedException(
            "VP9 iDCT 4x4 kernel uses ArrayView<byte> writes that ILGPU " +
            "lowers to atomic RMW. WebGL has no atomics support, so the " +
            "kernel produces non-bit-exact output on this backend. Run " +
            "this kernel on WebGPU / Wasm / CUDA / OpenCL / CPU.");
    }
}
