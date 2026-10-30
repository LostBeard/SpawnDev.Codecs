using SpawnDev.UnitTesting;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Demo.Shared.UnitTests;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.ILGPU.WebGPU.Backend;

namespace SpawnDev.Codecs.Demo.UnitTests;

/// <summary>
/// Browser-side WebGPU-backend entry point for SpawnDev.Codecs tests. Inherits
/// every cross-platform test from <see cref="CodecsTestBase"/> and runs ILGPU
/// kernel tests on the browser's WebGPU backend (WGSL codegen).
/// </summary>
// Discovery leaves out tests this lane can only skip: CpuOnly (CPU-only tests run once, in CpuCodecsTests); DesktopOnly (the Pillow / libwebp oracles need a desktop process).
[ExcludeCategories("CpuOnly", "DesktopOnly")]
public class WebGPUCodecsTests : CodecsTestBase
{
    /// <summary>
    /// LRU cap on compiled shaders the shared accelerator keeps (<see cref="WebGPUBackend.MaxCachedShaders"/>; 0 = unlimited).
    /// The whole lane is one class sharing ONE accelerator (<see cref="CodecsTestBase.AcquireAcceleratorOrSkipAsync"/>), so an
    /// unbounded cache keeps every kernel of all ~1150 tests alive and Chrome's GPU process grows over the lane (3.77 GB peak
    /// on 2026-10-30). An LRU cap keeps each test's working set warm and evicts the cold tail; clearing the cache per test
    /// recompiles everything (SpawnDev.ILGPU's own suite measured that, and caps at 64).
    /// Measured 2026-10-30, full lane, Chrome 156.0.8078.48: with no cap the GPU process climbed to 3.4 GB by the end; with
    /// 128 it held at 1.3-1.8 GB, and the lane took the same time (27.1 vs 27.3 min). A ~3.3 GB transient spike remains in both
    /// runs: a single compile of the VP8 GPU-pair kernels (GpuTranscodeDemo_Vp8), Plans/PLAN-Oversized-Kernels.md.
    /// </summary>
    public static int TestShaderCacheCap { get; set; } = 128;

    static WebGPUCodecsTests() => WebGPUBackend.MaxCachedShaders = TestShaderCacheCap;

    protected override async ValueTask<(Context, Accelerator)> CreateKernelAcceleratorAsync()
    {
        // builder.WebGPU() is async: it probes the browser's GPU adapter
        // before returning. Use the builder / ToContext / GetWebGPUDevices
        // pattern from the SpawnDev.ILGPU WebGPU test reference.
        var builder = Context.Create();
        await builder.WebGPU();
        var ctx = builder.ToContext();
        var devices = ctx.GetWebGPUDevices();
        if (devices.Count == 0)
            throw new InvalidOperationException("No WebGPU devices available on this runtime.");
        var acc = await devices[0].CreateAcceleratorAsync(ctx);
        return (ctx, acc);
    }
}
