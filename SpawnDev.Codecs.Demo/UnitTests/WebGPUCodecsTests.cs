using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Demo.Shared.UnitTests;
using SpawnDev.ILGPU.WebGPU;

namespace SpawnDev.Codecs.Demo.UnitTests;

/// <summary>
/// Browser-side WebGPU-backend entry point for SpawnDev.Codecs tests. Inherits
/// every cross-platform test from <see cref="CodecsTestBase"/> and runs ILGPU
/// kernel tests on the browser's WebGPU backend (WGSL codegen).
/// </summary>
public class WebGPUCodecsTests : CodecsTestBase
{
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
