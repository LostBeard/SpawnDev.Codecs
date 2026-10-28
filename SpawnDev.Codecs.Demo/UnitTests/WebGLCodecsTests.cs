using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Demo.Shared.UnitTests;
using SpawnDev.ILGPU.WebGL;
using SpawnDev.UnitTesting;

namespace SpawnDev.Codecs.Demo.UnitTests;

/// <summary>
/// Browser-side WebGL-backend entry point for SpawnDev.Codecs tests. Inherits every cross-platform test from
/// <see cref="CodecsTestBase"/> and runs the ILGPU kernel tests on the browser's WebGL2 backend (GLSL ES 3.0 through
/// Transform Feedback).
/// </summary>
/// <remarks>
/// WebGL has no atomics, shared memory or barriers, and a thread stores only at its own output slot (no scatter
/// stores). A test whose kernels need one of those declares it with <c>RequireFeatures</c> and is skipped here with
/// that reason; every other test runs. (Until 2026-10-26 this class threw for EVERY test, so no Codecs kernel had ever
/// run on WebGL.)
/// </remarks>
// CPU-only tests (Category = "CpuOnly") run once, in CpuCodecsTests: discovery leaves them out here.
[ExcludeCategories("CpuOnly")]
public class WebGLCodecsTests : CodecsTestBase
{
    protected override async ValueTask<(Context, Accelerator)> CreateKernelAcceleratorAsync()
    {
        var builder = Context.Create();
        await builder.WebGL();
        var ctx = builder.ToContext();
        var devices = ctx.GetWebGLDevices();
        if (devices.Count == 0)
        {
            ctx.Dispose();
            throw new UnsupportedTestException("No WebGL2 devices on this runtime.");
        }
        var acc = devices[0].CreateAccelerator(ctx);
        return (ctx, acc);
    }
}
