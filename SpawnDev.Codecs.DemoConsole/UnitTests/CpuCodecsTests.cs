using SpawnDev.UnitTesting;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using SpawnDev.Codecs.Demo.Shared.UnitTests;

namespace SpawnDev.Codecs.DemoConsole.UnitTests;

/// <summary>
/// Desktop CPU-backend entry point for SpawnDev.Codecs tests. Inherits every
/// cross-platform test from <see cref="CodecsTestBase"/> and runs ILGPU kernel
/// tests on the ILGPU CPU backend (IR interpreter).
/// </summary>
// Discovery leaves out tests this lane can only skip: BrowserOnly (TypedArray overloads need a browser).
[ExcludeCategories("BrowserOnly")]
public class CpuCodecsTests : CodecsTestBase
{
    /// <summary>The CPU-only tests (no accelerator) run here and nowhere else.</summary>
    protected override bool RunsCpuOnlyTests => true;

    protected override ValueTask<(Context, Accelerator)> CreateKernelAcceleratorAsync()
    {
        var ctx = Context.Create(b => b.CPU());
        var acc = ctx.CreateCPUAccelerator(0);
        return new ValueTask<(Context, Accelerator)>((ctx, acc));
    }
}
