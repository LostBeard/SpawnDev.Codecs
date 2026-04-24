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
public class CpuCodecsTests : CodecsTestBase
{
    protected override ValueTask<(Context, Accelerator)> CreateKernelAcceleratorAsync()
    {
        var ctx = Context.Create(b => b.CPU());
        var acc = ctx.CreateCPUAccelerator(0);
        return new ValueTask<(Context, Accelerator)>((ctx, acc));
    }
}
