using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Velocity;
using SpawnDev.Codecs.Demo.Shared.UnitTests;

namespace SpawnDev.Codecs.DemoConsole.UnitTests;

/// <summary>
/// Desktop Velocity-backend entry point for SpawnDev.Codecs tests: ILGPU's vectorized CPU backend, the CPU backend
/// applications get by default (AllAccelerators registers it instead of the IL-interpreting CPU accelerator that
/// <see cref="CpuCodecsTests"/> runs on). CPU-only tests stay in CpuCodecsTests.
/// </summary>
public class VelocityCodecsTests : CodecsTestBase
{
    protected override ValueTask<(Context, Accelerator)> CreateKernelAcceleratorAsync()
    {
        var ctx = Context.Create(b => b.Velocity());
        var acc = ctx.CreateVelocityAccelerator(0);
        return new ValueTask<(Context, Accelerator)>((ctx, acc));
    }
}
