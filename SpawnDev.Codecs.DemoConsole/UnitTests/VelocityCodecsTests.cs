using SpawnDev.UnitTesting;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Velocity;
using SpawnDev.Codecs.Demo.Shared.UnitTests;

namespace SpawnDev.Codecs.DemoConsole.UnitTests;

/// <summary>
/// Desktop Velocity-backend entry point for SpawnDev.Codecs tests: ILGPU's vectorized CPU backend, the CPU backend
/// applications get by default (AllAccelerators registers it instead of the IL-interpreting CPU accelerator that
/// <see cref="CpuCodecsTests"/> runs on). CPU-only tests stay in CpuCodecsTests. The device is Vector128, the one
/// AllAccelerators / HostCompute pick: <c>Velocity()</c> alone registers Scalar2 (two scalar lanes) first, and device 0 of
/// that is not what applications run.
/// </summary>
// Discovery leaves out tests this lane can only skip: CpuOnly (CPU-only tests run once, in CpuCodecsTests); BrowserOnly (TypedArray overloads need a browser).
[ExcludeCategories("CpuOnly", "BrowserOnly")]
public class VelocityCodecsTests : CodecsTestBase
{
    protected override ValueTask<(Context, Accelerator)> CreateKernelAcceleratorAsync()
    {
        var ctx = Context.Create(b => b.Velocity(VelocityDeviceType.Vector128));
        var acc = ctx.CreateVelocityAccelerator(0);
        return new ValueTask<(Context, Accelerator)>((ctx, acc));
    }
}
