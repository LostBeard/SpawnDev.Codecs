using SpawnDev.UnitTesting;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Velocity;
using SpawnDev.Codecs.Demo.Shared.UnitTests;

namespace SpawnDev.Codecs.DemoConsole.UnitTests;

/// <summary>
/// Desktop Velocity-backend entry point for SpawnDev.Codecs tests: ILGPU's vectorized CPU backend, the CPU backend
/// applications get by default (AllAccelerators registers it instead of the IL-interpreting CPU accelerator that
/// <see cref="CpuCodecsTests"/> runs on). CPU-only tests stay in CpuCodecsTests. The device is Vector128, which every
/// x64 CPU has (<c>Velocity()</c> alone registers Scalar2 first, which applications do not run). Since SpawnDev.ILGPU
/// 5.4.23, AllAccelerators picks Vector256 on AVX2 machines; set CODECS_VELOCITY_DEVICE=Vector256 to run this lane on
/// it (VP9 decode, ComfyUI files: reconstruction 113 -> 76 ms/frame, loop filter 466 -> 266, bit-exact).
/// </summary>
// Discovery leaves out tests this lane can only skip: CpuOnly (CPU-only tests run once, in CpuCodecsTests); BrowserOnly (TypedArray overloads need a browser).
[ExcludeCategories("CpuOnly", "BrowserOnly")]
public class VelocityCodecsTests : CodecsTestBase
{
    protected override ValueTask<(Context, Accelerator)> CreateKernelAcceleratorAsync()
    {
        // CODECS_VELOCITY_DEVICE=Vector256 (or another VelocityDeviceType name) runs this lane on that device instead, for
        // comparing Velocity targets on the same tests.
        var device = Enum.TryParse<VelocityDeviceType>(Environment.GetEnvironmentVariable("CODECS_VELOCITY_DEVICE"), out var d)
            ? d : VelocityDeviceType.Vector128;
        var ctx = Context.Create(b => b.Velocity(device));
        var acc = ctx.CreateVelocityAccelerator(0);
        return new ValueTask<(Context, Accelerator)>((ctx, acc));
    }
}
