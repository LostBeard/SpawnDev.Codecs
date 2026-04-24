using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using SpawnDev.Codecs.Demo.Shared.UnitTests;

namespace SpawnDev.Codecs.DemoConsole.UnitTests;

/// <summary>
/// Desktop CUDA-backend entry point for SpawnDev.Codecs tests. Inherits every
/// cross-platform test from <see cref="CodecsTestBase"/> and runs ILGPU kernel
/// tests on the first available NVIDIA GPU via PTX codegen.
/// </summary>
public class CudaCodecsTests : CodecsTestBase
{
    protected override ValueTask<(Context, Accelerator)> CreateKernelAcceleratorAsync()
    {
        var ctx = Context.Create(b => b.Cuda());
        var acc = ctx.CreateCudaAccelerator(0);
        return new ValueTask<(Context, Accelerator)>((ctx, acc));
    }
}
