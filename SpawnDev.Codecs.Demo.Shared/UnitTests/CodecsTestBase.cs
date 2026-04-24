using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Base class for cross-platform SpawnDev.Codecs tests. Concrete subclasses live in
/// <c>SpawnDev.Codecs.Demo</c> (browser, runs via PlaywrightMultiTest against Blazor WASM)
/// and <c>SpawnDev.Codecs.DemoConsole</c> (desktop, runs via PlaywrightMultiTest against
/// a published .NET console exe).
///
/// Tests are split into partial files by subject so each subject's coverage is self-contained
/// and easy to locate. See the matching <c>CodecsTestBase.*.cs</c> files.
/// </summary>
public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Create an <see cref="Accelerator"/> for ILGPU kernel tests on this runner's
    /// native backend. Default implementation creates the ILGPU CPU backend
    /// (IR interpreter; works in every runtime including Blazor WASM). Concrete
    /// runners override this to dispatch kernels on their hardware: CUDA for
    /// <c>CudaCodecsTests</c>, OpenCL for <c>OpenCLCodecsTests</c>, WebGPU for
    /// <c>WebGPUCodecsTests</c>, etc.
    ///
    /// The returned <see cref="Context"/> owns the accelerator's lifetime -
    /// callers must dispose both (<c>accelerator.Dispose(); context.Dispose()</c>)
    /// when the test finishes.
    /// </summary>
    protected virtual ValueTask<(Context, Accelerator)> CreateKernelAcceleratorAsync()
    {
        var ctx = Context.Create(b => b.CPU());
        var acc = ctx.CreateCPUAccelerator(0);
        return new ValueTask<(Context, Accelerator)>((ctx, acc));
    }
}
