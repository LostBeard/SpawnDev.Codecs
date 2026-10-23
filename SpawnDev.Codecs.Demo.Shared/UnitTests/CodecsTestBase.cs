using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;

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

    /// <summary>
    /// True in the ONE lane that runs the CPU-only tests (<c>CpuCodecsTests</c>, desktop). A CPU-only test never touches an
    /// accelerator (bit readers, tables, parsers, the CPU reference ports): it runs the same managed code on every lane, so
    /// running it again on CUDA / OpenCL / WebGPU / WebGL / Wasm only burns host CPU - in the browser lanes that is
    /// single-threaded .NET Wasm.
    /// </summary>
    protected virtual bool RunsCpuOnlyTests => false;

    /// <summary>First statement of every CPU-only test: skip unless this is the lane that runs them (see <see cref="RunsCpuOnlyTests"/>).</summary>
    protected void CpuOnlyTest()
    {
        if (!RunsCpuOnlyTests)
            throw new UnsupportedTestException("CPU-only test (no accelerator): it runs once, in the desktop CPU lane (CpuCodecsTests).");
    }

    /// <summary>
    /// Throw <see cref="UnsupportedTestException"/> if the supplied accelerator's
    /// backend cannot satisfy the requirements. Use at the top of any test whose
    /// kernels need features not available on every backend (atomics, shared
    /// memory, native f64, etc.). Skipped tests are reported as Unsupported, not
    /// Failed, by the SpawnDev.UnitTesting harness.
    /// </summary>
    protected static void RequireFeatures(Accelerator accelerator, AcceleratorRequirements requirements)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        ArgumentNullException.ThrowIfNull(requirements);
        if (!accelerator.Device.Satisfies(requirements))
        {
            throw new UnsupportedTestException(
                $"Accelerator {accelerator.AcceleratorType} does not satisfy required features: {requirements.Describe()}");
        }
    }

    /// <summary>
    /// Wrap <see cref="CreateKernelAcceleratorAsync"/> to convert
    /// <see cref="NotSupportedException"/> into <see cref="UnsupportedTestException"/>.
    /// Some backends (WebGL in particular) do eager kernel validation at
    /// accelerator creation and throw <see cref="NotSupportedException"/> when
    /// a kernel uses an unsupported feature (e.g. atomics on WebGL). The
    /// SpawnDev.UnitTesting harness only treats <see cref="UnsupportedTestException"/>
    /// as a Skip; <see cref="NotSupportedException"/> is reported as a Failure.
    /// Tests that depend on kernels that may fail eager validation should call
    /// this helper instead of <see cref="CreateKernelAcceleratorAsync"/> directly.
    /// </summary>
    protected async ValueTask<(Context, Accelerator)> AcquireAcceleratorOrSkipAsync()
    {
        try
        {
            return await CreateKernelAcceleratorAsync();
        }
        catch (NotSupportedException ex)
        {
            throw new UnsupportedTestException(ex.Message);
        }
    }
}
