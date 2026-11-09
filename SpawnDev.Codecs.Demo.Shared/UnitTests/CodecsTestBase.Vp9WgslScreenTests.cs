// Offline WebGPU screen of the VP9 decoder's kernels: generates each kernel's WGSL on the desktop with
// SpawnDev.ILGPU's ShaderCompiler (the same code generator the browser backend runs; no browser, no GPU) and checks
// 1. that it generates at all (WGSL cannot express some constructs, for example choosing at run time between views of
//    different buffers: the parse kernel did that until 2026-11-09), and
// 2. a size budget per kernel. The browser's shader compiler (Tint) drove Chrome's GPU process to many GB on giant
//    Codecs kernels; on WebGPU every helper taking a view is inlined at every call site, so a size regression here is a
//    browser-crash regression there. Budgets are today's sizes plus 10%; tighten them when ILGPU emits view-taking
//    helpers as WGSL functions. VP9_WGSL_PRINT=1 prints the measured sizes in budget-table form.
// GLSL (WebGL) is not screened yet: SpawnDev.ILGPU 5.4.21's GLSL generator overflows the stack on these kernels.

using System.Reflection;
using SpawnDev.Codecs.Video.Vp9.Decoder;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    // (class, kernel method, WGSL character budget). Measured 2026-11-09 on SpawnDev.ILGPU 5.4.21: ParseKernel 87.1 MB
    // (821,758 lines), InterPass / IntraWave 33.7 MB each, loop-filter Wavefront 10.8 MB, CompressedHeader 6.4 MB; the
    // browser GPU lanes keep these fenced (GiantShader) until they shrink.
    private static readonly (Type Owner, string Kernel, long Budget)[] Vp9WgslKernels =
    {
        (typeof(Vp9CompressedHeaderKernels), "CopyContextKernel", 3734),
        (typeof(Vp9CompressedHeaderKernels), "CompressedHeaderKernel", 7089732),
        (typeof(Vp9ParseKernels), "ParseKernel", 95805552),
        (typeof(Vp9ReconKernels), "InterPassKernel", 37019010),
        (typeof(Vp9ReconKernels), "IntraWaveKernel", 37024471),
        (typeof(Vp9LoopFilterKernels), "FrameInitKernel", 24780),
        (typeof(Vp9LoopFilterKernels), "WavefrontKernel", 11848236),
        (typeof(Vp9AdaptKernels), "SumCountsKernel", 3579),
        (typeof(Vp9AdaptKernels), "AdaptKernel", 404973),
        (typeof(Vp9OutputKernels), "PackPlanesKernel", 19703),
    };

    [TestMethod(Timeout = 1800000, Category = "CpuOnly")]
    public async Task Vp9DecoderKernels_GenerateWgsl_WithinSizeBudget()
    {
        await Task.Yield();
        bool print = !OperatingSystem.IsBrowser() && Environment.GetEnvironmentVariable("VP9_WGSL_PRINT") == "1";
        var failures = new List<string>();
        foreach (var (owner, kernel, budget) in Vp9WgslKernels)
        {
            var method = owner.GetMethod(kernel, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
                ?? throw new InvalidOperationException($"{owner.Name}.{kernel} not found");
            long chars, lines;
            try
            {
                string wgsl = ShaderCompiler.Generate(method, CapabilityProfiles.WebGPUBaseline).Source ?? "";
                chars = wgsl.Length;
                lines = 1;
                foreach (char c in wgsl) if (c == '\n') lines++;
            }
            catch (Exception ex)
            {
                // 5.4.21 wraps UnsupportedKernelFeatureException (AggregateException -> InternalCompilerException -> ...)
                var inner = ex;
                while (inner.InnerException != null) inner = inner.InnerException;
                failures.Add($"{owner.Name}.{kernel}: no WGSL: {inner.GetType().Name}: {inner.Message}");
                continue;
            }
            Console.WriteLine($"VP9_WGSL {owner.Name}.{kernel}: {chars:N0} chars, {lines:N0} lines");
            if (print) Console.WriteLine($"VP9_WGSL_BUDGET (typeof({owner.Name}), \"{kernel}\", {chars + chars / 10}),");
            else if (budget <= 0) failures.Add($"{owner.Name}.{kernel}: no budget set ({chars:N0} chars); run with VP9_WGSL_PRINT=1");
            else if (chars > budget) failures.Add($"{owner.Name}.{kernel}: {chars:N0} chars, budget {budget:N0}");
        }
        if (print) throw new UnsupportedTestException("VP9_WGSL_PRINT=1: printed the budget table instead of checking it.");
        True(failures.Count == 0, string.Join("\n", failures));
    }
}
