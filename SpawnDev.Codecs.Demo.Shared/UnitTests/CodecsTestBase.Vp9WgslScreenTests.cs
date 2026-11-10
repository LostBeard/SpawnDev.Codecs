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
    // (class, kernel method, WGSL character budget), each today's size + 10%. Measured 2026-11-09 on SpawnDev.ILGPU
    // 5.4.23-local.3, which emits view-taking helpers, ref state and LocalMemory views as WGSL functions instead of
    // inlining them. Re-measured 2026-11-10 on 5.4.23 after the reconstruction and loop-filter kernels moved from 64-bit
    // to 32-bit pixel / coefficient / scratch offsets (64-bit math is emulated in WGSL): loop-filter Wavefront 9.0 MB
    // (fully inlined 189k -> 116k lines; then 0.75 MB, about 14k inlined, after its call-site rewrite: one filter body per size), InterPredict 5.4 MB (134k -> 76k; then 0.21 MB, about 4.1k inlined, after its call-site rewrite: one convolve body), IntraWave 1.3 MB (2.7 MB; 116k -> 61k; 0.72 MB, about 16k, with one inverse transform site per size),
    // InterResidual 1.1 MB (2.4 MB; 102k -> 56k; 0.51 MB, about 12k, likewise),
    // CompressedHeader 0.43 MB (6.4 MB before the table-driven rewrite: one DiffUpdate call site instead of one per
    // syntax element), ParseKernel 0.55 MB (11,563 lines, about 11,900 with every function inlined as D3D does; before
    // the call-site rewrite 9.0 MB / 113,564 lines, but ParseBlock was a 111k-line WGSL function called from 6 places,
    // so about 763k lines inlined). A WGSL function's call sites multiply it on D3D, so the line count alone can hide
    // a regression: VP9_WGSL_DUMP=dir writes the WGSL to look at.
    private static readonly (Type Owner, string Kernel, long Budget)[] Vp9WgslKernels =
    {
        (typeof(Vp9CompressedHeaderKernels), "CopyContextKernel", 3734),
        (typeof(Vp9CompressedHeaderKernels), "CompressedHeaderKernel", 471586),
        (typeof(Vp9ParseKernels), "ParseKernel", 600774),
        (typeof(Vp9ReconKernels), "InterPredictKernel", 229652),
        (typeof(Vp9ReconKernels), "InterResidualKernel", 563233),
        (typeof(Vp9ReconKernels), "IntraWaveKernel", 794108),
        (typeof(Vp9LoopFilterKernels), "FrameInitKernel", 24780),
        (typeof(Vp9LoopFilterKernels), "WavefrontKernel", 829819),
        (typeof(Vp9AdaptKernels), "SumCountsKernel", 3579),
        (typeof(Vp9AdaptKernels), "AdaptKernel", 354152),
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
                // diagnostics: VP9_WGSL_DUMP=dir writes <class>.<kernel>.wgsl (to see which code repeats)
                string? dump = OperatingSystem.IsBrowser() ? null : Environment.GetEnvironmentVariable("VP9_WGSL_DUMP");
                if (dump != null) File.WriteAllText(Path.Combine(dump, $"{owner.Name}.{kernel}.wgsl"), wgsl);
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
