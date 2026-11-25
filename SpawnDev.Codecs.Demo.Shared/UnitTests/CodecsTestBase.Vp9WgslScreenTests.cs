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
using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.Codecs.Video.Vp9.Decoder;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    // (class, kernel method, WGSL character budget), each today's size + 10%. Re-measured 2026-11-25 on SpawnDev.ILGPU 5.4.49
    // (VP9_WGSL_PRINT=1). Its optimizer order (5.4.46+) grew three: FrameInitKernel 22.5k -> 30.9k chars (+41%),
    // CompressedHeaderKernel +5%, PackPlanesKernel +4%. FrameInitKernel and the lossy encoder's refresh kernel (the other helper
    // that grew) run no faster or slower for it (nsys, RTX 4070, 2026-11-24); Data owns the growth and keeps the order, which
    // shrinks the FLAC decoders by half (FlacWgslKernels below). Measured 2026-11-09 on SpawnDev.ILGPU
    // 5.4.23-local.3, which emits view-taking helpers, ref state and LocalMemory views as WGSL functions instead of
    // inlining them. Re-measured 2026-11-10 on 5.4.23 after the reconstruction and loop-filter kernels moved from 64-bit
    // to 32-bit pixel / coefficient / scratch offsets (64-bit math is emulated in WGSL): loop-filter Wavefront 9.0 MB
    // (fully inlined 189k -> 116k lines; then 0.75 MB, about 14k inlined, after its call-site rewrite: one filter body per size), InterPredict 5.4 MB (134k -> 76k; then 0.28 MB, about 5.1k inlined, after its call-site rewrite: one call site per convolve), IntraWave 1.3 MB (2.7 MB; 116k -> 61k; 0.72 MB, about 16k, with one inverse transform site per size),
    // InterResidual 1.1 MB (2.4 MB; 102k -> 56k; 0.51 MB, about 12k, likewise),
    // CompressedHeader 0.43 MB (6.4 MB before the table-driven rewrite: one DiffUpdate call site instead of one per
    // syntax element), ParseKernel 0.55 MB (11,563 lines, about 11,900 with every function inlined as D3D does; before
    // the call-site rewrite 9.0 MB / 113,564 lines, but ParseBlock was a 111k-line WGSL function called from 6 places,
    // so about 763k lines inlined). A WGSL function's call sites multiply it on D3D, so the line count alone can hide
    // a regression: VP9_WGSL_DUMP=dir writes the WGSL to look at.
    // FLAC's stream decoder: the largest WGSL in the library after the VP9 parse (FindSyncKernel and DecodeFrameKernel scan and
    // decode a frame in one thread). ILGPU 5.4.46's callee-first optimize halved the text of both and 5.4.45's order doubles it, so a
    // swing like that must fail here. Measured 2026-11-25 on ILGPU 5.4.49 (its optimizer order), each +10%.
    private static readonly (Type Owner, string Kernel, long Budget)[] FlacWgslKernels =
    {
        (typeof(FlacStreamDecodeKernels), "FindSyncKernel", 379781),
        (typeof(FlacStreamDecodeKernels), "DecodeFrameKernel", 859585),
        (typeof(FlacStreamDecodeKernels), "WalkKernel", 464730),
    };

    private static readonly (Type Owner, string Kernel, long Budget)[] Vp9WgslKernels =
    {
        (typeof(Vp9CompressedHeaderKernels), "CopyContextKernel", 3654),
        (typeof(Vp9CompressedHeaderKernels), "CompressedHeaderKernel", 498092),
        (typeof(Vp9ParseKernels), "ParseKernel", 564875),
        (typeof(Vp9ReconKernels), "InterPredictKernel", 296954),
        (typeof(Vp9ReconKernels), "InterResidualKernel", 550611),
        (typeof(Vp9ReconKernels), "IntraWaveKernel", 776724),
        (typeof(Vp9LoopFilterKernels), "FrameInitKernel", 34001),
        (typeof(Vp9LoopFilterKernels), "WavefrontKernel", 593986),
        (typeof(Vp9LoopFilterKernels), "MaskKernel", 207478),   // 2026-11-15: the masks moved out of WavefrontKernel (754k -> 561k); 11-16: + the horizontal-edge plan (161k -> 195k, Wavefront 561k -> 548k)
        (typeof(Vp9AdaptKernels), "SumCountsKernel", 3499),
        (typeof(Vp9AdaptKernels), "AdaptKernel", 344954),
        (typeof(Vp9OutputKernels), "PackPlanesKernel", 20485),
    };

    [TestMethod(Timeout = 1800000, Category = "CpuOnly")]
    public async Task Vp9DecoderKernels_GenerateWgsl_WithinSizeBudget()
    {
        await Task.Yield();
        CheckWgslSizes(Vp9WgslKernels, "VP9_WGSL");
    }

    [TestMethod(Timeout = 1800000, Category = "CpuOnly")]
    public async Task FlacDecoderKernels_GenerateWgsl_WithinSizeBudget()
    {
        await Task.Yield();
        CheckWgslSizes(FlacWgslKernels, "FLAC_WGSL");
    }

    /// <summary>
    /// Generates each kernel's WGSL and checks it against its budget (<paramref name="label"/>_PRINT=1 prints the budget
    /// table instead of checking; <paramref name="label"/>_DUMP=dir writes the text).
    /// </summary>
    private static void CheckWgslSizes((Type Owner, string Kernel, long Budget)[] kernels, string label)
    {
        bool print = !OperatingSystem.IsBrowser() && Environment.GetEnvironmentVariable(label + "_PRINT") == "1";
        var failures = new List<string>();
        foreach (var (owner, kernel, budget) in kernels)
        {
            var method = owner.GetMethod(kernel, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
                ?? throw new InvalidOperationException($"{owner.Name}.{kernel} not found");
            long chars, lines;
            try
            {
                string wgsl = ShaderCompiler.Generate(method, CapabilityProfiles.WebGPUBaseline).Source ?? "";
                // diagnostics: VP9_WGSL_DUMP=dir writes <class>.<kernel>.wgsl (to see which code repeats)
                string? dump = OperatingSystem.IsBrowser() ? null : Environment.GetEnvironmentVariable(label + "_DUMP");
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
            Console.WriteLine($"{label} {owner.Name}.{kernel}: {chars:N0} chars, {lines:N0} lines");
            if (print) Console.WriteLine($"{label}_BUDGET (typeof({owner.Name}), \"{kernel}\", {chars + chars / 10}),");
            else if (budget <= 0) failures.Add($"{owner.Name}.{kernel}: no budget set ({chars:N0} chars); run with {label}_PRINT=1");
            else if (chars > budget) failures.Add($"{owner.Name}.{kernel}: {chars:N0} chars, budget {budget:N0}");
        }
        if (print) throw new UnsupportedTestException(label + "_PRINT=1: printed the budget table instead of checking it.");
        True(failures.Count == 0, string.Join("\n", failures));
    }
}
