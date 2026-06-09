using System.Reflection;
using System.Text.RegularExpressions;
using ILGPU;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.ILGPU.Wasm;
using SpawnDev.ILGPU.Wasm.Backend;

namespace SpawnDev.Codecs.DemoConsole;

/// <summary>
/// Offline (desktop, no browser) Wasm COMPILE measurement for the VP9 frame-entropy walker.
/// Mirrors SpawnDev.ILGPU.DemoConsole/WasmCompileDump: a desktop WasmAccelerator runs the
/// IL→wasm codegen path fully (LoadAutoGroupedStreamKernel compiles eagerly, before any dispatch),
/// so constructing <see cref="Vp9FrameEntropyKernel"/> compiles <c>BatchEncodeFrameKernel</c> and the
/// WasmBackend logs (under VerboseLogging) a "[Wasm-Helper] 'EncodeFrameBody' ... locals=N ..." line —
/// EncodeFrameBody is [NoInlining] so it is emitted as its own function and gets its own locals count.
///
/// This is the before/after vehicle for Geordi's CumulativeInlinedILBudget fix (ILGPU 4.9.16-local.1):
///   - budget INERT  → the SB64→leaf→coef tree all inlines into EncodeFrameBody → high locals (~52K)
///   - budget FIRES  → the deep tree is bounded → EncodeFrameBody locals drop
/// We never dispatch (that needs Web Workers); we only compile and read the emitted locals + the
/// Inliner diagnostic counters (read via reflection so this compiles against any ILGPU version).
///
/// Run: dotnet run --project SpawnDev.Codecs.DemoConsole -c Release -- wasm-vp9-dump
/// </summary>
internal static class WasmVp9Dump
{
    public static async Task<int> Run()
    {
        Console.WriteLine("=== Wasm offline compile measurement: VP9 EncodeFrameBody locals ===");
        Console.WriteLine($"ILGPU: {typeof(Context).Assembly.GetName().Version}");

        // Is the tunable budget present (fix build)? If so, sweep it to find the value that splits
        // EncodeFrameBody under the V8/wabt ~50K-locals-per-function ceiling. On a pre-fix ILGPU the
        // field is absent and we just do one default-budget compile.
        var budgetMember = GetInlinerStatic("CumulativeInlinedILBudget");
        var budgets = budgetMember != null
            ? new long[] { -1 /*default*/, 8192, 4096, 2048, 1024, 512, 256 }
            : new long[] { -1 };

        Console.WriteLine($"  tunable CumulativeInlinedILBudget present: {budgetMember != null}");
        Console.WriteLine();
        Console.WriteLine("  budget   EncodeFrameBody  maxHelper(name)            helpers  skipCount  maxCumIL");
        Console.WriteLine("  -------  ---------------  ------------------------  -------  ---------  --------");

        foreach (var budget in budgets)
        {
            var r = await CompileOnce(budget, budgetMember);
            string bcol = budget < 0 ? "default" : budget.ToString();
            Console.WriteLine($"  {bcol,-7}  {r.efb,15}  {r.maxName + "(" + r.maxLocals + ")",-24}  {r.helpers,7}  {r.skip,9}  {r.maxIL,8}");
        }

        Console.WriteLine();
        Console.WriteLine("Note: V8 kV8MaxWasmFunctionLocals / wabt cap ~50000 per emitted function. EncodeFrameBody must land under that.");
        Console.WriteLine("=== wasm-vp9-dump done ===");
        return 0;
    }

    private readonly record struct CompileResult(long efb, long maxLocals, string maxName, long helpers, string skip, string maxIL);

    private static async Task<CompileResult> CompileOnce(long budget, MemberInfo? budgetMember)
    {
        // Fresh context per compile so the kernel cache doesn't return a previous budget's result.
        var context = Context.Create().Wasm().ToContext();
        WasmAccelerator accelerator = await context.CreateWasmAcceleratorAsync();

        if (budget >= 0 && budgetMember != null) SetInlinerStatic(budgetMember, budget);
        ResetInlinerCounter("CumulativeBudgetSkipCount");
        ResetInlinerCounter("MaxCumulativeInlinedIL");

        var prevVerbose = WasmBackend.VerboseLogging;
        var prevOut = Console.Out;
        var sw = new FilteringLineWriter(l => l.Contains("[Wasm-Helper]") || l.Contains("Kernel params="));
        WasmBackend.VerboseLogging = true;
        Console.SetOut(sw);
        IDisposable? kernel = null;
        try { kernel = new Vp9FrameEntropyKernel(accelerator); }
        catch { /* compile may still have logged the helper line before throwing */ }
        finally { Console.SetOut(prevOut); WasmBackend.VerboseLogging = prevVerbose; }

        var helperRx = new Regex(@"\[Wasm-Helper\] '([^']+)'.*?locals=(\d+)");
        var kernelRx = new Regex(@"Kernel params=.*?locals=(\d+).*?helpers=(\d+)");
        long efb = -1, helpers = -1, maxLocals = -1; string maxName = "";
        foreach (var raw in sw.Captured.Split('\n'))
        {
            var hm = helperRx.Match(raw);
            if (hm.Success)
            {
                long locals = long.Parse(hm.Groups[2].Value);
                if (locals > maxLocals) { maxLocals = locals; maxName = hm.Groups[1].Value; }
                if (hm.Groups[1].Value.Contains("EncodeFrameBody")) efb = locals;
            }
            var km = kernelRx.Match(raw);
            if (km.Success) helpers = long.Parse(km.Groups[2].Value);
        }

        string skip = ReadInlinerCounter("CumulativeBudgetSkipCount");
        string maxIL = ReadInlinerCounter("MaxCumulativeInlinedIL");
        kernel?.Dispose();
        accelerator.Dispose();
        context.Dispose();
        return new CompileResult(efb, maxLocals, maxName, helpers, skip, maxIL);
    }

    private static MemberInfo? GetInlinerStatic(string name)
    {
        var inliner = typeof(Context).Assembly.GetType("ILGPU.IR.Transformations.Inliner");
        if (inliner == null) return null;
        return (MemberInfo?)inliner.GetField(name, BindingFlags.Public | BindingFlags.Static | BindingFlags.NonPublic)
               ?? inliner.GetProperty(name, BindingFlags.Public | BindingFlags.Static | BindingFlags.NonPublic);
    }
    private static void SetInlinerStatic(MemberInfo m, long value)
    {
        try { if (m is FieldInfo f) f.SetValue(null, Convert.ChangeType(value, f.FieldType)); else if (m is PropertyInfo p && p.CanWrite) p.SetValue(null, Convert.ChangeType(value, p.PropertyType)); }
        catch { }
    }
    private static void ResetInlinerCounter(string name) { var m = GetInlinerStatic(name); if (m != null) SetInlinerStatic(m, 0); }
    private static string ReadInlinerCounter(string name)
    {
        var m = GetInlinerStatic(name);
        if (m == null) return "n/a";
        try { object? v = m is FieldInfo f ? f.GetValue(null) : ((PropertyInfo)m).GetValue(null); return v?.ToString() ?? "?"; }
        catch { return "err"; }
    }

    /// <summary>A line-buffering TextWriter that retains ONLY lines matching a predicate (bounded memory).
    /// Everything else is discarded — needed because the budget-inert walker's VerboseLogging is gigabytes.</summary>
    private sealed class FilteringLineWriter : TextWriter
    {
        private readonly Func<string, bool> _keep;
        private readonly System.Text.StringBuilder _line = new();
        private readonly List<string> _kept = new();
        public FilteringLineWriter(Func<string, bool> keep) => _keep = keep;
        public override System.Text.Encoding Encoding => System.Text.Encoding.UTF8;
        public string Captured => string.Join("\n", _kept);
        public override void Write(char c)
        {
            if (c == '\n') { Flush(); return; }
            if (c != '\r') _line.Append(c);
        }
        public override void Write(string? s)
        {
            if (s == null) return;
            foreach (var c in s) Write(c);
        }
        public override void Flush()
        {
            if (_line.Length == 0) return;
            var l = _line.ToString();
            _line.Clear();
            if (_keep(l) && _kept.Count < 100_000) _kept.Add(l);
        }
    }
}
