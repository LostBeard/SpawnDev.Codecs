using Microsoft.Playwright;
using Microsoft.Testing.Platform.Requests;
using PlaywrightMultiTest;
using SpawnDev.UnitTesting;

public class ProjectTest
{
    public TestableProject Project { get; }
    public string Name { get; }
    public string? TestClassName { get; }
    public string? TestTypeName { get; }
    public string? TestMethodName { get; }
    public string? TestPageUrl { get; }
    public TestResult Result { get; set; }
    public string? ResultMessage { get; set; }
    public Func<IPage, Task> TestFunc { get; set; }
    public ProjectTest(TestableProject testableProject, string name)
    {
        Project = testableProject;
        Name = name;
        SetSuccess();
    }
    public ProjectTest(TestableProject testableProject, string typeName, string methodName, string testPage = "")
    {
        Project = testableProject;
        TestTypeName = typeName;
        TestMethodName = methodName;
        Name = $"{TestTypeName}.{TestMethodName}";
        TestClassName = $"{TestTypeName}-{TestMethodName}";
        TestPageUrl = testPage;
        TestFunc = RunTest;
    }
    public void SetSuccess()
    {
        Result = TestResult.Success;
        TestFunc = (page) => Task.CompletedTask;
    }
    public void SetError(string? err = null)
    {
        Result = TestResult.Error;
        TestFunc = async (page) =>
        {
            throw new Exception(string.IsNullOrWhiteSpace(err) ? "Failed" : err);
        };
    }
    public void SetDefault()
    {
        TestFunc = (page) => RunTest(page);
    }

    /// <summary>
    /// Checks if the Blazor error UI (#blazor-error-ui) is visible on the page.
    /// </summary>
    private static async Task<bool> IsBlazorErrorVisible(IPage page)
    {
        return await page.EvaluateAsync<bool>(
            "() => { var el = document.getElementById('blazor-error-ui'); return el != null && getComputedStyle(el).display !== 'none'; }");
    }

    /// <summary>
    /// Dismisses the Blazor error UI by reloading the page if it's visible.
    /// Returns the error text if one was found, null otherwise.
    /// </summary>
    private static async Task<string?> DismissBlazorErrorIfVisible(IPage page)
    {
        var visible = await IsBlazorErrorVisible(page);
        if (!visible) return null;
        var errorText = await page.EvaluateAsync<string>(
            "() => { var el = document.getElementById('blazor-error-ui'); return el ? el.innerText : ''; }");
        return string.IsNullOrWhiteSpace(errorText) ? "Blazor unhandled error (no message)" : errorText.Trim();
    }

    /// <summary>
    /// Per-page record of a dead .NET runtime. Once Mono asserts or the runtime exits, the page's DOM is frozen: the Run
    /// buttons stay enabled and the rows keep their old state, so reading them reports PASS for every later test (a run
    /// on 2026-10-25 logged 2,489 fake passes at ~250 ms each after one runtime exit). The test running when it died is
    /// failed with the runtime's message and the page is reloaded before the next test.
    /// </summary>
    private sealed class PageRuntime
    {
        public string? DeadReason;
    }

    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<IPage, PageRuntime> PageRuntimes = new();

    private static PageRuntime WatchRuntime(IPage page)
    {
        if (PageRuntimes.TryGetValue(page, out var state)) return state;
        state = new PageRuntime();
        PageRuntimes.Add(page, state);
        page.Console += (_, msg) =>
        {
            if (msg.Type != "error" || state.DeadReason != null) return;
            var text = msg.Text;
            if (text.Contains("runtime already exited") || text.Contains("MONO_WASM: Assert failed") || text.Contains("[MONO] * Assertion at"))
                state.DeadReason = Clip(text, 400);
        };
        page.Crash += (_, _) => state.DeadReason ??= "the page crashed";
        return state;
    }

    private static string Clip(string text, int max) => text.Length <= max ? text : text.Substring(0, max) + $"... ({text.Length} chars)";

    public async Task RunTest(IPage page)
    {
        var runtime = WatchRuntime(page);
        try
        {
            // Codec kernels (entropy walkers, transform chains) currently
            // compile to ~750KB Wasm functions with ~52K locals (5K inlined
            // call sites - same root cause as the WGSL walker fn-def
            // codegen issue). Cold-compile cost on the first dispatch can
            // exceed Playwright's 30s default actionability timeout while
            // V8 tier-up runs. Bump per-page default timeout to 5 minutes
            // so the click action's post-click stability wait survives
            // cold compile until the underlying ILGPU codegen path emits
            // helpers as Wasm fn-defs (Geordi's lane). Cheap on hot tests
            // (timeout doesn't fire when the page is responsive).
            page.SetDefaultTimeout(300_000);

            var rowSelector = $"tr.{TestClassName}";

            // make sure we are on the test page this test is on, with a live runtime
            if (page.Url != TestPageUrl || runtime.DeadReason != null)
            {
                runtime.DeadReason = null;
                await page.GotoAsync(TestPageUrl);

                // wait for test to load (5min - codec test pages enumerate
                // hundreds of test rows; bootstrap can exceed 30s under V8
                // tier-up while the kernel surface is still cold-compiling)
                await page.WaitForSelectorAsync(rowSelector, new() { Timeout = 300_000 });
            }

            // Check if Blazor error UI is already visible before this test
            var hadPreExistingBlazorError = await IsBlazorErrorVisible(page);

            // Capture console messages (errors + warnings) during the test
            var consoleErrors = new List<string>();
            var consoleWarnings = new List<string>();
            void OnConsole(object? sender, IConsoleMessage msg)
            {
                if (msg.Type == "error")
                    consoleErrors.Add(msg.Text);
                else if (msg.Type == "warning")
                    consoleWarnings.Add(msg.Text);
            }
            page.Console += OnConsole;

            // run the test
            var row = page.Locator(rowSelector);

            // find the button within THIS specific row
            var runButton = row.GetByRole(AriaRole.Button, new() { Name = "Run" });

            // wait for test button to be enabled
            await page.WaitForConditionAsync(async () =>
            {
                return await runButton.IsEnabledAsync();
            });

            // Self-test of the dead-runtime guard: PMT_SELFTEST_EXIT_RUNTIME=<test name> exits the page's .NET runtime
            // (dotnet.js RuntimeAPI.exit) just before that test, so it must FAIL and the next test must pass after a reload.
            var exitBefore = Environment.GetEnvironmentVariable("PMT_SELFTEST_EXIT_RUNTIME");
            if (!string.IsNullOrEmpty(exitBefore) && Name.EndsWith(exitBefore, StringComparison.Ordinal))
                await page.EvaluateAsync("() => { try { globalThis.getDotnetRuntime(0).exit(1, 'PMT self-test'); } catch (e) { console.error('MONO_WASM: Assert failed: PMT self-test exit threw ' + e); } }");

            // click the button to start the process for this row (button will be disabled)
            await runButton.ClickAsync();

            // wait for test to finish (button will become enabled again) or for the runtime to die
            await page.WaitForConditionAsync(async () =>
            {
                return runtime.DeadReason != null || await runButton.IsEnabledAsync();
            });

            // Stop capturing console
            page.Console -= OnConsole;

            // A dead runtime leaves the row as it was: never read it as a result (the next test reloads the page).
            if (runtime.DeadReason != null)
                throw new Exception($"The page's .NET runtime died during this test: {runtime.DeadReason}");

            // Only flag a Blazor error if it appeared NEW during this test (wasn't already there)
            string? blazorError = null;
            if (!hadPreExistingBlazorError)
                blazorError = await DismissBlazorErrorIfVisible(page);

            // current state text
            var stateMessage = await row.Locator(".test-state").TextContentAsync();

            //  check for error  class
            var wasError = await row.EvaluateAsync<bool>("el => el.classList.contains('test-error')");

            //  check for error  class
            var unsupported = await row.EvaluateAsync<bool>("el => el.classList.contains('test-unsupported')");

            // Log console errors/warnings to stderr for diagnostics
            if (consoleErrors.Count > 0 || consoleWarnings.Count > 0)
            {
                Console.Error.WriteLine($"[{Name}] Console: {consoleErrors.Count} error(s), {consoleWarnings.Count} warning(s)");
                foreach (var err in consoleErrors.Take(10))
                    Console.Error.WriteLine($"  ERROR: {Clip(err, 2000)}");
                foreach (var warn in consoleWarnings.Take(10))
                    Console.Error.WriteLine($"  WARN: {Clip(warn, 2000)}");
            }

            ResultMessage = stateMessage;

            if (unsupported)
            {
                Result = TestResult.Unsupported;
                if (string.IsNullOrWhiteSpace(stateMessage))
                {
                    stateMessage = "Skipped";
                }
            }
            else if (wasError || blazorError != null)
            {
                Result = TestResult.Error;
                if (blazorError != null && !wasError)
                {
                    // Test reported success but Blazor framework threw an unhandled error
                    stateMessage = $"Blazor error during test: {blazorError}";
                }
                else if (string.IsNullOrWhiteSpace(stateMessage))
                {
                    stateMessage = "Failed";
                }
                throw new Exception(stateMessage);
            }
            else
            {
                Result = TestResult.Success;
                if (string.IsNullOrWhiteSpace(stateMessage))
                {
                    stateMessage = "Success";
                }
            }
        }
        catch (Exception ex)
        {
            throw new Exception($"Test {Name} failed with error: {ex.Message}");
        }
    }
}
