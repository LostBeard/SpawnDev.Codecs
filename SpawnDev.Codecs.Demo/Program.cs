using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using SpawnDev.Codecs.Demo;
using SpawnDev.Codecs.Demo.UnitTests;
using SpawnDev.SpawnJS;

Console.WriteLine("[SpawnDev.Codecs.Demo] booting.");

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.Services.AddSpawnJSRuntime(out var JS);
// Slot lifetime is manual in SpawnJS; the watcher names leaks from owned wrappers / callbacks, but it captures a full
// StackTrace(true) on every SpawnJS object, which the WASM interpreter pays dearly for. Measured by Data 10-27 on
// WebPRgbaToYuvGpu_AllFixtures (WebGL): ~8 s per conversion with it on (305 ms per tiny Allocate1D), 3 s for the whole
// test with it off. Opt in when hunting a leak: ?watchdispose=1 on the URL.
SpawnJSRuntime.EnableIDisposableWatcher =
    JS.Get<string>("location.search")?.Contains("watchdispose=1", StringComparison.OrdinalIgnoreCase) ?? false;
// ?env=NAME=value;NAME2=value2 (PlaywrightMultiTest's PMT_PAGE_ENV): environment variables for this runtime, so a test's
// diagnostics switch read with Environment.GetEnvironmentVariable (VP9_STAGE_TIMES=1, ...) works in the browser lanes too.
foreach (var part in (JS.Get<string>("location.search") ?? "").TrimStart('?').Split('&'))
{
    if (!part.StartsWith("env=", StringComparison.Ordinal)) continue;
    foreach (var pair in Uri.UnescapeDataString(part.Substring(4)).Split(';', StringSplitOptions.RemoveEmptyEntries))
    {
        int eq = pair.IndexOf('=');
        if (eq > 0) Environment.SetEnvironmentVariable(pair.Substring(0, eq), pair.Substring(eq + 1));
    }
}
builder.Services.AddSingleton(sp => new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });

builder.Services.AddSingleton<WebGPUCodecsTests>();
builder.Services.AddSingleton<WebGLCodecsTests>();
builder.Services.AddSingleton<WasmCodecsTests>();

builder.Services.AddSingleton<SpawnDev.ILGPU.Services.ShaderDebugService>();

builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

await builder.Build().SpawnJSRunAsync();
