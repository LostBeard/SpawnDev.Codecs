using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using SpawnDev.BlazorJS;
using SpawnDev.Codecs.Demo;

// Planning phase (2026-04-23). Minimal Blazor WASM bootstrap.
// Phase 1 (Opus) adds ILGPU accelerator, ShaderDebugService, and UnitTest registrations.
Console.WriteLine("[SpawnDev.Codecs.Demo] Planning phase — no codec implementations yet.");

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.Services.AddBlazorJSRuntime();
builder.Services.AddSingleton(sp => new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });

builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

await builder.Build().BlazorJSRunAsync();
