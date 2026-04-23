using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using SpawnDev.BlazorJS;
using SpawnDev.Codecs.Demo;
using SpawnDev.Codecs.Demo.UnitTests;

// Planning phase (2026-04-23). Minimal Blazor WASM bootstrap with unit-test registration.
// Phase 1 (Opus) adds ILGPU accelerator + ShaderDebugService once the first kernels land.
Console.WriteLine("[SpawnDev.Codecs.Demo] Phase 0 scaffolding + Phase 1a entropy coders + Opus packet parser.");

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.Services.AddBlazorJSRuntime();
builder.Services.AddSingleton(sp => new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });

// SpawnDev.UnitTesting: register one concrete test class per browser backend so
// PlaywrightMultiTest + UnitTestsView discover them. Each class currently inherits
// all tests from CodecsTestBase; when CELT ILGPU kernels land each will override
// accelerator creation to exercise its specific ILGPU backend.
// (IServiceCollection is already singleton-registered by BackgroundServiceManager via
//  AddBlazorJSRuntime -> GetBlazorJSRuntime -> GetBackgroundServiceManager, so the
//  /tests page's [Inject] IServiceCollection resolves without any extra wiring.)
builder.Services.AddSingleton<WebGPUCodecsTests>();
builder.Services.AddSingleton<WebGLCodecsTests>();
builder.Services.AddSingleton<WasmCodecsTests>();

builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

await builder.Build().BlazorJSRunAsync();
