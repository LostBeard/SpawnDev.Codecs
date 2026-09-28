using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using SpawnDev.Codecs.Demo;
using SpawnDev.Codecs.Demo.UnitTests;
using SpawnDev.SpawnJS;

Console.WriteLine("[SpawnDev.Codecs.Demo] booting.");

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.Services.AddSpawnJSRuntime();
// Slot lifetime is manual in SpawnJS; watcher names leaks from owned wrappers/callbacks.
SpawnJSRuntime.EnableIDisposableWatcher = true;
builder.Services.AddSingleton(sp => new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });

builder.Services.AddSingleton<WebGPUCodecsTests>();
builder.Services.AddSingleton<WebGLCodecsTests>();
builder.Services.AddSingleton<WasmCodecsTests>();

builder.Services.AddSingleton<SpawnDev.ILGPU.Services.ShaderDebugService>();

builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

await builder.Build().SpawnJSRunAsync();
