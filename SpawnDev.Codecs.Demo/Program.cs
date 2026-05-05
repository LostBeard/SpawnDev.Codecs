using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using SpawnDev.BlazorJS;
using SpawnDev.Codecs.Demo;
using SpawnDev.Codecs.Demo.UnitTests;

Console.WriteLine("[SpawnDev.Codecs.Demo] booting.");

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.Services.AddBlazorJSRuntime();
builder.Services.AddSingleton(sp => new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });

builder.Services.AddSingleton<WebGPUCodecsTests>();
builder.Services.AddSingleton<WebGLCodecsTests>();
builder.Services.AddSingleton<WasmCodecsTests>();

builder.Services.AddSingleton<SpawnDev.ILGPU.Services.ShaderDebugService>();

builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

await builder.Build().BlazorJSRunAsync();
