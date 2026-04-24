using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Demo.Shared.UnitTests;
using SpawnDev.ILGPU.Wasm;

namespace SpawnDev.Codecs.Demo.UnitTests;

/// <summary>
/// Browser-side Wasm-backend entry point for SpawnDev.Codecs tests. Inherits
/// every cross-platform test from <see cref="CodecsTestBase"/> and runs ILGPU
/// kernel tests on the pure-WebAssembly backend (no GPU involvement).
/// </summary>
public class WasmCodecsTests : CodecsTestBase
{
    protected override async ValueTask<(Context, Accelerator)> CreateKernelAcceleratorAsync()
    {
        var builder = Context.Create();
        builder.Wasm();
        var ctx = builder.ToContext();
        var acc = await ctx.CreateWasmAcceleratorAsync();
        return (ctx, acc);
    }
}
