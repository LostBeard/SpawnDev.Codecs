using Microsoft.Extensions.DependencyInjection;
using SpawnDev.Codecs.DemoConsole;
using SpawnDev.Codecs.DemoConsole.UnitTests;
using SpawnDev.UnitTesting;

// Diagnostic commands (offline, no test harness). `wasm-vp9-dump` measures the VP9
// frame-entropy walker's EncodeFrameBody Wasm locals — the before/after for Geordi's
// CumulativeInlinedILBudget fix (ILGPU 4.9.16-local.1).
if (args.Length > 0 && args[0] == "wasm-vp9-dump")
    return await WasmVp9Dump.Run();

// Run SpawnDev.UnitTesting unit tests on desktop .NET via PlaywrightMultiTest harness.
try
{
    var services = new ServiceCollection();
    // One concrete test class per desktop ILGPU backend. Each currently inherits all
    // tests from CodecsTestBase; CELT kernel work will override per-backend accelerator.
    services.AddSingleton<CpuCodecsTests>();
    services.AddSingleton<CudaCodecsTests>();
    services.AddSingleton<OpenCLCodecsTests>();
    var sp = services.BuildServiceProvider();
    var runner = new UnitTestRunner(sp, true);
    await ConsoleRunner.Run(args, runner);
}
catch (Exception ex)
{
    Console.Error.WriteLine(ex.Message);
    return 1;
}
return 0;
