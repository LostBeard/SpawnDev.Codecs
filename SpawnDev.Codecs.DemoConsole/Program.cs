using Microsoft.Extensions.DependencyInjection;
using SpawnDev.Codecs.DemoConsole;
using SpawnDev.Codecs.DemoConsole.UnitTests;
using SpawnDev.UnitTesting;

// Diagnostic commands (offline, no test harness). `wasm-vp9-dump` measures the VP9
// frame-entropy walker's EncodeFrameBody Wasm locals — the before/after for Geordi's
// CumulativeInlinedILBudget fix (ILGPU 4.9.16-local.1).
if (args.Length > 0 && args[0] == "wasm-vp9-dump")
    return await WasmVp9Dump.Run();

// `weightpack-bench <file.safetensors> [maxMB] [cuda|opencl|cpu] [reps]`: WeightPackGpu on real model weights.
if (args.Length > 0 && args[0] == "weightpack-bench")
    return await WeightPackBench.Run(args);

// `gdeflate-bench <file.gdf> <original> [cuda|opencl|cpu] [reps]`: GDeflateDecompressorGpu on a reference stream.
if (args.Length > 0 && args[0] == "gdeflate-bench")
    return await GDeflateBench.Run(args);

// `gdeflate-encode <in> <out.gdf> [level] [cpu|cuda|opencl]`: GDeflateCompressorGpu, for the reference check.
if (args.Length > 0 && args[0] == "gdeflate-encode")
    return await GDeflateEncode.Run(args);

// `zlib-encode <in> <out.zz> [level] [cpu|cuda|opencl] [time] [wave]`: ZlibCompressorGpu speed bench.
if (args.Length > 0 && args[0] == "zlib-encode")
    return await ZlibEncode.Run(args);

// Run SpawnDev.UnitTesting unit tests on desktop .NET via PlaywrightMultiTest harness.
try
{
    var services = new ServiceCollection();
    // One concrete test class per desktop ILGPU backend. Each currently inherits all
    // tests from CodecsTestBase; CELT kernel work will override per-backend accelerator.
    services.AddSingleton<CpuCodecsTests>();
    services.AddSingleton<CudaCodecsTests>();
    services.AddSingleton<OpenCLCodecsTests>();
    services.AddSingleton<VelocityCodecsTests>();
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
