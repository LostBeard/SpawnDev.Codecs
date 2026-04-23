using Microsoft.Extensions.DependencyInjection;
using SpawnDev.Codecs.DemoConsole.UnitTests;
using SpawnDev.UnitTesting;

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
