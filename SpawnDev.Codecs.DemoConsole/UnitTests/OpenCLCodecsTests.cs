using SpawnDev.Codecs.Demo.Shared.UnitTests;

namespace SpawnDev.Codecs.DemoConsole.UnitTests;

/// <summary>
/// Desktop OpenCL-backend entry point for SpawnDev.Codecs tests. Inherits all
/// cross-platform tests from <see cref="CodecsTestBase"/>. When CELT ILGPU kernels land
/// in Phase 1a, this class will override accelerator creation to use the ILGPU OpenCL
/// backend so kernel tests run on any OpenCL-capable device (GPU or CPU).
/// </summary>
public class OpenCLCodecsTests : CodecsTestBase
{
    public OpenCLCodecsTests() : base() { }
}
