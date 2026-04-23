using SpawnDev.Codecs.Demo.Shared.UnitTests;

namespace SpawnDev.Codecs.DemoConsole.UnitTests;

/// <summary>
/// Desktop CPU-backend entry point for SpawnDev.Codecs tests. Inherits all
/// cross-platform tests from <see cref="CodecsTestBase"/>. When CELT ILGPU kernels land
/// in Phase 1a, this class will override accelerator creation to use the ILGPU CPU
/// backend so kernel tests run on CPU via the standard ILGPU emulator.
/// </summary>
public class CpuCodecsTests : CodecsTestBase
{
    public CpuCodecsTests() : base() { }
}
