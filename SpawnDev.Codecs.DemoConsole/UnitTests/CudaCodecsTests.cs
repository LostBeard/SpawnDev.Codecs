using SpawnDev.Codecs.Demo.Shared.UnitTests;

namespace SpawnDev.Codecs.DemoConsole.UnitTests;

/// <summary>
/// Desktop CUDA-backend entry point for SpawnDev.Codecs tests. Inherits all
/// cross-platform tests from <see cref="CodecsTestBase"/>. When CELT ILGPU kernels land
/// in Phase 1a, this class will override accelerator creation to use the ILGPU CUDA
/// backend so kernel tests run on NVIDIA GPUs (PTX codegen).
/// </summary>
public class CudaCodecsTests : CodecsTestBase
{
    public CudaCodecsTests() : base() { }
}
