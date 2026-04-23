using SpawnDev.Codecs.Demo.Shared.UnitTests;

namespace SpawnDev.Codecs.Demo.UnitTests;

/// <summary>
/// Browser-side WebGPU-backend entry point for SpawnDev.Codecs tests. Inherits all
/// cross-platform tests from <see cref="CodecsTestBase"/>. When CELT ILGPU kernels land
/// in Phase 1a, this class will override accelerator creation to use
/// <c>SpawnDev.ILGPU.WebGPU</c> so kernel tests run on the browser's WebGPU backend.
/// </summary>
public class WebGPUCodecsTests : CodecsTestBase
{
    public WebGPUCodecsTests() : base() { }
}
