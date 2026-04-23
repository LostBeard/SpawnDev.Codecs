using SpawnDev.Codecs.Demo.Shared.UnitTests;

namespace SpawnDev.Codecs.Demo.UnitTests;

/// <summary>
/// Browser-side WebGL-backend entry point for SpawnDev.Codecs tests. Inherits all
/// cross-platform tests from <see cref="CodecsTestBase"/>. When CELT ILGPU kernels land
/// in Phase 1a, this class will override accelerator creation to use
/// <c>SpawnDev.ILGPU.WebGL</c> so kernel tests run on the browser's WebGL2 backend.
/// </summary>
public class WebGLCodecsTests : CodecsTestBase
{
    public WebGLCodecsTests() : base() { }
}
