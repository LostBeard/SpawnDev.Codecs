using SpawnDev.Codecs.Demo.Shared.UnitTests;

namespace SpawnDev.Codecs.Demo.UnitTests;

/// <summary>
/// Browser-side Wasm-backend entry point for SpawnDev.Codecs tests. Inherits all
/// cross-platform tests from <see cref="CodecsTestBase"/>. When CELT ILGPU kernels land
/// in Phase 1a, this class will override accelerator creation to use the ILGPU Wasm
/// backend so kernel tests run in pure WebAssembly (no GPU).
/// </summary>
public class WasmCodecsTests : CodecsTestBase
{
    public WasmCodecsTests() : base() { }
}
