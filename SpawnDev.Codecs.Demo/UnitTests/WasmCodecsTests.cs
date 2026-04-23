using SpawnDev.Codecs.Demo.Shared.UnitTests;

namespace SpawnDev.Codecs.Demo.UnitTests;

/// <summary>
/// Concrete browser-side entry point for SpawnDev.Codecs tests. Inherits all cross-platform
/// tests from <see cref="CodecsTestBase"/>. Browser-only tests (if any) are added here
/// as additional <c>[TestMethod]</c> methods.
/// </summary>
public class WasmCodecsTests : CodecsTestBase
{
    public WasmCodecsTests() : base() { }
}
