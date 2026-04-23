namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Base class for cross-platform SpawnDev.Codecs tests. Concrete subclasses live in
/// <c>SpawnDev.Codecs.Demo</c> (browser, runs via PlaywrightMultiTest against Blazor WASM)
/// and <c>SpawnDev.Codecs.DemoConsole</c> (desktop, runs via PlaywrightMultiTest against
/// a published .NET console exe).
///
/// Tests are split into partial files by subject so each subject's coverage is self-contained
/// and easy to locate. See the matching <c>CodecsTestBase.*.cs</c> files.
/// </summary>
public abstract partial class CodecsTestBase
{
}
