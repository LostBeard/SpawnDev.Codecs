namespace SpawnDev.Codecs.Tests;

/// <summary>
/// Placeholder test so the Tests project has at least one xunit discovery target
/// while we build out Phase 1 (Opus). Delete when the first real codec tests land.
/// </summary>
public class PlaceholderTests
{
    [Fact]
    public void PlanningPhase_ProjectIsBuildable()
    {
        Assert.Equal("Planning (2026-04-23). No codec implementations yet.", SpawnDevCodecs.Status);
    }
}
