// Vp9StreamValidator tests against valid + invalid VP9 inputs.

using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9StreamValidator_BbbFixture_ReportsNoErrors()
    {
        using var stream = LoadBigBuckBunnyWebM();
        var container = new MatroskaContainer(stream);
        var video = container.Tracks.First(t => t.IsVideo);

        var packets = container.Frames
            .Where(f => f.TrackNumber == video.TrackNumber)
            .Select(f => (ReadOnlyMemory<byte>)f.Data);

        var result = Vp9StreamValidator.Validate(packets);
        Equal(true, result.IsValid);
        Equal(0, result.CountBy(Av1ValidationSeverity.Error));
    }

    [TestMethod]
    public void Vp9StreamValidator_EmptyPacketStream_ReportsError()
    {
        var result = Vp9StreamValidator.Validate(Array.Empty<ReadOnlyMemory<byte>>());
        Equal(false, result.IsValid);
        True(result.Findings.Any(f =>
            f.Severity == Av1ValidationSeverity.Error
            && f.Message.Contains("no parseable")),
            "expected 'no parseable coded frame' error");
    }

    [TestMethod]
    public void Vp9StreamValidator_BadPacketBytes_ReportsError()
    {
        // Just random bytes - won't even parse as a frame header.
        var bad = new byte[100];
        for (int i = 0; i < bad.Length; i++) bad[i] = (byte)i;
        var result = Vp9StreamValidator.Validate(new[] { (ReadOnlyMemory<byte>)bad });
        Equal(false, result.IsValid);
        True(result.CountBy(Av1ValidationSeverity.Error) >= 1,
            "expected at least one error");
    }
}
