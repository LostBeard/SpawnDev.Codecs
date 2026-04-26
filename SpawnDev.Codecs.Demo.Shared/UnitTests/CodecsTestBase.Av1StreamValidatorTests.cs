// Av1StreamValidator tests against valid + invalid AV1 inputs.

using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Av1StreamValidator_BbbFixture_ReportsNoErrors()
    {
        var bytes = LoadAv1Fixture();
        var result = Av1StreamValidator.Validate(bytes);
        Equal(true, result.IsValid);
        Equal(0, result.CountBy(Av1ValidationSeverity.Error));
        // BBB is well-formed - no warnings on the headline checks.
        // (FourCc=AV01, dims match SH, every TU starts with TD, has SH,
        // first frame is KeyFrame.)
        Equal(0, result.CountBy(Av1ValidationSeverity.Warning));
    }

    [TestMethod]
    public void Av1StreamValidator_RejectsBadIvfHeader()
    {
        // Stream that is NOT a valid IVF (just random bytes).
        var bytes = new byte[64];
        for (int i = 0; i < bytes.Length; i++) bytes[i] = (byte)(i + 1);
        var result = Av1StreamValidator.Validate(bytes);
        Equal(false, result.IsValid);
        True(result.CountBy(Av1ValidationSeverity.Error) >= 1,
            "expected at least one Error finding");
    }

    [TestMethod]
    public void Av1StreamValidator_StreamWithoutSequenceHeader_ReportsError()
    {
        // Build a minimal IVF with a TD OBU but no SH OBU.
        using var ms = new MemoryStream();
        var writer = new IvfWriter(ms, "AV01", 320, 180);
        var td = Av1ObuWriter.EmitObu(Av1ObuType.TemporalDelimiter, ReadOnlySpan<byte>.Empty);
        writer.WriteFrame(td, 0);
        writer.Finish();

        var result = Av1StreamValidator.Validate(ms.ToArray());
        True(result.Findings.Any(f => f.Severity == Av1ValidationSeverity.Error
            && f.Message.Contains("SequenceHeader")),
            "expected an error about missing SequenceHeader");
    }

    [TestMethod]
    public void Av1StreamValidator_FoundCount_MatchesIndividualCounts()
    {
        var bytes = LoadAv1Fixture();
        var result = Av1StreamValidator.Validate(bytes);
        int sum = result.CountBy(Av1ValidationSeverity.Info)
            + result.CountBy(Av1ValidationSeverity.Warning)
            + result.CountBy(Av1ValidationSeverity.Error);
        Equal(result.Findings.Count, sum);
    }
}
