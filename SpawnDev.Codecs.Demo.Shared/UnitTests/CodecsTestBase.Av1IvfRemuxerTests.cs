// Av1IvfRemuxer tests against the BBB AV1 fixture.

using SpawnDev.Codecs.Video.Av1;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Av1IvfRemuxer_BbbFixture_ProducesIdenticalSize()
    {
        var src = LoadAv1Fixture();
        var remuxed = Av1IvfRemuxer.RemuxToBytes(src);
        Equal(src.Length, remuxed.Length);
    }

    [TestMethod]
    public void Av1IvfRemuxer_BbbFixture_ProducesByteIdenticalOutput()
    {
        var src = LoadAv1Fixture();
        var remuxed = Av1IvfRemuxer.RemuxToBytes(src);
        Equal(src.Length, remuxed.Length);
        for (int i = 0; i < src.Length; i++)
        {
            if (src[i] != remuxed[i])
                throw new Exception(
                    $"Byte {i}: source 0x{src[i]:X2} vs remux 0x{remuxed[i]:X2}");
        }
    }

    [TestMethod]
    public void Av1IvfRemuxer_BbbConfigSubstitution_ProducesByteIdenticalOutput()
    {
        // Build the BBB-equivalent SH config from observed bits, substitute
        // it via the writer, and verify the remuxed output is byte-equivalent
        // to source. This proves the closed-loop SH writer path works
        // end-to-end at the IVF level.
        var src = LoadAv1Fixture();
        var bbbConfig = new Av1SequenceHeaderConfig
        {
            SeqProfile = 0, SeqLevelIdx0 = 0, MaxFrameWidth = 320, MaxFrameHeight = 180,
            BitDepth = 8, SubsamplingX = 1, SubsamplingY = 1,
            EnableFilterIntra = true, EnableIntraEdgeFilter = true,
            EnableMaskedCompound = true, EnableWarpedMotion = true,
            EnableOrderHint = true, EnableRefFrameMvs = true, OrderHintBitsMinus1 = 6,
            SeqChooseScreenContentTools = true, SeqChooseIntegerMv = true,
            EnableCdef = true, ColorDescriptionPresent = true,
            ColorPrimaries = 2, TransferCharacteristics = 2, MatrixCoefficients = 5,
        };
        var remuxed = Av1IvfRemuxer.RemuxToBytesWithShSubstitution(src, bbbConfig);

        Equal(src.Length, remuxed.Length);
        for (int i = 0; i < src.Length; i++)
        {
            if (src[i] != remuxed[i])
                throw new Exception(
                    $"Byte {i}: source 0x{src[i]:X2} vs remux 0x{remuxed[i]:X2}");
        }
    }

    [TestMethod]
    public void Av1IvfRemuxer_BbbFixture_RemuxedStreamReParseable()
    {
        // The remuxed bytes must round-trip through Av1StreamAnalyzer
        // identically to the source.
        var src = LoadAv1Fixture();
        var remuxed = Av1IvfRemuxer.RemuxToBytes(src);

        var srcSummary = Av1StreamAnalyzer.Analyze(src);
        var rmxSummary = Av1StreamAnalyzer.Analyze(remuxed);

        Equal(srcSummary.IvfHeader.Width, rmxSummary.IvfHeader.Width);
        Equal(srcSummary.IvfHeader.Height, rmxSummary.IvfHeader.Height);
        Equal(srcSummary.TotalTemporalUnits, rmxSummary.TotalTemporalUnits);
        Equal(srcSummary.CodedFrames.Count, rmxSummary.CodedFrames.Count);
        Equal(srcSummary.ShowExistingFrames.Count, rmxSummary.ShowExistingFrames.Count);
        Equal(srcSummary.ObuCounts.Count, rmxSummary.ObuCounts.Count);
        foreach (var kv in srcSummary.ObuCounts)
            Equal(kv.Value, rmxSummary.ObuCounts[kv.Key]);
    }
}
