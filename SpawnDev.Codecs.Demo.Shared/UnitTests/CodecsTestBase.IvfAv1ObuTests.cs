// End-to-end tests for IVF reader + AV1 OBU parser. Drives a real
// AV1 stream (bbb_180_2s.ivf, generated from BBB.webm via ffmpeg
// libaom-av1) through the IVF reader and into the OBU parser,
// verifying the per-frame OBU stream structure across all 60
// frames of the fixture.
//
// This is the entry-point for AV1 decode in pure .NET - first time
// SpawnDev.Codecs has parsed real AV1 bitstream data.

using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static byte[] LoadAv1Fixture()
    {
        var assembly = typeof(CodecsTestBase).Assembly;
        const string resourceName =
            "SpawnDev.Codecs.Demo.Shared.TestData.bbb_180_2s.ivf";
        using var stream = assembly.GetManifestResourceStream(resourceName)
            ?? throw new FileNotFoundException(
                $"Missing embedded resource '{resourceName}'.");
        var bytes = new byte[(int)stream.Length];
        stream.ReadExactly(bytes);
        return bytes;
    }

    [TestMethod]
    public void IvfReader_BbbAv1Fixture_HeaderMatchesExpected()
    {
        var bytes = LoadAv1Fixture();
        var header = IvfReader.ParseHeader(bytes);
        Equal("AV01", header.FourCc);
        Equal(320, header.Width);
        Equal(180, header.Height);
        // ffmpeg writes num_frames = 0 for live-encoded streams,
        // but for our 2-second / 60-frame conversion it should be set.
        True(header.NumFrames > 0,
            $"expected non-zero num_frames, got {header.NumFrames}");
    }

    [TestMethod]
    public void IvfReader_BbbAv1Fixture_FrameCount()
    {
        var bytes = LoadAv1Fixture();
        int count = 0;
        long lastPts = -1;
        foreach (var frame in IvfReader.EnumerateFrames(bytes))
        {
            count++;
            True(frame.Data.Length > 0, $"frame {count} has empty payload");
            True(frame.Pts >= lastPts,
                $"frame {count} pts {frame.Pts} not monotonically >= prev {lastPts}");
            lastPts = frame.Pts;
        }
        // 30fps * 2s = 60 frames.
        Equal(60, count);
    }

    [TestMethod]
    public void Av1ObuParser_BbbFirstFrame_StartsWithTemporalDelimiterOrSequenceHeader()
    {
        // First IVF frame should contain a TemporalDelimiter OBU at the
        // start of every Temporal Unit, often followed by SequenceHeader
        // (for the first TU in the stream) and a Frame OBU.
        var bytes = LoadAv1Fixture();
        var first = IvfReader.EnumerateFrames(bytes).First();

        var obuList = Av1ObuParser.EnumerateObus(first.Data).ToList();
        True(obuList.Count > 0, "first frame must produce at least one OBU");

        // Either TD or SequenceHeader should be in the first 2 OBUs.
        var seenTypes = obuList.Select(o => o.Type).Take(3).ToHashSet();
        True(
            seenTypes.Contains(Av1ObuType.TemporalDelimiter)
            || seenTypes.Contains(Av1ObuType.SequenceHeader),
            $"first frame OBUs {string.Join(',', seenTypes)} - expected TD or SH near start");
    }

    [TestMethod]
    public void Av1SequenceHeader_BbbFirstFrame_ParsesProfileAndDimensions()
    {
        // Find the SequenceHeader OBU in the first AV1 frame and parse it.
        var bytes = LoadAv1Fixture();
        var firstFrame = IvfReader.EnumerateFrames(bytes).First();

        Av1SequenceHeader? sh = null;
        foreach (var obu in Av1ObuParser.EnumerateObus(firstFrame.Data))
        {
            if (obu.Type == Av1ObuType.SequenceHeader)
            {
                sh = Av1SequenceHeaderParser.Parse(
                    firstFrame.Data.Span.Slice(obu.PayloadOffset, obu.PayloadLength));
                break;
            }
        }
        True(sh is not null, "no SequenceHeader OBU found in first frame");

        // BBB at 320x180, 8-bit, 4:2:0.
        Equal(0, sh!.SeqProfile);
        Equal(false, sh.StillPicture);
        Equal(320, sh.MaxFrameWidth);
        Equal(180, sh.MaxFrameHeight);
        Equal(8, sh.BitDepth);
        Equal(false, sh.Monochrome);
        Equal(1, sh.SubsamplingX);
        Equal(1, sh.SubsamplingY);
    }

    [TestMethod]
    public void Av1SequenceHeader_BbbFirstFrame_ParsesAdvancedFlags()
    {
        // Verify the parser surfaces every conditional bit observed in
        // the libaom-encoded BBB SH (decoded from the bitstream by hand:
        // see inspect_bbb_sh.cs).
        var bytes = LoadAv1Fixture();
        var firstFrame = IvfReader.EnumerateFrames(bytes).First();

        Av1SequenceHeader? sh = null;
        foreach (var obu in Av1ObuParser.EnumerateObus(firstFrame.Data))
        {
            if (obu.Type == Av1ObuType.SequenceHeader)
            {
                sh = Av1SequenceHeaderParser.Parse(
                    firstFrame.Data.Span.Slice(obu.PayloadOffset, obu.PayloadLength));
                break;
            }
        }
        True(sh is not null, "no SequenceHeader OBU found");

        // Tools that affect frame parsing - libaom-av1 set these:
        Equal(true, sh!.EnableFilterIntra);
        Equal(true, sh.EnableIntraEdgeFilter);
        Equal(false, sh.EnableInterintraCompound);
        Equal(true, sh.EnableMaskedCompound);
        Equal(true, sh.EnableWarpedMotion);
        Equal(false, sh.EnableDualFilter);
        Equal(true, sh.EnableOrderHint);
        Equal(false, sh.EnableJntComp);
        Equal(true, sh.EnableRefFrameMvs);
        Equal(6, sh.OrderHintBitsMinus1);
        Equal(true, sh.SeqChooseScreenContentTools);
        Equal(2, sh.SeqForceScreenContentTools); // SELECT
        Equal(true, sh.SeqChooseIntegerMv);
        Equal(2, sh.SeqForceIntegerMv); // SELECT
        Equal(false, sh.EnableSuperres);
        Equal(true, sh.EnableCdef);
        Equal(false, sh.EnableRestoration);
        Equal(true, sh.ColorDescriptionPresent);
        Equal(2, sh.ColorPrimaries);          // UNSPECIFIED
        Equal(2, sh.TransferCharacteristics); // UNSPECIFIED
        Equal(5, sh.MatrixCoefficients);      // BT.709
        Equal(0, sh.ChromaSamplePosition);
        Equal(false, sh.SeparateUvDeltas);
        Equal(false, sh.FilmGrainParamsPresent);
        Equal(false, sh.Use128x128Superblock);
        Equal(0, sh.SeqLevelIdx0);
    }

    [TestMethod]
    public void Av1FrameHeader_BbbFirstFrame_FirstFrameIsKey()
    {
        // The first IVF frame must be a keyframe.
        var bytes = LoadAv1Fixture();
        var firstFrame = IvfReader.EnumerateFrames(bytes).First();

        Av1SequenceHeader? sh = null;
        Av1FrameHeader? fh = null;
        foreach (var obu in Av1ObuParser.EnumerateObus(firstFrame.Data))
        {
            if (obu.Type == Av1ObuType.SequenceHeader)
            {
                sh = Av1SequenceHeaderParser.Parse(
                    firstFrame.Data.Span.Slice(obu.PayloadOffset, obu.PayloadLength));
            }
            else if (obu.IsCodedFrameData && sh is not null)
            {
                fh = Av1FrameHeaderParser.Parse(
                    firstFrame.Data.Span.Slice(obu.PayloadOffset, obu.PayloadLength), sh);
                break;
            }
        }
        True(sh is not null, "expected SH OBU before frame OBU");
        True(fh is not null, "expected Frame / FrameHeader OBU");
        Equal(Av1FrameType.KeyFrame, fh!.FrameType);
        Equal(true, fh.ShowFrame);
        Equal(true, fh.FrameIsIntra);
    }

    [TestMethod]
    public void Av1FrameHeader_BbbFirstFrame_ParsesPostPrefixFields()
    {
        // After my SH-driven prefix extension, the parser surfaces
        // disable_cdf_update + allow_screen_content_tools + force_integer_mv
        // for BBB's first keyframe. Verify these get plausible values.
        var bytes = LoadAv1Fixture();
        var firstFrame = IvfReader.EnumerateFrames(bytes).First();

        Av1SequenceHeader? sh = null;
        Av1FrameHeader? fh = null;
        foreach (var obu in Av1ObuParser.EnumerateObus(firstFrame.Data))
        {
            if (obu.Type == Av1ObuType.SequenceHeader)
                sh = Av1SequenceHeaderParser.Parse(
                    firstFrame.Data.Span.Slice(obu.PayloadOffset, obu.PayloadLength));
            else if (obu.IsCodedFrameData && sh is not null)
            {
                fh = Av1FrameHeaderParser.Parse(
                    firstFrame.Data.Span.Slice(obu.PayloadOffset, obu.PayloadLength), sh);
                break;
            }
        }
        True(sh is not null && fh is not null, "expected SH + FH");

        // BBB's first keyframe: disable_cdf_update is a 1-bit flag, value
        // is what libaom chose. Just assert it's parseable (no throw).
        // allow_screen_content_tools: BBB SH says SELECT (2), so parser
        // reads a 1-bit f(1) value -> resolves to 0 or 1.
        True(fh!.AllowScreenContentTools == 0 || fh.AllowScreenContentTools == 1,
            $"allow_screen_content_tools must be 0 or 1, got {fh.AllowScreenContentTools}");

        // force_integer_mv for an intra frame is forced to 1.
        Equal(true, fh.FrameIsIntra);
        Equal(1, fh.ForceIntegerMv);
    }

    [TestMethod]
    public void Av1FrameHeader_BbbAllFrames_FrameTypesPlausible()
    {
        // Walk every frame, count frame types. BBB should produce 1
        // keyframe + many inter.
        var bytes = LoadAv1Fixture();
        Av1SequenceHeader? sh = null;
        var typeCounts = new Dictionary<Av1FrameType, int>();
        int parsed = 0;
        foreach (var ivfFrame in IvfReader.EnumerateFrames(bytes))
        {
            foreach (var obu in Av1ObuParser.EnumerateObus(ivfFrame.Data))
            {
                if (obu.Type == Av1ObuType.SequenceHeader)
                {
                    sh = Av1SequenceHeaderParser.Parse(
                        ivfFrame.Data.Span.Slice(obu.PayloadOffset, obu.PayloadLength));
                }
                else if (obu.IsCodedFrameData && sh is not null)
                {
                    var fh = Av1FrameHeaderParser.Parse(
                        ivfFrame.Data.Span.Slice(obu.PayloadOffset, obu.PayloadLength), sh);
                    typeCounts.TryGetValue(fh.FrameType, out int c);
                    typeCounts[fh.FrameType] = c + 1;
                    parsed++;
                    break;
                }
            }
        }
        True(parsed >= 60, $"expected at least 60 frame headers parsed; got {parsed}");
        True(typeCounts.ContainsKey(Av1FrameType.KeyFrame),
            $"expected at least one keyframe; saw types {string.Join(',', typeCounts.Keys)}");
        True(typeCounts.ContainsKey(Av1FrameType.InterFrame),
            $"expected inter frames; saw types {string.Join(',', typeCounts.Keys)}");
    }

    [TestMethod]
    public void Av1ObuParser_BbbAllFrames_ParseWithoutErrors()
    {
        // Drive every frame through the OBU parser and verify all OBUs
        // declare a known type and have non-negative payload lengths.
        var bytes = LoadAv1Fixture();
        int frameCount = 0;
        int totalObus = 0;
        var typeCounts = new Dictionary<Av1ObuType, int>();
        foreach (var frame in IvfReader.EnumerateFrames(bytes))
        {
            frameCount++;
            foreach (var obu in Av1ObuParser.EnumerateObus(frame.Data))
            {
                totalObus++;
                True(obu.PayloadLength >= 0,
                    $"frame {frameCount}: OBU {obu.Type} payload {obu.PayloadLength} negative");
                True(obu.PayloadOffset + obu.PayloadLength <= frame.Data.Length,
                    $"frame {frameCount}: OBU {obu.Type} overruns frame");
                typeCounts.TryGetValue(obu.Type, out int c);
                typeCounts[obu.Type] = c + 1;
            }
        }
        True(frameCount > 0, "expected at least 1 frame");
        True(totalObus >= frameCount, $"expected at least one OBU per frame; got {totalObus} OBUs / {frameCount} frames");
        // libaom-av1 normally produces at least 1 SequenceHeader and 1+ Frame OBUs.
        True(typeCounts.ContainsKey(Av1ObuType.SequenceHeader),
            $"expected SequenceHeader; saw types: {string.Join(',', typeCounts.Keys)}");
        True(typeCounts.ContainsKey(Av1ObuType.Frame)
                || typeCounts.ContainsKey(Av1ObuType.TileGroup),
            $"expected Frame or TileGroup OBU; saw: {string.Join(',', typeCounts.Keys)}");
    }
}
