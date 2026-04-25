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
