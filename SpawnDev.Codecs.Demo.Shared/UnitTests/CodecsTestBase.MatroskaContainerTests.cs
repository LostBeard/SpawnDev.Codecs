// Tests for MatroskaContainer against the bundled Big Buck Bunny WebM
// fixture. Exercises the full production path: file bytes -> SpawnDev.EBML
// parse -> /Segment/Tracks walk -> MatroskaTrack records. This is the same
// code path the Phase 1b VP8 / VP9 / AV1 / Opus frame extractors will use
// to route packets to the right decoder.

using System.IO;
using System.Reflection;
using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Load the embedded Big Buck Bunny 10s WebM fixture. Works identically
    /// on desktop and Blazor WASM - the asset is compiled into the test
    /// assembly as a manifest resource.
    /// </summary>
    private static Stream LoadBigBuckBunnyWebM()
    {
        var assembly = typeof(CodecsTestBase).Assembly;
        const string resourceName =
            "SpawnDev.Codecs.Demo.Shared.TestData.Big_Buck_Bunny_180_10s.webm";
        var stream = assembly.GetManifestResourceStream(resourceName);
        if (stream is null)
        {
            var available = string.Join(",", assembly.GetManifestResourceNames());
            throw new FileNotFoundException(
                $"Missing embedded resource '{resourceName}'. Available: {available}");
        }
        return stream;
    }

    [TestMethod]
    public void MatroskaContainer_OnBigBuckBunnyWebM_ReportsWebMDocType()
    {
        using var stream = LoadBigBuckBunnyWebM();
        var container = new MatroskaContainer(stream);
        True(container.IsWebM, $"expected WebM, got DocType '{container.DocType}'");
        False(container.IsMatroska, "must not also be flagged as Matroska");
    }

    [TestMethod]
    public void MatroskaContainer_OnBigBuckBunnyWebM_EnumeratesTracks()
    {
        using var stream = LoadBigBuckBunnyWebM();
        var container = new MatroskaContainer(stream);
        var tracks = container.Tracks.ToList();
        // Big Buck Bunny WebM (VP8 video only) - at least one track exists.
        True(tracks.Count > 0, "expected at least one track");
        foreach (var t in tracks)
        {
            True(t.TrackNumber > 0, $"TrackNumber must be > 0, got {t.TrackNumber}");
            True(!string.IsNullOrEmpty(t.CodecId), "CodecID must be populated");
        }
    }

    [TestMethod]
    public void MatroskaContainer_BigBuckBunny_HasVideoTrack_WithVp9Codec()
    {
        // The bundled fixture is a VP9 WebM. Exercise the full detection
        // path: enumerate tracks, find the video one, check its codec ID
        // matches the Matroska codec-registry string for VP9.
        using var stream = LoadBigBuckBunnyWebM();
        var container = new MatroskaContainer(stream);
        var video = container.Tracks.FirstOrDefault(t => t.IsVideo);
        True(video != null, "expected at least one video track");
        Equal("V_VP9", video!.CodecId);
    }

    [TestMethod]
    public void MatroskaContainer_OnBigBuckBunnyWebM_YieldsVideoFrames()
    {
        using var stream = LoadBigBuckBunnyWebM();
        var container = new MatroskaContainer(stream);
        var videoTrack = container.Tracks.FirstOrDefault(t => t.IsVideo);
        True(videoTrack != null, "expected a video track");

        var videoFrames = container.Frames
            .Where(f => f.TrackNumber == videoTrack!.TrackNumber)
            .ToList();
        True(videoFrames.Count > 0, $"expected video frames on track {videoTrack!.TrackNumber}");
        // Every frame must have non-empty payload and be tagged with the
        // correct track number.
        foreach (var f in videoFrames)
        {
            True(f.Data.Length > 0, "video frame data must be non-empty");
            Equal(videoTrack.TrackNumber, f.TrackNumber);
        }
    }

    [TestMethod]
    public void MatroskaContainer_Frames_AreTimestampOrdered_WithinTrack()
    {
        using var stream = LoadBigBuckBunnyWebM();
        var container = new MatroskaContainer(stream);
        var videoTrack = container.Tracks.First(t => t.IsVideo);
        var frames = container.Frames
            .Where(f => f.TrackNumber == videoTrack.TrackNumber)
            .ToList();
        True(frames.Count > 1, "need at least 2 frames to test ordering");
        // Within a single track, timestamps should be strictly non-decreasing
        // (Matroska doesn't require strict monotonicity due to B-frames,
        // but the typical VP9 WebM we bundle has no B-frames).
        for (int i = 1; i < frames.Count; i++)
        {
            True(frames[i].Timestamp >= frames[i - 1].Timestamp,
                $"frame {i} ts {frames[i].Timestamp} < frame {i - 1} ts {frames[i - 1].Timestamp}");
        }
    }

    [TestMethod]
    public void MatroskaContainer_SegmentInfo_ExposesTimestampScale_ForBigBuckBunny()
    {
        using var stream = LoadBigBuckBunnyWebM();
        var container = new MatroskaContainer(stream);
        var info = container.SegmentInfo;
        // Default per Matroska spec is 1,000,000 (ticks = ms). Most WebM
        // encoders keep this; zero would be invalid (range="not 0").
        True(info.TimestampScaleNs > 0, "TimestampScale must be > 0");
    }

    [TestMethod]
    public void MatroskaContainer_SegmentInfo_DurationConvertsToRealTimeSpan()
    {
        using var stream = LoadBigBuckBunnyWebM();
        var container = new MatroskaContainer(stream);
        var info = container.SegmentInfo;
        // The fixture is ~10 seconds. We accept any Duration > 5s to keep
        // the test robust against re-encode variation.
        True(info.DurationTimeSpan.HasValue, "Duration must be populated for this WebM");
        True(info.DurationTimeSpan!.Value.TotalSeconds > 5,
            $"Duration must be > 5s, got {info.DurationTimeSpan.Value.TotalSeconds}s");
        True(info.DurationTimeSpan.Value.TotalSeconds < 60,
            $"Duration must be < 60s, got {info.DurationTimeSpan.Value.TotalSeconds}s");
    }

    [TestMethod]
    public void MatroskaContainer_SegmentInfo_DefaultsTimestampScale_WhenElementAbsent()
    {
        // Hand-build a minimal WebM with no /Segment/Info. The container
        // should still return a populated SegmentInfo object with the
        // spec-default TimestampScale of 1,000,000.
        var headerBytes = new byte[] { 0x1A, 0x45, 0xDF, 0xA3, 0x80 };
        // Segment element with size-unknown payload = 0 (just header).
        var segmentBytes = new byte[] { 0x18, 0x53, 0x80, 0x67, 0x80 };
        var buf = new byte[headerBytes.Length + segmentBytes.Length];
        Array.Copy(headerBytes, 0, buf, 0, headerBytes.Length);
        Array.Copy(segmentBytes, 0, buf, headerBytes.Length, segmentBytes.Length);
        using var stream = new MemoryStream(buf);
        var container = new MatroskaContainer(stream);
        var info = container.SegmentInfo;
        Equal(1_000_000UL, info.TimestampScaleNs);
        True(info.DurationTicks is null, "Duration must be null when absent");
    }

    [TestMethod]
    public void MatroskaContainer_Frames_FirstVideoFrame_IsKeyframe()
    {
        // The very first video frame in any well-formed stream must be a
        // keyframe - otherwise there's nothing for a decoder to reference.
        using var stream = LoadBigBuckBunnyWebM();
        var container = new MatroskaContainer(stream);
        var videoTrack = container.Tracks.First(t => t.IsVideo);
        var firstVideoFrame = container.Frames
            .First(f => f.TrackNumber == videoTrack.TrackNumber);
        True(firstVideoFrame.IsKeyframe,
            "first video frame must be a keyframe (SimpleBlock with bit 7 set)");
    }

    [TestMethod]
    public void MatroskaContainer_ThrowsOnNullStream()
    {
        bool threw = false;
        try { _ = new MatroskaContainer(null!); }
        catch (ArgumentNullException) { threw = true; }
        True(threw, "null stream must throw");
    }

    [TestMethod]
    public void MatroskaContainer_ThrowsOnNonEbmlStream()
    {
        using var garbage = new MemoryStream(new byte[] { 0, 0, 0, 0, 0, 0, 0, 0 });
        bool threw = false;
        try { _ = new MatroskaContainer(garbage); }
        catch (InvalidDataException) { threw = true; }
        True(threw, "non-EBML stream must throw InvalidDataException");
    }
}
