// Round-trip tests for Vp9FrameHeaderParser <-> Vp9FrameHeaderWriter.
// Build a known Vp9FrameHeader, serialize via the writer, parse via
// the parser, verify every field round-trips bit-exactly. This pins
// the encoder + decoder pair to agree on the wire format.
//
// Plus an "ffmpeg-style" check: the byte sequence produced by the
// writer for a fixed keyframe matches the byte sequence the existing
// BuildVp9Keyframe_Profile0_Bt601 test helper produces - so writer
// output is byte-identical to known-good hand-built bits.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static Vp9FrameHeader BuildKeyframeHeader(int w, int h)
    {
        return new Vp9FrameHeader
        {
            Profile = 0,
            ShowExistingFrame = false,
            FrameType = Vp9FrameType.Key,
            ShowFrame = true,
            ErrorResilientMode = false,
            IntraOnly = false,
            BitDepth = 8,
            ColorSpace = Vp9ColorSpace.Bt601,
            ColorRangeFull = false,
            SubsamplingX = true,
            SubsamplingY = true,
            FrameWidth = w,
            FrameHeight = h,
            RenderWidth = 0,
            RenderHeight = 0,
        };
    }

    [TestMethod]
    public void Vp9FrameHeaderWriter_Keyframe_RoundTripsViaParser()
    {
        var original = BuildKeyframeHeader(1920, 1080);
        var bytes = Vp9FrameHeaderWriter.WriteHeaderPrefix(original);
        var parsed = Vp9FrameHeaderParser.Parse(bytes);

        Equal(original.Profile, parsed.Profile);
        Equal(original.FrameType, parsed.FrameType);
        Equal(original.ShowFrame, parsed.ShowFrame);
        Equal(original.ErrorResilientMode, parsed.ErrorResilientMode);
        Equal(original.BitDepth, parsed.BitDepth);
        Equal(original.ColorSpace, parsed.ColorSpace);
        Equal(original.ColorRangeFull, parsed.ColorRangeFull);
        Equal(original.SubsamplingX, parsed.SubsamplingX);
        Equal(original.SubsamplingY, parsed.SubsamplingY);
        Equal(original.FrameWidth, parsed.FrameWidth);
        Equal(original.FrameHeight, parsed.FrameHeight);
    }

    [TestMethod]
    public void Vp9FrameHeaderWriter_Keyframe_BytesMatchHandBuilt()
    {
        // Hand-built helper from the parser test suite is the gold
        // reference. Writer output must match byte-for-byte.
        var handBuilt = BuildVp9Keyframe_Profile0_Bt601(640, 480);
        var via = Vp9FrameHeaderWriter.WriteHeaderPrefix(BuildKeyframeHeader(640, 480));
        Equal(handBuilt.Length, via.Length);
        for (int i = 0; i < handBuilt.Length; i++)
        {
            Equal(handBuilt[i], via[i]);
        }
    }

    [TestMethod]
    public void Vp9FrameHeaderWriter_DimensionRange_PreservesEdgeValues()
    {
        // Keyframes at the smallest and largest legal dimensions.
        foreach (var (w, h) in new[] { (1, 1), (320, 180), (3840, 2160), (65536, 65536) })
        {
            var bytes = Vp9FrameHeaderWriter.WriteHeaderPrefix(BuildKeyframeHeader(w, h));
            var parsed = Vp9FrameHeaderParser.Parse(bytes);
            Equal(w, parsed.FrameWidth);
            Equal(h, parsed.FrameHeight);
        }
    }

    [TestMethod]
    public void Vp9FrameHeaderWriter_RenderSizeOverride_RoundTrips()
    {
        var original = new Vp9FrameHeader
        {
            Profile = 0,
            ShowExistingFrame = false,
            FrameType = Vp9FrameType.Key,
            ShowFrame = true,
            ErrorResilientMode = false,
            IntraOnly = false,
            BitDepth = 8,
            ColorSpace = Vp9ColorSpace.Bt709,
            ColorRangeFull = true,
            SubsamplingX = true,
            SubsamplingY = true,
            FrameWidth = 1920,
            FrameHeight = 1080,
            RenderWidth = 1280,
            RenderHeight = 720,
        };
        var bytes = Vp9FrameHeaderWriter.WriteHeaderPrefix(original);
        var parsed = Vp9FrameHeaderParser.Parse(bytes);
        Equal(1920, parsed.FrameWidth);
        Equal(1080, parsed.FrameHeight);
        Equal(1280, parsed.RenderWidth);
        Equal(720, parsed.RenderHeight);
        Equal(true, parsed.ColorRangeFull);
        Equal(Vp9ColorSpace.Bt709, parsed.ColorSpace);
    }

    [TestMethod]
    public void Vp9FrameHeaderWriter_ShowExistingFrame_RoundTrips()
    {
        var original = new Vp9FrameHeader
        {
            Profile = 0,
            ShowExistingFrame = true,
            FrameToShowMapIdx = 5,
        };
        var bytes = Vp9FrameHeaderWriter.WriteHeaderPrefix(original);
        var parsed = Vp9FrameHeaderParser.Parse(bytes);
        Equal(true, parsed.ShowExistingFrame);
        Equal(5, parsed.FrameToShowMapIdx);
        Equal(0, parsed.Profile);
    }
}
