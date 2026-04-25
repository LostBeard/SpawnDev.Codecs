// Tests for Vp9FrameSizeWithRefsParser (slice 223).

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static readonly (int W, int H)[] StandardRefSizes = new[]
    {
        (1280, 720),  // LAST
        (1920, 1080), // GOLDEN
        (640, 480),   // ALTREF
    };

    [TestMethod]
    public void Vp9FrameSizeWithRefs_LastRefSelected_FirstBitOne()
    {
        // bit 0 = 1 -> use LAST (1280x720). render_override = 0.
        var data = BitsToBytes((1, 1), (0, 1));

        var info = Vp9FrameSizeWithRefsParser.Parse(data, StandardRefSizes);

        Equal(0, info.RefFoundIdx);
        Equal(1280, info.FrameWidth);
        Equal(720, info.FrameHeight);
        Equal(false, info.RenderSizeOverride);
    }

    [TestMethod]
    public void Vp9FrameSizeWithRefs_GoldenRefSelected_FirstBitZero_SecondOne()
    {
        // ref bits: 0, 1 -> GOLDEN (1920x1080). Third bit not read.
        var data = BitsToBytes((0, 1), (1, 1), (0, 1));

        var info = Vp9FrameSizeWithRefsParser.Parse(data, StandardRefSizes);

        Equal(1, info.RefFoundIdx);
        Equal(1920, info.FrameWidth);
        Equal(1080, info.FrameHeight);
    }

    [TestMethod]
    public void Vp9FrameSizeWithRefs_AltRefSelected_TwoZerosThenOne()
    {
        // ref bits: 0, 0, 1 -> ALTREF (640x480).
        var data = BitsToBytes((0, 1), (0, 1), (1, 1), (0, 1));

        var info = Vp9FrameSizeWithRefsParser.Parse(data, StandardRefSizes);

        Equal(2, info.RefFoundIdx);
        Equal(640, info.FrameWidth);
        Equal(480, info.FrameHeight);
    }

    [TestMethod]
    public void Vp9FrameSizeWithRefs_NoRefSelected_ReadsExplicitSize()
    {
        // ref bits: 0, 0, 0 -> read explicit 16+16 width/height.
        // Then render_override = 0.
        // frame_width = (1024-1) + 1 = 1024, frame_height = (768-1) + 1 = 768
        var data = BitsToBytes(
            (0, 1), (0, 1), (0, 1),
            (1023, 16), (767, 16),  // width-1, height-1
            (0, 1));                 // render_override = 0

        var info = Vp9FrameSizeWithRefsParser.Parse(data, StandardRefSizes);

        Equal(-1, info.RefFoundIdx);
        Equal(1024, info.FrameWidth);
        Equal(768, info.FrameHeight);
        Equal(false, info.RenderSizeOverride);
    }

    [TestMethod]
    public void Vp9FrameSizeWithRefs_RenderOverride_ReadsRenderSize()
    {
        // LAST selected, render_override = 1, render = 800x600.
        var data = BitsToBytes(
            (1, 1),                  // LAST selected
            (1, 1),                  // render_override = 1
            (799, 16), (599, 16));   // render width-1, height-1

        var info = Vp9FrameSizeWithRefsParser.Parse(data, StandardRefSizes);

        Equal(0, info.RefFoundIdx);
        Equal(true, info.RenderSizeOverride);
        Equal(800, info.RenderWidth);
        Equal(600, info.RenderHeight);
    }

    [TestMethod]
    public void Vp9FrameSizeWithRefs_RejectsTooFewRefs()
    {
        var data = new byte[8];
        Throws<ArgumentException>(() =>
            Vp9FrameSizeWithRefsParser.Parse(data, new (int, int)[] { (1, 1) }));
    }

    [TestMethod]
    public void Vp9FrameSizeWithRefs_Constants_MatchLibvpx()
    {
        Equal(3, Vp9FrameSizeWithRefsParser.RefsPerFrame);
    }
}
