// Tests for Vp9FrameHeaderParser + Vp9BitReader. Hand-builds synthetic
// VP9 uncompressed headers and verifies each field decodes correctly.
// Also parses every video keyframe out of the bundled Big Buck Bunny
// WebM and validates dimensions + profile consistency.

using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Assemble a VP9 uncompressed keyframe header with the given dimensions,
    /// profile 0, 8-bit, 4:2:0, Bt601, studio range, no render override.
    /// Bits are packed MSB-first; this helper exists so tests can reason in
    /// terms of VP9 semantics rather than hand-counting bits.
    /// </summary>
    private static byte[] BuildVp9Keyframe_Profile0_Bt601(int width, int height)
    {
        // Expected bit sequence:
        //   marker           2  = 10
        //   profile_low      1  = 0
        //   profile_high     1  = 0
        //   show_existing    1  = 0
        //   frame_type       1  = 0 (KEY)
        //   show_frame       1  = 1
        //   error_resilient  1  = 0
        //   sync_code        24 = 0x49 0x83 0x42
        //   color_config:
        //     color_space    3  = 1 (Bt601)
        //     color_range    1  = 0 (studio)
        //   frame_width_m1   16 = width-1 BE
        //   frame_height_m1  16 = height-1 BE
        //   render_different 1  = 0
        // Total bits before trailing: 2+1+1+1+1+1+1+24+3+1+16+16+1 = 69 bits.
        // Pack MSB-first via a BitWriter helper inline.
        var bw = new BitPacker();
        bw.Write(0b10, 2);     // marker
        bw.Write(0, 1);        // profile_low
        bw.Write(0, 1);        // profile_high
        bw.Write(0, 1);        // show_existing
        bw.Write(0, 1);        // frame_type = KEY
        bw.Write(1, 1);        // show_frame
        bw.Write(0, 1);        // error_resilient
        bw.Write(0x49, 8);
        bw.Write(0x83, 8);
        bw.Write(0x42, 8);
        bw.Write(1, 3);        // color_space = Bt601
        bw.Write(0, 1);        // color_range = studio
        bw.Write((uint)(width - 1), 16);
        bw.Write((uint)(height - 1), 16);
        bw.Write(0, 1);        // render_different
        return bw.ToBytes();
    }

    private sealed class BitPacker
    {
        private readonly List<byte> _bytes = new();
        private int _cur = 0;
        private int _bits = 0;
        public void Write(uint value, int nBits)
        {
            for (int i = nBits - 1; i >= 0; i--)
            {
                int bit = (int)((value >> i) & 1u);
                _cur = (_cur << 1) | bit;
                _bits++;
                if (_bits == 8)
                {
                    _bytes.Add((byte)_cur);
                    _cur = 0;
                    _bits = 0;
                }
            }
        }
        public byte[] ToBytes()
        {
            if (_bits > 0) _bytes.Add((byte)(_cur << (8 - _bits)));
            return _bytes.ToArray();
        }
    }

    // -------- BitReader unit tests ----------------------------------------

    [TestMethod]
    public void Vp9BitReader_ReadsMsbFirst_AcrossByteBoundaries()
    {
        // 2 bytes: 0b1010_1100 0b1111_0011 = 0xAC 0xF3.
        // Read 3 bits -> 0b101 = 5
        // Read 5 bits -> 0b01100 = 12
        // Read 8 bits -> 0b11110011 = 0xF3
        var data = new byte[] { 0xAC, 0xF3 };
        var r = new Vp9BitReader(data);
        Equal(5u, r.ReadBits(3));
        Equal(12u, r.ReadBits(5));
        Equal(0xF3u, r.ReadBits(8));
        Equal(0, r.BitsRemaining);
    }

    [TestMethod]
    public void Vp9BitReader_ReadsZeroBits_ReturnsZero()
    {
        var r = new Vp9BitReader(new byte[] { 0xFF });
        Equal(0u, r.ReadBits(0));
        Equal(8, r.BitsRemaining);
    }

    [TestMethod]
    public void Vp9BitReader_ReadingPastEnd_Throws()
    {
        var r = new Vp9BitReader(new byte[] { 0xFF });
        _ = r.ReadBits(8);
        bool threw = false;
        try { _ = r.ReadBits(1); } catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    // -------- Frame-header parse: synthetic ------------------------------

    [TestMethod]
    public void Vp9FrameHeader_Parse_Profile0_Keyframe_640x480()
    {
        var bytes = BuildVp9Keyframe_Profile0_Bt601(640, 480);
        var h = Vp9FrameHeaderParser.Parse(bytes);
        Equal(0, h.Profile);
        False(h.ShowExistingFrame);
        Equal(Vp9FrameType.Key, h.FrameType);
        True(h.ShowFrame);
        False(h.ErrorResilientMode);
        Equal(8, h.BitDepth);
        Equal(Vp9ColorSpace.Bt601, h.ColorSpace);
        False(h.ColorRangeFull);
        True(h.SubsamplingX);
        True(h.SubsamplingY);
        Equal(640, h.FrameWidth);
        Equal(480, h.FrameHeight);
        Equal(0, h.RenderWidth);
        Equal(0, h.RenderHeight);
    }

    [TestMethod]
    public void Vp9FrameHeader_Parse_Profile0_Keyframe_1920x1080()
    {
        var bytes = BuildVp9Keyframe_Profile0_Bt601(1920, 1080);
        var h = Vp9FrameHeaderParser.Parse(bytes);
        Equal(1920, h.FrameWidth);
        Equal(1080, h.FrameHeight);
    }

    [TestMethod]
    public void Vp9FrameHeader_Parse_InvalidFrameMarker_Throws()
    {
        // 0xFF first byte: top 2 bits are 11, not 10.
        bool threw = false;
        try { _ = Vp9FrameHeaderParser.Parse(new byte[] { 0xFF, 0xFF, 0xFF, 0xFF }); }
        catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void Vp9FrameHeader_Parse_ShowExistingFrame_EarlyReturns()
    {
        // marker 10, profile_low 0, profile_high 0, show_existing 1, map_idx 5.
        // Bit pattern: 10 0 0 1 101 ... then padding.
        // = 0b1000_1101 first byte = 0x8D.
        var bytes = new byte[] { 0x8D, 0x00 };
        var h = Vp9FrameHeaderParser.Parse(bytes);
        Equal(0, h.Profile);
        True(h.ShowExistingFrame);
        Equal(5, h.FrameToShowMapIdx);
        // No frame-size / color fields populated.
        Equal(0, h.FrameWidth);
        Equal(Vp9ColorSpace.Unknown, h.ColorSpace);
    }

    [TestMethod]
    public void Vp9FrameHeader_Parse_EmptyBuffer_Throws()
    {
        bool threw = false;
        try { _ = Vp9FrameHeaderParser.Parse(ReadOnlySpan<byte>.Empty); }
        catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    // -------- Frame-header parse: real-world integration -----------------

    [TestMethod]
    public void Vp9FrameHeader_Parse_BigBuckBunny_KeyframesHaveConsistentDimensions()
    {
        // Every keyframe in a stream must declare the same frame size.
        // Use the first keyframe to establish dimensions, then verify all
        // subsequent keyframes match.
        using var stream = LoadBigBuckBunnyWebM();
        var container = new MatroskaContainer(stream);
        var video = container.Tracks.First(t => t.IsVideo);
        Equal("V_VP9", video.CodecId);

        int? firstW = null, firstH = null;
        int keyframesSeen = 0;
        foreach (var frame in container.Frames.Where(f => f.TrackNumber == video.TrackNumber))
        {
            var sf = Vp9SuperframeParser.Parse(frame.Data);
            foreach (var slice in sf.Frames)
            {
                var sub = frame.Data.AsSpan(slice.Offset, slice.Length);
                var h = Vp9FrameHeaderParser.Parse(sub);
                if (h.FrameType == Vp9FrameType.Key && !h.ShowExistingFrame)
                {
                    if (firstW is null)
                    {
                        firstW = h.FrameWidth;
                        firstH = h.FrameHeight;
                        True(h.FrameWidth > 0 && h.FrameHeight > 0,
                            $"keyframe reports bogus dimensions {h.FrameWidth}x{h.FrameHeight}");
                    }
                    else
                    {
                        Equal(firstW.Value, h.FrameWidth);
                        Equal(firstH.Value, h.FrameHeight);
                    }
                    keyframesSeen++;
                }
            }
        }
        True(keyframesSeen >= 1, "expected at least one keyframe in the fixture");
    }

    [TestMethod]
    public void Vp9FrameHeader_Parse_BigBuckBunny_ProfileIsZero_BitDepthEight()
    {
        // The bundled Big Buck Bunny fixture is encoded as profile 0 VP9
        // (8-bit 4:2:0), which every mainline WebM encoder produces unless
        // explicitly configured otherwise. Pinning these values catches
        // regressions in the color-config branch of the parser.
        using var stream = LoadBigBuckBunnyWebM();
        var container = new MatroskaContainer(stream);
        var video = container.Tracks.First(t => t.IsVideo);
        var firstKey = container.Frames
            .Where(f => f.TrackNumber == video.TrackNumber)
            .SelectMany(f =>
            {
                var sf = Vp9SuperframeParser.Parse(f.Data);
                return sf.Frames.Select(s => Vp9FrameHeaderParser.Parse(
                    f.Data.AsSpan(s.Offset, s.Length).ToArray()));
            })
            .First(h => h.FrameType == Vp9FrameType.Key && !h.ShowExistingFrame);
        Equal(0, firstKey.Profile);
        Equal(8, firstKey.BitDepth);
        True(firstKey.SubsamplingX, "profile 0 is 4:2:0, expected SubsamplingX = true");
        True(firstKey.SubsamplingY, "profile 0 is 4:2:0, expected SubsamplingY = true");
    }
}
