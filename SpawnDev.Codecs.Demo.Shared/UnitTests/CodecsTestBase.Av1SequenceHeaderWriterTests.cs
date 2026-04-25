// Av1SequenceHeaderWriter round-trip tests against Av1SequenceHeaderParser.
// Emits a SH for known encoder configurations, parses it back, and
// verifies the parsed fields match what we asked the writer to emit.
//
// This is the first AV1 SH BYTES emitted by SpawnDev.Codecs that another
// AV1 parser (our own first, then ffmpeg) accepts as well-formed. The
// foundation for the AV1 ENCODER's bitstream output.

using SpawnDev.Codecs.Video.Av1;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Av1SequenceHeaderWriter_BbbConfig_RoundTripsThroughParser()
    {
        var cfg = new Av1SequenceHeaderConfig
        {
            SeqProfile = 0,
            SeqLevelIdx0 = 0,
            MaxFrameWidth = 320,
            MaxFrameHeight = 180,
            BitDepth = 8,
            Monochrome = false,
            SubsamplingX = 1,
            SubsamplingY = 1,
            ColorRangeFull = false,
        };

        byte[] payload = Av1SequenceHeaderWriter.EmitPayload(cfg);
        True(payload.Length > 0, "expected non-empty SH payload");

        var sh = Av1SequenceHeaderParser.Parse(payload);
        Equal(0, sh.SeqProfile);
        Equal(false, sh.StillPicture);
        Equal(false, sh.ReducedStillPictureHeader);
        Equal(320, sh.MaxFrameWidth);
        Equal(180, sh.MaxFrameHeight);
        Equal(8, sh.BitDepth);
        Equal(false, sh.Monochrome);
        Equal(1, sh.SubsamplingX);
        Equal(1, sh.SubsamplingY);
        Equal(false, sh.ColorRangeFull);
    }

    [TestMethod]
    public void Av1SequenceHeaderWriter_HighBitDepth10_RoundTrips()
    {
        var cfg = new Av1SequenceHeaderConfig
        {
            SeqProfile = 0,
            MaxFrameWidth = 1920,
            MaxFrameHeight = 1080,
            BitDepth = 10,
            SubsamplingX = 1,
            SubsamplingY = 1,
        };
        var payload = Av1SequenceHeaderWriter.EmitPayload(cfg);
        var sh = Av1SequenceHeaderParser.Parse(payload);
        Equal(0, sh.SeqProfile);
        Equal(1920, sh.MaxFrameWidth);
        Equal(1080, sh.MaxFrameHeight);
        Equal(10, sh.BitDepth);
        Equal(false, sh.Monochrome);
    }

    [TestMethod]
    public void Av1SequenceHeaderWriter_4kFrame_RoundTrips()
    {
        var cfg = new Av1SequenceHeaderConfig
        {
            SeqProfile = 0,
            MaxFrameWidth = 3840,
            MaxFrameHeight = 2160,
            BitDepth = 8,
            SubsamplingX = 1,
            SubsamplingY = 1,
        };
        var payload = Av1SequenceHeaderWriter.EmitPayload(cfg);
        var sh = Av1SequenceHeaderParser.Parse(payload);
        Equal(3840, sh.MaxFrameWidth);
        Equal(2160, sh.MaxFrameHeight);
    }

    [TestMethod]
    public void Av1SequenceHeaderWriter_Monochrome_RoundTrips()
    {
        var cfg = new Av1SequenceHeaderConfig
        {
            SeqProfile = 0,
            MaxFrameWidth = 640,
            MaxFrameHeight = 480,
            BitDepth = 8,
            Monochrome = true,
        };
        var payload = Av1SequenceHeaderWriter.EmitPayload(cfg);
        var sh = Av1SequenceHeaderParser.Parse(payload);
        Equal(true, sh.Monochrome);
        Equal(640, sh.MaxFrameWidth);
        Equal(480, sh.MaxFrameHeight);
        Equal(8, sh.BitDepth);
    }

    [TestMethod]
    public void Av1SequenceHeaderWriter_WrappedAsObu_StreamHasValidShape()
    {
        var cfg = new Av1SequenceHeaderConfig
        {
            SeqProfile = 0,
            MaxFrameWidth = 320,
            MaxFrameHeight = 180,
            BitDepth = 8,
            SubsamplingX = 1,
            SubsamplingY = 1,
        };
        var payload = Av1SequenceHeaderWriter.EmitPayload(cfg);
        var obuBytes = Av1ObuWriter.EmitObu(Av1ObuType.SequenceHeader, payload, hasSizeField: true);

        // Drive it back through the OBU parser - exactly one OBU, type SH.
        var parsedObus = Av1ObuParser.EnumerateObus(obuBytes).ToList();
        Equal(1, parsedObus.Count);
        Equal(Av1ObuType.SequenceHeader, parsedObus[0].Type);
        Equal(true, parsedObus[0].HasSizeField);
        Equal(payload.Length, parsedObus[0].PayloadLength);
    }

    [TestMethod]
    public void Av1SequenceHeaderWriter_RejectsInvalidConfigs()
    {
        // 12-bit on profile 0 is invalid.
        Throws<ArgumentException>(() => Av1SequenceHeaderWriter.EmitPayload(
            new Av1SequenceHeaderConfig
            {
                SeqProfile = 0, MaxFrameWidth = 100, MaxFrameHeight = 100, BitDepth = 12,
            }));
        // Bit depth 9 (invalid).
        Throws<ArgumentOutOfRangeException>(() => Av1SequenceHeaderWriter.EmitPayload(
            new Av1SequenceHeaderConfig
            {
                SeqProfile = 0, MaxFrameWidth = 100, MaxFrameHeight = 100, BitDepth = 9,
            }));
        // Negative width.
        Throws<ArgumentOutOfRangeException>(() => Av1SequenceHeaderWriter.EmitPayload(
            new Av1SequenceHeaderConfig
            {
                SeqProfile = 0, MaxFrameWidth = 0, MaxFrameHeight = 100, BitDepth = 8,
            }));
        // Profile out of range.
        Throws<ArgumentOutOfRangeException>(() => Av1SequenceHeaderWriter.EmitPayload(
            new Av1SequenceHeaderConfig
            {
                SeqProfile = 3, MaxFrameWidth = 100, MaxFrameHeight = 100, BitDepth = 8,
            }));
    }
}
