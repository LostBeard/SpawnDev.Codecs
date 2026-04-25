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
    public void Av1SequenceHeaderWriter_BbbConfig_BitExactMatchesLibaomBytes()
    {
        // Strongest spec validation: build the config that mirrors what
        // libaom-av1 chose for the BBB encode (observed by parsing the
        // source SH bit-by-bit) and verify our writer emits IDENTICAL
        // bytes. Same config in -> same bitstream out as the reference
        // encoder.
        var bytes = LoadAv1Fixture();
        var firstFrame = SpawnDev.Codecs.Container.Ivf.IvfReader.EnumerateFrames(bytes).First();
        byte[] sourceSh = Array.Empty<byte>();
        foreach (var obu in Av1ObuParser.EnumerateObus(firstFrame.Data))
        {
            if (obu.Type == Av1ObuType.SequenceHeader)
            {
                sourceSh = firstFrame.Data.Slice(obu.PayloadOffset, obu.PayloadLength).ToArray();
                break;
            }
        }
        True(sourceSh.Length > 0, "Could not extract source SH from BBB fixture.");

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
            Use128x128Superblock = false,
            EnableFilterIntra = true,
            EnableIntraEdgeFilter = true,
            EnableInterintraCompound = false,
            EnableMaskedCompound = true,
            EnableWarpedMotion = true,
            EnableDualFilter = false,
            EnableOrderHint = true,
            EnableJntComp = false,
            EnableRefFrameMvs = true,
            OrderHintBitsMinus1 = 6,
            SeqChooseScreenContentTools = true,
            SeqChooseIntegerMv = true,
            EnableSuperres = false,
            EnableCdef = true,
            EnableRestoration = false,
            ColorDescriptionPresent = true,
            ColorPrimaries = 2,
            TransferCharacteristics = 2,
            MatrixCoefficients = 5,
            ChromaSamplePosition = 0,
            SeparateUvDeltas = false,
            FilmGrainParamsPresent = false,
        };

        var emitted = Av1SequenceHeaderWriter.EmitPayload(cfg);
        Equal(sourceSh.Length, emitted.Length);
        for (int i = 0; i < sourceSh.Length; i++)
        {
            if (sourceSh[i] != emitted[i])
                throw new Exception(
                    $"BBB SH byte {i}: source 0x{sourceSh[i]:X2} vs emitted 0x{emitted[i]:X2}");
        }
    }

    [TestMethod]
    public void Av1SequenceHeaderWriter_BbbSh_ParseToConfigEchoesBitExact()
    {
        // The full closed loop: parse a real libaom-encoded SH OBU, build
        // the writer config straight from the parsed Av1SequenceHeader via
        // Av1SequenceHeaderConfig.FromHeader, and verify the writer emits
        // back the IDENTICAL source bytes. This proves the parser surfaces
        // every conditional bit it reads, and the writer round-trips it.
        var bytes = LoadAv1Fixture();
        var firstFrame = SpawnDev.Codecs.Container.Ivf.IvfReader.EnumerateFrames(bytes).First();
        byte[] sourceSh = Array.Empty<byte>();
        foreach (var obu in Av1ObuParser.EnumerateObus(firstFrame.Data))
        {
            if (obu.Type == Av1ObuType.SequenceHeader)
            {
                sourceSh = firstFrame.Data.Slice(obu.PayloadOffset, obu.PayloadLength).ToArray();
                break;
            }
        }
        True(sourceSh.Length > 0, "no SH OBU found in BBB first frame");

        var sh = Av1SequenceHeaderParser.Parse(sourceSh);
        var cfg = Av1SequenceHeaderConfig.FromHeader(sh);
        var emitted = Av1SequenceHeaderWriter.EmitPayload(cfg);

        Equal(sourceSh.Length, emitted.Length);
        for (int i = 0; i < sourceSh.Length; i++)
        {
            if (sourceSh[i] != emitted[i])
                throw new Exception(
                    $"BBB SH echo byte {i}: source 0x{sourceSh[i]:X2} vs emitted 0x{emitted[i]:X2}");
        }
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
