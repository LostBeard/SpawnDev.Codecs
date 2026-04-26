// Av1FrameHeaderWriter round-trip tests. Emits frame headers via the
// writer, parses them back via the parser, verifies fields match.

using SpawnDev.Codecs.Video.Av1;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    private static Av1SequenceHeader BuildBbbLikeSequenceHeader() => new()
    {
        SeqProfile = 0,
        StillPicture = false,
        ReducedStillPictureHeader = false,
        MaxFrameWidth = 320,
        MaxFrameHeight = 180,
        BitDepth = 8,
        Monochrome = false,
        SubsamplingX = 1,
        SubsamplingY = 1,
        ColorRangeFull = false,
        FrameIdNumbersPresent = false,
        FrameIdLengthMinus7 = 0,
        Use128x128Superblock = false,
        EnableFilterIntra = false,
        EnableIntraEdgeFilter = false,
    };

    [TestMethod]
    public void Av1FrameHeaderWriter_VisibleKeyFrame_RoundTripsThroughParser()
    {
        var sh = BuildBbbLikeSequenceHeader();
        var cfg = new Av1FrameHeaderConfig
        {
            ShowExistingFrame = false,
            FrameType = Av1FrameType.KeyFrame,
            ShowFrame = true,
            // Parser's invariant: visible KeyFrame + SwitchFrame have implicit ErrorResilientMode=true
            ErrorResilientMode = true,
        };
        var payload = Av1FrameHeaderWriter.EmitPayload(cfg, sh);
        var fh = Av1FrameHeaderParser.Parse(payload, sh);
        Equal(false, fh.ShowExistingFrame);
        Equal(Av1FrameType.KeyFrame, fh.FrameType);
        Equal(true, fh.ShowFrame);
        Equal(true, fh.FrameIsIntra);
        Equal(true, fh.ErrorResilientMode);
    }

    [TestMethod]
    public void Av1FrameHeaderWriter_VisibleInterFrame_ErrorResilientFalse_RoundTrips()
    {
        var sh = BuildBbbLikeSequenceHeader();
        var cfg = new Av1FrameHeaderConfig
        {
            ShowExistingFrame = false,
            FrameType = Av1FrameType.InterFrame,
            ShowFrame = true,
            ErrorResilientMode = false,
        };
        var payload = Av1FrameHeaderWriter.EmitPayload(cfg, sh);
        var fh = Av1FrameHeaderParser.Parse(payload, sh);
        Equal(Av1FrameType.InterFrame, fh.FrameType);
        Equal(true, fh.ShowFrame);
        Equal(true, fh.ShowableFrame);
        Equal(false, fh.FrameIsIntra);
        Equal(false, fh.ErrorResilientMode);
    }

    [TestMethod]
    public void Av1FrameHeaderWriter_HiddenFrame_ShowableTrue_RoundTrips()
    {
        var sh = BuildBbbLikeSequenceHeader();
        var cfg = new Av1FrameHeaderConfig
        {
            FrameType = Av1FrameType.InterFrame,
            ShowFrame = false,
            ShowableFrame = true,
            ErrorResilientMode = false,
        };
        var payload = Av1FrameHeaderWriter.EmitPayload(cfg, sh);
        var fh = Av1FrameHeaderParser.Parse(payload, sh);
        Equal(false, fh.ShowFrame);
        Equal(true, fh.ShowableFrame);
        Equal(false, fh.ErrorResilientMode);
    }

    [TestMethod]
    public void Av1FrameHeaderWriter_ShowExistingFrame_RoundTrips()
    {
        var sh = BuildBbbLikeSequenceHeader();
        var cfg = new Av1FrameHeaderConfig
        {
            ShowExistingFrame = true,
            FrameToShowMapIdx = 5,
        };
        var payload = Av1FrameHeaderWriter.EmitPayload(cfg, sh);
        var fh = Av1FrameHeaderParser.Parse(payload, sh);
        Equal(true, fh.ShowExistingFrame);
        Equal(5, fh.FrameToShowMapIdx);
    }

    [TestMethod]
    public void Av1FrameHeaderWriter_IntraOnlyFrame_RoundTrips()
    {
        var sh = BuildBbbLikeSequenceHeader();
        var cfg = new Av1FrameHeaderConfig
        {
            FrameType = Av1FrameType.IntraOnlyFrame,
            ShowFrame = true,
            ErrorResilientMode = false,
        };
        var payload = Av1FrameHeaderWriter.EmitPayload(cfg, sh);
        var fh = Av1FrameHeaderParser.Parse(payload, sh);
        Equal(Av1FrameType.IntraOnlyFrame, fh.FrameType);
        Equal(true, fh.FrameIsIntra);
        Equal(true, fh.ShowableFrame);
        Equal(false, fh.ErrorResilientMode);
    }

    [TestMethod]
    public void Av1FrameHeaderWriter_SwitchFrame_RoundTrips()
    {
        var sh = BuildBbbLikeSequenceHeader();
        var cfg = new Av1FrameHeaderConfig
        {
            FrameType = Av1FrameType.SwitchFrame,
            ShowFrame = true,
            // SwitchFrame has implicit ErrorResilientMode = true and
            // implicit FrameSizeOverride = true.
            ErrorResilientMode = true,
            FrameSizeOverride = true,
        };
        var payload = Av1FrameHeaderWriter.EmitPayload(cfg, sh);
        var fh = Av1FrameHeaderParser.Parse(payload, sh);
        Equal(Av1FrameType.SwitchFrame, fh.FrameType);
        Equal(true, fh.ErrorResilientMode);
        Equal(true, fh.FrameSizeOverride);
    }

    [TestMethod]
    public void Av1FrameHeaderWriter_FromHeader_RoundTripsViaWriter()
    {
        // Closed loop: build a FrameHeader by emitting a known config,
        // parse it back, convert to config via FromHeader, re-emit,
        // verify byte-equivalent output.
        var sh = BuildBbbLikeSequenceHeader();
        var originalCfg = new Av1FrameHeaderConfig
        {
            FrameType = Av1FrameType.InterFrame,
            ShowFrame = true,
            ErrorResilientMode = false,
            DisableCdfUpdate = true,
            RefreshFrameFlags = 0xAB,
        };

        var firstEmit = Av1FrameHeaderWriter.EmitPayload(originalCfg, sh);
        var fh = Av1FrameHeaderParser.Parse(firstEmit, sh);
        var roundTripCfg = Av1FrameHeaderConfig.FromHeader(fh);
        var secondEmit = Av1FrameHeaderWriter.EmitPayload(roundTripCfg, sh);

        Equal(firstEmit.Length, secondEmit.Length);
        for (int i = 0; i < firstEmit.Length; i++)
        {
            if (firstEmit[i] != secondEmit[i])
                throw new Exception(
                    $"FH round-trip byte {i}: first 0x{firstEmit[i]:X2} vs second 0x{secondEmit[i]:X2}");
        }
    }

    [TestMethod]
    public void Av1FrameHeaderWriter_RejectsImplicitErrorResilientMisconfig()
    {
        var sh = BuildBbbLikeSequenceHeader();
        // SwitchFrame requires implicit ErrorResilientMode = true
        Throws<ArgumentException>(() => Av1FrameHeaderWriter.EmitPayload(
            new Av1FrameHeaderConfig
            {
                FrameType = Av1FrameType.SwitchFrame,
                ShowFrame = true,
                ErrorResilientMode = false,
            }, sh));
        // Visible KeyFrame requires implicit ErrorResilientMode = true
        Throws<ArgumentException>(() => Av1FrameHeaderWriter.EmitPayload(
            new Av1FrameHeaderConfig
            {
                FrameType = Av1FrameType.KeyFrame,
                ShowFrame = true,
                ErrorResilientMode = false,
            }, sh));
        // ShowExistingFrame map idx must fit in 3 bits
        Throws<ArgumentOutOfRangeException>(() => Av1FrameHeaderWriter.EmitPayload(
            new Av1FrameHeaderConfig
            {
                ShowExistingFrame = true,
                FrameToShowMapIdx = 8,
            }, sh));
    }

    [TestMethod]
    public void Av1FrameHeaderWriter_ReducedStillPictureHeader_EmptyHeader()
    {
        // When SH says reduced_still_picture_header=true, parser ignores
        // the header bytes and returns a fixed key/show/intra value. The
        // writer matches that by writing only trailing-bits.
        var sh = new Av1SequenceHeader
        {
            SeqProfile = 0,
            StillPicture = true,
            ReducedStillPictureHeader = true,
            MaxFrameWidth = 320, MaxFrameHeight = 180, BitDepth = 8,
            Monochrome = false, SubsamplingX = 1, SubsamplingY = 1,
            ColorRangeFull = false, FrameIdNumbersPresent = false,
            FrameIdLengthMinus7 = 0, Use128x128Superblock = false,
            EnableFilterIntra = false, EnableIntraEdgeFilter = false,
        };
        var cfg = new Av1FrameHeaderConfig
        {
            FrameType = Av1FrameType.KeyFrame,
            ShowFrame = true,
            ErrorResilientMode = false,
        };
        var payload = Av1FrameHeaderWriter.EmitPayload(cfg, sh);
        var fh = Av1FrameHeaderParser.Parse(payload, sh);
        Equal(Av1FrameType.KeyFrame, fh.FrameType);
        Equal(true, fh.ShowFrame);
        Equal(true, fh.FrameIsIntra);
    }
}
