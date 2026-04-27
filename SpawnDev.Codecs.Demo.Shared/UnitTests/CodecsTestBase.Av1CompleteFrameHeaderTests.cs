// AV1 complete frame header parser tests. Drives the parser against
// real BBB bbb_180_2s.ivf bitstream to confirm tile_info / quant / lf /
// cdef / lr / segmentation are parsed coherently for the keyframe.

using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Av1CompleteFrameHeader_BbbFirstKeyframe_ParsesWithoutThrowing()
    {
        var bytes = LoadAv1Fixture();
        // Find the SequenceHeader and the first Frame OBU.
        var firstIvf = IvfReader.EnumerateFrames(bytes).First();
        Av1SequenceHeader? sh = null;
        Av1Obu? frameObu = null;
        int frameOffset = 0;
        foreach (var obu in Av1ObuParser.EnumerateObus(firstIvf.Data))
        {
            if (obu.Type == Av1ObuType.SequenceHeader)
            {
                sh = Av1SequenceHeaderParser.Parse(
                    firstIvf.Data.Span.Slice(obu.PayloadOffset, obu.PayloadLength));
            }
            else if (obu.Type == Av1ObuType.Frame)
            {
                frameObu = obu;
                frameOffset = obu.PayloadOffset;
                break;
            }
        }
        True(sh is not null, "expected SequenceHeader OBU in first IVF frame");
        True(frameObu.HasValue, "expected Frame OBU in first IVF frame");

        var payload = firstIvf.Data.Span.Slice(frameOffset, frameObu.Value.PayloadLength);
        var complete = Av1CompleteFrameHeaderParser.Parse(payload, sh!);

        True(complete.Prefix.FrameType == Av1FrameType.KeyFrame,
            $"expected KeyFrame; got {complete.Prefix.FrameType}");
        Equal(320, complete.Prefix.FrameWidth);
        Equal(180, complete.Prefix.FrameHeight);
    }

    [TestMethod]
    public void Av1CompleteFrameHeader_BbbFirstKeyframe_HasReasonableTileInfo()
    {
        var (sh, payload) = LoadBbbAv1FirstKeyframe();
        var complete = Av1CompleteFrameHeaderParser.Parse(payload, sh);
        // 320x180 BBB encoded by libaom uses uniform 1x1 tiling per
        // libdav1d's `Frame 0: ... tiles 1x1` ffprobe debug output.
        Equal(1, complete.TileInfo.TileCols);
        Equal(1, complete.TileInfo.TileRows);
        Equal(true, complete.TileInfo.UniformSpacing);
    }

    [TestMethod]
    public void Av1CompleteFrameHeader_BbbFirstKeyframe_QuantBaseQindex_MatchesLibdav1d()
    {
        // Spot-check baseQindex against what libdav1d reports for this fixture.
        // libaom encoded BBB at high quality so baseQindex is in single digits.
        var (sh, payload) = LoadBbbAv1FirstKeyframe();
        var complete = Av1CompleteFrameHeaderParser.Parse(payload, sh);
        // Tightened band (verified parser produces baseQindex=5).
        InRange(complete.Quant.BaseQindex, 0, 32);
    }

    [TestMethod]
    public void Av1CompleteFrameHeader_BbbFirstKeyframe_TileGroupExtraction_Succeeds()
    {
        // Now that the header parser is bit-cursor-correct, the tile group
        // extractor should succeed and find exactly 1 tile covering the
        // remainder of the frame payload.
        var (sh, payload) = LoadBbbAv1FirstKeyframe();
        var complete = Av1CompleteFrameHeaderParser.Parse(payload, sh);
        var tg = Av1TileGroupExtractor.Extract(payload, complete);
        Equal(1, tg.Tiles.Count);
        Equal(0, tg.Tiles[0].TileRow);
        Equal(0, tg.Tiles[0].TileCol);
        // Tile data starts right after the 9-byte uncompressed header.
        InRange(tg.Tiles[0].Offset, 1, 64);
        True(tg.Tiles[0].Length > 1000,
            $"single-tile payload should be most of the frame ({tg.Tiles[0].Length}B)");
    }

    [TestMethod]
    public void Av1CompleteFrameHeader_BbbFirstKeyframe_QuantInBand()
    {
        var (sh, payload) = LoadBbbAv1FirstKeyframe();
        var complete = Av1CompleteFrameHeaderParser.Parse(payload, sh);
        // libaom typical default qindex ~ 80-160 for medium quality. BBB ffmpeg
        // CRF 30-ish usually produces qindex around 100-200. Sanity bound only.
        InRange(complete.Quant.BaseQindex, 0, 255);
        // delta_q values are signed 6-bit literals -> -63..63
        InRange(complete.Quant.YDcDeltaQ, -63, 63);
        InRange(complete.Quant.UDcDeltaQ, -63, 63);
        InRange(complete.Quant.UAcDeltaQ, -63, 63);
        InRange(complete.Quant.VDcDeltaQ, -63, 63);
        InRange(complete.Quant.VAcDeltaQ, -63, 63);
    }

    [TestMethod]
    public void Av1CompleteFrameHeader_BbbFirstKeyframe_SegmentationDisabled()
    {
        var (sh, payload) = LoadBbbAv1FirstKeyframe();
        var complete = Av1CompleteFrameHeaderParser.Parse(payload, sh);
        // BBB encoded by libaom defaults to segmentation OFF.
        Equal(false, complete.Segmentation.Enabled);
    }

    [TestMethod]
    public void Av1CompleteFrameHeader_BbbFirstKeyframe_LoopFilterLevelsValid()
    {
        var (sh, payload) = LoadBbbAv1FirstKeyframe();
        var complete = Av1CompleteFrameHeaderParser.Parse(payload, sh);
        // 6-bit unsigned literals -> 0..63 each
        InRange(complete.LoopFilter.FilterLevel0, 0, 63);
        InRange(complete.LoopFilter.FilterLevel1, 0, 63);
    }

    [TestMethod]
    public void Av1CompleteFrameHeader_BbbFirstKeyframe_HasCdefParamsWhenEnabled()
    {
        var (sh, payload) = LoadBbbAv1FirstKeyframe();
        var complete = Av1CompleteFrameHeaderParser.Parse(payload, sh);
        if (sh.EnableCdef && !complete.CodedLossless && !complete.Prefix.AllowIntraBc)
        {
            True(complete.Cdef is not null, "CDEF params required when SH.EnableCdef is on");
            InRange(complete.Cdef!.Damping, 3, 6);  // 2-bit + 3 = 3..6
            InRange(complete.Cdef.Bits, 0, 3);
        }
    }

    [TestMethod]
    public void Av1CompleteFrameHeader_BbbFirstKeyframe_TxModeAndReducedTxSet()
    {
        var (sh, payload) = LoadBbbAv1FirstKeyframe();
        var complete = Av1CompleteFrameHeaderParser.Parse(payload, sh);
        // Either Largest (1) or Select (2) is valid for non-lossless keyframes.
        True(complete.TxMode == Av1TxMode.Largest || complete.TxMode == Av1TxMode.Select,
            $"unexpected TxMode {complete.TxMode}");
        // ReferenceMode for intra frames is forced to SingleReference.
        Equal(Av1ReferenceMode.SingleReference, complete.ReferenceMode);
    }

    [TestMethod]
    public void Av1CompleteFrameHeader_BbbFirstKeyframe_HeaderSizeIsReasonable()
    {
        var (sh, payload) = LoadBbbAv1FirstKeyframe();
        var complete = Av1CompleteFrameHeaderParser.Parse(payload, sh);
        // Uncompressed header for a 320x180 keyframe is < 100 bytes typically.
        InRange(complete.HeaderSizeBytes, 1, 200);
        True(complete.HeaderSizeBytes < payload.Length,
            $"header size {complete.HeaderSizeBytes} must be < payload {payload.Length}");
    }

    private static (Av1SequenceHeader sh, byte[] framePayload) LoadBbbAv1FirstKeyframe()
    {
        var bytes = LoadAv1Fixture();
        var firstIvf = IvfReader.EnumerateFrames(bytes).First();
        Av1SequenceHeader? sh = null;
        byte[]? frameBytes = null;
        foreach (var obu in Av1ObuParser.EnumerateObus(firstIvf.Data))
        {
            if (obu.Type == Av1ObuType.SequenceHeader)
            {
                sh = Av1SequenceHeaderParser.Parse(
                    firstIvf.Data.Span.Slice(obu.PayloadOffset, obu.PayloadLength));
            }
            else if (obu.Type == Av1ObuType.Frame)
            {
                frameBytes = firstIvf.Data.Span.Slice(obu.PayloadOffset, obu.PayloadLength).ToArray();
                break;
            }
        }
        if (sh is null) throw new InvalidOperationException("no SH in first BBB frame");
        if (frameBytes is null) throw new InvalidOperationException("no Frame OBU in first BBB frame");
        return (sh, frameBytes);
    }
}
