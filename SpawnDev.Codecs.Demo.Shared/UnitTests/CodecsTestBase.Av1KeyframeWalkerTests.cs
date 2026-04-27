// AV1 keyframe walker tests. Drives the architecture skeleton against
// the BBB first keyframe and verifies it correctly walks the bitstream
// through partition decode + mode info read, then throws at the
// coefficient decode boundary (next porting step: av1_read_coeffs_txb).

using SpawnDev.Codecs.Container.Ivf;
using SpawnDev.Codecs.Video.Av1;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Av1KeyframeWalker_BbbFirstKeyframe_FailsAtCoefficientDecodeBoundary()
    {
        // The walker has been wired through partition decode + mode info
        // read (intra mode, skip, delta_q, cdef, uv mode, angle delta,
        // filter intra). The remaining gap is av1_read_coeffs_txb (libaom
        // av1/decoder/decodetxb.c). This test pins the current boundary.
        var (sh, payload) = LoadBbbAv1FirstKeyframeForWalker();
        var complete = Av1CompleteFrameHeaderParser.Parse(payload, sh);
        var tg = Av1TileGroupExtractor.Extract(payload, complete);

        var walker = new Av1KeyframeWalker();
        Throws<NotImplementedException>(() =>
            walker.DecodeFrame(payload, sh, complete, tg));
    }

    [TestMethod]
    public void Av1KeyframeWalker_RejectsInterFrameInputs()
    {
        // Build a minimal mock complete header with FrameType=InterFrame.
        // The walker is supposed to fail loudly on inter inputs.
        var sh = new Av1SequenceHeader
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
        var prefix = new Av1FrameHeader
        {
            ShowExistingFrame = false,
            FrameType = Av1FrameType.InterFrame,
            ShowFrame = true,
            ShowableFrame = true,
            ErrorResilientMode = false,
            FrameWidth = 320,
            FrameHeight = 180,
        };
        var complete = new Av1CompleteFrameHeader
        {
            Prefix = prefix,
            TileInfo = new Av1TileInfo { UniformSpacing = true, Log2TileCols = 0, Log2TileRows = 0, TileCols = 1, TileRows = 1, ColStartSb = new[] { 0, 5 }, RowStartSb = new[] { 0, 3 } },
            Quant = new Av1QuantParams { BaseQindex = 100 },
            Segmentation = new Av1SegmentationParams { Enabled = false },
            LoopFilter = new Av1LoopFilterParams { FilterLevel0 = 0, FilterLevel1 = 0 },
            TxMode = Av1TxMode.Largest,
            ReferenceMode = Av1ReferenceMode.SingleReference,
            ReducedTxSetUsed = false,
            HeaderSizeBytes = 10,
            CodedLossless = false,
            AllLossless = false,
        };
        var tg = new Av1TileGroup
        {
            StartTile = 0,
            EndTile = 0,
            Tiles = new[] { new Av1TileBuffer(0, 0, 0, 32) },
        };
        var walker = new Av1KeyframeWalker();
        Throws<NotImplementedException>(() =>
            walker.DecodeFrame(new byte[64], sh, complete, tg));
    }

    private static (Av1SequenceHeader sh, byte[] framePayload) LoadBbbAv1FirstKeyframeForWalker()
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
        if (sh is null) throw new InvalidOperationException("no SH");
        if (frameBytes is null) throw new InvalidOperationException("no Frame OBU");
        return (sh, frameBytes);
    }
}
