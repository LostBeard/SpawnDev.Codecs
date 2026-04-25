// Tests for Vp9CompleteUncompressedHeaderParser - the full
// uncompressed header parser that extends Vp9FrameHeaderParser
// with refresh_frame_flags, ref frame info, allow_high_precision_mv,
// interp_filter, frame_context, loop_filter, quantization,
// segmentation, tile_info, and first_partition_size (header_size).
//
// Drives the BBB fixture's first keyframe through the parser and
// verifies non-zero / sane values come back. Uses the parser's
// FirstPartitionSize as the gate for compressed-header decoding.

using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9CompleteUncompressedHeader_BbbFirstKeyframe_ParsesAllFields()
    {
        using var stream = LoadBigBuckBunnyWebM();
        var container = new MatroskaContainer(stream);
        var video = container.Tracks.First(t => t.IsVideo);

        var firstPacket = container.Frames
            .First(f => f.TrackNumber == video.TrackNumber);
        var sf = Vp9SuperframeParser.Parse(firstPacket.Data);
        var firstFrame = firstPacket.Data.AsSpan(sf.Frames[0].Offset, sf.Frames[0].Length);

        var header = Vp9CompleteUncompressedHeaderParser.Parse(firstFrame);

        // Prefix fields - dimensions match BBB.
        Equal(0, header.FrameHeader.Profile);
        Equal(Vp9FrameType.Key, header.FrameHeader.FrameType);
        Equal(true, header.FrameHeader.ShowFrame);
        Equal(320, header.FrameHeader.FrameWidth);
        Equal(180, header.FrameHeader.FrameHeight);

        // Implicit refresh_frame_flags = 0xff for keyframes.
        Equal((byte)0xff, header.RefreshFrameFlags);

        // Interp filter / allow_hp_mv default for keyframes.
        Equal(false, header.AllowHighPrecisionMv);
        Equal(Vp9InterpFilter.EightTap, header.InterpFilter);

        // FirstPartitionSize must be sensible (1..frame_length).
        True(header.FirstPartitionSize > 0,
            $"first_partition_size must be > 0; got {header.FirstPartitionSize}");
        True(header.FirstPartitionSize < firstFrame.Length,
            $"first_partition_size {header.FirstPartitionSize} must be < frame length {firstFrame.Length}");

        // The uncompressed header itself ends on a byte boundary.
        True(header.UncompressedHeaderSizeBytes > 0,
            "uncompressed header size must be positive");
        True(header.UncompressedHeaderSizeBytes < firstFrame.Length,
            $"uncompressed header size {header.UncompressedHeaderSizeBytes} must fit in frame {firstFrame.Length}");

        // Together: compressed header + uncompressed header < total frame.
        int total = header.UncompressedHeaderSizeBytes + header.FirstPartitionSize;
        True(total <= firstFrame.Length,
            $"uncompressed ({header.UncompressedHeaderSizeBytes}) + compressed ({header.FirstPartitionSize}) " +
            $"must not exceed frame length {firstFrame.Length}");
    }

    [TestMethod]
    public void Vp9CompleteUncompressedHeader_AllVisibleBbbFrames_ParseWithoutThrowing()
    {
        // Smoke test: the complete header parser handles every variant
        // present in the BBB fixture. Without ref frame sizes, inter
        // frames using frame_size_with_refs will fall back to (0,0)
        // dimensions when found_ref is set, but the parser should not
        // throw.
        using var stream = LoadBigBuckBunnyWebM();
        var container = new MatroskaContainer(stream);
        var video = container.Tracks.First(t => t.IsVideo);

        var refSizes = new (int Width, int Height)[3];
        int parsed = 0;
        int keyframes = 0;
        int interFrames = 0;
        foreach (var pkt in container.Frames.Where(f => f.TrackNumber == video.TrackNumber))
        {
            var sf = Vp9SuperframeParser.Parse(pkt.Data);
            foreach (var slice in sf.Frames)
            {
                var span = pkt.Data.AsSpan(slice.Offset, slice.Length);
                var h = Vp9CompleteUncompressedHeaderParser.Parse(span, refSizes);
                parsed++;
                if (h.FrameHeader.FrameType == Vp9FrameType.Key)
                {
                    keyframes++;
                    refSizes[0] = (h.FrameHeader.FrameWidth, h.FrameHeader.FrameHeight);
                    refSizes[1] = (h.FrameHeader.FrameWidth, h.FrameHeader.FrameHeight);
                    refSizes[2] = (h.FrameHeader.FrameWidth, h.FrameHeader.FrameHeight);
                }
                else if (!h.FrameHeader.ShowExistingFrame)
                {
                    interFrames++;
                }
            }
        }
        True(parsed > 0, "fixture must contain VP9 frames");
        True(keyframes >= 1, "fixture must contain at least one keyframe");
        True(interFrames >= 1, "fixture must contain at least one inter frame");
    }

    [TestMethod]
    public void Vp9CompleteUncompressedHeader_FirstPartitionSize_FitsInFrame()
    {
        // Every frame in the BBB fixture must satisfy:
        //   uncompressed_header_size + first_partition_size <= frame_length
        // Otherwise the compressed header wouldn't fit and the bool decoder
        // would underrun.
        using var stream = LoadBigBuckBunnyWebM();
        var container = new MatroskaContainer(stream);
        var video = container.Tracks.First(t => t.IsVideo);

        var refSizes = new (int Width, int Height)[3];
        foreach (var pkt in container.Frames.Where(f => f.TrackNumber == video.TrackNumber))
        {
            var sf = Vp9SuperframeParser.Parse(pkt.Data);
            foreach (var slice in sf.Frames)
            {
                var span = pkt.Data.AsSpan(slice.Offset, slice.Length);
                var h = Vp9CompleteUncompressedHeaderParser.Parse(span, refSizes);
                if (h.FrameHeader.ShowExistingFrame) continue;
                int total = h.UncompressedHeaderSizeBytes + h.FirstPartitionSize;
                True(total <= span.Length,
                    $"frame: {span.Length}B; uncompressed {h.UncompressedHeaderSizeBytes} + " +
                    $"compressed {h.FirstPartitionSize} = {total}");
                if (h.FrameHeader.FrameType == Vp9FrameType.Key)
                {
                    refSizes[0] = (h.FrameHeader.FrameWidth, h.FrameHeader.FrameHeight);
                    refSizes[1] = (h.FrameHeader.FrameWidth, h.FrameHeader.FrameHeight);
                    refSizes[2] = (h.FrameHeader.FrameWidth, h.FrameHeader.FrameHeight);
                }
            }
        }
    }
}
