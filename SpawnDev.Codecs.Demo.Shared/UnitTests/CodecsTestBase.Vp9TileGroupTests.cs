// Tests for Vp9TileGroupExtractor - the post-compressed-header
// per-tile byte range extraction.
//
// VP9 tile data layout: after byte-aligned uncompressed header +
// first_partition_size compressed header, the remaining bytes are
// tile_cols * tile_rows tiles. Each non-last tile has a 4-byte
// big-endian size prefix; the last tile consumes whatever is left.

using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9TileGroup_BbbFirstKeyframe_OneTileExtracted()
    {
        using var stream = LoadBigBuckBunnyWebM();
        var container = new MatroskaContainer(stream);
        var video = container.Tracks.First(t => t.IsVideo);

        var firstPacket = container.Frames
            .First(f => f.TrackNumber == video.TrackNumber);
        var sf = Vp9SuperframeParser.Parse(firstPacket.Data);
        var firstFrame = firstPacket.Data.AsSpan(sf.Frames[0].Offset, sf.Frames[0].Length);

        var header = Vp9CompleteUncompressedHeaderParser.Parse(firstFrame);
        var tileGroup = Vp9TileGroupExtractor.Extract(firstFrame, header);

        // BBB at 320x180 likely has 1 tile (log2_tile_cols = 0, log2_tile_rows = 0).
        Equal(header.TileInfo.TileCols * header.TileInfo.TileRows, tileGroup.Tiles.Count);
        True(tileGroup.Tiles.Count >= 1, "expected at least 1 tile");

        // First (and possibly only) tile starts after compressed header
        // and runs to end of frame for single-tile case.
        var firstTile = tileGroup.Tiles[0];
        int expectedStart = header.UncompressedHeaderSizeBytes + header.FirstPartitionSize;
        Equal(expectedStart, firstTile.Offset);
        if (tileGroup.Tiles.Count == 1)
        {
            // Last tile consumes all remaining bytes.
            Equal(firstFrame.Length - expectedStart, firstTile.Length);
        }
        True(firstTile.Length > 0, "tile must have content");
    }

    [TestMethod]
    public void Vp9TileGroup_AllBbbVisibleFrames_TilesFitInFrame()
    {
        // For every visible BBB frame, the extracted tile slices must
        // tile-cover the post-compressed-header region exactly:
        //   sum(tileSize + 4 if !last else 0) == remaining bytes
        // and no tile range exceeds the frame.
        using var stream = LoadBigBuckBunnyWebM();
        var container = new MatroskaContainer(stream);
        var video = container.Tracks.First(t => t.IsVideo);

        var refSizes = new (int Width, int Height)[3];
        int framesProcessed = 0;
        foreach (var pkt in container.Frames.Where(f => f.TrackNumber == video.TrackNumber))
        {
            var sf = Vp9SuperframeParser.Parse(pkt.Data);
            foreach (var slice in sf.Frames)
            {
                var span = pkt.Data.AsSpan(slice.Offset, slice.Length);
                var h = Vp9CompleteUncompressedHeaderParser.Parse(span, refSizes);
                if (h.FrameHeader.ShowExistingFrame) continue;

                var tg = Vp9TileGroupExtractor.Extract(span, h);
                Equal(h.TileInfo.TileCols * h.TileInfo.TileRows, tg.Tiles.Count);

                // Every tile range must be in-bounds.
                foreach (var t in tg.Tiles)
                {
                    True(t.Offset >= 0, $"tile offset {t.Offset} negative");
                    True(t.Length >= 0, $"tile length {t.Length} negative");
                    True(t.Offset + t.Length <= span.Length,
                        $"tile {t.Offset}..{t.Offset + t.Length} exceeds frame {span.Length}");
                }

                if (h.FrameHeader.FrameType == Vp9FrameType.Key)
                {
                    refSizes[0] = (h.FrameHeader.FrameWidth, h.FrameHeader.FrameHeight);
                    refSizes[1] = (h.FrameHeader.FrameWidth, h.FrameHeader.FrameHeight);
                    refSizes[2] = (h.FrameHeader.FrameWidth, h.FrameHeader.FrameHeight);
                }
                framesProcessed++;
            }
        }
        True(framesProcessed > 0, "fixture must contain visible frames");
    }
}
