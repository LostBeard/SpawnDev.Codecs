// Tests for Vp9SuperframeParser. Hand-builds VP9 Annex B.1 superframe
// index bytes and verifies the parser unpacks every frame slice
// correctly. Also exercises the "not a superframe" fast path and the
// malformed-index rejection cases.

using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    /// <summary>
    /// Build a VP9 superframe packet: concatenated frame bytes, then a
    /// little-endian size index of n*bpf bytes, then the marker byte.
    /// </summary>
    private static byte[] BuildVp9Superframe(int[] frameSizes, int bytesPerSize)
    {
        if (frameSizes.Length < 1 || frameSizes.Length > 8)
            throw new ArgumentException("frame count must be 1..8", nameof(frameSizes));
        if (bytesPerSize < 1 || bytesPerSize > 4)
            throw new ArgumentException("bytesPerSize must be 1..4", nameof(bytesPerSize));

        int total = 0;
        foreach (var s in frameSizes) total += s;
        int indexSize = 1 + frameSizes.Length * bytesPerSize;
        var packet = new byte[total + indexSize];

        // Pattern each frame with distinct bytes so the test can verify
        // the correct slice was returned for each frame.
        int pos = 0;
        for (int f = 0; f < frameSizes.Length; f++)
        {
            for (int i = 0; i < frameSizes[f]; i++)
                packet[pos + i] = (byte)(0x10 + f);
            pos += frameSizes[f];
        }
        // Size index: little-endian.
        for (int f = 0; f < frameSizes.Length; f++)
        {
            int sz = frameSizes[f];
            for (int b = 0; b < bytesPerSize; b++)
                packet[pos + b] = (byte)((sz >> (8 * b)) & 0xFF);
            pos += bytesPerSize;
        }
        // Marker byte: bits 7-5 = 110, bits 4-3 = bpfm1, bits 2-0 = nfm1.
        byte marker = 0b1100_0000;
        marker |= (byte)(((bytesPerSize - 1) & 0x03) << 3);
        marker |= (byte)((frameSizes.Length - 1) & 0x07);
        packet[pos] = marker;
        return packet;
    }

    [TestMethod]
    public void Vp9Superframe_NoIndex_ReturnsSingleFrame()
    {
        // Non-superframe: last byte just happens to not match 0b110.....
        var packet = new byte[] { 0x85, 0x49, 0x83, 0x42, 0x00, 0x00 };
        var sf = Vp9SuperframeParser.Parse(packet);
        False(sf.HadIndex);
        Equal(1, sf.Frames.Count);
        Equal(0, sf.Frames[0].Offset);
        Equal(packet.Length, sf.Frames[0].Length);
    }

    [TestMethod]
    public void Vp9Superframe_TwoFrames_1ByteSize_Unpacks()
    {
        var packet = BuildVp9Superframe(new[] { 50, 30 }, bytesPerSize: 1);
        var sf = Vp9SuperframeParser.Parse(packet);
        True(sf.HadIndex);
        Equal(2, sf.Frames.Count);
        Equal(0, sf.Frames[0].Offset);
        Equal(50, sf.Frames[0].Length);
        Equal(50, sf.Frames[1].Offset);
        Equal(30, sf.Frames[1].Length);
        // Verify slice content matches the per-frame pattern we wrote.
        Equal((byte)0x10, packet[sf.Frames[0].Offset]);
        Equal((byte)0x11, packet[sf.Frames[1].Offset]);
    }

    [TestMethod]
    public void Vp9Superframe_EightFrames_2ByteSize_Unpacks()
    {
        // Exercise the max frame count (3-bit field saturates at 7 for n=8).
        var sizes = new[] { 100, 200, 150, 75, 300, 500, 50, 400 };
        var packet = BuildVp9Superframe(sizes, bytesPerSize: 2);
        var sf = Vp9SuperframeParser.Parse(packet);
        True(sf.HadIndex);
        Equal(8, sf.Frames.Count);
        int expectedOffset = 0;
        for (int i = 0; i < 8; i++)
        {
            Equal(expectedOffset, sf.Frames[i].Offset);
            Equal(sizes[i], sf.Frames[i].Length);
            expectedOffset += sizes[i];
        }
    }

    [TestMethod]
    public void Vp9Superframe_4ByteSize_HandlesLargeFrames()
    {
        // 4-byte size fields support up to ~4GB frames. Verify the parser
        // reads the full 4 bytes correctly for a mid-sized value.
        var sizes = new[] { 70_000, 50_000 }; // neither fits in 2 bytes
        var packet = BuildVp9Superframe(sizes, bytesPerSize: 4);
        var sf = Vp9SuperframeParser.Parse(packet);
        Equal(2, sf.Frames.Count);
        Equal(70_000, sf.Frames[0].Length);
        Equal(50_000, sf.Frames[1].Length);
    }

    [TestMethod]
    public void Vp9Superframe_EmptyPacket_Throws()
    {
        bool threw = false;
        try { _ = Vp9SuperframeParser.Parse(ReadOnlySpan<byte>.Empty); }
        catch (InvalidDataException) { threw = true; }
        True(threw, "empty packet must throw");
    }

    [TestMethod]
    public void Vp9Superframe_IndexOverrunsPacket_Throws()
    {
        // A malicious or corrupt marker claiming 8 frames × 4 bytes = 32
        // bytes of index in a 4-byte packet must be rejected.
        // 0xDF = 0b110_11_111: marker=110, bpfm1=3 (4 bytes each),
        // nfm1=7 (8 frames). Index size = 1 + 8*4 = 33 bytes, won't fit.
        var packet = new byte[] { 0x00, 0x00, 0x00, 0xDF };
        bool threw = false;
        try { _ = Vp9SuperframeParser.Parse(packet); }
        catch (InvalidDataException) { threw = true; }
        True(threw, "index overrun must throw");
    }

    [TestMethod]
    public void Vp9Superframe_FrameSizeOverrunsBeforeIndex_Throws()
    {
        // Build a valid-looking superframe but claim frame 0 is larger
        // than the available space before the index.
        var packet = BuildVp9Superframe(new[] { 10, 10 }, bytesPerSize: 1);
        // Corrupt the first size field: claim frame 0 is 1000 bytes.
        // Index starts at packet.Length - (1 + 2*1) = length - 3.
        int sizesStart = packet.Length - 3;
        packet[sizesStart] = 250; // unsigned 250 still overruns the 20B payload space
        bool threw = false;
        try { _ = Vp9SuperframeParser.Parse(packet); }
        catch (InvalidDataException) { threw = true; }
        True(threw, "frame-size overrun must throw");
    }

    [TestMethod]
    public void Vp9Superframe_SingleFrame_WithIndex_Unpacks()
    {
        // n=1 is valid - index marker just marks a 1-frame wrapper.
        var packet = BuildVp9Superframe(new[] { 42 }, bytesPerSize: 1);
        var sf = Vp9SuperframeParser.Parse(packet);
        True(sf.HadIndex);
        Equal(1, sf.Frames.Count);
        Equal(0, sf.Frames[0].Offset);
        Equal(42, sf.Frames[0].Length);
    }

    [TestMethod]
    public void Vp9Superframe_RealBigBuckBunnyVideoFrames_AllParseCleanly()
    {
        // Integration test: pull every VP9 video packet out of the bundled
        // WebM, run it through the superframe parser, and verify each
        // yields at least one non-empty frame slice with total bytes not
        // exceeding the packet. Catches marker-layout regressions against
        // real encoder output.
        using var stream = LoadBigBuckBunnyWebM();
        var container = new SpawnDev.Codecs.Container.Matroska.MatroskaContainer(stream);
        var video = container.Tracks.First(t => t.IsVideo);
        Equal("V_VP9", video.CodecId);

        int packetsWithIndex = 0;
        int totalFrames = 0;
        int packetsWithoutIndex = 0;
        foreach (var frame in container.Frames.Where(f => f.TrackNumber == video.TrackNumber))
        {
            var sf = Vp9SuperframeParser.Parse(frame.Data);
            True(sf.Frames.Count >= 1, "every VP9 packet must yield at least one frame slice");
            int declared = 0;
            foreach (var slice in sf.Frames)
            {
                True(slice.Length > 0, $"frame slice at offset {slice.Offset} must have positive length");
                True(slice.Offset + slice.Length <= frame.Data.Length,
                    $"slice end {slice.Offset + slice.Length} must be <= packet length {frame.Data.Length}");
                declared += slice.Length;
            }
            True(declared <= frame.Data.Length, "declared sizes must fit in packet");
            totalFrames += sf.Frames.Count;
            if (sf.HadIndex) packetsWithIndex++;
            else packetsWithoutIndex++;
        }
        // At least SOME packets parsed. (The split between index / no-index
        // varies by encoder - we don't assert exact counts, just that the
        // parser walked the whole stream without throwing.)
        True(totalFrames > 0, "must have parsed at least one frame across the whole stream");
    }
}
