// Vp9SuperframeWriter round-trip tests against the VP9 superframe parser.

using SpawnDev.Codecs.Container.Matroska;
using SpawnDev.Codecs.Video.Vp9;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

public abstract partial class CodecsTestBase
{
    [TestMethod]
    public void Vp9SuperframeWriter_SingleFrame_RoundTripsViaParser()
    {
        var frame = new byte[] { 0x80, 0x49, 0x83, 0x42, 0x01, 0x02, 0x03, 0x04 };
        var packet = Vp9SuperframeWriter.Emit(new[] { frame });
        // Single frame produces verbatim output (no marker).
        Equal(frame.Length, packet.Length);
        for (int i = 0; i < frame.Length; i++) Equal(frame[i], packet[i]);

        var parsed = Vp9SuperframeParser.Parse(packet);
        Equal(false, parsed.HadIndex);
        Equal(1, parsed.Frames.Count);
        Equal(0, parsed.Frames[0].Offset);
        Equal(frame.Length, parsed.Frames[0].Length);
    }

    [TestMethod]
    public void Vp9SuperframeWriter_TwoFrames_RoundTripsThroughParser()
    {
        var f1 = new byte[100];
        for (int i = 0; i < f1.Length; i++) f1[i] = (byte)i;
        var f2 = new byte[200];
        for (int i = 0; i < f2.Length; i++) f2[i] = (byte)(i * 3);

        var packet = Vp9SuperframeWriter.Emit(new[] { f1, f2 });
        // 2 frames + 2 size bytes (1 byte each since both <= 255) + 1 marker byte
        Equal(100 + 200 + 2 + 1, packet.Length);

        var parsed = Vp9SuperframeParser.Parse(packet);
        Equal(true, parsed.HadIndex);
        Equal(2, parsed.Frames.Count);
        Equal(0, parsed.Frames[0].Offset);
        Equal(100, parsed.Frames[0].Length);
        Equal(100, parsed.Frames[1].Offset);
        Equal(200, parsed.Frames[1].Length);

        // Re-extract and verify each frame's bytes.
        for (int i = 0; i < 100; i++) Equal(f1[i], packet[i]);
        for (int i = 0; i < 200; i++) Equal(f2[i], packet[100 + i]);
    }

    [TestMethod]
    public void Vp9SuperframeWriter_LargeFrames_UsesMultiByteSizes()
    {
        var f1 = new byte[300];   // > 255 forces 2-byte size field
        var f2 = new byte[1000];
        for (int i = 0; i < f1.Length; i++) f1[i] = 0x42;
        for (int i = 0; i < f2.Length; i++) f2[i] = 0x99;

        var packet = Vp9SuperframeWriter.Emit(new[] { f1, f2 });
        // 2 frames + 2 * 2 size bytes + 1 marker = 1305
        Equal(300 + 1000 + 4 + 1, packet.Length);

        var parsed = Vp9SuperframeParser.Parse(packet);
        Equal(2, parsed.Frames.Count);
        Equal(300, parsed.Frames[0].Length);
        Equal(1000, parsed.Frames[1].Length);
    }

    [TestMethod]
    public void Vp9SuperframeWriter_RejectsInvalidInput()
    {
        Throws<ArgumentNullException>(() => Vp9SuperframeWriter.Emit(null!));
        Throws<ArgumentException>(() => Vp9SuperframeWriter.Emit(Array.Empty<byte[]>()));
        var nineFrames = new byte[9][];
        for (int i = 0; i < 9; i++) nineFrames[i] = new byte[10];
        Throws<ArgumentException>(() => Vp9SuperframeWriter.Emit(nineFrames));
    }

    [TestMethod]
    public void Vp9SuperframeWriter_BbbPackets_RoundTripBitExact()
    {
        // Walk every BBB.webm video packet, parse + re-emit through writer,
        // verify every byte matches. BBB has no superframes in the fixture,
        // so each packet is a single frame.
        using var stream = LoadBigBuckBunnyWebM();
        var container = new MatroskaContainer(stream);
        var video = container.Tracks.First(t => t.IsVideo);

        int packetCount = 0;
        foreach (var pkt in container.Frames.Where(f => f.TrackNumber == video.TrackNumber))
        {
            packetCount++;
            var data = pkt.Data.ToArray();
            var parsed = Vp9SuperframeParser.Parse(data);
            // Extract each frame slice as a byte array and re-emit.
            var frames = new byte[parsed.Frames.Count][];
            for (int i = 0; i < parsed.Frames.Count; i++)
            {
                var slice = parsed.Frames[i];
                var fbytes = new byte[slice.Length];
                Buffer.BlockCopy(data, slice.Offset, fbytes, 0, slice.Length);
                frames[i] = fbytes;
            }
            var reEmitted = Vp9SuperframeWriter.Emit(frames);
            Equal(data.Length, reEmitted.Length);
            for (int i = 0; i < data.Length; i++)
            {
                if (data[i] != reEmitted[i])
                    throw new Exception(
                        $"packet {packetCount} byte {i}: src 0x{data[i]:X2} vs emit 0x{reEmitted[i]:X2}");
            }
        }
        Equal(300, packetCount);
    }
}
