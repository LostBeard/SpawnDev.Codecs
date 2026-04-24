using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.Codecs.Container.Ogg;
using SpawnDev.UnitTesting;
using static SpawnDev.Codecs.Demo.Shared.UnitTests.TestHelpers;

namespace SpawnDev.Codecs.Demo.Shared.UnitTests;

/// <summary>
/// Tests for <see cref="FlacOggDecoder"/>. Each test encodes PCM to native
/// FLAC via FlacEncoder, splits the native stream into (mapping header +
/// individual frames) Ogg packets, writes them via OggPageWriter, then
/// decodes back through FlacOggDecoder and asserts the PCM matches.
/// </summary>
public abstract partial class CodecsTestBase
{
    /// <summary>Split a native FLAC byte stream into the mapping header packet + individual frame packets.</summary>
    private static List<byte[]> NativeFlacToFlacInOggPackets(byte[] nativeFlac)
    {
        // "fLaC" + first metadata block header + 34-byte STREAMINFO payload = 4 + 4 + 34 = 42 bytes.
        // (We assume STREAMINFO is the only metadata block our encoder emits.)
        const int streamInfoPayloadOffset = 4 + 4;
        var streamInfoPayload = nativeFlac.AsSpan(streamInfoPayloadOffset, 34).ToArray();

        // Build the mapping header packet per FLAC-to-Ogg mapping spec.
        var mapping = new List<byte>
        {
            0x7F,
            (byte)'F', (byte)'L', (byte)'A', (byte)'C',
            1, 0,              // major / minor version
            0, 0,              // header packet count hint (unused on decode)
        };
        mapping.AddRange(new byte[] { (byte)'f', (byte)'L', (byte)'a', (byte)'C' });
        // Native STREAMINFO metadata block header: isLast=1, type=0, length=34.
        mapping.AddRange(new byte[] { 0x80, 0x00, 0x00, 0x22 });
        mapping.AddRange(streamInfoPayload);

        // Iterate the rest of the native FLAC extracting each frame's byte range.
        var (streamInfo, audioOffset) = FlacMetadataParser.ReadStreamPrelude(nativeFlac);
        int pos = audioOffset;
        var framePackets = new List<byte[]>();
        while (pos < nativeFlac.Length)
        {
            var frame = FlacFrameDecoder.Decode(nativeFlac.AsSpan(pos), streamInfo);
            framePackets.Add(nativeFlac.AsSpan(pos, frame.FrameBytesConsumed).ToArray());
            pos += frame.FrameBytesConsumed;
        }

        var packets = new List<byte[]> { mapping.ToArray() };
        packets.AddRange(framePackets);
        return packets;
    }

    private static byte[] BuildFlacOggBytes(byte[] nativeFlac, uint serial = 0xABCDEF01)
    {
        var packets = NativeFlacToFlacInOggPackets(nativeFlac);
        var outgoing = new List<OggOutgoingPacket>();
        long running = 0;
        for (int i = 0; i < packets.Count; i++)
        {
            outgoing.Add(new OggOutgoingPacket { Data = packets[i], GranulePosition = running });
            running += 1024; // arbitrary - decoder doesn't rely on granule
        }
        return OggPageWriter.WriteStream(serial, outgoing);
    }

    [TestMethod]
    public void FlacOgg_MonoSine_RoundtripsThroughNativeAndOggWrapping()
    {
        var input = GenerateSineInt(samplesPerChannel: 1024, channels: 1, sampleRateHz: 44100, bps: 16);
        byte[] nativeFlac = FlacEncoder.EncodeStream(input, 44100, 1, 16, blockSize: 1024);
        byte[] ogg = BuildFlacOggBytes(nativeFlac);
        var decoded = FlacOggDecoder.Decode(ogg);
        Equal(44100, decoded.StreamInfo.SampleRateHz);
        Equal(1, decoded.StreamInfo.Channels);
        Equal(1024, decoded.TotalSamplesPerChannel);
        EqualInts(input, decoded.InterleavedSamples);
    }

    [TestMethod]
    public void FlacOgg_StereoMultiframe_Roundtrips()
    {
        var input = GenerateSineInt(samplesPerChannel: 512, channels: 2, sampleRateHz: 48000, bps: 16);
        byte[] nativeFlac = FlacEncoder.EncodeStream(input, 48000, 2, 16, blockSize: 128);
        byte[] ogg = BuildFlacOggBytes(nativeFlac);
        var decoded = FlacOggDecoder.Decode(ogg);
        Equal(2, decoded.StreamInfo.Channels);
        Equal(512, decoded.TotalSamplesPerChannel);
        EqualInts(input, decoded.InterleavedSamples);
    }

    [TestMethod]
    public void FlacOgg_24BitMono_Roundtrips()
    {
        var input = GenerateSineInt(samplesPerChannel: 300, channels: 1, sampleRateHz: 96000, bps: 24);
        byte[] nativeFlac = FlacEncoder.EncodeStream(input, 96000, 1, 24, blockSize: 128);
        byte[] ogg = BuildFlacOggBytes(nativeFlac);
        var decoded = FlacOggDecoder.Decode(ogg);
        Equal(24, decoded.StreamInfo.BitsPerSample);
        EqualInts(input, decoded.InterleavedSamples);
    }

    [TestMethod]
    public void FlacOgg_BadMappingMarker_Throws()
    {
        var input = GenerateSineInt(128, 1, 44100, 16);
        byte[] nativeFlac = FlacEncoder.EncodeStream(input, 44100, 1, 16, blockSize: 128);
        var packets = NativeFlacToFlacInOggPackets(nativeFlac);
        packets[0][0] = 0xFF; // mapping marker should be 0x7F
        var outgoing = packets.Select((p, i) => new OggOutgoingPacket { Data = p, GranulePosition = i * 1024 }).ToList();
        byte[] ogg = OggPageWriter.WriteStream(1, outgoing);
        bool threw = false;
        try { _ = FlacOggDecoder.Decode(ogg); } catch (InvalidDataException) { threw = true; }
        True(threw);
    }

    [TestMethod]
    public void FlacOgg_MissingFlacMagic_Throws()
    {
        var input = GenerateSineInt(128, 1, 44100, 16);
        byte[] nativeFlac = FlacEncoder.EncodeStream(input, 44100, 1, 16, blockSize: 128);
        var packets = NativeFlacToFlacInOggPackets(nativeFlac);
        packets[0][1] = (byte)'X'; // corrupt "FLAC" magic
        var outgoing = packets.Select((p, i) => new OggOutgoingPacket { Data = p, GranulePosition = i * 1024 }).ToList();
        byte[] ogg = OggPageWriter.WriteStream(1, outgoing);
        bool threw = false;
        try { _ = FlacOggDecoder.Decode(ogg); } catch (InvalidDataException) { threw = true; }
        True(threw);
    }
}
